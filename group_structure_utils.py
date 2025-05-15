"""
Utilities for analyzing and visualizing company group structures from Companies House data.
"""
import pathlib as _pl
import logging # For logging
from datetime import datetime, timedelta # Added timedelta
import json # For parsing JSON content
import re # For regex-based searching in text/xhtml
import html # For unescaping HTML entities
from typing import Optional, Dict, List, Any, Callable, Tuple, TypeAlias # Added Callable, TypeAlias

JSONObj: TypeAlias = Dict[str, Any]

# Create a logger for this module
try:
    import logging
    logger = logging.getLogger(__name__)
except Exception as e:
    print(f"Warning: Could not set up logger in group_structure_utils.py: {e}")
    class DummyLogger:
        def debug(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
    logger = DummyLogger()

# Generic alias: bytes (binary doc) + str (mime/ext)  →  (text, confidence, error)
OCRHandlerType: TypeAlias = Callable[[bytes, str], tuple[str, int, str | None]]

# Try to override with the richer alias from `text_extraction_utils`, if available
try:
    from text_extraction_utils import extract_text_from_document, OCRHandlerType as _LibOCRType  # noqa: E402
    OCRHandlerType = _LibOCRType  # type: ignore[assignment]
except ImportError as _imp_err:  # library absent – keep generic alias
    logger.warning("text_extraction_utils not importable: %s; using generic OCRHandlerType", _imp_err)
    def extract_text_from_document(*args: Any, **kwargs: Any) -> tuple[str, int, str]:
        logger.error("extract_text_from_document called but backend not available")
        return "", 0, "OCR backend missing"

# Handle imports carefully with clear error messages
try:
    from text_extraction_utils import extract_text_from_document, OCRHandlerType as TextExtractionOCRHandlerType
    # OCRHandlerType = TextExtractionOCRHandlerType      # ← delete / comment
except ImportError as e:
    logger.error(f"Failed to import from text_extraction_utils: {e}")
    # OCRHandlerType = Any                               # ← delete / comment
    def extract_text_from_document(*args, **kwargs):
        logger.error("extract_text_from_document called but not available")
        return f"Error: Function not available due to import issues", 0, "Import failed"

# Import ch_api_utils functions with error handling
try:
    from ch_api_utils import get_ch_documents_metadata, get_company_profile, get_company_pscs, _fetch_document_content_from_ch
    import ch_api_utils
except ImportError as e:
    logger.error(f"Failed to import from ch_api_utils: {e}")
    ch_api_utils = None
    def get_ch_documents_metadata(*args, **kwargs):
        logger.error("get_ch_documents_metadata called but not available")
        return [], None, "Function not available due to import issues"
    def get_company_profile(*args, **kwargs):
        logger.error("get_company_profile called but not available")
        return None
    def get_company_pscs(*args, **kwargs):
        logger.error("get_company_pscs called but not available")
        return None, "Function not available due to import issues"
    def _fetch_document_content_from_ch(*args, **kwargs):
        logger.error("_fetch_document_content_from_ch called but not available")
        return {}, [], "Function not available due to import issues"

# Import Graphviz with error handling    
# Import AWS Textract utilities with error handling
try:
    from aws_textract_utils import perform_textract_ocr
except ImportError as e:
    logger.error(f"Failed to import from aws_textract_utils: {e}")
    def perform_textract_ocr(*args, **kwargs):
        logger.error("perform_textract_ocr called but not available")
        return "Error: OCR function not available due to import issues", 0, "Import failed"
# Import streamlit with error handling
try:
    import streamlit as st # type: ignore
except ImportError as e:
    logger.error(f"Failed to import streamlit: {e}")
    # Create a minimal mock st object to prevent errors
    class MockStreamlit:
        def markdown(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def expander(self, *args, **kwargs): return self
        def text_input(self, *args, **kwargs): return ""
        def button(self, *args, **kwargs): return False
        def spinner(self, *args, **kwargs): 
            class MockSpinner:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return MockSpinner()
        def tabs(self, *args, **kwargs): 
            return [self, self, self]
        def __enter__(self): return self
        def __exit__(self, *args): pass
        
    # Replace st with our mock if import failed
    st = MockStreamlit()

# Ensure company profile is imported if used directly
# This is already handled in the block above if ch_api_utils import failed

# Constants for metadata fetching
RELEVANT_CATEGORIES_FOR_GROUP_STRUCTURE = ["accounts", "confirmation-statement"] # Added annual-return for older filings
METADATA_YEARS_TO_SCAN = 10 # Scan last 10 years for relevant documents
ITEMS_PER_PAGE_API = 100
MAX_DOCS_TO_SCAN_PER_CATEGORY_API = 200 # Max raw items to look through per category via API
TARGET_DOCS_PER_CATEGORY_IN_RANGE_API = 50 # Target to find within the date range per category

# Placeholder for actual API call functions (these would live in ch_api_utils.py ideally)
# def get_company_profile_psc(company_number: str, api_key: str, logger: logging.Logger):
#     logger.info(f"Simulating API call for PSC data for {company_number}")
#     # Simulate API response structure for PSC
#     if company_number == "12345678": # Child
#         return {"items": [{"name": "ULTIMATE PARENT LTD", "natures_of_control": ["ownership-of-shares-75-to-100-percent"], "links": {"company": "/company/08888888"}}], "active_count": 1}
#     elif company_number == "08888888": # Parent
#         return {"items": [], "active_count": 0} # No corporate PSCs, might be individual or top-level
#     return {"items": [], "active_count": 0}

# def get_company_filing_history_accounts(company_number: str, api_key: str, logger: logging.Logger):
#     logger.info(f"Simulating API call for filing history (accounts) for {company_number}")
#     # Simulate finding group accounts
#     if company_number == "08888888":
#         return {"items": [
#             {"type": "AA", "date": "2023-12-31", "description": "GROUP ACCOUNTS MADE UP TO 31/12/2023", "links": {"document_metadata": "url_to_group_accounts_metadata"}},
#             {"type": "AA", "date": "2022-12-31", "description": "FULL ACCOUNTS MADE UP TO 31/12/2022", "links": {"document_metadata": "url_to_full_accounts_metadata"}}
#         ]}
#     return {"items": []}

# Renamed function from find_ultimate_parent
def extract_parent_timeline(company_number: str, api_key: str, logger: logging.Logger, parsed_docs_data: list = None):
    """
    Extracts a timeline of parent company information from parsed documents.
    Args:
        company_number (str): The company number being analyzed.
        api_key (str): Companies House API key.
        logger (logging.Logger): Logger instance.
        parsed_docs_data (list, optional): List of dictionaries, where each dict is a doc_bundle
                                           containing ['metadata'] and ['parsed_data'].

    Returns:
        tuple: (parent_timeline, messages)
               parent_timeline: List of dicts, each detailing a parent mention:
                                {'date': str, 'parent_name': str, 'parent_number': str, 
                                 'source_doc_desc': str, 's409_related': bool}
               messages: List of strings for reporting.
    """
    messages = ["Parent Company Timeline Analysis:"]
    logger.info(f"Extracting parent company timeline for {company_number} from parsed documents.")
    parent_timeline = []

    if not parsed_docs_data:
        messages.append("No parsed document data provided for parent timeline generation.")
        logger.warning("find_ultimate_parent called without parsed_docs_data.")
        return parent_timeline, messages

    # Helper to get document date for sorting
    def get_doc_date(doc_bundle):
        parsed_data = doc_bundle.get('parsed_data')
        if parsed_data and parsed_data.get('document_date'):
            try:
                return datetime.strptime(parsed_data['document_date'], '%Y-%m-%d')
            except (ValueError, TypeError):
                # Try to get from metadata if parsed_data is missing it
                metadata = doc_bundle.get('metadata')
                if metadata and metadata.get('date'):
                    try:
                        return datetime.strptime(metadata['date'], '%Y-%m-%d')
                    except (ValueError, TypeError):
                        pass # Fall through
                return datetime.min # Oldest possible date if parsing fails
        # Fallback to metadata if parsed_data date is missing
        metadata = doc_bundle.get('metadata')
        if metadata and metadata.get('date'):
            try:
                return datetime.strptime(metadata['date'], '%Y-%m-%d')
            except (ValueError, TypeError):
                pass # Fall through
        return datetime.min

    # Sort documents by date (oldest first for timeline reporting)
    sorted_docs = sorted(parsed_docs_data, key=get_doc_date)

    last_reported_parent_key = None

    for doc_bundle in sorted_docs:
        parsed = doc_bundle.get('parsed_data')
        metadata = doc_bundle.get('metadata') # Get metadata for doc description and date fallback

        if not parsed or not metadata: # Skip if essential data is missing
            logger.debug("Skipping doc_bundle due to missing parsed_data or metadata.")
            continue
        
        # Use date from parsed_data if available, else from metadata
        doc_date_str = parsed.get("document_date") or metadata.get("date", "N/A")
        doc_desc = parsed.get("document_description") or metadata.get("description", "Unknown Document")

        parent_name = parsed.get("parent_company_name")
        parent_number = parsed.get("parent_company_number")
        s409_found = parsed.get("s409_info_found", False)

        if parent_name or parent_number: # If any parent information is found
            # Normalize parent_number to None if it's an empty string or whitespace
            current_parent_number = parent_number.strip() if parent_number and parent_number.strip() else None
            current_parent_name = parent_name.strip() if parent_name and parent_name.strip() else None

            # Avoid logging the same parent repeatedly if it hasn't changed from the previous doc in the timeline
            current_parent_key = (current_parent_name, current_parent_number)
            
            # Log if it's a new parent or the first one in the timeline
            if current_parent_key != last_reported_parent_key or not parent_timeline:
                entry = {
                    'date': doc_date_str,
                    'parent_name': current_parent_name,
                    'parent_number': current_parent_number,
                    'source_doc_desc': doc_desc,
                    's409_related': s409_found
                }
                parent_timeline.append(entry)
                message = f"- On {doc_date_str} ({doc_desc}): Reported parent as '{current_parent_name or 'N/A'}' (CRN: {current_parent_number or 'N/A'}). S409 related: {s409_found}." # Corrected f-string
                messages.append(message)
                logger.info(f"Timeline: {company_number} - {message}")
                last_reported_parent_key = current_parent_key
            else:
                logger.debug(f"Timeline: {company_number} - On {doc_date_str} ({doc_desc}): Parent '{current_parent_name or 'N/A'}' (CRN: {current_parent_number or 'N/A'}) consistent with previous entry.")
        elif s409_found: # S409 found but no explicit parent name/number in this parsed doc
            # This might indicate the company is part of a group but the specific parent isn't extracted by the parser from this doc
            # Or it's availing exemption because its own parent (elsewhere disclosed) files group accounts.
            # We can add an entry indicating S409 was mentioned, which is a data point for group status.
            current_parent_key = ("S409_INFO_ONLY", "S409_INFO_ONLY") # Special key for this case
            if current_parent_key != last_reported_parent_key or not parent_timeline:
                entry = {
                    'date': doc_date_str,
                    'parent_name': None, # No specific parent extracted here
                    'parent_number': None,
                    'source_doc_desc': doc_desc,
                    's409_related': True
                }
                parent_timeline.append(entry)
                message = f"- On {doc_date_str} ({doc_desc}): S409 information found, but no specific parent details extracted from this document. Implies group context."
                messages.append(message)
                logger.info(f"Timeline: {company_number} - {message}")
                last_reported_parent_key = current_parent_key

    if not parent_timeline:
        messages.append("No specific parent company mentions found in the analyzed documents to build a timeline.")
        logger.info(f"No parent company timeline generated for {company_number} as no mentions found.")
    else:
        messages.insert(1, f"Found {len(parent_timeline)} distinct parent changes/mentions over time:")

    return parent_timeline, messages


def _get_corporate_psc_parent_info(company_number: str, api_key: str, logger: logging.Logger) -> Optional[Dict[str, str]]:
    """
    Fetches and parses PSC data to find a corporate parent.
    Returns a dict {'name': name, 'number': crn} or None.
    """
    logger.info(f"Attempting to fetch PSC data for {company_number} to identify corporate parents.")
    if not hasattr(ch_api_utils, 'get_company_pscs'):
        logger.warning("PSC check skipped: 'get_company_pscs' function not found in ch_api_utils.")
        return None

    try:
        pscs_data, psc_error = get_company_pscs(company_number, api_key) # Assuming this function exists and returns (data, error_msg)
        
        if psc_error:
            logger.error(f"Error fetching PSC data for {company_number}: {psc_error}")
            return None
        
        if not pscs_data or not pscs_data.get("items"):
            logger.info(f"No PSC items found for {company_number}.")
            return None

        corporate_pscs = []
        for psc in pscs_data["items"]:
            if psc.get("kind") == "corporate-entity-person-with-significant-control":
                psc_name = psc.get("name")
                identification = psc.get("identification")
                if identification:
                    psc_crn = identification.get("registration_number")
                    # Ensure the corporate PSC is not the company itself
                    if psc_crn and psc_crn.strip().upper() != company_number.strip().upper():
                        # Prioritize PSCs with strong control natures, e.g., >50% or >75%
                        # For simplicity, we'll take the first valid one. More complex logic can be added.
                        # Example: Check natures_of_control for "ownership-of-shares-more-than-75-percent" etc.
                        corporate_pscs.append({"name": psc_name, "number": psc_crn, "data": psc}) 
                        logger.info(f"Found corporate PSC for {company_number}: {psc_name} (CRN: {psc_crn})")
        
        if not corporate_pscs:
            logger.info(f"No external corporate PSCs identified for {company_number}.")
            return None

        # Simple selection: return the first one found.
        # Could be enhanced to pick based on 'natures_of_control' strength.
        selected_psc = corporate_pscs[0] 
        logger.info(f"Selected corporate PSC for {company_number} as potential parent: {selected_psc['name']} (CRN: {selected_psc['number']})")
        return {"name": selected_psc["name"], "number": selected_psc["number"]}

    except Exception as e:
        logger.error(f"Unexpected error while processing PSC data for {company_number}: {e}", exc_info=True)
        return None


def analyze_company_group_structure(company_number: str, api_key: str, base_scratch_dir: _pl.Path, logger: logging.Logger, ocr_handler: Optional[OCRHandlerType] = None):
    """
    Analyzes the group structure of a company by fetching and reviewing filing metadata.
    Ensures document content is fetched in the order: JSON, then XHTML, then PDF (with OCR fallback if needed).
    """
    logger.info(f"Starting group structure analysis for {company_number}. OCR Handler: {type(ocr_handler).__name__ if ocr_handler else 'None'}.")
    report_messages = [f"Group Structure Analysis Report for {company_number} (Metadata Fetch Stage):"]
    parent_timeline = [] # Initialize parent_timeline
    visualization_data = None  # DOT language string for Graphviz - will remain placeholder for now
    suggested_ultimate_parent_cn = None # Placeholder for now
    downloaded_documents_content = [] # Initialize downloaded_documents_content

    try:
        # Determine date range for metadata fetching
        end_year = datetime.now().year
        start_year = end_year - METADATA_YEARS_TO_SCAN + 1
        logger.info(f"Setting document metadata scan range: {start_year}-{end_year} for categories: {RELEVANT_CATEGORIES_FOR_GROUP_STRUCTURE}")

        # Call get_ch_documents_metadata
        doc_metadata_items, company_profile, meta_error = get_ch_documents_metadata(
            company_no=company_number,
            api_key=api_key,
            categories=RELEVANT_CATEGORIES_FOR_GROUP_STRUCTURE,
            items_per_page=ITEMS_PER_PAGE_API,
            max_docs_to_fetch_meta=MAX_DOCS_TO_SCAN_PER_CATEGORY_API,
            target_docs_per_category_in_date_range=TARGET_DOCS_PER_CATEGORY_IN_RANGE_API,
            year_range=(start_year, end_year)
        )

        if meta_error:
            logger.error(f"Error fetching metadata for {company_number}: {meta_error}")
            report_messages.append(f"Error fetching document metadata: {meta_error}")
            # Optionally, still try to use company_profile if available
            if company_profile:
                report_messages.append(f"Company Profile for {company_number}: Name - {company_profile.get('company_name', 'N/A')}, Status - {company_profile.get('company_status', 'N/A')}")

        if company_profile:
            logger.info(f"Retrieved profile for {company_number}: {company_profile.get('company_name')}")
            report_messages.append(f"Company: {company_profile.get('company_name', company_number)} (Status: {company_profile.get('company_status', 'N/A')})")
            if company_profile.get("accounts", {}).get("accounting_requirement") == "group":
                 report_messages.append("Company profile suggests it prepares group accounts.")
            elif company_profile.get("accounts", {}).get("accounting_requirement") == "full" or company_profile.get("accounts", {}).get("accounting_requirement") == "small":
                 report_messages.append("Company profile suggests it prepares individual accounts (may include S409 if parent).")


        if not doc_metadata_items:
            logger.warning(f"No document metadata found for {company_number} matching criteria ({RELEVANT_CATEGORIES_FOR_GROUP_STRUCTURE}, {start_year}-{end_year}).")
            report_messages.append(f"No relevant document filings (accounts, confirmation statements) found within the last {METADATA_YEARS_TO_SCAN} years.")
        else:
            report_messages.append(f"Found {len(doc_metadata_items)} potentially relevant document filings in the last {METADATA_YEARS_TO_SCAN} years.")
            logger.info(f"Found {len(doc_metadata_items)} metadata items for {company_number}.")

            identified_accounts_docs = []
            identified_cs_docs = []

            for item in doc_metadata_items:
                doc_date = item.get("date", "N/A")
                doc_type = item.get("type", "N/A").upper()
                doc_desc = item.get("description", "N/A")
                doc_category = item.get("category", "N/A")
                
                log_message = f"  - Date: {doc_date}, Type: {doc_type}, Category: {doc_category}, Description: {doc_desc}"

                is_group_accounts = "group" in doc_desc.lower() or doc_type in ["AA", "GROUP"] # AA can be group accounts
                is_consolidated_accounts = "consolidated" in doc_desc.lower()
                is_full_accounts = "full" in doc_desc.lower() or doc_type in ["AA", "FULL"]
                is_small_full_accounts = "small full" in doc_desc.lower()

                if doc_category == "accounts":
                    if is_group_accounts or is_consolidated_accounts or is_full_accounts or is_small_full_accounts:
                        identified_accounts_docs.append(item)
                        report_messages.append(f"  -> Identified Accounts: {doc_date} - {doc_desc} (Type: {doc_type})")
                        logger.info(f"Identified relevant accounts: {log_message}")
                    else:
                        logger.debug(f"Skipping non-target accounts: {log_message}")
                elif doc_category == "confirmation-statement" or doc_type.startswith("CS") or doc_type.startswith("AR"): # AR for annual returns
                    identified_cs_docs.append(item)
                    report_messages.append(f"  -> Identified Confirmation Statement/Return: {doc_date} - {doc_desc} (Type: {doc_type})")
                    logger.info(f"Identified CS/AR: {log_message}")
            
            report_messages.append(f"Found {len(identified_accounts_docs)} relevant accounts filings and {len(identified_cs_docs)} confirmation statements/annual returns.")
            
            selected_docs_for_download = [] # Initialize list for documents selected for download

            if not identified_accounts_docs and not identified_cs_docs:
                report_messages.append("No specific group, full accounts, or confirmation statements identified in the fetched metadata for detailed S409 analysis.")
            else:
                report_messages.append("\\nDocument Selection Strategy:")
                logger.info("Starting document selection strategy.")

                # Helper to parse date string, robust to "N/A"
                def parse_filing_date(doc_item):
                    date_str = doc_item.get("date")
                    if date_str and date_str != "N/A":
                        try:
                            return datetime.strptime(date_str, "%Y-%m-%d")
                        except ValueError:
                            logger.warning(f"Could not parse date: {date_str} for doc {doc_item.get('description')}")
                            return datetime.min # Treat unparseable dates as very old
                    return datetime.min

                current_year = datetime.now().year
                for year_offset in range(METADATA_YEARS_TO_SCAN):
                    target_year = current_year - year_offset
                    
                    # --- Select Accounts for the target_year ---
                    yearly_accounts_candidates = [
                        doc for doc in identified_accounts_docs 
                        if doc.get("date", "").startswith(str(target_year))
                    ]
                    
                    selected_account_doc_for_year = None
                    if yearly_accounts_candidates:
                        # Sort by date (latest first), then by preference (group > full > small full)
                        yearly_accounts_candidates.sort(key=lambda d: (
                            parse_filing_date(d),
                            "group" in d.get("description", "").lower(), # True for group is better
                            "full" in d.get("description", "").lower() and not "small" in d.get("description", "").lower(), # True for full (not small) is better
                            "small full" in d.get("description", "").lower() # True for small full is next
                        ), reverse=True)
                        
                        selected_account_doc_for_year = yearly_accounts_candidates[0] # Assign the best candidate

                        if selected_account_doc_for_year: # Now this check makes sense
                            # Check for duplicates before adding
                            doc_meta_link = selected_account_doc_for_year.get("links", {}).get("document_metadata")
                            if doc_meta_link and not any(sel_doc.get("links", {}).get("document_metadata") == doc_meta_link for sel_doc in selected_docs_for_download):
                                selected_docs_for_download.append(selected_account_doc_for_year)
                                report_messages.append(f"  - Selected for Year {target_year} (Accounts): {selected_account_doc_for_year.get('date')} - {selected_account_doc_for_year.get('description')} (Type: {selected_account_doc_for_year.get('type')})")
                                logger.info(f"Selected Accounts for {target_year}: {selected_account_doc_for_year.get('description')}")
                            elif not doc_meta_link:
                                logger.warning(f"Accounts document for {target_year} ({selected_account_doc_for_year.get('description')}) has no document_metadata link.")
                            else: # Duplicate
                                logger.info(f"Accounts document for {target_year} ({selected_account_doc_for_year.get('description')}) already selected.")
                        # This else was tied to the "if selected_account_doc_for_year" which was always None before.
                        # Now, if yearly_accounts_candidates was populated, selected_account_doc_for_year will be too.
                        # The case for "no suitable accounts document found" is if yearly_accounts_candidates is empty.
                    else:
                        logger.info(f"No suitable accounts document found for {target_year} among identified accounts filings.")


                    # --- Select Confirmation Statement / Annual Return for the target_year ---
                    yearly_cs_candidates = [
                        doc for doc in identified_cs_docs
                        if doc.get("date", "").startswith(str(target_year))
                    ]
                    if yearly_cs_candidates:
                        # Sort by date (latest first)
                        yearly_cs_candidates.sort(key=parse_filing_date, reverse=True)
                        selected_cs_doc_for_year = yearly_cs_candidates[0]
                        # Check for duplicates before adding
                        cs_doc_meta_link = selected_cs_doc_for_year.get("links", {}).get("document_metadata")
                        if cs_doc_meta_link and not any(sel_doc.get("links", {}).get("document_metadata") == cs_doc_meta_link for sel_doc in selected_docs_for_download):
                            selected_docs_for_download.append(selected_cs_doc_for_year)
                            report_messages.append(f"  - Selected for Year {target_year} (CS/AR): {selected_cs_doc_for_year.get('date')} - {selected_cs_doc_for_year.get('description')} (Type: {selected_cs_doc_for_year.get('type')})")
                            logger.info(f"Selected CS/AR for {target_year}: {selected_cs_doc_for_year.get('description')}")
                        elif not cs_doc_meta_link:
                            logger.warning(f"CS/AR document for {target_year} ({selected_cs_doc_for_year.get('description')}) has no document_metadata link.")
                        else: # Duplicate
                            logger.info(f"CS/AR document for {target_year} ({selected_cs_doc_for_year.get('description')}) already selected.")
                    else:
                        logger.info(f"No CS/AR document found for {target_year}.")

                # Initialize downloaded_documents_content before the document processing block
                downloaded_documents_content = []

                if selected_docs_for_download:
                    report_messages.append(f"\\nTotal documents selected for download: {len(selected_docs_for_download)}")
                    logger.info(f"Total documents selected for download: {len(selected_docs_for_download)}")
                    
                    report_messages.append("\\nAttempting to download and parse selected documents:")
                    logger.info("Starting download and parsing process for selected documents.")

                    for doc_item_to_download in selected_docs_for_download:
                        # doc_link = doc_item_to_download.get("links", {}).get("document_metadata") # No longer directly used here
                        doc_description = doc_item_to_download.get('description', 'Unknown document')
                        doc_date_str = doc_item_to_download.get('date', 'N/A')

                        # The doc_link is inside doc_item_to_download["links"]["document_metadata"]
                        # which _fetch_document_content_from_ch will use internally.
                        if not doc_item_to_download.get("links", {}).get("document_metadata"):
                            logger.warning(f"No document_metadata link for {doc_description}. Skipping download.")
                            report_messages.append(f"  - Skipping {doc_description} (no download link).")
                            parsed_data_bundle = {
                                "metadata": doc_item_to_download,
                                "parsed_data": {"errors": ["No download link found"], "document_description": doc_description, "document_date": doc_date_str}
                            }
                            downloaded_documents_content.append(parsed_data_bundle)
                            continue

                        logger.info(f"Processing document: {doc_description} ({doc_date_str}) for company {company_number}")

                        # Directly fetch using the function from ch_api_utils
                        # It takes company_no (our company_number) and item_details (our doc_item_to_download)
                        # It returns: content_data, content_type_fetched, error_message_or_None
                        raw_content, content_type_fetched, fetch_error_msg = _fetch_document_content_from_ch(
                            company_number,       # company_no argument
                            doc_item_to_download  # item_details argument
                        )

                        # --- Update for new return structure: content_dict, fetched_types, error_msg ---
                        if fetch_error_msg:
                            logger.error(f"Failed to download/fetch content for {doc_description}: {fetch_error_msg}")
                            report_messages.append(f"  - Failed to download {doc_description}: {fetch_error_msg}")
                            parsed_data_bundle = {
                                "metadata": doc_item_to_download,
                                "parsed_data": {"errors": [f"Download/fetch failed: {fetch_error_msg}"], "document_description": doc_description, "document_date": doc_date_str}
                            }
                            downloaded_documents_content.append(parsed_data_bundle)
                            continue

                        if raw_content:
                            logger.info(f"Successfully fetched content for {doc_description}. Types fetched: {content_type_fetched}. Parsing...")
                            report_messages.append(f"  - Downloaded {doc_description} (Types: {content_type_fetched}). Parsing...")

                            # Prefer JSON, then XHTML, then PDF for parsing
                            parse_type = None
                            parse_content = None
                            for t in ["json", "xhtml", "pdf"]:
                                if t in raw_content:
                                    parse_type = t
                                    parse_content = raw_content[t]
                                    break

                            if not parse_type:
                                logger.warning(f"No usable content type found for parsing for {doc_description}. Types fetched: {content_type_fetched}")
                                report_messages.append(f"  - No usable content type found for parsing for {doc_description}. Types fetched: {content_type_fetched}")
                                parsed_data_bundle = {
                                    "metadata": doc_item_to_download,
                                    "parsed_data": {"errors": ["No usable content type found for parsing"], "document_description": doc_description, "document_date": doc_date_str}
                                }
                                downloaded_documents_content.append(parsed_data_bundle)
                                continue

                            # Corrected argument order for _parse_document_content
                            parsed_info = _parse_document_content(
                                doc_content_data=raw_content, # The dictionary containing the actual fetched content (e.g., {'xhtml': '...', 'pdf_bytes': b'...'})
                                fetched_content_types=content_type_fetched, # List of types successfully fetched (e.g., ['xhtml', 'pdf'])
                                company_no=company_number,        # The company number being analyzed
                                doc_metadata=doc_item_to_download, # The metadata dictionary for the document
                                logger=logger,                 # The logger instance
                                ocr_handler=ocr_handler        # Pass the ocr_handler
                            )

                            current_doc_bundle = {
                                "metadata": doc_item_to_download,
                                "parsed_data": parsed_info
                            }
                            downloaded_documents_content.append(current_doc_bundle)

                            if parsed_info.get("errors"):
                                report_messages.append(f"    - Parsed {doc_description} with errors: {parsed_info['errors']}")
                            else:
                                subs_count = len(parsed_info.get('subsidiaries', []))
                                parent_name_info = parsed_info.get('parent_company_name', 'N/A')
                                parent_crn_info = parsed_info.get('parent_company_number', 'N/A')
                                parent_display = f"Parent: {parent_name_info} ({parent_crn_info})" if parsed_info.get('parent_company_name') else "No parent info extracted"
                                report_messages.append(f"    - Parsed {doc_description} successfully. S409: {parsed_info.get('s409_info_found', False)}. Subsidiaries: {subs_count}. {parent_display}.")
                        else:
                            logger.warning(f"No content returned for {doc_description}, and no specific error from fetcher. This is unexpected.")
                            report_messages.append(f"  - No content returned for {doc_description} (unexpected state).")
                            parsed_data_bundle = {
                                "metadata": doc_item_to_download,
                                "parsed_data": {"errors": ["No content returned from download and no specific error (unexpected)"], "document_description": doc_description, "document_date": doc_date_str}
                            }
                            downloaded_documents_content.append(parsed_data_bundle)
                    # End of for loop for doc_item_to_download

        # --- Parent Timeline Extraction and Ultimate Parent Determination ---
        logger.info(f"Extracting parent timeline and determining current ultimate parent for {company_number}.")
        
        parent_timeline, parent_identification_messages = extract_parent_timeline(
            company_number, 
            api_key, 
            logger, 
            parsed_docs_data=downloaded_documents_content
        )
        report_messages.extend(parent_identification_messages) # Add timeline messages to the main report

        # Determine current ultimate parent from the timeline
        # Variables to store findings from document analysis
        doc_suggested_ultimate_parent_cn = None
        doc_is_input_ultimate = True 
        doc_ultimate_parent_name_for_report = None

        if parent_timeline:
            latest_parent_entry_with_crn = None
            for entry in reversed(parent_timeline):
                if entry.get('parent_number'):
                    latest_parent_entry_with_crn = entry
                    break
            
            if latest_parent_entry_with_crn:
                potential_parent_crn = latest_parent_entry_with_crn['parent_number']
                potential_parent_name = latest_parent_entry_with_crn['parent_name']
                if potential_parent_crn.strip().upper() != company_number.strip().upper():
                    doc_suggested_ultimate_parent_cn = potential_parent_crn
                    doc_ultimate_parent_name_for_report = potential_parent_name or potential_parent_crn
                    doc_is_input_ultimate = False
                    report_messages.append(f"Docs: Based on latest document ({latest_parent_entry_with_crn['date']}), parent appears to be: {doc_ultimate_parent_name_for_report} (CRN: {doc_suggested_ultimate_parent_cn}).")
                    logger.info(f"Docs: Parent for {company_number} from timeline: {doc_ultimate_parent_name_for_report} ({doc_suggested_ultimate_parent_cn})")
                else:
                    doc_suggested_ultimate_parent_cn = company_number
                    doc_ultimate_parent_name_for_report = company_profile.get('company_name', company_number) if company_profile else company_number
                    report_messages.append(f"Docs: Latest documents indicate {doc_ultimate_parent_name_for_report} (CRN: {company_number}) is its own parent.")
                    logger.info(f"Docs: Input company {company_number} confirmed as parent from timeline (self-reference).")
            else:
                report_messages.append("Docs: No parent with CRN identified in timeline. Assuming input company is ultimate or parent is by name only.")
                logger.info(f"Docs: No parent with CRN in timeline for {company_number}.")
                doc_suggested_ultimate_parent_cn = company_number 
                doc_ultimate_parent_name_for_report = company_profile.get('company_name', company_number) if company_profile else company_number
        else:
            report_messages.append("Docs: Parent timeline empty. Assuming input company is ultimate parent.")
            logger.info(f"Docs: Parent timeline empty for {company_number}.")
            doc_suggested_ultimate_parent_cn = company_number
            doc_ultimate_parent_name_for_report = company_profile.get('company_name', company_number) if company_profile else company_number

        # --- PSC Analysis Fallback ---
        logger.info(f"Attempting PSC analysis for {company_number}.")
        report_messages.append("\\n--- PSC (Persons with Significant Control) Analysis ---")
        psc_parent_info = _get_corporate_psc_parent_info(company_number, api_key, logger)
        
        # Initialize final decision variables with document-based findings
        final_suggested_ultimate_parent_cn = doc_suggested_ultimate_parent_cn
        final_ultimate_parent_name = doc_ultimate_parent_name_for_report
        final_is_input_ultimate = doc_is_input_ultimate

        if psc_parent_info:
            psc_parent_name = psc_parent_info['name']
            psc_parent_crn = psc_parent_info['number']
            report_messages.append(f"PSC Analysis: Found corporate PSC: {psc_parent_name} (CRN: {psc_parent_crn}).")
            logger.info(f"PSC for {company_number}: Found corporate PSC {psc_parent_name} ({psc_parent_crn}).")

            if final_is_input_ultimate and psc_parent_crn.strip().upper() != company_number.strip().upper():
                report_messages.append(f"PSC data suggests a different ultimate parent: {psc_parent_name} (CRN: {psc_parent_crn}). Overriding document-based finding (or lack thereof).")
                logger.info(f"PSC Override: Using PSC {psc_parent_name} ({psc_parent_crn}) as ultimate parent for {company_number}.")
                final_suggested_ultimate_parent_cn = psc_parent_crn
                final_ultimate_parent_name = psc_parent_name
                final_is_input_ultimate = False
            elif not final_is_input_ultimate and final_suggested_ultimate_parent_cn.strip().upper() != psc_parent_crn.strip().upper() and psc_parent_crn.strip().upper() != company_number.strip().upper():
                report_messages.append(f"PSC Warning: Corporate PSC ({psc_parent_name}, CRN: {psc_parent_crn}) differs from document-derived parent ({final_ultimate_parent_name}, CRN: {final_suggested_ultimate_parent_cn}). Document-derived parent will be prioritized. Further investigation may be needed.")
                logger.warning(f"PSC Conflict for {company_number}: Doc parent {final_ultimate_parent_name} ({final_suggested_ultimate_parent_cn}), PSC parent {psc_parent_name} ({psc_parent_crn}). Sticking with doc parent.")
            elif final_suggested_ultimate_parent_cn.strip().upper() == psc_parent_crn.strip().upper():
                report_messages.append(f"PSC Confirmation: Corporate PSC ({psc_parent_name}, CRN: {psc_parent_crn}) aligns with document-derived parent.")
                logger.info(f"PSC Confirmation for {company_number}: PSC {psc_parent_name} ({psc_parent_crn}) matches doc parent.")
            # If psc_parent_crn is company_number itself, it's not an external parent, so no change to final_is_input_ultimate needed here.
        else:
            report_messages.append("PSC Analysis: No overriding corporate PSC identified or PSC data unavailable/not configured.")
            logger.info(f"PSC for {company_number}: No overriding corporate PSC identified.")

        # Update suggested_ultimate_parent_cn for return based on final decision
        suggested_ultimate_parent_cn = final_suggested_ultimate_parent_cn

        # --- Prepare for DOT Graph Generation ---
        # Fetch profile for the final suggested ultimate parent if it's different from the input company
        final_ultimate_parent_crn_for_graph = company_number 
        ultimate_parent_name_for_graph = company_profile.get('company_name', company_number) if company_profile else company_number

        if not final_is_input_ultimate and final_suggested_ultimate_parent_cn:
            final_ultimate_parent_crn_for_graph = final_suggested_ultimate_parent_cn
            # Try to get profile for this parent
            parent_profile = ch_api_utils.get_company_profile(final_suggested_ultimate_parent_cn, api_key)
            if parent_profile:
                ultimate_parent_name_for_graph = parent_profile.get('company_name', final_suggested_ultimate_parent_cn)
                report_messages.append(f"Ultimate Parent Profile ({final_suggested_ultimate_parent_cn}): {ultimate_parent_name_for_graph}, Status: {parent_profile.get('company_status', 'N/A')}.")
            else:
                ultimate_parent_name_for_graph = final_ultimate_parent_name # Use name from docs/PSC if profile fetch fails
                report_messages.append(f"Could not fetch profile for ultimate parent CRN: {final_suggested_ultimate_parent_cn}. Using name: {ultimate_parent_name_for_graph}.")
        elif final_is_input_ultimate : # Analyzed company is ultimate
            final_ultimate_parent_crn_for_graph = company_number
            ultimate_parent_name_for_graph = company_profile.get('company_name', company_number) if company_profile else company_number
            report_messages.append(f"Final determination: {ultimate_parent_name_for_graph} (CRN: {company_number}) is the ultimate parent entity in this context.")


        # --- DOT Graph Generation (using final_ultimate_parent_crn_for_graph and ultimate_parent_name_for_graph) ---
        logger.info(f"Generating DOT string for visualization. Analyzed: {company_number}. Ultimate Parent: {final_ultimate_parent_crn_for_graph}")

        dot_lines = [
            "digraph CompanyGroup {",
            "  rankdir=TB;",
            "  node [shape=box, style=filled, fontname=\"sans-serif\"];",
            "  edge [fontname=\"sans-serif\"];",
        ]
        
        analyzed_company_name_for_graph = company_profile.get('company_name', company_number) if company_profile else company_number
        
        # Node for the analyzed company
        dot_lines.append(f'  "{company_number}" [label="{analyzed_company_name_for_graph}\\n({company_number})\\nAnalyzed Entity", color="orange"];')

        # Node and edge for the ultimate parent
        if final_ultimate_parent_crn_for_graph.strip().upper() != company_number.strip().upper():
            up_name_label = ultimate_parent_name_for_graph if ultimate_parent_name_for_graph else final_ultimate_parent_crn_for_graph
            dot_lines.append(f'  "{final_ultimate_parent_crn_for_graph}" [label="{up_name_label}\\n({final_ultimate_parent_crn_for_graph})\\nCurrent Ultimate Parent", color="lightgreen"];')
            dot_lines.append(f'  "{final_ultimate_parent_crn_for_graph}" -> "{company_number}" [label="owns (implied)"];') # Changed label slightly
        else: # Analyzed company is the ultimate parent
            # Modify the analyzed company's node to reflect it's also the ultimate parent
            dot_lines[-1] = f'  "{company_number}" [label="{analyzed_company_name_for_graph}\\n({company_number})\\nAnalyzed Entity & Current Ultimate Parent", color="lightgreen"];'
        
        all_identified_subsidiaries = {} # This should be populated from parsed_docs_content
        if downloaded_documents_content:
            for doc_bundle in downloaded_documents_content:
                parsed_data = doc_bundle.get("parsed_data")
                if parsed_data and parsed_data.get("subsidiaries"):
                    for sub in parsed_data["subsidiaries"]:
                        sub_name = sub.get("name", "Unknown Subsidiary")
                        sub_number = sub.get("number")
                        if sub_number and sub_number not in all_identified_subsidiaries: # Corrected syntax here
                            all_identified_subsidiaries[sub_number] = sub_name
                            # Add subsidiary node and edge to DOT
                            dot_lines.append(f'  "{sub_number}" [label="{sub_name}\\n({sub_number})\\nIdentified Subsidiary", color="lightblue"];')
                            dot_lines.append(f'  "{company_number}" -> "{sub_number}" [label="subsidiary of"];')
                        elif sub_number and sub_number in all_identified_subsidiaries:
                            logger.debug(f"Subsidiary {sub_name} ({sub_number}) already added to graph.")
                        elif not sub_number and sub_name != "Unknown Subsidiary": # Subsidiary by name only
                            # Create a unique ID for name-only subsidiary to avoid node conflicts if multiple have same name
                            name_only_id = f"name_only_sub_{sub_name.replace(' ', '_').lower()}"
                            if name_only_id not in all_identified_subsidiaries: # Check if this conceptual ID is already used
                                all_identified_subsidiaries[name_only_id] = sub_name # Store it to prevent re-adding
                                dot_lines.append(f'  "{name_only_id}" [label="{sub_name}\\n(CRN not identified)\\nIdentified Subsidiary", color="lightgray"];')
                                dot_lines.append(f'  "{company_number}" -> "{name_only_id}" [label="subsidiary of (name only)"];')


        dot_lines.append("}")
        visualization_data = "\n".join(dot_lines)
        report_messages.append("\\nGroup structure visualization DOT string generated.")
        logger.info("DOT string generated.")

        # --- Enhance Report Messages ---
        parsed_summary = ["\\nSummary of Information from Parsed Documents:"]
        doc_evidence_count = 0
        if downloaded_documents_content: # Check if the list is not empty
            for doc_bundle in downloaded_documents_content: # Corrected variable name
                parsed = doc_bundle.get('parsed_data')
                meta = doc_bundle.get('metadata')
                if parsed and meta:
                    doc_desc = meta.get('description', 'Unknown Document')
                    doc_date = meta.get('date', 'N/A')
                    if parsed.get('parent_company_name') or parsed.get('s409_info_found') or parsed.get('subsidiaries'):
                        doc_evidence_count +=1
                        summary_line = f"- Doc ({doc_date}, {doc_desc}):"
                        if parsed.get('parent_company_name'):
                            summary_line += f" Mentions parent: {parsed.get('parent_company_name')} ({parsed.get('parent_company_number', 'CRN N/A')})."
                        if parsed.get('s409_info_found'):
                            summary_line += " Contains S409/group exemption related statements."
                        if parsed.get('subsidiaries'):
                            summary_line += f" Mentions {len(parsed.get('subsidiaries'))} potential subsidiaries."
                        parsed_summary.append(summary_line)
        if doc_evidence_count > 0:
            report_messages.extend(parsed_summary)
        else:
            report_messages.append("\\nNo specific group structure information (parent, S409, subsidiaries) was extracted by the parser from the selected documents.")

    except Exception as e:
        logger.error(f"Error during group structure metadata analysis for {company_number}: {e}", exc_info=True)
        report_messages.append(f"An unexpected error occurred during metadata analysis: {e}")
    
    logger.info(f"Metadata fetching stage for {company_number} complete. Report messages: {len(report_messages)}")
    # Return parent_timeline along with other results
    return visualization_data, report_messages, suggested_ultimate_parent_cn, parent_timeline

# --- Helper function for parsing document content ---
def _parse_document_content(
    doc_content_data: Dict[str, Any], 
    fetched_content_types: List[str], # This argument might be redundant if doc_content_data.keys() is used directly
    company_no: str, 
    doc_metadata: Dict[str, Any], 
    logger: logging.Logger,
    ocr_handler: Optional[OCRHandlerType] = None  # Added ocr_handler parameter
) -> Dict[str, Any]:
    """
    Parses document content, prioritizing JSON, XHTML, XML, then PDF (with OCR).
    Extracts text and attempts to find basic parent/S409 info.
    """
    # Ensure imports for typing and re are available if not already at module level
    # from typing import Dict, List, Any, Optional (already imported)
    # import re (already imported)

    parsed_result = {
        "document_date": doc_metadata.get("date"),
        "document_description": doc_metadata.get("description"),
        "text_content": None,
        "parent_company_name": None, 
        "parent_company_number": None,
        "s409_info_found": False, 
        "extraction_error": None,
        "source_format_parsed": None,
        "pages_ocrd": 0 
    }

    content_to_parse = None
    content_type_to_parse = None

    preferred_types = ["json", "xhtml", "xml"]

    for doc_type in preferred_types:
        if doc_type in doc_content_data:
            content_to_parse = doc_content_data[doc_type]
            content_type_to_parse = doc_type
            logger.info(f"GS: Found {doc_type.upper()} content for parsing for '{parsed_result.get('document_description', 'N/A')}'.")
            break
    
    if not content_to_parse and 'pdf' in doc_content_data:
        if ocr_handler: # Only consider PDF if an OCR handler is provided
            content_to_parse = doc_content_data['pdf']
            content_type_to_parse = 'pdf'
            logger.info(f"GS: No JSON/XHTML/XML. Using PDF content with OCR for '{parsed_result.get('document_description', 'N/A')}'.")
        else:
            logger.warning(f"GS: PDF content available for '{parsed_result.get('document_description', 'N/A')}', but no OCR handler provided for group structure analysis. PDF will not be processed.")
            parsed_result["extraction_error"] = "PDF content found, but OCR handler not provided for group structure analysis."

    if content_to_parse and content_type_to_parse:
        parsed_result["source_format_parsed"] = content_type_to_parse.upper()
        try:
            actual_ocr_handler_for_extraction = ocr_handler if content_type_to_parse == 'pdf' else None
            
            if actual_ocr_handler_for_extraction:
                logger.info(f"GS: Calling extract_text_from_document with OCR for PDF: '{parsed_result.get('document_description', 'N/A')}'.")
            elif content_type_to_parse == 'pdf' and not actual_ocr_handler_for_extraction: # Should not happen if logic above is correct
                 logger.warning(f"GS: PDF selected for parsing, but OCR handler is effectively None for this call for '{parsed_result.get('document_description', 'N/A')}'.")

            extracted_text, pages_processed_in_doc, extraction_error = extract_text_from_document(
                document_content=content_to_parse,
                content_type_input=content_type_to_parse,
                company_number_for_logging=company_no,
                ocr_handler=actual_ocr_handler_for_extraction 
            )
            
            if content_type_to_parse == 'pdf' and actual_ocr_handler_for_extraction and pages_processed_in_doc > 0:
                parsed_result["pages_ocrd"] = pages_processed_in_doc
                logger.info(f"GS: OCR processed {pages_processed_in_doc} pages for PDF: '{parsed_result.get('document_description', 'N/A')}'.")

            if extraction_error:
                logger.error(f"GS: Error extracting text from {content_type_to_parse.upper()} for '{parsed_result.get('document_description', 'N/A')}': {extraction_error}")
                parsed_result["extraction_error"] = str(extraction_error)
            elif not extracted_text: # Check if extracted_text is None or empty
                logger.warning(f"GS: No text extracted from {content_type_to_parse.upper()} for '{parsed_result.get('document_description', 'N/A')}'.")
                parsed_result["text_content"] = "" 
            else:
                parsed_result["text_content"] = extracted_text
                logger.info(f"GS: Successfully extracted text ({len(extracted_text)} chars) from {content_type_to_parse.upper()} for '{parsed_result.get('document_description', 'N/A')}'.")
                
                # Placeholder for more sophisticated parsing logic of the extracted_text
                text_lower = extracted_text.lower()
                # Example: Search for "ultimate parent company is" followed by the name.
                # This regex is basic and might need refinement.
                parent_match = re.search(r"(?:ultimate parent company is|the parent company is|parent undertaking is)\s*([^,.(]+)", text_lower)
                if parent_match:
                    parent_name_candidate = parent_match.group(1).strip()
                    # Avoid matching the company itself if its name appears in such a phrase
                    # This requires company_profile to be available or company_name passed in
                    # For now, a simple assignment.
                    parsed_result["parent_company_name"] = html.unescape(parent_name_candidate.title()) # Basic cleaning
                    logger.info(f"GS: Potential parent name found: '{parsed_result['parent_company_name']}' in '{parsed_result.get('document_description', 'N/A')}'.")

                if "s409" in text_lower or "section 409" in text_lower or "s.409" in text_lower:
                    parsed_result["s409_info_found"] = True
                    logger.info(f"GS: S409/Section 409 mention found in '{parsed_result.get('document_description', 'N/A')}'.")

        except Exception as e_extract:
            logger.error(f"GS: Exception during text extraction from {content_type_to_parse.upper()} for '{parsed_result.get('document_description', 'N/A')}': {e_extract}", exc_info=True)
            parsed_result["extraction_error"] = f"General extraction exception: {str(e_extract)}"
    
    elif not parsed_result["extraction_error"]: 
        # This case means no content_to_parse was set (e.g. PDF without OCR handler, or no suitable types at all)
        # and no specific error was set during the type selection.
        logger.warning(f"GS: No suitable content (JSON, XHTML, XML, or PDF with OCR) was processed for doc: '{parsed_result.get('document_description', 'N/A')}' for company {company_no}.")
        # If the error was already set (e.g. "PDF content found, but OCR handler not provided..."), don't overwrite.
        if not parsed_result["extraction_error"]:
             parsed_result["extraction_error"] = "No suitable content type found or processed."
    return parsed_result

def render_group_structure_ui(api_key: str, base_scratch_dir: _pl.Path, logger: logging.Logger, ocr_handler: Optional[OCRHandlerType] = None):
    """
    Renders the UI for Company Group Structure Analysis.
    """
    st.header("🏢 Company Group Structure Analysis")
    st.markdown("""
        Enter a company number to analyze its group structure. The tool will attempt to identify parent companies
        and subsidiaries by reviewing Companies House filings.
    """)

    # Initialize session state keys if they don't exist (idempotent)
    if 'group_structure_cn_for_analysis' not in st.session_state:
        st.session_state.group_structure_cn_for_analysis = ""
    if 'group_structure_report' not in st.session_state:
        st.session_state.group_structure_report = []
    if 'group_structure_viz_data' not in st.session_state:
        st.session_state.group_structure_viz_data = None
    if 'suggested_parent_cn_for_rerun' not in st.session_state: # For re-running with a suggested parent
        st.session_state.suggested_parent_cn_for_rerun = None
    if 'group_structure_parent_timeline' not in st.session_state:
        st.session_state.group_structure_parent_timeline = []


    # Company Number Input
    current_cn_input = st.text_input(
        "Enter Company Number for Group Analysis",
        value=st.session_state.group_structure_cn_for_analysis,
        key="group_structure_cn_input_field",
        placeholder="E.g., 00445790"
    )
    # Update session state if the input field changes
    if current_cn_input != st.session_state.group_structure_cn_for_analysis:
        st.session_state.group_structure_cn_for_analysis = current_cn_input
        # Clear previous results when company number changes
        st.session_state.group_structure_report = []
        st.session_state.group_structure_viz_data = None
        st.session_state.suggested_parent_cn_for_rerun = None
        st.session_state.group_structure_parent_timeline = []


    col_analyze, col_rerun_parent = st.columns([1,1])

    with col_analyze:
        if st.button("🔍 Analyze Structure", key="analyze_group_structure_button", help="Fetch and analyze filings to determine group structure.", type="primary"):
            if not st.session_state.group_structure_cn_for_analysis:
                st.warning("Please enter a company number.")
            elif not api_key:
                st.error("Companies House API Key is not configured. Please set it in the application settings or .env file.")
            else:
                cn_to_analyze = st.session_state.group_structure_cn_for_analysis.strip().upper()
                st.session_state.group_structure_cn_for_analysis = cn_to_analyze # Store the cleaned version

                with st.spinner(f"Analyzing group structure for {cn_to_analyze}... This may take a few minutes."):
                    try:
                        logger.info(f"UI: Initiating group structure analysis for {cn_to_analyze}.")
                        # Call the backend analysis function
                        viz_data, report_msgs, suggested_parent_cn, parent_timeline_data = analyze_company_group_structure(
                            company_number=cn_to_analyze,
                            api_key=api_key,
                            base_scratch_dir=base_scratch_dir, # Ensure this path is valid and writable
                            logger=logger,
                            ocr_handler=ocr_handler
                        )
                        st.session_state.group_structure_viz_data = viz_data
                        st.session_state.group_structure_report = report_msgs
                        st.session_state.suggested_parent_cn_for_rerun = suggested_parent_cn
                        st.session_state.group_structure_parent_timeline = parent_timeline_data

                        logger.info(f"UI: Analysis complete for {cn_to_analyze}. Suggested parent: {suggested_parent_cn}")
                        st.success(f"Analysis complete for {cn_to_analyze}.")

                    except Exception as e:
                        logger.error(f"UI: Error during group structure analysis for {cn_to_analyze}: {e}", exc_info=True)
                        st.error(f"An error occurred during analysis: {e}")
                        st.session_state.group_structure_report = [f"Error: {e}"]
                        st.session_state.group_structure_viz_data = None
                        st.session_state.suggested_parent_cn_for_rerun = None
                        st.session_state.group_structure_parent_timeline = []
    
    with col_rerun_parent:
        if st.session_state.get('suggested_parent_cn_for_rerun') and st.session_state.suggested_parent_cn_for_rerun != st.session_state.group_structure_cn_for_analysis:
            if st.button(f"↪️ Analyze Suggested Parent: {st.session_state.suggested_parent_cn_for_rerun}", key="rerun_with_suggested_parent_button"):
                st.session_state.group_structure_cn_for_analysis = st.session_state.suggested_parent_cn_for_rerun
                # Clear previous specific results before re-running for the new parent
                st.session_state.group_structure_report = []
                st.session_state.group_structure_viz_data = None
                st.session_state.suggested_parent_cn_for_rerun = None # Clear this so the button doesn't reappear for the same parent
                st.session_state.group_structure_parent_timeline = []
                logger.info(f"UI: User triggered re-analysis for suggested parent: {st.session_state.group_structure_cn_for_analysis}")
                st.rerun() # Rerun to trigger analysis with the new company number in the input field

    # --- Display Results ---
    if st.session_state.group_structure_report or st.session_state.group_structure_viz_data:
        st.markdown("---")
        st.subheader("Analysis Results")

        tab_report, tab_viz, tab_timeline = st.tabs(["📜 Report Log", "📊 Visual Diagram", "⏳ Parent Timeline"])

        with tab_report:
            if st.session_state.group_structure_report:
                st.markdown("##### Group Structure Report & Analysis Log")
                # Join report messages into a single string with markdown line breaks
                report_display_text = "\n".join(st.session_state.group_structure_report).replace("\\n", "\n---\n")
                st.text_area("Report Details:", value=report_display_text, height=400, disabled=True, key="group_structure_report_text_area")
            else:
                st.info("No report generated yet, or analysis was not run.")

        with tab_viz:
            if st.session_state.group_structure_viz_data:
                st.markdown("##### Visual Diagram of Identified Structure")
                try:
                    st.graphviz_chart(st.session_state.group_structure_viz_data, use_container_width=True)
                except Exception as e_gv:
                    st.error(f"Could not render visualization: {e_gv}. Ensure Graphviz is installed and in PATH if rendering locally, or check DOT string syntax.")
                    logger.error(f"Error rendering Graphviz chart: {e_gv}. DOT data: {st.session_state.group_structure_viz_data[:500]}...") # Log snippet of DOT
            else:
                st.info("No visualization data available. Run analysis to generate the diagram.")
        
        with tab_timeline:
            if st.session_state.group_structure_parent_timeline:
                st.markdown("##### Parent Company Timeline")
                st.write("This timeline shows mentions of parent companies found in historical documents, ordered by document date (oldest first).")
                
                # Convert timeline data to a DataFrame for better display
                try:
                    import pandas as pd # type: ignore
                    df_timeline = pd.DataFrame(st.session_state.group_structure_parent_timeline)
                    # Select and rename columns for display
                    df_timeline_display = df_timeline.rename(columns={
                        'date': 'Document Date',
                        'parent_name': 'Reported Parent Name',
                        'parent_number': 'Reported Parent CRN',
                        'source_doc_desc': 'Source Document',
                        's409_related': 'S409 Mentioned'
                    })
                    st.dataframe(df_timeline_display, use_container_width=True)
                except ImportError:
                    st.warning("Pandas library not available. Displaying timeline as raw data.")
                    for entry in st.session_state.group_structure_parent_timeline:
                        st.json(entry) # Display as JSON if pandas is not there
                except Exception as e_timeline_df:
                    st.error(f"Could not display parent timeline: {e_timeline_df}")
                    logger.error(f"Error creating DataFrame for parent timeline: {e_timeline_df}")

            else:
                st.info("No parent company timeline data extracted from documents, or analysis not run.")
    else:
        st.info("Enter a company number and click 'Analyze Structure' to see results.")

# ... rest of the file (e.g., _parse_document_content, analyze_company_group_structure, etc.)
# Ensure analyze_company_group_structure and its dependencies are correctly defined in this file or imported.

# Add this at the end of the file to ensure the function is always defined
# This is a fallback implementation that will be used if there are import errors
# that prevent the real implementation from working

# Check if the function is already defined
if 'analyze_company_group_structure' not in globals():
    def analyze_company_group_structure(company_number, api_key, base_scratch_dir, logger, ocr_handler=None):
        """
        Stub implementation of analyze_company_group_structure that returns placeholder data.
        This will be used if the real implementation fails due to import errors.
        """
        logger.warning(f"Using stub implementation of analyze_company_group_structure for {company_number}")
        
        # Return placeholder data
        viz_data = """
        digraph G {
            rankdir=BT;
            node [shape=box, style=filled, color=lightblue];
            
            "COMPANY: {0}" [label="COMPANY: {0}\nNOTE: This is a placeholder graph.\nReal analysis unavailable due to import errors."];
        }
        """.format(company_number)
        
        report_messages = [
            f"Analysis for company {company_number} could not be performed.",
            "The Group Structure analysis functionality is not fully available.",
            "Check console logs for more details about import errors."
        ]
        
        suggested_parent = None
        timeline_data = []
        
        return viz_data, report_messages, suggested_parent, timeline_data
