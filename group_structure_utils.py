"""
Utilities for analyzing and visualizing company group structures from Companies House data.
Refactored for a staged, user-driven analysis process.
v4: Addressed Pylance diagnostics: added hashlib import and st.text mock.
"""
import pathlib as _pl
import logging 
from datetime import datetime, timedelta 
import json 
import re 
import html 
import hashlib # Added import for hashlib
from typing import Optional, Dict, List, Any, Callable, Tuple, TypeAlias, Literal, Set
from collections import defaultdict

# --- Logger Setup ---
try:
    from config import logger, MIN_MEANINGFUL_TEXT_LEN, get_openai_client, OPENAI_MODEL_DEFAULT
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("group_structure_utils: Using fallback logger configuration.")
    MIN_MEANINGFUL_TEXT_LEN = 200
    def get_openai_client() -> Optional[object]:
        logger.warning("group_structure_utils: get_openai_client fallback used. OpenAI features unavailable.")
        return None
    OPENAI_MODEL_DEFAULT = "gpt-4o"

# --- Type Aliases ---
JSONObj: TypeAlias = Dict[str, Any]
AnalysisMode = Literal["profile_check", "find_parent", "list_subsidiaries", "subsidiary_details"]

# --- OCR Handler Type Definition ---
try:
    from text_extraction_utils import extract_text_from_document, OCRHandlerType
    logger.info("group_structure_utils: Successfully imported OCRHandlerType and extract_text_from_document from text_extraction_utils.")
except ImportError as _imp_err:
    logger.warning(f"group_structure_utils: text_extraction_utils not importable: {_imp_err}. Using generic OCRHandlerType and dummy extract_text_from_document.")
    OCRHandlerType = Callable[[bytes, str], Tuple[str, int, Optional[str]]] 
    def extract_text_from_document(*args: Any, **kwargs: Any) -> Tuple[str, int, Optional[str]]:
        logger.error("group_structure_utils: extract_text_from_document called but backend (text_extraction_utils) not available.")
        return "Error: OCR backend missing", 0, "OCR backend missing"

# --- CH API Utilities Import ---
ch_api_utils_available = False
try:
    from ch_api_utils import get_ch_documents_metadata, get_company_profile, get_company_pscs, _fetch_document_content_from_ch
    ch_api_utils_available = True
    logger.info("group_structure_utils: Successfully imported functions from ch_api_utils.")
except ImportError as e_ch_api:
    logger.error(f"group_structure_utils: Failed to import from ch_api_utils: {e_ch_api}. Defining stubs.")
    def get_ch_documents_metadata(*args: Any, **kwargs: Any) -> Tuple[List[Any], Optional[Dict[str, Any]], Optional[str]]: 
        return [], None, "CH API utils (get_ch_documents_metadata) not available."
    def get_company_profile(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return None
    def _fetch_document_content_from_ch(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
        return {}, [], "CH API utils (_fetch_document_content_from_ch) not available."

# --- AWS Textract Utilities Import ---
try:
    from aws_textract_utils import perform_textract_ocr
except ImportError:
    def perform_textract_ocr(*args: Any, **kwargs: Any) -> Tuple[str, int, Optional[str]]: # type: ignore
        logger.error("group_structure_utils: perform_textract_ocr (stub) called."); return "OCR N/A",0,"OCR N/A"

# --- Streamlit Import & Mocking ---
class MockSessionState:
    def __init__(self): 
        self._state: Dict[str, Any] = {}
    def __getattr__(self, name: str) -> Any: 
        return self._state.get(name)
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_state": 
            super().__setattr__(name, value)
        else: 
            self._state[name] = value
    def __setitem__(self, key: str, value: Any) -> None: 
        self._state[key] = value
    def get(self, name: str, default: Any = None) -> Any: 
        return self._state.get(name, default)
    def __contains__(self, key: str) -> bool: 
        return key in self._state
    def update(self, *args: Any, **kwargs: Any) -> None: 
        self._state.update(*args, **kwargs)
    def clear(self): 
        self._state.clear() 

class MockStreamlit: 
    def __init__(self): 
        self.session_state = MockSessionState()
    def markdown(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.markdown called")
    def info(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.info called")
    def warning(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.warning called")
    def error(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.error called")
    def expander(self, *args: Any, **kwargs: Any) -> 'MockStreamlit': 
        logger.debug("MockStreamlit.expander called")
        return self
    def text_input(self, *args: Any, **kwargs: Any) -> str: 
        logger.debug("MockStreamlit.text_input called")
        return kwargs.get('value', "")
    def button(self, *args: Any, **kwargs: Any) -> bool: 
        logger.debug("MockStreamlit.button called")
        return False
    def spinner(self, *args: Any, **kwargs: Any) -> 'MockSpinner': 
        logger.debug("MockStreamlit.spinner called")
        return MockSpinner()
    def tabs(self, *args: Any, **kwargs: Any) -> List['MockStreamlit']: 
        logger.debug("MockStreamlit.tabs called")
        return [self, self, self] # type: ignore
    def __enter__(self) -> 'MockStreamlit': return self
    def __exit__(self, *args: Any): pass
    def graphviz_chart(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.graphviz_chart called")
    def subheader(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.subheader called")
    def text_area(self, *args: Any, **kwargs: Any) -> str: 
        logger.debug("MockStreamlit.text_area called")
        return kwargs.get('value', "")
    def success(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.success called")
    def rerun(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.rerun called")
    def columns(self, *args: Any, **kwargs: Any) -> List['MockStreamlit']: 
        logger.debug("MockStreamlit.columns called")
        return [self, self] # type: ignore
    def write(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.write called")
    def dataframe(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.dataframe called")
    def json(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.json called")
    def header(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.header called") 
    def checkbox(self, *args: Any, **kwargs: Any) -> bool: 
        logger.debug("MockStreamlit.checkbox called")
        return kwargs.get('value', False) 
    def caption(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.caption called") 
    def multiselect(self, *args: Any, **kwargs: Any) -> List[Any]: 
        logger.debug("MockStreamlit.multiselect called")
        return kwargs.get('default', [])
    def code(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.code called")
    def text(self, *args: Any, **kwargs: Any): logger.debug("MockStreamlit.text called") # Added st.text mock

class MockSpinner:
    def __enter__(self): return self
    def __exit__(self, *args: Any): pass

try:
    import streamlit as st
    logger.info("group_structure_utils: Successfully imported streamlit.")
except ImportError:
    logger.warning("group_structure_utils: Failed to import streamlit. Using MockStreamlit.")
    st = MockStreamlit() # type: ignore

# --- Constants ---
RELEVANT_CATEGORIES_FOR_GROUP_STRUCTURE = ["accounts", "confirmation-statement", "annual-return"]
METADATA_YEARS_TO_SCAN = 10 
ACCOUNTS_PRIORITY_ORDER = [
    "group",
    "consolidated",
    "legacy",
    "full accounts",
    "full",
    "small full",
    "small",
    "medium",
    "interim",
    "initial",
    "dormant",
    "micro-entity",
    "abridged",
    "filleted",
]
CRN_REGEX = r'\b([A-Z0-9]{2}\d{6}|\d{6,8})\b'
COMPANY_SUFFIX_REGEX = r'\b(?:Limited|Ltd|Plc|Llp|Lp|Sarl|Gmbh|Bv|Inc|Incorporated|Company|Undertaking|Group|Partnership|Ventures|Holdings|Industries|Solutions|Services|Technologies|Enterprises|Associates|Consulting|Capital|Investments|Trading|Logistics|Properties|Estates|Developments|Management|Resources|Systems|International|Global)\b'
SUBSIDIARY_VIZ_LIMIT: Optional[int] = 20

# --- Function Definitions ---

def _parse_document_content(
    doc_content_data: Dict[str, Any], fetched_content_types: List[str], 
    company_no: str, doc_metadata: Dict[str, Any], logger_param: logging.Logger, 
    ocr_handler: Optional[OCRHandlerType] = None, is_priority_for_ocr: bool = False
) -> Dict[str, Any]:
    """
    Parses document content, prioritizing JSON, then XHTML/XML, then PDF (with OCR if handler provided).
    Refined subsidiary name and CRN extraction.
    """
    parsed_result: Dict[str, Any] = {
        "document_date": doc_metadata.get("date"), "document_description": doc_metadata.get("description"),
        "text_content": None, "parent_company_name": None, "parent_company_number": None,
        "s409_info_found": False, "extraction_error": None, "source_format_parsed": None,
        "pages_ocrd": 0, "subsidiaries": [] 
    }
    content_to_parse: Any = None 
    content_type_to_parse: Optional[str] = None

    for doc_type in ["json", "xhtml", "xml"]: 
        if doc_type in doc_content_data and doc_content_data[doc_type] is not None:
            content_to_parse, content_type_to_parse = doc_content_data[doc_type], doc_type
            logger_param.info(f"GS Parse: Prioritizing {doc_type.upper()} for doc '{doc_metadata.get('description', 'N/A')}'")
            break
    
    if content_to_parse is None and 'pdf' in doc_content_data and doc_content_data['pdf'] is not None:
        if ocr_handler: 
            content_to_parse, content_type_to_parse = doc_content_data['pdf'], 'pdf'
            logger_param.info(f"GS Parse: PDF '{doc_metadata.get('description')}' found. "
                              f"OCR handler is active for Group Structure tab, attempting OCR. "
                              f"(Original priority flag for this doc type: {is_priority_for_ocr})")
        else:
            parsed_result["extraction_error"] = "PDF content found, but no OCR handler is active for Group Structure analysis (e.g., Textract not enabled for this tab or not available)."
            logger_param.warning(f"GS Parse: PDF '{doc_metadata.get('description', 'N/A')}' likely needs OCR, "
                                 f"but no OCR handler is active for the Group Structure tab.")

    if content_to_parse is not None and content_type_to_parse is not None:
        parsed_result["source_format_parsed"] = content_type_to_parse.upper()
        try:
            actual_ocr_handler_for_call = (
                ocr_handler if (content_type_to_parse == 'pdf' and ocr_handler) else None
            )
            extracted_text, pages_processed, extraction_error = extract_text_from_document(
                content_to_parse, content_type_to_parse, company_no, actual_ocr_handler_for_call
            )

            if content_type_to_parse == 'pdf' and actual_ocr_handler_for_call and pages_processed > 0:
                parsed_result["pages_ocrd"] = pages_processed
                if is_priority_for_ocr:
                    logger.info(
                        f"OCR executed for prioritized PDF '{doc_metadata.get('description')}'."
                    )
                else:
                    logger.info(
                        f"OCR triggered for '{doc_metadata.get('description')}' due to low or no text."
                    )
            if extraction_error: parsed_result["extraction_error"] = str(extraction_error)
            elif not extracted_text: parsed_result["text_content"] = ""
            else:
                parsed_result["text_content"] = extracted_text
                text_lower = extracted_text.lower()
                if any(s in text_lower for s in ["s409", "section 409", "s.409", "group accounts exemption", "exempt from audit", "parent undertaking prepares group accounts"]):
                    parsed_result["s409_info_found"] = True
                
                parent_match = re.search(
                    r"(?:ultimate parent company is|the parent company is|parent undertaking is|the immediate parent undertaking is)\s*([^,.(]+?)(?:,\s*a company incorporated in [^,.]+)?(?:\s*\(?(?:company no\.?|registered number|crn)?[:\s]*([A-Z0-9]+)\)?)?", 
                    text_lower, re.IGNORECASE
                )
                if parent_match:
                    p_name = html.unescape(parent_match.group(1).strip().title())
                    p_crn = parent_match.group(2).strip().upper() if parent_match.group(2) else None
                    if company_no.lower() not in p_name.lower() and (not p_crn or company_no.lower() != p_crn.lower()):
                        parsed_result["parent_company_name"] = p_name
                        if p_crn: parsed_result["parent_company_number"] = p_crn
                
                subs_list: List[Dict[str,Optional[str]]] = []
                subs_section_pattern = r"(?:subsidiary undertakings|principal subsidiary undertakings|investment in subsidiaries|details of related undertakings|group companies)([\s\S]+?)(?:\n\n[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]|\Z|notes to the financial statements|directors' report)" 
                subs_section_match = re.search(subs_section_pattern, extracted_text, re.IGNORECASE) 

                if subs_section_match:
                    section_text = subs_section_match.group(1)
                    for line_num, line in enumerate(section_text.split('\n')):
                        line_clean = line.strip()
                        if not line_clean or len(line_clean) < 3 or line_clean.lower().startswith(("note","total", "carrying value", "country of incorporation", "proportion of voting rights", "%", "£", "$", "€", "name of subsidiary", "registered office")): 
                            continue
                        
                        crn_match = re.search(CRN_REGEX, line_clean)
                        sub_crn = crn_match.group(1).upper() if crn_match else None
                        
                        potential_name_segment = line_clean
                        if sub_crn: 
                            parts_by_crn = re.split(f'({CRN_REGEX})', line_clean) 
                            name_candidate_idx = -1
                            for i, part in enumerate(parts_by_crn):
                                if sub_crn in part:
                                    if i > 0 and parts_by_crn[i-1].strip(): 
                                        name_candidate_idx = i-1
                                    elif i < len(parts_by_crn) - 1 and parts_by_crn[i+1].strip(): 
                                        name_candidate_idx = i+1
                                    break
                            if name_candidate_idx != -1:
                                potential_name_segment = parts_by_crn[name_candidate_idx].strip()
                            else: 
                                potential_name_segment = re.sub(CRN_REGEX, '', line_clean).strip()
                        
                        common_trails_and_prefixes = [
                            r'\(?incorporated in [^)]+\)?', r'registered in [^)]+\)?',
                            r'-\s*(?:100% owned|subsidiary|ordinary shares|share capital|\d+% holding)',
                            r',\s*(?:united kingdom|england|scotland|wales|northern ireland|ireland|usa|etc\.)',
                            r'company registration number', r'registered number', r'company no\.?',
                            r'whose registered office is', r'principal place of business',
                            r'nature of business', r'country of operation',
                            r'class of shares held', r'proportion of nominal value of issued shares held'
                        ]
                        cleaned_name_candidate = potential_name_segment
                        for pattern in common_trails_and_prefixes:
                            cleaned_name_candidate = re.sub(pattern, '', cleaned_name_candidate, flags=re.IGNORECASE).strip()
                        
                        cleaned_name_candidate = re.sub(r'^[^\w(]+', '', cleaned_name_candidate) 
                        cleaned_name_candidate = re.sub(r'[^\w)]+$', '', cleaned_name_candidate) 
                        cleaned_name_candidate = html.unescape(cleaned_name_candidate.strip().title())

                        if not cleaned_name_candidate or len(cleaned_name_candidate) < 3 or len(cleaned_name_candidate) > 100: continue
                        if re.fullmatch(r'[\d\s.,%£$€()*#@"&]+', cleaned_name_candidate): continue 
                        if not re.search(COMPANY_SUFFIX_REGEX, cleaned_name_candidate, re.IGNORECASE) and not sub_crn: 
                            if len(cleaned_name_candidate.split()) > 5 : continue 
                        if len(cleaned_name_candidate.split()) > 7 : continue 

                        is_not_parent = not (parsed_result.get("parent_company_name") and parsed_result.get("parent_company_name","").lower() in cleaned_name_candidate.lower())
                        is_not_self = company_no.lower() not in cleaned_name_candidate.lower()
                        
                        if is_not_parent and is_not_self:
                            subs_list.append({"name": cleaned_name_candidate, "number": sub_crn})
                            logger_param.debug(f"GS Parse: Added potential subsidiary: Name='{cleaned_name_candidate}', CRN='{sub_crn}' from line: '{line_clean[:100]}'")
                
                if subs_list: 
                    parsed_result["subsidiaries"] = list({frozenset(item.items()):item for item in subs_list}.values())
        
        except Exception as e: 
            parsed_result["extraction_error"] = f"Extraction exception: {str(e)}"
            logger_param.error(f"GS Parse: Exception during text extraction for doc '{doc_metadata.get('description', 'N/A')}': {e}", exc_info=True)

    elif not parsed_result.get("extraction_error"): 
        parsed_result["extraction_error"] = "No suitable content type found or processed based on priority and availability for Group Structure analysis."
        logger_param.warning(f"GS Parse: No content processed for doc '{doc_metadata.get('description', 'N/A')}'. "
                             f"Fetched types: {fetched_content_types}. OCR handler present: {bool(ocr_handler)}.")
    
    return parsed_result

def extract_parent_timeline(
    company_number: str, 
    api_key: str, 
    logger_param: logging.Logger, 
    parsed_docs_data: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    messages = ["Parent Company Timeline Analysis:"]
    logger_param.info(f"Extracting parent company timeline for {company_number} from parsed documents.")
    parent_timeline: List[Dict[str, Any]] = []

    if not parsed_docs_data:
        messages.append("No parsed document data provided for parent timeline generation.")
        logger_param.warning("extract_parent_timeline called without parsed_docs_data.")
        return parent_timeline, messages

    def get_doc_date(doc_bundle: Dict[str, Any]) -> datetime:
        parsed_data = doc_bundle.get('parsed_data')
        if parsed_data and parsed_data.get('document_date') and parsed_data['document_date'] != "N/A":
            try: return datetime.strptime(parsed_data['document_date'], '%Y-%m-%d')
            except (ValueError, TypeError): pass
        
        metadata = doc_bundle.get('metadata')
        if metadata and metadata.get('date') and metadata['date'] != "N/A":
            try: return datetime.strptime(metadata['date'], '%Y-%m-%d')
            except (ValueError, TypeError): pass
        return datetime.min 

    sorted_docs = sorted(parsed_docs_data, key=get_doc_date)
    last_reported_parent_key: Optional[Tuple[Optional[str], Optional[str]]] = None

    for doc_bundle in sorted_docs:
        parsed = doc_bundle.get('parsed_data')
        metadata = doc_bundle.get('metadata')
        if not parsed or not metadata: 
            continue

        doc_date_str = parsed.get("document_date") or metadata.get("date", "N/A")
        doc_desc = parsed.get("document_description") or metadata.get("description", "Unknown Document")
        
        parent_name = parsed.get("parent_company_name")
        parent_number = parsed.get("parent_company_number")
        s409_found = parsed.get("s409_info_found", False)

        if parent_name or parent_number: 
            current_parent_number = parent_number.strip().upper() if parent_number and isinstance(parent_number, str) and parent_number.strip() else None
            current_parent_name = parent_name.strip().title() if parent_name and isinstance(parent_name, str) and parent_name.strip() else None
            current_parent_key = (current_parent_name, current_parent_number)

            if current_parent_key != last_reported_parent_key or not parent_timeline:
                entry: Dict[str, Any] = {
                    'date': doc_date_str, 
                    'parent_name': current_parent_name, 
                    'parent_number': current_parent_number,
                    'source_doc_desc': doc_desc, 
                    's409_related': s409_found 
                }
                parent_timeline.append(entry)
                message = f"- On {doc_date_str} ({doc_desc}): Reported parent as '{current_parent_name or 'N/A'}' (CRN: {current_parent_number or 'N/A'}). S409 related: {s409_found}."
                messages.append(message)
                last_reported_parent_key = current_parent_key
        elif s409_found: 
            current_parent_key = ("S409_INFO_ONLY_IMPLIES_GROUP", "S409_INFO_ONLY_IMPLIES_GROUP") 
            if current_parent_key != last_reported_parent_key or not parent_timeline:
                entry = {
                    'date': doc_date_str, 
                    'parent_name': None, 
                    'parent_number': None,
                    'source_doc_desc': doc_desc, 
                    's409_related': True
                }
                parent_timeline.append(entry)
                message = f"- On {doc_date_str} ({doc_desc}): S409 information found, implying group context, but no specific parent details extracted from this document."
                messages.append(message)
                last_reported_parent_key = current_parent_key
    
    if not parent_timeline:
        messages.append("No specific parent company mentions or S409 group context indicators found in the analyzed documents to build a timeline.")
    else:
        messages.insert(1, f"Found {len(parent_timeline)} distinct parent changes/mentions or group context indicators over time:")
    
    return parent_timeline, messages

def _get_corporate_psc_parent_info(company_number: str, api_key: str, logger_param: logging.Logger) -> Optional[Dict[str, str]]:
    logger_param.info(f"Attempting to fetch PSC data for {company_number} to identify corporate parents.")
    if not ch_api_utils_available: 
        logger_param.warning("PSC check skipped: 'ch_api_utils' module not available.")
        return None
    try:
        pscs_data, psc_error = get_company_pscs(company_number, api_key)
        if psc_error: 
            logger_param.error(f"Error fetching PSC data for {company_number}: {psc_error}")
            return None
        if not pscs_data or not pscs_data.get("items"): 
            logger_param.info(f"No PSC items found for {company_number}.")
            return None
        
        corporate_pscs: List[Dict[str, Any]] = []
        for psc in pscs_data.get("items", []):
            if psc.get("kind") == "corporate-entity-person-with-significant-control":
                psc_name = psc.get("name")
                identification = psc.get("identification")
                if identification and isinstance(identification, dict):
                    psc_crn = identification.get("registration_number")
                    if psc_crn and isinstance(psc_crn, str) and psc_crn.strip().upper() != company_number.strip().upper():
                        corporate_pscs.append({"name": psc_name, "number": psc_crn, "data": psc})
        
        if not corporate_pscs: 
            logger_param.info(f"No external corporate PSCs identified for {company_number}.")
            return None
        
        selected_psc = corporate_pscs[0] 
        logger_param.info(f"Identified corporate PSC for {company_number}: {selected_psc.get('name')} ({selected_psc.get('number')})")
        return {"name": str(selected_psc["name"]), "number": str(selected_psc["number"])}

    except Exception as e: 
        logger_param.error(f"Unexpected error processing PSC data for {company_number}: {e}", exc_info=True)
        return None

def _fetch_and_parse_selected_documents(
    company_number: str, docs_to_process_metadata: List[Dict[str, Any]], api_key: str, 
    logger_param: logging.Logger, ocr_handler: Optional[OCRHandlerType],
    is_topco_analyzing_subs: bool 
) -> List[Dict[str, Any]]:
    processed_content_list = []
    if not ch_api_utils_available: 
        logger_param.error("_fetch_and_parse_selected_documents: ch_api_utils not available. Cannot fetch/parse.")
        return processed_content_list 

    for doc_item_meta in docs_to_process_metadata:
        doc_desc = doc_item_meta.get('description','N/A')
        doc_date_str = doc_item_meta.get('date','N/A')
        doc_category = doc_item_meta.get("category","N/A") 

        priority_for_ocr_flag = False
        if doc_category == "accounts":
            if is_topco_analyzing_subs and ("group" in doc_desc.lower() or "consolidated" in doc_desc.lower()):
                priority_for_ocr_flag = True
            elif not is_topco_analyzing_subs and "full" in doc_desc.lower(): 
                priority_for_ocr_flag = True
        
        if not doc_item_meta.get("links", {}).get("document_metadata"):
            logger_param.warning(f"No download link (document_metadata) for doc: {doc_desc} ({doc_date_str}). Skipping.")
            processed_content_list.append({
                "metadata": doc_item_meta, 
                "parsed_data": {"errors": ["No download link"], "document_description": doc_desc, "document_date": doc_date_str}
            })
            continue

        raw_content_dict, fetched_types_list, fetch_err_msg = _fetch_document_content_from_ch(company_number, doc_item_meta)
        
        if fetch_err_msg or not raw_content_dict:
            logger_param.error(f"Failed to fetch content for doc {doc_desc} ({doc_date_str}): {fetch_err_msg or 'No content dictionary returned'}")
            processed_content_list.append({
                "metadata": doc_item_meta, 
                "parsed_data": {"errors": [f"Fetch fail: {fetch_err_msg or 'No content'}"], "document_description": doc_desc, "document_date": doc_date_str}
            })
            continue
        
        parsed_info_dict = _parse_document_content(
            raw_content_dict, fetched_types_list, company_number, 
            doc_item_meta, logger_param, ocr_handler, priority_for_ocr_flag
        )
        processed_content_list.append({"metadata": doc_item_meta, "parsed_data": parsed_info_dict})
    
    return processed_content_list

# --- Main Orchestrator Function ---
def analyze_company_group_structure(
    company_number: str,
    api_key: str,
    base_scratch_dir: _pl.Path,
    logger: logging.Logger,
    ocr_handler: Optional[OCRHandlerType] = None,
    analysis_mode: AnalysisMode = "profile_check",
    target_subsidiary_crns: Optional[List[str]] = None,
    session_data: Optional[Dict[str, Any]] = None,
    known_structure: Optional[Dict[str, List[Dict[str, str]]]] = None,
    use_public_sources_gpt: bool = False,
    *,
    docs_metadata_to_process: Optional[List[Dict[str, Any]]] = None,
    years_to_scan: int = METADATA_YEARS_TO_SCAN,
    metadata_only: bool = False,
) -> Dict[str, Any]:
    """High level orchestrator for group structure workflows.

    If ``metadata_only`` is ``True`` the function returns filing metadata so the
    caller can select which documents to process.  When ``docs_metadata_to_process``
    is provided, only those filings will be fetched and parsed. ``years_to_scan``
    controls the lookback window when fetching metadata. ``known_structure`` can
    be supplied to merge a pre-existing list of subsidiaries into the results.
    When ``use_public_sources_gpt`` is ``True`` the function will query GPT-4.1
    for a preliminary list of subsidiaries based on public information.
    """
    results: Dict[str, Any] = {
        "company_number_analyzed": company_number, "analysis_mode_executed": analysis_mode,
        "report_messages": [f"Group Structure Analysis for {company_number} - Mode: {analysis_mode}"],
        "company_profile": session_data.get("company_profile") if session_data else None,
        "is_inferred_topco": session_data.get("is_inferred_topco", False) if session_data else False,
        "parent_timeline": session_data.get("parent_timeline", []) if session_data else [],
        "subsidiary_evolution": session_data.get("subsidiary_evolution", defaultdict(list)) if session_data else defaultdict(list),
        "identified_parent_crn": session_data.get("identified_parent_crn") if session_data else None,
        "visualization_data": None, "subsidiary_details_list": [],
        "downloaded_documents_content": [] 
    }
    if not known_structure and use_public_sources_gpt:
        company_name_for_gpt = results.get("company_profile", {}).get("company_name", company_number)
        gpt_subs = gpt_fetch_public_group_structure(company_name_for_gpt, company_number, logger)
        if gpt_subs:
            known_structure = {company_number: gpt_subs}

    if known_structure:
        preset_subs = known_structure.get(company_number, [])
        if preset_subs:
            existing = {frozenset(s.items()) for s in results["subsidiary_evolution"].get(0, [])}
            for sub in preset_subs:
                if isinstance(sub, dict) and frozenset(sub.items()) not in existing:
                    results["subsidiary_evolution"].setdefault(0, []).append(sub)
    report_messages = results["report_messages"] 

    if not ch_api_utils_available:
        report_messages.append("CRITICAL ERROR: CH API utilities not available. Group structure analysis cannot proceed."); 
        logger.critical("analyze_company_group_structure: ch_api_utils not available.")
        return results

    if analysis_mode == "profile_check" or not results["company_profile"]:
        logger.info(f"GS Mode '{analysis_mode}': Fetching/updating profile for {company_number}.")
        profile = get_company_profile(company_number, api_key)
        results["company_profile"] = profile
        if profile and isinstance(profile, dict) :
            report_messages.append(f"Profile for {profile.get('company_name', company_number)} (Status: {profile.get('company_status', 'N/A')}) fetched.")
            acc_req = profile.get("accounts", {}).get("accounting_requirement")
            if acc_req == "group": 
                results["is_inferred_topco"] = True
                report_messages.append("Profile indicates TopCo status (group accounts requirement).")
            elif acc_req: 
                report_messages.append(f"Profile accounting requirement: {acc_req}.")
                results["is_inferred_topco"] = False 
            else: 
                results["is_inferred_topco"] = session_data.get("user_stated_topco", False) if session_data else False
                report_messages.append(f"Profile accounting requirement not specified. TopCo status set to: {results['is_inferred_topco']} (based on user input/default).")
        else:
            report_messages.append(f"Failed to fetch company profile for {company_number} or profile is invalid.")
            results["is_inferred_topco"] = session_data.get("user_stated_topco", False) if session_data else False 
            report_messages.append(f"Proceeding with TopCo status as: {results['is_inferred_topco']} (profile unavailable).")
    else:
        logger.info(f"GS Mode '{analysis_mode}': Using existing profile for {company_number}.")
        if session_data and "is_inferred_topco" in session_data:
             results["is_inferred_topco"] = session_data["is_inferred_topco"]
        elif results["company_profile"]: 
            acc_req = results["company_profile"].get("accounts", {}).get("accounting_requirement")
            results["is_inferred_topco"] = (acc_req == "group")

    docs_to_process_metadata: List[Dict[str, Any]] = []
    if analysis_mode in ["find_parent", "list_subsidiaries"] and docs_metadata_to_process is None:
        end_year = datetime.now().year
        start_year = end_year - max(1, years_to_scan) + 1
        categories = RELEVANT_CATEGORIES_FOR_GROUP_STRUCTURE
        report_messages.append(
            f"Mode '{analysis_mode}': Fetching filings for {company_number} over {years_to_scan} year(s)."
        )
        doc_items, _, meta_err = get_ch_documents_metadata(
            company_no=company_number,
            api_key=api_key,
            categories=categories,
            items_per_page=100,
            max_docs_to_fetch_meta=150,
            target_docs_per_category_in_date_range=(3 if analysis_mode == "find_parent" else 10),
            year_range=(start_year, end_year),
        )
        if meta_err:
            report_messages.append(f"Metadata fetch warning: {meta_err}")
        if doc_items:
            report_messages.append(
                f"Found {len(doc_items)} potentially relevant filings for {analysis_mode} mode."
            )
            docs_to_process_metadata = doc_items
        else:
            report_messages.append(f"No relevant filings found for {analysis_mode} mode.")
        results["retrieved_doc_metadata"] = docs_to_process_metadata
        if docs_to_process_metadata:
            for idx, itm in enumerate(docs_to_process_metadata):
                desc = str(itm.get("description", "")).lower()
                if any(keyword in desc for keyword in ACCOUNTS_PRIORITY_ORDER):
                    results["recommended_doc_idx"] = idx
                    break
        if metadata_only:
            return results
    elif docs_metadata_to_process is not None:
        docs_to_process_metadata = docs_metadata_to_process

    parsed_docs_data_list: List[Dict[str, Any]] = []
    if docs_to_process_metadata:
        logger.info(f"GS Mode '{analysis_mode}': Processing {len(docs_to_process_metadata)} selected filings for {company_number}.")
        parsed_docs_data_list = _fetch_and_parse_selected_documents(
            company_number, docs_to_process_metadata, api_key, logger, ocr_handler,
            is_topco_analyzing_subs=(analysis_mode == "list_subsidiaries" and results["is_inferred_topco"])
        )
        results["downloaded_documents_content"] = parsed_docs_data_list 
        report_messages.append(f"Processed {len(parsed_docs_data_list)} documents. Check logs for details of each.")

    if analysis_mode == "find_parent":
        logger.info(f"GS Mode 'find_parent': Extracting parent timeline for {company_number}.")
        parent_timeline_data, pt_msgs = extract_parent_timeline(company_number, api_key, logger, parsed_docs_data_list)
        results["parent_timeline"] = parent_timeline_data
        report_messages.extend(pt_msgs)
        latest_parent_entry = next((e for e in reversed(results["parent_timeline"]) if e.get('parent_number')), None)
        if latest_parent_entry and isinstance(latest_parent_entry.get('parent_number'), str):
            results["identified_parent_crn"] = latest_parent_entry['parent_number']
        # The PSC-based parent lookup has been removed in favor of explicit user input or GPT analysis
        
        latest_parent_entry_with_crn = next((e for e in reversed(results["parent_timeline"]) if e.get('parent_number')), None)
        if latest_parent_entry_with_crn and isinstance(latest_parent_entry_with_crn.get('parent_number'), str):
            results["identified_parent_crn"] = latest_parent_entry_with_crn['parent_number']
            report_messages.append(f"Identified most recent parent CRN from filings: {results['identified_parent_crn']}")
        
        if not results["identified_parent_crn"]:
            report_messages.append("No parent CRN found in filings timeline. Checking PSC data as fallback.")
            psc_parent_info = _get_corporate_psc_parent_info(company_number, api_key, logger)
            if psc_parent_info and isinstance(psc_parent_info.get('number'), str):
                results["identified_parent_crn"] = psc_parent_info['number']
                report_messages.append(f"Identified potential parent CRN from PSC data: {results['identified_parent_crn']} (Name: {psc_parent_info.get('name', 'N/A')})")
            else:
                report_messages.append("No corporate parent identified from PSC data.")
        
        dot_lines_viz = ["digraph G { rankdir=TB; node[shape=box, style=rounded];"]
        current_co_name_viz = results.get("company_profile", {}).get('company_name', company_number)
        dot_lines_viz.append(f'"{company_number}" [label="{current_co_name_viz}\\n({company_number})\\nAnalyzed Company", fillcolor=lightblue, style="filled,rounded"];')
        if results["identified_parent_crn"] and results["identified_parent_crn"] != company_number:
            parent_crn_str_viz = str(results["identified_parent_crn"])
            parent_profile_viz = get_company_profile(parent_crn_str_viz, api_key) 
            parent_name_viz = parent_profile_viz.get('company_name', parent_crn_str_viz) if parent_profile_viz else parent_crn_str_viz
            dot_lines_viz.append(f'"{parent_crn_str_viz}" [label="{parent_name_viz}\\n({parent_crn_str_viz})\\nIdentified Parent", fillcolor=lightgreen, style="filled,rounded"];')
            dot_lines_viz.append(f'"{parent_crn_str_viz}" -> "{company_number}";')
        dot_lines_viz.append("}")
        results["visualization_data"] = "\n".join(dot_lines_viz)

    elif analysis_mode == "list_subsidiaries":
        logger.info(f"GS Mode 'list_subsidiaries': Extracting subsidiary evolution for {company_number}.")
        current_subs_evolution = results.get("subsidiary_evolution", defaultdict(list)) 
        
        dot_lines = ["digraph G { rankdir=TB; node[shape=box];"]
        topco_name_viz = results.get("company_profile", {}).get('company_name', company_number) # Safe access
        dot_lines.append(f'"{company_number}" [label="{topco_name_viz}\\n({company_number})\\nTopCo", color=lightgreen];')
        latest_year_subs = max(results["subsidiary_evolution"].keys()) if results["subsidiary_evolution"] else None
        if latest_year_subs:
            subs_for_viz = results["subsidiary_evolution"][latest_year_subs]
            # Limit number of subsidiaries displayed in the graph if SUBSIDIARY_VIZ_LIMIT is set
            if SUBSIDIARY_VIZ_LIMIT is not None:
                subs_for_viz = subs_for_viz[:SUBSIDIARY_VIZ_LIMIT]
            for sub in subs_for_viz:
                sub_name, sub_crn = sub.get("name","N/A"), sub.get("number","N/A")
                node_id = sub_crn if sub_crn else f"{sub_name}_name_only"
                dot_lines.append(f'"{node_id}" [label="{sub_name}\\n({sub_crn if sub_crn else "CRN N/A"})"];')
                dot_lines.append(f'"{company_number}" -> "{node_id}";')
        dot_lines.append("}")
        results["visualization_data"] = "\n".join(dot_lines)

    elif analysis_mode == "subsidiary_details":
        if target_subsidiary_crns:
            report_messages.append(f"Fetching profile details for {len(target_subsidiary_crns)} selected subsidiaries.")
            for sub_crn_target in target_subsidiary_crns:
                sub_profile_detail = get_company_profile(sub_crn_target, api_key)
                sub_detail_entry = {"crn": sub_crn_target, "name": "N/A", "status": "N/A", "type": "N/A"}
                if sub_profile_detail and isinstance(sub_profile_detail, dict):
                    sub_detail_entry["name"] = sub_profile_detail.get("company_name", sub_crn_target)
                    sub_detail_entry["status"] = sub_profile_detail.get("company_status", "N/A")
                    sub_detail_entry["type"] = sub_profile_detail.get("type", "N/A")
                results["subsidiary_details_list"].append(sub_detail_entry)
            report_messages.append("Subsidiary profile fetching complete.")
        else: 
            report_messages.append("No target subsidiaries provided for detail fetching in 'subsidiary_details' mode.")
            logger.info("GS Mode 'subsidiary_details': No target_subsidiary_crns provided.")
            
    logger.info(f"Group structure analysis for {company_number} (Mode: {analysis_mode}) completed.")
    return results

def render_group_structure_ui(api_key: str, base_scratch_dir: _pl.Path, logger: logging.Logger, ocr_handler: Optional[OCRHandlerType] = None):
    st.header("🏢 Company Group Structure Analysis") # type: ignore
    default_session_state = {
        'gs_current_company_crn': "",
        'gs_user_stated_topco': False,
        'gs_analysis_results': {},
        'gs_current_stage': "initial_input",
        'gs_selected_subs_for_details': [],
        'gs_years_choice': 5,
        'gs_next_analysis_mode': None,
    }
    for key, default_val in default_session_state.items():
        if key not in st.session_state: 
            if isinstance(st.session_state, MockSessionState):
                setattr(st.session_state, key, default_val)
            else: 
                st.session_state[key] = default_val


    input_crn_ui = st.text_input(
        "Enter Company Number to Analyze:", 
        st.session_state.get('gs_current_company_crn', ""), 
        key="gs_crn_input_widget" 
    )
    if input_crn_ui != st.session_state.get('gs_current_company_crn'):
        st.session_state.gs_current_company_crn = input_crn_ui.strip().upper()
        st.session_state.gs_analysis_results = {}
        st.session_state.gs_current_stage = "initial_input"
        st.session_state.gs_user_stated_topco = False 
        st.session_state.gs_selected_subs_for_details = []
        st.session_state.gs_selected_subs_for_details_display = []
        st.session_state.gs_show_downloaded_docs_expander = False
    
    st.session_state.gs_user_stated_topco = st.checkbox(
        "This company is the UK Top Parent",
        st.session_state.get('gs_user_stated_topco', False),
        key="gs_topco_checkbox_main",
    )  # type: ignore

    st.session_state.gs_years_choice = st.selectbox(
        "How many years of filings to retrieve?",
        options=[2, 5, 10],
        index=[2, 5, 10].index(st.session_state.get('gs_years_choice', 5)),
        key="gs_years_choice_select",
    )

    if st.button("1. Fetch Profile & Determine Analysis Path", key="gs_fetch_profile_btn_widget"):
        if not st.session_state.get('gs_current_company_crn'): 
            st.warning("Please enter a company number.")
        elif not api_key:
            st.error("Companies House API Key is not configured. Cannot fetch profile.")
        else:
            with st.spinner(f"Fetching profile for {st.session_state.get('gs_current_company_crn')}..."):
                current_session_data_for_analysis = {
                    "company_profile": None, 
                    "is_inferred_topco": st.session_state.get('gs_user_stated_topco'), 
                    "user_stated_topco": st.session_state.get('gs_user_stated_topco') 
                }
                analysis_results = analyze_company_group_structure(
                    company_number=st.session_state.get('gs_current_company_crn',''), 
                    api_key=api_key, 
                    base_scratch_dir=base_scratch_dir,
                    logger_param=logger, 
                    ocr_handler=ocr_handler, 
                    analysis_mode="profile_check", 
                    session_data=current_session_data_for_analysis
                )
                st.session_state.gs_analysis_results = analysis_results
                st.session_state.gs_current_stage = "profile_reviewed"
                st.session_state.gs_show_downloaded_docs_expander = False 
                st.rerun()

    results_from_analysis = st.session_state.get('gs_analysis_results', {})
    if results_from_analysis and st.session_state.get('gs_current_stage') in ["profile_reviewed", "parent_analyzed", "subs_listed", "subs_detailed"]:
        st.markdown("---"); 
        st.subheader(f"Analysis Dashboard for: {results_from_analysis.get('company_number_analyzed', '')}")
        
        profile_data = results_from_analysis.get("company_profile")
        if profile_data and isinstance(profile_data, dict) : 
            st.markdown(f"**Name:** {profile_data.get('company_name', 'N/A')} | **Status:** {profile_data.get('company_status', 'N/A')}")
            st.caption(f"Inferred TopCo based on profile: {results_from_analysis.get('is_inferred_topco', 'Unknown')}")
        else: 
            st.warning("Company profile not fetched or unavailable.")
    
        downloaded_docs = results_from_analysis.get("downloaded_documents_content", [])
        if downloaded_docs:
            show_docs_expander_current_val = st.session_state.get('gs_show_downloaded_docs_expander', False)
            new_checkbox_val = st.checkbox("Show/Hide Processed Document Details", value=show_docs_expander_current_val, key=f"gs_show_docs_toggle_{st.session_state.gs_current_stage}")
            if new_checkbox_val != show_docs_expander_current_val:
                 st.session_state.gs_show_downloaded_docs_expander = new_checkbox_val
                 st.rerun() 

            if st.session_state.gs_show_downloaded_docs_expander:
                with st.expander("Details of Processed Documents for Group Structure Analysis", expanded=True):
                    for idx, doc_bundle in enumerate(downloaded_docs):
                        meta = doc_bundle.get("metadata", {})
                        parsed = doc_bundle.get("parsed_data", {})
                        st.markdown(f"**Doc {idx+1}: {meta.get('description', 'N/A')} ({meta.get('date', 'N/A')})**")
                        st.caption(f"Source Format Parsed: {parsed.get('source_format_parsed', 'N/A')}, OCR Pages: {parsed.get('pages_ocrd', 0)}")
                        if parsed.get("extraction_error"):
                            st.error(f"Extraction Error: {parsed.get('extraction_error')}")
                        elif parsed.get("text_content") is not None:
                             if callable(getattr(st, "code", None)): 
                                st.code(f"Text Length: {len(parsed.get('text_content', ''))} chars. Parent Found: {parsed.get('parent_company_name', 'No')}. Subs Found: {len(parsed.get('subsidiaries',[]))}", language=None)
                             else: 
                                # Use st.text as a fallback if st.code is not available (e.g. in older Streamlit or very basic mock)
                                if callable(getattr(st, "text", None)):
                                    st.text(f"Text Length: {len(parsed.get('text_content', ''))} chars. Parent Found: {parsed.get('parent_company_name', 'No')}. Subs Found: {len(parsed.get('subsidiaries',[]))}")
                                else: # Absolute fallback
                                    st.write(f"Text Length: {len(parsed.get('text_content', ''))} chars. Parent Found: {parsed.get('parent_company_name', 'No')}. Subs Found: {len(parsed.get('subsidiaries',[]))}")
                        else:
                            st.info("No text content extracted or extraction not attempted for this format.")


    if st.session_state.get('gs_current_stage') == "profile_reviewed":
        is_topco = results.get("is_inferred_topco", False)
        col1, col2 = st.columns(2) # type: ignore
        with col1:
            if is_topco:
                if st.button("2a. Select Group Accounts to Analyze", key="gs_analyze_subs_btn"):  # type: ignore
                    with st.spinner(
                        f"Fetching group accounts for {results.get('company_number_analyzed')}..."
                    ):  # type: ignore
                        updated_results = analyze_company_group_structure(
                            results.get('company_number_analyzed', ''),
                            api_key,
                            base_scratch_dir,
                            logger,
                            ocr_handler,
                            "list_subsidiaries",
                            session_data=results,
                            metadata_only=True,
                            years_to_scan=st.session_state.get('gs_years_choice', 5),
                        )
                        st.session_state.gs_analysis_results = updated_results  # type: ignore
                        st.session_state.gs_current_stage = "docs_listed"  # type: ignore
                        st.session_state.gs_next_analysis_mode = "list_subsidiaries"  # type: ignore
                        st.rerun()  # type: ignore
            else:
                if st.button("2b. Find Parent Company", key="gs_find_parent_btn"):  # type: ignore
                    with st.spinner(
                        f"Fetching filings for {results.get('company_number_analyzed')}..."
                    ):  # type: ignore
                        updated_results = analyze_company_group_structure(
                            results.get('company_number_analyzed', ''),
                            api_key,
                            base_scratch_dir,
                            logger,
                            ocr_handler,
                            "find_parent",
                            session_data=results,
                            metadata_only=True,
                            years_to_scan=st.session_state.get('gs_years_choice', 5),
                        )
                        st.session_state.gs_analysis_results = updated_results  # type: ignore
                        st.session_state.gs_current_stage = "docs_listed"  # type: ignore

                        st.session_state.gs_next_analysis_mode = "find_parent"  # type: ignore
                        st.rerun()  # type: ignore

    if st.session_state.get('gs_current_stage') == "docs_listed":
        docs_meta = results.get("retrieved_doc_metadata", [])
        if not docs_meta:
            st.warning("No filings found to select.")  # type: ignore
        else:
            st.subheader("Select Filings to Analyze")  # type: ignore
            options_labels = [
                f"{d.get('date','N/A')} | {d.get('category','N/A')} | {d.get('type','N/A')} | {d.get('description','N/A')}"
                for d in docs_meta
            ]
            default_sel: List[str] = []
            rec_idx = results.get("recommended_doc_idx")
            if rec_idx is not None and 0 <= rec_idx < len(options_labels):
                options_labels[rec_idx] += " [RECOMMENDED]"
                default_sel.append(options_labels[rec_idx])
                st.caption("Latest group/consolidated accounts pre-selected")  # type: ignore
            selected_display = st.multiselect(
                "Choose filings:",
                options=options_labels,
                default=default_sel,
                key="gs_doc_multiselect",
            )  # type: ignore
            if st.button("Analyze Selected Filings", key="gs_process_docs_btn"):
                indices = [options_labels.index(lbl) for lbl in selected_display]
                chosen = [docs_meta[i] for i in indices]
                if not chosen:
                    st.warning("Please select at least one filing.")  # type: ignore
                else:
                    with st.spinner("Processing selected filings..."):
                        updated_results = analyze_company_group_structure(
                            results.get('company_number_analyzed', ''),
                            api_key,
                            base_scratch_dir,
                            logger,
                            ocr_handler,
                            st.session_state.get('gs_next_analysis_mode', 'find_parent'),
                            session_data=results,
                            docs_metadata_to_process=chosen,
                        )
                        next_stage = (
                            "subs_listed"
                            if st.session_state.get('gs_next_analysis_mode') == "list_subsidiaries"
                            else "parent_analyzed"
                        )
                        st.session_state.gs_analysis_results = updated_results  # type: ignore
                        st.session_state.gs_current_stage = next_stage  # type: ignore
                        st.rerun()  # type: ignore
    
    if st.session_state.get('gs_current_stage') == "parent_analyzed":
        st.subheader("Parent Company Analysis Results");
        
        parent_timeline_viz = results_from_analysis.get("parent_timeline", [])
        if parent_timeline_viz:
            with st.expander("Parent Company Timeline (from filings)", expanded=False):
                for entry in parent_timeline_viz:
                    st.markdown(f"- **{entry.get('date', 'N/A')}** ({entry.get('source_doc_desc','N/A')}): Parent '{entry.get('parent_name', 'N/A')}' (CRN: {entry.get('parent_number', 'N/A')}). S409: {entry.get('s409_related')}")

        identified_parent_crn_val = results_from_analysis.get("identified_parent_crn")
        if identified_parent_crn_val and identified_parent_crn_val != results_from_analysis.get("company_number_analyzed"):
            st.success(f"Potential Parent Company Identified: CRN **{identified_parent_crn_val}**")
            if st.button(f"Analyze This Parent ({identified_parent_crn_val}) as TopCo", key=f"gs_analyze_identified_parent_{identified_parent_crn_val}"):
                st.session_state.gs_current_company_crn = identified_parent_crn_val
                st.session_state.gs_user_stated_topco = True 
                st.session_state.gs_analysis_results = {} 
                st.session_state.gs_current_stage = "initial_input" 
                st.rerun()
        elif not parent_timeline_viz : 
             st.info("No parent company information found in the analyzed documents or PSC data.")

        if results_from_analysis.get("visualization_data"): 
            st.graphviz_chart(results_from_analysis["visualization_data"])

    if st.session_state.get('gs_current_stage') == "subs_listed":
        st.subheader(f"Subsidiary Analysis for TopCo: {results.get('company_number_analyzed')}") # type: ignore
        if results.get("visualization_data"): st.graphviz_chart(results["visualization_data"]) # type: ignore
        subs_evo = results.get("subsidiary_evolution", {})
        all_subs = get_all_subsidiaries(subs_evo)
        if all_subs:
            # Display a deduplicated list aggregated across all years
            st.markdown("---"); st.subheader("Stage 3: Deeper Dive into Selected Subsidiaries") # type: ignore
            subs_options = [f"{s.get('name', 'N/A')} ({s.get('number', 'N/A')})" for s in all_subs if s.get('number')]
            selected_subs_display = st.multiselect("Select subsidiaries for status check:", options=subs_options, key="gs_subs_multiselect") # type: ignore
            crns_to_fetch = [re.search(CRN_REGEX, disp).group(1) for disp in selected_subs_display if re.search(CRN_REGEX, disp)] if selected_subs_display else []
            st.session_state.gs_selected_subs_for_details = crns_to_fetch # type: ignore
            if st.button("Fetch Status for Selected Subsidiaries", key="gs_fetch_sub_details_btn"): # type: ignore
                if not crns_to_fetch: st.warning("Please select subsidiaries with CRNs.") # type: ignore
                else:
                    with st.spinner("Fetching details..."): # type: ignore
                        updated_results = analyze_company_group_structure(results.get('company_number_analyzed',''), api_key, base_scratch_dir, logger, ocr_handler, "subsidiary_details", target_subsidiary_crns=crns_to_fetch, session_data=results)
                        st.session_state.gs_analysis_results, st.session_state.gs_current_stage = updated_results, "subs_detailed" # type: ignore
                        st.rerun() # type: ignore
    
    if st.session_state.get('gs_current_stage') == "subs_detailed":
        st.subheader("Selected Subsidiary Details") # type: ignore
        sub_details_list = results.get("subsidiary_details_list", [])
        if sub_details_list:
            for detail in sub_details_list:
                st.markdown(f"- **{detail.get('name')}** ({detail.get('crn')}) - Status: `{detail.get('status')}`") # type: ignore
                if st.button(f"Analyze {detail.get('crn')} in detail", key=f"gs_analyze_sub_detail_{detail.get('crn')}"): # type: ignore
                    st.session_state.gs_current_company_crn, st.session_state.gs_user_stated_topco = detail.get('crn'), False # type: ignore
                    st.session_state.gs_analysis_results, st.session_state.gs_current_stage = {}, "initial_input" # type: ignore
                    st.rerun() # type: ignore

        if latest_year_with_subs_data and subs_evolution_data.get(latest_year_with_subs_data):
            st.markdown("---"); st.subheader("Stage 3: Deeper Dive into Selected Subsidiaries")
            
            subs_for_selection = subs_evolution_data[latest_year_with_subs_data]
            subs_options_display = [
                f"{s.get('name', 'N/A')} ({s.get('number', 'CRN N/A')})" 
                for s in subs_for_selection 
                if s.get('number') and s.get('number') != 'N/A' 
            ]
            
            if subs_options_display:
                multiselect_key = f"gs_subs_multiselect_widget_{st.session_state.gs_current_company_crn}_{latest_year_with_subs_data}"
                selected_subs_display_names = st.multiselect(
                    "Select subsidiaries with CRNs for status check:", 
                    options=subs_options_display, 
                    default=st.session_state.get('gs_selected_subs_for_details_display', []), 
                    key=multiselect_key
                )
                crns_to_fetch_details = []
                if selected_subs_display_names:
                    for disp_name in selected_subs_display_names:
                        match = re.search(CRN_REGEX, disp_name)
                        if match: crns_to_fetch_details.append(match.group(1))
                
                if st.session_state.get('gs_selected_subs_for_details') != crns_to_fetch_details:
                    st.session_state.gs_selected_subs_for_details = crns_to_fetch_details
                if st.session_state.get('gs_selected_subs_for_details_display') != selected_subs_display_names:
                     st.session_state.gs_selected_subs_for_details_display = selected_subs_display_names


                if st.button("Fetch Status for Selected Subsidiaries", key="gs_fetch_sub_details_btn_widget"):
                    if not st.session_state.gs_selected_subs_for_details: 
                        st.warning("Please select subsidiaries with valid CRNs from the list.")
                    else:
                        with st.spinner("Fetching details for selected subsidiaries..."):
                            updated_results = analyze_company_group_structure(
                                results_from_analysis.get('company_number_analyzed',''), api_key, base_scratch_dir, 
                                logger, ocr_handler, "subsidiary_details", 
                                target_subsidiary_crns=st.session_state.gs_selected_subs_for_details, 
                                session_data=results_from_analysis
                            )
                            st.session_state.gs_analysis_results = updated_results
                            st.session_state.gs_current_stage = "subs_detailed"
                            st.rerun()
            else:
                st.info("No subsidiaries with identifiable CRNs found in the latest filings to select for detailed status check.")
        elif not subs_evolution_data: 
             st.info("No subsidiary information was extracted from the analyzed documents.")
        else: 
             st.info("No subsidiaries found in the filings for the analyzed years.")


    if st.session_state.get('gs_current_stage') == "subs_detailed":
        st.subheader("Selected Subsidiary Details")
        subsidiary_details_list_data = results_from_analysis.get("subsidiary_details_list", [])
        if subsidiary_details_list_data:
            for detail_item in subsidiary_details_list_data:
                st.markdown(f"- **{detail_item.get('name', 'N/A')}** (CRN: {detail_item.get('crn', 'N/A')}) - Status: `{detail_item.get('status', 'N/A')}` - Type: `{detail_item.get('type', 'N/A')}`")
                if detail_item.get('crn') and detail_item.get('crn') != 'N/A':
                    if st.button(f"Analyze {detail_item.get('crn')} in Detail", key=f"gs_analyze_sub_detail_{detail_item.get('crn')}_btn"):
                        st.session_state.gs_current_company_crn = detail_item.get('crn')
                        st.session_state.gs_user_stated_topco = False 
                        st.session_state.gs_analysis_results = {} 
                        st.session_state.gs_current_stage = "initial_input" 
                        st.rerun()
        else:
            st.info("No details fetched for selected subsidiaries, or no subsidiaries were selected.")

    if results_from_analysis and results_from_analysis.get("report_messages") and len(results_from_analysis.get("report_messages",[])) > 1:
        with st.expander("Detailed Analysis Log (Group Structure)", expanded=False):
            for msg_item in results_from_analysis.get("report_messages", []): 
                st.caption(msg_item)

def find_subsidiaries(company: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Recursively finds all subsidiaries in a company hierarchy.
    
    Args:
        company (Dict[str, Any]): A dictionary containing company information including subsidiaries
        
    Returns:
        List[Dict[str, Any]]: A flat list of all subsidiaries found in the hierarchy
        
    Raises:
        ValueError: If company is None
    """
    if company is None:
        raise ValueError("Company cannot be None")
        
    subsidiaries = []
    
    # Get direct subsidiaries
    direct_subs = company.get("subsidiaries", [])
    subsidiaries.extend(direct_subs)
    
    # Recursively find subsidiaries of subsidiaries
    for sub in direct_subs:
        if isinstance(sub, dict):
            sub_subs = find_subsidiaries(sub)
            subsidiaries.extend(sub_subs)
            
    return subsidiaries

def build_company_tree(company_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a hierarchical tree structure from company data.
    
    Args:
        company_data (Dict[str, Any]): Company data including subsidiaries
        
    Returns:
        Dict[str, Any]: Tree structure with company hierarchy
        
    Raises:
        ValueError: If company_data is None or invalid
    """
    if company_data is None:
        raise ValueError("Company data cannot be None")
        
    if not isinstance(company_data, dict):
        raise ValueError("Company data must be a dictionary")
        
    if not company_data:  # Handle empty dict case
        return {}
        
    tree = {
        "name": company_data.get("company_name", "Unknown"),
        "number": company_data.get("company_number", "N/A"),
        "status": company_data.get("company_status", "Unknown"),
        "type": company_data.get("type", "Unknown"),
        "children": []
    }
    
    # Process subsidiaries
    subsidiaries = company_data.get("subsidiaries", [])
    for sub in subsidiaries:
        if isinstance(sub, dict):
            child_tree = build_company_tree(sub)
            tree["children"].append(child_tree)
            
    return tree

def validate_company_relationship(parent_type: str, child_type: str) -> bool:
    # Handle generic parent/subsidiary relationship
    pt = parent_type.lower()
    ct = child_type.lower()
    if (pt == "parent" and ct == "subsidiary") or (pt == "subsidiary" and ct == "sub-subsidiary"):
        return True
    valid_relationships = {
        "ltd": ["ltd", "plc", "llp", "charity"],
        "plc": ["ltd", "plc", "llp"],
        "llp": ["ltd", "llp"],
        "charity": ["ltd", "charity"],
        "holding": ["ltd", "plc", "llp", "charity"]
    }
    if pt not in valid_relationships:
        return False
    return ct in valid_relationships[pt]

def format_group_structure(company_hierarchy: Dict[str, Any]) -> str:
    """
    Formats company hierarchy into a readable string representation.
    
    Args:
        company_hierarchy (Dict[str, Any]): Company hierarchy data
        
    Returns:
        str: Formatted string representation of the hierarchy
        
    Raises:
        ValueError: If company_hierarchy is None or invalid
    """
    if company_hierarchy is None:
        raise ValueError("Company hierarchy cannot be None")
        
    if not isinstance(company_hierarchy, dict):
        raise ValueError("Company hierarchy must be a dictionary")
        
    if not company_hierarchy:  # Handle empty dict case
        return ""
        
    def format_node(node: Dict[str, Any], level: int = 0) -> str:
        indent = "  " * level
        result = []
        
        # Format current node
        name = node.get("company_name", "Unknown")
        number = node.get("company_number", "N/A")
        status = node.get("company_status", "Unknown")
        type_info = node.get("type", "Unknown")
        
        node_str = f"{indent}• {name} ({number})\n"
        node_str += f"{indent}  Status: {status}\n"
        node_str += f"{indent}  Type: {type_info}\n"
        result.append(node_str)
        
        # Format subsidiaries
        subsidiaries = node.get("subsidiaries", [])
        for sub in subsidiaries:
            if isinstance(sub, dict):
                result.append(format_node(sub, level + 1))
                
        return "\n".join(result)
    
    # Start with parent company
    parent = company_hierarchy.get("parent", {})
    if not parent:
        return ""
        
    return format_node(parent)

def _compute_max_depth(node: dict) -> int:
    if not node or not isinstance(node, dict):
        return 0
    subs = node.get("subsidiaries", [])
    if not subs:
        return 1 if node else 0
    return 1 + max((_compute_max_depth(sub) for sub in subs if isinstance(sub, dict)), default=0)

def analyze_company_group_structure(
    company_data: dict,
    api_key: str = "",
    base_scratch_dir: Any = None,
    logger_param: Any = None,
    ocr_handler: Any = None,
    analysis_mode: str = "profile_check",
    target_subsidiary_crns: Any = None,
    session_data: Any = None
) -> dict:
    if company_data is None:
        raise ValueError("Company data cannot be None")
    if not isinstance(company_data, dict):
        raise ValueError("Company data must be a dictionary")
    if logger_param is None:
        class DummyLogger:
            def info(self, msg): pass
        logger_param = DummyLogger()
    parent = company_data.get("parent", {})
    subs = find_subsidiaries(parent)
    total_companies = 1 + len(subs) if parent else 0
    max_depth = _compute_max_depth(parent) - 1 if parent else 0
    results = {
        "company_hierarchy": build_company_tree(parent),
        "subsidiaries": subs,
        "analysis_mode": analysis_mode,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "api_key_provided": bool(api_key),
            "ocr_handler_provided": bool(ocr_handler),
            "target_subsidiaries": target_subsidiary_crns or []
        },
        "total_companies": total_companies,
        "max_depth": max_depth
    }
    results["formatted_output"] = format_group_structure(company_data)
    logger_param.info(f"Completed {analysis_mode} analysis for company structure")
    return results
