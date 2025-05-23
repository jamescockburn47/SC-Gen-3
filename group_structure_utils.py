"""
Utilities for analyzing and visualizing company group structures from Companies House data.
Refactored for a staged, user-driven analysis process.
"""
import pathlib as _pl
import logging 
from datetime import datetime, timedelta 
import json 
import re 
import html 
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
    from ch_api_utils import get_ch_documents_metadata, get_company_profile, _fetch_document_content_from_ch
    import ch_api_utils 
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
    def __init__(self): self._state: Dict[str, Any] = {}
    def __getattr__(self, name: str) -> Any: return self._state.get(name)
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_state": super().__setattr__(name, value)
        else: self._state[name] = value
    def get(self, name: str, default: Any = None) -> Any: return self._state.get(name, default)
    def __contains__(self, key: str) -> bool: return key in self._state
    def update(self, *args: Any, **kwargs: Any) -> None: self._state.update(*args, **kwargs)
    def clear(self): self._state.clear() 

class MockStreamlit: 
    def __init__(self): self.session_state = MockSessionState()
    # Corrected: Each method definition on a new line
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
    def text_area(self, *args: Any, **kwargs: Any) -> str: # Added return type for consistency
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
# Maximum number of subsidiaries to visualise in diagrams. Set to ``None`` for no limit.
SUBSIDIARY_VIZ_LIMIT: Optional[int] = 20

# --- Function Definitions (Order Matters for Pylance) ---

def _parse_document_content(
    doc_content_data: Dict[str, Any], fetched_content_types: List[str], 
    company_no: str, doc_metadata: Dict[str, Any], logger: logging.Logger,
    ocr_handler: Optional[OCRHandlerType] = None, is_priority_for_ocr: bool = False
) -> Dict[str, Any]:
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
            content_to_parse, content_type_to_parse = doc_content_data[doc_type], doc_type; break
    
    if content_to_parse is None and 'pdf' in doc_content_data and doc_content_data['pdf'] is not None:
        content_to_parse, content_type_to_parse = doc_content_data['pdf'], 'pdf'
        if ocr_handler:
            if is_priority_for_ocr:
                logger.info(
                    f"GS Parse: PDF '{doc_metadata.get('description')}' prioritized for OCR if needed."
                )
            else:
                logger.info(
                    f"GS Parse: PDF '{doc_metadata.get('description')}' will be parsed; OCR available if needed."
                )
        else:
            logger.info(
                f"GS Parse: PDF '{doc_metadata.get('description')}' will be parsed without OCR handler."
            )

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
                parent_match = re.search(r"(?:ultimate parent company is|the parent company is|parent undertaking is|the immediate parent undertaking is)\s*([^,.(]+?)(?:,\s*a company incorporated in [^,.]+)?(?:\s*\(?(?:company no\.?|registered number|crn)?[:\s]*([A-Z0-9]+)\)?)?", text_lower, re.IGNORECASE)
                if parent_match:
                    p_name = html.unescape(parent_match.group(1).strip().title())
                    p_crn = parent_match.group(2).strip().upper() if parent_match.group(2) else None
                    if company_no.lower() not in p_name.lower() and (not p_crn or company_no.lower() != p_crn.lower()):
                        parsed_result["parent_company_name"] = p_name
                subs_list: List[Dict[str, Optional[str]]] = []
                subs_section_pattern = r"(?:subsidiary undertakings|principal subsidiary undertakings|investment in subsidiaries|details of related undertakings|group companies)([\s\S]+?)(?:\n\n[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]|\Z|notes to the financial statements|directors' report)"
                subs_section_match = re.search(subs_section_pattern, text_lower, re.IGNORECASE)
                if subs_section_match:
                    section_text = subs_section_match.group(1)
                    lines_to_scan = section_text.split('\n')
                else:
                    logger.info(
                        f"{company_no}: Subsidiary section not located; scanning entire document for potential matches."
                    )
                    lines_to_scan = extracted_text.split('\n')

                for line in lines_to_scan:
                    line_clean = line.strip()
                    if not line_clean or len(line_clean) < 3 or line_clean.lower().startswith(("note","total", "carrying value", "country of incorporation", "proportion of voting rights", "%", "£", "$", "€")):
                        continue
                    crn_match = re.search(CRN_REGEX, line_clean)
                    sub_crn = crn_match.group(1).upper() if crn_match else None
                    sub_name_candidate = line_clean
                    if crn_match and crn_match.group(0):
                        sub_name_candidate = sub_name_candidate.replace(crn_match.group(0), '').strip()
                    common_trails = [r'\s*\(?incorporated in [^)]+\)?', r'\s*registered in [^)]+\)?', r'\s*-\s*(?:100% owned|subsidiary|ordinary shares|share capital|\d+% holding)', r'\s*,\s*(?:united kingdom|england|scotland|wales|northern ireland|ireland|usa|etc\.)']
                    for trail in common_trails:
                        sub_name_candidate = re.sub(trail, '', sub_name_candidate, flags=re.IGNORECASE).strip()
                    sub_name_candidate = re.sub(r'\s*[,.]?$', '', sub_name_candidate).strip()
                    if (
                        re.search(r'\b(?:limited|ltd|plc|llp|lp|sarl|gmbh|bv|inc\.|incorporated|company|undertaking|group)\b', sub_name_candidate, re.IGNORECASE)
                        or (sub_crn and len(sub_name_candidate) > 3)
                    ) and not (
                        parsed_result.get("parent_company_name") and parsed_result.get("parent_company_name", "").lower() in sub_name_candidate.lower() and len(sub_name_candidate) > 3
                    ) and company_no.lower() not in sub_name_candidate.lower():
                        if not re.fullmatch(r'[\d\s.,%£$€()*]+', sub_name_candidate) and len(sub_name_candidate) > 2:
                            subs_list.append({"name": html.unescape(sub_name_candidate.title()), "number": sub_crn})
                if subs_list:
                    parsed_result["subsidiaries"] = list({frozenset(item.items()): item for item in subs_list}.values())
        except Exception as e: parsed_result["extraction_error"] = f"Extraction exception: {str(e)}"
    elif not parsed_result.get("extraction_error"): 
        parsed_result["extraction_error"] = "No suitable content type found or processed based on priority and availability."
    return parsed_result

def extract_parent_timeline(
    company_number: str, 
    api_key: str, 
    logger: logging.Logger, 
    parsed_docs_data: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    messages = ["Parent Company Timeline Analysis:"]
    logger.info(f"Extracting parent company timeline for {company_number} from parsed documents.")
    parent_timeline: List[Dict[str, Any]] = []
    if not parsed_docs_data:
        messages.append("No parsed document data provided for parent timeline generation.")
        logger.warning("extract_parent_timeline called without parsed_docs_data.")
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
        if not parsed or not metadata: continue
        doc_date_str = parsed.get("document_date") or metadata.get("date", "N/A")
        doc_desc = parsed.get("document_description") or metadata.get("description", "Unknown Document")
        parent_name = parsed.get("parent_company_name")
        parent_number = parsed.get("parent_company_number")
        s409_found = parsed.get("s409_info_found", False)
        if parent_name or parent_number:
            current_parent_number = parent_number.strip() if parent_number and isinstance(parent_number, str) and parent_number.strip() else None
            current_parent_name = parent_name.strip() if parent_name and isinstance(parent_name, str) and parent_name.strip() else None
            current_parent_key = (current_parent_name, current_parent_number)
            if current_parent_key != last_reported_parent_key or not parent_timeline:
                entry: Dict[str, Any] = {'date': doc_date_str, 'parent_name': current_parent_name, 'parent_number': current_parent_number,
                         'source_doc_desc': doc_desc, 's409_related': s409_found}
                parent_timeline.append(entry)
                message = f"- On {doc_date_str} ({doc_desc}): Reported parent as '{current_parent_name or 'N/A'}' (CRN: {current_parent_number or 'N/A'}). S409 related: {s409_found}."
                messages.append(message)
                last_reported_parent_key = current_parent_key
        elif s409_found: 
            current_parent_key = ("S409_INFO_ONLY", "S409_INFO_ONLY") 
            if current_parent_key != last_reported_parent_key or not parent_timeline:
                entry = {'date': doc_date_str, 'parent_name': None, 'parent_number': None,
                         'source_doc_desc': doc_desc, 's409_related': True}
                parent_timeline.append(entry)
                message = f"- On {doc_date_str} ({doc_desc}): S409 information found, but no specific parent details extracted. Implies group context."
                messages.append(message)
                last_reported_parent_key = current_parent_key
    if not parent_timeline: messages.append("No specific parent company mentions found in the analyzed documents to build a timeline.")
    else: messages.insert(1, f"Found {len(parent_timeline)} distinct parent changes/mentions over time:")
    return parent_timeline, messages


def gpt_fetch_public_group_structure(company_name: str, company_number: str, logger: logging.Logger) -> List[Dict[str, Optional[str]]]:
    """Use GPT-4.1 to gather subsidiary information from public sources."""
    openai_client = get_openai_client()
    if not openai_client:
        logger.warning("OpenAI client unavailable; skipping GPT public structure lookup.")
        return []
    try:
        prompt = (
            f"Provide a concise JSON list of known subsidiaries for {company_name} (CRN {company_number}) "
            f"based on publicly available sources. Return the format: {{'subsidiaries': [{{'name': 'XYZ Ltd', 'number': '12345678'}}]}}."
        )
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL_DEFAULT,
            messages=[
                {"role": "system", "content": "You are a corporate research assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.2,
        )
        raw_content = response.choices[0].message.content.strip()
        data = json.loads(raw_content)
        subs = data.get("subsidiaries", []) if isinstance(data, dict) else []
        cleaned: List[Dict[str, Optional[str]]] = []
        for item in subs:
            if isinstance(item, dict) and "name" in item:
                cleaned.append({"name": str(item.get("name")).strip(), "number": str(item.get("number") or "").strip() or None})
        return cleaned
    except Exception as e:
        logger.error(f"GPT public structure lookup failed for {company_number}: {e}", exc_info=True)
        return []


def _fetch_and_parse_selected_documents(
    company_number: str, docs_to_process: List[Dict[str, Any]], api_key: str, 
    logger: logging.Logger, ocr_handler: Optional[OCRHandlerType],
    is_topco_analyzing_subs: bool
) -> List[Dict[str, Any]]:
    processed_content_list = []
    if not ch_api_utils_available: return processed_content_list 
    for doc_item in docs_to_process:
        doc_desc, doc_date_str, doc_category = doc_item.get('description','N/A'), doc_item.get('date','N/A'), doc_item.get("category","N/A")
        priority_for_ocr = False
        if doc_category == "accounts":
            desc_lower = doc_desc.lower()
            if is_topco_analyzing_subs and any(keyword in desc_lower for keyword in ACCOUNTS_PRIORITY_ORDER):
                priority_for_ocr = True
            elif not is_topco_analyzing_subs and "full" in desc_lower:
                priority_for_ocr = True
        if not doc_item.get("links", {}).get("document_metadata"):
            processed_content_list.append({"metadata":doc_item, "parsed_data":{"errors":["No download link"], "document_description":doc_desc, "document_date":doc_date_str}}); continue
        raw_content_dict, fetched_types_list, fetch_err_msg = _fetch_document_content_from_ch(company_number, doc_item)
        if fetch_err_msg or not raw_content_dict:
            processed_content_list.append({"metadata":doc_item, "parsed_data":{"errors":[f"Fetch fail: {fetch_err_msg or 'No content'}"], "document_description":doc_desc, "document_date":doc_date_str}}); continue
        parsed_info = _parse_document_content(raw_content_dict, fetched_types_list, company_number, doc_item, logger, ocr_handler, priority_for_ocr)
        processed_content_list.append({"metadata":doc_item, "parsed_data":parsed_info})
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
        "visualization_data": None, "subsidiary_details_list": []
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
        report_messages.append("CRITICAL ERROR: CH API utilities not available."); return results

    if analysis_mode == "profile_check" or not results["company_profile"]:
        profile = get_company_profile(company_number, api_key)
        results["company_profile"] = profile
        if profile and isinstance(profile, dict):
            report_messages.append(f"Profile: {profile.get('company_name', company_number)} (Status: {profile.get('company_status', 'N/A')})")
            acc_req = profile.get("accounts", {}).get("accounting_requirement")
            if acc_req == "group": results["is_inferred_topco"] = True; report_messages.append("Profile suggests TopCo (group accounts).")
            elif acc_req: report_messages.append(f"Profile accounting: {acc_req}."); results["is_inferred_topco"] = False
        else:
            report_messages.append("Failed to fetch company profile or profile is invalid.")
            results["is_inferred_topco"] = session_data.get("user_stated_topco", False) if session_data else False
            report_messages.append(f"Proceeding with TopCo status as: {results['is_inferred_topco']} (profile unavailable).")

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

    parsed_docs_data: List[Dict[str, Any]] = []
    if docs_to_process_metadata:
        parsed_docs_data = _fetch_and_parse_selected_documents(
            company_number, docs_to_process_metadata, api_key, logger, ocr_handler,
            is_topco_analyzing_subs=(analysis_mode == "list_subsidiaries" and results["is_inferred_topco"])
        )
        results["downloaded_documents_content"] = parsed_docs_data

    if analysis_mode == "find_parent":
        pt, pt_msgs = extract_parent_timeline(company_number, api_key, logger, parsed_docs_data)
        results["parent_timeline"] = pt 
        report_messages.extend(pt_msgs)
        latest_parent_entry = next((e for e in reversed(results["parent_timeline"]) if e.get('parent_number')), None)
        if latest_parent_entry and isinstance(latest_parent_entry.get('parent_number'), str):
            results["identified_parent_crn"] = latest_parent_entry['parent_number']
        # The PSC-based parent lookup has been removed in favor of explicit user input or GPT analysis
        
        dot_lines = ["digraph G { rankdir=TB; node[shape=box];"]
        node_name = results.get("company_profile", {}).get('company_name', company_number) # Safe access
        dot_lines.append(f'"{company_number}" [label="{node_name}\\n({company_number})\\nAnalyzed"];')
        if results["identified_parent_crn"] and results["identified_parent_crn"] != company_number:
            parent_crn_str = str(results["identified_parent_crn"]) # Ensure string
            parent_profile_viz = get_company_profile(parent_crn_str, api_key)
            parent_name_viz = parent_profile_viz.get('company_name', parent_crn_str) if parent_profile_viz else parent_crn_str
            dot_lines.append(f'"{parent_crn_str}" [label="{parent_name_viz}\\n({parent_crn_str})\\nIdentified Parent", color=lightgreen];')
            dot_lines.append(f'"{parent_crn_str}" -> "{company_number}";')
        dot_lines.append("}")
        results["visualization_data"] = "\n".join(dot_lines)

    elif analysis_mode == "list_subsidiaries":
        current_subs_evolution = results["subsidiary_evolution"] 
        for doc_bundle in parsed_docs_data:
            parsed, meta = doc_bundle.get("parsed_data",{}), doc_bundle.get("metadata",{})
            doc_date_str = meta.get("date", "N/A")
            subs_in_doc = parsed.get("subsidiaries", []) # Ensure it's a list
            if subs_in_doc and doc_date_str != "N/A":
                try:
                    year = datetime.strptime(doc_date_str, "%Y-%m-%d").year
                    existing_subs_repr = {frozenset(s.items()) for s in current_subs_evolution.get(year, [])}
                    new_subs = [s for s in subs_in_doc if isinstance(s, dict) and frozenset(s.items()) not in existing_subs_repr] # Check type
                    current_subs_evolution[year].extend(new_subs)
                except ValueError: pass
        results["subsidiary_evolution"] = {k:v for k,v in current_subs_evolution.items()}
        
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
            report_messages.append(f"Fetching details for {len(target_subsidiary_crns)} selected subsidiaries.")
            for sub_crn in target_subsidiary_crns:
                sub_profile = get_company_profile(sub_crn, api_key)
                sub_detail = {"crn": sub_crn, "name": "N/A", "status": "N/A"}
                if sub_profile and isinstance(sub_profile, dict):
                    sub_detail["name"] = sub_profile.get("company_name", sub_crn)
                    sub_detail["status"] = sub_profile.get("company_status", "N/A")
                results["subsidiary_details_list"].append(sub_detail)
        else: report_messages.append("No target subsidiaries provided for detail fetching.")
    return results

def get_all_subsidiaries(subsidiary_evolution: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Combine subsidiary lists from all years, deduplicating entries."""
    all_subs: List[Dict[str, Any]] = []
    seen: Set[frozenset] = set()
    for year in sorted(subsidiary_evolution.keys()):
        for sub in subsidiary_evolution.get(year, []):
            if not isinstance(sub, dict):
                continue
            repr_set = frozenset(sub.items())
            if repr_set not in seen:
                seen.add(repr_set)
                all_subs.append(sub)
    return all_subs

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
        if key not in st.session_state: st.session_state[key] = default_val # type: ignore

    input_crn = st.text_input("Enter Company Number to Analyze:", st.session_state.get('gs_current_company_crn', ""), key="gs_crn_input_main") # type: ignore
    if input_crn != st.session_state.get('gs_current_company_crn'):
        st.session_state.gs_current_company_crn = input_crn.strip().upper() # type: ignore
        st.session_state.gs_analysis_results, st.session_state.gs_current_stage, st.session_state.gs_user_stated_topco = {}, "initial_input", False # type: ignore
    
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

    if st.button("1. Fetch Profile & Determine Path", key="gs_fetch_profile_btn"): # type: ignore
        if not st.session_state.get('gs_current_company_crn'): st.warning("Please enter a company number.") # type: ignore
        else:
            with st.spinner(f"Fetching profile for {st.session_state.get('gs_current_company_crn')}..."): # type: ignore
                current_session_data = {"company_profile": None, "is_inferred_topco": st.session_state.get('gs_user_stated_topco'), "user_stated_topco": st.session_state.get('gs_user_stated_topco')}
                results = analyze_company_group_structure(
                    company_number=st.session_state.get('gs_current_company_crn',''), api_key=api_key, base_scratch_dir=base_scratch_dir,  # Added default for get
                    logger=logger, ocr_handler=ocr_handler, analysis_mode="profile_check", session_data=current_session_data
                )
                st.session_state.gs_analysis_results, st.session_state.gs_current_stage = results, "profile_reviewed" # type: ignore
                st.rerun() # type: ignore

    results = st.session_state.get('gs_analysis_results', {})
    if results and st.session_state.get('gs_current_stage') in ["profile_reviewed", "parent_analyzed", "subs_listed", "subs_detailed"]:
        st.markdown("---"); st.subheader(f"Analysis for: {results.get('company_number_analyzed', '')}") # type: ignore
        profile = results.get("company_profile")
        if profile and isinstance(profile, dict) : 
            st.markdown(f"**Name:** {profile.get('company_name', 'N/A')} | **Status:** {profile.get('company_status', 'N/A')}") # type: ignore
            st.caption(f"Inferred TopCo: {results.get('is_inferred_topco', 'Unknown')}") # type: ignore
        else: st.warning("Company profile not fetched or unavailable.") # type: ignore
    
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
        st.subheader("Parent Company Analysis Results"); # type: ignore
        identified_parent_crn = results.get("identified_parent_crn")
        if identified_parent_crn and identified_parent_crn != results.get("company_number_analyzed"):
            st.success(f"Potential Parent Identified: CRN {identified_parent_crn}") # type: ignore
            if st.button(f"Analyze Parent ({identified_parent_crn})", key=f"gs_analyze_parent_{identified_parent_crn}"): # type: ignore
                st.session_state.gs_current_company_crn, st.session_state.gs_user_stated_topco = identified_parent_crn, True # type: ignore
                st.session_state.gs_analysis_results, st.session_state.gs_current_stage = {}, "initial_input" # type: ignore
                st.rerun() # type: ignore
        if results.get("visualization_data"): st.graphviz_chart(results["visualization_data"]) # type: ignore

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

    if results and results.get("report_messages") and len(results.get("report_messages",[])) > 1:
        with st.expander("Detailed Analysis Log", expanded=False): # type: ignore
            for msg in results.get("report_messages", []): st.caption(msg) # type: ignore
