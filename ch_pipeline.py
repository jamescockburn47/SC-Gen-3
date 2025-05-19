# ch_pipeline.py

from __future__ import annotations
import datetime
import json
import logging
import os
import re
import shutil
import subprocess # Not used in the current version, consider removing if not needed
import tempfile # Not used in the current version, consider removing if not needed
from collections import Counter, defaultdict # Counter not used
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypeAlias, Any # Make sure TypeAlias is used or remove

# Attempt to import config first as it defines logger and other constants
try:
    import config
except ImportError as import_err:
    # Try loading config relative to this file
    import importlib.util
    import sys
    config_path = Path(__file__).resolve().parent / "config.py"
    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec and spec.loader:
            config = importlib.util.module_from_spec(spec)
            sys.modules["config"] = config
            spec.loader.exec_module(config)
        else:
            raise import_err
    except Exception as e_load:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load config.py from {config_path}: {e_load}")
        raise

from config import (
    APPLICATION_SCRATCH_DIR,
    CH_API_MAX_RETRY,  # Not directly used in this file after refactor, but ch_api_utils uses it
    CH_API_RETRY_BACKOFF_FACTOR,  # Not directly used
    CH_API_RETRY_STATUS_FORCELIST,
    CH_API_TIMEOUT,  # Not directly used
    logger,
    CH_API_USER_AGENT,
    MIN_MEANINGFUL_TEXT_LEN,
)


try:
    import xml.etree.ElementTree as ET # Standard library, should be fine
except ImportError:
    logger.error("Failed to import xml.etree.ElementTree. XML parsing will fail.")
    ET = None # type: ignore

try:
    import xmltodict
except ImportError:
    logger.warning("xmltodict library not installed. Some XML/iXBRL processing might be affected.")
    xmltodict = None # type: ignore

try:
    import openai
except ImportError:
    logger.warning("openai library not installed. OpenAI dependent features will fail.")
    openai = None # type: ignore

try:
    import google.generativeai as genai
except ImportError:
    logger.warning("google-generativeai library not installed. Gemini dependent features will fail.")
    genai = None # type: ignore

try:
    import requests
    from requests.adapters import Retry, HTTPAdapter
    RequestException = requests.exceptions.RequestException
except ImportError:
    logger.warning("requests library not installed. All CH API calls will fail.")
    requests = None # type: ignore
    Retry = None # type: ignore
    HTTPAdapter = None # type: ignore
    RequestException = Exception # type: ignore


# CH API Interactions (already robustly imported in previous versions)
from ch_api_utils import (
    get_ch_documents_metadata,
    _fetch_document_content_from_ch,
    get_company_profile,
)

# Text Extraction Utilities & Type Alias
try:
    from text_extraction_utils import extract_text_from_document, OCRHandlerType
    TEXT_EXTRACTOR_AVAILABLE = True
    logger.info("text_extraction_utils.py and OCRHandlerType found and imported successfully.")
except ImportError:
    logger.error("text_extraction_utils.py not found or OCRHandlerType missing. Text extraction will fail or be limited.")
    TEXT_EXTRACTOR_AVAILABLE = False
    # Define a dummy OCRHandlerType for type hinting if import fails
    OCRHandlerType = Callable[[bytes, str], Tuple[str, int, Optional[str]]] # type: ignore
    def extract_text_from_document(*args: Any, **kwargs: Any) -> Tuple[str, int, Optional[str]]:
        return "Error: text_extraction_utils.py not found.", 0, "text_extraction_utils.py is missing."

# AI Summarization
from ai_utils import gpt_summarise_ch_docs, gemini_summarise_ch_docs

# Optional AWS Textract Import
try:
    from aws_textract_utils import perform_textract_ocr, get_textract_cost_estimation, _initialize_aws_clients as initialize_textract_aws_clients
    TEXTRACT_AVAILABLE = True # This refers to the module being importable
    logger.info("aws_textract_utils.py found and imported successfully.")
except ImportError:
    perform_textract_ocr = None # type: ignore
    get_textract_cost_estimation = None # type: ignore
    initialize_textract_aws_clients = None # type: ignore
    TEXTRACT_AVAILABLE = False
    logger.warning("aws_textract_utils.py not found. Textract-related functions will be disabled.")

# ---------------------------------------------------------------------------- #
# Global Configurations and Constants (now mostly from config.py)
# ---------------------------------------------------------------------------- #
CH_API_BASE_URL = "https://api.company-information.service.gov.uk"
# CH_API_USER_AGENT is imported from config
DEFAULT_HEADERS = {"User-Agent": CH_API_USER_AGENT if CH_API_USER_AGENT else "StrategicComplianceGen3/1.0 (Default)"}
SCRATCH_DIR = Path(APPLICATION_SCRATCH_DIR).expanduser() 
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
# CH_API_RETRY_STATUS_FORCELIST is imported from config

# ---------------------------------------------------------------------------- #
# Helper Functions (Disk, Network, Parsing)
# ---------------------------------------------------------------------------- #

def _save_raw_document_content(
    doc_content: Union[bytes, str, Dict[str, Any]], # Updated Dict hint
    doc_type_str: str, 
    company_no: str,
    ch_doc_transaction_id: str, 
    doc_year: int,
    scratch_dir: Path
) -> Optional[Path]:
    """
    Persist the fetched document to the scratch directory and return the path,
    or None on failure.
    """
    file_extension_map = {"pdf": "pdf", "xhtml": "xhtml", "json": "json", "xml": "xml"}
    file_extension = file_extension_map.get(doc_type_str.lower(), "dat")

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    safe_transaction_id = re.sub(r'[^\w\-.]', '_', ch_doc_transaction_id)
    doc_filename_prefix = f"{company_no}_{safe_transaction_id}_{doc_year}_{timestamp_str}"
    doc_save_path = scratch_dir / f"{doc_filename_prefix}.{file_extension}"

    try:
        if isinstance(doc_content, bytes):
            doc_save_path.write_bytes(doc_content)
        elif isinstance(doc_content, str):
            doc_save_path.write_text(doc_content, encoding="utf-8")
        elif isinstance(doc_content, dict) and doc_type_str.lower() == "json":
            with open(doc_save_path, "w", encoding="utf-8") as f_json:
                json.dump(doc_content, f_json, indent=2, default=str)
        else:
            logger.error(
                "Cannot save document: unsupported content type %s for '%s'",
                type(doc_content), doc_type_str,
            )
            return None
        logger.debug("Saved fetched %s document to %s", doc_type_str.upper(), doc_save_path.name)
        return doc_save_path
    except IOError as e_save:
        logger.error("Failed to save %s (%s): %s", doc_type_str.upper(), doc_save_path.name, e_save)
        return None


def _cleanup_scratch_directory(scratch_dir: Path, keep_days: int):
    if keep_days < 0:
        logger.info(f"Scratch directory cleanup skipped for {scratch_dir} (keep_days: {keep_days}).")
        return

    timestamp_cutoff = datetime.datetime.now().timestamp() - keep_days * 86400
    num_files_cleaned = 0
    try:
        for item_path in scratch_dir.iterdir():
            if item_path.is_file():
                try:
                    if item_path.stat().st_mtime < timestamp_cutoff or keep_days == 0:
                        item_path.unlink()
                        num_files_cleaned += 1
                except OSError as e_unlink:
                    logger.warning(f"Could not delete old file {item_path.name} from scratch: {e_unlink}")
        if num_files_cleaned > 0:
            logger.info(f"Cleaned up {num_files_cleaned} old files from scratch directory: {scratch_dir}")
    except Exception as e_cleanup:
        logger.error(f"Error during scratch directory cleanup ({scratch_dir}): {e_cleanup}")

def get_relevant_filings_metadata(
    company_number: str,
    api_key: str,
    years_back: int = 5,
    categories_to_fetch: List[str] = ["accounts", "confirmation-statement", "capital", "mortgage", "officers"],
    filters: Optional[Dict[str, Union[str, List[str]]]] = None,
    items_per_page_api: int = 50,
    max_docs_to_scan_api: int = 200,
    target_docs_in_range_api: int = 20
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    current_year = datetime.datetime.now().year
    start_year = current_year - years_back
    end_year = current_year
    year_range_tuple = (start_year, end_year)

    logger.info(f"Fetching CH metadata for {company_number}, years: {start_year}-{end_year}, categories: {categories_to_fetch}")

    all_filings_metadata, company_profile, error_msg = get_ch_documents_metadata(
        company_no=company_number, api_key=api_key, categories=categories_to_fetch,
        items_per_page=items_per_page_api, max_docs_to_fetch_meta=max_docs_to_scan_api,
        target_docs_per_category_in_date_range=target_docs_in_range_api, year_range=year_range_tuple
    )

    if error_msg:
        logger.error(f"Error fetching metadata via ch_api_utils for {company_number}: {error_msg}")
        return [], company_profile, error_msg
    if not all_filings_metadata:
        logger.info(f"No filings metadata returned from ch_api_utils for {company_number} with given criteria.")
        return [], company_profile, None

    relevant_filings_after_filter = []
    if filters:
        for filing in all_filings_metadata:
            match = True
            for key, value_filter in filters.items():
                if key in filing:
                    if isinstance(value_filter, list):
                        if filing[key] not in value_filter: match = False; break
                    else:
                        if filing[key] != value_filter: match = False; break
                else: match = False; break
            if match: relevant_filings_after_filter.append(filing)
        logger.info(f"Applied local filters to {len(all_filings_metadata)} API results, yielding {len(relevant_filings_after_filter)} filings for {company_number}.")
        return relevant_filings_after_filter, company_profile, None
    else:
        return all_filings_metadata, company_profile, None

class CompanyHouseDocumentPipeline:
    def __init__(
        self, company_number: str, ch_api_key: Optional[str] = None,
        text_extractor_available: bool = TEXT_EXTRACTOR_AVAILABLE,
        textract_available: bool = TEXTRACT_AVAILABLE, # Module availability
        use_textract_for_ocr_if_available: bool = True,
        scratch_dir: Path = SCRATCH_DIR, 
        keep_days_in_scratch: int = 14,
    ) -> None:
        self.company_number = company_number
        self.ch_api_key = ch_api_key
        self.text_extractor_available = text_extractor_available
        self.textract_available = textract_available 
        self.use_textract_for_ocr_if_available = use_textract_for_ocr_if_available
        self.scratch_dir = scratch_dir 
        self.keep_days_in_scratch = keep_days_in_scratch
        self.company_scratch_dir = self.scratch_dir / self.company_number
        self.company_scratch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialised CompanyHouseDocumentPipeline for {self.company_number} | API Key: {'Yes' if ch_api_key else 'No'} | TextExt: {self.text_extractor_available} | TextractMod: {self.textract_available} | UseTextract: {self.use_textract_for_ocr_if_available}")

    def run(self, years_back: int = 5, categories_to_fetch: List[str] = ["accounts", "confirmation-statement"]) -> Dict[str, Any]:
        _cleanup_scratch_directory(self.company_scratch_dir, self.keep_days_in_scratch)
        if not self.ch_api_key:
            logger.error(f"Pipeline run for {self.company_number} aborted: CH API Key missing.")
            return {"company_number": self.company_number, "error": "CH API Key missing.", "filings_metadata_count": 0, "filings_downloaded_count": 0, "filings_processed_count": 0, "summary": {}}
        try:
            filings_metadata, company_profile, meta_error = get_relevant_filings_metadata(
                company_number=self.company_number, api_key=self.ch_api_key, years_back=years_back, categories_to_fetch=categories_to_fetch
            )
            if meta_error: raise Exception(f"Metadata fetching failed: {meta_error}")
            logger.info(f"Fetched {len(filings_metadata)} filings metadata for {self.company_number} (last {years_back} yrs, cats: {categories_to_fetch})")
            downloaded_docs_info = self._download_filings(filings_metadata)
            processed_docs_details = self._process_documents(downloaded_docs_info)
            summary_by_year = self._summarise_documents(processed_docs_details)
            return {"company_number": self.company_number, "company_profile": company_profile, "filings_metadata_count": len(filings_metadata),
                    "filings_downloaded_count": len(downloaded_docs_info), "filings_processed_count": len(processed_docs_details),
                    "summary_by_year": summary_by_year, "processed_document_details": processed_docs_details}
        except Exception as e_run:
            logger.exception(f"Pipeline run failed for {self.company_number}: {e_run}")
            return {"company_number": self.company_number, "error": str(e_run), "filings_metadata_count": 0, "filings_downloaded_count": 0, "filings_processed_count": 0, "summary": {}}

    def _download_filings(self, filings_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        downloaded_docs_output = []
        for filing_item_meta in filings_metadata_list:
            try:
                content_data_dict, fetched_types_list, fetch_error = _fetch_document_content_from_ch(self.company_number, filing_item_meta)
                if fetch_error: logger.error(f"Failed to fetch content for doc TX_ID {filing_item_meta.get('transaction_id', 'N/A')}: {fetch_error}"); continue
                if not content_data_dict or not fetched_types_list: logger.warning(f"No content/types for doc TX_ID {filing_item_meta.get('transaction_id', 'N/A')}"); continue
                content_to_save, content_type_to_save_str = None, None
                for pref_type in ["json", "xhtml", "xml", "pdf"]:
                    if pref_type in content_data_dict and content_data_dict[pref_type] is not None:
                        content_to_save, content_type_to_save_str = content_data_dict[pref_type], pref_type; break
                if not content_to_save or not content_type_to_save_str: logger.warning(f"No suitable primary content for doc TX_ID {filing_item_meta.get('transaction_id', 'N/A')}"); continue
                
                filing_date_str = filing_item_meta.get("date", "")
                filing_year = datetime.datetime.strptime(filing_date_str, "%Y-%m-%d").year if filing_date_str else datetime.datetime.now().year
                transaction_id = filing_item_meta.get("transaction_id", f"UnknownID_{filing_date_str.replace('-','') if filing_date_str else 'NODATE'}")
                local_path = _save_raw_document_content(content_to_save, content_type_to_save_str, self.company_number, transaction_id, filing_year, self.company_scratch_dir)
                if local_path:
                    downloaded_docs_output.append({"local_path": local_path, "original_metadata": filing_item_meta, "saved_content_type": content_type_to_save_str, "all_fetched_types": fetched_types_list})
                    logger.info(f"Saved {content_type_to_save_str.upper()} for doc TX_ID {transaction_id} ({self.company_number}) to {local_path.name}")
            except Exception as e_dl_loop: logger.exception(f"Error downloading/saving doc TX_ID {filing_item_meta.get('transaction_id', 'N/A_dl_loop')}: {e_dl_loop}")
        return downloaded_docs_output

    def _process_documents(self, downloaded_docs_info_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_docs_results = []
        for doc_info in downloaded_docs_info_list:
            local_path = doc_info.get("local_path")
            saved_content_type = doc_info.get("saved_content_type")
            original_meta = doc_info.get("original_metadata", {})
            tx_id_log = original_meta.get('transaction_id', local_path.name if local_path else 'N/A_proc')
            current_doc_result = {**doc_info, "extracted_text": None, "num_pages_ocrd": 0, "extraction_error": None}
            if not local_path or not Path(local_path).exists() or not saved_content_type: # Check Path object for exists()
                logger.warning(f"Skipping processing for {tx_id_log}: path invalid/missing or type missing.")
                current_doc_result["extraction_error"] = "Missing path/type or file does not exist"
                processed_docs_results.append(current_doc_result); continue
            try:
                doc_content_for_extraction: Union[bytes, str, Dict[str, Any]]
                if saved_content_type == "pdf": doc_content_for_extraction = Path(local_path).read_bytes()
                elif saved_content_type == "json": doc_content_for_extraction = json.loads(Path(local_path).read_text(encoding="utf-8"))
                else: doc_content_for_extraction = Path(local_path).read_text(encoding="utf-8", errors="ignore")

                ocr_handler_to_use: Optional[OCRHandlerType] = None # Explicitly Optional
                if saved_content_type == "pdf" and self.use_textract_for_ocr_if_available and TEXTRACT_AVAILABLE and perform_textract_ocr:
                    ocr_handler_to_use = perform_textract_ocr
                elif saved_content_type == "pdf":
                    logger.warning(f"PDF {tx_id_log}: Textract OCR not used/available for this pipeline instance.")
                    current_doc_result["extraction_error"] = "PDF found, Textract OCR not configured for use."
                    processed_docs_results.append(current_doc_result); continue
                
                text, num_pages, extractor_error_msg = extract_text_from_document(
                    doc_content_input=doc_content_for_extraction, content_type_input=saved_content_type,
                    company_no_for_logging=self.company_number, ocr_handler=ocr_handler_to_use
                )
                current_doc_result.update({"extracted_text": text, "num_pages_ocrd": num_pages if ocr_handler_to_use else 0, "extraction_error": extractor_error_msg})
                if extractor_error_msg: logger.error(f"Extraction error for {tx_id_log} ({saved_content_type}): {extractor_error_msg}")
                elif not text or len(text.strip()) < MIN_MEANINGFUL_TEXT_LEN: logger.warning(f"Short text from {tx_id_log} ({saved_content_type}): {len(text.strip()) if text else 0} chars.")
                else: logger.info(f"Extracted text from {tx_id_log} ({saved_content_type}, {len(text)} chars). OCR pages: {current_doc_result['num_pages_ocrd']}")
            except Exception as e_proc:
                logger.exception(f"Failed to process {tx_id_log} ({saved_content_type}): {e_proc}")
                current_doc_result["extraction_error"] = str(e_proc)
            processed_docs_results.append(current_doc_result)
        return processed_docs_results

    def _summarise_documents(self, processed_docs_details: List[Dict[str, Any]]) -> Dict[int, str]:
        docs_by_year: Dict[int, List[str]] = defaultdict(list)
        for doc_detail in processed_docs_details:
            if doc_detail.get("extracted_text") and not doc_detail.get("extraction_error"):
                try:
                    meta_date_str = doc_detail.get("original_metadata", {}).get("date")
                    if meta_date_str:
                        filing_year = datetime.datetime.strptime(meta_date_str, "%Y-%m-%d").year
                        docs_by_year[filing_year].append(doc_detail["extracted_text"])
                except ValueError: logger.warning(f"Could not parse date for {doc_detail.get('local_path', 'Unknown Doc')}")
        annual_summaries = {}
        for year, texts_for_year in docs_by_year.items():
            if not texts_for_year: continue
            combined_text_for_year = "\n\n--- Next Document ---\n\n".join(texts_for_year)
            MAX_CHARS_FOR_ANNUAL_SUMMARY = 750_000 
            if len(combined_text_for_year) > MAX_CHARS_FOR_ANNUAL_SUMMARY:
                logger.warning(f"Combined text for {year} ({self.company_number}, {len(combined_text_for_year)} chars) exceeds limit. Truncating.")
                combined_text_for_year = combined_text_for_year[:MAX_CHARS_FOR_ANNUAL_SUMMARY]
            try:
                summary_text = "Error in summarization"
                # Check config for API keys to decide which service to use
                # Ensure config is the imported module, not the dummy class if import failed
                gemini_key_present = hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY # type: ignore
                openai_key_present = hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY # type: ignore

                if genai and gemini_key_present: 
                    logger.info(f"Summarizing {len(texts_for_year)} docs for {self.company_number} (Year {year}) using Gemini.")
                    summary_text, _, _ = gemini_summarise_ch_docs(
                        text_to_summarize=combined_text_for_year, company_no=f"{self.company_number}_Year_{year}",
                        specific_instructions=f"Summarize key events, financial trends, governance changes for {self.company_number} in {year}.",
                        model_name=config.GEMINI_MODEL_DEFAULT if hasattr(config, 'GEMINI_MODEL_DEFAULT') else "gemini-1.5-pro-latest" # type: ignore
                    )
                elif openai and openai_key_present: 
                    logger.info(f"Summarizing {len(texts_for_year)} docs for {self.company_number} (Year {year}) using OpenAI.")
                    summary_text, _, _ = gpt_summarise_ch_docs(
                        text_to_summarize=combined_text_for_year, company_no=f"{self.company_number}_Year_{year}",
                        specific_instructions=f"Summarize key events, financial trends, governance changes for {self.company_number} in {year}.",
                        model_to_use=config.OPENAI_MODEL_DEFAULT if hasattr(config, 'OPENAI_MODEL_DEFAULT') else "gpt-4o-mini" # type: ignore
                    )
                else:
                    logger.warning(f"No AI summarization client (Gemini/OpenAI) available/configured for {self.company_number} (Year {year}).")
                    summary_text = "AI summarization service not available."
                annual_summaries[year] = summary_text
                logger.info(f"Summarized {len(texts_for_year)} docs for {self.company_number} (Year {year}). Length: {len(summary_text)} chars.")
            except Exception as e_sum:
                logger.warning(f"Failed to summarise {len(texts_for_year)} docs for {self.company_number} (Year {year}): {e_sum}", exc_info=True)
                annual_summaries[year] = f"Error during summarization for year {year}: {e_sum}"
        return annual_summaries

def run_batch_company_analysis(
    company_numbers_list: List[str],
    selected_filings_metadata_by_company: Dict[str, List[Dict[str, Any]]], 
    company_profiles_map: Dict[str, Dict[str, Any]], 
    ch_api_key_batch: str,
    model_prices_gbp: Dict[str, float], 
    specific_ai_instructions: str = "",
    filter_keywords_str: Optional[List[str]] = None, 
    base_scratch_dir: Path = SCRATCH_DIR, 
    keep_days: int = 7,
    use_textract_ocr: bool = False 
) -> Tuple[Optional[Path], Dict[str, Any]]:
    run_timestamp = datetime.datetime.now()
    run_id = run_timestamp.strftime("%Y%m%d_%H%M%S")
    batch_output_dir = base_scratch_dir / f"batch_run_{run_id}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_scratch_directory(base_scratch_dir, keep_days)
    all_processed_docs_data = []
    total_docs_analyzed_count, total_textract_pages_processed_count = 0, 0
    total_ai_summarization_cost_gbp, total_prompt_tokens_all_docs, total_completion_tokens_all_docs = 0.0, 0, 0
    ocr_handler_for_extraction: Optional[OCRHandlerType] = None
    if use_textract_ocr and TEXTRACT_AVAILABLE and perform_textract_ocr:
        if initialize_textract_aws_clients and initialize_textract_aws_clients():
            ocr_handler_for_extraction = perform_textract_ocr
            logger.info(f"Batch Run {run_id}: AWS Textract OCR will be used.")
        else: logger.warning(f"Batch Run {run_id}: Textract OCR requested, but AWS clients init failed. OCR skipped.")
    elif use_textract_ocr: logger.warning(f"Batch Run {run_id}: Textract OCR requested, but module/function not available. OCR skipped.")

    for company_no in company_numbers_list:
        logger.info(f"Batch Run {run_id}: Processing company {company_no}")
        company_scratch_dir = batch_output_dir / company_no
        company_scratch_dir.mkdir(exist_ok=True)
        company_filings_to_process = selected_filings_metadata_by_company.get(company_no, [])
        company_profile_data = company_profiles_map.get(company_no)
        if not company_filings_to_process: logger.warning(f"Batch Run {run_id}: No filings for {company_no}. Skipping."); continue

        for filing_meta in company_filings_to_process:
            total_docs_analyzed_count += 1
            doc_date_str = filing_meta.get("date", "N/A")
            doc_result_row: Dict[str, Any] = { # Initialize with Nones for Optional fields
                "company_number": company_no, "company_name": company_profile_data.get("company_name", "N/A") if company_profile_data else "N/A",
                "document_type": filing_meta.get("type", "N/A"), "document_date": doc_date_str,
                "document_description": filing_meta.get("description", "N/A"), "transaction_id": filing_meta.get("transaction_id", "N/A"),
                "api_category": filing_meta.get("category", "N/A"), "text_extraction_status": "Pending", "extracted_text_length": 0,
                "ocr_pages_processed": 0, "ai_summary_status": "Pending", "summary": "", "ai_model_used_for_summary": "N/A",
                "prompt_tokens_summary": 0, "completion_tokens_summary": 0, "cost_gbp_summary": 0.0,
                "local_path_raw_doc": None, "processing_error": None # Explicitly None
            }
            try:
                logger.info(f"Batch Run {run_id} ({company_no}): Fetching content for doc TX_ID: {doc_result_row['transaction_id']}")
                content_dict, fetched_types, fetch_err = _fetch_document_content_from_ch(company_no, filing_meta)
                if fetch_err: raise Exception(f"Content fetch failed: {fetch_err}")
                if not content_dict or not fetched_types: raise Exception("No content fetched from API.")
                primary_content_data, primary_content_type_str = None, None
                for pref_type in ["json", "xhtml", "xml", "pdf"]:
                    if pref_type in content_dict and content_dict[pref_type] is not None:
                        primary_content_data, primary_content_type_str = content_dict[pref_type], pref_type; break
                if not primary_content_data or not primary_content_type_str: raise Exception(f"No suitable primary content. Fetched: {fetched_types}")
                
                current_year_for_save = datetime.datetime.strptime(doc_date_str, "%Y-%m-%d").year if doc_date_str != "N/A" else run_timestamp.year
                saved_path = _save_raw_document_content(primary_content_data, primary_content_type_str, company_no, doc_result_row['transaction_id'], current_year_for_save, company_scratch_dir)
                doc_result_row["local_path_raw_doc"] = str(saved_path) if saved_path else None

                logger.info(f"Batch Run {run_id} ({company_no}): Extracting text from {primary_content_type_str} for doc TX_ID: {doc_result_row['transaction_id']}")
                extracted_text, pages_ocrd, extraction_err = extract_text_from_document(
                    primary_content_data, primary_content_type_str, f"{company_no}_{doc_result_row['transaction_id']}", ocr_handler_for_extraction
                )
                doc_result_row["text_extraction_status"] = f"Error: {extraction_err}" if extraction_err else ("Success (Short Text)" if not extracted_text or len(extracted_text.strip()) < MIN_MEANINGFUL_TEXT_LEN else "Success")
                if extraction_err: logger.warning(f"Batch Run {run_id} ({company_no}): Extraction error for TX_ID {doc_result_row['transaction_id']} ({primary_content_type_str}): {extraction_err}")
                doc_result_row["extracted_text_length"] = len(extracted_text.strip()) if extracted_text else 0
                if pages_ocrd > 0: doc_result_row["ocr_pages_processed"] = pages_ocrd; total_textract_pages_processed_count += pages_ocrd

                if doc_result_row["text_extraction_status"] == "Success" and extracted_text:
                    logger.info(f"Batch Run {run_id} ({company_no}): Summarizing doc TX_ID: {doc_result_row['transaction_id']}")
                    summary_content, p_tokens, c_tokens = "AI Summarization Skipped (No suitable model/config)", 0, 0
                    ai_model_to_use_for_summary = "" # Will be set based on availability
                    summarizer_func_to_call = None
                    
                    gemini_key_ok = genai and hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY # type: ignore
                    openai_key_ok = openai and hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY # type: ignore

                    if gemini_key_ok:
                        ai_model_to_use_for_summary = config.GEMINI_MODEL_DEFAULT if hasattr(config, 'GEMINI_MODEL_DEFAULT') else "gemini-1.5-pro-latest" # type: ignore
                        summarizer_func_to_call = gemini_summarise_ch_docs
                        logger.debug(f"Batch Run {run_id} ({company_no}): Using Gemini ('{ai_model_to_use_for_summary}') for CH summary.")
                    elif openai_key_ok:
                        ai_model_to_use_for_summary = config.OPENAI_MODEL_DEFAULT if hasattr(config, 'OPENAI_MODEL_DEFAULT') else "gpt-4o-mini" # type: ignore
                        summarizer_func_to_call = gpt_summarise_ch_docs
                        logger.debug(f"Batch Run {run_id} ({company_no}): Using OpenAI ('{ai_model_to_use_for_summary}') for CH summary (Gemini unavailable).")
                    
                    doc_result_row["ai_model_used_for_summary"] = ai_model_to_use_for_summary if ai_model_to_use_for_summary else "N/A"

                    if summarizer_func_to_call and ai_model_to_use_for_summary:
                        # Correct parameter name for gpt_summarise_ch_docs
                        if summarizer_func_to_call == gpt_summarise_ch_docs:
                            summary_content, p_tokens, c_tokens = summarizer_func_to_call(
                                text_to_summarize=extracted_text, company_no=company_no,
                                specific_instructions=specific_ai_instructions, model_to_use=ai_model_to_use_for_summary # Changed here
                            )
                        else: # For gemini_summarise_ch_docs
                             summary_content, p_tokens, c_tokens = summarizer_func_to_call(
                                text_to_summarize=extracted_text, company_no=company_no,
                                specific_instructions=specific_ai_instructions, model_name=ai_model_to_use_for_summary
                            )
                        doc_result_row["ai_summary_status"] = "Success" if "Error:" not in summary_content else f"Error: {summary_content}"
                    else: doc_result_row["ai_summary_status"] = "Error: No AI summarization client configured."
                    doc_result_row.update({"summary": summary_content, "prompt_tokens_summary": p_tokens, "completion_tokens_summary": c_tokens})
                    total_prompt_tokens_all_docs += p_tokens; total_completion_tokens_all_docs += c_tokens
                    cost_this_summary = ((p_tokens + c_tokens) / 1000) * model_prices_gbp.get(ai_model_to_use_for_summary, 0.0)
                    doc_result_row["cost_gbp_summary"] = round(cost_this_summary, 5); total_ai_summarization_cost_gbp += cost_this_summary
                elif doc_result_row["text_extraction_status"] != "Success": doc_result_row["ai_summary_status"] = "Skipped (Text Extraction Failed/Short)"
                else: doc_result_row["ai_summary_status"] = "Skipped (Text Too Short for Summary)"
            except Exception as e_doc_proc:
                logger.error(f"Batch Run {run_id} ({company_no}): Error processing doc TX_ID {doc_result_row['transaction_id']}: {e_doc_proc}", exc_info=True)
                doc_result_row["processing_error"] = str(e_doc_proc)
                doc_result_row["text_extraction_status"] = doc_result_row["text_extraction_status"] if doc_result_row["text_extraction_status"] != "Pending" else "Failed"
                doc_result_row["ai_summary_status"] = doc_result_row["ai_summary_status"] if doc_result_row["ai_summary_status"] != "Pending" else "Skipped (Processing Error)"
            all_processed_docs_data.append(doc_result_row)

    output_csv_path: Optional[Path] = None # Initialize
    if all_processed_docs_data:
        try:
            import pandas as pd
            df_results = pd.DataFrame(all_processed_docs_data)
            csv_columns = ["company_number", "company_name", "document_type", "document_date", "document_description", "transaction_id", "api_category",
                           "text_extraction_status", "extracted_text_length", "ocr_pages_processed", "ai_summary_status", "summary", 
                           "ai_model_used_for_summary", "prompt_tokens_summary", "completion_tokens_summary", "cost_gbp_summary",
                           "local_path_raw_doc", "processing_error"]
            for col in csv_columns:
                if col not in df_results.columns: df_results[col] = None 
            output_csv_path = batch_output_dir / f"ch_analysis_digest_{run_id}.csv"
            df_results[csv_columns].to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Batch Run {run_id}: Digest CSV saved to {output_csv_path}")
        except ImportError: logger.error(f"Batch Run {run_id}: Pandas library not found. Cannot create CSV digest.")
        except Exception as e_csv: logger.error(f"Batch Run {run_id}: Error writing CSV digest: {e_csv}", exc_info=True)

    output_json_path = batch_output_dir / f"ch_analysis_detailed_data_{run_id}.json"
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f_json_out: json.dump(all_processed_docs_data, f_json_out, indent=2)
        logger.info(f"Batch Run {run_id}: Detailed JSON data saved to {output_json_path}")
    except Exception as e_json_out: logger.error(f"Batch Run {run_id}: Error writing detailed JSON: {e_json_out}", exc_info=True)

    output_docx_path: Optional[Path] = None # Initialize
    try:
        from docx import Document as DocxDocument # Ensure this is imported if used
        report_doc = DocxDocument()
        report_doc.add_heading(f"Companies House Analysis Report - Run ID: {run_id}", level=0)
        # ... (rest of DOCX generation) ...
        output_docx_path = batch_output_dir / f"ch_analysis_report_{run_id}.docx" # Assign here
        report_doc.save(output_docx_path)
        logger.info(f"Batch Run {run_id}: DOCX report saved to {output_docx_path}")
    except ImportError: logger.error(f"Batch Run {run_id}: python-docx library not found. Cannot create DOCX.")
    except Exception as e_docx: logger.error(f"Batch Run {run_id}: Error writing DOCX report: {e_docx}", exc_info=True)

    batch_metrics = {
        "run_id": run_id, "run_timestamp": run_timestamp.isoformat(), "total_companies_processed": len(company_numbers_list), 
        "total_documents_analyzed": total_docs_analyzed_count, "total_textract_pages_processed": total_textract_pages_processed_count,
        "total_ai_summarization_cost_gbp": round(total_ai_summarization_cost_gbp, 5), "total_prompt_tokens": total_prompt_tokens_all_docs,
        "total_completion_tokens": total_completion_tokens_all_docs, "output_csv_path": str(output_csv_path) if output_csv_path else None,
        "output_json_path": str(output_json_path) if output_json_path else None, "output_docx_path": str(output_docx_path) if output_docx_path else None,
        "batch_output_directory": str(batch_output_dir)
    }
    if use_textract_ocr and get_textract_cost_estimation and total_textract_pages_processed_count > 0:
        num_pdfs_ocrd_approx_metrics = sum(1 for d in all_processed_docs_data if d.get("ocr_pages_processed", 0) > 0)
        if get_textract_cost_estimation: # Check if function is not None
             batch_metrics["aws_textract_costs"] = get_textract_cost_estimation(total_textract_pages_processed_count, num_pdfs_ocrd_approx_metrics)

    logger.info(f"Batch Run {run_id}: Analysis complete. Metrics: {batch_metrics}")
    return output_csv_path, batch_metrics
