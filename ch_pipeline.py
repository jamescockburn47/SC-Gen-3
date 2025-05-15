# ch_pipeline.py

# When building docs_to_process_by_company in app.py, do NOT expect 'raw_metadata' or 'display_name' keys.
# Instead, always use the document dict itself for both display and processing.
# For display, generate display_name dynamically in the UI, e.g.:
# display_name = f"{doc_detail.get('description', 'N/A')} | {doc_detail.get('date', 'N/A')} | {doc_detail.get('type', 'N/A')}"
# When collecting selected documents for processing, append the entire doc_detail dict, not doc_detail

from __future__ import annotations
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypeAlias, Any
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
import xmltodict
import xml.etree.ElementTree as ET
import openai
import google.generativeai as genai


import requests
from requests.adapters import Retry, HTTPAdapter

# CH API Interactions
from ch_api_utils import (
    get_ch_documents_metadata,
    _fetch_document_content_from_ch,
    get_company_profile,          # ← only if referenced later in the file
)

from config import (
    APPLICATION_SCRATCH_DIR,
    CH_API_MAX_RETRY,
    CH_API_RETRY_BACKOFF_FACTOR,
    CH_API_RETRY_STATUS_FORCELIST,
    CH_API_TIMEOUT,
)

# ---------------------------------------------------------------------------- #
# Optional Text Extraction Import (local OCR handler)
# ---------------------------------------------------------------------------- #
try:
    from text_extraction_utils import extract_text_from_document
    TEXT_EXTRACTOR_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("text_extraction_utils.py found and imported successfully.")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("text_extraction_utils.py not found. Text extraction will fail.")
    def extract_text_from_document(*args, **kwargs) -> Tuple[str, int, Optional[str]]:
        return "Error: text_extraction_utils.py not found.", 0, "text_extraction_utils.py is missing."

# AI Summarization
from ai_utils import gpt_summarise_ch_docs, gemini_summarise_ch_docs

# Optional AWS Textract Import
try:
    from aws_textract_utils import perform_textract_ocr, get_textract_cost_estimation, _initialize_aws_clients as initialize_textract_aws_clients
    TEXTRACT_AVAILABLE = True
    logger.info("aws_textract_utils.py found and imported successfully.")
except ImportError:
    perform_textract_ocr = None
    get_textract_cost_estimation = None
    initialize_textract_aws_clients = None
    TEXTRACT_AVAILABLE = False
    logger.warning("aws_textract_utils.py not found. Textract-related functions will be disabled.")

# ---------------------------------------------------------------------------- #
# Global Configurations and Constants
# ---------------------------------------------------------------------------- #
CH_API_BASE_URL = "https://api.company-information.service.gov.uk"
USER_AGENT = "StrategicComplianceGen3/1.0 (contact: ops@strategiccompliance.ai)"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}
SCRATCH_DIR = Path(APPLICATION_SCRATCH_DIR).expanduser()
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
CH_API_RETRY_STATUS_FORLIST = [429, 500, 502, 503, 504]
# ---------------------------------------------------------------------------- #
# Helper Functions (Disk, Network, Parsing)
# ---------------------------------------------------------------------------- #

def _save_raw_document_content(
    doc_content: Union[bytes, str, Dict],
    doc_type_str: str,
    company_no: str,
    ch_doc_code: str,  # filing identifier
    doc_year: int,  # e.g. 2024
    scratch_dir: Path
) -> Optional[Path]:
    """
    Persist the fetched document to the scratch directory and return the path,
    or None on failure.
    """
    file_extension_map = {"pdf": "pdf", "xhtml": "xhtml", "json": "json"}
    file_extension = file_extension_map.get(doc_type_str.lower(), "dat")

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    doc_filename_prefix = f"{company_no}_{ch_doc_code}_{doc_year}_{timestamp_str}"
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
        return  # negative keep_days acts as "never clean"

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

# NEW FUNCTION

def get_relevant_filings_metadata(
    ch_api: CompanyHouseAPI,
    company_number: str,
    years_back: int = 5,
    filters: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> List[Dict[str, Any]]:
    """Return a list of filings metadata dictionaries for the given company that match the filters."""
    min_year = datetime.datetime.now().year - years_back
    company_filings = ch_api.get_company_filings(company_number)

    relevant_filings = []
    for filing in company_filings:
        filing_date = datetime.datetime.strptime(filing["date"], "%Y-%m-%d")
        if filing_date.year >= min_year:
            if filters:
                match = True
                for key, value in filters.items():
                    if key in filing and filing[key] not in value:
                        match = False
                        break
                if match:
                    relevant_filings.append(filing)
            else:
                relevant_filings.append(filing)

    return relevant_filings

# ---------------------------------------------------------------------------- #
# Core Pipeline Classes / Functions
# ---------------------------------------------------------------------------- #

class CompanyHouseDocumentPipeline:
    """Pipeline that downloads, processes, OCRs, and summarises Companies House filings."""

    def __init__(
        self,
        company_number: str,
        ch_api_key: Optional[str] = None,
        text_extractor_available: bool = TEXT_EXTRACTOR_AVAILABLE,
        textract_available: bool = TEXTRACT_AVAILABLE,
        scratch_dir: Path = SCRATCH_DIR,
        keep_days_in_scratch: int = 14,
    ) -> None:
        self.company_number = company_number
        self.text_extractor_available = text_extractor_available
        self.textract_available = textract_available
        self.scratch_dir = scratch_dir
        self.ch_api = CompanyHouseAPI(
            api_key=ch_api_key,
            max_retry=CH_API_MAX_RETRY,
            backoff_factor=CH_API_RETRY_BACKOFF_FACTOR,
            status_forcelist=CH_API_RETRY_STATUS_FORLIST,
            timeout=CH_API_TIMEOUT,
            user_agent=USER_AGENT,
        )
        self.keep_days_in_scratch = keep_days_in_scratch

        # Prepare scratch directory for this company
        self.company_scratch_dir = self.scratch_dir / self.company_number
        self.company_scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialised CompanyHouseDocumentPipeline for %s | text_extractor=%s | textract=%s",
            self.company_number,
            self.text_extractor_available,
            self.textract_available,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run(self, years_back: int = 5) -> Dict[str, Any]:
        """Main entry point for the pipeline. Returns a dict with results and summary."""
        _cleanup_scratch_directory(self.company_scratch_dir, self.keep_days_in_scratch)

        try:
            # 1. Get recent filings metadata
            filings_metadata = get_relevant_filings_metadata(
                self.ch_api, self.company_number, years_back=years_back
            )
            logger.info(
                "Fetched %d recent filings for %s within last %d years",
                len(filings_metadata),
                self.company_number,
                years_back,
            )

            # 2. Download filings
            downloaded_docs = self._download_filings(filings_metadata)

            # 3. Extract text (OCR if needed)
            processed_docs = self._process_documents(downloaded_docs)

            # 4. Summarise documents
            summary = self._summarise_documents(processed_docs)

            return {
                "company_number": self.company_number,
                "filings_downloaded": len(downloaded_docs),
                "filings_processed": len(processed_docs),
                "summary": summary,
            }

        except Exception as e_run:
            logger.exception("Pipeline run failed for %s: %s", self.company_number, e_run)
            raise

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _download_filings(self, filings_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        downloaded = []
        for filing in filings_metadata:
            try:
                filing_type = filing.get("type", "?")
                filing_year = datetime.datetime.strptime(filing["date"], "%Y-%m-%d").year
                doc_content, content_type = self.ch_api.download_document(
                    filing["company_number"], filing["filing_id"], return_content_type=True
                )
                local_path = _save_raw_document_content(
                    doc_content,
                    content_type,
                    filing["company_number"],
                    filing["filing_id"],
                    filing_year,
                    self.company_scratch_dir,
                )
                if local_path:
                    filing["local_path"] = local_path
                    downloaded.append(filing)
                    logger.info(
                        "Downloaded %s document %s for %s to %s",
                        filing_type,
                        filing["filing_id"],
                        filing["company_number"],
                        local_path.name,
                    )
            except CompanyHouseAPINotFoundError:
                logger.warning("Document %s not found for %s", filing["filing_id"], filing["company_number"])
            except (CompanyHouseAPIRateLimitError, CompanyHouseAPIAuthError) as e_ch:
                logger.error("Company House API error for %s: %s", filing["filing_id"], e_ch)
            except Exception as e_dl:
                logger.exception("Unexpected error downloading %s: %s", filing["filing_id"], e_dl)
        return downloaded

    def _process_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_docs = []
        for doc in docs:
            local_path: Path = doc.get("local_path")  # type: ignore[assignment]
            if not local_path or not local_path.exists():
                continue
            try:
                if local_path.suffix.lower() == ".pdf" and self.text_extractor_available:
                    text, num_pages, extractor_used = extract_text_from_document(local_path)
                elif local_path.suffix.lower() == ".pdf" and self.textract_available:
                    text, num_pages, extractor_used = perform_textract_ocr(local_path)
                else:
                    text = local_path.read_text(encoding="utf-8", errors="ignore")
                    num_pages = text.count("\f") or text.count("\n\f")  # crude heuristic
                    extractor_used = "text_from_file"

                doc["extracted_text"] = text
                doc["num_pages"] = num_pages
                doc["extractor_used"] = extractor_used
                processed_docs.append(doc)
                logger.info(
                    "Processed %s (%d pages) for %s via %s",
                    local_path.name,
                    num_pages,
                    doc.get("company_number"),
                    extractor_used,
                )
            except Exception as e_proc:
                logger.exception("Failed to process %s: %s", local_path.name, e_proc)
        return processed_docs

    def _summarise_documents(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        docs_by_year: Dict[int, List[str]] = defaultdict(list)
        for doc in docs:
            filing_year = datetime.datetime.strptime(doc["date"], "%Y-%m-%d").year
            docs_by_year[filing_year].append(doc.get("extracted_text", ""))

        annual_summaries = {}
        for year, texts in docs_by_year.items():
            combined_text = "\n\n".join(texts)[:60000]  # guard token limit
            try:
                # Prefer Gemini if available; fallback to GPT
                if os.getenv("GEMINI_API_KEY"):
                    summary = gemini_summarise_ch_docs(combined_text)
                else:
                    summary = gpt_summarise_ch_docs(combined_text)
                annual_summaries[year] = summary
                logger.info("Summarised %d docs for %s (%d)", len(texts), self.company_number, year)
            except Exception as e_sum:
                logger.warning("Failed to summarise %d docs for %s (%d): %s", len(texts), self.company_number, year, e_sum)
        return annual_summaries

# End of ch_pipeline.py
