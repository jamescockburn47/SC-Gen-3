# text_extraction_utils.py

import logging
import re
import json
from io import BytesIO
from typing import Tuple, Optional, Callable, Dict, Union

# Lightweight PDF parsing via pdfminer.six
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdfparser import PDFSyntaxError

# PyPDF2 remains unused in this module
from bs4 import BeautifulSoup

from config import MIN_MEANINGFUL_TEXT_LEN, logger

# Type alias for the OCR handler function
# Takes (pdf_bytes, company_no_for_logging)
# Returns (extracted_text, pages_processed, error_message_or_none)
OCRHandlerType = Callable[[bytes, str], Tuple[str, int, Optional[str]]]

def _reconstruct_text_from_ch_json(doc_content: Dict, company_no_for_logging: str) -> str:
    """
    Basic reconstruction of text from Companies House JSON content.
    This is a heuristic-based approach for iXBRL JSON.
    """
    logger.info(f"{company_no_for_logging}: Reconstructing text from CH JSON content.")
    text_parts = []

    def extract_values_from_node(node: Union[Dict, list], current_prefix: str = ""):
        if isinstance(node, dict):
            for key, value in node.items():
                # Heuristic: try to get meaningful keys, avoid purely technical ones.
                # Focus on longer string values that are not URLs or simple references.
                if isinstance(value, str) and len(value) > 20 and not value.startswith("http"):
                    if "contextRef" not in key and "unitRef" not in key and "dimension" not in key:
                        # Attempt to make key more readable for context
                        readable_key = key.replace("Value", "").replace("TextBlock", "")
                        # Remove common iXBRL prefixes for brevity if they exist
                        prefixes_to_remove = ["uk-gaap:", "uk-bus:", "core:", "ref:", "nonNumeric:", "num:", "link:", "xbrli:"]
                        for prefix in prefixes_to_remove:
                            if readable_key.startswith(prefix):
                                readable_key = readable_key[len(prefix):]
                        
                        # CamelCase/PascalCase to Title Case for better readability
                        readable_key = re.sub(r'([a-z])([A-Z])', r'\1 \2', readable_key).replace("_", " ").title()
                        
                        # Avoid adding overly generic or purely structural keys if value is simple
                        if not (readable_key.lower() in ["value", "text"] and len(value.split()) < 5):
                             text_parts.append(f"{current_prefix}{readable_key.strip()}: {value.strip()}")
                        else:
                            text_parts.append(f"{current_prefix}{value.strip()}") # Just add the value

                elif isinstance(value, (dict, list)):
                    extract_values_from_node(value, current_prefix) # Recursive call

        elif isinstance(node, list):
            for item in node:
                extract_values_from_node(item, current_prefix) # Process items in a list

    try:
        # Start extraction from known common top-level keys or the whole document
        # Common iXBRL data might be under 'facts', 'instance', or directly at root
        if 'facts' in doc_content and isinstance(doc_content['facts'], dict):
            for fact_group in doc_content['facts'].values(): # facts are often grouped by concept
                extract_values_from_node(fact_group)
        elif 'instance' in doc_content and isinstance(doc_content['instance'], dict) :
             extract_values_from_node(doc_content['instance'])
        else:
            extract_values_from_node(doc_content) # Process the whole JSON if no obvious entry points
            
        if not text_parts:
            extracted_text = "JSON content was available but yielded no reconstructable text with current generic logic."
            logger.warning(f"{company_no_for_logging}: CH JSON processing (generic) yielded no text. Full JSON sample: {json.dumps(doc_content)[:500]}")
        else:
            extracted_text = "\n".join(text_parts)
            # Further clean up: reduce excessive blank lines that might result from structure
            extracted_text = re.sub(r'\n\s*\n', '\n', extracted_text).strip()
            logger.info(f"{company_no_for_logging}: Textual representation generated from CH JSON ({len(extracted_text)} chars).")
            
    except Exception as e:
        logger.error(f"{company_no_for_logging}: Failed to process CH JSON content for text reconstruction: {e}", exc_info=True)
        extracted_text = f"Error: Could not process CH JSON content for text. Details: {str(e)}"
    return extracted_text

MIN_CHARS_FOR_CH_XHTML_SUMMARY = 500 # Increased threshold for CH XHTML
MIN_CHARS_FOR_GENERAL_XHTML_SUMMARY = 200 # Default for other XHTML

def _extract_text_from_xhtml(xhtml_content: str, company_no_for_logging: str, is_ch_document: bool = False) -> str:
    """Extracts text from XHTML content using BeautifulSoup."""
    logger.info(f"{company_no_for_logging}: Extracting text from XHTML content (CH Doc: {is_ch_document}).")
    try:
        if not isinstance(xhtml_content, str):
            logger.warning(f"{company_no_for_logging}: XHTML content not string (type: {type(xhtml_content)}), converting.")
            xhtml_content = str(xhtml_content)
        
        soup = BeautifulSoup(xhtml_content, "html.parser") 
        
        # For CH documents, be less aggressive with tag stripping initially to preserve table-like data if possible.
        # For general XHTML, the existing more aggressive stripping is fine.
        if not is_ch_document:
            tags_to_remove = ["script", "style", "head", "meta", "link", "title", "header", "footer", "nav", "aside", "form", "button", "input", "noscript"]
            for tag_name in tags_to_remove:
                for tag_element in soup.find_all(tag_name):
                    tag_element.decompose()
        else: # For CH documents, more conservative removal - keep tables, divs, spans more readily
            tags_to_remove_ch = ["script", "style", "head", "meta", "link", "title", "header", "footer", "nav", "aside", "form", "button", "input", "noscript"]
            for tag_name in tags_to_remove_ch:
                for tag_element in soup.find_all(tag_name):
                    tag_element.decompose()
            # Consider if specific ix:nonNumeric, ix:nonFraction elements need special handling for text extraction if they are being missed.
            # For now, relying on get_text() should pick up their textual content.

        body_tag = soup.find("body")
        if body_tag:
            # For CH docs, try to preserve more structure from tables by replacing TDs/THs with tabs/newlines
            if is_ch_document:
                for table in body_tag.find_all("table"):
                    for row in table.find_all("tr"):
                        cell_texts = []
                        for cell in row.find_all(["td", "th"]):
                            cell_texts.append(cell.get_text(separator=" ", strip=True))
                        row.replace_with("\t".join(cell_texts) + "\n") # Replace row with tab-separated values
            text_content = body_tag.get_text(separator=" ", strip=True)
        else:
            text_content = soup.get_text(separator=" ", strip=True)

        text_content = re.sub(r'\s\s+', ' ', text_content).strip()
        text_content = re.sub(r'(\n\s*){2,}', '\n\n', text_content).strip()

        min_len_threshold = MIN_CHARS_FOR_CH_XHTML_SUMMARY if is_ch_document else MIN_CHARS_FOR_GENERAL_XHTML_SUMMARY

        if len(text_content) < min_len_threshold:
            logger.warning(f"{company_no_for_logging}: XHTML parsing (CH Doc: {is_ch_document}) yielded short text ({len(text_content)} chars, threshold {min_len_threshold}). Preview: '{text_content[:200]}'") # Increased preview
        else:
            logger.info(f"{company_no_for_logging}: Text extracted from XHTML (CH Doc: {is_ch_document}, {len(text_content)} chars).")
        return text_content
    except Exception as e_parse_xhtml:
        logger.error(f"{company_no_for_logging}: Failed to parse XHTML (CH Doc: {is_ch_document}): {e_parse_xhtml}", exc_info=True)
        return f"Error: Could not parse XHTML content. Details: {str(e_parse_xhtml)}"


def extract_text_from_document(
    doc_content_input: Union[bytes, str, Dict],
    content_type_input: str,
    company_no_for_logging: str = "N/A_DocExtract",
    ocr_handler: Optional[OCRHandlerType] = None
) -> Tuple[str, int, Optional[str]]:
    """
    Extracts text from various document content types.
    For PDFs, relies *only* on the ocr_handler if provided.
    Standard library PDF parsing (PyPDF2, pdfminer) is removed for CH pipeline.

    Args:
        doc_content_input: The document content (bytes for PDF, str for XHTML, dict for JSON).
        content_type_input: The type of content ("pdf", "xhtml", "json").
        company_no_for_logging: Identifier for logging.
        ocr_handler: Optional function to call for PDF OCR.
                     Expected signature: ocr_handler(pdf_bytes, log_id) -> (text, pages_processed, error_msg_or_None)

    Returns:
        A tuple: (extracted_text_str, pages_processed_by_ocr_int, error_message_str_or_None).
        pages_processed_by_ocr_int is 0 if OCR was not used or failed.
        error_message_str_or_None contains an error message if a significant failure occurred.
    """
    extracted_text = ""
    pages_ocrd = 0
    error_msg = None

    if content_type_input == "json":
        if isinstance(doc_content_input, dict):
            extracted_text = _reconstruct_text_from_ch_json(doc_content_input, company_no_for_logging)
        else:
            error_msg = f"Expected dict for JSON content, got {type(doc_content_input)}. Cannot process."
            logger.error(f"{company_no_for_logging}: {error_msg}")
            extracted_text = f"Error: {error_msg}" # Return error in text if this happens
        # No pages processed for JSON

    elif content_type_input == "xhtml" or content_type_input == "xml": # Added "xml"
        if isinstance(doc_content_input, str):
            # Pass is_ch_document=True if it's a CH document, assuming a naming convention or metadata flag
            # For this example, we'll assume company_no_for_logging containing typical CH number implies it's a CH doc.
            # A more robust way would be to pass a specific flag from ch_pipeline.py
            is_ch_doc_heuristic = bool(re.match(r"^\d{8}$|^[A-Z]{2}\d{6}$", company_no_for_logging))
            # For XML, treat it similarly to XHTML for text extraction using BeautifulSoup
            # as iXBRL is XHTML-based.
            extracted_text = _extract_text_from_xhtml(doc_content_input, company_no_for_logging, is_ch_document=is_ch_doc_heuristic)
            logger.info(f"{company_no_for_logging}: Extracted text from {content_type_input.upper()} using XHTML/HTML parsing logic.")
        else:
            error_msg = f"Expected str for {content_type_input.upper()} content, got {type(doc_content_input)}. Cannot process."
            logger.error(f"{company_no_for_logging}: {error_msg}")
            extracted_text = f"Error: {error_msg}"
        # No pages processed for XHTML/XML

    elif content_type_input == "pdf":
        if not isinstance(doc_content_input, bytes):
            error_msg = f"Expected bytes for PDF content, got {type(doc_content_input)}. Cannot process."
            logger.error(f"{company_no_for_logging}: {error_msg}")
            extracted_text = f"Error: {error_msg}"
            return extracted_text, 0, error_msg  # Early exit

        pdf_text = ""
        parse_error = None
        try:
            with BytesIO(doc_content_input) as pdf_buf:
                pdf_text = pdfminer_extract(pdf_buf) or ""
            pdf_text = re.sub(r"\s+", " ", pdf_text).strip()
            if pdf_text:
                logger.info(f"{company_no_for_logging}: Parsed PDF text with pdfminer ({len(pdf_text)} chars).")
        except Exception as e_pdf:
            parse_error = str(e_pdf)
            logger.warning(f"{company_no_for_logging}: pdfminer failed to parse PDF: {e_pdf}")
            pdf_text = ""

        if pdf_text and len(pdf_text) >= MIN_MEANINGFUL_TEXT_LEN:
            extracted_text = pdf_text
        else:
            if pdf_text:
                logger.info(f"{company_no_for_logging}: Parsed PDF text length {len(pdf_text)} below threshold {MIN_MEANINGFUL_TEXT_LEN}.")
            if ocr_handler:
                logger.info(f"{company_no_for_logging}: Falling back to OCR for PDF.")
                ocr_text, pages_ocrd_by_handler, ocr_err = ocr_handler(doc_content_input, company_no_for_logging)
                pages_ocrd = pages_ocrd_by_handler
                if ocr_err:
                    error_msg = f"OCR failed: {ocr_err}"
                    logger.error(f"{company_no_for_logging}: {error_msg}")
                    extracted_text = ocr_text if ocr_text else (pdf_text if pdf_text else f"Error: {error_msg}")
                elif not ocr_text or len(ocr_text.strip()) < MIN_MEANINGFUL_TEXT_LEN:
                    short_text_msg = f"OCR yielded short or no text ({len(ocr_text.strip()) if ocr_text else 0} chars)."
                    logger.warning(f"{company_no_for_logging}: {short_text_msg} Preview: '{ocr_text[:100] if ocr_text else ''}...'")
                    extracted_text = ocr_text if ocr_text else pdf_text
                else:
                    extracted_text = ocr_text
                    logger.info(f"{company_no_for_logging}: Text extracted from PDF via OCR ({len(extracted_text)} chars, {pages_ocrd} pages).")
            else:
                extracted_text = pdf_text
                if parse_error:
                    error_msg = f"PDF parse error: {parse_error}"
                elif not pdf_text:
                    error_msg = "PDF parsing yielded no text and OCR not available."
                else:
                    error_msg = f"PDF parsing produced short text ({len(pdf_text)} chars) and no OCR handler provided."
                logger.warning(f"{company_no_for_logging}: {error_msg}")

    else:
        error_msg = f"Unknown content_type '{content_type_input}' for text extraction."
        logger.error(f"{company_no_for_logging}: {error_msg}")
        extracted_text = f"Error: {error_msg}"

    # Final check on extracted_text before returning
    if "Error:" in extracted_text and not error_msg : # If text contains "Error:" but error_msg is not set
        error_msg = extracted_text # Promote the text's error message

    if not extracted_text.strip() and not error_msg:
        # If text is empty/whitespace but no explicit error was flagged,
        # it means extraction happened but found nothing. This isn't an "error" in the process.
        logger.info(f"{company_no_for_logging}: Text extraction resulted in empty content for type '{content_type_input}'.")
        
    return extracted_text, pages_ocrd, error_msg