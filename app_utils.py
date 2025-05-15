# app_utils.py

import json
import re
import logging # Keep for local logger if needed, but prefer central
from typing import Tuple, Optional, List, Dict, Union, Callable # Added Union
import io

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader # Keep for direct PDF text extraction attempt
from docx import Document

# Import from config
from config import (
    get_openai_client, 
    get_ch_session, 
    PROTO_TEXT_FALLBACK, # Fallback protocol text
    LOADED_PROTO_TEXT, # Dynamically loaded protocol text from app.py
    logger # Use central logger from config
)

# AWS SDK (boto3) for Textract, only if needed directly here.
# It's better if aws_textract_utils.perform_textract_ocr is called via a handler.
# However, the original had a direct Textract call. Let's refine this.
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    NoCredentialsError = None
    PartialCredentialsError = None
    BOTO3_AVAILABLE = False
    logger.info("boto3 library not found. Direct AWS calls in app_utils will be skipped if any.")

from aws_textract_utils import perform_textract_ocr as aws_perform_textract_ocr # Renamed for clarity
from aws_textract_utils import TEXTRACT_AVAILABLE as AWS_TEXTRACT_MODULE_AVAILABLE


def _word_cap(word_count: int) -> int:
    """Determines a reasonable word cap for summaries based on input word count."""
    if word_count <= 2000:
        return max(150, int(word_count * 0.15))
    elif word_count <= 10000:
        return 300
    else:
        return min(500, int(word_count * 0.05))

def summarise_with_title(
    text: str,
    model_name_selected: str, # This implies the main app's selected model for general tasks
    topic: str, 
    # protocol_text: str # This was passed, but now we use LOADED_PROTO_TEXT from config
) -> Tuple[str, str]:
    """
    Generates a short title and summary for UI display of uploaded documents.
    Uses a cost-effective OpenAI model for this task.
    Uses LOADED_PROTO_TEXT from config.
    """
    if not text or not text.strip():
        return "Empty Content", "No text was provided for summarization."

    word_count = len(text.split())
    summary_word_cap = _word_cap(word_count)
    # Limit text to avoid excessive token usage for a quick title/summary
    text_to_summarise = text[:15000] # Approx 3.7k tokens, adjust if needed
    max_tokens_for_response = int(summary_word_cap * 2.0) # Allow more tokens for JSON structure and content

    # Use a specific, cost-effective model for this utility, overriding model_name_selected
    openai_model_for_this_task = "gpt-4o-mini" 
    openai_client = get_openai_client()

    if not openai_client:
        logger.error(f"OpenAI client not available for summarise_with_title (topic: {topic}).")
        return "Summarization Error", "OpenAI client not configured."

    # Using LOADED_PROTO_TEXT from config, which app.py updates
    current_protocol_text = LOADED_PROTO_TEXT 

    prompt = (
        f"Return ONLY valid JSON in the format {{\"title\": \"<A concise title of less than 12 words>\", "
        f"\"summary\": \"<A summary of approximately {summary_word_cap} words, capturing the essence of the text>\"}}.\n\n"
        f"Analyze the following text:\n---\n{text_to_summarise}\n---"
    )
    raw_response_content = ""
    title = "Error in Summarization"
    summary = "Could not generate summary due to an issue."

    try:
        response = openai_client.chat.completions.create(
            model=openai_model_for_this_task,
            temperature=0.2,
            max_tokens=max_tokens_for_response,
            messages=[
                {"role": "system", "content": current_protocol_text}, 
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} # Request JSON object response
        )
        raw_response_content = response.choices[0].message.content.strip()
        data = json.loads(raw_response_content)
        title = str(data.get("title", "Title Missing"))
        summary = str(data.get("summary", "Summary Missing"))
        logger.info(f"Successfully generated title/summary for topic '{topic}' using {openai_model_for_this_task}.")
    except json.JSONDecodeError as e_json:
        logger.error(f"JSONDecodeError in summarise_with_title (topic: {topic}, model: {openai_model_for_this_task}): {e_json}. Raw response: {raw_response_content[:200]}...")
        title = "Summarization Format Error"
        summary = f"Failed to parse AI response as JSON. Preview: {raw_response_content[:150]}..."
    except Exception as e:
        logger.error(f"Exception in summarise_with_title (topic: {topic}, model: {openai_model_for_this_task}): {e}. Raw response: {raw_response_content[:200]}...", exc_info=True)
        # Try to make a title from the beginning of the raw response if it's not JSON
        first_part = raw_response_content.split(".")[0][:75].strip() if raw_response_content and isinstance(raw_response_content, str) else ""
        title = first_part if first_part else f"Summarization Failed ({type(e).__name__})"
        summary = raw_response_content if raw_response_content and isinstance(raw_response_content, str) else "No response content or malformed."
    return title, summary


def fetch_url_content(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches and extracts text content from a URL. Returns (text, error_message)."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Use the CH session for potentially better rate limiting / consistent client if it's a CH domain,
        # otherwise a new requests.get. For generic URLs, new requests.get is fine.
        # For simplicity here, using direct requests.
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch URL '{url}': {e}")
        return None, f"Fetch URL Error: {e.__class__.__name__} for {url}"

    try:
        # Attempt to decode using UTF-8, then fall back to response.apparent_encoding or ISO-8859-1
        try:
            html_content = response.content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                html_content = response.content.decode(response.apparent_encoding)
            except (UnicodeDecodeError, TypeError): # TypeError if apparent_encoding is None
                html_content = response.content.decode('ISO-8859-1', errors='replace')
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove common non-content tags
        for s_tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "input", "meta", "link"]):
            s_tag.decompose()

        # Try to find main content areas
        main_content_tags = soup.find_all(['main', 'article', 'div.content', 'div.main-content', 'div.post-body', 'section.content-section'])
        content_text = ""
        if main_content_tags:
            content_text = " ".join(tag.get_text(" ", strip=True) for tag in main_content_tags)
        
        # Fallback if specific main content tags are not found or yield little text
        if not content_text.strip() or len(content_text.split()) < 50 : # Increased threshold
            body_tag = soup.find("body")
            if body_tag:
                content_text = body_tag.get_text(" ", strip=True)
            else: # If no body tag, get all text (should be rare for valid HTML)
                content_text = soup.get_text(" ", strip=True)
        
        # Clean up whitespace
        content_text = re.sub(r'\s\s+', ' ', content_text).strip()
        content_text = re.sub(r'(\n\s*){3,}', '\n\n', content_text).strip() # Reduce multiple newlines

        if not content_text.strip():
            logger.info(f"No significant text extracted from URL '{url}' after parsing.")
            return None, f"No text content found at {url} after parsing."
        
        logger.info(f"Successfully extracted text from URL '{url}' ({len(content_text)} chars).")
        return content_text, None
    except Exception as e:
        logger.error(f"Failed to process content from URL '{url}': {e}", exc_info=True)
        return None, f"Process URL Error: {str(e)} for {url}"


def find_company_number(query: str, ch_api_key: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """
    Searches Companies House for a company number.
    Returns (company_number, error_message, first_match_details).
    Uses ch_api_key if provided, otherwise relies on config.CH_API_KEY.
    """
    # Use get_ch_session, passing the specific API key if provided.
    # This ensures the session is correctly authenticated.
    ch_session = get_ch_session(api_key_override=ch_api_key) 
    
    if not ch_api_key and not getattr(ch_session.auth, 'username', None): # Check if session has auth if key not passed
        # This condition might be tricky if CH_API_KEY was set in env and session was pre-auth'd
        # A clearer check: if not (ch_api_key or config.CH_API_KEY):
        if not ch_session.auth: # Simpler: if session has no auth tuple
             return None, "Companies House API Key is missing or not configured for the session.", None

    if not query or not query.strip():
        return None, "Please enter a company name or number to search.", None

    query_cleaned = query.strip().upper()
    
    # Regex for typical UK company numbers (allows for optional prefix, 6-8 digits, or alphanumeric like Scottish)
    company_no_match = re.fullmatch(r"((SC|NI|OC|SO|R[0-9])?([0-9]{6,8})|[A-Z0-9]{8})", query_cleaned.replace(" ", ""))
    
    if company_no_match:
        potential_no = query_cleaned.replace(" ", "")
        # If it's purely numeric and short, zfill it (common for older English/Welsh numbers)
        if potential_no.isdigit() and len(potential_no) < 8:
            formatted_no = potential_no.zfill(8)
            # Check if this formatted number is a valid pattern (e.g. 8 digits)
            if re.fullmatch(r"[0-9]{8}", formatted_no):
                 logger.info(f"Using provided/formatted company number: {formatted_no}")
                 # Return a minimal dict as we don't have full company details yet
                 return formatted_no, None, {"company_number": formatted_no, "title": "Input as Number (Formatted)"}

        # If it matches a more complex pattern (like SC, NI, or already 8 chars)
        if re.fullmatch(r"[A-Z0-9]{8}|(SC|NI|OC|SO|R[0-9]?)[0-9]{6}", potential_no): # Adjusted regex slightly
            logger.info(f"Using provided company number: {potential_no}")
            return potential_no, None, {"company_number": potential_no, "title": "Input as Number/Code"}

    logger.info(f"Searching Companies House for name/number: '{query}' (Cleaned: '{query_cleaned}')")
    search_url = "https://api.company-information.service.gov.uk/search/companies"
    params = {'q': query_cleaned, 'items_per_page': 5} # Fetch a few items

    try:
        response = ch_session.get(search_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Companies House search API error for '{query}': {e}", exc_info=True)
        return None, f"Companies House Search Error: {e}", None
    except json.JSONDecodeError as e_json:
        logger.error(f"Failed to decode JSON from CH search for '{query}': {e_json}. Response text: {getattr(response, 'text', 'N/A')[:200]}", exc_info=True)
        return None, "Companies House Search Error: Could not parse response.", None

    if not items:
        logger.warning(f"No company found for query '{query}'.")
        return None, f"No company found for '{query}'.", None

    # Prefer exact match on company number if query was a number
    if query_cleaned.isalnum(): # Check if the cleaned query could be a company number
        for item in items:
            if item.get("company_number") == query_cleaned:
                logger.info(f"Exact company number match found via search: {item.get('title')} ({item.get('company_number')})")
                return item.get("company_number"), None, item

    # Fallback to the first match if no exact number match or query was a name
    first_match = items[0]
    company_number_found = first_match.get("company_number")
    company_name_found = first_match.get("title", "N/A")
    
    if company_number_found:
        logger.info(f"Found via search (first result): {company_name_found} ({company_number_found}). Using this number.")
        return company_number_found, None, first_match
    else:
        logger.warning(f"First match for '{query}' ({company_name_found}) has no company number in API response.")
        return None, "First match found but no company number available in the result.", first_match


def extract_text_from_uploaded_file(
    file_obj: io.BytesIO, 
    file_name: str,
    # Adding an optional OCR handler for consistency, though Textract is directly used for PDF here
    ocr_handler: Optional[Callable[[bytes, str], Tuple[str, int, Optional[str]]]] = None 
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts text from an uploaded file object (PDF, DOCX, TXT).
    For PDFs, it will attempt direct text extraction first, then use AWS Textract via
    aws_textract_utils if direct extraction fails or yields minimal text, and if Textract is available.
    Returns (text, error_message).
    """
    file_ext = file_name.split(".")[-1].lower() if '.' in file_name else ''
    text_content: Optional[str] = None
    error_message: Optional[str] = None
    
    try:
        file_obj.seek(0) # Ensure stream is at the beginning

        if file_ext == "pdf":
            # 1. Try PyPDF2 for direct text extraction
            try:
                reader = PdfReader(file_obj)
                pdf_text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
                if pdf_text_parts:
                    text_content = "\n".join(pdf_text_parts).strip()
                    logger.info(f"Successfully extracted text from PDF '{file_name}' using PyPDF2.")
            except Exception as e_pypdf:
                logger.warning(f"PyPDF2 failed to extract text from PDF '{file_name}': {e_pypdf}. Will try OCR if available.")
                text_content = None # Ensure it's None so OCR is attempted

            # 2. If PyPDF2 fails or extracts too little, and Textract is available, use it.
            #    The user request is "PDF/OCR via aws". So if it's PDF, Textract is the OCR method.
            if not text_content or len(text_content) < 50: # Threshold for "minimal text"
                if AWS_TEXTRACT_MODULE_AVAILABLE and aws_perform_textract_ocr:
                    logger.info(f"Direct PDF extraction for '{file_name}' yielded minimal/no text. Attempting AWS Textract OCR.")
                    file_obj.seek(0) # Reset stream for Textract
                    pdf_bytes = file_obj.getvalue()
                    # Using the imported aws_perform_textract_ocr from aws_textract_utils
                    ocr_text, pages_ocrd, ocr_error = aws_perform_textract_ocr(pdf_bytes, file_name)
                    if ocr_error:
                        error_message = f"Textract OCR failed for '{file_name}': {ocr_error}"
                        logger.error(error_message)
                        # Keep PyPDF2 text if it existed, even if minimal, otherwise None
                    elif ocr_text:
                        text_content = ocr_text.strip()
                        logger.info(f"Successfully extracted text from PDF '{file_name}' using AWS Textract ({pages_ocrd} pages).")
                    else:
                        logger.warning(f"Textract OCR for '{file_name}' returned no text but no error.")
                        # Keep PyPDF2 text if it existed
                elif text_content: # PyPDF2 got something, but it was minimal, and Textract not available
                    logger.info(f"Direct PDF extraction for '{file_name}' yielded minimal text, Textract OCR not available/used.")
                else: # PyPDF2 failed, and Textract not available
                    error_message = f"Failed to extract text from PDF '{file_name}'. Direct extraction failed and Textract OCR is not available/configured."
                    logger.warning(error_message)
        
        elif file_ext == "docx":
            doc = Document(file_obj)
            text_content = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
            logger.info(f"Successfully extracted text from DOCX: {file_name}")
        
        elif file_ext == "txt":
            try:
                text_content = file_obj.getvalue().decode("utf-8", "ignore").strip()
            except Exception as e_txt_decode: # Catch potential errors if getvalue() isn't bytes or decode fails
                logger.error(f"Error decoding TXT file {file_name} as UTF-8: {e_txt_decode}")
                file_obj.seek(0) # Reset and try with default system encoding or ISO-8859-1
                try:
                    text_content = file_obj.read().decode(errors='replace').strip() # More general read
                except Exception as e_txt_read_fallback:
                    error_message = f"Could not decode TXT file {file_name}: {e_txt_read_fallback}"
            if text_content: logger.info(f"Successfully extracted text from TXT: {file_name}")

        else:
            error_message = f"Unsupported file type: '{file_ext}' for file '{file_name}'"
            logger.warning(error_message)

        # Final check for empty content after processing
        if text_content is not None and not text_content.strip():
            text_content = None # Set to None if it's just whitespace
            if not error_message: # Don't overwrite an existing error
                 logger.info(f"No text content extracted from {file_name} (empty after extraction process).")
        elif text_content and not error_message:
             logger.info(f"Extraction complete for {file_name}. Text length: {len(text_content)} chars.")

    except Exception as e:
        logger.error(f"Error reading or processing uploaded file {file_name} (ext: {file_ext}): {e}", exc_info=True)
        error_message = f"File Read/Process Error for {file_name}: {e}"
        text_content = None # Ensure text_content is None on error
        
    return text_content, error_message
