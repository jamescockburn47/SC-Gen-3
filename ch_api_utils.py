# ch_api_utils.py

import logging
import time
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime

import requests
import re

from config import (
    get_ch_session,
    CH_API_BASE_URL,
    CH_DOCUMENT_API_BASE_URL,
    logger # Use central logger
)

# Cache for company profiles to reduce redundant API calls within a single application run
_company_profile_cache: Dict[str, Dict[str, Any]] = {}

def get_company_profile(company_no: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the company profile from Companies House API.
    Uses a local cache to avoid redundant calls for the same company number within a run.
    The provided api_key is used to configure the session if not already set by CH_API_KEY env var.
    """
    if company_no in _company_profile_cache:
        logger.debug(f"Using cached profile for company {company_no}.")
        return _company_profile_cache[company_no]

    ch_session = get_ch_session(api_key_override=api_key) 
    company_profile_url = f"{CH_API_BASE_URL}/company/{company_no}"
    
    try:
        time.sleep(0.3) # Small delay for CH API politeness
        profile_resp = ch_session.get(company_profile_url, timeout=30)
        profile_resp.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        company_profile_data = profile_resp.json()
        _company_profile_cache[company_no] = company_profile_data
        logger.info(f"Fetched and cached profile for company {company_no}.")
        return company_profile_data
    except requests.exceptions.HTTPError as e_http:
        # Log specific HTTP errors, e.g., 404 Not Found
        logger.error(f"HTTP error fetching company profile {company_no}: {e_http.response.status_code} - {e_http.response.text[:200]}")
        if e_http.response.status_code == 404:
            _company_profile_cache[company_no] = {"error": "Company not found", "status_code": 404} # Cache not found
        return None # Or return a dict indicating error like: {"error": str(e_http), "status_code": e_http.response.status_code}
    except requests.exceptions.RequestException as e_req:
        logger.error(f"API request failed for company profile {company_no}: {e_req}", exc_info=True)
        return None
    except json.JSONDecodeError as e_json:
        logger.error(f"Failed to decode JSON for company profile {company_no}: {e_json}. Response text: {getattr(profile_resp, 'text', 'N/A')[:200]}", exc_info=True)
        return None

def _fetch_document_content_from_ch(
    company_no: str,
    item_details: Dict[str, Any] # This is the document metadata item
) -> Tuple[Dict[str, Union[str, bytes, Dict]], List[str], Optional[str]]:
    """
    Fetches the content of a single document from Companies House, prioritizing formats:
    1. JSON (if eligible for the document type)
    2. XHTML
    3. XML (as a fallback if XHTML is not available or if it's iXBRL)
    4. PDF (as a last resort)

    Args:
        company_no: The company number.
        item_details: The metadata dictionary for the document from filing history.

    Returns:
        A tuple: (content_dict, list_of_fetched_types, error_message_or_None).
        content_dict: {"json": ..., "xhtml": ..., "xml": ..., "pdf": ...} (only keys for successful fetches)
        list_of_fetched_types: List of successfully fetched types in order of preference (e.g., ["json", "xhtml"])
        error_message_or_None: An error message if all fetch attempts fail, otherwise None.
    """
    ch_session = get_ch_session() # Relies on global config for API key or override
    doc_meta_link = item_details.get("links", {}).get("document_metadata", "")
    
    if not doc_meta_link:
        err_msg = f"No document_metadata link for item: {item_details.get('transaction_id', 'N/A')}"
        logger.warning(f"{company_no}: {err_msg}")
        return {"error": err_msg}, [], err_msg
    
    # Extract document ID from the metadata link
    doc_id_match = re.search(r"/document/([^/]+)", doc_meta_link)
    if not doc_id_match:
        err_msg = f"Could not parse document ID from metadata link: {doc_meta_link}"
        logger.error(f"{company_no}: {err_msg}")
        return {"error": err_msg}, [], err_msg
    doc_id = doc_id_match.group(1)

    content_url = f"{CH_DOCUMENT_API_BASE_URL}/document/{doc_id}/content"
    doc_ch_type_code = item_details.get("type", "UNKNOWN").upper() # CH Form Type (e.g., AA, CS01)
    doc_description = item_details.get('description', 'N/A')
    request_delay = 0.35 # Seconds to wait between CH API calls

    content_dict: Dict[str, Union[str, bytes, Dict]] = {}
    fetched_types: List[str] = []
    errors_encountered: List[str] = []

    # Define eligible types for JSON content (typically accounts)
    json_eligible_types = [
        "AA", "AP01", "AP02", "AP03", "AP04", "CH01", "CH02", "CH03", "CH04", 
        "TM01", "TM02", "CS01", "PSC01", "PSC02", "PSC03", "PSC04", "PSC05", 
        "PSC06", "PSC07", "PSC08", "PSC09", "MR01", "MR02", "MR04", "MR05",
        "ACCOUNTS TYPE FULL", "ACCOUNTS TYPE MEDIUM", "ACCOUNTS TYPE SMALL", 
        "ACCOUNTS TYPE MICROENTITY", "ACCOUNTS TYPE GROUP", "ACCOUNTS TYPE INTERIM", 
        "ACCOUNTS TYPE INITIAL", "ACCOUNTS TYPE DORMANT"
    ] # This list can be refined based on CH API documentation for JSON availability.

    # 1. Attempt JSON if eligible
    if doc_ch_type_code in json_eligible_types:
        logger.debug(f"{company_no}: Doc {doc_id} (Type: {doc_ch_type_code}) is eligible for JSON. Attempting fetch.")
        try:
            time.sleep(request_delay)
            resp = ch_session.get(content_url, headers={"Accept": "application/json"}, timeout=45)
            resp.raise_for_status()
            if "application/json" in resp.headers.get("Content-Type", "").lower():
                content_dict["json"] = resp.json()
                fetched_types.append("json")
                logger.info(f"{company_no}: Successfully fetched JSON for doc {doc_id}.")
            else:
                logger.debug(f"{company_no}: JSON request for doc {doc_id} returned Content-Type: {resp.headers.get('Content-Type', '')}. Preview: {resp.text[:100]}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [404, 406, 503]: # Not available or temporarily unavailable
                logger.info(f"{company_no}: JSON not available (HTTP {e.response.status_code}) for doc {doc_id}.")
            else: # Other HTTP error
                errors_encountered.append(f"JSON HTTP error {e.response.status_code}")
                logger.warning(f"{company_no}: JSON fetch HTTP error for doc {doc_id}: {e}")
        except json.JSONDecodeError as e_json:
            errors_encountered.append(f"JSON decode error: {e_json}")
            logger.error(f"{company_no}: Failed to decode JSON for doc {doc_id}: {e_json}. Response: {getattr(resp, 'text', '')[:200]}", exc_info=True)
        except requests.exceptions.RequestException as e_req:
            errors_encountered.append(f"JSON request error: {e_req}")
            logger.warning(f"{company_no}: JSON fetch RequestException for doc {doc_id}: {e_req}", exc_info=True)
    
    # 2. Attempt XHTML (if JSON not fetched or preferred)
    if "json" not in content_dict: # Only proceed if JSON wasn't successfully fetched
        logger.debug(f"{company_no}: Attempting XHTML for doc {doc_id} (Type: {doc_ch_type_code}).")
        try:
            time.sleep(request_delay)
            resp = ch_session.get(content_url, headers={"Accept": "application/xhtml+xml"}, timeout=45)
            resp.raise_for_status()
            if "application/xhtml+xml" in resp.headers.get("Content-Type", "").lower():
                content_dict["xhtml"] = resp.text
                fetched_types.append("xhtml")
                logger.info(f"{company_no}: Successfully fetched XHTML for doc {doc_id}.")
            else:
                logger.debug(f"{company_no}: XHTML request for doc {doc_id} returned Content-Type: {resp.headers.get('Content-Type', '')}. Preview: {resp.text[:100]}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [404, 406, 503]:
                logger.info(f"{company_no}: XHTML not available (HTTP {e.response.status_code}) for doc {doc_id}.")
            else:
                errors_encountered.append(f"XHTML HTTP error {e.response.status_code}")
                logger.warning(f"{company_no}: XHTML fetch HTTP error for doc {doc_id}: {e}")
        except requests.exceptions.RequestException as e_req:
            errors_encountered.append(f"XHTML request error: {e_req}")
            logger.warning(f"{company_no}: XHTML fetch RequestException for doc {doc_id}: {e_req}", exc_info=True)

    # 3. Attempt XML (if JSON and XHTML not fetched) - useful for iXBRL accounts
    if "json" not in content_dict and "xhtml" not in content_dict:
        logger.debug(f"{company_no}: Attempting XML for doc {doc_id} (Type: {doc_ch_type_code}).")
        try:
            time.sleep(request_delay)
            resp = ch_session.get(content_url, headers={"Accept": "application/xml"}, timeout=45)
            resp.raise_for_status()
            # CH sometimes returns PDF even if XML is requested if XML is not available.
            # Also, ensure it's actual XML and not something else. iXBRL is XML with HTML inside.
            content_type_header = resp.headers.get("Content-Type", "").lower()
            if "application/xml" in content_type_header or ("text/xml" in content_type_header):
                # Basic check for iXBRL (often starts with <html> tag within the XML)
                # A more robust check might involve parsing the start of the content.
                if b"<html" in resp.content[:500] or b"<ix:header>" in resp.content[:500]: # Check for common iXBRL patterns
                    content_dict["xml"] = resp.text # Store as text
                    fetched_types.append("xml") # Could also be "ixbrl"
                    logger.info(f"{company_no}: Successfully fetched XML/iXBRL for doc {doc_id}.")
                else: # It's XML but maybe not iXBRL, still store it
                    content_dict["xml"] = resp.text
                    fetched_types.append("xml")
                    logger.info(f"{company_no}: Successfully fetched generic XML for doc {doc_id}.")
            elif "application/pdf" in content_type_header:
                 logger.info(f"{company_no}: XML request for doc {doc_id} returned PDF instead. Will fall back to PDF if needed.")
            else:
                logger.debug(f"{company_no}: XML request for doc {doc_id} returned Content-Type: {content_type_header}. Preview: {resp.text[:100]}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [404, 406, 503]:
                logger.info(f"{company_no}: XML not available (HTTP {e.response.status_code}) for doc {doc_id}.")
            else:
                errors_encountered.append(f"XML HTTP error {e.response.status_code}")
                logger.warning(f"{company_no}: XML fetch HTTP error for doc {doc_id}: {e}")
        except requests.exceptions.RequestException as e_req:
            errors_encountered.append(f"XML request error: {e_req}")
            logger.warning(f"{company_no}: XML fetch RequestException for doc {doc_id}: {e_req}", exc_info=True)

    # 4. Attempt PDF (if JSON, XHTML, and XML not fetched) - This is the last resort.
    if not fetched_types: # If no preferred format was obtained
        logger.info(f"{company_no}: No JSON, XHTML, or XML fetched for doc {doc_id}. Attempting PDF as last resort.")
        try:
            time.sleep(request_delay)
            resp = ch_session.get(content_url, headers={"Accept": "application/pdf"}, timeout=120) # Longer timeout for potentially large PDFs
            resp.raise_for_status()
            if "application/pdf" in resp.headers.get("Content-Type", "").lower():
                content_dict["pdf"] = resp.content # PDF content is bytes
                fetched_types.append("pdf")
                logger.info(f"{company_no}: Successfully fetched PDF for doc {doc_id} (Size: {len(resp.content)} bytes).")
            else:
                err_msg = f"PDF request for doc {doc_id} returned Content-Type {resp.headers.get('Content-Type', '')} instead of PDF. Preview: {resp.text[:100]}"
                errors_encountered.append(err_msg)
                logger.warning(f"{company_no}: {err_msg}")
        except requests.exceptions.RequestException as e_req:
            err_msg = f"PDF fetch failed for doc {doc_id}: {e_req}"
            errors_encountered.append(err_msg)
            logger.error(f"{company_no}: {err_msg}", exc_info=True)
        except Exception as e_pdf_other:
            err_msg = f"Unexpected error fetching PDF for doc {doc_id}: {e_pdf_other}"
            errors_encountered.append(err_msg)
            logger.error(f"{company_no}: {err_msg}", exc_info=True)

    if fetched_types: # If at least one format was successfully fetched
        return content_dict, fetched_types, None
    
    # If all attempts failed
    final_error_summary = (
        f"All attempts to fetch content for doc {doc_id} (Type: {doc_ch_type_code}, Desc: {doc_description}) failed. "
        f"Errors: {'; '.join(errors_encountered) if errors_encountered else 'No specific errors logged, but no content retrieved.'}"
    )
    logger.warning(f"{company_no}: {final_error_summary}")
    return content_dict, [], final_error_summary


def get_ch_documents_metadata(
    company_no: str,
    api_key: str, 
    categories: List[str],
    items_per_page: int,
    max_docs_to_fetch_meta: int,
    target_docs_per_category_in_date_range: int,
    year_range: Tuple[int, int]
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetches filing history metadata from Companies House for specified categories,
    filtering by year range as data is paged.
    """
    ch_session = get_ch_session(api_key_override=api_key)
    start_filter_year, end_filter_year = year_range
    logger.info(
        f"Fetching CH document metadata for {company_no} | Categories: {categories} | "
        f"Years: {start_filter_year}-{end_filter_year} | Target/cat: {target_docs_per_category_in_date_range} | "
        f"Scan limit/cat: {max_docs_to_fetch_meta}"
    )

    if not categories:
        logger.warning(f"No categories specified for {company_no}. Skipping metadata fetch.")
        # Try to get profile even if no categories, as it might be useful
        profile_data_no_cat = get_company_profile(company_no, api_key)
        return [], profile_data_no_cat, "No categories specified for metadata fetch."

    company_profile_data = get_company_profile(company_no, api_key)
    if not company_profile_data or company_profile_data.get("error"): # Check if profile fetch failed
        err_msg = f"Failed to retrieve company profile for {company_no}."
        if company_profile_data and company_profile_data.get("error"):
            err_msg += f" Reason: {company_profile_data.get('error')}"
        logger.error(err_msg)
        return [], None, err_msg

    filings_url_path = company_profile_data.get("links", {}).get("filing_history")
    if not filings_url_path:
        err_msg = f"No 'filing_history' link found in profile for company {company_no}."
        logger.warning(err_msg)
        return [], company_profile_data, err_msg 

    full_filings_url = filings_url_path if filings_url_path.startswith("http") else f"{CH_API_BASE_URL}{filings_url_path}"

    all_docs_in_date_range: List[Dict[str, Any]] = []
    processed_categories_log = set() # To avoid redundant logging for the same category
    total_api_calls_for_filings = 0
    overall_fetch_error: Optional[str] = None

    for category_to_fetch in categories:
        cat_lower = category_to_fetch.lower().strip()
        if not cat_lower or cat_lower in processed_categories_log:
            continue
        processed_categories_log.add(cat_lower)
        
        logger.info(f"{company_no}: Category '{cat_lower}': Targeting {target_docs_per_category_in_date_range} docs in range {start_filter_year}-{end_filter_year}.")

        current_start_index = 0
        docs_found_for_this_category_in_range: List[Dict[str, Any]] = []
        items_scanned_from_api_for_category = 0 

        while True: 
            if items_scanned_from_api_for_category >= max_docs_to_fetch_meta:
                logger.info(f"{company_no}: Category '{cat_lower}': Reached API scan limit ({max_docs_to_fetch_meta} items).")
                break

            params = {
                "category": cat_lower,
                "items_per_page": min(items_per_page, 100), # CH API max is 100
                "start_index": current_start_index,
            }
            try:
                time.sleep(0.51) # Adhere to CH rate limits (20 calls per second, but play safe)
                total_api_calls_for_filings += 1
                logger.debug(f"{company_no}: Filings API call #{total_api_calls_for_filings} to {full_filings_url} with params: {params}")
                
                resp = ch_session.get(full_filings_url, params=params, timeout=45)
                resp.raise_for_status()
                api_response_data = resp.json()
            except requests.exceptions.RequestException as e_req_filings:
                err_msg = f"API request failed for filings (cat: {cat_lower}, company: {company_no}): {e_req_filings}"
                logger.error(err_msg, exc_info=True)
                overall_fetch_error = err_msg # Store the first critical error
                break # Stop processing this category on critical error
            except json.JSONDecodeError as e_json_filings:
                err_msg = f"Failed to decode JSON for filings (cat: {cat_lower}, company: {company_no}): {e_json_filings}. Response: {getattr(resp, 'text', '')[:200]}"
                logger.error(err_msg, exc_info=True)
                overall_fetch_error = err_msg
                break 
            
            if overall_fetch_error: break # If error occurred in try-except, break from category loop

            items_on_this_page = api_response_data.get("items", [])
            api_total_count_for_this_category = api_response_data.get("total_count", 0)
            
            if not items_on_this_page:
                logger.info(f"{company_no}: Category '{cat_lower}': No more items from API at start_index {current_start_index}.")
                break 

            for item in items_on_this_page:
                if len(docs_found_for_this_category_in_range) >= target_docs_per_category_in_date_range:
                    break 
                item_date_str = item.get("date")
                if item_date_str:
                    try:
                        item_datetime_obj = datetime.strptime(item_date_str, "%Y-%m-%d")
                        item_year = item_datetime_obj.year
                        if start_filter_year <= item_year <= end_filter_year:
                            item['_parsed_date_obj'] = item_datetime_obj # For potential sorting later
                            item['company_number'] = company_no # Ensure company number is in each doc item
                            docs_found_for_this_category_in_range.append(item)
                    except ValueError:
                        logger.warning(f"{company_no}: Cat '{cat_lower}': Could not parse date '{item_date_str}' for item TX_ID: {item.get('transaction_id', 'N/A')}. Skipping date filter.")
                else: # No date on item, cannot filter by year
                    logger.warning(f"{company_no}: Cat '{cat_lower}': Item TX_ID: {item.get('transaction_id', 'N/A')} has no date. Cannot apply year filter.")
            
            items_scanned_from_api_for_category += len(items_on_this_page)

            logger.debug(
                f"{company_no}: Cat '{cat_lower}': Page (start_idx {current_start_index}) scanned {len(items_on_this_page)} items. "
                f"In range for cat: {len(docs_found_for_this_category_in_range)}/{target_docs_per_category_in_date_range}. "
                f"Total API items scanned for cat: {items_scanned_from_api_for_category}/{max_docs_to_fetch_meta} (API reports {api_total_count_for_this_category} total for cat)."
            )

            if len(docs_found_for_this_category_in_range) >= target_docs_per_category_in_date_range:
                logger.info(f"{company_no}: Cat '{cat_lower}': Met target of {target_docs_per_category_in_date_range} docs in date range.")
                break 
            if items_scanned_from_api_for_category >= max_docs_to_fetch_meta:
                logger.info(f"{company_no}: Cat '{cat_lower}': Reached API scan limit of {max_docs_to_fetch_meta}.")
                break
            if current_start_index + items_per_page >= api_total_count_for_this_category and api_total_count_for_this_category > 0: # All items for category scanned
                logger.info(f"{company_no}: Cat '{cat_lower}': Scanned all ~{api_total_count_for_this_category} available API items.")
                break
            
            current_start_index += items_per_page
        
        if overall_fetch_error: break # Break from outer category loop if a critical error occurred

        all_docs_in_date_range.extend(docs_found_for_this_category_in_range)
        logger.info(f"{company_no}: Cat '{cat_lower}': Added {len(docs_found_for_this_category_in_range)} docs. Total docs in date range so far: {len(all_docs_in_date_range)}.")

    # Sort all collected documents by date (most recent first) if needed, using '_parsed_date_obj'
    all_docs_in_date_range.sort(key=lambda x: x.get('_parsed_date_obj', datetime.min), reverse=True)
    # Clean up the temporary parsed date object if not needed downstream
    for item in all_docs_in_date_range:
        item.pop('_parsed_date_obj', None)

    logger.info(
        f"{company_no}: Completed metadata fetch. Found {len(all_docs_in_date_range)} docs in date range across all categories. "
        f"Total CH Filings API calls: {total_api_calls_for_filings} (excluding profile call)."
    )
    return all_docs_in_date_range, company_profile_data, overall_fetch_error

def get_company_pscs(company_no: str, api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetches Persons with Significant Control (PSC) data for a company.
    Returns the full JSON response from the PSC list endpoint and an optional error message.
    """
    ch_session = get_ch_session(api_key_override=api_key)
    psc_list_url = f"{CH_API_BASE_URL}/company/{company_no}/persons-with-significant-control"
    logger.info(f"Fetching PSC data for company {company_no}.")

    try:
        time.sleep(0.3) # API politeness
        response = ch_session.get(psc_list_url, timeout=30)
        response.raise_for_status()
        psc_data = response.json()
        logger.info(f"Successfully fetched PSC data for {company_no}. Found {psc_data.get('active_count', 0)} active PSCs.")
        return psc_data, None
    except requests.exceptions.HTTPError as e_http:
        if e_http.response.status_code == 404:
            logger.info(f"No PSC data found for company {company_no} (HTTP 404). This may be normal for some company types or if none registered.")
            return {"items": [], "active_count": 0, "total_results": 0, "message": "No PSC data found (404)"}, None # Return empty-like structure
        logger.error(f"HTTP error fetching PSC data for {company_no}: {e_http.response.status_code} - {e_http.response.text[:200]}")
        return None, f"PSC API HTTP Error: {e_http.response.status_code}"
    except requests.exceptions.RequestException as e_req:
        logger.error(f"Request failed for PSC data for {company_no}: {e_req}", exc_info=True)
        return None, f"PSC API Request Error: {e_req}"
    except json.JSONDecodeError as e_json:
        logger.error(f"Failed to decode JSON for PSC data {company_no}: {e_json}. Response: {getattr(response, 'text', '')[:200]}", exc_info=True)
        return None, "PSC API JSON Decode Error"

# ---------------------------------------------------------------------
# Legacy compatibility shim – remove once the new code path is universal
# ---------------------------------------------------------------------
class CompanyHouseAPIRateLimitError(Exception): ...
class CompanyHouseAPIAuthError(Exception): ...
class CompanyHouseAPINotFoundError(Exception): ...

class CompanyHouseAPI:
    """
    DO NOT use for new code.  Only here so that historical references in
    ch_pipeline, group_structure_utils, etc. keep importing.
    """
    def __init__(self, *args, **kwargs): pass

    # expose just the methods ch_pipeline actually calls
    def get_company_filings(self, company_number: str, **_) -> list[dict]:
        return get_ch_documents_metadata(company_number)[0]

    def download_document(
        self, company_number: str, document_id: str, *, stream: bool = False, **_
    ):
        return _fetch_document_content_from_ch(company_number, document_id, stream=stream)
