# aws_textract_utils.py

import logging
import time
import uuid
import re # Import re for sanitizing filenames
from typing import Tuple, Optional, Dict, Any # Added Dict, Any

# Attempt to import boto3 and set TEXTRACT_AVAILABLE flag
try:
    import boto3
    from botocore.exceptions import ClientError as BotoClientError, NoCredentialsError, PartialCredentialsError
    TEXTRACT_AVAILABLE = True # boto3 is available
except ImportError:
    boto3 = None # type: ignore
    BotoClientError = None # type: ignore
    NoCredentialsError = None # type: ignore
    PartialCredentialsError = None # type: ignore
    TEXTRACT_AVAILABLE = False # boto3 is not available
    # logger will be available from config, but if this file is used standalone, it might not be.
    # For robustness, could add a local logger initialization here if config.logger isn't guaranteed.
    # However, assuming it's part of the larger app where config.logger is set.

# Import AWS configuration from config.py
# This assumes config.py and its logger are correctly set up and importable.
try:
    from config import (
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_REGION_DEFAULT,
        S3_TEXTRACT_BUCKET,
        AWS_PRICING_CONFIG, 
        logger # Use the centrally configured logger
    )
except ImportError:
    # Fallback if config.py is not found or causes issues during isolated testing/Pylance checks
    # This is primarily for Pylance to have some defaults if config is problematic.
    # In a running app, config.py should be correctly structured and importable.
    logger = logging.getLogger(__name__) # Fallback logger
    logger.warning("Could not import from config.py in aws_textract_utils. Using fallback logger and default AWS values might be undefined.")
    AWS_ACCESS_KEY_ID = None
    AWS_SECRET_ACCESS_KEY = None
    AWS_REGION_DEFAULT = "us-east-1" # A common default, but should be from config
    S3_TEXTRACT_BUCKET = None
    AWS_PRICING_CONFIG = {}


_textract_client = None
_s3_client = None
_aws_clients_initialized = False # Tracks if our _initialize_aws_clients has run successfully

def _initialize_aws_clients() -> bool:
    """Initializes AWS clients if not already done. Returns True on success."""
    global _textract_client, _s3_client, _aws_clients_initialized
    if _aws_clients_initialized:
        return True
    
    if not TEXTRACT_AVAILABLE: 
        logger.error("Cannot initialize AWS clients because boto3 library is not available.")
        return False

    if not AWS_REGION_DEFAULT:
        logger.error("AWS_DEFAULT_REGION not set in config. Cannot initialize AWS Textract/S3 clients.")
        return False
    if not S3_TEXTRACT_BUCKET: # Check S3_TEXTRACT_BUCKET from config
        logger.error("S3_TEXTRACT_BUCKET not set in config. Cannot initialize S3 client for Textract.")
        return False

    try:
        session_params: Dict[str, Any] = {"region_name": AWS_REGION_DEFAULT} # Type hint for session_params
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY: 
            session_params["aws_access_key_id"] = AWS_ACCESS_KEY_ID
            session_params["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        
        if boto3 is None: 
            logger.error("boto3 is None, cannot create Boto session (Import failed).")
            return False

        boto_session = boto3.Session(**session_params)
        _textract_client = boto_session.client("textract")
        _s3_client = boto_session.client("s3")
        
        logger.info(f"AWS Textract and S3 clients initialized for region {AWS_REGION_DEFAULT} and bucket {S3_TEXTRACT_BUCKET}.")
        _aws_clients_initialized = True
        return True
    except (NoCredentialsError, PartialCredentialsError) as e_creds: # type: ignore 
        logger.error(f"AWS credentials not found or incomplete: {e_creds}. Textract functionality will be unavailable.")
    except BotoClientError as e_boto_init: # type: ignore 
        logger.error(f"BotoClientError initializing AWS clients: {e_boto_init}. Textract functionality may be unavailable.")
    except Exception as e_aws_init:
        logger.error(f"Failed to initialize AWS clients: {e_aws_init}. Textract functionality may be unavailable.", exc_info=True)
    
    _aws_clients_initialized = False 
    return False


def perform_textract_ocr(
    pdf_bytes: bytes,
    company_no_for_logging: str = "N/A_Textract"
) -> Tuple[str, int, Optional[str]]:
    """
    Performs OCR on PDF bytes using AWS Textract via S3.
    Returns (extracted_text, pages_processed, error_message).
    Error_message is None if successful.
    """
    # Initialize job_final_status at the beginning of the function
    job_final_status: str = "PRE_JOB_STATE" 

    if not _initialize_aws_clients(): 
        err_msg = "AWS Textract/S3 clients not available or failed to initialize. OCR skipped."
        return "", 0, err_msg
    
    if not _textract_client or not _s3_client: # Should be caught by above, but as defense
        err_msg = "AWS Textract/S3 client instances are None after initialization attempt. OCR skipped."
        logger.error(f"{company_no_for_logging}: {err_msg}")
        return "", 0, err_msg

    logger.info(f"{company_no_for_logging}: AWS Textract: Initiating OCR (Bucket: {S3_TEXTRACT_BUCKET}).")
    extracted_text = ""
    textract_pages_processed = 0
    safe_company_no_log = re.sub(r'[^\w\-.]', '_', company_no_for_logging) # Sanitize for S3 key
    s3_object_key = f"temp-textract-inputs/{safe_company_no_log}_{uuid.uuid4()}.pdf"
    textract_job_id: Optional[str] = None # Initialize textract_job_id
    upload_successful = False
    error_occurred_msg: Optional[str] = None # Initialize error_occurred_msg

    try:
        logger.debug(f"{company_no_for_logging}: AWS Textract: Uploading PDF ({len(pdf_bytes)} bytes) to S3: s3://{S3_TEXTRACT_BUCKET}/{s3_object_key}")
        _s3_client.put_object(Bucket=S3_TEXTRACT_BUCKET, Key=s3_object_key, Body=pdf_bytes)
        upload_successful = True
        logger.info(f"{company_no_for_logging}: AWS Textract: PDF uploaded to S3.")

        logger.debug(f"{company_no_for_logging}: AWS Textract: Starting job for S3 object: {s3_object_key}")
        start_response = _textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': S3_TEXTRACT_BUCKET, 'Name': s3_object_key}}
        )
        textract_job_id = start_response.get('JobId') # Use .get() for safety
        if not textract_job_id:
            logger.error(f"{company_no_for_logging}: Textract start_document_text_detection did not return a JobId. Response: {start_response}")
            job_final_status = "JOB_ID_MISSING"
            raise Exception("Textract job ID not returned from start_document_text_detection.")
            
        logger.info(f"{company_no_for_logging}: AWS Textract: Job started: {textract_job_id}")

        textract_text_parts: List[str] = [] # Type hint
        next_token_for_textract_results: Optional[str] = None # Type hint
        job_final_status = "UNKNOWN" # Re-initialize after job start attempt
        
        max_poll_attempts_config = 240 
        polling_interval_seconds_config = 1.5 
        logger.info(f"{company_no_for_logging}: AWS Textract: Polling job {textract_job_id} up to {max_poll_attempts_config} times...")

        for poll_attempt in range(max_poll_attempts_config):
            time.sleep(polling_interval_seconds_config)
            get_params_textract: Dict[str, Any] = {'JobId': textract_job_id} 
            if next_token_for_textract_results:
                get_params_textract['NextToken'] = next_token_for_textract_results
            
            try:
                get_response_textract = _textract_client.get_document_text_detection(**get_params_textract)
                job_status_textract = get_response_textract['JobStatus']
            except BotoClientError as e_get_boto: # type: ignore 
                logger.warning(f"{company_no_for_logging}: Textract BotoClientError during poll (Attempt {poll_attempt + 1}): {e_get_boto}. Retrying.")
                if 'AccessDeniedException' in str(e_get_boto) or 'InvalidS3ObjectException' in str(e_get_boto) : 
                    error_occurred_msg = f"Textract job failed due to S3 access/object issue: {e_get_boto}"
                    logger.error(f"{company_no_for_logging}: {error_occurred_msg}")
                    job_final_status = "POLLING_S3_ERROR"
                    break
                continue 
            except Exception as e_poll_gen:
                error_occurred_msg = f"Textract generic error during poll (Attempt {poll_attempt + 1}): {e_poll_gen}"
                logger.error(f"{company_no_for_logging}: {error_occurred_msg}", exc_info=True)
                job_final_status = "POLLING_EXCEPTION"
                break 

            if job_status_textract == 'SUCCEEDED':
                blocks_textract = get_response_textract.get('Blocks', [])
                for block in blocks_textract:
                    if block.get('BlockType') == 'LINE': 
                        textract_text_parts.append(block.get('Text', ''))
                
                doc_meta = get_response_textract.get('DocumentMetadata')
                if doc_meta and isinstance(doc_meta.get('Pages'), int): # Check doc_meta before accessing
                    textract_pages_processed = max(textract_pages_processed, doc_meta['Pages']) 
                
                next_token_for_textract_results = get_response_textract.get('NextToken')
                if not next_token_for_textract_results:
                    extracted_text = "\n".join(filter(None, textract_text_parts)).strip()
                    job_final_status = 'SUCCEEDED'
                    logger.info(f"{company_no_for_logging}: Textract: Job {textract_job_id} SUCCEEDED. Extracted {len(extracted_text)} chars. Processed {textract_pages_processed} pages.")
                    break 
                else:
                    logger.debug(f"{company_no_for_logging}: Textract: Job {textract_job_id} SUCCEEDED (partial), fetching next page...")
            elif job_status_textract == 'FAILED':
                job_final_status = 'FAILED'
                status_msg_textract = get_response_textract.get('StatusMessage', 'No error message from Textract.')
                error_occurred_msg = f"Textract: Job {textract_job_id} FAILED. StatusMessage: {status_msg_textract}"
                logger.error(f"{company_no_for_logging}: {error_occurred_msg}")
                break
            elif job_status_textract == 'IN_PROGRESS':
                if (poll_attempt + 1) % 30 == 0: 
                    logger.debug(f"{company_no_for_logging}: Textract: Job {textract_job_id} IN_PROGRESS (Poll {poll_attempt + 1}/{max_poll_attempts_config})...")
            else: 
                job_final_status = job_status_textract
                error_occurred_msg = f"Textract: Job {textract_job_id} unhandled status: {job_status_textract}."
                logger.warning(f"{company_no_for_logging}: {error_occurred_msg} Response: {get_response_textract}")
                break 
        else: 
            error_occurred_msg = f"Textract: Job {textract_job_id} timed out polling after {max_poll_attempts_config} attempts."
            logger.warning(f"{company_no_for_logging}: {error_occurred_msg}")
            job_final_status = 'TIMEOUT_POLLING'

    except (NoCredentialsError, PartialCredentialsError) as e_creds_runtime: # type: ignore
        error_occurred_msg = f"AWS credentials error during Textract operation: {e_creds_runtime}"
        job_final_status = "CREDENTIALS_ERROR_RUNTIME" # Set status
        logger.error(f"{company_no_for_logging}: {error_occurred_msg}", exc_info=True)
    except BotoClientError as e_aws_boto_outer: # type: ignore
        error_code_aws = "UnknownAWSBotoClientError"
        error_message_aws = str(e_aws_boto_outer)
        if hasattr(e_aws_boto_outer, 'response') and e_aws_boto_outer.response and isinstance(e_aws_boto_outer.response.get('Error'), dict): # type: ignore
            error_details = e_aws_boto_outer.response['Error'] # type: ignore
            error_code_aws = error_details.get('Code', error_code_aws)
            error_message_aws = error_details.get('Message', error_message_aws)
        
        error_occurred_msg = f"Textract BotoClientError (S3/Textract): {error_code_aws} - {error_message_aws} (Job ID: {textract_job_id})"
        job_final_status = f"BOTO_ERROR_{error_code_aws}" # Set status
        logger.error(f"{company_no_for_logging}: {error_occurred_msg}", exc_info=True)
        if error_code_aws in ('AccessDenied', 'UnauthorizedOperation', 'AccessDeniedException'):
            logger.error(f"CRITICAL CHECK: AWS Permissions for S3 bucket '{S3_TEXTRACT_BUCKET}' (GetObject, PutObject, DeleteObject) and Textract (StartDocumentTextDetection, GetDocumentTextDetection) might be insufficient or misconfigured!")
    except Exception as e_generic_outer_textract:
        error_occurred_msg = f"Textract: Unexpected generic error: {e_generic_outer_textract}"
        job_final_status = "GENERIC_OUTER_EXCEPTION" # Set status
        logger.error(f"{company_no_for_logging}: {error_occurred_msg}", exc_info=True)
    finally:
        if upload_successful and S3_TEXTRACT_BUCKET and _s3_client: 
            try:
                logger.debug(f"{company_no_for_logging}: Textract: Deleting temporary S3 object: s3://{S3_TEXTRACT_BUCKET}/{s3_object_key}")
                _s3_client.delete_object(Bucket=S3_TEXTRACT_BUCKET, Key=s3_object_key)
                logger.info(f"{company_no_for_logging}: Textract: Temp S3 object {s3_object_key} deleted.")
            except Exception as e_del_s3_final:
                logger.error(f"{company_no_for_logging}: Textract: FAILED to delete S3 object {s3_object_key}: {e_del_s3_final}. Manual cleanup may be required in '{S3_TEXTRACT_BUCKET}'.")

    # Check job_final_status which is now guaranteed to be initialized
    if not extracted_text and not error_occurred_msg: 
        if job_final_status != 'SUCCEEDED':
            error_occurred_msg = f"Textract: OCR yielded no text. Final job status for {textract_job_id or 'N/A'} was '{job_final_status}'."
            logger.warning(f"{company_no_for_logging}: {error_occurred_msg}")
        else: 
            logger.info(f"{company_no_for_logging}: Textract: OCR job {textract_job_id or 'N/A'} SUCCEEDED but returned no text (possibly blank or image-only PDF with no detectable text).")

    return extracted_text, textract_pages_processed, error_occurred_msg

def get_textract_cost_estimation(pages_processed: int, num_pdfs_processed: int) -> Dict[str, Any]: # Changed dict to Dict[str, Any]
    """Estimates AWS Textract and related S3 costs."""
    if not _aws_clients_initialized: 
        return {
            "notes": "AWS clients not initialized, or Textract was not used. Cost estimation skipped.",
            "total_estimated_aws_cost_usd_for_ocr": 0.0,
            "total_estimated_aws_cost_gbp_for_ocr": 0.0
        }

    # Ensure AWS_PRICING_CONFIG is not None and is a dict
    current_aws_pricing_config = AWS_PRICING_CONFIG if isinstance(AWS_PRICING_CONFIG, dict) else {}

    s3_put_cost_usd = num_pdfs_processed * current_aws_pricing_config.get("s3_put_request_per_pdf_to_textract", 0.000005) 
    textract_page_processing_cost_usd = pages_processed * current_aws_pricing_config.get("textract_per_page", 0.0015) 
    
    total_aws_cost_usd = s3_put_cost_usd + textract_page_processing_cost_usd
    exchange_rate = 0.80 # Default
    try:
        # Ensure the value from config is explicitly cast to float
        config_rate = current_aws_pricing_config.get("usd_to_gbp_exchange_rate", 0.80)
        if config_rate is not None: # Check if it's not None before casting
            exchange_rate = float(config_rate)
    except (ValueError, TypeError): # Catch if casting fails or if config_rate is not suitable for float()
        logger.warning(f"Could not parse USD_TO_GBP_EXCHANGE_RATE ('{config_rate}') as float, using default {exchange_rate}.")

    total_aws_cost_gbp = total_aws_cost_usd * exchange_rate

    return {
        "num_pdfs_sent_to_textract_s3": num_pdfs_processed,
        "num_pages_processed_by_textract": pages_processed,
        "estimated_s3_put_cost_usd_for_textract_pdfs": round(s3_put_cost_usd, 5),
        "estimated_textract_page_processing_cost_usd": round(textract_page_processing_cost_usd, 5),
        "total_estimated_aws_cost_usd_for_ocr": round(total_aws_cost_usd, 5),
        "total_estimated_aws_cost_gbp_for_ocr": round(total_aws_cost_gbp, 5),
        "notes": "AWS costs cover Textract page processing and S3 PUTs for PDFs sent to Textract. Other S3 costs (storage, GETs) not included."
    }
