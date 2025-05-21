# config.py

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Any, Tuple

from dotenv import load_dotenv
import openai
import requests

# Attempt to import Google Generative AI
try:
    import google.generativeai as genai
except ImportError:
    genai = None # Set to None if not installed

# Attempt to import Google API core exceptions for specific error handling
try:
    from google.api_core import exceptions as GoogleAPICoreExceptions
except Exception:
    GoogleAPICoreExceptions = None

load_dotenv()

# Local directory used by ch_pipeline to cache downloaded documents
APPLICATION_SCRATCH_DIR = Path(
    os.getenv("APPLICATION_SCRATCH_DIR", Path(__file__).resolve().parent / "scratch")
)
APPLICATION_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("strategic_counsel_app") # Changed to a more specific root logger name

# Suppress verbose logs from libraries
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING) # Added for noisy urllib3 logs often seen with requests

# Suppress specific warnings
from PyPDF2 import errors as PyPDF2Errors
warnings.filterwarnings("ignore", message="incorrect startxref pointer.*")
warnings.filterwarnings("ignore", category=PyPDF2Errors.PdfReadWarning)


# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CH_API_KEY = os.getenv("CH_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") # For Textract
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") # For Textract
# AWS_REGION_DEFAULT is used for Textract. It's sourced from the AWS_DEFAULT_REGION env var.
AWS_REGION_DEFAULT = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")
S3_TEXTRACT_BUCKET = os.getenv("S3_TEXTRACT_BUCKET") # For Textract

# --- Google Drive Integration ---
GOOGLE_CLIENT_SECRET_FILE = os.getenv(
    "GOOGLE_CLIENT_SECRET_FILE",
    str(Path(__file__).resolve().parent / "client_secret.json"),
)
GOOGLE_TOKEN_FILE = os.getenv(
    "GOOGLE_TOKEN_FILE",
    str(Path(__file__).resolve().parent / "google_token.json"),
)
GOOGLE_API_SCOPES = ["https://www.googleapis.com/auth/drive"]
ENABLE_GOOGLE_DRIVE_INTEGRATION = os.getenv("ENABLE_GOOGLE_DRIVE_INTEGRATION", "false").lower() == "true"

# --- Model Configuration ---
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o")
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL_FOR_SUMMARIES", "gemini-1.5-flash-latest") # More specific name
GEMINI_MODEL_FOR_PROTOCOL_CHECK = os.getenv("GEMINI_MODEL_FOR_PROTOCOL_CHECK", "gemini-1.5-flash-latest") # Model for protocol check
PROTOCOL_CHECK_MODEL_PROVIDER = os.getenv("PROTOCOL_CHECK_MODEL_PROVIDER", "gemini")  # "gemini" or "openai"

# --- Application Constants ---
MIN_MEANINGFUL_TEXT_LEN = 200
# Limit the number of filings processed for each company during a pipeline run
MAX_DOCS_TO_PROCESS_PER_COMPANY = int(os.getenv("MAX_DOCS_PER_COMPANY_PIPELINE", "20"))
MAX_TEXTRACT_WORKERS = int(os.getenv("MAX_TEXTRACT_WORKERS", "4"))
CH_API_BASE_URL = "https://api.company-information.service.gov.uk"
CH_DOCUMENT_API_BASE_URL = "https://document-api.company-information.service.gov.uk"

# ───────────────────────────────────────────────────────────────
# Companies-House API tunables – used by ch_pipeline & ch_api_utils
# ----------------------------------------------------------------

CH_API_KEY: str | None = os.getenv("CH_API_KEY")           # ← set in .env or CI
CH_API_MAX_RETRY: int = int(os.getenv("CH_API_MAX_RETRY", 3))
CH_API_BACKOFF_FACTOR: float = float(os.getenv("CH_API_BACKOFF_FACTOR", 0.6))
CH_API_RETRY_BACKOFF_FACTOR = CH_API_BACKOFF_FACTOR   # ← legacy alias ✔
CH_API_TIMEOUT: int = int(os.getenv("CH_API_TIMEOUT", 30))  # seconds
CH_API_USER_AGENT: str = os.getenv(
    "CH_API_USER_AGENT",
    "StrategicCounselGen3/0.1 (+https://example.com)",
)

# Retry-forcelist defined earlier; keep alias for legacy import spellings
CH_API_RETRY_STATUS_FORCELIST: list[int] = [429, 500, 502, 503, 504]
CH_API_RETRY_STATUS_FORLIST = CH_API_RETRY_STATUS_FORCELIST
# ───────────────────────────────────────────────────────────────



# --- Protocol Text Fallback ---
# This will be the default. app.py will try to load strategic_protocols.txt
# and can update this value if successful.
PROTO_TEXT_FALLBACK = "You are a helpful AI assistant. Please provide concise and factual responses."
# These will be updated by app.py after attempting to load strategic_protocols.txt
LOADED_PROTO_PATH_NAME = "strategic_protocols.txt (Not loaded yet or using fallback)"
LOADED_PROTO_TEXT = PROTO_TEXT_FALLBACK


# --- AWS Pricing (relevant if Textract is used) ---
AWS_PRICING_CONFIG = {
    "textract_per_page": 0.0015, # Cost per page for Textract Detect Document Text
    "s3_put_request_per_pdf_to_textract": 0.000005, # S3 Standard PUT request cost
    "usd_to_gbp_exchange_rate": os.getenv("USD_TO_GBP_RATE", 0.80) # Exchange rate
}

# --- Initialize API Clients ---
_openai_client: Optional[openai.OpenAI] = None
_ch_session: Optional[requests.Session] = None
_ch_api_key_used: Optional[str] = None # To track which key the current CH session is using
# Gemini models are initialized on demand via get_gemini_model

# Flag to ensure Textract (and S3) clients are initialized only once if needed globally
# aws_textract_utils.py handles its own client initialization.
# This flag is for other potential global AWS service clients if added here.
_aws_clients_globally_initialized = False
CH_PIPELINE_TEXTRACT_FLAG = False # Will be updated by app.py based on aws_textract_utils availability


def get_openai_client() -> Optional[openai.OpenAI]:
    global _openai_client
    if not _openai_client:
        if OPENAI_API_KEY:
            try:
                _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialized.")
            except Exception as e_openai_init:
                logger.error(f"Error setting up OpenAI client: {e_openai_init}")
                _openai_client = None # Ensure it's None on failure
        else:
            logger.warning("OPENAI_API_KEY not found. OpenAI calls will fail.")
            _openai_client = None
    return _openai_client


def check_openai_model(model_name: str) -> Tuple[bool, str]:
    """Verify that the requested OpenAI model can be retrieved."""
    client = get_openai_client()
    if not client:
        msg = "OpenAI client not available."
        logger.error(msg)
        return False, msg

    try:
        client.models.retrieve(model_name)
        logger.info(f"OpenAI model '{model_name}' is available.")
        return True, ""
    except Exception as e:
        msg = (
            f"Model '{model_name}' not found; check OPENAI_MODEL or run client.models.list()"
        )
        logger.error(msg)
        return False, msg


_google_drive_service: Optional[Any] = None

def get_google_drive_service() -> Optional[Any]:
    """Return an authorized Google Drive service instance or ``None`` if unavailable."""
    global _google_drive_service
    if _google_drive_service:
        return _google_drive_service

    if not ENABLE_GOOGLE_DRIVE_INTEGRATION:
        logger.info("Google Drive integration disabled via configuration.")
        return None

    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request

        creds = None
        token_path = Path(GOOGLE_TOKEN_FILE)
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), GOOGLE_API_SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    GOOGLE_CLIENT_SECRET_FILE, GOOGLE_API_SCOPES
                )
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())

        _google_drive_service = build("drive", "v3", credentials=creds)
        logger.info("Google Drive service initialised.")
        return _google_drive_service
    except Exception as e:
        logger.error(f"Failed to initialise Google Drive service: {e}")
        return None

def get_ch_session(api_key_override: Optional[str] = None) -> requests.Session:
    """
    Returns a Companies House API session.
    Uses api_key_override if provided, otherwise CH_API_KEY from environment.
    Reinitializes session if the key changes or if session doesn't exist.
    """
    global _ch_session, _ch_api_key_used
    
    key_to_use = api_key_override if api_key_override else CH_API_KEY

    if not key_to_use:
        # If there's an existing session (even if it had a key), and now no key is provided,
        # re-initialize a key-less session.
        if _ch_session is None or _ch_api_key_used is not None:
            _ch_session = requests.Session()
            _ch_api_key_used = None # Mark that this session has no key
            logger.info("Initialized Companies House session without an API key (rate limits may apply).")
        return _ch_session

    # If session doesn't exist, or if the key to be used is different from the one the current session uses
    if _ch_session is None or _ch_api_key_used != key_to_use:
        logger.info(f"Initializing/Re-initializing Companies House session with API key.")
        _ch_session = requests.Session()
        _ch_session.auth = (key_to_use, "")
        # Standard headers for CH API
        _ch_session.headers.update({'User-Agent': 'StrategicCounselApp/1.0'})
        _ch_api_key_used = key_to_use
        logger.info(f"Companies House session configured with API key (ending ...{key_to_use[-4:] if len(key_to_use) > 4 else '****'}).")
    return _ch_session


_gemini_sdk_configured = False

def get_gemini_model(model_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Initialize and return a Gemini GenerativeModel.

    Returns a tuple ``(model, error_message)``. ``model`` will be ``None`` if
    initialization failed, in which case ``error_message`` contains details
    about the failure.
    """
    global _gemini_sdk_configured

    if not genai:
        msg = "google-generativeai library not installed. Gemini models will not be available."
        logger.warning(msg)
        return None, msg

    if not GEMINI_API_KEY:
        msg = "GEMINI_API_KEY not found. Gemini models will fail."
        logger.warning(msg)
        return None, msg

    if not _gemini_sdk_configured:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Google Generative AI SDK configured with API key.")
            _gemini_sdk_configured = True
        except Exception as e_gemini_config:
            msg = f"Error configuring Google Generative AI SDK: {e_gemini_config}"
            logger.error(msg)
            return None, msg

    try:
        model = genai.GenerativeModel(model_name)
        logger.info(f"Gemini model '{model_name}' initialized.")
        return model, None
    except Exception as e:
        if GoogleAPICoreExceptions and isinstance(e, GoogleAPICoreExceptions.NotFound):
            msg = (
                f"Model '{model_name}' not found; check GEMINI_MODEL_FOR_SUMMARIES "
                "or run genai.list_models()"
            )
            logger.error(msg)
            return None, msg

        msg = f"Failed to initialize Gemini model '{model_name}': {e}"
        logger.error(msg, exc_info=True)
        return None, msg

# Base path for the application (useful for file operations in app.py and other modules)
APP_BASE_PATH = Path(__file__).resolve().parent
