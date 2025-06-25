# config.py

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Any, Tuple
from functools import lru_cache, wraps
import time

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

# --- Performance Optimization: Caching Decorator ---
def timed_cache(timeout_seconds: int = 300):
    """Cache with timeout for expensive operations."""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < timeout_seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            
            # Clean old entries
            if len(cache) > 100:  # Limit cache size
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# Local directory used by ch_pipeline to cache downloaded documents
APPLICATION_SCRATCH_DIR = Path(
    os.getenv("APPLICATION_SCRATCH_DIR", Path(__file__).resolve().parent / "scratch")
)
APPLICATION_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Optimize logging configuration
class OptimizedFormatter(logging.Formatter):
    """Optimized formatter that reduces overhead for frequent logging."""
    
    def format(self, record):
        # Cache formatted time to reduce overhead
        if not hasattr(self, '_cached_time') or self._cached_time[0] != int(record.created):
            self._cached_time = (int(record.created), self.formatTime(record))
        
        record.asctime = self._cached_time[1]
        return super().format(record)

# Configure logging with optimized formatter
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Use optimized formatter for file handlers
logger = logging.getLogger("strategic_counsel_app")
for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setFormatter(OptimizedFormatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        ))

# Suppress verbose logs from libraries
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Suppress specific warnings - Make PyPDF2 import conditional
try:
    from PyPDF2 import errors as PyPDF2Errors
    warnings.filterwarnings("ignore", message="incorrect startxref pointer.*")
    warnings.filterwarnings("ignore", category=PyPDF2Errors.PdfReadWarning)
except ImportError:
    logger.warning("PyPDF2 not available. PDF-related warnings will not be suppressed.")
    PyPDF2Errors = None

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CH_API_KEY = os.getenv("CH_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_DEFAULT = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")
S3_TEXTRACT_BUCKET = os.getenv("S3_TEXTRACT_BUCKET")

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
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL_FOR_SUMMARIES", "gemini-1.5-flash-latest")
GEMINI_MODEL_FOR_PROTOCOL_CHECK = os.getenv("GEMINI_MODEL_FOR_PROTOCOL_CHECK", "gemini-1.5-flash-latest")
GEMINI_2_5_PRO_MODEL = "gemini-2.5-pro-latest"
PROTOCOL_CHECK_MODEL_PROVIDER = os.getenv("PROTOCOL_CHECK_MODEL_PROVIDER", "gemini")

# --- Application Constants ---
MIN_MEANINGFUL_TEXT_LEN = 200
MAX_DOCS_TO_PROCESS_PER_COMPANY = int(os.getenv("MAX_DOCS_PER_COMPANY_PIPELINE", "20"))
MAX_TEXTRACT_WORKERS = int(os.getenv("MAX_TEXTRACT_WORKERS", "4"))
CH_API_BASE_URL = "https://api.company-information.service.gov.uk"
CH_API_DOCUMENT_API_BASE_URL = "https://document-api.company-information.service.gov.uk"
# Backward compatibility
CH_DOCUMENT_API_BASE_URL = CH_API_DOCUMENT_API_BASE_URL

# --- Companies House API Configuration ---
CH_API_MAX_RETRY: int = int(os.getenv("CH_API_MAX_RETRY", 3))
CH_API_BACKOFF_FACTOR: float = float(os.getenv("CH_API_BACKOFF_FACTOR", 0.6))
CH_API_RETRY_BACKOFF_FACTOR = CH_API_BACKOFF_FACTOR
CH_API_TIMEOUT: int = int(os.getenv("CH_API_TIMEOUT", 30))
CH_API_USER_AGENT: str = os.getenv(
    "CH_API_USER_AGENT",
    "StrategicCounselGen3/0.1 (+https://example.com)",
)
USER_AGENT = CH_API_USER_AGENT

CH_API_RETRY_STATUS_FORCELIST: list[int] = [429, 500, 502, 503, 504]
CH_API_RETRY_STATUS_FORLIST = CH_API_RETRY_STATUS_FORCELIST

# --- Protocol Text Configuration ---
PROTO_TEXT_FALLBACK = "You are a helpful AI assistant. Please provide concise and factual responses."
LOADED_PROTO_PATH_NAME = "strategic_protocols.txt (Not loaded yet or using fallback)"
LOADED_PROTO_TEXT = PROTO_TEXT_FALLBACK

# --- AWS Pricing Configuration ---
AWS_PRICING_CONFIG = {
    "textract_per_page": 0.0015,
    "s3_put_request_per_pdf_to_textract": 0.000005,
    "usd_to_gbp_exchange_rate": os.getenv("USD_TO_GBP_RATE", 0.80)
}

# --- Optimized API Client Management ---
_openai_client: Optional[openai.OpenAI] = None
_ch_session: Optional[requests.Session] = None
_ch_api_key_used: Optional[str] = None
_aws_clients_globally_initialized = False
CH_PIPELINE_TEXTRACT_FLAG = False

@lru_cache(maxsize=1)
def get_openai_client() -> Optional[openai.OpenAI]:
    """Cached OpenAI client initialization."""
    global _openai_client
    if not _openai_client:
        if OPENAI_API_KEY:
            try:
                _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialized and cached.")
            except Exception as e_openai_init:
                logger.error(f"Error setting up OpenAI client: {e_openai_init}")
                _openai_client = None
        else:
            logger.warning("OPENAI_API_KEY not found. OpenAI calls will fail.")
            _openai_client = None
    return _openai_client

@timed_cache(timeout_seconds=3600)  # Cache for 1 hour
def check_openai_model(model_name: str) -> Tuple[bool, str]:
    """Cached OpenAI model verification."""
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
        msg = f"Model '{model_name}' not found; check OPENAI_MODEL or run client.models.list()"
        logger.error(msg)
        return False, msg

# --- Google Drive Service with Caching ---
_google_drive_service: Optional[Any] = None

@lru_cache(maxsize=1)
def get_google_drive_service() -> Optional[Any]:
    """Cached Google Drive service initialization."""
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
        logger.info("Google Drive service initialised and cached.")
        return _google_drive_service
    except Exception as e:
        logger.error(f"Failed to initialise Google Drive service: {e}")
        return None

# --- Optimized CH Session Management ---
def get_ch_session(api_key_override: Optional[str] = None) -> requests.Session:
    """Optimized Companies House API session with connection pooling."""
    global _ch_session, _ch_api_key_used
    
    key_to_use = api_key_override if api_key_override else CH_API_KEY

    if not key_to_use:
        if _ch_session is None or _ch_api_key_used is not None:
            _ch_session = requests.Session()
            # Optimize connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=3
            )
            _ch_session.mount('http://', adapter)
            _ch_session.mount('https://', adapter)
            _ch_api_key_used = None
            logger.info("Initialized optimized Companies House session without API key.")
        return _ch_session

    if _ch_session is None or _ch_api_key_used != key_to_use:
        logger.info("Initializing/Re-initializing optimized Companies House session with API key.")
        _ch_session = requests.Session()
        _ch_session.auth = (key_to_use, "")
        
        # Optimize connection pooling and retries
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504]
            )
        )
        _ch_session.mount('http://', adapter)
        _ch_session.mount('https://', adapter)
        
        _ch_session.headers.update({
            'User-Agent': CH_API_USER_AGENT,
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        _ch_api_key_used = key_to_use
        logger.info(f"Optimized Companies House session configured with API key.")
    return _ch_session

# --- Optimized Gemini Model Management ---
_gemini_sdk_configured = False
_gemini_models_cache = {}

@timed_cache(timeout_seconds=1800)  # Cache for 30 minutes
def get_gemini_model(model_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Cached Gemini model initialization."""
    global _gemini_sdk_configured, _gemini_models_cache

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

    # Check cache first
    if model_name in _gemini_models_cache:
        return _gemini_models_cache[model_name], None

    try:
        model = genai.GenerativeModel(model_name)
        _gemini_models_cache[model_name] = model
        logger.info(f"Gemini model '{model_name}' initialized and cached.")
        return model, None
    except Exception as e:
        if GoogleAPICoreExceptions and isinstance(e, GoogleAPICoreExceptions.NotFound):
            msg = f"Model '{model_name}' not found; check GEMINI_MODEL_FOR_SUMMARIES or run genai.list_models()"
            logger.error(msg)
            return None, msg

        msg = f"Failed to initialize Gemini model '{model_name}': {e}"
        logger.error(msg, exc_info=True)
        return None, msg

# Base path for the application
APP_BASE_PATH = Path(__file__).resolve().parent
