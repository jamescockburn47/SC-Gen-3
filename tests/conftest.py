import pytest
import os
import sys
from pathlib import Path
import logging
from unittest.mock import MagicMock, patch
import streamlit as st
from dotenv import load_dotenv

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def load_env_vars():
    """Fixture to ensure environment variables are loaded for each test"""
    # Verify critical environment variables are present
    required_vars = [
        "OPENAI_API_KEY",
        "CH_API_KEY",
        "GEMINI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "S3_TEXTRACT_BUCKET"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.fail(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    yield

@pytest.fixture
def mock_env_vars():
    """Fixture to set up mock environment variables for testing"""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "CH_API_KEY": "test-ch-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "AWS_ACCESS_KEY_ID": "test-aws-key",
        "AWS_SECRET_ACCESS_KEY": "test-aws-secret",
        "AWS_DEFAULT_REGION": "eu-west-2",
        "S3_TEXTRACT_BUCKET": "test-bucket",
        "LOG_LEVEL": "DEBUG"
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def mock_openai_client():
    """Fixture to mock OpenAI client"""
    with patch("openai.OpenAI") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

@pytest.fixture
def mock_gemini_model():
    """Fixture to mock Gemini model"""
    with patch("google.generativeai.GenerativeModel") as mock:
        model = MagicMock()
        mock.return_value = model
        yield model

@pytest.fixture
def mock_ch_session():
    """Fixture to mock Companies House API session"""
    with patch("requests.Session") as mock:
        session = MagicMock()
        mock.return_value = session
        yield session

@pytest.fixture
def test_data_dir():
    """Fixture to provide test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_pdf_file(test_data_dir):
    """Fixture to provide a sample PDF file for testing"""
    return test_data_dir / "sample.pdf"

@pytest.fixture
def sample_docx_file(test_data_dir):
    """Fixture to provide a sample DOCX file for testing"""
    return test_data_dir / "sample.docx"

@pytest.fixture
def sample_company_data():
    """Fixture to provide sample company data for testing"""
    return {
        "company_number": "12345678",
        "company_name": "Test Company Ltd",
        "registered_office_address": {
            "address_line_1": "123 Test Street",
            "postal_code": "TE1 1ST"
        },
        "date_of_creation": "2020-01-01",
        "type": "ltd"
    }

@pytest.fixture(autouse=True)
def init_streamlit_session_state():
    """Fixture to initialize Streamlit session state for each test"""
    # Only initialize if not already present
    if not hasattr(st, 'session_state') or not hasattr(st.session_state, '__getitem__'):
        try:
            from streamlit.runtime.state import SafeSessionState
            st.session_state = SafeSessionState()
        except Exception:
            # Fallback for older/newer Streamlit versions
            st.session_state = dict()
    
    # Set required keys with values from environment variables where applicable
    st.session_state.setdefault('current_topic', 'test_topic')
    st.session_state.setdefault('ocr_method', 'aws')
    st.session_state.setdefault('ocr_method_radio', 0)
    st.session_state.setdefault('processed_summaries', [])
    st.session_state.setdefault('selected_summary_texts', [])
    st.session_state.setdefault('document_processing_complete', True)
    st.session_state.setdefault('ch_analysis_summaries_for_injection', [])
    st.session_state.setdefault('ch_available_documents', [])
    st.session_state.setdefault('ch_document_selection', {})
    st.session_state.setdefault('ch_company_profiles_map', {})
    
    # Set API keys from environment variables
    st.session_state.setdefault('openai_api_key', os.getenv('OPENAI_API_KEY'))
    st.session_state.setdefault('ch_api_key', os.getenv('CH_API_KEY'))
    st.session_state.setdefault('gemini_api_key', os.getenv('GEMINI_API_KEY'))
    st.session_state.setdefault('aws_access_key', os.getenv('AWS_ACCESS_KEY_ID'))
    st.session_state.setdefault('aws_secret_key', os.getenv('AWS_SECRET_ACCESS_KEY'))
    st.session_state.setdefault('aws_region', os.getenv('AWS_DEFAULT_REGION'))
    st.session_state.setdefault('s3_bucket', os.getenv('S3_TEXTRACT_BUCKET'))

@pytest.fixture
def mock_aws_credentials():
    """Fixture to provide AWS credentials for testing"""
    return {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'region_name': os.getenv('AWS_DEFAULT_REGION')
    }

@pytest.fixture
def mock_ch_credentials():
    """Fixture to provide Companies House credentials for testing"""
    return {
        'api_key': os.getenv('CH_API_KEY')
    }

@pytest.fixture
def mock_ai_credentials():
    """Fixture to provide AI service credentials for testing"""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'gemini_api_key': os.getenv('GEMINI_API_KEY')
    } 