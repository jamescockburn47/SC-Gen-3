import pytest
import os
import sys
from pathlib import Path
import logging
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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