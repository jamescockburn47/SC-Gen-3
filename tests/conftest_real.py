import pytest
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

@pytest.fixture(scope="session")
def test_env():
    """Fixture to ensure test environment is properly configured"""
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
        pytest.skip(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {var: os.getenv(var) for var in required_vars}

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture to provide test data directory"""
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

@pytest.fixture(scope="session")
def sample_pdf_file(test_data_dir):
    """Fixture to provide a sample PDF file for testing"""
    pdf_path = test_data_dir / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF file not found. Please add a sample.pdf file to tests/test_data/")
    return pdf_path

@pytest.fixture(scope="session")
def sample_docx_file(test_data_dir):
    """Fixture to provide a sample DOCX file for testing"""
    docx_path = test_data_dir / "sample.docx"
    if not docx_path.exists():
        pytest.skip("Sample DOCX file not found. Please add a sample.docx file to tests/test_data/")
    return docx_path

@pytest.fixture(scope="session")
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

@pytest.fixture(scope="session")
def scratch_dir():
    """Fixture to provide a scratch directory for test outputs"""
    scratch = Path(__file__).parent.parent / "scratch" / "tests"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch

@pytest.fixture(autouse=True)
def cleanup_scratch(scratch_dir):
    """Fixture to clean up scratch directory after each test"""
    yield
    for file in scratch_dir.glob("*"):
        try:
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                import shutil
                shutil.rmtree(file)
        except Exception as e:
            logger.warning(f"Failed to clean up {file}: {e}") 