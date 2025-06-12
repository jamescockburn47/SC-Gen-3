import pytest
from unittest.mock import Mock, patch
from app_utils import (
    summarise_with_title,
    fetch_url_content,
    find_company_number,
    extract_text_from_uploaded_file
)

@pytest.fixture
def mock_requests():
    with patch('requests.get') as mock_get:
        yield mock_get

@pytest.fixture
def mock_textract():
    with patch('app_utils.extract_text_with_textract') as mock:
        yield mock

def test_summarise_with_title():
    # Test successful summarization
    result = summarise_with_title("Test content", "Test Title")
    assert isinstance(result, str)
    assert len(result) > 0

def test_fetch_url_content(mock_requests):
    # Test successful URL content fetch
    mock_response = Mock()
    mock_response.text = "Test content"
    mock_response.status_code = 200
    mock_requests.return_value = mock_response

    result = fetch_url_content("http://test.com")
    assert result == "Test content"

def test_fetch_url_content_error(mock_requests):
    # Test error handling
    mock_requests.side_effect = Exception("Network Error")
    
    result = fetch_url_content("http://test.com")
    assert result is None

def test_find_company_number():
    # Test company number extraction
    text = "Company number: 12345678"
    result = find_company_number(text)
    assert result == "12345678"

def test_find_company_number_not_found():
    # Test when company number is not found
    text = "No company number here"
    result = find_company_number(text)
    assert result is None

def test_extract_text_from_uploaded_file(mock_textract):
    # Test successful text extraction
    mock_textract.return_value = "Extracted text"
    
    mock_file = Mock()
    mock_file.name = "test.pdf"
    
    result = extract_text_from_uploaded_file(mock_file)
    assert result == "Extracted text"

def test_extract_text_from_uploaded_file_error(mock_textract):
    # Test error handling
    mock_textract.side_effect = Exception("Extraction Error")
    
    mock_file = Mock()
    mock_file.name = "test.pdf"
    
    result = extract_text_from_uploaded_file(mock_file)
    assert result is None 