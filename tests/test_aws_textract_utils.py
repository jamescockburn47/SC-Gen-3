import pytest
from unittest.mock import Mock, patch
from aws_textract_utils import (
    perform_textract_ocr,
    get_textract_cost_estimation,
    _initialize_aws_clients
)

@pytest.fixture
def mock_boto3():
    with patch('boto3.Session') as mock_session:
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        yield mock_client

def test_initialize_aws_clients(mock_boto3):
    # Test successful initialization
    result = _initialize_aws_clients()
    assert result is True

def test_initialize_aws_clients_error():
    # Test initialization error
    with patch('boto3.Session', side_effect=Exception("AWS Error")):
        result = _initialize_aws_clients()
        assert result is False

def test_perform_textract_ocr(mock_boto3):
    # Test successful OCR
    mock_boto3.start_document_text_detection.return_value = {"JobId": "test-job"}
    mock_boto3.get_document_text_detection.return_value = {
        "JobStatus": "SUCCEEDED",
        "Blocks": [{"BlockType": "LINE", "Text": "Test text"}],
        "DocumentMetadata": {"Pages": 1}
    }
    
    result, pages, error = perform_textract_ocr(b"Test PDF content")
    assert result == "Test text"
    assert pages == 1
    assert error is None

def test_perform_textract_ocr_error(mock_boto3):
    # Test OCR error
    mock_boto3.start_document_text_detection.side_effect = Exception("OCR Error")
    
    result, pages, error = perform_textract_ocr(b"Test PDF content")
    assert result == ""
    assert pages == 0
    assert "OCR Error" in error

def test_get_textract_cost_estimation():
    # Test cost estimation
    result = get_textract_cost_estimation(10, 2)
    assert isinstance(result, dict)
    assert "total_cost" in result
    assert "cost_per_page" in result 