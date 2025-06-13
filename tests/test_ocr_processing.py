import pytest
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import time

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aws_textract_utils import (
    initialize_textract_aws_clients,
    perform_textract_ocr,
    process_textract_response
)
from google_drive_utils import (
    extract_text_from_google_doc,
    download_file_bytes
)
from ocr_processing import process_document_ocr, extract_text_aws, extract_text_tesseract

@pytest.fixture
def mock_aws_clients():
    """Fixture to mock AWS clients"""
    with patch('boto3.client') as mock_client:
        # Mock Textract client
        textract_client = MagicMock()
        textract_client.start_document_analysis.return_value = {
            'JobId': 'test-job-id'
        }
        textract_client.get_document_analysis.return_value = {
            'JobStatus': 'SUCCEEDED',
            'Blocks': [
                {
                    'BlockType': 'PAGE',
                    'Text': 'Test page content'
                },
                {
                    'BlockType': 'LINE',
                    'Text': 'Test line content'
                }
            ]
        }
        
        # Mock S3 client
        s3_client = MagicMock()
        s3_client.upload_fileobj.return_value = None
        s3_client.delete_object.return_value = None
        
        # Set up the mock to return different clients
        def get_client(service_name, *args, **kwargs):
            if service_name == 'textract':
                return textract_client
            elif service_name == 's3':
                return s3_client
            return MagicMock()
        
        mock_client.side_effect = get_client
        yield {
            'textract': textract_client,
            's3': s3_client
        }

@pytest.fixture
def mock_google_service():
    """Fixture to mock Google Drive service"""
    with patch('google.oauth2.service_account.Credentials.from_service_account_file') as mock_creds:
        with patch('googleapiclient.discovery.build') as mock_build:
            service = MagicMock()
            mock_build.return_value = service
            yield service

@pytest.fixture
def test_data_dir():
    """Fixture to provide test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_pdf_file(test_data_dir):
    """Fixture to provide a sample PDF file for testing"""
    return test_data_dir / "sample.pdf"

def test_aws_textract_initialization(mock_aws_clients):
    """Test AWS Textract client initialization"""
    # Test successful initialization
    result = initialize_textract_aws_clients()
    assert result is True
    mock_aws_clients['textract'].assert_called_once()
    mock_aws_clients['s3'].assert_called_once()

    # Test initialization failure
    with patch('boto3.client', side_effect=ClientError({}, 'Operation')):
        result = initialize_textract_aws_clients()
        assert result is False

def test_textract_ocr_processing(mock_aws_clients):
    """Test Textract OCR processing"""
    # Test successful OCR processing
    result = perform_textract_ocr(
        file_path="test.pdf",
        bucket_name="test-bucket",
        textract_client=mock_aws_clients['textract'],
        s3_client=mock_aws_clients['s3']
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Test page content" in result
    assert "Test line content" in result

    # Test OCR processing with invalid file
    with patch('builtins.open', side_effect=FileNotFoundError):
        result = perform_textract_ocr(
            file_path="nonexistent.pdf",
            bucket_name="test-bucket",
            textract_client=mock_aws_clients['textract'],
            s3_client=mock_aws_clients['s3']
        )
        assert result is None

def test_textract_response_processing():
    """Test Textract response processing"""
    # Test processing valid response
    response = {
        'Blocks': [
            {
                'BlockType': 'PAGE',
                'Text': 'Page 1'
            },
            {
                'BlockType': 'LINE',
                'Text': 'Line 1'
            },
            {
                'BlockType': 'WORD',
                'Text': 'Word 1'
            }
        ]
    }
    result = process_textract_response(response)
    assert isinstance(result, str)
    assert "Page 1" in result
    assert "Line 1" in result

    # Test processing empty response
    result = process_textract_response({'Blocks': []})
    assert result == ""

    # Test processing invalid response
    result = process_textract_response({})
    assert result == ""

def test_google_drive_text_extraction(mock_google_service):
    """Test Google Drive text extraction"""
    # Mock successful text extraction
    mock_google_service.files().export().execute.return_value = b"Test document content"
    
    result = extract_text_from_google_doc(mock_google_service, "test-doc-id")
    assert isinstance(result, str)
    assert result == "Test document content"

    # Test error handling
    mock_google_service.files().export().execute.side_effect = Exception("API Error")
    result = extract_text_from_google_doc(mock_google_service, "test-doc-id")
    assert result is None

def test_google_drive_file_download(mock_google_service):
    """Test Google Drive file download"""
    # Mock successful file download
    mock_google_service.files().get_media().execute.return_value = b"Test file content"
    
    result = download_file_bytes(mock_google_service, "test-file-id")
    assert isinstance(result, bytes)
    assert result == b"Test file content"

    # Test error handling
    mock_google_service.files().get_media().execute.side_effect = Exception("API Error")
    result = download_file_bytes(mock_google_service, "test-file-id")
    assert result is None

def test_ocr_error_handling(mock_aws_clients):
    """Test OCR error handling"""
    # Test S3 upload error
    mock_aws_clients['s3'].upload_fileobj.side_effect = ClientError({}, 'Operation')
    result = perform_textract_ocr(
        file_path="test.pdf",
        bucket_name="test-bucket",
        textract_client=mock_aws_clients['textract'],
        s3_client=mock_aws_clients['s3']
    )
    assert result is None

    # Test Textract job error
    mock_aws_clients['textract'].get_document_analysis.return_value = {
        'JobStatus': 'FAILED',
        'StatusMessage': 'Job failed'
    }
    result = perform_textract_ocr(
        file_path="test.pdf",
        bucket_name="test-bucket",
        textract_client=mock_aws_clients['textract'],
        s3_client=mock_aws_clients['s3']
    )
    assert result is None

def test_ocr_integration(mock_aws_clients, mock_google_service):
    """Test OCR integration with different providers"""
    # Test AWS Textract integration
    with patch('aws_textract_utils.initialize_textract_aws_clients', return_value=True):
        result = perform_textract_ocr(
            file_path="test.pdf",
            bucket_name="test-bucket",
            textract_client=mock_aws_clients['textract'],
            s3_client=mock_aws_clients['s3']
        )
        assert isinstance(result, str)
        assert len(result) > 0

    # Test Google Drive integration
    mock_google_service.files().export().execute.return_value = b"Test document content"
    result = extract_text_from_google_doc(mock_google_service, "test-doc-id")
    assert isinstance(result, str)
    assert result == "Test document content"

def test_ocr_performance(mock_aws_clients):
    """Test OCR performance with large documents"""
    # Mock large document response
    large_response = {
        'Blocks': [
            {
                'BlockType': 'PAGE',
                'Text': f'Page {i}'
            }
            for i in range(100)  # Simulate 100 pages
        ]
    }
    mock_aws_clients['textract'].get_document_analysis.return_value = large_response

    # Test processing large document
    result = perform_textract_ocr(
        file_path="large.pdf",
        bucket_name="test-bucket",
        textract_client=mock_aws_clients['textract'],
        s3_client=mock_aws_clients['s3']
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Page 0" in result
    assert "Page 99" in result

def test_aws_textract_integration(mock_aws_credentials):
    """Test AWS Textract integration with real credentials"""
    # Initialize AWS Textract client
    textract_client = boto3.client('textract', **mock_aws_credentials)
    
    # Test file
    test_file = Path("tests/test_data/sample.pdf")
    assert test_file.exists(), "Test file does not exist"
    
    # Process document with real AWS Textract
    result = extract_text_aws(test_file)
    assert result is not None, "AWS Textract processing failed"
    assert 'text' in result, "AWS Textract result missing text"
    assert len(result['text']) > 0, "AWS Textract result is empty"

def test_tesseract_processing(sample_pdf_file):
    """Test Tesseract OCR processing"""
    # Process document with Tesseract
    result = extract_text_tesseract(sample_pdf_file)
    assert result is not None, "Tesseract processing failed"
    assert 'text' in result, "Tesseract result missing text"
    assert len(result['text']) > 0, "Tesseract result is empty"

def test_ocr_method_selection(mock_aws_credentials):
    """Test OCR method selection with real services"""
    # Test file
    test_file = Path("tests/test_data/sample.pdf")
    
    # Test AWS method
    result_aws = process_document_ocr(test_file, method='aws')
    assert result_aws is not None, "AWS OCR processing failed"
    
    # Test Tesseract method
    result_tesseract = process_document_ocr(test_file, method='tesseract')
    assert result_tesseract is not None, "Tesseract OCR processing failed"
    
    # Compare results
    assert len(result_aws['text']) > 0, "AWS OCR result is empty"
    assert len(result_tesseract['text']) > 0, "Tesseract OCR result is empty"

def test_error_handling(mock_aws_credentials):
    """Test error handling with real services"""
    # Test with invalid file
    invalid_file = Path("nonexistent.pdf")
    result = process_document_ocr(invalid_file, method='aws')
    assert result is None, "Should handle invalid file gracefully"
    
    # Test with invalid method
    test_file = Path("tests/test_data/sample.pdf")
    result = process_document_ocr(test_file, method='invalid_method')
    assert result is None, "Should handle invalid method gracefully"

def test_performance(mock_aws_credentials):
    """Test OCR performance with real services"""
    test_file = Path("tests/test_data/sample.pdf")
    
    # Test AWS Textract performance
    start_time = time.time()
    result = process_document_ocr(test_file, method='aws')
    aws_time = time.time() - start_time
    assert aws_time < 30, "AWS Textract processing took too long"
    
    # Test Tesseract performance
    start_time = time.time()
    result = process_document_ocr(test_file, method='tesseract')
    tesseract_time = time.time() - start_time
    assert tesseract_time < 60, "Tesseract processing took too long"

def test_large_document_processing(mock_aws_credentials):
    """Test processing of large documents with real services"""
    # Test with a large PDF file
    large_file = Path("tests/test_data/large_sample.pdf")
    if large_file.exists():
        result = process_document_ocr(large_file, method='aws')
        assert result is not None, "Failed to process large document"
        assert 'text' in result, "Large document processing result missing text"
        assert len(result['text']) > 0, "Large document processing result is empty"

def test_multiple_document_processing(mock_aws_credentials):
    """Test processing multiple documents with real services"""
    # Process multiple documents
    test_files = [
        Path("tests/test_data/sample.pdf"),
        Path("tests/test_data/sample2.pdf"),
        Path("tests/test_data/sample3.pdf")
    ]
    
    for test_file in test_files:
        if test_file.exists():
            result = process_document_ocr(test_file, method='aws')
            assert result is not None, f"Failed to process {test_file}"
            assert 'text' in result, f"Result missing text for {test_file}"
            assert len(result['text']) > 0, f"Empty result for {test_file}" 