import pytest
from unittest.mock import Mock, patch
from text_extraction_utils import (
    extract_text_from_document,
    OCRHandlerType
)

@pytest.fixture
def mock_pdf():
    with patch('PyPDF2.PdfReader') as mock:
        yield mock

@pytest.fixture
def mock_textract():
    with patch('text_extraction_utils.extract_text_with_textract') as mock:
        yield mock

def test_extract_text_from_document_pdf(mock_pdf):
    mock_pdf.return_value.pages = [Mock(extract_text=lambda: "PDF text")]
    result = extract_text_from_document("test.pdf", b"PDF content")
    assert result == "PDF text"

def test_extract_text_from_document_textract(mock_textract):
    mock_textract.return_value = "Textract text"
    result = extract_text_from_document("test.pdf", b"PDF content", OCRHandlerType.TEXTRACT)
    assert result == "Textract text"

def test_extract_text_from_document_error():
    result = extract_text_from_document("test.xyz", b"Invalid content")
    assert result is None

def test_ocr_handler_type_enum():
    assert hasattr(OCRHandlerType, '__call__') 