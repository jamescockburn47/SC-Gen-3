import pytest
from pathlib import Path
from text_extraction_utils import extract_text_from_document
import PyPDF2

def test_extract_text_from_document_pdf(sample_pdf_file):
    """Test PDF text extraction using PyPDF2"""
    text, pages, error = extract_text_from_document(sample_pdf_file)
    assert text is not None
    assert isinstance(text, str)
    assert pages > 0
    assert error is None

def test_extract_text_from_document_invalid_file(scratch_dir):
    """Test text extraction with an invalid file"""
    invalid_file = scratch_dir / "invalid.pdf"
    invalid_file.write_text("This is not a valid PDF")
    text, pages, error = extract_text_from_document(invalid_file)
    assert text is None
    assert pages == 0
    assert error is not None
    assert "Failed to extract text" in str(error)

def test_extract_text_from_document_empty_pdf(scratch_dir):
    """Test text extraction with an empty PDF"""
    empty_pdf = scratch_dir / "empty.pdf"
    # Create an empty PDF
    pdf = PyPDF2.PdfWriter()
    with open(empty_pdf, "wb") as f:
        pdf.write(f)
    text, pages, error = extract_text_from_document(empty_pdf)
    assert text == ""
    assert pages == 0
    assert error is None 