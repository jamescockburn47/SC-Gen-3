import pytest
from pathlib import Path
from aws_textract_utils import _initialize_aws_clients, perform_textract_ocr
import boto3

def test_initialize_aws_clients():
    """Test AWS client initialization"""
    assert _initialize_aws_clients() is True

def test_perform_textract_ocr_bytes(sample_pdf_file):
    """Test Textract OCR on a sample PDF (bytes input)"""
    _initialize_aws_clients()
    with open(sample_pdf_file, "rb") as f:
        pdf_bytes = f.read()
    text, pages, error = perform_textract_ocr(pdf_bytes, company_no_for_logging="TEST_COMPANY")
    assert isinstance(text, str)
    assert isinstance(pages, int)
    # error may be None or a string depending on AWS setup

def test_perform_textract_ocr(sample_pdf_file, test_env):
    """Test Textract OCR on a sample PDF"""
    session, textract_client = initialize_aws_clients()
    
    # Upload the PDF to S3
    s3_client = session.client('s3')
    bucket = test_env['S3_TEXTRACT_BUCKET']
    key = f"test/{sample_pdf_file.name}"
    
    try:
        s3_client.upload_file(str(sample_pdf_file), bucket, key)
        
        # Perform OCR
        text, pages, error = perform_textract_ocr(textract_client, bucket, key)
        assert text is not None
        assert isinstance(text, str)
        assert pages > 0
        assert error is None
        
    finally:
        # Clean up
        try:
            s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            pytest.fail(f"Failed to clean up S3: {e}")

def test_perform_textract_ocr_invalid_file(scratch_dir, test_env):
    """Test Textract OCR with an invalid file"""
    session, textract_client = initialize_aws_clients()
    s3_client = session.client('s3')
    bucket = test_env['S3_TEXTRACT_BUCKET']
    
    # Create an invalid PDF
    invalid_file = scratch_dir / "invalid.pdf"
    invalid_file.write_text("This is not a valid PDF")
    key = f"test/{invalid_file.name}"
    
    try:
        s3_client.upload_file(str(invalid_file), bucket, key)
        
        # Attempt OCR
        text, pages, error = perform_textract_ocr(textract_client, bucket, key)
        assert text is None
        assert pages == 0
        assert error is not None
        
    finally:
        # Clean up
        try:
            s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            pytest.fail(f"Failed to clean up S3: {e}")

def test_perform_textract_ocr_nonexistent_file(test_env):
    """Test Textract OCR with a nonexistent file"""
    session, textract_client = initialize_aws_clients()
    bucket = test_env['S3_TEXTRACT_BUCKET']
    key = "nonexistent.pdf"
    
    text, pages, error = perform_textract_ocr(textract_client, bucket, key)
    assert text is None
    assert pages == 0
    assert error is not None
    assert "NoSuchKey" in str(error) 