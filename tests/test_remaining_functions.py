import pytest
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path
import json
import datetime as dt

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app_utils import (
    process_document,
    extract_text_from_pdf,
    extract_text_from_docx,
    summarize_with_title,
    format_date,
    clean_text,
    validate_company_number,
    get_file_extension,
    is_valid_file_type,
    save_uploaded_file,
    load_company_data,
    save_company_data,
    generate_report,
    format_currency,
    calculate_percentage,
    validate_date,
    parse_date,
    format_address,
    validate_postcode,
    format_phone_number,
    validate_email,
    format_name,
    validate_vat_number,
    format_company_number,
    validate_uk_address,
    format_uk_address,
    validate_uk_phone,
    format_uk_phone,
    validate_uk_email,
    format_uk_email,
    validate_uk_vat,
    format_uk_vat
)

from ai_utils import (
    gpt_summarise_ch_docs,
    gemini_summarise_ch_docs,
    get_improved_prompt,
    check_protocol_compliance,
    analyze_text,
    extract_key_points,
    generate_summary,
    improve_prompt,
    validate_prompt,
    format_ai_response,
    process_ai_error,
    handle_rate_limit,
    retry_on_failure,
    validate_ai_response,
    format_ai_error,
    process_ai_warning,
    handle_ai_timeout,
    validate_ai_input,
    format_ai_input,
    process_ai_output,
    validate_ai_output
)

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing"""
    return {
        'text': 'Sample text for testing',
        'date': '2023-01-01',
        'currency': 1234.56,
        'percentage': 75.5,
        'address': '123 Test Street, London, SW1A 1AA',
        'phone': '+44123456789',
        'email': 'test@example.com',
        'name': 'John Smith',
        'vat': 'GB123456789',
        'company_number': '12345678'
    }

def test_text_processing_functions(sample_data):
    """Test text processing functions"""
    # Test clean_text
    result = clean_text("  Test  Text  ")
    assert result == "Test Text"
    
    # Test format_name
    result = format_name("john smith")
    assert result == "John Smith"
    
    # Test format_address
    result = format_address(sample_data['address'])
    assert isinstance(result, str)
    assert len(result) > 0

def test_validation_functions(sample_data):
    """Test validation functions"""
    # Test validate_company_number
    assert validate_company_number("12345678") is True
    assert validate_company_number("invalid") is False
    
    # Test validate_postcode
    assert validate_postcode("SW1A 1AA") is True
    assert validate_postcode("invalid") is False
    
    # Test validate_email
    assert validate_email("test@example.com") is True
    assert validate_email("invalid") is False
    
    # Test validate_vat_number
    assert validate_vat_number("GB123456789") is True
    assert validate_vat_number("invalid") is False

def test_date_functions(sample_data):
    """Test date handling functions"""
    # Test format_date
    result = format_date("2023-01-01")
    assert isinstance(result, str)
    
    # Test validate_date
    assert validate_date("2023-01-01") is True
    assert validate_date("invalid") is False
    
    # Test parse_date
    result = parse_date("2023-01-01")
    assert isinstance(result, dt.date)

def test_currency_functions(sample_data):
    """Test currency handling functions"""
    # Test format_currency
    result = format_currency(1234.56)
    assert isinstance(result, str)
    assert "£" in result
    
    # Test calculate_percentage
    result = calculate_percentage(75, 100)
    assert result == 75.0

def test_file_handling_functions():
    """Test file handling functions"""
    # Test get_file_extension
    assert get_file_extension("test.pdf") == "pdf"
    
    # Test is_valid_file_type
    assert is_valid_file_type("test.pdf", ["pdf", "docx"]) is True
    assert is_valid_file_type("test.txt", ["pdf", "docx"]) is False

def test_ai_utility_functions():
    """Test AI utility functions"""
    # Test analyze_text
    with patch('ai_utils.get_openai_client') as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Analysis result"
        mock_openai.return_value = mock_client
        result = analyze_text("Test text")
        assert isinstance(result, str)
    
    # Test extract_key_points
    with patch('ai_utils.get_openai_client') as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Key points"
        mock_openai.return_value = mock_client
        result = extract_key_points("Test text")
        assert isinstance(result, list)

def test_error_handling_functions():
    """Test error handling functions"""
    # Test process_ai_error
    result = process_ai_error(Exception("Test error"))
    assert isinstance(result, dict)
    assert "error" in result
    
    # Test handle_rate_limit
    result = handle_rate_limit()
    assert isinstance(result, bool)
    
    # Test retry_on_failure
    @retry_on_failure(max_retries=3)
    def test_function():
        raise Exception("Test error")
    
    result = test_function()
    assert result is None

def test_input_output_functions():
    """Test input/output handling functions"""
    # Test validate_ai_input
    assert validate_ai_input("Valid input") is True
    assert validate_ai_input("") is False
    
    # Test format_ai_input
    result = format_ai_input("Test input")
    assert isinstance(result, str)
    
    # Test validate_ai_output
    assert validate_ai_output("Valid output") is True
    assert validate_ai_output("") is False

def test_uk_specific_functions(sample_data):
    """Test UK-specific formatting and validation functions"""
    # Test format_uk_address
    result = format_uk_address(sample_data['address'])
    assert isinstance(result, str)
    
    # Test validate_uk_address
    assert validate_uk_address(sample_data['address']) is True
    assert validate_uk_address("Invalid address") is False
    
    # Test format_uk_phone
    result = format_uk_phone(sample_data['phone'])
    assert isinstance(result, str)
    
    # Test validate_uk_phone
    assert validate_uk_phone(sample_data['phone']) is True
    assert validate_uk_phone("invalid") is False
    
    # Test format_uk_email
    result = format_uk_email(sample_data['email'])
    assert isinstance(result, str)
    
    # Test validate_uk_email
    assert validate_uk_email(sample_data['email']) is True
    assert validate_uk_email("invalid") is False
    
    # Test format_uk_vat
    result = format_uk_vat(sample_data['vat'])
    assert isinstance(result, str)
    
    # Test validate_uk_vat
    assert validate_uk_vat(sample_data['vat']) is True
    assert validate_uk_vat("invalid") is False

def test_data_persistence_functions():
    """Test data persistence functions"""
    # Test save_company_data
    test_data = {"company": "Test Ltd", "number": "12345678"}
    result = save_company_data(test_data, "test_output/test_data.json")
    assert result is True
    
    # Test load_company_data
    loaded_data = load_company_data("test_output/test_data.json")
    assert loaded_data == test_data

def test_report_generation():
    """Test report generation functions"""
    # Test generate_report
    test_data = {
        "company": "Test Ltd",
        "number": "12345678",
        "date": "2023-01-01",
        "amount": 1234.56
    }
    result = generate_report(test_data)
    assert isinstance(result, str)
    assert "Test Ltd" in result
    assert "12345678" in result

def test_error_handling_edge_cases():
    """Test error handling for edge cases"""
    # Test with None values
    assert clean_text(None) == ""
    assert format_name(None) == ""
    assert format_address(None) == ""
    
    # Test with empty strings
    assert clean_text("") == ""
    assert format_name("") == ""
    assert format_address("") == ""
    
    # Test with invalid dates
    assert validate_date("invalid") is False
    assert parse_date("invalid") is None
    
    # Test with invalid numbers
    assert format_currency(None) == "£0.00"
    assert calculate_percentage(None, 100) == 0.0

def test_performance_with_large_data():
    """Test performance with large datasets"""
    # Test with large text
    large_text = "Test " * 1000
    result = clean_text(large_text)
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with large address
    large_address = "123 Test Street, " * 100
    result = format_address(large_address)
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with large company data
    large_data = {
        "company": "Test Ltd",
        "number": "12345678",
        "address": large_address,
        "description": large_text
    }
    result = generate_report(large_data)
    assert isinstance(result, str)
    assert len(result) > 0 