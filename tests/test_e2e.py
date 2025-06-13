import pytest
import streamlit as st
import os
from pathlib import Path
import sys
import time
from datetime import datetime

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import main
from document_processing import process_document
from ch_integration import get_company_profile
from ocr_processing import process_document_ocr
from ai_utils import generate_summary

@pytest.fixture
def test_data_dir():
    """Fixture to provide test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def real_pdf_file(test_data_dir):
    """Fixture to provide a real PDF file for testing"""
    return test_data_dir / "real_sample.pdf"

@pytest.fixture
def real_company_number():
    """Fixture to provide a real company number for testing"""
    return "00000006"  # Example: Apple Inc.

def test_full_document_processing_flow(real_pdf_file):
    """Test the complete document processing flow with real files"""
    # Initialize session state
    st.session_state.clear()
    st.session_state['current_topic'] = 'test_topic'
    st.session_state['ocr_method'] = 'aws'
    st.session_state['ocr_method_radio'] = 0
    
    # Test document upload and processing
    assert os.path.exists(real_pdf_file), f"Test file {real_pdf_file} does not exist"
    
    # Process document with real OCR
    result = process_document_ocr(real_pdf_file, method='aws')
    assert result is not None, "OCR processing failed"
    assert 'text' in result, "OCR result missing text"
    
    # Generate summary with real AI
    summary = generate_summary(result['text'])
    assert summary is not None, "Summary generation failed"
    assert len(summary) > 0, "Generated summary is empty"
    
    # Verify session state updates
    assert 'processed_summaries' in st.session_state
    assert len(st.session_state['processed_summaries']) > 0

def test_companies_house_integration(real_company_number):
    """Test real Companies House API integration"""
    # Get real company profile
    profile = get_company_profile(real_company_number)
    assert profile is not None, "Failed to get company profile"
    assert 'company_name' in profile, "Company profile missing name"
    assert 'registered_office_address' in profile, "Company profile missing address"
    
    # Verify document availability
    assert 'ch_available_documents' in st.session_state
    assert len(st.session_state['ch_available_documents']) > 0

def test_application_flow():
    """Test the complete application flow"""
    # Initialize session state
    st.session_state.clear()
    
    # Test OCR method selection
    st.session_state['ocr_method_radio'] = 0
    assert st.session_state['ocr_method'] == 'aws'
    
    st.session_state['ocr_method_radio'] = 1
    assert st.session_state['ocr_method'] == 'tesseract'
    
    # Test document processing completion
    st.session_state['document_processing_complete'] = False
    assert not st.session_state['document_processing_complete']
    
    # Test summary selection
    test_summaries = ["Summary 1", "Summary 2", "Summary 3"]
    st.session_state['processed_summaries'] = test_summaries
    st.session_state['selected_summary_texts'] = [test_summaries[0]]
    assert len(st.session_state['selected_summary_texts']) == 1

def test_error_handling():
    """Test application error handling with real scenarios"""
    # Test with invalid file
    invalid_file = Path("nonexistent.pdf")
    result = process_document(invalid_file)
    assert result is None, "Should handle invalid file gracefully"
    
    # Test with invalid company number
    invalid_company = "99999999"
    profile = get_company_profile(invalid_company)
    assert profile is None, "Should handle invalid company number gracefully"
    
    # Test with invalid OCR method
    st.session_state['ocr_method'] = 'invalid_method'
    result = process_document_ocr(Path("test.pdf"), method='invalid_method')
    assert result is None, "Should handle invalid OCR method gracefully"

def test_performance():
    """Test application performance with real operations"""
    start_time = time.time()
    
    # Test document processing performance
    result = process_document_ocr(real_pdf_file, method='aws')
    processing_time = time.time() - start_time
    assert processing_time < 30, "Document processing took too long"
    
    # Test summary generation performance
    start_time = time.time()
    summary = generate_summary(result['text'])
    summary_time = time.time() - start_time
    assert summary_time < 10, "Summary generation took too long"
    
    # Test Companies House API performance
    start_time = time.time()
    profile = get_company_profile(real_company_number)
    api_time = time.time() - start_time
    assert api_time < 5, "Companies House API call took too long"

def test_data_persistence():
    """Test data persistence across session state updates"""
    # Initialize test data
    test_data = {
        'summaries': ["Test Summary 1", "Test Summary 2"],
        'selections': ["Test Summary 1"],
        'profiles': {"12345678": {"name": "Test Company"}}
    }
    
    # Update session state
    st.session_state['processed_summaries'] = test_data['summaries']
    st.session_state['selected_summary_texts'] = test_data['selections']
    st.session_state['ch_company_profiles_map'] = test_data['profiles']
    
    # Verify persistence
    assert st.session_state['processed_summaries'] == test_data['summaries']
    assert st.session_state['selected_summary_texts'] == test_data['selections']
    assert st.session_state['ch_company_profiles_map'] == test_data['profiles']
    
    # Test data clearing
    st.session_state.clear()
    assert 'processed_summaries' not in st.session_state
    assert 'selected_summary_texts' not in st.session_state
    assert 'ch_company_profiles_map' not in st.session_state 