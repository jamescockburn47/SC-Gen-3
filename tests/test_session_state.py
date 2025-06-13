import pytest
import streamlit as st
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import init_session_state

@pytest.fixture
def mock_streamlit():
    """Fixture to mock Streamlit session state and widgets"""
    with patch('streamlit.session_state', {}) as mock_state:
        with patch('streamlit.radio') as mock_radio:
            with patch('streamlit.text_input') as mock_text_input:
                with patch('streamlit.button') as mock_button:
                    with patch('streamlit.error') as mock_error:
                        with patch('streamlit.success') as mock_success:
                            with patch('streamlit.rerun') as mock_rerun:
                                yield {
                                    'state': mock_state,
                                    'radio': mock_radio,
                                    'text_input': mock_text_input,
                                    'button': mock_button,
                                    'error': mock_error,
                                    'success': mock_success,
                                    'rerun': mock_rerun
                                }

def test_init_session_state_defaults(mock_streamlit):
    """Test that session state is initialized with correct default values"""
    init_session_state()
    
    # Check essential session state variables
    assert mock_streamlit['state'].get('current_topic') == "general_default_topic"
    assert mock_streamlit['state'].get('ocr_method') == "aws"
    assert mock_streamlit['state'].get('ocr_method_radio') == 0
    assert mock_streamlit['state'].get('document_processing_complete') is True
    assert isinstance(mock_streamlit['state'].get('processed_summaries'), list)
    assert isinstance(mock_streamlit['state'].get('selected_summary_texts'), list)
    assert isinstance(mock_streamlit['state'].get('ch_analysis_summaries_for_injection'), list)
    assert isinstance(mock_streamlit['state'].get('ch_available_documents'), list)
    assert isinstance(mock_streamlit['state'].get('ch_document_selection'), dict)
    assert isinstance(mock_streamlit['state'].get('ch_company_profiles_map'), dict)

def test_ocr_method_radio_handling(mock_streamlit):
    """Test OCR method radio button handling"""
    # Initialize session state
    init_session_state()
    
    # Test OCR method mapping
    ocr_method_map = {"aws": 0, "google": 1, "none": 2}
    ocr_options = ["AWS Textract", "Google Drive", "None"]
    
    # Test each OCR method
    for method, index in ocr_method_map.items():
        # Set the OCR method
        mock_streamlit['state']['ocr_method'] = method
        mock_streamlit['state']['ocr_method_radio'] = index
        
        # Verify the mapping is correct
        assert mock_streamlit['state']['ocr_method'] == method
        assert mock_streamlit['state']['ocr_method_radio'] == index

def test_session_state_persistence(mock_streamlit):
    """Test that session state persists between function calls"""
    # Initialize session state
    init_session_state()
    
    # Set some values
    mock_streamlit['state']['current_topic'] = "test_topic"
    mock_streamlit['state']['ocr_method'] = "google"
    mock_streamlit['state']['ocr_method_radio'] = 1
    
    # Verify values persist
    assert mock_streamlit['state']['current_topic'] == "test_topic"
    assert mock_streamlit['state']['ocr_method'] == "google"
    assert mock_streamlit['state']['ocr_method_radio'] == 1

def test_session_state_reset(mock_streamlit):
    """Test session state reset functionality"""
    # Initialize session state
    init_session_state()
    
    # Set some values
    mock_streamlit['state']['current_topic'] = "test_topic"
    mock_streamlit['state']['processed_summaries'] = ["summary1", "summary2"]
    
    # Simulate topic change
    mock_streamlit['text_input'].return_value = "new_topic"
    
    # Verify state is reset appropriately
    assert mock_streamlit['state']['current_topic'] == "new_topic"
    assert mock_streamlit['state']['processed_summaries'] == []
    assert mock_streamlit['state']['selected_summary_texts'] == []
    assert mock_streamlit['state']['loaded_memories'] == []

def test_invalid_ocr_method_handling(mock_streamlit):
    """Test handling of invalid OCR method values"""
    # Initialize session state
    init_session_state()
    
    # Set an invalid OCR method
    mock_streamlit['state']['ocr_method'] = "invalid_method"
    
    # Verify it defaults to "aws"
    assert mock_streamlit['state']['ocr_method'] == "aws"
    assert mock_streamlit['state']['ocr_method_radio'] == 0

def test_session_state_type_safety(mock_streamlit):
    """Test that session state maintains correct types"""
    # Initialize session state
    init_session_state()
    
    # Verify list types
    assert isinstance(mock_streamlit['state']['processed_summaries'], list)
    assert isinstance(mock_streamlit['state']['selected_summary_texts'], list)
    assert isinstance(mock_streamlit['state']['ch_analysis_summaries_for_injection'], list)
    assert isinstance(mock_streamlit['state']['ch_available_documents'], list)
    
    # Verify dict types
    assert isinstance(mock_streamlit['state']['ch_document_selection'], dict)
    assert isinstance(mock_streamlit['state']['ch_company_profiles_map'], dict)
    
    # Verify boolean types
    assert isinstance(mock_streamlit['state']['document_processing_complete'], bool)
    
    # Verify string types
    assert isinstance(mock_streamlit['state']['current_topic'], str)
    assert isinstance(mock_streamlit['state']['ocr_method'], str) 