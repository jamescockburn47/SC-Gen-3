import pytest
import streamlit as st
import os
from pathlib import Path
import sys
import time
from datetime import datetime
import boto3
from openai import OpenAI
import requests

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import main
from document_processing import process_document
from ch_integration import get_company_profile
from ocr_processing import process_document_ocr
from ai_utils import generate_summary
from group_structure_utils import (
    process_group_structure,
    validate_group_structure,
    generate_group_report
)
from timeline_utils import (
    process_timeline_data,
    generate_timeline_report
)
from strategic_analysis import (
    analyze_strategic_position,
    generate_strategic_recommendations,
    calculate_risk_metrics
)

@pytest.fixture
def test_data_dir():
    """Fixture to provide test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_documents():
    """Fixture to provide sample documents for testing"""
    return {
        'board_minutes': test_data_dir / "board_minutes.pdf",
        'strategy_doc': test_data_dir / "strategy_document.pdf",
        'financial_report': test_data_dir / "financial_report.pdf",
        'legal_doc': test_data_dir / "legal_document.pdf"
    }

def test_dashboard_initialization(mock_aws_credentials, mock_ch_credentials, mock_ai_credentials):
    """Test the initialization of the Strategic Counsel dashboard with real credentials"""
    # Clear session state
    st.session_state.clear()
    
    # Initialize dashboard with real credentials
    main()
    
    # Verify core session state variables and credentials
    assert st.session_state['aws_access_key'] == os.getenv('AWS_ACCESS_KEY_ID')
    assert st.session_state['aws_secret_key'] == os.getenv('AWS_SECRET_ACCESS_KEY')
    assert st.session_state['aws_region'] == os.getenv('AWS_DEFAULT_REGION')
    assert st.session_state['ch_api_key'] == os.getenv('CH_API_KEY')
    assert st.session_state['openai_api_key'] == os.getenv('OPENAI_API_KEY')
    assert st.session_state['gemini_api_key'] == os.getenv('GEMINI_API_KEY')

def test_document_processing_workflow(mock_aws_credentials, mock_ai_credentials):
    """Test the complete document processing workflow with real AWS and AI services"""
    # Initialize AWS Textract client
    textract_client = boto3.client('textract', **mock_aws_credentials)
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=mock_ai_credentials['openai_api_key'])
    
    # Test document processing with real services
    test_file = Path("tests/test_data/sample.pdf")
    assert test_file.exists(), "Test file does not exist"
    
    # Process document with real OCR
    result = process_document_ocr(test_file, method='aws')
    assert result is not None, "OCR processing failed"
    assert 'text' in result, "OCR result missing text"
    
    # Generate summary with real AI
    summary = generate_summary(result['text'])
    assert summary is not None, "Summary generation failed"
    assert len(summary) > 0, "Generated summary is empty"

def test_companies_house_integration(mock_ch_credentials):
    """Test real Companies House API integration"""
    # Get real company profile using actual API key
    company_number = "00000006"  # Example: Apple Inc.
    profile = get_company_profile(company_number)
    
    assert profile is not None, "Failed to get company profile"
    assert 'company_name' in profile, "Company profile missing name"
    assert 'registered_office_address' in profile, "Company profile missing address"
    
    # Test document retrieval
    documents = get_company_documents(company_number)
    assert documents is not None, "Failed to get company documents"
    assert len(documents) > 0, "No documents found"

def test_strategic_analysis_workflow(mock_ai_credentials):
    """Test the strategic analysis workflow with real AI services"""
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=mock_ai_credentials['openai_api_key'])
    
    # Sample document text
    sample_text = "This is a sample strategic document for testing purposes."
    
    # Perform strategic analysis
    analysis = analyze_strategic_position([sample_text])
    assert analysis is not None, "Strategic analysis failed"
    assert 'strengths' in analysis
    assert 'weaknesses' in analysis
    assert 'opportunities' in analysis
    assert 'threats' in analysis
    
    # Generate recommendations
    recommendations = generate_strategic_recommendations(analysis)
    assert recommendations is not None, "Failed to generate recommendations"
    assert len(recommendations) > 0, "No recommendations generated"

def test_group_structure_analysis(mock_ch_credentials):
    """Test the group structure analysis with real Companies House data"""
    # Get real group structure data
    parent_company = "00000006"  # Example: Apple Inc.
    group_data = get_group_structure(parent_company)
    
    assert group_data is not None, "Failed to get group structure"
    
    # Process group structure
    processed = process_group_structure(group_data)
    assert processed is not None, "Failed to process group structure"
    
    # Validate structure
    validation = validate_group_structure(processed)
    assert validation['is_valid'], "Group structure validation failed"
    
    # Generate report
    report = generate_group_report(processed)
    assert report is not None, "Failed to generate group report"

def test_timeline_analysis(mock_ai_credentials):
    """Test the timeline analysis with real AI services"""
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=mock_ai_credentials['openai_api_key'])
    
    # Sample timeline data
    timeline_data = {
        'events': [
            {
                'date': '2023-01-01',
                'description': 'Board Meeting',
                'category': 'governance'
            }
        ]
    }
    
    # Process timeline
    processed = process_timeline_data(timeline_data)
    assert processed is not None, "Failed to process timeline"
    
    # Generate report
    report = generate_timeline_report(processed)
    assert report is not None, "Failed to generate timeline report"

def test_dashboard_interactions():
    """Test dashboard user interactions with real session state"""
    # Test topic selection
    st.session_state['current_topic'] = 'strategic_analysis'
    assert st.session_state['current_topic'] == 'strategic_analysis'
    
    # Test document selection
    st.session_state['selected_documents'] = ['sample.pdf']
    assert len(st.session_state['selected_documents']) == 1
    
    # Test summary selection
    st.session_state['selected_summary_texts'] = ['Test Summary']
    assert len(st.session_state['selected_summary_texts']) == 1

def test_error_handling_and_recovery(mock_aws_credentials, mock_ch_credentials, mock_ai_credentials):
    """Test error handling and recovery with real services"""
    # Test invalid document handling
    invalid_doc = Path("nonexistent.pdf")
    result = process_document(invalid_doc)
    assert result is None, "Should handle invalid document gracefully"
    
    # Test invalid company number
    invalid_company = "99999999"
    profile = get_company_profile(invalid_company)
    assert profile is None, "Should handle invalid company number gracefully"
    
    # Test invalid API key handling
    with pytest.raises(Exception):
        openai_client = OpenAI(api_key="invalid_key")
        openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )

def test_performance_metrics(mock_aws_credentials, mock_ai_credentials):
    """Test dashboard performance with real services"""
    start_time = time.time()
    
    # Test document processing performance
    test_file = Path("tests/test_data/sample.pdf")
    result = process_document_ocr(test_file, method='aws')
    processing_time = time.time() - start_time
    assert processing_time < 30, "Document processing took too long"
    
    # Test AI analysis performance
    start_time = time.time()
    analysis = analyze_strategic_position([result['text']])
    analysis_time = time.time() - start_time
    assert analysis_time < 15, "Analysis generation took too long"

def test_data_persistence_and_state_management():
    """Test data persistence and state management with real data"""
    # Initialize test data
    test_data = {
        'summaries': ["Real Summary 1", "Real Summary 2"],
        'analysis': {
            'strengths': ["Real Strength 1"],
            'weaknesses': ["Real Weakness 1"]
        },
        'recommendations': ["Real Recommendation 1"]
    }
    
    # Update session state
    st.session_state['processed_summaries'] = test_data['summaries']
    st.session_state['current_analysis'] = test_data['analysis']
    st.session_state['current_recommendations'] = test_data['recommendations']
    
    # Verify persistence
    assert st.session_state['processed_summaries'] == test_data['summaries']
    assert st.session_state['current_analysis'] == test_data['analysis']
    assert st.session_state['current_recommendations'] == test_data['recommendations'] 