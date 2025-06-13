import pytest
import streamlit as st
from app import (
    init_session_state,
    render_about_page,
    render_instructions_page,
    render_consult_counsel_page
)
from pathlib import Path

def test_init_session_state():
    """Test session state initialization"""
    # Clear any existing session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    init_session_state()
    
    assert "company_number" in st.session_state
    assert "analysis_result" in st.session_state
    assert "group_structure" in st.session_state
    assert "file_upload" in st.session_state

def test_render_about_page():
    """Test about page rendering"""
    # Clear any existing session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    init_session_state()
    render_about_page()
    
    # Verify that session state is maintained
    assert "company_number" in st.session_state
    assert "analysis_result" in st.session_state
    assert "group_structure" in st.session_state
    assert "file_upload" in st.session_state

def test_render_instructions_page():
    """Test instructions page rendering"""
    # Clear any existing session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    init_session_state()
    render_instructions_page()
    
    # Verify that session state is maintained
    assert "company_number" in st.session_state
    assert "analysis_result" in st.session_state
    assert "group_structure" in st.session_state
    assert "file_upload" in st.session_state

def test_render_consult_counsel_page():
    """Test consult counsel page rendering"""
    # Clear any existing session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    init_session_state()
    render_consult_counsel_page()
    
    # Verify that session state is maintained
    assert "company_number" in st.session_state
    assert "analysis_result" in st.session_state
    assert "group_structure" in st.session_state
    assert "file_upload" in st.session_state

def test_session_state_persistence():
    """Test session state persistence across page renders"""
    # Clear any existing session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    init_session_state()
    
    # Set some values
    st.session_state["company_number"] = "12345678"
    st.session_state["analysis_result"] = {"test": "data"}
    
    # Render different pages
    render_about_page()
    render_instructions_page()
    render_consult_counsel_page()
    
    # Verify values persist
    assert st.session_state["company_number"] == "12345678"
    assert st.session_state["analysis_result"] == {"test": "data"}

def test_file_upload_handling(scratch_dir):
    """Test file upload handling"""
    # Clear any existing session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    init_session_state()
    
    # Create a test file
    test_file = scratch_dir / "test.pdf"
    test_file.write_text("Test content")
    
    # Simulate file upload
    st.session_state["file_upload"] = test_file
    
    # Render consult counsel page
    render_consult_counsel_page()
    
    # Verify file was processed
    assert "file_upload" in st.session_state
    assert st.session_state["file_upload"] == test_file 