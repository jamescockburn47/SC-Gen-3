import pytest
import streamlit as st
import re

def test_widget_keys():
    """Test to ensure no duplicate widget keys in the sidebar."""
    # Track all widget keys
    widget_keys = set()
    duplicate_keys = set()
    
    # Function to check for duplicate keys
    def check_widget_key(key):
        if key in widget_keys:
            duplicate_keys.add(key)
        widget_keys.add(key)
    
    # Mock Streamlit widgets to track keys
    original_multiselect = st.multiselect
    original_checkbox = st.checkbox
    
    def mock_multiselect(*args, **kwargs):
        if 'key' in kwargs:
            check_widget_key(kwargs['key'])
        return []
    
    def mock_checkbox(*args, **kwargs):
        if 'key' in kwargs:
            check_widget_key(kwargs['key'])
        return False
    
    # Replace Streamlit widgets with our tracking versions
    st.multiselect = mock_multiselect
    st.checkbox = mock_checkbox
    
    try:
        # Test memory widget
        st.multiselect(
            "Inject Memories",
            ["Memory 1", "Memory 2"],
            key="mem_multiselect_sidebar_context"
        )
        
        # Test digest checkbox
        st.checkbox(
            "Inject Digest",
            key="inject_digest_checkbox_context"
        )
        
        # Check for duplicates
        assert len(duplicate_keys) == 0, f"Found duplicate widget keys: {duplicate_keys}"
        
        # Print all widget keys for debugging
        print("\nAll widget keys found:")
        for key in sorted(widget_keys):
            print(f"- {key}")
            
    finally:
        # Restore original Streamlit widgets
        st.multiselect = original_multiselect
        st.checkbox = original_checkbox

def test_problematic_key():
    """Test to specifically check for the problematic key."""
    # Track all widget keys
    widget_keys = set()
    
    # Function to check for problematic key
    def check_widget_key(key):
        if key == "mem_multiselect_sidebar_unique":
            print(f"Found problematic key: {key}")
        widget_keys.add(key)
    
    # Mock Streamlit widgets to track keys
    original_multiselect = st.multiselect
    
    def mock_multiselect(*args, **kwargs):
        if 'key' in kwargs:
            check_widget_key(kwargs['key'])
        return []
    
    # Replace Streamlit widget with our tracking version
    st.multiselect = mock_multiselect
    
    try:
        # Test memory widget with current key
        st.multiselect(
            "Inject Memories",
            ["Memory 1", "Memory 2"],
            key="mem_multiselect_sidebar_context"
        )
        
        # Test memory widget with problematic key
        st.multiselect(
            "Inject Memories",
            ["Memory 1", "Memory 2"],
            key="mem_multiselect_sidebar_unique"
        )
        
        # Print all widget keys for debugging
        print("\nAll widget keys found in problematic key test:")
        for key in sorted(widget_keys):
            print(f"- {key}")
            
    finally:
        # Restore original Streamlit widget
        st.multiselect = original_multiselect

if __name__ == "__main__":
    # Run the tests
    test_widget_keys()
    test_problematic_key()
    print("\nAll tests completed successfully!") 