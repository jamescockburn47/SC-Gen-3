#!/usr/bin/env python3
"""
Simple test to verify text visibility in Streamlit app
"""

import streamlit as st

def main():
    st.set_page_config(
        page_title="Text Visibility Test",
        page_icon="👁️",
        layout="wide"
    )
    
    # Load external CSS
    try:
        with open('static/harcus_parker_style.css', 'r') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found")
    
    st.title("👁️ Text Visibility Test")
    st.markdown("This page tests if text is visible with our custom CSS.")
    
    # Test various text elements
    st.header("Header Text")
    st.subheader("Subheader Text")
    st.write("Regular paragraph text")
    st.markdown("**Bold text** and *italic text*")
    
    # Test form elements
    st.text_input("Text Input", "Sample text")
    st.text_area("Text Area", "Sample text area content")
    st.selectbox("Select Box", ["Option 1", "Option 2", "Option 3"])
    st.multiselect("Multi Select", ["Option A", "Option B", "Option C"])
    st.number_input("Number Input", value=42)
    st.checkbox("Checkbox")
    st.radio("Radio Buttons", ["Choice 1", "Choice 2"])
    st.slider("Slider", 0, 100, 50)
    
    # Test buttons
    if st.button("Test Button"):
        st.success("Button clicked!")
    
    # Test alerts
    st.info("This is an info message")
    st.warning("This is a warning message")
    st.error("This is an error message")
    st.success("This is a success message")
    
    # Test data
    import pandas as pd
    df = pd.DataFrame({
        'Column 1': [1, 2, 3, 4],
        'Column 2': ['A', 'B', 'C', 'D']
    })
    st.dataframe(df)
    
    # Test metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Metric 1", "100", "10")
    col2.metric("Metric 2", "200", "-20")
    col3.metric("Metric 3", "300", "30")
    
    # Test expander
    with st.expander("Click to expand"):
        st.write("This is content inside an expander")
    
    # Test tabs
    tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
    with tab1:
        st.write("Content in tab 1")
    with tab2:
        st.write("Content in tab 2")
    with tab3:
        st.write("Content in tab 3")
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. **Check if all text is visible** - Should be black text on white background
    2. **Check form elements** - Inputs should have white background and black text
    3. **Check buttons** - Should be blue with white text
    4. **Check alerts** - Should have colored borders but readable text
    5. **Check dataframes** - Should be readable
    6. **Check tabs and expanders** - Should be functional and readable
    """)
    
    if st.button("✅ All Text Visible"):
        st.balloons()
        st.success("Great! Text visibility is working correctly.")

if __name__ == "__main__":
    main() 