import streamlit as st

"""Instructions page for Strategic Counsel - Comprehensive usage guide"""


def render_instructions_page():
    """Render comprehensive step-by-step usage instructions."""
    
    st.markdown("## 📖 Instructions: Using Strategic Counsel")
    st.markdown("*(Complete step-by-step guide for all features)*")
    st.markdown("---")
    
    # Quick Start Guide
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    **Getting Started in 3 Steps:**
    1. **Set your Topic ID** in the sidebar (e.g., "Client ABC Case")
    2. **Choose your AI Model** from the sidebar dropdown
    3. **Start with AI Consultation** for immediate legal guidance
    """)
    
    st.markdown("---")
    
    # AI Consultation Instructions
    st.subheader("🤖 AI Legal Consultation")
    
    with st.expander("Step-by-Step AI Consultation Guide", expanded=True):
        st.markdown("""
        **1. Basic Consultation (No Documents Required)**
        - Navigate to the **🤖 AI Consultation** tab
        - Select a legal topic from the dropdown (Corporate Governance, Contract Law, etc.)
        - Type your legal question or describe your situation
        - Choose your preferred AI model (GPT-4.1 recommended for complex issues)
        - Click **"🤖 Get AI Consultation"**
        
        **2. Enhanced Consultation (With Documents)**
        - Expand the **"📁 Optional: Upload Supporting Documents"** section
        - Upload relevant PDF, DOCX, or TXT files
        - Click **"Process Uploaded Documents"** to extract and summarize content
        - The AI will use this context to provide more targeted advice
        
        **3. Understanding the Response**
        - Review the AI's comprehensive legal guidance
        - Check the estimated cost for the consultation
        - The response includes practical recommendations and risk considerations
        
        **💡 Tips:**
        - Be specific in your questions for better responses
        - Use the topic selection to guide the AI's expertise area
        - Upload relevant documents for more contextual advice
        - Monitor costs using the built-in estimation
        """)
    
    st.markdown("---")
    
    # Citation Verification Instructions
    st.subheader("⚖️ AI Citation Verification")
    
    with st.expander("Step-by-Step Citation Verification Guide", expanded=False):
        st.markdown("""
        **How Citation Verification Works:**
        When you receive an AI consultation, the system automatically:
        1. **Detects** legal citations (case law and legislation) in the response
        2. **Verifies** them by searching legal databases (Bailii, Casemine)
        3. **Displays** verification results with clear status indicators
        
        **Understanding Verification Results:**
        - **✅ Automatically Verified Citations** - Found and confirmed in legal databases
        - **⚠️ Unverified Citations** - Could not be automatically verified
        
        **Manual Citation Enhancement (For Unverified Citations):**
        1. **Provide Specific Links** - Enter Bailii or legislation.gov.uk URLs for unverified citations
        2. **AI Content Analysis** - The system fetches and analyzes the provided content
        3. **Intelligent Validation** - Uses GPT-4o to confirm the case supports your legal argument
        4. **Detailed Feedback** - Shows exactly why citations were accepted or rejected
        
        **What the AI Validation Checks:**
        - ✅ **Correct Document**: Is this the right case/legislation?
        - ✅ **Supports Proposition**: Does it actually support the legal point being made?
        - ✅ **Legal Principle**: What does this case/legislation establish?
        - ✅ **Confidence Level**: High/Medium/Low confidence in the analysis
        
        **Validation Results:**
        - **Accepted**: Updates consultation with properly linked citations
        - **Rejected**: Explains why (wrong document, doesn't support proposition, etc.)
        - **Error**: Shows technical issues with the provided link
        
        **💡 Tips:**
        - Look for unverified citations after each consultation
        - Use specific Bailii URLs (e.g., bailii.org/uk/cases/UKHL/1975/2.html)
        - Provide legislation.gov.uk links for statutes
        - Review the detailed AI analysis to understand why citations were accepted/rejected
        - The system prevents hallucinated or incorrect legal references
        """)
    
    st.markdown("---")
    
    # Companies House Analysis Instructions
    st.subheader("🏢 Companies House Analysis")
    
    with st.expander("Step-by-Step Companies House Guide", expanded=False):
        st.markdown("""
        **1. Prepare Company Information**
        - Navigate to the **🏢 Companies House Analysis** tab
        - Enter UK company numbers (one per line or comma-separated)
        - Set the date range for document search
        - Select document categories to analyze
        
        **2. Search for Documents**
        - Click **"🔍 Search for Available Documents"**
        - Review the list of found documents
        - Uncheck any documents you don't want to include
        
        **3. Configure Analysis**
        - Add optional AI instructions for specific focus areas
        - Enter keywords to highlight or filter in analysis
        - Click **"📊 Run Analysis on Selected Documents"**
        
        **4. Review Results**
        - Check the narrative summary and key findings
        - Review detailed document information
        - Download the full report as DOCX
        - View processing metrics and cost estimates
        
        **💡 Tips:**
        - Use specific date ranges to focus on relevant periods
        - Select appropriate document categories for your analysis
        - Add AI instructions to focus on specific aspects (e.g., "Focus on financial risks")
        - Monitor costs as OCR processing may incur additional charges
        """)
    
    st.markdown("---")
    
    # Group Structure Instructions
    st.subheader("📊 Group Structure Visualization")
    
    with st.expander("Step-by-Step Group Structure Guide", expanded=False):
        st.markdown("""
        **1. Start Group Analysis**
        - Navigate to the **📊 Group Structure** tab
        - Enter a company number to begin analysis
        - Choose OCR method if needed (AWS Textract recommended for scanned documents)
        
        **2. Follow the Analysis Steps**
        - **Step 1:** Fetch company profile and basic information
        - **Step 2:** Identify parent companies and group structure
        - **Step 3:** Analyze subsidiaries and related entities
        - **Step 4:** Generate comprehensive group report
        
        **3. Review Results**
        - View the visual group structure
        - Check detailed analysis reports
        - Export results for further use
        
        **💡 Tips:**
        - Start with a known company number
        - Use AWS Textract for better OCR results on scanned documents
        - Review each step before proceeding to the next
        - The analysis can be time-consuming for complex group structures
        """)
    
    st.markdown("---")
    
    # Configuration and Settings
    st.subheader("⚙️ Configuration & Settings")
    
    with st.expander("Essential Configuration Guide", expanded=False):
        st.markdown("""
        **1. API Keys Setup**
        - **OpenAI API Key:** Required for GPT models
        - **Google Gemini API Key:** Required for Gemini models
        - **Companies House API Key:** Required for UK company data
        - **AWS Credentials:** Optional for Textract OCR
        
        **2. Model Selection**
        - **GPT-4.1:** Best for complex legal analysis (highest cost)
        - **GPT-4.1-mini:** Good balance of quality and cost
        - **Gemini models:** Alternative AI models
        - **o3/o4 models:** Latest OpenAI models
        
        **3. Performance Settings**
        - **Caching:** Enabled by default to reduce costs
        - **Memory Management:** Automatic for large document sets
        - **Batch Processing:** Available for multiple companies
        
        **4. Cost Management**
        - Monitor estimated costs before running analysis
        - Use caching to avoid redundant processing
        - Choose appropriate models for your budget
        """)
    
    st.markdown("---")
    
    # Best Practices
    st.subheader("💡 Best Practices & Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 For Best Results:**
        - Keep topic IDs consistent for related work
        - Be specific in your questions and instructions
        - Upload relevant documents for enhanced context
        - Verify unverified citations with specific Bailii/legislation.gov.uk links
        - Use appropriate document categories in CH analysis
        - Monitor costs using built-in estimation tools
        - Review AI outputs and citation validation before making decisions
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Important Notes:**
        - This is a professional tool for legal analysis
        - All AI outputs should be reviewed by qualified professionals
        - Document processing may incur AI model costs
        - AWS Textract OCR has additional costs
        - Keep API keys secure and monitor usage
        - Regular backups of important work are recommended
        """)
    
    st.markdown("---")
    
    # Troubleshooting
    st.subheader("🔧 Troubleshooting")
    
    with st.expander("Common Issues & Solutions", expanded=False):
        st.markdown("""
        **Text Visibility Issues:**
        - Use the text visibility test at http://localhost:8502
        - Check browser settings and zoom levels
        - Try refreshing the page
        
        **API Connection Issues:**
        - Verify API keys are correctly configured
        - Check internet connection
        - Ensure API quotas haven't been exceeded
        
        **Document Processing Issues:**
        - Ensure documents are in supported formats (PDF, DOCX, TXT)
        - Check file size limits
        - Try different OCR methods for scanned documents
        
        **Performance Issues:**
        - Use caching to reduce processing time
        - Process documents in smaller batches
        - Monitor system resources
        
        **Getting Help:**
        - Check the About tab for system information
        - Review application logs for error details
        - Test with the provided test scripts
        """)
    
    st.markdown("---")
    
    # Advanced Features
    st.subheader("🚀 Advanced Features")
    
    with st.expander("Advanced Usage Guide", expanded=False):
        st.markdown("""
        **1. Batch Processing**
        - Process multiple companies simultaneously
        - Use CSV export for large datasets
        - Combine results for comprehensive analysis
        
        **2. Custom Protocols**
        - Modify strategic_protocols.txt for custom AI behavior
        - Add specific legal frameworks or requirements
        - Customize AI response patterns
        
        **3. Integration Workflows**
        - Use Companies House data in AI consultations
        - Combine document analysis with legal guidance
        - Create comprehensive client reports
        
        **4. Performance Optimization**
        - Use appropriate AI models for different tasks
        - Leverage caching for repeated analysis
        - Monitor and optimize costs
        """)
    
    st.markdown("---")
    
    # Quick Reference
    st.subheader("📋 Quick Reference")
    
    st.markdown("""
    **Essential Commands:**
    - **Start App:** `./manage_streamlit.sh start`
    - **Check Status:** `./manage_streamlit.sh status`
    - **Restart App:** `./manage_streamlit.sh restart`
    - **Test Visibility:** `./test_visibility.sh`
    
    **Key URLs:**
    - **Main App:** http://localhost:8501
    - **Visibility Test:** http://localhost:8502
    
    **Important Files:**
    - **Configuration:** `.streamlit/config.toml`
    - **CSS Theme:** `static/harcus_parker_style.css`
    - **Protocols:** `strategic_protocols.txt`
    """)
    
    st.markdown("---")
    st.caption("Strategic Counsel v3.0 | Complete Usage Guide")
    st.caption("For technical support, check the About tab or GitHub repository")

