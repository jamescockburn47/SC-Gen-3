# about_page.py
"""
Contains the content and rendering logic for the 'About' tab in the Strategic Counsel application.
Version 3.0: Updated for current application state with UI improvements and optimizations.
"""
import streamlit as st
import config
from pathlib import Path

def render_about_page():
    """Renders all the content for the 'About' tab."""

    st.markdown("## ℹ️ About Strategic Counsel")
    st.markdown("*(Version 3.0 - AI Legal Analysis Platform)*")
    st.markdown("---")
    
    st.markdown("""
    **Strategic Counsel (SC)** is an advanced AI-powered legal analysis platform designed to serve as an 
    intelligent partner for legal professionals, corporate analysts, and strategic advisors. The platform 
    combines cutting-edge AI models with specialized legal and corporate data analysis tools to provide 
    comprehensive insights and assistance.
    """)

    # Key Features Overview
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🤖 AI-Powered Features")
        st.markdown("""
        - **Independent AI Consultation** - Direct legal guidance without document uploads
        - **AI Citation Verification** - Automatic validation of legal case citations and legislation
        - **Document Analysis** - Process PDFs, DOCX, and TXT files with AI summarization
        - **Web Content Analysis** - Extract and analyze content from URLs
        - **Multi-Model Support** - OpenAI GPT-4.1, Gemini, and other advanced models
        - **Cost Optimization** - Smart caching and performance monitoring
        """)
    
    with col2:
        st.markdown("### 🏢 Corporate Analysis Tools")
        st.markdown("""
        - **Companies House Integration** - UK company filings and document analysis
        - **Group Structure Visualization** - Corporate hierarchy mapping
        - **Financial Document Processing** - Accounts, confirmation statements, etc.
        - **OCR Capabilities** - AWS Textract integration for scanned documents
        - **Batch Processing** - Multi-company analysis workflows
        """)

    st.markdown("---")
    
    # Core Capabilities
    st.subheader("🎯 Core Capabilities")
    
    with st.expander("AI Legal Consultation", expanded=True):
        st.markdown("""
        **Independent Legal Guidance**
        - Ask legal questions directly without requiring document uploads
        - Choose from 10 predefined legal topics (Corporate Governance, Contract Law, Employment Law, etc.)
        - Optional document context for enhanced analysis
        - Real-time cost estimation for AI model usage
        - Export consultations to DOCX format
        
        **Smart Context Management**
        - Session-based topic organization
        - Persistent memory for ongoing matters
        - Dynamic context injection from uploaded documents
        - Web content integration for comprehensive analysis
        """)

    with st.expander("AI Citation Verification", expanded=False):
        st.markdown("""
        **Intelligent Legal Citation Validation**
        - **Automatic Detection** - Extracts case law and legislation references from AI responses
        - **Web Verification** - Searches Bailii, Casemine, and other legal databases automatically
        - **Manual Enhancement** - Allows legal professionals to provide specific Bailii/legislation.gov.uk links
        - **AI Content Analysis** - Uses GPT-4o to verify that cases actually support legal propositions
        
        **Sophisticated Validation Process**
        - Fetches actual case/legislation content from provided URLs
        - Analyzes whether the document is the correct case/legislation
        - Confirms the legal principle established by the case
        - Validates that the citation supports the specific legal argument
        - Provides detailed explanations for acceptance or rejection
        
        **Professional Quality Control**
        - High/medium/low confidence ratings for all validations
        - Detailed legal analysis showing why citations were accepted/rejected
        - Protocol compliance checks with updated consultation text
        - Prevents hallucinated or incorrect legal references
        """)
        
    with st.expander("Protocol Compliance System", expanded=False):
        st.markdown("""
        **Strategic Protocol Enforcement**
        - **Real-time Compliance** - Monitors AI responses against legal protocols
        - **Citation Discipline** - Prevents hallucinated legal authorities
        - **Memory Integrity** - Safeguards against false context assumptions
        - **Export Fidelity** - Maintains document structure and formatting
        - **Adversarial Reasoning** - Promotes doctrinal logic over narrative framing
        
        **Automated Quality Assurance**
        - Gemini-powered compliance analysis
        - Section-by-section protocol adherence reporting
        - Red flag identification for protocol violations
        - Detailed compliance reports for all AI consultations
        """)

    with st.expander("Companies House Analysis", expanded=False):
        st.markdown("""
        **Comprehensive UK Company Analysis**
        - Search and retrieve company filings from Companies House API
        - Process multiple document types: Accounts, Confirmation Statements, Officers, etc.
        - Advanced text extraction with optional AWS Textract OCR
        - AI-powered summarization of complex legal documents
        - Batch processing for multiple companies
        - Export results as CSV and DOCX reports
        
        **Document Categories Available:**
        - Accounts and financial statements
        - Annual confirmation statements
        - Director and officer appointments
        - Share capital changes
        - Mortgages and charges
        - Insolvency proceedings
        - Persons with significant control
        - Company name changes
        - Registered office changes
        """)

    with st.expander("Group Structure Visualization", expanded=False):
        st.markdown("""
        **Corporate Hierarchy Analysis**
        - Map parent-subsidiary relationships
        - Visualize complex corporate structures
        - Process group-wide document analysis
        - Identify ultimate beneficial owners
        - Generate comprehensive group reports
        """)

    with st.expander("Performance Optimizations", expanded=False):
        st.markdown("""
        **Advanced Caching and Performance**
        - Smart document caching to reduce processing costs
        - Memory management for large document sets
        - Session state optimization
        - Batch processing capabilities
        - Real-time performance monitoring
        
        **Cost Management**
        - Token usage tracking
        - Cost estimation for all AI operations
        - Efficient model selection based on task requirements
        - Caching to avoid redundant processing
        """)

    st.markdown("---")
    
    # Technology Stack
    st.subheader("🛠️ Technology Stack")
    
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown("**Frontend & Framework**")
        st.markdown("""
        - **Streamlit** - Modern web application framework
        - **Custom CSS** - Advanced theming and UI customization
        - **JavaScript Integration** - Dynamic content management
        - **Responsive Design** - Works on desktop and mobile
        """)
        
        st.markdown("**AI & Machine Learning**")
        st.markdown("""
        - **OpenAI GPT-4.1** - Latest GPT model for complex analysis
        - **Google Gemini** - Alternative AI model for diverse tasks
        - **Multi-Model Architecture** - Fallback and optimization
        - **Custom Prompts** - Legal and corporate-specific guidance
        """)
    
    with tech_col2:
        st.markdown("**Data Processing**")
        st.markdown("""
        - **PyPDF2** - PDF text extraction
        - **python-docx** - Word document processing
        - **BeautifulSoup4** - Web content parsing
        - **Pandas** - Data manipulation and analysis
        - **AWS Textract** - Advanced OCR capabilities
        """)
        
        st.markdown("**APIs & Integration**")
        st.markdown("""
        - **Companies House API** - UK company data
        - **OpenAI API** - GPT model access
        - **Google Gemini API** - Alternative AI models
        - **AWS Services** - Textract OCR and cloud processing
        """)

    st.markdown("---")
    
    # System Information
    st.subheader("📊 System Information")
    
    # Get system information
    try:
        app_base_path = str(config.APP_BASE_PATH) if hasattr(config, 'APP_BASE_PATH') else "N/A"
        textract_available = getattr(config, 'CH_PIPELINE_TEXTRACT_FLAG', "Unknown")
        proto_path = getattr(config, 'PROTO_PATH', Path("strategic_protocols.txt"))
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Application Path", app_base_path)
            st.metric("AWS Textract", "Available" if textract_available else "Not Available")
            st.metric("Protocol File", proto_path.name if hasattr(proto_path, 'name') else str(proto_path))
        
        with info_col2:
            st.metric("Python Version", "3.10+")
            st.metric("Streamlit Version", "1.28+")
            st.metric("AI Models", "GPT-4.1, Gemini, o3, o4-mini")
            
    except Exception as e:
        st.warning(f"Could not retrieve system information: {e}")

    st.markdown("---")
    
    # Usage Guidelines
    st.subheader("📋 Usage Guidelines")
    st.markdown("""
    **Best Practices:**
    - Start with the AI Consultation tab for general legal questions
    - Use Companies House Analysis for UK company research
    - Upload relevant documents for enhanced context
    - Keep topic IDs consistent for related work
    - Monitor costs using the built-in estimation tools
    
    **Important Notes:**
    - This is a professional tool designed for legal and corporate analysis
    - All AI outputs should be reviewed by qualified professionals
    - Document processing may incur costs for AI model usage
    - AWS Textract OCR is optional and may incur additional costs
    """)

    st.markdown("---")
    
    # Support and Documentation
    st.subheader("📚 Support & Documentation")
    st.markdown("""
    **Available Resources:**
    - **Instructions Tab** - Step-by-step usage guide
    - **GitHub Repository** - Source code and documentation
    - **Configuration Files** - Customizable settings
    - **Test Suite** - Text visibility and functionality tests
    
    **Getting Help:**
    - Check the Instructions tab for detailed usage information
    - Review system logs for troubleshooting
    - Ensure all required API keys are configured
    - Test with the provided test scripts if issues arise
    """)

    st.markdown("---")
    st.caption("Strategic Counsel v3.0 | AI-Powered Legal Analysis Platform")
    st.caption("Built with Streamlit, OpenAI, and Google Gemini APIs")

