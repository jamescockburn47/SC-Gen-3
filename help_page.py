# help_page.py
"""
Modern, consolidated Help & About page for Strategic Counsel
Combines instructions and about information in a user-friendly format
"""

import streamlit as st
import config
from pathlib import Path

def render_help_page():
    """Render the modern, consolidated help page"""
    
    # Header with logo and tagline
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>⚖️ Strategic Counsel</h1>
        <h3 style="color: #0066cc; margin-top: -1rem;">Multi-Agent AI Legal Analysis Platform</h3>
        <p style="color: #666; font-size: 1.1rem;">Professional legal AI with automatic model orchestration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick navigation
    st.markdown("### 🧭 Quick Navigation")
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        if st.button("🚀 Getting Started", use_container_width=True):
            st.session_state.help_section = "getting_started"
    with nav_col2:
        if st.button("🤖 Multi-Agent RAG", use_container_width=True):
            st.session_state.help_section = "multi_agent"
    with nav_col3:
        if st.button("📚 All Features", use_container_width=True):
            st.session_state.help_section = "features"
    with nav_col4:
        if st.button("🔧 Troubleshooting", use_container_width=True):
            st.session_state.help_section = "troubleshooting"
    
    st.markdown("---")
    
    # Main content based on selection
    section = getattr(st.session_state, 'help_section', 'overview')
    
    if section == "getting_started":
        render_getting_started()
    elif section == "multi_agent":
        render_multi_agent_guide()
    elif section == "features":
        render_all_features()
    elif section == "troubleshooting":
        render_troubleshooting()
    else:
        render_overview()

def render_overview():
    """Render the main overview section"""
    
    st.markdown("## 🎯 What is Strategic Counsel?")
    
    st.info("""
    **Strategic Counsel** is a professional AI legal analysis platform featuring a **sophisticated multi-agent system** 
    that automatically orchestrates multiple AI models to provide comprehensive legal analysis.
    
    🔹 **5 Specialized AI Agents** working in parallel  
    🔹 **Automatic task assignment** based on query complexity  
    🔹 **Advanced document processing** with legal intelligence  
    🔹 **Complete privacy** - everything runs locally  
    """)
    
    # Feature highlights
    st.markdown("### ✨ Key Capabilities")
    
    tab1, tab2, tab3 = st.tabs(["🤖 Multi-Agent RAG", "⚖️ Legal AI", "🏢 Corporate Analysis"])
    
    with tab1:
        st.markdown("""
        **Revolutionary Multi-Agent Document Analysis**
        
        - 🧠 **deepseek-llm:67b** - Master analyst for complex legal synthesis
        - ⚖️ **mixtral** - Legal expert for contract analysis and risk assessment  
        - 📝 **deepseek-llm:7b** - Content processor for summaries and extraction
        - 🔍 **mistral** - Information specialist for entities and dates
        - ⚡ **phi3** - Quick responder for fast analysis
        
        **Smart Features:**
        - Automatically selects the best agents for your query
        - Runs multiple models in parallel for comprehensive analysis
        - Provides agent execution breakdowns with confidence scores
        - Synthesizes results from all agents into coherent answers
        """)
    
    with tab2:
        st.markdown("""
        **Professional Legal AI Consultation**
        
        🔹 **Direct Legal Guidance** - Ask questions without uploading documents  
        🔹 **Citation Verification** - Automatic validation of legal references  
        🔹 **Protocol Compliance** - Quality control and hallucination prevention  
        🔹 **Export Capabilities** - Professional DOCX reports  
        
        **Legal Topic Specializations:**
        - Corporate Governance & Compliance
        - Contract Law & Commercial Agreements  
        - Employment Law & HR Issues
        - Intellectual Property & Technology
        - Litigation & Dispute Resolution
        - And 5 more specialized areas...
        """)
    
    with tab3:
        st.markdown("""
        **UK Corporate Intelligence**
        
        🔹 **Companies House Integration** - Real-time UK company data  
        🔹 **Group Structure Mapping** - Corporate hierarchy visualization  
        🔹 **Document Analysis** - AI-powered filing summaries  
        🔹 **Batch Processing** - Multi-company analysis workflows  
        
        **Document Types Supported:**
        - Annual accounts and financial statements
        - Confirmation statements and officer details
        - Share capital changes and charges
        - Insolvency and dissolution records
        """)
    
    # Quick start
    st.markdown("### 🚀 Ready to Start?")
    
    start_col1, start_col2 = st.columns(2)
    
    with start_col1:
        st.success("""
        **🎯 For Legal Questions:**
        1. Go to **📚 Document RAG** tab
        2. Upload your legal documents
        3. Ask questions in natural language
        4. Watch the multi-agent system work!
        """)
    
    with start_col2:
        st.info("""
        **🏢 For Company Research:**
        1. Go to **🏢 Companies House** tab
        2. Enter UK company numbers
        3. Select document types to analyze
        4. Get AI-powered insights
        """)

def render_getting_started():
    """Render the getting started guide"""
    
    st.markdown("## 🚀 Getting Started with Strategic Counsel")
    
    # System check
    with st.expander("🔍 System Status Check", expanded=True):
        st.markdown("**Before we begin, let's check your system:**")
        
        try:
            # Check if multi-agent system is available
            try:
                from multi_agent_rag_orchestrator import get_orchestrator
                st.success("✅ Multi-Agent RAG System: Ready")
            except ImportError:
                st.warning("⚠️ Multi-Agent RAG System: Not available (falling back to single model)")
            
            # Check Ollama
            try:
                from local_rag_pipeline import rag_session_manager
                st.success("✅ RAG Pipeline: Available")
            except ImportError:
                st.error("❌ RAG Pipeline: Not available")
                
        except Exception as e:
            st.error(f"System check failed: {e}")
    
    # Step-by-step guide
    st.markdown("### 📋 Step-by-Step Guide")
    
    step_tab1, step_tab2, step_tab3 = st.tabs(["📚 Document Analysis", "🤖 AI Consultation", "🏢 Company Research"])
    
    with step_tab1:
        st.markdown("#### 🎯 Multi-Agent Document Analysis")
        
        with st.expander("📖 Step 1: Upload Documents", expanded=True):
            st.markdown("""
            **Go to the 📚 Document RAG tab:**
            
            1. **Select your matter/topic** in the sidebar (e.g., "Client ABC Contract Review")
            2. **Upload documents** - PDF, DOCX, TXT, DOC, RTF files supported
            3. **Wait for processing** - Documents are chunked and vectorized
            4. **See the status** - Check document count and vector index size
            
            **💡 Tips:**
            - Upload related documents together for better cross-analysis
            - Use descriptive matter names for organization
            - Larger documents take longer to process but provide richer context
            """)
        
        with st.expander("🤖 Step 2: Ask Questions"):
            st.markdown("""
            **The magic happens here:**
            
            1. **Type your question** in natural language
            2. **Watch the multi-agent analysis**:
               - System analyzes your query complexity
               - Automatically assigns specialized agents
               - Shows real-time execution status
            3. **Review comprehensive results**:
               - Synthesized answer from multiple agents
               - Agent execution breakdown
               - Source citations with similarity scores
               - Task-specific findings
            
            **Example Questions:**
            - "What are the key risks and obligations in these contracts?"
            - "Who are the parties involved and what are their responsibilities?"
            - "Summarize the main legal issues and provide recommendations"
            - "Extract all important dates and deadlines"
            """)

def render_multi_agent_guide():
    """Render detailed multi-agent system guide"""
    
    st.markdown("## 🤖 Multi-Agent RAG System")
    
    st.success("""
    **Revolutionary AI orchestration:** Strategic Counsel automatically coordinates 5 specialized AI agents 
    to provide comprehensive legal analysis. No more manual model selection - the system intelligently 
    assigns the best agents for each task type!
    """)
    
    # Agent overview
    st.markdown("### 🎯 Your AI Legal Team")
    
    agent_col1, agent_col2 = st.columns(2)
    
    with agent_col1:
        with st.expander("🧠 deepseek-llm:67b - Master Analyst", expanded=True):
            st.markdown("""
            **Role:** Senior Legal Advisor
            
            **Specializations:**
            - Complex legal synthesis and integration
            - Compliance checking and regulatory analysis  
            - Final answer synthesis from all agents
            - Cross-jurisdictional legal analysis
            
            **When Active:** Complex queries requiring deep legal reasoning
            
            **Concurrency:** 1 task (large model, highest quality)
            """)
        
        with st.expander("⚖️ mixtral:latest - Legal Expert"):
            st.markdown("""
            **Role:** Contract & Risk Specialist
            
            **Specializations:**
            - Contract analysis and interpretation
            - Legal clause extraction and analysis
            - Risk assessment and identification
            - Obligation mapping and compliance
            
            **When Active:** Legal document analysis, contract review
            
            **Concurrency:** 2 parallel tasks
            """)

def render_all_features():
    """Render comprehensive features overview"""
    
    st.markdown("## 📚 Complete Feature Guide")
    
    st.info("Comprehensive feature documentation available here...")

def render_troubleshooting():
    """Render troubleshooting and support section"""
    
    st.markdown("## 🔧 Troubleshooting & Support")
    
    st.info("Troubleshooting guide available here...") 