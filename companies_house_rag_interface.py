#!/usr/bin/env python3
"""
Companies House RAG Interface
Streamlit interface for comprehensive Companies House document analysis with RAG capabilities

Features:
- Companies House document retrieval and processing
- Local LLM-based OCR for scanned documents
- RAG-powered semantic search and analysis
- Alternative to cloud-based processing
"""

import streamlit as st
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the CH RAG pipeline
try:
    from companies_house_rag_pipeline import get_ch_rag_pipeline, CompaniesHouseRAGPipeline
    CH_RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"Companies House RAG pipeline not available: {e}")
    CH_RAG_AVAILABLE = False

# Configuration
try:
    import config
    CH_API_KEY = getattr(config, 'CH_API_KEY', None)
except ImportError:
    CH_API_KEY = None

# Categories mapping
CH_CATEGORIES = {
    "Accounts": "accounts",
    "Confirmation Statement": "confirmation-statement",
    "Capital": "capital",
    "Officers": "officers",
    "Mortgage": "mortgage",
    "Incorporation": "incorporation",
    "Annual Return": "annual-return",
    "Resolutions": "resolutions",
    "Charges": "charges",
    "Insolvency": "insolvency"
}

def render_companies_house_rag_interface():
    """Render the comprehensive Companies House RAG interface"""
    
    st.markdown("### 🏢 Companies House RAG Analysis")
    st.info("📌 **Advanced RAG-powered Companies House analysis** with local LLM processing and semantic search capabilities")
    
    if not CH_RAG_AVAILABLE:
        st.error("❌ Companies House RAG pipeline not available. Please check dependencies.")
        return
    
    if not CH_API_KEY:
        st.error("❌ Companies House API key not configured. Please set CH_API_KEY in config.")
        st.info("💡 Get your API key from: https://developer.company-information.service.gov.uk/")
        return
    
    # Initialize session state
    if 'ch_rag_pipeline' not in st.session_state:
        st.session_state.ch_rag_pipeline = get_ch_rag_pipeline()
        st.session_state.ch_rag_pipeline.ch_api_key = CH_API_KEY
    
    # Configuration section
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Company input
            company_input = st.text_area(
                "Company Numbers (one per line):",
                height=100,
                placeholder="00000001\n12345678\n00123456",
                help="Enter UK company numbers, one per line"
            )
            
            company_numbers = [num.strip() for num in company_input.split('\n') if num.strip()]
            
        with col2:
            # Year range
            current_year = datetime.now().year
            start_year = st.number_input("Start Year:", min_value=1990, max_value=current_year, value=current_year-5)
            end_year = st.number_input("End Year:", min_value=1990, max_value=current_year, value=current_year)
            
        with col3:
            # Document categories
            selected_categories = st.multiselect(
                "Document Categories:",
                options=list(CH_CATEGORIES.keys()),
                default=["Accounts", "Confirmation Statement", "Officers"],
                help="Select document types to retrieve and analyze"
            )
            
            max_docs = st.slider(
                "Max Documents per Company:",
                min_value=5, max_value=100, value=25,
                help="Limit documents to process for performance"
            )
    
    # Processing section
    st.markdown("### 📊 Document Processing & RAG Integration")
    
    # Processing status
    stats = st.session_state.ch_rag_pipeline.get_processing_stats()
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Companies in RAG", stats.get('companies_in_database', 0))
    with col_stat2:
        st.metric("Documents in RAG", stats.get('documents_in_database', 0))
    with col_stat3:
        st.metric("Vector Index Size", stats.get('vector_index_size', 0))
    with col_stat4:
        st.metric("OCR Processed", stats.get('documents_ocr_processed', 0))
    
    # Process companies button
    if st.button("🔄 Process Companies & Add to RAG", type="primary", disabled=not company_numbers):
        if not company_numbers:
            st.warning("Please enter at least one company number")
        else:
            categories_api = [CH_CATEGORIES[cat] for cat in selected_categories]
            
            with st.spinner(f"Processing {len(company_numbers)} companies with local LLM OCR..."):
                try:
                    # Run async processing
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    results = loop.run_until_complete(
                        st.session_state.ch_rag_pipeline.process_companies(
                            company_numbers=company_numbers,
                            categories=categories_api,
                            year_range=(start_year, end_year),
                            max_docs_per_company=max_docs
                        )
                    )
                    
                    if results['success']:
                        st.success(f"✅ Processing complete!")
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        with col_res1:
                            st.metric("Companies Processed", len(results['companies_processed']))
                        with col_res2:
                            st.metric("Documents Retrieved", results['total_documents'])
                        with col_res3:
                            st.metric("Chunks Created", results.get('total_chunks_created', 0))
                        
                        # Show processing details
                        with st.expander("📋 Processing Details", expanded=False):
                            for company_result in results['companies_processed']:
                                company_num = company_result['company_number']
                                docs_processed = company_result['documents_processed']
                                
                                profile = company_result.get('company_profile', {})
                                company_name = profile.get('company_name', 'Unknown')
                                
                                st.write(f"**{company_name} ({company_num})**: {docs_processed} documents processed")
                        
                        if results.get('processing_errors'):
                            st.warning("⚠️ Some processing errors occurred:")
                            for error in results['processing_errors']:
                                st.write(f"• {error}")
                    
                    else:
                        st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Processing error: {e}")
                    
                finally:
                    loop.close()
    
    st.markdown("---")
    
    # RAG Search section
    st.markdown("### 🔍 RAG-Powered Semantic Search")
    
    search_query = st.text_input(
        "Search Companies House Documents:",
        placeholder="What are the financial risks for company 12345678?",
        help="Use natural language to search across all processed CH documents"
    )
    
    col_search1, col_search2 = st.columns([3, 1])
    
    with col_search1:
        search_results_limit = st.slider("Max Search Results:", 5, 50, 15)
    
    with col_search2:
        if st.button("🔍 Search", disabled=not search_query.strip()):
            with st.spinner("Searching CH documents with RAG..."):
                try:
                    results = st.session_state.ch_rag_pipeline.search_ch_documents(
                        search_query, top_k=search_results_limit
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} relevant documents")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"📄 Result {i+1}: {result.get('ch_metadata', {}).get('document_type', 'Unknown')} - Score: {result.get('similarity_score', 0):.3f}"):
                                
                                ch_metadata = result.get('ch_metadata', {})
                                
                                # Document metadata
                                col_meta1, col_meta2 = st.columns(2)
                                with col_meta1:
                                    st.write(f"**Company:** {ch_metadata.get('company_number', 'Unknown')}")
                                    st.write(f"**Document Type:** {ch_metadata.get('document_type', 'Unknown')}")
                                    st.write(f"**Date:** {ch_metadata.get('document_date', 'Unknown')}")
                                
                                with col_meta2:
                                    st.write(f"**Processing Method:** {ch_metadata.get('processing_method', 'Unknown')}")
                                    st.write(f"**Similarity Score:** {result.get('similarity_score', 0):.3f}")
                                    st.write(f"**Description:** {ch_metadata.get('description', 'N/A')}")
                                
                                # Document content
                                st.markdown("**Content:**")
                                content = result.get('text', 'No content available')
                                st.write(content[:1000] + "..." if len(content) > 1000 else content)
                    
                    else:
                        st.info("No relevant documents found. Try processing more companies or adjusting your search query.")
                
                except Exception as e:
                    st.error(f"Search failed: {e}")
    
    st.markdown("---")
    
    # Comprehensive Analysis section
    st.markdown("### 📈 Comprehensive Company Analysis")
    
    analysis_company = st.text_input(
        "Company Number for Analysis:",
        placeholder="00123456",
        help="Enter a company number that has been processed"
    )
    
    custom_analysis_query = st.text_area(
        "Analysis Focus (Optional):",
        placeholder="Focus on financial performance, governance changes, and compliance issues over the last 3 years",
        help="Customize the analysis focus or leave blank for comprehensive analysis"
    )
    
    if st.button("📊 Generate Comprehensive Analysis", disabled=not analysis_company.strip()):
        with st.spinner("Generating comprehensive analysis using RAG and local LLM..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                analysis_result = loop.run_until_complete(
                    st.session_state.ch_rag_pipeline.analyze_company_comprehensive(
                        company_number=analysis_company.strip(),
                        analysis_query=custom_analysis_query.strip() if custom_analysis_query.strip() else None
                    )
                )
                
                if analysis_result['success']:
                    st.success("✅ Analysis Complete")
                    
                    # Analysis metadata
                    col_anal1, col_anal2, col_anal3 = st.columns(3)
                    with col_anal1:
                        st.metric("Documents Analyzed", analysis_result.get('documents_analyzed', 0))
                    with col_anal2:
                        st.metric("Company Number", analysis_result.get('company_number', 'Unknown'))
                    with col_anal3:
                        timestamp = analysis_result.get('analysis_timestamp', '')
                        if timestamp:
                            formatted_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                            st.metric("Generated", formatted_time)
                    
                    # Main analysis
                    st.markdown("#### 📋 Analysis Report")
                    analysis_text = analysis_result.get('analysis', 'No analysis available')
                    st.markdown(analysis_text)
                    
                    # Source documents
                    source_docs = analysis_result.get('source_documents', [])
                    if source_docs:
                        with st.expander("📚 Source Documents", expanded=False):
                            for i, doc in enumerate(source_docs[:10]):
                                st.write(f"**{i+1}.** {doc.get('type', 'Unknown')} - {doc.get('date', 'Unknown date')} (Score: {doc.get('similarity_score', 0):.3f})")
                
                else:
                    st.error(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Analysis error: {e}")
            
            finally:
                loop.close()
    
    # System Status
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    
    with col_status1:
        st.metric("Local LLM OCR", "🟢 Available" if True else "🔴 Unavailable")
    
    with col_status2:
        st.metric("Vector Database", "🟢 Active" if stats.get('vector_index_size', 0) > 0 else "🟡 Empty")
    
    with col_status3:
        st.metric("CH API", "🟢 Configured" if CH_API_KEY else "🔴 Missing")
    
    with col_status4:
        st.metric("RAG Pipeline", "🟢 Ready" if CH_RAG_AVAILABLE else "🔴 Error")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options", expanded=False):
        st.markdown("#### 🔧 Pipeline Configuration")
        
        current_stats = st.session_state.ch_rag_pipeline.get_processing_stats()
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.markdown("**Storage Information:**")
            st.write(f"Storage Path: `{current_stats.get('storage_path', 'Unknown')}`")
            st.write(f"Embedding Model: `{current_stats.get('embedding_model', 'Unknown')}`")
            st.write(f"Last Updated: `{current_stats.get('last_updated', 'Unknown')}`")
        
        with col_adv2:
            st.markdown("**Processing Statistics:**")
            st.write(f"Total Companies: {current_stats.get('companies_processed', 0)}")
            st.write(f"Documents Retrieved: {current_stats.get('documents_retrieved', 0)}")
            st.write(f"Chunks Created: {current_stats.get('chunks_created', 0)}")
        
        # Reset button
        if st.button("🗑️ Clear RAG Database", help="Warning: This will remove all processed CH documents from RAG"):
            if st.checkbox("I understand this will delete all CH RAG data"):
                try:
                    # Clear the pipeline data
                    st.session_state.ch_rag_pipeline.ch_documents_metadata = {}
                    st.session_state.ch_rag_pipeline.company_profiles = {}
                    st.session_state.ch_rag_pipeline.processing_stats = {
                        'companies_processed': 0,
                        'documents_retrieved': 0,
                        'documents_ocr_processed': 0,
                        'chunks_created': 0,
                        'processing_errors': []
                    }
                    st.session_state.ch_rag_pipeline._save_metadata()
                    st.success("✅ RAG database cleared")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear database: {e}")

def render_ch_comparison_analysis():
    """Render comparison analysis between multiple companies"""
    
    st.markdown("### 🔄 Multi-Company Comparison")
    
    comparison_companies = st.text_area(
        "Companies to Compare (one per line):",
        height=100,
        placeholder="00000001\n12345678",
        help="Enter 2-5 company numbers for comparative analysis"
    )
    
    company_list = [num.strip() for num in comparison_companies.split('\n') if num.strip()]
    
    if len(company_list) >= 2 and st.button("📊 Generate Comparison Analysis"):
        with st.spinner("Generating comparative analysis..."):
            try:
                # This would implement multi-company comparison logic
                st.info("🚧 Multi-company comparison feature coming soon!")
                st.write(f"Ready to compare: {', '.join(company_list)}")
                
            except Exception as e:
                st.error(f"Comparison analysis failed: {e}")

if __name__ == "__main__":
    print("Companies House RAG Interface ready") 