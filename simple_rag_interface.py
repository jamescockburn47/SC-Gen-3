#!/usr/bin/env python3
"""
Simple RAG Interface
===================

Clean, minimal interface for document Q&A.
No clutter, just ask questions and get answers.
"""

import streamlit as st
import asyncio
from typing import Dict, Any, List
from local_rag_pipeline import rag_session_manager

async def simple_rag_query(query: str, matter_id: str) -> Dict[str, Any]:
    """Simple RAG query using the proper enhanced RAG pipeline with timeline optimization"""
    
    try:
        # Import the proper answer generation function
        from enhanced_rag_interface import get_protocol_compliant_answer
        
        # Detect if this is a timeline query
        timeline_keywords = ['timeline', 'chronological', 'sequence', 'dates', 'history', 'progression', 'order', 'when', 'date']
        is_timeline_query = any(keyword in query.lower() for keyword in timeline_keywords)
        
        # Use enhanced settings for timeline queries
        if is_timeline_query:
            max_chunks = 50  # Much more comprehensive for timelines
            st.info("🕒 **Timeline Query Detected** - Using comprehensive document coverage for chronological analysis")
        else:
            max_chunks = 25  # Standard enhanced retrieval
        
        # Use the sophisticated RAG pipeline with optimal settings
        result = await get_protocol_compliant_answer(
            query=query,
            matter_id=matter_id,
            model="mistral:latest",  # Use the best model for accuracy
            max_chunks=max_chunks,  # Dynamic based on query type
            anonymise=False  # Keep simple for this interface
        )
        
        # Return the result in the expected format
        return {
            'answer': result.get('answer', 'No answer generated'),
            'sources': result.get('sources', []),
            'chunks_used': result.get('context_chunks_used', 0),
            'model_used': result.get('model_used', 'mistral:latest'),
            'generation_time': result.get('generation_time', 0),
            'protocol_compliance': result.get('protocol_compliance', {}),
            'graph_data': None,  # Knowledge graph handled by the pipeline
            'is_timeline_query': is_timeline_query,
            'max_chunks_used': max_chunks
        }
        
    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'error': str(e)
        }

def render_simple_rag_interface():
    """Simple, clean RAG interface with minimal clutter"""
    
    st.title("📄 Document Q&A")
    st.markdown("Ask questions about your documents. Advanced AI analysis with simple interface.")
    
    try:
        # Get available pipelines
        manager = rag_session_manager
        
        # Use correct attribute name
        available_matters = list(manager.active_pipelines.keys()) if manager.active_pipelines else []
        
        if not available_matters:
            st.info("📭 No documents loaded yet. Please go to Document Management to upload documents first.")
            return
        
        # Simple matter selection
        if len(available_matters) == 1:
            matter_id = available_matters[0]
            st.info(f"📁 Using matter: **{matter_id}**")
        else:
            matter_id = st.selectbox(
                "📁 Select documents:",
                available_matters,
                key="matter_selection"
            )
        
        # Get pipeline and check status
        pipeline = manager.get_or_create_pipeline(matter_id)
        status = pipeline.get_document_status()
        
        if status['total_documents'] == 0:
            st.warning("📭 No documents found in this matter. Please upload documents first.")
            return
        
        # Show simple status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Documents", status['total_documents'])
        with col2:
            st.metric("📝 Text Chunks", status['total_chunks'])
        with col3:
            st.metric("🧠 AI Model", "Mistral", help="Using mistral:latest for best accuracy")
        
        st.markdown("---")
        
        # Simple query interface
        st.markdown("#### 💬 Ask your question:")
        
        # Quick query buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Summary", key="quick_summary"):
                st.session_state['simple_query'] = "Provide a comprehensive summary of the key points in these documents"
        with col2:
            if st.button("📋 Key Facts", key="quick_facts"):
                st.session_state['simple_query'] = "What are the most important facts and findings?"
        with col3:
            if st.button("⚖️ Legal Issues", key="quick_legal"):
                st.session_state['simple_query'] = "What are the main legal issues discussed?"
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            value=st.session_state.get('simple_query', ''),
            placeholder="Example: Who are the parties involved? What are the key allegations? What damages are claimed?",
            height=100,
            key="query_input"
        )
        
        # Search button
        if st.button("🔍 Ask AI", type="primary", disabled=not query.strip()):
            if query.strip():
                with st.spinner("🧠 AI is analyzing your documents and generating answer..."):
                    try:
                        result = asyncio.run(simple_rag_query(query.strip(), matter_id))
                        
                        if result['answer'] and not result['answer'].startswith('Error'):
                            # Display answer
                            st.markdown("#### ✅ AI Analysis:")
                            st.success(result['answer'])
                            
                            # Show AI performance metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Sources Used", len(result['sources']))
                            with col2:
                                chunks_used = result.get('chunks_used', 0)
                                max_chunks = result.get('max_chunks_used', 25)
                                st.metric("⚡ Chunks Analyzed", f"{chunks_used}/{max_chunks}")
                            with col3:
                                if result.get('generation_time'):
                                    st.metric("🕒 Response Time", f"{result['generation_time']:.1f}s")
                            
                            # Show timeline-specific enhancements
                            if result.get('is_timeline_query'):
                                st.info(f"🕒 **Timeline Analysis Active**: Analyzed {result.get('max_chunks_used', 50)} text sections with larger 600-word chunks for better chronological context preservation")
                            
                            # Protocol compliance indicator
                            compliance = result.get('protocol_compliance', {})
                            overall_score = compliance.get('overall_score', 0)
                            if overall_score >= 0.8:
                                st.success(f"🛡️ **High Quality Response**: {overall_score:.1%} protocol compliance")
                            elif overall_score >= 0.6:
                                st.warning(f"🛡️ **Good Response**: {overall_score:.1%} protocol compliance") 
                            else:
                                st.info(f"🛡️ **Response Generated**: {overall_score:.1%} protocol compliance")
                            
                            # Display sources
                            if result['sources']:
                                with st.expander(f"📚 **Sources** ({len(result['sources'])} documents analyzed)", expanded=False):
                                    for i, source in enumerate(result['sources'], 1):
                                        st.markdown(f"**Source {i}: {source['document']}**")
                                        if source.get('similarity_score'):
                                            st.markdown(f"*Relevance: {source['similarity_score']:.1%}*")
                                        st.text(source['text_preview'])
                                        if i < len(result['sources']):
                                            st.markdown("---")
                            
                        else:
                            st.error(f"❌ {result['answer']}")
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("💡 Try refreshing the page or rephrasing your question.")
        
        # Clear query button
        if st.button("🗑️ Clear", key="clear_query"):
            st.session_state['simple_query'] = ''
            st.rerun()
        
        # Show system info
        with st.expander("ℹ️ System Information", expanded=False):
            st.markdown("**Enhanced Features Active:**")
            st.markdown("- 🧠 **Mistral AI** for intelligent analysis")
            st.markdown("- 🔍 **25 chunks** enhanced retrieval")
            st.markdown("- 🛡️ **Protocol compliance** checking")
            st.markdown("- 📊 **Citation tracking** for source verification")
            st.markdown("- ⚡ **GPU acceleration** for fast processing")
    
    except Exception as e:
        st.error(f"❌ Error accessing documents: {str(e)}")
        
        # Show debug info
        with st.expander("🔧 Debug Information", expanded=False):
            st.code(f"Error: {str(e)}")
            st.code(f"Available session manager: {hasattr(globals(), 'rag_session_manager')}")
            if 'rag_session_manager' in globals():
                manager = globals()['rag_session_manager']
                st.code(f"Manager type: {type(manager)}")
                st.code(f"Manager attributes: {dir(manager)}") 