#!/usr/bin/env python3
"""
Quick RAG Fix - Bypass Multi-Agent Issues
Simple, working RAG interface while multi-agent system gets fixed
"""

import asyncio
import streamlit as st
from local_rag_pipeline import rag_session_manager

async def simple_rag_query(matter_id: str, query: str, model: str = "phi3:latest"):
    """Simple RAG query without multi-agent complexity"""
    
    try:
        # Get pipeline
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        
        # Check documents
        status = pipeline.get_document_status()
        if status['total_documents'] == 0:
            return {
                'success': False,
                'error': 'No documents loaded',
                'answer': '',
                'sources': []
            }
        
        # Generate answer
        result = await pipeline.generate_rag_answer(
            query, model, max_context_chunks=5, temperature=0.1
        )
        
        # Ensure required fields exist
        result['success'] = True
        result['context_chunks'] = result.get('context_chunks', len(result.get('sources', [])))
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'answer': f'Error: {str(e)}',
            'sources': [],
            'context_chunks': 0
        }

def render_simple_rag_ui():
    """Render a simple RAG interface without multi-agent complications"""
    
    st.markdown("### 🔧 Simple RAG Interface (Multi-Agent Bypassed)")
    st.info("This is a simplified interface while we fix the multi-agent system issues.")
    
    # Model selection
    available_models = ["phi3:latest", "deepseek-llm:7b", "mistral:latest", "mixtral:latest"]
    selected_model = st.selectbox(
        "Select Model:",
        available_models,
        index=0,  # Default to phi3 (fastest)
        help="phi3=fastest, deepseek-7b=balanced, mistral=professional, mixtral=complex"
    )
    
    # Query input
    query = st.text_area(
        "Ask about your documents:",
        height=100,
        placeholder="What are the key legal issues in this case?"
    )
    
    if st.button("🧠 Generate Answer", type="primary", disabled=not query.strip()):
        if query.strip():
            with st.spinner(f"Generating answer with {selected_model}..."):
                # Run async function
                result = asyncio.run(simple_rag_query(
                    "Corporate Governance", 
                    query, 
                    selected_model
                ))
                
                if result['success']:
                    st.markdown("#### 🤖 Answer")
                    st.markdown(result['answer'])
                    
                    # Sources
                    if result.get('sources'):
                        st.markdown("#### 📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {i}: {source['document']} (Score: {source['similarity_score']:.3f})"):
                                st.write(source['text_preview'])
                    
                    # Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sources Used", len(result.get('sources', [])))
                    with col2:
                        st.metric("Context Chunks", result.get('context_chunks', 0))
                    with col3:
                        st.metric("Model", selected_model)
                
                else:
                    st.error(f"❌ {result['error']}")

if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else "phi3:latest"
        
        print(f"Testing RAG: {query} with {model}")
        result = asyncio.run(simple_rag_query("Corporate Governance", query, model))
        
        if result['success']:
            print(f"✅ Answer: {result['answer'][:200]}...")
            print(f"📚 Sources: {len(result['sources'])}")
        else:
            print(f"❌ Error: {result['error']}")
    else:
        print("Usage: python3 quick_rag_fix.py 'your question' [model_name]")
        print("Available models: phi3:latest, deepseek-llm:7b, mistral:latest, mixtral:latest") 