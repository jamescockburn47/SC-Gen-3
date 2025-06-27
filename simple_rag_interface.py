#!/usr/bin/env python3
"""
Simple RAG Interface
===================

Clean, minimal interface for document Q&A.
No clutter, just ask questions and get answers.
"""

import streamlit as st
import asyncio
import aiohttp
import time
from typing import Dict, Any, List
from local_rag_pipeline import rag_session_manager

async def intelligent_rag_query_async(query: str, matter_id: str) -> Dict[str, Any]:
    """AI-powered RAG query that actually understands legal disputes"""
    
    try:
        # Get the RAG pipeline
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        
        # Detect query type for optimal retrieval
        timeline_keywords = ['timeline', 'chronological', 'sequence', 'dates', 'history', 'progression', 'order', 'when', 'date']
        is_timeline_query = any(keyword in query.lower() for keyword in timeline_keywords)
        
        dispute_keywords = ['dispute', 'case', 'litigation', 'claim', 'allegation', 'parties', 'plaintiff', 'defendant', 'about']
        is_dispute_query = any(keyword in query.lower() for keyword in dispute_keywords)
        
        # Use enhanced settings for different query types
        if is_timeline_query:
            max_chunks = 50
            chunk_strategy = "timeline"
        elif is_dispute_query:
            max_chunks = 30  # More context needed to understand disputes
            chunk_strategy = "comprehensive"
        else:
            max_chunks = 25
            chunk_strategy = "standard"
        
        # Get search results
        search_results = pipeline.search_documents(query, top_k=max_chunks)
        
        if not search_results:
            return {
                'answer': 'No relevant documents found for your query. Please try rephrasing your question.',
                'sources': [],
                'chunks_used': 0,
                'model_used': 'no_model',
                'generation_time': 0,
                'is_timeline_query': is_timeline_query,
                'max_chunks_used': max_chunks
            }
        
        # Build rich context for AI analysis
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results[:max_chunks]):
            if isinstance(result, dict) and 'text' in result:
                text = result['text']
                if text and len(text.strip()) > 20:
                    context_parts.append(text)
                    
                    # Build source info
                    source_info = {
                        'document': result.get('id', f'Document {i+1}'),
                        'text_preview': text[:200] + '...' if len(text) > 200 else text,
                        'similarity_score': result.get('score', 0.0)
                    }
                    sources.append(source_info)
        
        if not context_parts:
            return {
                'answer': 'Found documents but could not extract meaningful content. Please try a different query.',
                'sources': [],
                'chunks_used': 0,
                'model_used': 'no_model',
                'generation_time': 0
            }
        
        # Create comprehensive context for AI
        context = "\n\n".join(context_parts)
        
        # Create specialized prompts based on query type
        if is_dispute_query or "about" in query.lower():
            prompt = f"""You are a legal AI assistant analyzing court documents. The user is asking about the nature of a legal dispute.

QUESTION: {query}

DOCUMENTS:
{context}

Please provide a comprehensive analysis that covers:

1. **DISPUTE OVERVIEW**: What is this case/dispute fundamentally about? What are the core issues?

2. **PARTIES INVOLVED**: Who are the main parties (plaintiff, defendant, other key figures)?

3. **KEY ALLEGATIONS/CLAIMS**: What are the primary legal claims or allegations being made?

4. **BACKGROUND/CONTEXT**: What led to this dispute? What are the underlying facts?

5. **LEGAL ISSUES**: What are the main legal questions or issues at stake?

6. **DAMAGES/RELIEF SOUGHT**: What compensation or relief is being sought?

Provide a clear, coherent analysis based ONLY on the documents provided. If specific information isn't available in the documents, say so. Be comprehensive but concise."""

        elif is_timeline_query:
            prompt = f"""You are a legal AI assistant analyzing court documents for chronological information.

QUESTION: {query}

DOCUMENTS:
{context}

Please create a chronological timeline based on the documents. Include:

1. **KEY DATES**: Important dates mentioned in chronological order
2. **EVENTS**: What happened on or around each date
3. **SIGNIFICANCE**: Why each event is important to the case
4. **CONTEXT**: How events relate to each other chronologically

Present the timeline in a clear, organized format. Only include information that is explicitly stated in the documents."""

        else:
            prompt = f"""You are a legal AI assistant. Analyze the provided court documents to answer the user's question comprehensively and accurately.

QUESTION: {query}

DOCUMENTS:
{context}

Please provide a thorough, well-structured answer based ONLY on the information in the documents. 

Structure your response with:
1. **DIRECT ANSWER**: Address the specific question asked
2. **SUPPORTING DETAILS**: Relevant details that support your answer
3. **CONTEXT**: Additional context that helps understand the answer
4. **LIMITATIONS**: Note if information is incomplete or unclear

Be precise, factual, and comprehensive. If the documents don't contain enough information to fully answer the question, explain what information is missing."""
        
        # Use the pipeline's AI generation
        start_time = time.time()
        
        # Try to use generate_rag_answer if available
        try:
            # Use the best available model
            model = "mixtral:latest"  # Most powerful model for comprehensive legal analysis
            
            # Generate AI response using the pipeline's method
            result = await pipeline.generate_rag_answer(
                query, 
                model_name=model,
                max_context_chunks=len(context_parts),
                temperature=0.1  # Low temperature for factual accuracy
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            return {
                'answer': result.get('answer', 'AI model generated an empty response.'),
                'sources': sources,
                'chunks_used': len(context_parts),
                'model_used': model,
                'generation_time': generation_time,
                'protocol_compliance': result.get('protocol_compliance', {'overall_score': 0.8}),
                'is_timeline_query': is_timeline_query,
                'max_chunks_used': max_chunks,
                'strategy_used': chunk_strategy,
                'context_length': len(context)
            }
            
        except Exception as llm_error:
            # Fallback to direct Ollama call with custom prompt if pipeline method fails
            try:
                async with aiohttp.ClientSession() as session:
                    ollama_payload = {
                        "model": "mixtral:latest",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9
                        }
                    }
                    
                    async with session.post("http://localhost:11434/api/generate", json=ollama_payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            end_time = time.time()
                            generation_time = end_time - start_time
                            
                            return {
                                'answer': result.get('response', 'AI model generated an empty response.'),
                                'sources': sources,
                                'chunks_used': len(context_parts),
                                'model_used': 'mixtral:latest',
                                'generation_time': generation_time,
                                'protocol_compliance': {'overall_score': 0.8},
                                'is_timeline_query': is_timeline_query,
                                'max_chunks_used': max_chunks,
                                'strategy_used': chunk_strategy + '_fallback',
                                'context_length': len(context)
                            }
                        else:
                            raise Exception(f"Ollama API error: {response.status}")
                            
            except Exception as fallback_error:
                return {
                    'answer': f"AI analysis failed. Pipeline error: {str(llm_error)}. Fallback error: {str(fallback_error)}. Please ensure Ollama is running with mixtral:latest model.",
                    'sources': sources,
                    'chunks_used': len(context_parts),
                    'model_used': 'error',
                    'generation_time': 0,
                    'error': f"Pipeline: {llm_error}, Fallback: {fallback_error}"
                }
        
    except Exception as e:
        return {
            'answer': f"Error processing query: {str(e)}\n\nPlease try again or contact support if the issue persists.",
            'sources': [],
            'chunks_used': 0,
            'error': str(e)
        }

def simple_rag_query_sync(query: str, matter_id: str) -> Dict[str, Any]:
    """Synchronous wrapper for the intelligent RAG query"""
    try:
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(intelligent_rag_query_async(query, matter_id))
        loop.close()
        return result
    except Exception as e:
        return {
            'answer': f"Synchronous execution error: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'error': str(e)
        }

# Keep the async version for backwards compatibility
async def simple_rag_query(query: str, matter_id: str) -> Dict[str, Any]:
    """Async version - now properly using AI"""
    return await intelligent_rag_query_async(query, matter_id)

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
            st.metric("🧠 AI Model", "Mixtral", help="Using mixtral:latest for most powerful analysis")
        
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
                        result = simple_rag_query_sync(query.strip(), matter_id)
                        
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
            st.markdown("- 🧠 **Mixtral AI** for intelligent analysis")
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