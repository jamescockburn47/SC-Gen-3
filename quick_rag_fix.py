#!/usr/bin/env python3
"""
Quick RAG fix for hallucination issues
This bypasses the multi-agent system and uses direct, controlled prompts
"""

import streamlit as st
import asyncio
import aiohttp
from local_rag_pipeline import rag_session_manager

def create_strict_prompt(query: str, context: str) -> str:
    """Create a strict prompt that prevents hallucination"""
    return f"""You are a document analysis assistant. You MUST ONLY use information from the provided documents below.

STRICT RULES:
- Only state facts that are explicitly written in the documents
- If information is not in the documents, say "This information is not provided in the documents"
- Quote directly from documents when possible
- Use the exact format [Source 1], [Source 2] etc. for citations
- Do NOT make assumptions or inferences beyond what is written
- Do NOT use placeholder text like "[DATE]" or "[Source X, Page XX]"

QUERY: {query}

DOCUMENTS:
{context}

RESPONSE (based ONLY on the provided documents):"""

async def get_non_hallucinating_answer(query: str, matter_id: str = 'Corporate Governance', model: str = 'phi3:latest'):
    """Get answer that strictly follows document content"""
    
    try:
        # Get RAG pipeline
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        
        # Search for relevant chunks
        chunks = pipeline.search_documents(query, top_k=5)
        
        if not chunks:
            return {
                'answer': "No relevant documents found for your query.",
                'sources': [],
                'model_used': model,
                'debug_info': 'No chunks returned from search'
            }
        
        # Build context with real content validation
        context_parts = []
        valid_sources = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '').strip()
            if len(chunk_text) > 20:  # Ensure chunk has meaningful content
                context_parts.append(f"[Source {i+1}] {chunk_text}")
                valid_sources.append({
                    'chunk_id': chunk['id'],
                    'document': chunk.get('document_info', {}).get('filename', 'Unknown'),
                    'similarity_score': chunk['similarity_score'],
                    'text_preview': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                })
        
        if not context_parts:
            return {
                'answer': "Document chunks found but contain insufficient content for analysis.",
                'sources': [],
                'model_used': model,
                'debug_info': 'Empty or too-short chunks'
            }
        
        context = "\n\n".join(context_parts)
        prompt = create_strict_prompt(query, context)
        
        # Generate answer with strict controls
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0,  # Most deterministic
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
            
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get('response', 'No response').strip()
                    
                    # Validate response for hallucination indicators
                    hallucination_flags = []
                    
                    # Check for placeholder patterns
                    placeholders = ['[OR:', '[DATE]', '[SOURCE X', 'Page XX', '[UNVERIFIED]']
                    for placeholder in placeholders:
                        if placeholder in answer:
                            hallucination_flags.append(f"Contains placeholder: {placeholder}")
                    
                    # Check for vague language
                    vague_phrases = ['I think', 'probably', 'might be', 'could be', 'generally speaking']
                    for phrase in vague_phrases:
                        if phrase.lower() in answer.lower():
                            hallucination_flags.append(f"Contains uncertain language: {phrase}")
                    
                    return {
                        'answer': answer,
                        'sources': valid_sources,
                        'model_used': model,
                        'context_chunks_used': len(context_parts),
                        'hallucination_flags': hallucination_flags,
                        'debug_info': f"Prompt length: {len(prompt)} chars, Context length: {len(context)} chars"
                    }
                else:
                    error_text = await response.text()
                    return {
                        'answer': f"Error: HTTP {response.status} - {error_text}",
                        'sources': [],
                        'model_used': model,
                        'debug_info': f"HTTP error: {response.status}"
                    }
    
    except Exception as e:
        return {
            'answer': f"System error: {str(e)}",
            'sources': [],
            'model_used': model,
            'debug_info': f"Exception: {type(e).__name__}: {str(e)}"
        }

def render_simple_rag_ui():
    """Render a simple RAG interface without multi-agent complications"""
    
    st.markdown("### 🔧 Anti-Hallucination RAG Interface")
    st.info("📌 This interface uses strict prompting to prevent hallucinations and placeholder responses.")
    
    # Model selection
    available_models = ["phi3:latest", "deepseek-llm:7b", "mistral:latest", "mixtral:latest"]
    selected_model = st.selectbox(
        "Select Model:",
        available_models,
        index=0,  # Default to phi3 (fastest)
        help="phi3=fastest, deepseek-7b=balanced, mistral=professional, mixtral=complex"
    )
    
    # Matter selection
    matter_options = ["Corporate Governance", "Contract Analysis", "Litigation Review", "Compliance Audit"]
    selected_matter = st.selectbox(
        "Select Matter:",
        matter_options,
        index=0
    )
    
    # Query input
    query = st.text_area(
        "Ask about your documents:",
        height=100,
        placeholder="What is the case number? Who are the parties involved? What are the key legal issues?"
    )
    
    if st.button("🧠 Generate Strict Answer", type="primary", disabled=not query.strip()):
        with st.spinner(f"Analyzing documents with {selected_model}..."):
            try:
                # Run async function
                result = asyncio.run(get_non_hallucinating_answer(query, selected_matter, selected_model))
                
                # Display results
                st.markdown("### 📋 Analysis Result")
                
                # Show answer
                st.markdown("#### Answer:")
                st.write(result['answer'])
                
                # Show hallucination warnings if any
                if result.get('hallucination_flags'):
                    st.warning("⚠️ Potential hallucination indicators detected:")
                    for flag in result['hallucination_flags']:
                        st.write(f"• {flag}")
                
                # Show sources
                if result.get('sources'):
                    st.markdown("#### Sources Used:")
                    for i, source in enumerate(result['sources']):
                        with st.expander(f"Source {i+1}: {source['document']} (Score: {source['similarity_score']:.3f})"):
                            st.write(source['text_preview'])
                
                # Show metadata
                with st.expander("🔍 Debug Information"):
                    st.write(f"**Model Used:** {result['model_used']}")
                    st.write(f"**Context Chunks:** {result.get('context_chunks_used', 0)}")
                    st.write(f"**Debug Info:** {result.get('debug_info', 'None')}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Test function for debugging
async def test_quick_fix():
    """Test the quick fix with a simple query"""
    query = "What is the case number?"
    result = await get_non_hallucinating_answer(query)
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result.get('sources', []))}")
    print(f"Hallucination flags: {result.get('hallucination_flags', [])}")

if __name__ == "__main__":
    print("Testing anti-hallucination RAG...")
    asyncio.run(test_quick_fix()) 