#!/usr/bin/env python3
"""
Fast RAG - Optimized for Speed
Uses shorter context and optimized model settings
"""

import asyncio
import aiohttp
from local_rag_pipeline import rag_session_manager

async def fast_rag_answer(query: str, max_context_length: int = 500):
    """Ultra-fast RAG with minimal context"""
    
    print("🚀 Fast RAG - Optimized for Speed")
    print("=" * 40)
    
    try:
        # Get pipeline
        pipeline = rag_session_manager.get_or_create_pipeline('Corporate Governance')
        status = pipeline.get_document_status()
        
        print(f"📚 Documents: {status['total_documents']}, Chunks: {status['total_chunks']}")
        
        if status['total_chunks'] == 0:
            return "❌ No documents loaded"
        
        # Quick search with limited results
        print("🔍 Quick search...")
        chunks = pipeline.search_documents(query, top_k=2)  # Reduced from 5 to 2
        
        if not chunks:
            return "❌ No relevant content found"
        
        print(f"✅ Found {len(chunks)} chunks")
        
        # Build minimal context (much shorter)
        context_parts = []
        for i, chunk in enumerate(chunks):
            short_text = chunk['text'][:max_context_length]  # Limit chunk size
            context_parts.append(f"[{i+1}] {short_text}")
        
        context = "\\n".join(context_parts)
        
        # Ultra-simple prompt for speed
        prompt = f"Question: {query}\\n\\nContext: {context}\\n\\nBrief answer:"
        
        print(f"🤖 Generating answer (context: {len(context)} chars)...")
        
        # Optimized Ollama call
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            payload = {
                "model": "phi3:latest",
                "prompt": prompt,
                "temperature": 0.0,  # More deterministic
                "top_p": 0.9,
                "num_predict": 100,  # Limit response length for speed
                "stream": False
            }
            
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get('response', 'No response').strip()
                    
                    print(f"✅ Generated: {len(answer)} chars")
                    return answer
                else:
                    return f"❌ API Error: {response.status}"
    
    except asyncio.TimeoutError:
        return "❌ Still too slow - try restarting Ollama"
    except Exception as e:
        return f"❌ Error: {str(e)}"

async def main():
    """Run fast RAG test"""
    
    queries = [
        "Who are the parties?",
        "What is the case number?", 
        "What are the main claims?"
    ]
    
    for query in queries:
        print(f"\\n❓ Query: {query}")
        answer = await fast_rag_answer(query, max_context_length=300)
        print(f"💬 Answer: {answer}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main()) 