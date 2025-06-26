#!/usr/bin/env python3
"""
Quick Working RAG - Minimal Example with Timeout
"""

import asyncio
import aiohttp
from local_rag_pipeline import rag_session_manager

async def quick_rag_with_timeout(query: str, timeout_seconds: int = 20):
    """RAG query with timeout protection"""
    
    try:
        pipeline = rag_session_manager.get_or_create_pipeline('Corporate Governance')
        
        # Check documents
        status = pipeline.get_document_status()
        print(f"📚 Found {status['total_documents']} documents, {status['total_chunks']} chunks")
        
        if status['total_chunks'] == 0:
            return "❌ No documents loaded"
        
        # Search documents
        print("🔍 Searching documents...")
        chunks = pipeline.search_documents(query, top_k=3)
        
        if not chunks:
            return "❌ No relevant chunks found"
        
        print(f"✅ Found {len(chunks)} relevant chunks")
        
        # Build simple context
        context = "\\n\\n".join([f"[{i+1}] {chunk['text'][:300]}..." for i, chunk in enumerate(chunks)])
        
        # Simple prompt
        prompt = f"""Based on the following documents, answer this question: {query}

Documents:
{context}

Answer based only on the documents above:"""
        
        print("🤖 Generating answer with timeout protection...")
        
        # Direct Ollama call with timeout
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as session:
            payload = {
                "model": "phi3:latest",
                "prompt": prompt,
                "temperature": 0.1,
                "stream": False
            }
            
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get('response', 'No response generated')
                    
                    print(f"✅ Answer generated ({len(answer)} chars)")
                    return answer
                else:
                    error_text = await response.text()
                    return f"❌ API Error {response.status}: {error_text}"
    
    except asyncio.TimeoutError:
        return f"❌ Timeout after {timeout_seconds} seconds - model too slow"
    except Exception as e:
        return f"❌ Error: {str(e)}"

async def main():
    """Test the quick RAG system"""
    
    print("🧪 Quick Working RAG Test")
    print("=" * 40)
    
    # Test with short timeout
    query = "Who are the main parties in this case?"
    print(f"❓ Query: {query}")
    print(f"⏱️  Timeout: 20 seconds")
    print()
    
    answer = await quick_rag_with_timeout(query, 20)
    print()
    print("💬 Answer:")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main()) 