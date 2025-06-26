#!/usr/bin/env python3
"""
Simple direct test to isolate the hanging issue
"""

import asyncio
import aiohttp

async def test_direct_ollama():
    """Test Ollama directly with aiohttp"""
    
    print("🔍 DIRECT OLLAMA TEST")
    print("=" * 30)
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": "phi3:latest",
                "prompt": "What is 2+2? Answer briefly.",
                "stream": False
            }
            
            print("Sending request to Ollama...")
            async with session.post(
                "http://localhost:11434/api/generate", 
                json=payload
            ) as response:
                print(f"Response status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    print(f"Response: {result.get('response', 'No response')}")
                else:
                    error_text = await response.text()
                    print(f"Error: {error_text}")
                    
    except asyncio.TimeoutError:
        print("❌ Request timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error: {e}")

async def test_rag_search_only():
    """Test just the RAG search without generation"""
    
    print("\n🔍 RAG SEARCH TEST")
    print("=" * 30)
    
    try:
        from local_rag_pipeline import rag_session_manager
        
        pipeline = rag_session_manager.get_or_create_pipeline('Corporate Governance')
        
        print("Testing document search...")
        results = pipeline.search_documents("case number", top_k=2)
        
        print(f"Found {len(results)} chunks")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.get('document', 'Unknown')} (score: {result.get('similarity_score', 0):.3f})")
            
    except Exception as e:
        print(f"❌ Search error: {e}")

async def main():
    await test_direct_ollama()
    await test_rag_search_only()

if __name__ == "__main__":
    asyncio.run(main()) 