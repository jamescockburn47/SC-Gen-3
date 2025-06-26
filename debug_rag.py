#!/usr/bin/env python3
"""
Debug RAG pipeline step by step
"""

import asyncio
import aiohttp
from local_rag_pipeline import rag_session_manager, create_general_legal_prompt

async def test_step_by_step():
    """Test each step of the RAG pipeline"""
    
    print("🔍 STEP-BY-STEP RAG DEBUG")
    print("=" * 40)
    
    # Step 1: Get pipeline
    print("Step 1: Getting RAG pipeline...")
    pipeline = rag_session_manager.get_or_create_pipeline('Corporate Governance')
    print("✅ Pipeline loaded")
    
    # Step 2: Search documents
    print("\nStep 2: Searching documents...")
    query = "What is the case number?"
    results = pipeline.search_documents(query, top_k=2)
    print(f"✅ Found {len(results)} chunks")
    
    # Step 3: Build context
    print("\nStep 3: Building context...")
    context_parts = []
    for i, chunk in enumerate(results):
        context_parts.append(f"[Source {i+1}] {chunk['text']}")
    context = "\n\n".join(context_parts)
    print(f"✅ Context built ({len(context)} chars)")
    
    # Step 4: Create prompt (without protocols first)
    print("\nStep 4: Testing simple prompt...")
    simple_prompt = f"""You are a legal AI assistant analyzing documents.

Query: {query}

Documents:
{context}

Answer based only on the documents provided:"""
    
    print(f"✅ Simple prompt created ({len(simple_prompt)} chars)")
    
    # Step 5: Test with simple prompt
    print("\nStep 5: Testing Ollama with simple prompt...")
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": "phi3:latest",
                "prompt": simple_prompt,
                "stream": False,
                "temperature": 0.1
            }
            
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get('response', 'No response')
                    print(f"✅ Simple prompt works! Answer: {answer[:100]}...")
                else:
                    print(f"❌ Error {response.status}")
                    return
    except Exception as e:
        print(f"❌ Simple prompt failed: {e}")
        return
    
    # Step 6: Test with full protocol prompt
    print("\nStep 6: Testing with strategic protocols...")
    try:
        protocols = pipeline.strategic_protocols
        protocol_prompt = create_general_legal_prompt(query, context, protocols)
        print(f"Protocol prompt length: {len(protocol_prompt)} chars")
        
        if len(protocol_prompt) > 16000:  # Check if prompt is too long
            print("⚠️ Protocol prompt is very long, might cause timeout")
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": "phi3:latest", 
                "prompt": protocol_prompt,
                "stream": False,
                "temperature": 0.1
            }
            
            print("Sending protocol prompt to Ollama...")
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get('response', 'No response')
                    print(f"✅ Protocol prompt works! Answer: {answer[:100]}...")
                else:
                    error_text = await response.text()
                    print(f"❌ Protocol prompt failed: {response.status} - {error_text}")
    except asyncio.TimeoutError:
        print("❌ Protocol prompt timed out (too long/complex)")
    except Exception as e:
        print(f"❌ Protocol prompt error: {e}")
    
    # Step 7: Test MCP enforcement
    print("\nStep 7: Testing MCP enforcement...")
    try:
        from mcp_rag_server import mcp_rag_server
        
        # Test query validation
        is_valid, msg, metadata = mcp_rag_server.validate_query('Corporate Governance', query)
        print(f"✅ Query validation: {is_valid} - {msg}")
        
        # Test response enforcement on mock result
        mock_result = {
            'answer': 'Test answer',
            'sources': [],
            'query': query,
            'model_used': 'phi3:latest'
        }
        
        is_compliant, compliance_msg, enforcement_metadata = mcp_rag_server.enforce_protocol_on_response(
            'Corporate Governance', mock_result
        )
        print(f"✅ Response enforcement: {is_compliant} - {compliance_msg}")
        
    except Exception as e:
        print(f"❌ MCP enforcement error: {e}")

if __name__ == "__main__":
    asyncio.run(test_step_by_step()) 