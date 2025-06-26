#!/usr/bin/env python3
"""
Simple test to demonstrate MCP protocol enforcement
"""

import asyncio
from local_rag_pipeline import rag_session_manager

async def test_protocols():
    """Test MCP protocol enforcement"""
    
    print("🔍 MCP PROTOCOL ENFORCEMENT TEST")
    print("=" * 50)
    
    pipeline = rag_session_manager.get_or_create_pipeline('Corporate Governance')
    query = "What is the case number?"
    
    # Test with protocol enforcement
    print("WITH Protocol Enforcement:")
    result = await pipeline.generate_rag_answer(
        query, 'phi3:latest', 
        max_context_chunks=2, 
        temperature=0.1, 
        enforce_protocols=True
    )
    
    print(f"  Protocol Enforced: {result.get('protocol_enforcement_requested')}")
    print(f"  Protocol Compliant: {result.get('protocol_compliance')}")
    if result.get('compliance_message'):
        print(f"  Compliance Message: {result['compliance_message']}")
    print(f"  Answer: {result.get('answer', 'No answer')[:150]}...")
    
    print("\n" + "-" * 50)
    
    # Test without protocol enforcement
    print("WITHOUT Protocol Enforcement:")
    result2 = await pipeline.generate_rag_answer(
        query, 'phi3:latest', 
        max_context_chunks=2, 
        temperature=0.1, 
        enforce_protocols=False
    )
    
    print(f"  Protocol Enforced: {result2.get('protocol_enforcement_requested')}")
    print(f"  Answer: {result2.get('answer', 'No answer')[:150]}...")

if __name__ == "__main__":
    asyncio.run(test_protocols()) 