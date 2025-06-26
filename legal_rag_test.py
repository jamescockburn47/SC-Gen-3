#!/usr/bin/env python3
"""
Legal RAG Test - With MCP Protocol Enforcement
Tests RAG system with Strategic Counsel protocols applied
"""

import asyncio
from local_rag_pipeline import rag_session_manager

async def protocol_enforced_rag_test(query: str, model: str = "phi3:latest"):
    """Test RAG with MCP protocol enforcement enabled"""
    
    print(f"⚖️  Protocol-Enforced Legal RAG Test")
    print(f"Query: {query}")
    print(f"Model: {model}")
    print("=" * 60)
    
    try:
        # Get pipeline  
        pipeline = rag_session_manager.get_or_create_pipeline('Corporate Governance')
        
        # Generate answer with MCP protocol enforcement
        result = await pipeline.generate_rag_answer(
            query, 
            model, 
            max_context_chunks=3, 
            temperature=0.1,
            enforce_protocols=True  # This enables MCP enforcement
        )
        
        # Display results
        if result.get('answer'):
            print("✅ SUCCESS!")
            
            # Protocol compliance status
            if result.get('protocol_compliance') is not None:
                compliance_status = "✅ COMPLIANT" if result['protocol_compliance'] else "⚠️ NON-COMPLIANT"
                print(f"🔒 Protocol Status: {compliance_status}")
                if result.get('compliance_message'):
                    print(f"   Compliance Note: {result['compliance_message']}")
            
            print(f"📄 Answer ({len(result['answer'])} chars):")
            print(result['answer'])
            print()
            
            if result.get('sources'):
                print("📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['document']} (Score: {source['similarity_score']:.3f})")
                    print(f"     Preview: {source['text_preview'][:100]}...")
                print()
            
            print(f"🔧 Metadata:")
            print(f"  Model: {result.get('model_used')}")
            print(f"  Chunks: {result.get('context_chunks', 'N/A')}")
            print(f"  Sources: {len(result.get('sources', []))}")
            print(f"  Protocol Enforcement: {result.get('protocol_enforcement_requested', 'Unknown')}")
            
            # Show any enforcement metadata
            if result.get('enforcement_metadata'):
                enforcement = result['enforcement_metadata']
                if enforcement.get('warnings'):
                    print(f"  ⚠️ Warnings: {len(enforcement['warnings'])}")
                    for warning in enforcement['warnings']:
                        print(f"     - {warning}")
            
        else:
            print("❌ FAILED - No answer generated")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_non_protocol_vs_protocol():
    """Compare results with and without protocol enforcement"""
    
    test_query = "What are the main legal claims in this case?"
    model = "phi3:latest"
    
    print("🔍 COMPARISON TEST: Protocol vs Non-Protocol")
    print("=" * 60)
    
    pipeline = rag_session_manager.get_or_create_pipeline('Corporate Governance')
    
    # Test without protocol enforcement
    print("\n1️⃣ WITHOUT MCP Protocol Enforcement:")
    print("-" * 40)
    try:
        result_no_protocol = await pipeline.generate_rag_answer(
            test_query, model, max_context_chunks=3, temperature=0.1, enforce_protocols=False
        )
        print(f"Answer: {result_no_protocol.get('answer', 'No answer')[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with protocol enforcement  
    print("\n2️⃣ WITH MCP Protocol Enforcement:")
    print("-" * 40)
    try:
        result_with_protocol = await pipeline.generate_rag_answer(
            test_query, model, max_context_chunks=3, temperature=0.1, enforce_protocols=True
        )
        print(f"Answer: {result_with_protocol.get('answer', 'No answer')[:200]}...")
        print(f"Protocol Compliant: {result_with_protocol.get('protocol_compliance', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    """Run comprehensive legal RAG tests with protocols"""
    
    legal_queries = [
        "What is the case number?",
        "Who are the claimants in this case?", 
        "What are the main legal claims being made?",
        "What court is this case filed in?",
        "What damages or remedies are being sought?",
        "Identify any contract terms mentioned in the documents",
        "What legal entities are involved?",
        "Are there any compliance obligations mentioned?"
    ]
    
    print("🧪 LEGAL RAG SYSTEM TEST - WITH MCP PROTOCOLS")
    print("Testing with Strategic Counsel Protocol enforcement")
    print("=" * 80)
    
    # First run comparison test
    await test_non_protocol_vs_protocol()
    print("\n" + "=" * 80)
    
    # Then run individual protocol-enforced tests
    for i, query in enumerate(legal_queries, 1):
        print(f"\n🔍 PROTOCOL TEST {i}/{len(legal_queries)}")
        await protocol_enforced_rag_test(query)
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 