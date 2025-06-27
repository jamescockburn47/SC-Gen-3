#!/usr/bin/env python3
"""
Test the new intelligent RAG functionality to verify it produces proper analysis
instead of just text fragments
"""

import asyncio
from simple_rag_interface import intelligent_rag_query_async

async def test_intelligent_rag():
    """Test the new AI-powered RAG system"""
    
    print("🧠 TESTING INTELLIGENT RAG SYSTEM")
    print("=" * 50)
    
    # Test queries that should get proper analysis
    test_queries = [
        "What is this dispute about?",
        "Who are the parties involved in this case?", 
        "What are the main legal issues?",
        "Summarize the key allegations"
    ]
    
    matter_id = "Corporate Governance"  # Or whatever matter you have documents in
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 30)
        
        try:
            result = await intelligent_rag_query_async(query, matter_id)
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                continue
                
            answer = result.get('answer', '')
            
            # Check if we got a proper analysis vs fragments
            if len(answer) > 200 and not answer.startswith('Error'):
                print(f"✅ Generated proper analysis ({len(answer)} chars)")
                
                # Check for structured analysis
                if '**' in answer or 'DISPUTE OVERVIEW' in answer or 'PARTIES INVOLVED' in answer:
                    print("✅ Answer has structured format")
                else:
                    print("⚠️  Answer lacks structured format")
                
                # Check if it explains the dispute coherently
                if any(word in answer.lower() for word in ['dispute', 'case', 'parties', 'claim', 'allegation']):
                    print("✅ Answer contains legal dispute terms")
                else:
                    print("⚠️  Answer may not explain the dispute properly")
                
                # Show preview
                preview = answer[:300].replace('\n', ' ')
                print(f"📝 Preview: {preview}...")
                
            else:
                print(f"❌ Answer too short or error: {answer[:100]}...")
                
            # Show metadata
            print(f"📊 Sources: {len(result.get('sources', []))}")
            print(f"🧠 Model: {result.get('model_used', 'Unknown')}")
            print(f"⚡ Generation Time: {result.get('generation_time', 0):.1f}s")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")

async def main():
    await test_intelligent_rag()

if __name__ == "__main__":
    asyncio.run(main()) 