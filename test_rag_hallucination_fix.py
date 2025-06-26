#!/usr/bin/env python3
"""
Test script to verify RAG hallucination fixes
"""

import asyncio
import sys
from quick_rag_fix import get_non_hallucinating_answer
from local_rag_pipeline import rag_session_manager

async def test_rag_fixes():
    """Test the anti-hallucination RAG system"""
    
    print("🔍 Testing Anti-Hallucination RAG System")
    print("=" * 50)
    
    # Test queries that commonly cause hallucination
    test_queries = [
        "What is the case number?",
        "Who are the parties involved in this dispute?",
        "What are the main legal claims?",
        "What dates are mentioned in the documents?",
        "What are the key legal issues?"
    ]
    
    # Test with different models
    test_models = ["phi3:latest", "deepseek-llm:7b", "mistral:latest"]
    
    print(f"Testing {len(test_queries)} queries with {len(test_models)} models")
    print("-" * 50)
    
    # Check if documents are loaded
    pipeline = rag_session_manager.get_or_create_pipeline('Corporate Governance')
    status = pipeline.get_document_status()
    
    print(f"📄 Documents loaded: {status['total_documents']}")
    print(f"📄 Total chunks: {status['total_chunks']}")
    
    if status['total_documents'] == 0:
        print("❌ No documents loaded! Please upload documents first.")
        print("   1. Run: streamlit run app.py")
        print("   2. Go to Document RAG tab")
        print("   3. Upload some PDF/DOCX files")
        return
    
    print("-" * 50)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        for model in test_models:
            print(f"   Testing with {model}...")
            
            try:
                result = await get_non_hallucinating_answer(query, 'Corporate Governance', model)
                
                # Check for hallucination indicators
                answer = result['answer']
                flags = result.get('hallucination_flags', [])
                sources = result.get('sources', [])
                
                print(f"   ✅ {model}: {len(answer)} chars, {len(sources)} sources")
                
                if flags:
                    print(f"   ⚠️  Hallucination flags: {flags}")
                
                # Check for specific problematic patterns
                problematic_patterns = ['[OR:', '[DATE]', 'Page XX', '[UNVERIFIED]']
                found_patterns = [p for p in problematic_patterns if p in answer]
                
                if found_patterns:
                    print(f"   ❌ Found problematic patterns: {found_patterns}")
                    print(f"   📝 Answer preview: {answer[:100]}...")
                else:
                    print(f"   ✅ No problematic patterns detected")
                
                print(f"   📝 Answer preview: {answer[:100]}...")
                
            except Exception as e:
                print(f"   ❌ {model}: Error - {str(e)}")
            
            print()
    
    print("🎯 Test Summary:")
    print("- If you see '❌ Found problematic patterns', the hallucination issue persists")
    print("- If you see '✅ No problematic patterns detected', the fix is working")
    print("- Look for placeholder text like '[OR:', '[DATE]', 'Page XX' in answers")

async def test_single_query(query: str, model: str = "phi3:latest"):
    """Test a single query for detailed analysis"""
    
    print(f"🔍 Testing: {query}")
    print(f"🤖 Model: {model}")
    print("-" * 50)
    
    result = await get_non_hallucinating_answer(query, 'Corporate Governance', model)
    
    print(f"📋 Answer:")
    print(result['answer'])
    print()
    
    print(f"🚩 Hallucination flags: {result.get('hallucination_flags', 'None')}")
    print(f"📚 Sources found: {len(result.get('sources', []))}")
    print(f"🔧 Debug info: {result.get('debug_info', 'None')}")
    
    if result.get('sources'):
        print("\n📄 Sources:")
        for i, source in enumerate(result['sources']):
            print(f"   {i+1}. {source['document']} (Score: {source['similarity_score']:.3f})")
            print(f"      Preview: {source['text_preview'][:100]}...")

def main():
    """Main function to handle command line arguments"""
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else "phi3:latest"
        asyncio.run(test_single_query(query, model))
    else:
        asyncio.run(test_rag_fixes())

if __name__ == "__main__":
    main() 