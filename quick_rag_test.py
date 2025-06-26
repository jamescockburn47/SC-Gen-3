#!/usr/bin/env python3
"""
Quick RAG Test with Working Models
Tests RAG system with your 123 chunks using appropriate models
"""

import asyncio
import time
from local_rag_pipeline import rag_session_manager

async def test_rag_system():
    """Test RAG system with your loaded documents"""
    
    print("🧪 Quick RAG Test")
    print("=" * 40)
    
    # Get RAG pipeline for test matter
    print("📚 Getting RAG pipeline...")
    pipeline = rag_session_manager.get_or_create_pipeline("test_matter")
    
    # Check document status
    status = pipeline.get_document_status()
    print(f"✅ Documents loaded: {status['total_documents']}")
    print(f"✅ Chunks available: {status['total_chunks']}")
    print(f"✅ Vector index size: {status['vector_index_size']}")
    
    if status['total_chunks'] == 0:
        print("❌ No documents loaded! Please upload documents first.")
        return
    
    # Test queries with appropriate models
    test_queries = [
        "What is this case about?",
        "Who are the main parties involved?",
        "What are the key legal issues mentioned?"
    ]
    
    # Use smaller, faster models
    test_models = ["phi3:latest", "deepseek-llm:7b", "mistral:latest"]
    
    print(f"\n🚀 Testing {len(test_queries)} queries with {len(test_models)} models...")
    
    results = {}
    
    for model in test_models:
        print(f"\n🧠 Testing model: {model}")
        model_results = []
        
        for i, query in enumerate(test_queries):
            print(f"   Query {i+1}: {query[:50]}...")
            
            start_time = time.time()
            
            try:
                # Test document search first
                search_results = pipeline.search_documents(query, top_k=3)
                
                if not search_results:
                    print(f"   ❌ No search results found")
                    continue
                
                # Generate RAG answer
                result = await pipeline.generate_rag_answer(
                    query, model, max_context_chunks=3, temperature=0.1
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if result.get('answer'):
                    print(f"   ✅ Response in {response_time:.1f}s")
                    print(f"   📄 Answer length: {len(result['answer'])} chars")
                    print(f"   🔗 Sources: {len(result.get('sources', []))}")
                    
                    model_results.append({
                        'query': query,
                        'response_time': response_time,
                        'answer_length': len(result['answer']),
                        'sources_count': len(result.get('sources', [])),
                        'success': True
                    })
                else:
                    print(f"   ❌ No answer generated")
                    model_results.append({
                        'query': query,
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })
                
            except Exception as e:
                end_time = time.time()
                print(f"   ❌ Error: {str(e)}")
                model_results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'response_time': end_time - start_time
                })
        
        results[model] = model_results
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 40)
    
    for model, model_results in results.items():
        successful = [r for r in model_results if r.get('success')]
        failed = [r for r in model_results if not r.get('success')]
        
        print(f"\n🧠 {model}:")
        print(f"   ✅ Successful: {len(successful)}/{len(model_results)}")
        
        if successful:
            avg_time = sum(r['response_time'] for r in successful) / len(successful)
            avg_length = sum(r['answer_length'] for r in successful) / len(successful)
            print(f"   ⏱️  Avg response time: {avg_time:.1f}s")
            print(f"   📄 Avg answer length: {avg_length:.0f} chars")
        
        if failed:
            print(f"   ❌ Failed: {len(failed)}")
            for f in failed:
                print(f"      • {f.get('error', 'Unknown error')}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    successful_models = [
        model for model, results in results.items() 
        if any(r.get('success') for r in results)
    ]
    
    if successful_models:
        print("✅ Working models for your RAG system:")
        for model in successful_models:
            model_results = results[model]
            successful = [r for r in model_results if r.get('success')]
            if successful:
                avg_time = sum(r['response_time'] for r in successful) / len(successful)
                print(f"   • {model} (avg: {avg_time:.1f}s)")
    else:
        print("❌ No models working properly - check Ollama setup")

async def main():
    await test_rag_system()

if __name__ == "__main__":
    asyncio.run(main()) 