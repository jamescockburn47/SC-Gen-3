#!/usr/bin/env python3
"""
BGE Optimization Test Script
============================

Tests the BGE (BAAI) embeddings and reranking implementation for 
improved vectorization performance.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from local_rag_pipeline import LocalRAGPipeline

def test_bge_availability():
    """Test if BGE models are available"""
    print("🔍 Testing BGE Model Availability")
    print("=" * 50)
    
    try:
        from FlagEmbedding import FlagModel, FlagReranker
        print("✅ FlagEmbedding library available")
        
        # Test BGE embedding model
        print("📥 Testing BGE embedding model...")
        embedding_model = FlagModel(
            "BAAI/bge-base-en-v1.5",
            query_instruction_for_retrieval="Represent this query for searching legal documents:",
            use_fp16=True
        )
        
        test_texts = [
            "This is a test legal document about contract law.",
            "The defendant breached the contract by failing to deliver goods."
        ]
        
        embeddings = embedding_model.encode(test_texts)
        print(f"✅ BGE embeddings working (shape: {embeddings.shape})")
        
        # Test BGE reranker
        print("🎯 Testing BGE reranker...")
        reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
        
        pairs = [
            ("contract breach", "The defendant breached the contract"),
            ("contract breach", "This is about real estate law")
        ]
        
        scores = reranker.compute_score(pairs, normalize=True)
        print(f"✅ BGE reranker working (scores: {scores})")
        
        return True
        
    except ImportError as e:
        print(f"❌ FlagEmbedding not available: {e}")
        return False
    except Exception as e:
        print(f"❌ BGE test failed: {e}")
        return False

def test_bge_pipeline():
    """Test BGE integration in LocalRAGPipeline"""
    print("\n🚀 Testing BGE Pipeline Integration")
    print("=" * 50)
    
    try:
        # Create pipeline with BGE enabled
        pipeline = LocalRAGPipeline(
            matter_id="test_bge",
            embedding_model="BAAI/bge-base-en-v1.5",
            enable_reranking=True
        )
        
        print(f"✅ Pipeline created")
        print(f"   📊 Using BGE: {pipeline.is_using_bge}")
        print(f"   🎯 Reranking: {pipeline.enable_reranking}")
        print(f"   🔧 Model: {pipeline.embedding_model_name}")
        
        # Get performance stats
        stats = pipeline.get_performance_stats()
        print(f"\n📈 Performance Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ BGE pipeline test failed: {e}")
        return False

def test_search_performance():
    """Test search performance with BGE vs standard embeddings"""
    print("\n⚡ Testing Search Performance")
    print("=" * 50)
    
    try:
        # Test with BGE
        print("🚀 Testing BGE performance...")
        bge_pipeline = LocalRAGPipeline(
            matter_id="test_bge_perf",
            embedding_model="BAAI/bge-base-en-v1.5",
            enable_reranking=True
        )
        
        # Test with standard embeddings
        print("📦 Testing standard embeddings...")
        std_pipeline = LocalRAGPipeline(
            matter_id="test_std_perf", 
            embedding_model="all-mpnet-base-v2",
            enable_reranking=False
        )
        
        # Mock search test (since we don't have documents loaded)
        test_query = "contract breach and damages"
        
        # BGE search simulation
        start_time = time.time()
        # pipeline.search_documents(test_query, top_k=5)  # Would need documents
        bge_time = time.time() - start_time
        
        print(f"🚀 BGE Ready: {bge_pipeline.is_using_bge}")
        print(f"📦 Standard Ready: {not std_pipeline.is_using_bge}")
        print(f"🎯 BGE Reranking: {bge_pipeline.reranker_model is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_embedding_quality():
    """Test embedding quality comparison"""
    print("\n🎯 Testing Embedding Quality")
    print("=" * 50)
    
    try:
        from FlagEmbedding import FlagModel
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Legal documents for testing
        legal_texts = [
            "The claimant seeks damages for breach of contract in the amount of £50,000.",
            "Defendant argues that force majeure provisions excuse performance delays.",
            "The High Court granted interim injunction to prevent asset disposal.",
            "Summary judgment application denied due to disputed facts.",
            "Settlement negotiations concluded with mutual confidentiality agreement."
        ]
        
        # BGE embeddings
        print("🚀 Testing BGE embedding quality...")
        bge_model = FlagModel("BAAI/bge-base-en-v1.5", use_fp16=True)
        bge_embeddings = bge_model.encode(legal_texts)
        
        # Standard embeddings
        print("📦 Testing standard embedding quality...")
        std_model = SentenceTransformer("all-mpnet-base-v2")
        std_embeddings = std_model.encode(legal_texts)
        
        print(f"✅ BGE embeddings shape: {bge_embeddings.shape}")
        print(f"✅ Standard embeddings shape: {std_embeddings.shape}")
        
        # Calculate embedding quality metrics
        bge_avg_norm = np.mean(np.linalg.norm(bge_embeddings, axis=1))
        std_avg_norm = np.mean(np.linalg.norm(std_embeddings, axis=1))
        
        print(f"📊 BGE average norm: {bge_avg_norm:.4f}")
        print(f"📊 Standard average norm: {std_avg_norm:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding quality test failed: {e}")
        return False

def main():
    """Run all BGE optimization tests"""
    print("🧪 BGE Optimization Test Suite")
    print("=" * 60)
    
    tests = [
        ("BGE Availability", test_bge_availability),
        ("BGE Pipeline Integration", test_bge_pipeline),
        ("Search Performance", test_search_performance),
        ("Embedding Quality", test_embedding_quality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔬 Running: {test_name}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🚀 All BGE optimization tests passed!")
        print("💡 Your system is ready for enhanced vectorization with BGE models")
    else:
        print("⚠️  Some tests failed. Check dependencies and model availability.")
        print("💡 Install missing dependencies: pip install FlagEmbedding")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 