#!/usr/bin/env python3
"""
Simple test script to verify RAG system is working
Run this after fixing dependencies to ensure everything is ready
"""

import sys
import asyncio
from datetime import datetime

def test_imports():
    """Test all required imports"""
    print("🧪 Testing RAG System Dependencies...")
    print("")
    
    results = {}
    
    # Test core dependencies
    try:
        import faiss
        print("✅ FAISS: Available")
        results['faiss'] = True
    except ImportError as e:
        print(f"❌ FAISS: {e}")
        results['faiss'] = False
    
    try:
        import numpy as np
        print("✅ NumPy: Available")
        results['numpy'] = True
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        results['numpy'] = False
    
    try:
        import aiohttp
        print("✅ aiohttp: Available")
        results['aiohttp'] = True
    except ImportError as e:
        print(f"❌ aiohttp: {e}")
        results['aiohttp'] = False
    
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers: Available")
        results['sentence_transformers'] = True
    except ImportError as e:
        print(f"❌ sentence-transformers: {e}")
        results['sentence_transformers'] = False
    
    return results

def test_rag_components():
    """Test RAG pipeline components"""
    print("")
    print("🔧 Testing RAG Components...")
    
    try:
        # Test RAG pipeline import
        from local_rag_pipeline import LocalRAGPipeline, rag_session_manager
        print("✅ RAG Pipeline: Importable")
        
        # Test MCP server import
        from mcp_rag_server import mcp_rag_server
        print("✅ MCP Server: Importable")
        
        # Test optimizer import
        try:
            from rag_config_optimizer import rag_optimizer
            print("✅ Hardware Optimizer: Available")
        except ImportError:
            print("⚠️ Hardware Optimizer: Not available (optional)")
        
        return True
        
    except ImportError as e:
        print(f"❌ RAG Components: {e}")
        return False

async def test_ollama_connection():
    """Test Ollama connectivity"""
    print("")
    print("🔗 Testing Ollama Connection...")
    
    try:
        from local_rag_pipeline import LocalRAGPipeline
        
        # Create test pipeline
        pipeline = LocalRAGPipeline("test_connection")
        
        # Test model query
        models = await pipeline.query_ollama_models()
        
        if models:
            print(f"✅ Ollama: Connected ({len(models)} models available)")
            print("📊 Available models:")
            for model in models[:5]:  # Show first 5
                name = model['name']
                size_mb = model.get('size', 0) / (1024 * 1024)
                print(f"   • {name} ({size_mb:.1f} MB)")
            return True
        else:
            print("❌ Ollama: No models found")
            return False
            
    except Exception as e:
        print(f"❌ Ollama: Connection failed - {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def test_embedding_model():
    """Test embedding model initialization"""
    print("")
    print("🎯 Testing Embedding Model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try to initialize a small model
        print("📥 Loading embedding model (this may take a moment)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test encoding
        test_text = "This is a test sentence for embedding."
        embedding = model.encode([test_text])
        
        print(f"✅ Embedding Model: Working (dimension: {embedding.shape[1]})")
        return True
        
    except Exception as e:
        print(f"❌ Embedding Model: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RAG System Test Suite")
    print("=" * 50)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Test 1: Dependencies
    deps_results = test_imports()
    deps_ok = all(deps_results.values())
    
    # Test 2: RAG Components
    rag_ok = test_rag_components()
    
    # Test 3: Ollama (async)
    try:
        ollama_ok = asyncio.run(test_ollama_connection())
    except Exception as e:
        print(f"❌ Ollama Test Failed: {e}")
        ollama_ok = False
    
    # Test 4: Embedding Model (only if sentence_transformers works)
    if deps_results.get('sentence_transformers', False):
        embedding_ok = test_embedding_model()
    else:
        embedding_ok = False
        print("")
        print("⏭️ Skipping embedding test (sentence-transformers not available)")
    
    # Summary
    print("")
    print("=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Dependencies", deps_ok),
        ("RAG Components", rag_ok),
        ("Ollama Connection", ollama_ok),
        ("Embedding Model", embedding_ok)
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 RAG system is ready to use!")
        print("")
        print("Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Go to '📚 Document RAG' tab")
        print("3. Upload documents and start querying!")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("🔧 Run the fix script first:")
        print("   ./fix_rag_dependencies.sh")
        print("")
        print("If issues persist:")
        print("1. Check Ollama is running: ollama serve")
        print("2. Verify models are installed: ollama list")
        print("3. Check Python package versions")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 