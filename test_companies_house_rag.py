#!/usr/bin/env python3
"""
Test script for Companies House RAG Pipeline
Comprehensive testing of CH document retrieval, local OCR, and RAG integration
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

def test_ch_rag_availability():
    """Test if all required components are available"""
    print("🔧 Testing Companies House RAG availability...")
    
    results = {
        'ch_rag_pipeline': False,
        'config': False,
        'dependencies': []
    }
    
    try:
        from companies_house_rag_pipeline import get_ch_rag_pipeline
        results['ch_rag_pipeline'] = True
        print("✅ Companies House RAG pipeline available")
    except ImportError as e:
        print(f"❌ CH RAG pipeline not available: {e}")
    
    try:
        import config
        results['config'] = True
        ch_api_key = getattr(config, 'CH_API_KEY', None)
        if ch_api_key:
            print("✅ CH API key configured")
        else:
            print("⚠️ CH API key not configured")
    except ImportError:
        print("❌ Config not available")
    
    # Test dependencies
    dependencies = [
        ('aiohttp', 'HTTP client for async requests'),
        ('faiss-cpu', 'Vector similarity search'),
        ('sentence-transformers', 'Text embeddings'),
        ('PyPDF2', 'PDF text extraction')
    ]
    
    for dep_name, description in dependencies:
        try:
            if dep_name == 'faiss-cpu':
                import faiss
            elif dep_name == 'sentence-transformers':
                from sentence_transformers import SentenceTransformer
            elif dep_name == 'PyPDF2':
                import PyPDF2
            else:
                __import__(dep_name)
            
            results['dependencies'].append(dep_name)
            print(f"✅ {dep_name}: {description}")
        except ImportError:
            print(f"❌ {dep_name}: {description} - not available")
    
    return results

async def test_ch_rag_pipeline():
    """Test the complete CH RAG pipeline functionality"""
    print("\n🧪 Testing Companies House RAG Pipeline...")
    
    try:
        from companies_house_rag_pipeline import get_ch_rag_pipeline
        import config
        
        # Get pipeline instance
        pipeline = get_ch_rag_pipeline("test_ch_rag")
        pipeline.ch_api_key = getattr(config, 'CH_API_KEY', None)
        
        if not pipeline.ch_api_key:
            print("❌ Cannot test pipeline: CH API key not configured")
            return False
        
        print("✅ Pipeline initialized")
        
        # Test processing stats
        stats = pipeline.get_processing_stats()
        print(f"📊 Current stats: {stats['companies_in_database']} companies, {stats['documents_in_database']} documents")
        
        # Test with a well-known company (Companies House itself)
        test_company = "00006398"  # Companies House company number
        print(f"\n🏢 Testing with company: {test_company}")
        
        # Test document processing
        print("⏳ Processing company documents...")
        results = await pipeline.process_companies(
            company_numbers=[test_company],
            categories=['accounts', 'officers'],
            year_range=(2022, 2024),
            max_docs_per_company=5
        )
        
        if results['success']:
            print(f"✅ Processing successful: {results['total_documents']} documents")
            
            # Test search functionality
            if results['total_documents'] > 0:
                print("\n🔍 Testing search functionality...")
                search_results = pipeline.search_ch_documents(
                    f"company {test_company} financial information", top_k=3
                )
                
                if search_results:
                    print(f"✅ Search successful: {len(search_results)} results")
                    
                    # Test comprehensive analysis
                    print("\n📈 Testing comprehensive analysis...")
                    analysis = await pipeline.analyze_company_comprehensive(
                        test_company, 
                        "Provide overview of company activities and financial status"
                    )
                    
                    if analysis['success']:
                        print(f"✅ Analysis successful: {len(analysis['analysis'])} characters")
                        print(f"📄 Analyzed {analysis['documents_analyzed']} documents")
                        return True
                    else:
                        print(f"❌ Analysis failed: {analysis.get('error')}")
                        return False
                else:
                    print("❌ Search returned no results")
                    return False
            else:
                print("⚠️ No documents processed - cannot test search/analysis")
                return True  # Processing worked, just no documents
        else:
            print(f"❌ Processing failed: {results.get('error')}")
            return False
    
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_ch_rag_interface():
    """Test the Streamlit interface components"""
    print("\n🖥️ Testing CH RAG Interface...")
    
    try:
        from companies_house_rag_interface import render_companies_house_rag_interface
        print("✅ Interface module imported successfully")
        
        # Test categories mapping
        from companies_house_rag_interface import CH_CATEGORIES
        print(f"✅ Categories available: {len(CH_CATEGORIES)} types")
        for name, api_code in CH_CATEGORIES.items():
            print(f"   • {name}: {api_code}")
        
        return True
    
    except ImportError as e:
        print(f"❌ Interface not available: {e}")
        return False

def test_integration_with_main_app():
    """Test integration with the main Streamlit app"""
    print("\n🔗 Testing integration with main app...")
    
    try:
        # Check if app.py can import the interface
        import sys
        from pathlib import Path
        
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))
        
        # Test import
        from companies_house_rag_interface import render_companies_house_rag_interface
        print("✅ Interface can be imported by main app")
        
        # Check if we can modify app.py to include CH RAG tab
        app_py = Path("app.py")
        if app_py.exists():
            print("✅ app.py found")
            
            with open(app_py, 'r') as f:
                content = f.read()
            
            if 'companies_house_rag_interface' in content:
                print("✅ CH RAG interface already integrated in app.py")
            else:
                print("⚠️ CH RAG interface not yet integrated in app.py")
                print("💡 Add to app.py: from companies_house_rag_interface import render_companies_house_rag_interface")
        else:
            print("❌ app.py not found")
        
        return True
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Companies House RAG Pipeline Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test availability
    availability = test_ch_rag_availability()
    test_results.append(("Availability", availability['ch_rag_pipeline'] and availability['config']))
    
    # Test pipeline if available
    if availability['ch_rag_pipeline'] and availability['config']:
        pipeline_result = await test_ch_rag_pipeline()
        test_results.append(("Pipeline", pipeline_result))
    else:
        print("\n⏭️ Skipping pipeline test - dependencies not available")
        test_results.append(("Pipeline", False))
    
    # Test interface
    interface_result = test_ch_rag_interface()
    test_results.append(("Interface", interface_result))
    
    # Test integration
    integration_result = test_integration_with_main_app()
    test_results.append(("Integration", integration_result))
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} {status}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Companies House RAG pipeline is ready!")
    else:
        print("⚠️ Some tests failed. Check the issues above.")
        print("\n💡 Setup instructions:")
        print("   1. pip install aiohttp faiss-cpu sentence-transformers PyPDF2")
        print("   2. Set CH_API_KEY in config.py")
        print("   3. Ensure all dependencies are properly installed")

if __name__ == "__main__":
    asyncio.run(main()) 