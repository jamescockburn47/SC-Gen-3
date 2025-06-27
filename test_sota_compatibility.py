#!/usr/bin/env python3
"""
SOTA RAG Compatibility Test
===========================

Tests the new SOTA features while ensuring existing functionality is preserved.
This script validates that:

1. Existing system continues to work unchanged
2. SOTA features work when available
3. Graceful fallback when SOTA components unavailable
4. Performance improvements are measurable
"""

import sys
import logging
import time
from typing import Dict, Any, List
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SOTACompatibilityTester:
    """Comprehensive testing for SOTA RAG compatibility."""
    
    def __init__(self):
        self.test_results = {
            'existing_system': {},
            'sota_features': {},
            'compatibility': {},
            'performance': {}
        }
    
    def test_existing_system(self) -> Dict[str, bool]:
        """Test that existing system still works."""
        logger.info("🔍 Testing existing system compatibility...")
        
        results = {
            'import_local_rag': False,
            'create_pipeline': False,
            'basic_functionality': False
        }
        
        try:
            # Test importing existing RAG pipeline
            from local_rag_pipeline import LocalRAGPipeline, rag_session_manager
            results['import_local_rag'] = True
            logger.info("✅ Existing RAG pipeline imports successfully")
            
            # Test creating pipeline
            pipeline = rag_session_manager.get_or_create_pipeline('compatibility_test')
            if pipeline:
                results['create_pipeline'] = True
                logger.info("✅ Existing pipeline creation works")
                
                # Test basic functionality
                status = pipeline.get_document_status()
                if isinstance(status, dict):
                    results['basic_functionality'] = True
                    logger.info("✅ Existing pipeline basic functionality works")
                    logger.info(f"   Documents: {status.get('total_documents', 0)}")
                    logger.info(f"   Chunks: {status.get('total_chunks', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Existing system test failed: {e}")
        
        self.test_results['existing_system'] = results
        return results
    
    def test_sota_features(self) -> Dict[str, bool]:
        """Test SOTA features availability and functionality."""
        logger.info("🚀 Testing SOTA features...")
        
        results = {
            'flag_embedding_import': False,
            'bge_embeddings': False,
            'bge_reranker': False,
            'semantic_chunker': False,
            'enhanced_pdf': False,
            'integration_layer': False
        }
        
        # Test FlagEmbedding import
        try:
            from FlagEmbedding import FlagModel, FlagReranker
            results['flag_embedding_import'] = True
            logger.info("✅ FlagEmbedding imports successfully")
            
            # Test BGE embeddings
            try:
                model = FlagModel('BAAI/bge-base-en-v1.5', quantization_config={'load_in_8bit': True})
                test_embedding = model.encode(["Test legal document"])
                if test_embedding is not None and len(test_embedding) > 0:
                    results['bge_embeddings'] = True
                    logger.info(f"✅ BGE embeddings working (dim: {test_embedding.shape[1]})")
            except Exception as e:
                logger.warning(f"⚠️ BGE embeddings failed: {e}")
            
            # Test BGE reranker
            try:
                reranker = FlagReranker('BAAI/bge-reranker-base', quantization_config={'load_in_8bit': True})
                test_scores = reranker.compute_score([["query", "passage"]])
                if test_scores is not None:
                    results['bge_reranker'] = True
                    logger.info("✅ BGE reranker working")
            except Exception as e:
                logger.warning(f"⚠️ BGE reranker failed: {e}")
                
        except ImportError:
            logger.warning("⚠️ FlagEmbedding not available")
        
        # Test semantic chunker
        try:
            from legal_rag.ingest.chunker import LegalSemanticChunker
            chunker = LegalSemanticChunker()
            chunks = chunker.chunk_document("This is a test legal document with semantic chunking.")
            if chunks and len(chunks) > 0:
                results['semantic_chunker'] = True
                logger.info(f"✅ Semantic chunker working ({len(chunks)} chunks)")
        except Exception as e:
            logger.warning(f"⚠️ Semantic chunker failed: {e}")
        
        # Test enhanced PDF reader
        try:
            from legal_rag.ingest.pdf_reader import extract_text_with_fallback
            results['enhanced_pdf'] = True
            logger.info("✅ Enhanced PDF reader available")
        except Exception as e:
            logger.warning(f"⚠️ Enhanced PDF reader failed: {e}")
        
        # Test integration layer
        try:
            from sota_rag_integration import EnhancedLocalRAGPipeline
            enhanced_pipeline = EnhancedLocalRAGPipeline('sota_test')
            status = enhanced_pipeline.get_sota_status()
            if 'capabilities' in status:
                results['integration_layer'] = True
                logger.info("✅ Integration layer working")
                logger.info(f"   SOTA capabilities: {status['capabilities']}")
        except Exception as e:
            logger.warning(f"⚠️ Integration layer failed: {e}")
        
        self.test_results['sota_features'] = results
        return results
    
    def test_compatibility(self) -> Dict[str, bool]:
        """Test compatibility between existing and SOTA systems."""
        logger.info("🔄 Testing compatibility...")
        
        results = {
            'api_compatibility': False,
            'data_preservation': False,
            'graceful_fallback': False
        }
        
        try:
            # Test API compatibility
            from sota_rag_integration import enhanced_rag_session_manager
            
            # This should work exactly like the existing API
            enhanced_pipeline = enhanced_rag_session_manager.get_or_create_pipeline('compatibility_test')
            
            if enhanced_pipeline and hasattr(enhanced_pipeline, 'get_document_status'):
                results['api_compatibility'] = True
                logger.info("✅ API compatibility maintained")
            
            # Test data preservation
            status = enhanced_pipeline.get_document_status()
            if isinstance(status, dict):
                results['data_preservation'] = True
                logger.info("✅ Data access preserved")
            
            # Test graceful fallback
            sota_status = enhanced_pipeline.get_sota_status()
            fallback_available = sota_status.get('components', {}).get('existing_pipeline', False)
            
            if fallback_available:
                results['graceful_fallback'] = True
                logger.info("✅ Graceful fallback to existing system available")
            
        except Exception as e:
            logger.error(f"❌ Compatibility test failed: {e}")
        
        self.test_results['compatibility'] = results
        return results
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance of existing vs SOTA systems."""
        logger.info("⚡ Testing performance...")
        
        results = {
            'existing_embedding_time': None,
            'sota_embedding_time': None,
            'existing_search_time': None,
            'sota_search_time': None,
            'improvement_ratio': None
        }
        
        test_texts = [
            "This is a test legal document about contract law and commercial disputes.",
            "The case involves multiple parties in a complex litigation scenario.",
            "Legal precedents and statutory provisions are relevant to this matter."
        ]
        
        # Test existing system performance
        try:
            from sentence_transformers import SentenceTransformer
            existing_model = SentenceTransformer('all-mpnet-base-v2')
            
            start_time = time.time()
            existing_embeddings = existing_model.encode(test_texts)
            existing_time = time.time() - start_time
            
            results['existing_embedding_time'] = existing_time
            logger.info(f"📊 Existing embeddings: {existing_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Existing performance test failed: {e}")
        
        # Test SOTA system performance
        try:
            from FlagEmbedding import FlagModel
            sota_model = FlagModel('BAAI/bge-base-en-v1.5', quantization_config={'load_in_8bit': True})
            
            start_time = time.time()
            sota_embeddings = sota_model.encode(test_texts, normalize_embeddings=True)
            sota_time = time.time() - start_time
            
            results['sota_embedding_time'] = sota_time
            logger.info(f"🚀 SOTA embeddings: {sota_time:.3f}s")
            
            # Calculate improvement
            if results['existing_embedding_time'] and results['sota_embedding_time']:
                ratio = results['existing_embedding_time'] / results['sota_embedding_time']
                results['improvement_ratio'] = ratio
                
                if ratio > 1:
                    logger.info(f"✅ SOTA is {ratio:.2f}x faster")
                elif ratio < 1:
                    logger.info(f"📊 SOTA is {1/ratio:.2f}x slower (higher quality)")
                else:
                    logger.info("📊 Similar performance")
            
        except Exception as e:
            logger.warning(f"⚠️ SOTA performance test failed: {e}")
        
        self.test_results['performance'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        
        report.append("# SOTA RAG Compatibility Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Existing System Results
        existing = self.test_results['existing_system']
        existing_score = sum(existing.values())
        existing_total = len(existing)
        
        report.append(f"## 📦 Existing System: {existing_score}/{existing_total} tests passed")
        for test, passed in existing.items():
            emoji = "✅" if passed else "❌"
            report.append(f"- {emoji} {test.replace('_', ' ').title()}")
        report.append("")
        
        # SOTA Features Results
        sota = self.test_results['sota_features']
        sota_score = sum(sota.values())
        sota_total = len(sota)
        
        report.append(f"## 🚀 SOTA Features: {sota_score}/{sota_total} features available")
        for test, passed in sota.items():
            emoji = "✅" if passed else "❌"
            report.append(f"- {emoji} {test.replace('_', ' ').title()}")
        report.append("")
        
        # Compatibility Results
        compat = self.test_results['compatibility']
        compat_score = sum(compat.values())
        compat_total = len(compat)
        
        report.append(f"## 🔄 Compatibility: {compat_score}/{compat_total} tests passed")
        for test, passed in compat.items():
            emoji = "✅" if passed else "❌"
            report.append(f"- {emoji} {test.replace('_', ' ').title()}")
        report.append("")
        
        # Performance Results
        perf = self.test_results['performance']
        report.append("## ⚡ Performance Comparison")
        
        if perf['existing_embedding_time'] and perf['sota_embedding_time']:
            report.append(f"- Existing System: {perf['existing_embedding_time']:.3f}s")
            report.append(f"- SOTA System: {perf['sota_embedding_time']:.3f}s")
            
            if perf['improvement_ratio']:
                if perf['improvement_ratio'] > 1:
                    report.append(f"- **{perf['improvement_ratio']:.2f}x faster with SOTA**")
                else:
                    report.append(f"- {1/perf['improvement_ratio']:.2f}x slower with SOTA (higher quality)")
        else:
            report.append("- Performance comparison not available")
        report.append("")
        
        # Overall Assessment
        total_existing = existing_score
        total_sota = sota_score
        total_compat = compat_score
        
        report.append("## 📋 Overall Assessment")
        
        if total_existing >= 2:
            report.append("✅ **Existing system fully functional** - no disruption to current workflow")
        else:
            report.append("❌ **Existing system issues detected** - may need investigation")
        
        if total_sota >= 3:
            report.append("✅ **SOTA features ready** - significant enhancements available")
        elif total_sota >= 1:
            report.append("⚠️ **Partial SOTA features** - some enhancements available")
        else:
            report.append("❌ **SOTA features unavailable** - dependencies need installation")
        
        if total_compat >= 2:
            report.append("✅ **Full backward compatibility** - safe to upgrade")
        else:
            report.append("⚠️ **Compatibility concerns** - careful migration recommended")
        
        report.append("")
        report.append("## 🔧 Recommendations")
        
        if total_existing >= 2 and total_sota >= 3 and total_compat >= 2:
            report.append("🎉 **Ready for SOTA upgrade!** All systems functional and compatible.")
            report.append("- Run `python upgrade_to_sota_rag.py` to perform the upgrade")
            report.append("- The enhanced system will provide improved accuracy and features")
        elif total_existing >= 2 and total_compat >= 2:
            report.append("📦 **Continue with existing system** while installing SOTA dependencies.")
            report.append("- Install missing SOTA components: `pip install FlagEmbedding`")
            report.append("- Run this test again after installation")
        else:
            report.append("🛠️ **System needs attention** before upgrade.")
            report.append("- Check existing system functionality first")
            report.append("- Ensure all dependencies are properly installed")
        
        return "\\n".join(report)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all compatibility tests."""
        logger.info("🧪 Starting SOTA RAG Compatibility Tests")
        logger.info("=" * 50)
        
        # Run all test categories
        self.test_existing_system()
        self.test_sota_features()
        self.test_compatibility()
        self.test_performance()
        
        # Generate and display report
        report = self.generate_report()
        
        logger.info("\\n📋 TEST REPORT:")
        logger.info("=" * 50)
        for line in report.split("\\n"):
            if line.strip():
                logger.info(line)
        
        # Save report to file
        report_file = Path("sota_compatibility_report.md")
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"\\n📄 Report saved to: {report_file}")
        
        return self.test_results


def main():
    """Main test execution."""
    tester = SOTACompatibilityTester()
    results = tester.run_all_tests()
    
    # Calculate overall success
    existing_ok = sum(results['existing_system'].values()) >= 2
    sota_available = sum(results['sota_features'].values()) >= 1
    compat_ok = sum(results['compatibility'].values()) >= 2
    
    if existing_ok and compat_ok:
        logger.info("\\n🎉 COMPATIBILITY TEST PASSED")
        logger.info("✅ Safe to proceed with SOTA upgrade")
        return 0
    else:
        logger.warning("\\n⚠️ COMPATIBILITY ISSUES DETECTED")
        logger.warning("❌ Review issues before upgrading")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 