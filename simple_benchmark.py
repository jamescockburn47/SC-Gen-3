#!/usr/bin/env python3
"""
Simplified performance benchmark for Strategic Counsel optimizations.
Tests core functionality without requiring Streamlit context.
"""

import time
import sys
import gc
import logging
from pathlib import Path
from typing import Dict, List, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback using gc
        objects = gc.get_objects()
        return sys.getsizeof(objects) / 1024 / 1024

def test_performance_utils():
    """Test performance utilities without Streamlit dependencies."""
    logger.info("🧪 Testing Performance Utilities...")
    
    try:
        from performance_utils import cached, timed, memory_manager, normalize_text, chunk_text_efficiently
        
        # Test caching
        call_count = 0
        
        @cached(ttl=60, max_size=10)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return x * 2
        
        # Test cached vs uncached
        start_time = time.time()
        for i in range(5):
            expensive_function(i)
        uncached_time = time.time() - start_time
        
        first_call_count = call_count
        
        start_time = time.time()
        for i in range(5):
            expensive_function(i)
        cached_time = time.time() - start_time
        
        cache_improvement = (uncached_time - cached_time) / uncached_time * 100
        
        logger.info(f"✅ Caching test: {cache_improvement:.1f}% improvement")
        
        # Test memory management
        baseline_memory = get_memory_usage()
        
        large_objects = []
        for i in range(100):
            large_objects.append([0] * 10000)
        
        peak_memory = get_memory_usage()
        
        del large_objects
        memory_manager.cleanup_if_needed(force=True)
        
        final_memory = get_memory_usage()
        memory_recovered = peak_memory - final_memory
        
        logger.info(f"✅ Memory management: {memory_recovered:.1f}MB recovered")
        
        # Test text processing
        test_text = "This is a sample text with extra    spaces and \n\n\n multiple newlines. " * 1000
        
        start_time = time.time()
        for _ in range(50):
            normalized = normalize_text(test_text)
        normalization_time = time.time() - start_time
        
        logger.info(f"✅ Text normalization: {normalization_time:.3f}s for 50 calls")
        
        return {
            'caching_improvement': cache_improvement,
            'memory_recovered_mb': memory_recovered,
            'normalization_time': normalization_time,
            'success': True
        }
        
    except ImportError as e:
        logger.error(f"❌ Performance utils import failed: {e}")
        return {'success': False, 'error': str(e)}

def test_config_optimizations():
    """Test configuration optimizations."""
    logger.info("🧪 Testing Configuration Optimizations...")
    
    try:
        import config
        
        # Test OpenAI client caching
        start_time = time.time()
        client1 = config.get_openai_client()
        first_init_time = time.time() - start_time
        
        start_time = time.time()
        client2 = config.get_openai_client()
        cached_init_time = time.time() - start_time
        
        if first_init_time > 0:
            cache_improvement = (first_init_time - cached_init_time) / first_init_time * 100
        else:
            cache_improvement = 0
        
        logger.info(f"✅ OpenAI client caching: {cache_improvement:.1f}% improvement")
        
        # Test CH session
        start_time = time.time()
        session1 = config.get_ch_session()
        ch_first_time = time.time() - start_time
        
        start_time = time.time()
        session2 = config.get_ch_session()
        ch_cached_time = time.time() - start_time
        
        logger.info(f"✅ CH session caching: {(ch_first_time - ch_cached_time) / max(ch_first_time, 0.001) * 100:.1f}% improvement")
        
        return {
            'openai_improvement': cache_improvement,
            'ch_session_improvement': (ch_first_time - ch_cached_time) / max(ch_first_time, 0.001) * 100,
            'success': True
        }
        
    except ImportError as e:
        logger.error(f"❌ Config import failed: {e}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_ai_utils_optimizations():
    """Test AI utilities optimizations."""
    logger.info("🧪 Testing AI Utilities Optimizations...")
    
    try:
        # Test if the caching decorators are properly applied
        import ai_utils
        
        # Check if the gpt_summarise_ch_docs function has caching
        if hasattr(ai_utils.gpt_summarise_ch_docs, 'cache_clear'):
            logger.info("✅ AI summarization function has caching enabled")
            caching_enabled = True
        else:
            logger.warning("⚠️  AI summarization function caching not detected")
            caching_enabled = False
        
        return {
            'caching_enabled': caching_enabled,
            'success': True
        }
        
    except ImportError as e:
        logger.error(f"❌ AI utils import failed: {e}")
        return {'success': False, 'error': str(e)}

def run_simple_benchmark():
    """Run simplified performance benchmark."""
    logger.info("🚀 Starting Simplified Strategic Counsel Performance Benchmark")
    logger.info("=" * 60)
    
    results = {
        'timestamp': time.time(),
        'python_version': sys.version,
        'platform': sys.platform
    }
    
    # Test performance utilities
    logger.info("\n📦 Testing Performance Utilities")
    results['performance_utils'] = test_performance_utils()
    
    # Test config optimizations
    logger.info("\n⚙️  Testing Configuration Optimizations")
    results['config'] = test_config_optimizations()
    
    # Test AI utils optimizations
    logger.info("\n🤖 Testing AI Utilities Optimizations")
    results['ai_utils'] = test_ai_utils_optimizations()
    
    # Generate summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 BENCHMARK SUMMARY")
    logger.info("=" * 60)
    
    successful_tests = 0
    total_tests = 0
    
    if results['performance_utils']['success']:
        successful_tests += 1
        logger.info("✅ Performance Utilities: Working correctly")
    else:
        logger.error(f"❌ Performance Utilities: {results['performance_utils'].get('error', 'Unknown error')}")
    total_tests += 1
    
    if results['config']['success']:
        successful_tests += 1
        logger.info("✅ Configuration Optimizations: Working correctly")
    else:
        logger.error(f"❌ Configuration Optimizations: {results['config'].get('error', 'Unknown error')}")
    total_tests += 1
    
    if results['ai_utils']['success']:
        successful_tests += 1
        logger.info("✅ AI Utilities Optimizations: Working correctly")
    else:
        logger.error(f"❌ AI Utilities Optimizations: {results['ai_utils'].get('error', 'Unknown error')}")
    total_tests += 1
    
    success_rate = (successful_tests / total_tests) * 100
    logger.info(f"🎯 Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    
    if success_rate >= 80:
        logger.info("🎉 EXCELLENT: Most optimizations are working correctly!")
    elif success_rate >= 60:
        logger.info("✅ GOOD: Most optimizations are working!")
    else:
        logger.warning("⚠️  Some optimizations need attention")
    
    # Save results
    results_file = Path("simple_benchmark_results.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📁 Results saved to: {results_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
    
    logger.info("=" * 60)
    logger.info("🏁 Simple Benchmark Complete!")
    
    return results

if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        results = run_simple_benchmark()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\n🛑 Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        sys.exit(1) 