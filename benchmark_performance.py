#!/usr/bin/env python3
"""
Performance benchmarking script for Strategic Counsel optimizations.
Run this script to validate optimization improvements.
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

def benchmark_imports():
    """Benchmark import times for key modules."""
    results = {}
    
    # Test config import
    start_time = time.time()
    try:
        import config
        results['config'] = time.time() - start_time
        logger.info(f"✅ Config import: {results['config']:.3f}s")
    except ImportError as e:
        results['config'] = None
        logger.error(f"❌ Config import failed: {e}")
    
    # Test performance utils import
    start_time = time.time()
    try:
        import performance_utils
        results['performance_utils'] = time.time() - start_time
        logger.info(f"✅ Performance utils import: {results['performance_utils']:.3f}s")
    except ImportError as e:
        results['performance_utils'] = None
        logger.error(f"❌ Performance utils import failed: {e}")
    
    # Test session state manager import
    start_time = time.time()
    try:
        import session_state_manager
        results['session_state_manager'] = time.time() - start_time
        logger.info(f"✅ Session state manager import: {results['session_state_manager']:.3f}s")
    except ImportError as e:
        results['session_state_manager'] = None
        logger.error(f"❌ Session state manager import failed: {e}")
    
    return results

def benchmark_caching():
    """Benchmark caching performance."""
    try:
        from performance_utils import cached, smart_cache
        
        # Test function for caching
        call_count = 0
        
        @cached(ttl=60, max_size=10)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return x * 2
        
        # Benchmark cached vs uncached
        logger.info("🧪 Testing caching performance...")
        
        # First calls (cache misses)
        start_time = time.time()
        for i in range(5):
            expensive_function(i)
        uncached_time = time.time() - start_time
        
        # Reset call count
        first_call_count = call_count
        
        # Second calls (cache hits)
        start_time = time.time()
        for i in range(5):
            expensive_function(i)
        cached_time = time.time() - start_time
        
        cache_improvement = (uncached_time - cached_time) / uncached_time * 100
        
        logger.info(f"✅ Uncached calls: {uncached_time:.3f}s ({first_call_count} function calls)")
        logger.info(f"✅ Cached calls: {cached_time:.3f}s ({call_count - first_call_count} additional function calls)")
        logger.info(f"🚀 Cache improvement: {cache_improvement:.1f}%")
        
        return {
            'uncached_time': uncached_time,
            'cached_time': cached_time,
            'improvement_percent': cache_improvement,
            'cache_hits': 5 - (call_count - first_call_count)
        }
        
    except ImportError:
        logger.error("❌ Performance utils not available for caching test")
        return None

def benchmark_memory_management():
    """Benchmark memory management."""
    try:
        from performance_utils import memory_manager
        
        logger.info("🧪 Testing memory management...")
        
        # Get baseline memory
        baseline_memory = get_memory_usage()
        
        # Create some large objects
        large_objects = []
        for i in range(100):
            large_objects.append([0] * 10000)  # Create large lists
        
        peak_memory = get_memory_usage()
        
        # Test cleanup
        del large_objects
        memory_manager.cleanup_if_needed(force=True)
        
        final_memory = get_memory_usage()
        
        memory_recovered = peak_memory - final_memory
        recovery_percent = (memory_recovered / (peak_memory - baseline_memory)) * 100
        
        logger.info(f"✅ Baseline memory: {baseline_memory:.1f}MB")
        logger.info(f"✅ Peak memory: {peak_memory:.1f}MB")
        logger.info(f"✅ Final memory: {final_memory:.1f}MB")
        logger.info(f"🚀 Memory recovered: {memory_recovered:.1f}MB ({recovery_percent:.1f}%)")
        
        return {
            'baseline_memory': baseline_memory,
            'peak_memory': peak_memory,
            'final_memory': final_memory,
            'memory_recovered': memory_recovered,
            'recovery_percent': recovery_percent
        }
        
    except ImportError:
        logger.error("❌ Performance utils not available for memory management test")
        return None

def benchmark_api_clients():
    """Benchmark API client initialization."""
    try:
        import config
        
        logger.info("🧪 Testing API client performance...")
        
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
        
        logger.info(f"✅ First OpenAI client init: {first_init_time:.3f}s")
        logger.info(f"✅ Cached OpenAI client init: {cached_init_time:.3f}s")
        logger.info(f"🚀 Client caching improvement: {cache_improvement:.1f}%")
        
        # Test CH session
        start_time = time.time()
        session1 = config.get_ch_session()
        ch_first_time = time.time() - start_time
        
        start_time = time.time()
        session2 = config.get_ch_session()
        ch_cached_time = time.time() - start_time
        
        logger.info(f"✅ First CH session init: {ch_first_time:.3f}s")
        logger.info(f"✅ Cached CH session init: {ch_cached_time:.3f}s")
        
        return {
            'openai_first_init': first_init_time,
            'openai_cached_init': cached_init_time,
            'openai_improvement': cache_improvement,
            'ch_first_init': ch_first_time,
            'ch_cached_init': ch_cached_time
        }
        
    except ImportError:
        logger.error("❌ Config module not available for API client test")
        return None

def benchmark_text_processing():
    """Benchmark text processing optimizations."""
    try:
        from performance_utils import normalize_text, chunk_text_efficiently
        
        logger.info("🧪 Testing text processing performance...")
        
        # Generate test text
        test_text = "This is a sample text with extra    spaces and \n\n\n multiple newlines. " * 1000
        
        # Test text normalization
        start_time = time.time()
        for _ in range(100):
            normalized = normalize_text(test_text)
        normalization_time = time.time() - start_time
        
        # Test chunking
        large_text = test_text * 100  # Very large text
        start_time = time.time()
        chunks = chunk_text_efficiently(large_text, chunk_size=50000)
        chunking_time = time.time() - start_time
        
        logger.info(f"✅ Text normalization (100 calls): {normalization_time:.3f}s")
        logger.info(f"✅ Text chunking ({len(large_text)} chars): {chunking_time:.3f}s")
        logger.info(f"✅ Generated {len(chunks)} chunks")
        
        return {
            'normalization_time': normalization_time,
            'chunking_time': chunking_time,
            'chunks_generated': len(chunks),
            'text_size': len(large_text)
        }
        
    except ImportError:
        logger.error("❌ Performance utils not available for text processing test")
        return None

def run_full_benchmark():
    """Run complete performance benchmark suite."""
    logger.info("🚀 Starting Strategic Counsel Performance Benchmark")
    logger.info("=" * 60)
    
    results = {
        'timestamp': time.time(),
        'python_version': sys.version,
        'platform': sys.platform
    }
    
    # Run all benchmarks
    logger.info("\n📦 Testing Import Performance")
    results['imports'] = benchmark_imports()
    
    logger.info("\n💾 Testing Caching Performance")
    results['caching'] = benchmark_caching()
    
    logger.info("\n🧹 Testing Memory Management")
    results['memory'] = benchmark_memory_management()
    
    logger.info("\n🌐 Testing API Client Performance")
    results['api_clients'] = benchmark_api_clients()
    
    logger.info("\n📝 Testing Text Processing")
    results['text_processing'] = benchmark_text_processing()
    
    # Generate summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 BENCHMARK SUMMARY")
    logger.info("=" * 60)
    
    # Calculate overall scores
    performance_scores = []
    
    if results['caching'] and results['caching']['improvement_percent'] > 0:
        performance_scores.append(results['caching']['improvement_percent'])
        logger.info(f"✅ Caching Performance: {results['caching']['improvement_percent']:.1f}% improvement")
    
    if results['memory'] and results['memory']['recovery_percent'] > 0:
        performance_scores.append(results['memory']['recovery_percent'])
        logger.info(f"✅ Memory Management: {results['memory']['recovery_percent']:.1f}% recovery")
    
    if results['api_clients'] and results['api_clients']['openai_improvement'] > 0:
        performance_scores.append(results['api_clients']['openai_improvement'])
        logger.info(f"✅ API Client Caching: {results['api_clients']['openai_improvement']:.1f}% improvement")
    
    if performance_scores:
        avg_improvement = sum(performance_scores) / len(performance_scores)
        logger.info(f"🎯 Average Performance Improvement: {avg_improvement:.1f}%")
        
        if avg_improvement > 50:
            logger.info("🎉 EXCELLENT: Significant performance improvements detected!")
        elif avg_improvement > 25:
            logger.info("✅ GOOD: Notable performance improvements detected!")
        else:
            logger.info("📈 FAIR: Some performance improvements detected")
    else:
        logger.warning("⚠️  Unable to calculate performance improvements")
    
    # Save results
    results_file = Path("benchmark_results.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📁 Results saved to: {results_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
    
    logger.info("=" * 60)
    logger.info("🏁 Benchmark Complete!")
    
    return results

if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        results = run_full_benchmark()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\n🛑 Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        sys.exit(1) 