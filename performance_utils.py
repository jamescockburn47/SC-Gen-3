# performance_utils.py

"""
Performance optimization utilities for Strategic Counsel application.
Provides caching, memory management, and performance monitoring tools.
"""

import gc
import sys
import time
import hashlib
import threading
from contextlib import contextmanager
from functools import wraps, lru_cache
from typing import Any, Dict, Optional, Union, Callable, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# --- Memory Management ---

class MemoryManager:
    """Manages memory usage and provides cleanup utilities."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.max_cache_size = 100
            self.memory_threshold_mb = 500
            self.initialized = True
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback using sys.getsizeof approximation
            return sys.getsizeof(gc.get_objects()) / 1024 / 1024
    
    def cleanup_if_needed(self, force: bool = False) -> bool:
        """Clean up memory if usage exceeds threshold."""
        current_usage = self.get_memory_usage_mb()
        
        if force or current_usage > self.memory_threshold_mb:
            logger.info(f"Memory cleanup triggered. Current usage: {current_usage:.1f}MB")
            
            # Force garbage collection
            collected = gc.collect()
            
            new_usage = self.get_memory_usage_mb()
            logger.info(f"Memory cleanup completed. Collected {collected} objects. New usage: {new_usage:.1f}MB")
            
            return True
        return False
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager to monitor memory usage during operations."""
        start_memory = self.get_memory_usage_mb()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage_mb()
            end_time = time.time()
            
            logger.info(
                f"Memory usage for '{operation_name}': "
                f"{start_memory:.1f}MB -> {end_memory:.1f}MB "
                f"(+{end_memory - start_memory:.1f}MB) in {end_time - start_time:.2f}s"
            )

# Global memory manager instance
memory_manager = MemoryManager()

# --- Intelligent Caching ---

class SmartCache:
    """Smart caching system with size limits and TTL."""
    
    def __init__(self, max_size: int = 100, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_string = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp > self.default_ttl
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self._access_times.pop(key, None)
    
    def _enforce_size_limit(self):
        """Enforce cache size limit using LRU eviction."""
        if len(self.cache) <= self.max_size:
            return
        
        # Sort by access time (LRU)
        sorted_keys = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest entries
        keys_to_remove = sorted_keys[:len(self.cache) - self.max_size]
        for key, _ in keys_to_remove:
            self.cache.pop(key, None)
            self._access_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp <= self.default_ttl:
                    self._access_times[key] = time.time()
                    return value
                else:
                    # Expired
                    self.cache.pop(key, None)
                    self._access_times.pop(key, None)
            return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self._lock:
            now = time.time()
            self.cache[key] = (value, now)
            self._access_times[key] = now
            
            self._cleanup_expired()
            self._enforce_size_limit()
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_ratio': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
            }

# Global smart cache instance
smart_cache = SmartCache()

def cached(ttl: int = 3600, max_size: int = 100):
    """Decorator for caching function results with TTL."""
    def decorator(func: Callable) -> Callable:
        cache = SmartCache(max_size=max_size, default_ttl=ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._generate_key(*args, **kwargs)
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        # Add cache management methods using setattr to avoid type issues
        setattr(wrapper, 'cache_clear', cache.clear)
        setattr(wrapper, 'cache_stats', cache.stats)
        
        return wrapper
    return decorator

# --- Performance Monitoring ---

class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self._lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float):
        """Record timing for an operation."""
        with self._lock:
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            
            # Keep only last 100 measurements
            if len(self.metrics[operation]) > 100:
                self.metrics[operation] = self.metrics[operation][-100:]
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation."""
        with self._lock:
            if operation not in self.metrics or not self.metrics[operation]:
                return {}
            
            timings = self.metrics[operation]
            return {
                'count': len(timings),
                'average': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
                'total': sum(timings)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.metrics.keys()}

# Global performance monitor
perf_monitor = PerformanceMonitor()

def timed(operation_name: Optional[str] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                perf_monitor.record_timing(op_name, duration)
                
                if duration > 1.0:  # Log slow operations
                    logger.warning(f"Slow operation '{op_name}': {duration:.2f}s")
        
        return wrapper
    return decorator

# --- Text Processing Optimizations ---

@lru_cache(maxsize=1000)
def normalize_text(text: str) -> str:
    """Normalize text with caching for repeated content."""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common OCR artifacts
    text = re.sub(r'[^\w\s\.,!?;:()\-"]', '', text)
    
    return text

def chunk_text_efficiently(text: str, chunk_size: int = 100000, overlap: int = 1000) -> list:
    """Efficiently chunk large text with overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Adjust end to avoid breaking words
        if end < len(text):
            # Find the last space before the end
            while end > start and text[end] not in ' \n\t':
                end -= 1
            
            if end == start:  # No space found, use original end
                end = start + chunk_size
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        start = max(end - overlap, start + 1)
        
        if start >= len(text):
            break
    
    return chunks

# --- Batch Processing Utilities ---

def batch_process(items: list, batch_size: int = 10, max_workers: int = 4) -> list:
    """Process items in batches with optional threading."""
    if len(items) <= batch_size or max_workers == 1:
        return items
    
    from concurrent.futures import ThreadPoolExecutor
    import math
    
    # Split items into batches
    batches = [
        items[i:i + batch_size] 
        for i in range(0, len(items), batch_size)
    ]
    
    def process_batch(batch):
        return batch  # Override this in actual usage
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
        results = list(executor.map(process_batch, batches))
    
    # Flatten results
    flattened = []
    for batch_result in results:
        flattened.extend(batch_result)
    
    return flattened

# --- File I/O Optimizations ---

@cached(ttl=300)  # Cache file reads for 5 minutes
def read_file_cached(file_path: Union[str, Path]) -> str:
    """Read file with caching."""
    path = Path(file_path)
    return path.read_text(encoding='utf-8')

def write_file_atomic(file_path: Union[str, Path], content: str) -> None:
    """Write file atomically to prevent corruption."""
    path = Path(file_path)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    
    try:
        temp_path.write_text(content, encoding='utf-8')
        temp_path.rename(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

# --- Streamlit-specific Optimizations ---

def optimize_streamlit_session():
    """Optimize Streamlit session state for better performance."""
    try:
        import streamlit as st
        
        # Clean up old session state entries
        keys_to_remove = []
        for key in st.session_state:
            if key.startswith('temp_') or key.endswith('_cache'):
                # Remove temporary keys older than 1 hour
                if hasattr(st.session_state[key], 'timestamp'):
                    if time.time() - st.session_state[key].timestamp > 3600:
                        keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        # Force garbage collection
        memory_manager.cleanup_if_needed()
        
        logger.info(f"Streamlit session optimized. Removed {len(keys_to_remove)} old entries.")
        
    except ImportError:
        logger.warning("Streamlit not available for session optimization.")

# --- Performance Testing Utilities ---

@contextmanager
def performance_test(test_name: str):
    """Context manager for performance testing."""
    start_time = time.time()
    start_memory = memory_manager.get_memory_usage_mb()
    
    logger.info(f"Performance test '{test_name}' started")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = memory_manager.get_memory_usage_mb()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(
            f"Performance test '{test_name}' completed: "
            f"Duration: {duration:.2f}s, "
            f"Memory delta: {memory_delta:+.1f}MB"
        )

# --- Export functions for easy access ---
__all__ = [
    'memory_manager',
    'smart_cache', 
    'cached',
    'timed',
    'perf_monitor',
    'normalize_text',
    'chunk_text_efficiently',
    'batch_process',
    'read_file_cached',
    'write_file_atomic',
    'optimize_streamlit_session',
    'performance_test',
    'MemoryManager',
    'SmartCache',
    'PerformanceMonitor'
] 