# session_state_manager.py

"""
Optimized session state management for Strategic Counsel Streamlit application.
Provides efficient state management, caching, and cleanup utilities.
"""

import time
import hashlib
import logging
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    logger.warning("Streamlit not available. Session state manager will use fallback.")
    STREAMLIT_AVAILABLE = False
    # Create a mock streamlit module for type consistency
    class MockStreamlit:
        class session_state:
            _data = {}
            
            def __getitem__(self, key):
                return self._data.get(key)
            
            def __setitem__(self, key, value):
                self._data[key] = value
            
            def __contains__(self, key):
                return key in self._data
            
            def get(self, key, default=None):
                return self._data.get(key, default)
            
            def keys(self):
                return self._data.keys()
            
            def pop(self, key, default=None):
                return self._data.pop(key, default)
    
    st = MockStreamlit()

@dataclass
class SessionValue:
    """Wrapper for session state values with metadata."""
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if the value has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def access(self) -> Any:
        """Access the value and update metadata."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value

class OptimizedSessionStateManager:
    """Optimized session state manager for better performance."""
    
    def __init__(self):
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.max_memory_items = 1000
        self.default_ttl = 3600  # 1 hour
        
        # Initialize session state with optimized defaults
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize session state with optimized default values."""
        defaults = {
            "current_topic": "general_default_topic",
            "session_history": [],
            "loaded_memories": [],
            "processed_summaries": [],
            "selected_summary_texts": [],
            "latest_digest_content": "",
            "document_processing_complete": True,
            "ch_last_digest_path": None,
            "ch_last_df": None,
            "ch_last_narrative": None,
            "ch_last_batch_metrics": {},
            "consult_digest_model": "gpt-4o",
            "ch_analysis_summaries_for_injection": [],
            "ocr_method": "aws",
            "ocr_method_radio": 0,
            "user_instruction_main_text_area_value": "",
            "original_user_instruction_main": "",
            "user_instruction_main_is_improved": False,
            "additional_ai_instructions_ch_text_area_value": "",
            "original_additional_ai_instructions_ch": "",
            "additional_ai_instructions_ch_is_improved": False,
            "ch_available_documents": [],
            "ch_document_selection": {},
            "ch_start_year_input_main": datetime.now().year - 4,
            "ch_end_year_input_main": datetime.now().year,
            "group_structure_cn_for_analysis": "",
            "group_structure_report": [],
            "group_structure_viz_data": None,
            "suggested_parent_cn_for_rerun": None,
            "group_structure_parent_timeline": [],
            "latest_ai_response_for_protocol_check": None,
            "ch_company_profiles_map": {},
            # Performance tracking
            "_session_initialized": True,
            "_last_optimized": time.time(),
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def set_with_ttl(self, key: str, value: Any, ttl_seconds: int = None) -> None:
        """Set a session state value with TTL."""
        ttl = ttl_seconds or self.default_ttl
        expires_at = time.time() + ttl
        
        wrapped_value = SessionValue(
            value=value,
            expires_at=expires_at
        )
        
        st.session_state[f"_wrapped_{key}"] = wrapped_value
        st.session_state[key] = value  # Keep direct access for compatibility
    
    def get_with_metadata(self, key: str, default: Any = None) -> Any:
        """Get a value with metadata tracking."""
        wrapped_key = f"_wrapped_{key}"
        
        if wrapped_key in st.session_state:
            wrapped_value: SessionValue = st.session_state[wrapped_key]
            
            if wrapped_value.is_expired():
                # Clean up expired value
                del st.session_state[wrapped_key]
                if key in st.session_state:
                    del st.session_state[key]
                return default
            
            return wrapped_value.access()
        
        # Fallback to direct access
        return st.session_state.get(key, default)
    
    def reset_topic_state(self, new_topic: str) -> None:
        """Efficiently reset topic-specific state."""
        topic_specific_keys = [
            "session_history",
            "processed_summaries", 
            "selected_summary_texts",
            "loaded_memories",
            "latest_digest_content",
            "ch_last_digest_path",
            "ch_last_df", 
            "ch_last_narrative",
            "ch_last_batch_metrics",
            "ch_analysis_summaries_for_injection",
            "group_structure_cn_for_analysis",
            "group_structure_report",
            "group_structure_viz_data",
            "suggested_parent_cn_for_rerun",
            "group_structure_parent_timeline",
            "latest_ai_response_for_protocol_check",
            "ch_company_profiles_map"
        ]
        
        # Clear topic-specific state efficiently
        for key in topic_specific_keys:
            if key in st.session_state:
                del st.session_state[key]
            
            # Also clear wrapped versions
            wrapped_key = f"_wrapped_{key}"
            if wrapped_key in st.session_state:
                del st.session_state[wrapped_key]
        
        # Set new topic and reinitialize
        st.session_state.current_topic = new_topic
        self._initialize_defaults()
        
        logger.info(f"Topic state reset to: {new_topic}")
    
    def sync_ocr_state(self) -> None:
        """Ensure OCR method and radio index are synchronized."""
        ocr_method_map = {"aws": 0, "google": 1, "none": 2}
        current_method = st.session_state.get("ocr_method", "aws")
        
        # Validate and correct OCR method if needed
        if current_method not in ocr_method_map:
            current_method = "aws"
            st.session_state.ocr_method = current_method
        
        # Ensure radio index matches the method
        st.session_state.ocr_method_radio = ocr_method_map[current_method]
    
    def cleanup_expired(self, force: bool = False) -> int:
        """Clean up expired session state values."""
        current_time = time.time()
        
        if not force and (current_time - self.last_cleanup) < self.cleanup_interval:
            return 0
        
        cleaned_count = 0
        keys_to_remove = []
        
        # Find expired wrapped values
        for key in st.session_state.keys():
            if key.startswith("_wrapped_"):
                wrapped_value = st.session_state[key]
                if isinstance(wrapped_value, SessionValue) and wrapped_value.is_expired():
                    keys_to_remove.append(key)
                    # Also remove the unwrapped version
                    original_key = key.replace("_wrapped_", "")
                    if original_key in st.session_state:
                        keys_to_remove.append(original_key)
        
        # Remove expired keys
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
                cleaned_count += 1
        
        self.last_cleanup = current_time
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired session state entries")
        
        return cleaned_count
    
    def get_memory_usage_estimate(self) -> Dict[str, Any]:
        """Estimate memory usage of session state."""
        import sys
        
        total_size = 0
        item_count = 0
        largest_items = []
        
        for key, value in st.session_state.items():
            if key.startswith('_'):  # Skip internal keys
                continue
                
            try:
                size = sys.getsizeof(value)
                total_size += size
                item_count += 1
                
                largest_items.append((key, size))
            except Exception:
                # Skip items that can't be sized
                continue
        
        # Sort by size and keep top 10
        largest_items.sort(key=lambda x: x[1], reverse=True)
        largest_items = largest_items[:10]
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'item_count': item_count,
            'largest_items': largest_items,
            'average_item_size': total_size / max(item_count, 1)
        }
    
    def optimize_large_collections(self) -> None:
        """Optimize large collections in session state."""
        optimization_targets = [
            'session_history',
            'processed_summaries',
            'ch_analysis_summaries_for_injection'
        ]
        
        for key in optimization_targets:
            if key in st.session_state:
                collection = st.session_state[key]
                
                if isinstance(collection, list) and len(collection) > 100:
                    # Keep only the most recent 100 items
                    st.session_state[key] = collection[-100:]
                    logger.info(f"Optimized {key}: trimmed to 100 most recent items")
                
                elif isinstance(collection, dict) and len(collection) > 50:
                    # For dictionaries, keep a reasonable subset
                    # This is a simple approach - in practice, you might want more sophisticated logic
                    items = list(collection.items())[-50:]
                    st.session_state[key] = dict(items)
                    logger.info(f"Optimized {key}: trimmed to 50 most recent items")
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key for session state operations."""
        key_string = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def bulk_update(self, updates: Dict[str, Any]) -> None:
        """Efficiently update multiple session state values."""
        for key, value in updates.items():
            st.session_state[key] = value
        
        logger.debug(f"Bulk updated {len(updates)} session state values")
    
    def export_state(self, include_internal: bool = False) -> Dict[str, Any]:
        """Export current session state for debugging or backup."""
        state = {}
        
        for key, value in st.session_state.items():
            if not include_internal and key.startswith('_'):
                continue
            
            try:
                # Only include serializable values
                import json
                json.dumps(value)  # Test serialization
                state[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                state[key] = f"<non-serializable: {type(value).__name__}>"
        
        return state
    
    def performance_summary(self) -> Dict[str, Any]:
        """Get a summary of session state performance metrics."""
        memory_info = self.get_memory_usage_estimate()
        
        return {
            'memory_usage_mb': memory_info['total_size_mb'],
            'item_count': memory_info['item_count'],
            'last_cleanup': self.last_cleanup,
            'time_since_cleanup': time.time() - self.last_cleanup,
            'largest_items': memory_info['largest_items'][:5],  # Top 5 only
            'initialization_time': st.session_state.get('_last_optimized', 0)
        }

# Global session state manager instance
session_manager = OptimizedSessionStateManager()

# Convenience functions for common operations
def init_session_state():
    """Initialize session state with optimized defaults."""
    session_manager._initialize_defaults()
    session_manager.sync_ocr_state()

def reset_topic(new_topic: str):
    """Reset topic-specific session state."""
    session_manager.reset_topic_state(new_topic)

def optimize_session():
    """Perform session state optimization."""
    cleaned = session_manager.cleanup_expired()
    session_manager.optimize_large_collections()
    return cleaned

def get_session_stats():
    """Get session state performance statistics."""
    return session_manager.performance_summary()

# Export for easy access
__all__ = [
    'OptimizedSessionStateManager',
    'session_manager',
    'init_session_state', 
    'reset_topic',
    'optimize_session',
    'get_session_stats'
] 