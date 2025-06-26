# hierarchical_rag_adapter.py
"""
Adapter to integrate the new Hierarchical RAG Pipeline with existing interface
Provides backward compatibility while enabling SOTA hierarchical features
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

try:
    from hierarchical_rag_pipeline import HierarchicalRAGPipeline, get_hierarchical_rag_pipeline
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    print("Warning: Hierarchical RAG not available, falling back to basic RAG")

from local_rag_pipeline import LocalRAGPipeline, rag_session_manager
from config import logger

class AdaptiveRAGPipeline:
    """
    Adaptive RAG Pipeline that intelligently chooses between:
    1. Hierarchical RAG (SOTA) - for new documents and comprehensive analysis
    2. Legacy RAG - for existing documents and simple queries
    """
    
    def __init__(self, matter_id: str):
        self.matter_id = matter_id
        
        # Initialize both pipelines
        self.legacy_pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        
        if HIERARCHICAL_AVAILABLE:
            self.hierarchical_pipeline = get_hierarchical_rag_pipeline(matter_id)
            self.hierarchical_enabled = True
        else:
            self.hierarchical_pipeline = None
            self.hierarchical_enabled = False
    
    def get_document_status(self) -> Dict[str, Any]:
        """Get combined status from both pipelines"""
        
        # Get legacy status
        legacy_status = self.legacy_pipeline.get_document_status()
        
        # Get hierarchical status if available
        if self.hierarchical_enabled:
            hierarchical_status = self.hierarchical_pipeline.get_status()
            
            # Combine the information
            return {
                'total_documents': legacy_status['total_documents'] + hierarchical_status['total_documents'],
                'total_chunks': legacy_status['total_chunks'] + hierarchical_status['total_chunks'],
                'legacy_documents': legacy_status['total_documents'],
                'legacy_chunks': legacy_status['total_chunks'],
                'hierarchical_documents': hierarchical_status['total_documents'],
                'hierarchical_chunks': hierarchical_status['total_chunks'],
                'hierarchical_available': True,
                'storage_path': legacy_status['storage_path'],
                'configuration': legacy_status.get('configuration', {}),
                'documents': legacy_status['documents'] + hierarchical_status.get('documents', [])
            }
        else:
            # Only legacy available
            legacy_status['hierarchical_available'] = False
            legacy_status['hierarchical_documents'] = 0
            legacy_status['hierarchical_chunks'] = 0
            return legacy_status
    
    async def intelligent_search(self, query: str, max_chunks: int = 15) -> List[Dict[str, Any]]:
        """
        Intelligent search that uses the best available pipeline
        """
        
        # Analyze query complexity
        is_comprehensive = any(keyword in query.lower() for keyword in 
                             ['summarize', 'summarise', 'overview', 'comprehensive', 'all', 'entire'])
        
        is_cross_document = any(keyword in query.lower() for keyword in 
                               ['compare', 'contrast', 'between', 'versus', 'relationship'])
        
        # Decide which pipeline to use
        use_hierarchical = (
            self.hierarchical_enabled and 
            (is_comprehensive or is_cross_document) and
            self.hierarchical_pipeline.get_status()['total_chunks'] > 0
        )
        
        if use_hierarchical:
            logger.info(f"Using hierarchical search for query: {query[:50]}...")
            
            # Use hierarchical pipeline for complex queries
            hierarchical_results = await self.hierarchical_pipeline.intelligent_hierarchical_search(
                query, max_chunks
            )
            
            # Also get some legacy results for completeness
            legacy_chunk_count = max(2, max_chunks // 4)  # 25% from legacy
            legacy_results = self.legacy_pipeline.search_documents(query, legacy_chunk_count)
            
            # Combine and deduplicate results
            all_results = hierarchical_results + legacy_results
            return self._deduplicate_results(all_results, max_chunks)
        
        else:
            logger.info(f"Using legacy search for query: {query[:50]}...")
            # Use legacy pipeline for simple queries or when hierarchical not available
            return self.legacy_pipeline.search_documents(query, max_chunks)
    
    def _deduplicate_results(self, results: List[Dict[str, Any]], max_chunks: int) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on text similarity"""
        
        unique_results = []
        seen_texts = set()
        
        for result in results:
            # Use first 100 characters as deduplication key
            text_key = result.get('text', '')[:100].lower().strip()
            if text_key and text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)
                
                if len(unique_results) >= max_chunks:
                    break
        
        return unique_results
    
    def add_document_legacy(self, file_obj, filename: str, ocr_preference: str = "aws"):
        """Add document using legacy pipeline"""
        return self.legacy_pipeline.add_document(file_obj, filename, ocr_preference)
    
    async def add_document_hierarchical(self, file_obj, filename: str, ocr_preference: str = "aws"):
        """Add document using hierarchical pipeline"""
        if not self.hierarchical_enabled:
            return False, "Hierarchical RAG not available", {}
        
        return await self.hierarchical_pipeline.add_document(file_obj, filename, ocr_preference)
    
    def should_use_hierarchical_for_upload(self, filename: str) -> bool:
        """Determine if a document should use hierarchical processing"""
        
        if not self.hierarchical_enabled:
            return False
        
        # Use hierarchical for:
        # 1. Legal documents (likely to benefit from document summarization)
        # 2. Large documents (likely to benefit from hierarchical chunking)
        # 3. Structured documents (PDFs, Word docs)
        
        hierarchical_extensions = ['.pdf', '.docx', '.doc', '.rtf']
        hierarchical_keywords = ['contract', 'agreement', 'legal', 'policy', 'manual', 'report']
        
        filename_lower = filename.lower()
        
        # Check file extension
        if any(filename_lower.endswith(ext) for ext in hierarchical_extensions):
            return True
        
        # Check filename keywords
        if any(keyword in filename_lower for keyword in hierarchical_keywords):
            return True
        
        return False

# Global adaptive session manager
adaptive_rag_manager = {}

def get_adaptive_rag_pipeline(matter_id: str) -> AdaptiveRAGPipeline:
    """Get or create adaptive RAG pipeline instance"""
    global adaptive_rag_manager
    
    if matter_id not in adaptive_rag_manager:
        adaptive_rag_manager[matter_id] = AdaptiveRAGPipeline(matter_id)
    
    return adaptive_rag_manager[matter_id]

def get_rag_capabilities() -> Dict[str, Any]:
    """Get information about available RAG capabilities"""
    return {
        'legacy_rag_available': True,
        'hierarchical_rag_available': HIERARCHICAL_AVAILABLE,
        'adaptive_routing': True,
        'features': {
            'legacy': ['basic_chunking', 'vector_search', 'ollama_integration'],
            'hierarchical': ['document_summarization', 'multi_level_chunking', 'coarse_to_fine_search', 'query_complexity_analysis'] if HIERARCHICAL_AVAILABLE else [],
            'adaptive': ['intelligent_routing', 'result_deduplication', 'pipeline_selection']
        }
    } 