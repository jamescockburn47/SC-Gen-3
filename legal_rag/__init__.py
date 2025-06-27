"""
Strategic Counsel SOTA Legal RAG System
======================================

A state-of-the-art Retrieval-Augmented Generation system specifically designed for legal documents,
featuring BGE embeddings, semantic chunking, and hallucination-controlled outputs.

Components:
- legal_rag.ingest: Document ingestion and chunking
- legal_rag.rag: Retrieval and generation pipeline  
- legal_rag.cli: Command-line interface

Compatible with existing Strategic Counsel infrastructure while providing SOTA capabilities.
"""

__version__ = "2.0.0"
__author__ = "Strategic Counsel Team"

from typing import Dict, Any
import logging

# Configure logging for the legal RAG system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# System configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "reranker_model": "BAAI/bge-reranker-base", 
    "generator_model": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
    "chunk_size": 400,  # tokens
    "chunk_overlap": 80,  # tokens
    "vector_search_k": 25,
    "rerank_top_n": 8,
    "citation_verification": True,
    "hallucination_control": True
}

def get_version() -> str:
    """Get the current version of the legal RAG system."""
    return __version__

def get_config() -> Dict[str, Any]:
    """Get the default system configuration."""
    return DEFAULT_CONFIG.copy() 