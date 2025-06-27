"""
Legal RAG Ingestion Module
=========================

Document ingestion, chunking, and embedding pipeline with SOTA models.
"""

from .pdf_reader import extract_text_with_fallback
from .chunker import LegalSemanticChunker
from .embed import SOTAEmbeddingPipeline

__all__ = [
    'extract_text_with_fallback',
    'LegalSemanticChunker', 
    'SOTAEmbeddingPipeline'
] 