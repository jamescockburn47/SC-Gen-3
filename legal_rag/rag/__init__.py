"""
Legal RAG Retrieval and Generation Module
=========================================

SOTA retrieval and generation pipeline for legal documents.
"""

from .retriever import SOTALegalRetriever
from .generator import LlamaGenerator
from .chain import LegalRAGChain

__all__ = [
    'SOTALegalRetriever',
    'LlamaGenerator', 
    'LegalRAGChain'
] 