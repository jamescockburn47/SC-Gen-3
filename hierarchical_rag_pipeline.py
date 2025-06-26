# hierarchical_rag_pipeline.py
"""
Hierarchical RAG Pipeline - State-of-the-Art Implementation
Based on latest research: LongRAG, MacRAG, LongRefiner (2024-2025)

Key features:
1. Document-level summarization during upload
2. Multi-scale hierarchical chunking (document -> section -> paragraph -> sentence)
3. Coarse-to-fine retrieval strategy
4. Query-adaptive context selection
5. Intelligent coverage optimization
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import asyncio
import re
from dataclasses import dataclass
from enum import Enum

# Core dependencies
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import aiohttp
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Missing dependencies for hierarchical RAG: {e}")

from config import logger, APP_BASE_PATH
from app_utils import extract_text_from_uploaded_file

class QueryComplexity(Enum):
    """Query complexity levels for adaptive retrieval"""
    SIMPLE_FACT = "simple_fact"          # Single fact lookup
    DETAILED_ANALYSIS = "detailed"       # Multi-paragraph analysis
    COMPREHENSIVE = "comprehensive"      # Full document summarization
    CROSS_DOCUMENT = "cross_document"    # Multi-document synthesis

class DocumentLevel(Enum):
    """Hierarchical document levels"""
    DOCUMENT = "document"     # Full document summary
    SECTION = "section"       # Major sections/chapters
    PARAGRAPH = "paragraph"   # Paragraph-level chunks
    SENTENCE = "sentence"     # Sentence-level for precision

@dataclass
class HierarchicalChunk:
    """Enhanced chunk with hierarchical metadata"""
    id: str
    text: str
    level: DocumentLevel
    doc_id: str
    parent_id: Optional[str] = None
    children_ids: Optional[List[str]] = None
    section_title: Optional[str] = None
    paragraph_index: Optional[int] = None
    sentence_index: Optional[int] = None
    word_count: int = 0
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    created_at: str = ""

@dataclass
class DocumentSummary:
    """Document-level summary with metadata"""
    doc_id: str
    filename: str
    full_summary: str
    key_topics: List[str]
    section_summaries: Dict[str, str]
    total_sections: int
    total_paragraphs: int
    total_sentences: int
    content_type: str  # legal, technical, narrative, etc.
    created_at: str

class HierarchicalRAGPipeline:
    """
    Advanced hierarchical RAG pipeline implementing SOTA techniques
    """
    
    def __init__(self, 
                 matter_id: str,
                 embedding_model: str = "all-mpnet-base-v2",
                 ollama_base_url: str = "http://localhost:11434",
                 summarization_model: str = "mistral:latest"):
        
        self.matter_id = matter_id
        self.embedding_model_name = embedding_model
        self.ollama_base_url = ollama_base_url
        self.summarization_model = summarization_model
        
        # Initialize paths
        self.rag_base_path = APP_BASE_PATH / "hierarchical_rag" / matter_id
        self.vector_db_path = self.rag_base_path / "vector_db"
        self.documents_path = self.rag_base_path / "documents"
        self.summaries_path = self.rag_base_path / "summaries"
        self.metadata_path = self.rag_base_path / "hierarchical_metadata.json"
        
        # Create directories
        for path in [self.rag_base_path, self.vector_db_path, self.documents_path, self.summaries_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_model = None
        self.document_level_index = None  # For document summaries
        self.section_level_index = None   # For section summaries
        self.paragraph_level_index = None # For paragraph chunks
        self.sentence_level_index = None  # For sentence-level precision
        
        # Metadata storage
        self.document_summaries: Dict[str, DocumentSummary] = {}
        self.hierarchical_chunks: Dict[str, HierarchicalChunk] = {}
        self.document_metadata: Dict[str, Dict] = {}
        
        # Load existing data
        self._load_metadata()
        self._initialize_embedding_model()
        self._load_vector_indices()
    
    def _initialize_embedding_model(self) -> bool:
        """Initialize embedding model with GPU optimization"""
        if not DEPENDENCIES_AVAILABLE:
            logger.error("Dependencies not available for hierarchical RAG")
            return False
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # GPU optimization
            try:
                import torch
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.cuda()
                    logger.info("Enabled GPU acceleration for embedding model")
            except ImportError:
                pass
            
            logger.info(f"Initialized hierarchical embedding model: {self.embedding_model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False
    
    def _load_metadata(self):
        """Load hierarchical metadata"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    
                    # Load document summaries
                    for doc_id, summary_data in data.get('document_summaries', {}).items():
                        self.document_summaries[doc_id] = DocumentSummary(**summary_data)
                    
                    # Load hierarchical chunks
                    for chunk_id, chunk_data in data.get('hierarchical_chunks', {}).items():
                        chunk_data['level'] = DocumentLevel(chunk_data['level'])
                        if chunk_data.get('children_ids') is None:
                            chunk_data['children_ids'] = []
                        self.hierarchical_chunks[chunk_id] = HierarchicalChunk(**chunk_data)
                    
                    self.document_metadata = data.get('document_metadata', {})
                    
                logger.info(f"Loaded hierarchical metadata: {len(self.document_summaries)} documents, "
                          f"{len(self.hierarchical_chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to load hierarchical metadata: {e}")
    
    def _save_metadata(self):
        """Save hierarchical metadata"""
        try:
            # Convert to serializable format
            doc_summaries_data = {}
            for doc_id, summary in self.document_summaries.items():
                doc_summaries_data[doc_id] = {
                    'doc_id': summary.doc_id,
                    'filename': summary.filename,
                    'full_summary': summary.full_summary,
                    'key_topics': summary.key_topics,
                    'section_summaries': summary.section_summaries,
                    'total_sections': summary.total_sections,
                    'total_paragraphs': summary.total_paragraphs,
                    'total_sentences': summary.total_sentences,
                    'content_type': summary.content_type,
                    'created_at': summary.created_at
                }
            
            chunks_data = {}
            for chunk_id, chunk in self.hierarchical_chunks.items():
                chunks_data[chunk_id] = {
                    'id': chunk.id,
                    'text': chunk.text,
                    'level': chunk.level.value,
                    'doc_id': chunk.doc_id,
                    'parent_id': chunk.parent_id,
                    'children_ids': chunk.children_ids or [],
                    'section_title': chunk.section_title,
                    'paragraph_index': chunk.paragraph_index,
                    'sentence_index': chunk.sentence_index,
                    'word_count': chunk.word_count,
                    'summary': chunk.summary,
                    'keywords': chunk.keywords or [],
                    'similarity_score': chunk.similarity_score,
                    'created_at': chunk.created_at
                }
            
            data = {
                'document_summaries': doc_summaries_data,
                'hierarchical_chunks': chunks_data,
                'document_metadata': self.document_metadata,
                'configuration': {
                    'embedding_model': self.embedding_model_name,
                    'summarization_model': self.summarization_model,
                    'pipeline_version': '1.0.0-hierarchical'
                }
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save hierarchical metadata: {e}")
    
    def _load_vector_indices(self):
        """Load hierarchical vector indices"""
        if not DEPENDENCIES_AVAILABLE:
            return
        
        index_files = {
            'document': self.vector_db_path / "document_index.bin",
            'section': self.vector_db_path / "section_index.bin",
            'paragraph': self.vector_db_path / "paragraph_index.bin",
            'sentence': self.vector_db_path / "sentence_index.bin"
        }
        
        for level, path in index_files.items():
            if path.exists():
                try:
                    index = faiss.read_index(str(path))
                    setattr(self, f"{level}_level_index", index)
                    logger.info(f"Loaded {level} index with {index.ntotal} vectors")
                except Exception as e:
                    logger.error(f"Failed to load {level} index: {e}")
    
    def _save_vector_indices(self):
        """Save hierarchical vector indices"""
        if not DEPENDENCIES_AVAILABLE:
            return
        
        indices = {
            'document': self.document_level_index,
            'section': self.section_level_index,
            'paragraph': self.paragraph_level_index,
            'sentence': self.sentence_level_index
        }
        
        for level, index in indices.items():
            if index is not None:
                try:
                    path = self.vector_db_path / f"{level}_index.bin"
                    faiss.write_index(index, str(path))
                except Exception as e:
                    logger.error(f"Failed to save {level} index: {e}")
    
    async def _generate_document_summary(self, text: str, filename: str) -> DocumentSummary:
        """Generate comprehensive document summary using LLM"""
        
        # Analyze document structure
        sections = self._extract_sections(text)
        paragraphs = self._extract_paragraphs(text)
        sentences = self._extract_sentences(text)
        
        # Generate full document summary
        summary_prompt = f"""
Analyze this document comprehensively and provide:

1. A concise 2-3 paragraph summary of the entire document
2. 5-10 key topics/themes covered
3. Document type classification (legal, technical, business, academic, etc.)
4. Brief summary of each major section

Document filename: {filename}
Document content:
{text[:8000]}... (showing first 8000 characters)

Provide response in this exact format:
FULL_SUMMARY: [2-3 paragraph summary]
KEY_TOPICS: [topic1, topic2, topic3, ...]
CONTENT_TYPE: [classification]
SECTION_SUMMARIES: [For each major section, provide: Section Title | Brief summary]
"""
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.summarization_model,
                    "prompt": summary_prompt,
                    "stream": False,
                    "temperature": 0.1
                }
                
                async with session.post(f"{self.ollama_base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary_text = result.get('response', '')
                        
                        # Parse the structured response
                        full_summary = self._extract_section_from_response(summary_text, "FULL_SUMMARY")
                        topics_text = self._extract_section_from_response(summary_text, "KEY_TOPICS")
                        content_type = self._extract_section_from_response(summary_text, "CONTENT_TYPE")
                        sections_text = self._extract_section_from_response(summary_text, "SECTION_SUMMARIES")
                        
                        # Parse topics and sections
                        key_topics = [topic.strip() for topic in topics_text.split(',') if topic.strip()]
                        section_summaries = self._parse_section_summaries(sections_text)
                        
                        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
                        
                        return DocumentSummary(
                            doc_id=doc_id,
                            filename=filename,
                            full_summary=full_summary or "Summary generation failed",
                            key_topics=key_topics[:10],  # Limit to 10 topics
                            section_summaries=section_summaries,
                            total_sections=len(sections),
                            total_paragraphs=len(paragraphs),
                            total_sentences=len(sentences),
                            content_type=content_type or "unknown",
                            created_at=datetime.now().isoformat()
                        )
                        
        except Exception as e:
            logger.error(f"Failed to generate document summary: {e}")
            
        # Fallback summary
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return DocumentSummary(
            doc_id=doc_id,
            filename=filename,
            full_summary=f"Document analysis of {filename} containing approximately {len(text.split())} words.",
            key_topics=["document analysis"],
            section_summaries={},
            total_sections=len(sections),
            total_paragraphs=len(paragraphs),
            total_sentences=len(sentences),
            content_type="unknown",
            created_at=datetime.now().isoformat()
        )
    
    def _extract_section_from_response(self, response: str, section_name: str) -> str:
        """Extract specific section from LLM response"""
        pattern = f"{section_name}:\\s*(.+?)(?=\\n[A-Z_]+:|$)"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _parse_section_summaries(self, sections_text: str) -> Dict[str, str]:
        """Parse section summaries from response"""
        summaries = {}
        for line in sections_text.split('\n'):
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    title = parts[0].strip()
                    summary = parts[1].strip()
                    summaries[title] = summary
        return summaries
    
    def _extract_sections(self, text: str) -> List[str]:
        """Extract major sections from document"""
        # Look for common section headers
        section_patterns = [
            r'\n\s*\d+\.\s+[A-Z][^.\n]+\n',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]+\n',         # ALL CAPS headers
            r'\n\s*#{1,3}\s+[^#\n]+\n',      # Markdown headers
            r'\n\s*[A-Z][^.\n]{10,50}\n'     # Title case headers
        ]
        
        sections = []
        for pattern in section_patterns:
            matches = re.findall(pattern, text)
            sections.extend([match.strip() for match in matches])
        
        return list(set(sections))  # Remove duplicates
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from document"""
        # Split by double newlines, filter out short segments
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from document"""
        # Basic sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _create_hierarchical_chunks(self, text: str, doc_id: str, doc_summary: DocumentSummary) -> List[HierarchicalChunk]:
        """Create multi-level hierarchical chunks"""
        chunks = []
        
        # Level 1: Document-level chunk (summary)
        doc_chunk = HierarchicalChunk(
            id=f"{doc_id}_doc",
            text=doc_summary.full_summary,
            level=DocumentLevel.DOCUMENT,
            doc_id=doc_id,
            summary=doc_summary.full_summary,
            keywords=doc_summary.key_topics,
            word_count=len(doc_summary.full_summary.split()),
            created_at=datetime.now().isoformat()
        )
        chunks.append(doc_chunk)
        
        # Level 2: Section-level chunks
        sections = self._extract_paragraphs(text)  # Using paragraphs as sections for now
        section_children = []
        
        for i, section in enumerate(sections[:20]):  # Limit to 20 sections
            section_id = f"{doc_id}_section_{i}"
            section_chunk = HierarchicalChunk(
                id=section_id,
                text=section,
                level=DocumentLevel.SECTION,
                doc_id=doc_id,
                parent_id=doc_chunk.id,
                section_title=f"Section {i+1}",
                word_count=len(section.split()),
                created_at=datetime.now().isoformat()
            )
            chunks.append(section_chunk)
            section_children.append(section_id)
            
            # Level 3: Paragraph-level chunks (split long sections)
            if len(section.split()) > 150:  # Split long sections
                paragraph_chunks = self._split_into_paragraphs(section, section_id, doc_id, i)
                chunks.extend(paragraph_chunks)
                
                # Update parent-child relationships
                para_children = [chunk.id for chunk in paragraph_chunks]
                section_chunk.children_ids = para_children
        
        # Update document chunk children
        doc_chunk.children_ids = section_children
        
        return chunks
    
    def _split_into_paragraphs(self, section_text: str, parent_id: str, doc_id: str, section_index: int) -> List[HierarchicalChunk]:
        """Split section into paragraph-level chunks"""
        chunks = []
        sentences = self._extract_sentences(section_text)
        
        # Group sentences into paragraph-sized chunks
        chunk_size = 3  # sentences per paragraph chunk
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = '. '.join(chunk_sentences) + '.'
            
            para_chunk = HierarchicalChunk(
                id=f"{parent_id}_para_{i // chunk_size}",
                text=chunk_text,
                level=DocumentLevel.PARAGRAPH,
                doc_id=doc_id,
                parent_id=parent_id,
                paragraph_index=i // chunk_size,
                word_count=len(chunk_text.split()),
                created_at=datetime.now().isoformat()
            )
            chunks.append(para_chunk)
        
        return chunks
    
    def _embed_hierarchical_chunks(self, chunks: List[HierarchicalChunk]) -> bool:
        """Generate embeddings for all hierarchical chunks and build indices"""
        if not self.embedding_model or not DEPENDENCIES_AVAILABLE:
            return False
        
        try:
            # Group chunks by level
            chunks_by_level = {
                DocumentLevel.DOCUMENT: [],
                DocumentLevel.SECTION: [],
                DocumentLevel.PARAGRAPH: [],
                DocumentLevel.SENTENCE: []
            }
            
            for chunk in chunks:
                chunks_by_level[chunk.level].append(chunk)
            
            # Process each level
            for level, level_chunks in chunks_by_level.items():
                if not level_chunks:
                    continue
                
                # Generate embeddings
                texts = [chunk.text for chunk in level_chunks]
                embeddings = self.embedding_model.encode(texts)
                
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.astype(np.float32)
                    faiss.normalize_L2(embeddings)
                    
                    # Update chunks with embeddings
                    for chunk, embedding in zip(level_chunks, embeddings):
                        chunk.embedding = embedding
                    
                    # Create or update index for this level
                    index_attr = f"{level.value}_level_index"
                    current_index = getattr(self, index_attr)
                    
                    if current_index is None:
                        # Create new index
                        dimension = embeddings.shape[1]
                        new_index = faiss.IndexFlatIP(dimension)
                        new_index.add(embeddings)
                        setattr(self, index_attr, new_index)
                    else:
                        # Add to existing index
                        current_index.add(embeddings)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed hierarchical chunks: {e}")
            return False
    
    def classify_query_complexity(self, query: str) -> QueryComplexity:
        """Classify query complexity for adaptive retrieval"""
        query_lower = query.lower()
        
        # Simple fact queries
        simple_indicators = ['what is', 'who is', 'when did', 'where is', 'how much', 'is there']
        if any(indicator in query_lower for indicator in simple_indicators):
            return QueryComplexity.SIMPLE_FACT
        
        # Comprehensive analysis queries
        comprehensive_indicators = ['summarize', 'summarise', 'overview', 'comprehensive', 'all', 'entire', 'complete analysis']
        if any(indicator in query_lower for indicator in comprehensive_indicators):
            return QueryComplexity.COMPREHENSIVE
        
        # Cross-document queries
        cross_doc_indicators = ['compare', 'contrast', 'between', 'versus', 'relationship', 'across documents']
        if any(indicator in query_lower for indicator in cross_doc_indicators):
            return QueryComplexity.CROSS_DOCUMENT
        
        # Default to detailed analysis
        return QueryComplexity.DETAILED_ANALYSIS
    
    async def add_document(self, file_obj, filename: str, 
                          ocr_preference: str = "aws") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Add document with full hierarchical processing and vector database creation
        """
        if not self.embedding_model:
            return False, "Embedding model not available", {}
        
        # Extract text using existing utilities
        from app_utils import extract_text_from_uploaded_file
        text_content, error_message = extract_text_from_uploaded_file(
            file_obj, filename, ocr_preference
        )
        
        if not text_content:
            return False, f"Failed to extract text: {error_message}", {}
        
        try:
            logger.info(f"Processing document {filename} with hierarchical RAG...")
            
            # Step 1: Generate document summary using mistral
            doc_summary = await self._generate_document_summary(text_content, filename)
            logger.info(f"Generated document summary: {len(doc_summary.full_summary)} chars")
            
            # Step 2: Create hierarchical chunks
            hierarchical_chunks = self._create_hierarchical_chunks(text_content, doc_summary.doc_id, doc_summary)
            logger.info(f"Created {len(hierarchical_chunks)} hierarchical chunks")
            
            # Step 3: Generate embeddings and build vector indices
            if not self._embed_hierarchical_chunks(hierarchical_chunks):
                return False, "Failed to generate embeddings", {}
            logger.info(f"Generated embeddings for {len(hierarchical_chunks)} chunks")
            
            # Step 4: Store everything
            self.document_summaries[doc_summary.doc_id] = doc_summary
            
            for chunk in hierarchical_chunks:
                self.hierarchical_chunks[chunk.id] = chunk
            
            self.document_metadata[doc_summary.doc_id] = {
                'filename': filename,
                'text_length': len(text_content),
                'total_chunks': len(hierarchical_chunks),
                'created_at': datetime.now().isoformat(),
                'summary': doc_summary.full_summary[:200] + "..." if len(doc_summary.full_summary) > 200 else doc_summary.full_summary
            }
            
            # Step 5: Save to disk
            doc_path = self.documents_path / f"{doc_summary.doc_id}.txt"
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            summary_path = self.summaries_path / f"{doc_summary.doc_id}_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'full_summary': doc_summary.full_summary,
                    'key_topics': doc_summary.key_topics,
                    'section_summaries': doc_summary.section_summaries,
                    'content_type': doc_summary.content_type
                }, f, indent=2)
            
            self._save_metadata()
            self._save_vector_indices()
            
            logger.info(f"Successfully processed {filename} with hierarchical structure")
            
            return True, f"Successfully processed with hierarchical structure: {len(hierarchical_chunks)} chunks", {
                'doc_id': doc_summary.doc_id,
                'filename': filename,
                'summary': doc_summary.full_summary,
                'chunks_by_level': {
                    'document': len([c for c in hierarchical_chunks if c.level == DocumentLevel.DOCUMENT]),
                    'section': len([c for c in hierarchical_chunks if c.level == DocumentLevel.SECTION]),
                    'paragraph': len([c for c in hierarchical_chunks if c.level == DocumentLevel.PARAGRAPH]),
                    'sentence': len([c for c in hierarchical_chunks if c.level == DocumentLevel.SENTENCE])
                },
                'key_topics': doc_summary.key_topics,
                'content_type': doc_summary.content_type
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {filename}: {e}")
            return False, f"Processing error: {str(e)}", {}

    async def intelligent_hierarchical_search(self, query: str, max_total_chunks: int = 15) -> List[Dict[str, Any]]:
        """
        Intelligent hierarchical search with actual FAISS vector search
        Implements coarse-to-fine retrieval strategy
        """
        if not self.embedding_model or not DEPENDENCIES_AVAILABLE:
            return []
        
        try:
            # Step 1: Classify query complexity for adaptive chunking
            complexity = self.classify_query_complexity(query)
            chunk_allocation = self._get_chunk_allocation(complexity, max_total_chunks)
            
            # Step 2: Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.astype(np.float32)
                faiss.normalize_L2(query_embedding.astype(np.float32))
            
            # Step 3: Hierarchical search across levels
            selected_chunks = []
            
            # Document-level search (highest level summary)
            if self.document_level_index and chunk_allocation['document'] > 0:
                doc_chunks = self._search_level(
                    query_embedding, self.document_level_index, 
                    DocumentLevel.DOCUMENT, chunk_allocation['document']
                )
                selected_chunks.extend(doc_chunks)
            
            # Section-level search 
            if self.section_level_index and chunk_allocation['section'] > 0:
                section_chunks = self._search_level(
                    query_embedding, self.section_level_index,
                    DocumentLevel.SECTION, chunk_allocation['section']
                )
                selected_chunks.extend(section_chunks)
            
            # Paragraph-level search (detailed content)
            if self.paragraph_level_index and chunk_allocation['paragraph'] > 0:
                para_chunks = self._search_level(
                    query_embedding, self.paragraph_level_index,
                    DocumentLevel.PARAGRAPH, chunk_allocation['paragraph']
                )
                selected_chunks.extend(para_chunks)
            
            # Step 4: Re-rank and convert to legacy format
            final_chunks = self._rerank_and_convert(selected_chunks, max_total_chunks)
            
            logger.info(f"Hierarchical search returned {len(final_chunks)} chunks "
                       f"(complexity: {complexity.value}, allocation: {chunk_allocation})")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Hierarchical search failed: {e}")
            return []
    
    def _get_chunk_allocation(self, complexity: QueryComplexity, total_chunks: int) -> Dict[str, int]:
        """Determine optimal chunk allocation across hierarchy levels"""
        
        if complexity == QueryComplexity.SIMPLE_FACT:
            # Focus on precise, lower-level chunks
            return {
                'document': max(1, total_chunks // 8),     # 12.5%
                'section': max(1, total_chunks // 4),      # 25%
                'paragraph': max(2, total_chunks // 2),    # 50%
                'sentence': max(1, total_chunks // 8)      # 12.5%
            }
        
        elif complexity == QueryComplexity.COMPREHENSIVE:
            # Focus on higher-level summaries and overviews
            return {
                'document': max(2, total_chunks // 3),     # 33%
                'section': max(3, total_chunks // 2),      # 50%
                'paragraph': max(1, total_chunks // 6),    # 17%
                'sentence': 0                              # 0%
            }
        
        elif complexity == QueryComplexity.CROSS_DOCUMENT:
            # Balance across levels for comparison
            return {
                'document': max(2, total_chunks // 4),     # 25%
                'section': max(2, total_chunks // 3),      # 33%
                'paragraph': max(2, total_chunks // 3),    # 33%
                'sentence': max(1, total_chunks // 12)     # 9%
            }
        
        else:  # DETAILED_ANALYSIS
            # Balanced approach
            return {
                'document': max(1, total_chunks // 6),     # 17%
                'section': max(2, total_chunks // 3),      # 33%
                'paragraph': max(2, total_chunks // 3),    # 33%
                'sentence': max(1, total_chunks // 6)      # 17%
            }
    
    def _search_level(self, query_embedding: np.ndarray, index: faiss.Index, 
                     level: DocumentLevel, top_k: int) -> List[HierarchicalChunk]:
        """Search specific hierarchy level using FAISS"""
        if index.ntotal == 0:
            return []
        
        try:
            scores, indices = index.search(query_embedding.astype(np.float32), min(top_k, index.ntotal))
            
            chunks = []
            level_chunks = [chunk for chunk in self.hierarchical_chunks.values() if chunk.level == level]
            
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(level_chunks):
                    chunk = level_chunks[idx]
                    chunk.similarity_score = float(score)
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search {level.value} level: {e}")
            return []
    
    def _rerank_and_convert(self, chunks: List[HierarchicalChunk], max_chunks: int) -> List[Dict[str, Any]]:
        """Re-rank chunks and convert to legacy format for compatibility"""
        
        # Remove duplicates and sort by similarity
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            text_key = chunk.text[:100].lower().strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_chunks.append(chunk)
        
        # Sort by similarity score (descending)
        unique_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Convert to legacy format
        results = []
        for i, chunk in enumerate(unique_chunks[:max_chunks]):
            result = {
                'id': chunk.id,
                'text': chunk.text,
                'doc_id': chunk.doc_id,
                'similarity_score': chunk.similarity_score,
                'chunk_index': i,
                'word_start': 0,
                'word_end': chunk.word_count,
                'created_at': chunk.created_at,
                'document_info': self.document_metadata.get(chunk.doc_id, {}),
                'level': chunk.level.value,  # Additional hierarchical info
                'section_title': chunk.section_title,
                'summary': chunk.summary
            }
            results.append(result)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of hierarchical RAG system"""
        
        total_chunks = len(self.hierarchical_chunks)
        
        return {
            'total_documents': len(self.document_summaries),
            'total_chunks': total_chunks,
            'documents': list(self.document_metadata.values()),
            'hierarchical_structure': True,
            'embedding_model': self.embedding_model_name,
            'summarization_model': self.summarization_model,
            'storage_path': str(self.rag_base_path)
        }

# Compatibility functions for existing interface
def get_hierarchical_rag_pipeline(matter_id: str) -> HierarchicalRAGPipeline:
    """Get or create hierarchical RAG pipeline instance"""
    return HierarchicalRAGPipeline(matter_id) 