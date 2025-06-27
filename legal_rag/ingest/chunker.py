"""
Legal Semantic Chunker
======================

Semantic sentence splitter with recursive merge to 400-token target.
Enhanced metadata extraction for legal documents including jurisdiction, dates, versions.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

# Optional dependencies with fallbacks
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    English = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    dateparser = None

from legal_rag import logger, DEFAULT_CONFIG

class LegalSemanticChunker:
    """
    Advanced semantic chunker for legal documents.
    
    Features:
    - Semantic sentence segmentation with spaCy
    - Recursive merge to 400-token target with 80-token overlap
    - Legal metadata extraction (jurisdiction, dates, citations)
    - Document structure awareness (sections, pages)
    - Backward compatibility with existing chunking
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        enable_semantic: bool = True,
        enable_metadata: bool = True
    ):
        """
        Initialize legal semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (default: 400)
            chunk_overlap: Overlap between chunks in tokens (default: 80)
            enable_semantic: Use semantic sentence splitting
            enable_metadata: Extract legal metadata
        """
        self.config = DEFAULT_CONFIG.copy()
        self.chunk_size = chunk_size or self.config["chunk_size"]
        self.chunk_overlap = chunk_overlap or self.config["chunk_overlap"]
        self.enable_semantic = enable_semantic and SPACY_AVAILABLE
        self.enable_metadata = enable_metadata
        
        # Initialize spaCy model for semantic processing
        self.nlp = None
        if self.enable_semantic:
            self._initialize_spacy()
        
        # Initialize fallback splitter
        self.fallback_splitter = None
        if LANGCHAIN_AVAILABLE:
            self.fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 4,  # Approximate token-to-char ratio
                chunk_overlap=self.chunk_overlap * 4,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
            )
        
        # Legal patterns for metadata extraction
        self._compile_legal_patterns()
        
        logger.info(f"Initialized LegalSemanticChunker (semantic: {self.enable_semantic})")
    
    def _initialize_spacy(self):
        """Initialize spaCy model for semantic processing."""
        try:
            # Try to load full English model first
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ Loaded spaCy en_core_web_sm model")
            except OSError:
                # Fallback to basic English tokenizer
                self.nlp = English()
                self.nlp.add_pipe('sentencizer')
                logger.info("📦 Using spaCy basic English sentencizer")
                
        except Exception as e:
            logger.warning(f"spaCy initialization failed: {e}")
            self.enable_semantic = False
    
    def _compile_legal_patterns(self):
        """Compile regex patterns for legal metadata extraction."""
        # UK jurisdiction patterns
        self.jurisdiction_patterns = [
            r'\b(?:High Court|Court of Appeal|Supreme Court|Crown Court|County Court)\b',
            r'\b(?:England and Wales|Scotland|Northern Ireland)\b',
            r'\b(?:QBD|ChD|KB|QB|CA|SC|EWCA|EWHC|UKSC)\b'
        ]
        
        # Case citation patterns
        self.citation_patterns = [
            r'\[\d{4}\]\s+(?:EWCA|EWHC|UKSC|All ER|WLR|QB|Ch)\s+\d+',
            r'\(\d{4}\)\s+\d+\s+(?:All ER|WLR|QB|Ch)',
            r'\b(?:CPR|r\.|rule)\s+\d+(?:\.\d+)*\b'
        ]
        
        # Date patterns (various legal formats)
        self.date_patterns = [
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        
        # Compile patterns
        self.compiled_patterns = {
            'jurisdiction': [re.compile(p, re.IGNORECASE) for p in self.jurisdiction_patterns],
            'citation': [re.compile(p, re.IGNORECASE) for p in self.citation_patterns],
            'date': [re.compile(p, re.IGNORECASE) for p in self.date_patterns]
        }
    
    def extract_legal_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract legal metadata from document text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            'jurisdictions': [],
            'citations': [],
            'dates': [],
            'case_names': [],
            'sections': [],
            'pages': []
        }
        
        if not self.enable_metadata:
            return metadata
        
        try:
            # Extract jurisdictions
            for pattern in self.compiled_patterns['jurisdiction']:
                matches = pattern.findall(text)
                metadata['jurisdictions'].extend(matches)
            
            # Extract citations
            for pattern in self.compiled_patterns['citation']:
                matches = pattern.findall(text)
                metadata['citations'].extend(matches)
            
            # Extract dates
            for pattern in self.compiled_patterns['date']:
                matches = pattern.findall(text)
                metadata['dates'].extend(matches)
            
            # Parse dates if dateparser available
            if DATEPARSER_AVAILABLE and dateparser:
                parsed_dates = []
                for date_str in metadata['dates']:
                    parsed = dateparser.parse(date_str)
                    if parsed:
                        parsed_dates.append(parsed.isoformat())
                metadata['parsed_dates'] = parsed_dates
            
            # Extract case names (basic pattern)
            case_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            case_matches = re.findall(case_pattern, text)
            metadata['case_names'] = [f"{claimant} v {defendant}" for claimant, defendant in case_matches]
            
            # Extract section markers
            section_pattern = r'\b(?:Section|Para|Paragraph|§)\s+(\d+(?:\.\d+)*)\b'
            section_matches = re.findall(section_pattern, text, re.IGNORECASE)
            metadata['sections'] = section_matches
            
            # Extract page references
            page_pattern = r'\bpage\s+(\d+)\b'
            page_matches = re.findall(page_pattern, text, re.IGNORECASE)
            metadata['pages'] = [int(p) for p in page_matches]
            
            # Deduplicate lists
            for key in ['jurisdictions', 'citations', 'dates', 'case_names', 'sections']:
                metadata[key] = list(set(metadata[key]))
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def semantic_sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy for legal punctuation fidelity.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not self.enable_semantic or not self.nlp:
            # Fallback to simple splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return sentences
            
        except Exception as e:
            logger.warning(f"Semantic sentence splitting failed: {e}")
            # Fallback to regex
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def merge_sentences_to_chunks(
        self,
        sentences: List[str],
        target_size: int = None,
        overlap_size: int = None
    ) -> List[str]:
        """
        Recursively merge sentences to target chunk size with overlap.
        
        Args:
            sentences: List of sentences
            target_size: Target chunk size in tokens
            overlap_size: Overlap size in tokens
            
        Returns:
            List of merged chunks
        """
        if not sentences:
            return []
        
        target_size = target_size or self.chunk_size
        overlap_size = overlap_size or self.chunk_overlap
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        overlap_text = ""
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence would exceed target size
            if current_tokens + sentence_tokens > target_size and current_chunk:
                # Finalize current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, overlap_size)
                current_chunk = overlap_sentences + " " + sentence if overlap_sentences else sentence
                current_tokens = self.estimate_tokens(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_sentences(self, text: str, overlap_tokens: int) -> str:
        """Get the last few sentences for overlap."""
        sentences = self.semantic_sentence_split(text)
        if not sentences:
            return ""
        
        overlap_text = ""
        current_tokens = 0
        
        # Work backwards from the end
        for sentence in reversed(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            if current_tokens + sentence_tokens > overlap_tokens:
                break
            overlap_text = sentence + " " + overlap_text
            current_tokens += sentence_tokens
        
        return overlap_text.strip()
    
    def chunk_document(
        self,
        text: str,
        doc_id: str = None,
        doc_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Main chunking method that combines semantic splitting with metadata extraction.
        
        Args:
            text: Document text
            doc_id: Document identifier
            doc_metadata: Additional document metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text.strip():
            return []
        
        doc_id = doc_id or hashlib.sha256(text.encode()).hexdigest()[:16]
        doc_metadata = doc_metadata or {}
        
        try:
            # Extract document-level metadata
            document_metadata = self.extract_legal_metadata(text)
            
            # Semantic sentence splitting
            if self.enable_semantic:
                sentences = self.semantic_sentence_split(text)
                chunks_text = self.merge_sentences_to_chunks(sentences)
            else:
                # Fallback to existing chunking method
                if self.fallback_splitter:
                    chunks_text = self.fallback_splitter.split_text(text)
                else:
                    # Simple word-based chunking
                    words = text.split()
                    chunks_text = []
                    for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                        chunk_words = words[i:i + self.chunk_size]
                        chunks_text.append(' '.join(chunk_words))
            
            # Create chunk objects with metadata
            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                chunk_metadata = self.extract_legal_metadata(chunk_text)
                
                chunk = {
                    'id': f"{doc_id}_chunk_{i}",
                    'text': chunk_text,
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'token_count': self.estimate_tokens(chunk_text),
                    'created_at': datetime.now().isoformat(),
                    
                    # Legal metadata
                    'metadata': {
                        **doc_metadata,
                        'chunk_jurisdictions': chunk_metadata['jurisdictions'],
                        'chunk_citations': chunk_metadata['citations'],
                        'chunk_dates': chunk_metadata['dates'],
                        'chunk_case_names': chunk_metadata['case_names'],
                        'chunk_sections': chunk_metadata['sections'],
                        'chunk_pages': chunk_metadata['pages']
                    },
                    
                    # Document-level metadata
                    'document_metadata': document_metadata,
                    
                    # Processing info
                    'chunking_method': 'semantic' if self.enable_semantic else 'fallback',
                    'chunk_size_target': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }
                
                chunks.append(chunk)
            
            logger.info(f"✅ Created {len(chunks)} chunks from document {doc_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed for document {doc_id}: {e}")
            return []
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get chunking configuration and capabilities."""
        return {
            'config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'semantic_enabled': self.enable_semantic,
                'metadata_enabled': self.enable_metadata
            },
            'capabilities': {
                'spacy_available': SPACY_AVAILABLE,
                'langchain_available': LANGCHAIN_AVAILABLE,
                'dateparser_available': DATEPARSER_AVAILABLE,
                'nlp_model': str(self.nlp) if self.nlp else None
            }
        }


# Simple function for backward compatibility
def chunk_text_legacy(text: str, chunk_size: int = 400, chunk_overlap: int = 80) -> List[str]:
    """Legacy chunking function for backward compatibility."""
    chunker = LegalSemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_document(text)
    return [chunk['text'] for chunk in chunks] 