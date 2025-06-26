# advanced_vector_store.py

import re
import json
import spacy
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict, Counter

from config import logger

# Download spacy model if needed: python -m spacy download en_core_web_sm

@dataclass
class DocumentMetadata:
    """Rich metadata for documents"""
    document_id: str
    filename: str
    file_type: str
    upload_date: datetime
    file_size: int
    page_count: int
    
    # Legal-specific metadata
    document_type: str  # contract, agreement, memo, letter, etc.
    parties: List[str]
    legal_entities: List[str]
    key_dates: List[Dict[str, Any]]
    obligations: List[Dict[str, Any]]
    risks: List[str]
    jurisdiction: Optional[str]
    governing_law: Optional[str]
    
    # Content analysis
    language: str
    sentiment_score: float
    complexity_score: float
    confidentiality_level: str
    
    # Structural metadata
    sections: List[Dict[str, Any]]
    cross_references: List[str]
    citations: List[Dict[str, Any]]

@dataclass 
class ChunkMetadata:
    """Metadata for individual chunks"""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    
    # Hierarchical position
    section_title: Optional[str]
    subsection_title: Optional[str]
    paragraph_number: Optional[int]
    
    # Content classification
    content_type: str  # clause, definition, obligation, risk, etc.
    legal_concepts: List[str]
    entities_mentioned: List[str]
    dates_mentioned: List[str]
    
    # Semantic metadata
    main_topics: List[str]
    semantic_density: float
    reference_intensity: float  # How much it references other parts
    
    # Quality metrics
    completeness_score: float
    clarity_score: float

class LegalMetadataExtractor:
    """Extract sophisticated legal metadata from documents"""
    
    def __init__(self):
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Legal patterns
        self.legal_patterns = {
            'contract_types': [
                r'service\s+agreement', r'employment\s+contract', r'lease\s+agreement',
                r'purchase\s+agreement', r'license\s+agreement', r'partnership\s+agreement',
                r'confidentiality\s+agreement', r'non-disclosure\s+agreement', r'nda'
            ],
            'obligation_indicators': [
                r'shall\s+(?:not\s+)?(?:be\s+)?(?:required\s+to\s+)?', r'must\s+(?:not\s+)?',
                r'responsible\s+for', r'agrees?\s+to', r'undertakes?\s+to',
                r'covenant\s+(?:not\s+)?to', r'warrants?\s+that', r'represents?\s+that'
            ],
            'risk_indicators': [
                r'liability', r'damages', r'breach', r'default', r'penalty',
                r'indemnif(?:y|ication)', r'limitation\s+of\s+liability',
                r'force\s+majeure', r'disclaimer', r'at\s+(?:your|their)\s+own\s+risk'
            ],
            'date_patterns': [
                r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',  # MM/DD/YYYY
                r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
                r'\bwithin\s+\d+\s+(?:days?|weeks?|months?|years?)\b',
                r'\b(?:on\s+or\s+)?before\s+\d+[\/\-\.]\d+[\/\-\.]\d+\b'
            ]
        }
        
        # Legal entity patterns
        self.entity_suffixes = [
            'Inc\.?', 'LLC', 'Ltd\.?', 'Corp\.?', 'Corporation', 'Company', 'Co\.?',
            'LP', 'LLP', 'Partnership', 'Associates', 'Group', 'Holdings?'
        ]
    
    def extract_document_metadata(self, text: str, filename: str, file_size: int) -> DocumentMetadata:
        """Extract comprehensive metadata from a document"""
        
        # Basic document analysis
        doc_type = self._classify_document_type(text, filename)
        parties = self._extract_parties(text)
        legal_entities = self._extract_legal_entities(text)
        key_dates = self._extract_key_dates(text)
        obligations = self._extract_obligations(text)
        risks = self._extract_risks(text)
        
        # Legal jurisdiction analysis
        jurisdiction, governing_law = self._extract_jurisdiction_info(text)
        
        # Content analysis
        language = self._detect_language(text)
        sentiment = self._analyze_sentiment(text)
        complexity = self._calculate_complexity(text)
        confidentiality = self._assess_confidentiality_level(text)
        
        # Structural analysis
        sections = self._extract_sections(text)
        cross_refs = self._extract_cross_references(text)
        citations = self._extract_citations(text)
        
        return DocumentMetadata(
            document_id=self._generate_doc_id(filename),
            filename=filename,
            file_type=Path(filename).suffix.lower(),
            upload_date=datetime.now(),
            file_size=file_size,
            page_count=self._estimate_page_count(text),
            document_type=doc_type,
            parties=parties,
            legal_entities=legal_entities,
            key_dates=key_dates,
            obligations=obligations,
            risks=risks,
            jurisdiction=jurisdiction,
            governing_law=governing_law,
            language=language,
            sentiment_score=sentiment,
            complexity_score=complexity,
            confidentiality_level=confidentiality,
            sections=sections,
            cross_references=cross_refs,
            citations=citations
        )
    
    def extract_chunk_metadata(self, chunk_text: str, chunk_id: str, 
                             document_id: str, chunk_index: int,
                             start_char: int, end_char: int,
                             section_context: Dict[str, Any] = None) -> ChunkMetadata:
        """Extract metadata for individual chunks"""
        
        # Content classification
        content_type = self._classify_chunk_content(chunk_text)
        legal_concepts = self._extract_legal_concepts(chunk_text)
        entities = self._extract_entities_from_chunk(chunk_text)
        dates = self._extract_dates_from_chunk(chunk_text)
        
        # Semantic analysis
        topics = self._extract_main_topics(chunk_text)
        semantic_density = self._calculate_semantic_density(chunk_text)
        reference_intensity = self._calculate_reference_intensity(chunk_text)
        
        # Quality assessment
        completeness = self._assess_completeness(chunk_text)
        clarity = self._assess_clarity(chunk_text)
        
        # Section context
        section_title = section_context.get('section_title') if section_context else None
        subsection_title = section_context.get('subsection_title') if section_context else None
        paragraph_num = section_context.get('paragraph_number') if section_context else None
        
        return ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            section_title=section_title,
            subsection_title=subsection_title,
            paragraph_number=paragraph_num,
            content_type=content_type,
            legal_concepts=legal_concepts,
            entities_mentioned=entities,
            dates_mentioned=dates,
            main_topics=topics,
            semantic_density=semantic_density,
            reference_intensity=reference_intensity,
            completeness_score=completeness,
            clarity_score=clarity
        )
    
    def _classify_document_type(self, text: str, filename: str) -> str:
        """Classify the type of legal document"""
        
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Check filename first
        if any(term in filename_lower for term in ['contract', 'agreement']):
            return 'contract'
        elif any(term in filename_lower for term in ['memo', 'memorandum']):
            return 'memorandum'
        elif any(term in filename_lower for term in ['letter', 'correspondence']):
            return 'letter'
        
        # Check content patterns
        for pattern in self.legal_patterns['contract_types']:
            if re.search(pattern, text_lower):
                return 'contract'
        
        # More specific classifications
        if 'whereas' in text_lower and 'now therefore' in text_lower:
            return 'formal_agreement'
        elif 'attorney-client' in text_lower or 'privileged' in text_lower:
            return 'privileged_communication'
        elif re.search(r'court|tribunal|hearing|case\s+no', text_lower):
            return 'litigation_document'
        
        return 'general_legal_document'
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names from legal documents"""
        
        parties = []
        
        # Common party introduction patterns
        party_patterns = [
            r'between\s+([^,]+?)\s+(?:and|&)\s+([^,\n]+)',
            r'party\s+of\s+the\s+first\s+part[:\s]+([^,\n]+)',
            r'party\s+of\s+the\s+second\s+part[:\s]+([^,\n]+)',
            r'entered\s+into\s+by\s+and\s+between\s+([^,]+?)\s+and\s+([^,\n]+)'
        ]
        
        for pattern in party_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                for group in match.groups():
                    if group:
                        party = group.strip(' "\'')
                        if len(party) > 3 and party not in parties:
                            parties.append(party)
        
        return parties[:10]  # Limit to avoid noise
    
    def _extract_legal_entities(self, text: str) -> List[str]:
        """Extract legal entities (companies, organizations)"""
        
        entities = []
        
        # Pattern for legal entity suffixes
        suffix_pattern = '|'.join(self.entity_suffixes)
        entity_pattern = rf'\b([A-Z][A-Za-z\s&,.-]+?(?:{suffix_pattern}))\b'
        
        matches = re.finditer(entity_pattern, text)
        for match in matches:
            entity = match.group(1).strip()
            if len(entity) > 5 and entity not in entities:
                entities.append(entity)
        
        return entities[:20]  # Limit to avoid noise
    
    def _extract_key_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract important dates with context"""
        
        dates = []
        
        for pattern in self.legal_patterns['date_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group(0)
                
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                dates.append({
                    'date_text': date_text,
                    'context': context.strip(),
                    'position': match.start()
                })
        
        return dates[:15]  # Limit to most important dates

    def _extract_obligations(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal obligations and duties"""
        
        obligations = []
        
        for pattern in self.legal_patterns['obligation_indicators']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the full sentence containing the obligation
                start = max(0, text.rfind('.', 0, match.start()) + 1)
                end = text.find('.', match.end())
                if end == -1:
                    end = min(len(text), match.end() + 200)
                
                obligation_text = text[start:end].strip()
                
                if len(obligation_text) > 10:
                    obligations.append({
                        'text': obligation_text,
                        'type': self._classify_obligation_type(obligation_text),
                        'position': match.start()
                    })
        
        return obligations[:20]
    
    def _extract_risks(self, text: str) -> List[str]:
        """Extract risk-related content"""
        
        risks = []
        
        for pattern in self.legal_patterns['risk_indicators']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                
                if context not in risks:
                    risks.append(context)
        
        return risks[:15]

class AdvancedVectorStore:
    """
    Sophisticated vector store with hierarchical embeddings and rich metadata
    """
    
    def __init__(self, matter_id: str, embedding_model_name: str = "all-mpnet-base-v2"):
        self.matter_id = matter_id
        self.storage_path = Path(f"rag_storage/{matter_id}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding models - multiple for different purposes
        self.embedding_models = {
            'semantic': SentenceTransformer(embedding_model_name),
            'legal': SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1'),  # Better for Q&A
            'entity': SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')  # Good for entity matching
        }
        
        # Initialize metadata extractor
        self.metadata_extractor = LegalMetadataExtractor()
        
        # Vector stores for different embedding types
        self.vector_stores = {}
        self.metadata_store = {}
        self.document_metadata = {}
        
        # Load existing data
        self._load_existing_data()
    
    def add_document(self, text: str, filename: str, file_size: int) -> str:
        """Add a document with sophisticated processing"""
        
        logger.info(f"Processing document: {filename}")
        
        # Extract document-level metadata
        doc_metadata = self.metadata_extractor.extract_document_metadata(text, filename, file_size)
        doc_id = doc_metadata.document_id
        
        # Store document metadata
        self.document_metadata[doc_id] = doc_metadata
        
        # Hierarchical chunking with overlap
        chunks = self._create_hierarchical_chunks(text, doc_metadata)
        
        # Process each chunk with multiple embedding strategies
        for chunk_info in chunks:
            chunk_id = f"{doc_id}_{chunk_info['index']}"
            chunk_text = chunk_info['text']
            
            # Extract chunk metadata
            chunk_metadata = self.metadata_extractor.extract_chunk_metadata(
                chunk_text, chunk_id, doc_id, chunk_info['index'],
                chunk_info['start_char'], chunk_info['end_char'],
                chunk_info.get('section_context')
            )
            
            # Generate multiple types of embeddings
            embeddings = {}
            for emb_type, model in self.embedding_models.items():
                embeddings[emb_type] = model.encode([chunk_text])[0]
            
            # Store in appropriate vector stores
            for emb_type, embedding in embeddings.items():
                if emb_type not in self.vector_stores:
                    self.vector_stores[emb_type] = faiss.IndexFlatIP(len(embedding))
                
                # Normalize embedding for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                self.vector_stores[emb_type].add(embedding.reshape(1, -1))
            
            # Store metadata
            self.metadata_store[chunk_id] = {
                'chunk_metadata': chunk_metadata,
                'chunk_text': chunk_text,
                'document_metadata': doc_metadata,
                'embeddings_stored': list(embeddings.keys())
            }
        
        # Save to disk
        self._save_data()
        
        logger.info(f"Document processed: {len(chunks)} chunks created")
        return doc_id
    
    def search_advanced(self, query: str, top_k: int = 10, 
                       embedding_type: str = 'semantic',
                       metadata_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Advanced search with metadata filtering"""
        
        if embedding_type not in self.vector_stores:
            logger.warning(f"Embedding type {embedding_type} not available")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_models[embedding_type].encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search vector store
        scores, indices = self.vector_stores[embedding_type].search(
            query_embedding.reshape(1, -1), 
            min(top_k * 3, self.vector_stores[embedding_type].ntotal)  # Get more for filtering
        )
        
        # Apply metadata filters and rank results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata_store):
                chunk_id = list(self.metadata_store.keys())[idx]
                chunk_data = self.metadata_store[chunk_id]
                
                # Apply metadata filters
                if metadata_filters and not self._matches_filters(chunk_data, metadata_filters):
                    continue
                
                # Calculate enhanced relevance score
                relevance_score = self._calculate_enhanced_relevance(
                    query, chunk_data, score, embedding_type
                )
                
                results.append({
                    'chunk_id': chunk_id,
                    'text': chunk_data['chunk_text'],
                    'similarity_score': float(score),
                    'relevance_score': relevance_score,
                    'chunk_metadata': asdict(chunk_data['chunk_metadata']),
                    'document_info': {
                        'filename': chunk_data['document_metadata'].filename,
                        'document_type': chunk_data['document_metadata'].document_type,
                        'parties': chunk_data['document_metadata'].parties,
                    }
                })
        
        # Sort by enhanced relevance and return top results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]

# Global instances per matter
_vector_store_instances: Dict[str, AdvancedVectorStore] = {}

def get_advanced_vector_store(matter_id: str) -> AdvancedVectorStore:
    """Get or create advanced vector store instance for a matter"""
    if matter_id not in _vector_store_instances:
        _vector_store_instances[matter_id] = AdvancedVectorStore(matter_id)
    return _vector_store_instances[matter_id] 