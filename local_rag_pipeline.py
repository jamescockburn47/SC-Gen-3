# local_rag_pipeline.py

import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import asyncio

# Async HTTP import with error handling
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

# Vector database and embedding imports
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    SentenceTransformer = None

# BGE Models for SOTA performance
try:
    from FlagEmbedding import FlagModel, FlagReranker
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    FlagModel = None
    FlagReranker = None

# Document processing
import io
from app_utils import extract_text_from_uploaded_file
from config import logger, APP_BASE_PATH

# Import hardware optimizer
try:
    from rag_config_optimizer import rag_optimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    rag_optimizer = None

# Import MCP server for protocol enforcement
try:
    from mcp_rag_server import mcp_rag_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    mcp_rag_server = None

def load_strategic_protocols() -> str:
    """Load strategic protocols for RAG prompt generation"""
    try:
        protocols_path = APP_BASE_PATH / "strategic_protocols.txt"
        if protocols_path.exists():
            with open(protocols_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to load strategic protocols: {e}")
    
    # Fallback protocols
    return """
## Strategic Counsel Protocols (Fallback)

### M.A — Hallucination Containment & Citation Discipline
- No authority may be cited unless uploaded by user, verifiably public, or clearly flagged as [UNVERIFIED]
- Fictional case names, statutes, or legal doctrines are strictly prohibited
- Citations must include court, date, and jurisdiction where known

### M.B — Memory Integrity & Role Inference Safeguards  
- Role inference must derive only from documents or user instructions
- All assumed context must be tagged as [HYPOTHETICAL INFERENCE]
- Legal roles must be derived from document content, not AI guesswork

### M.D — Adversarial Reasoning & Legal Discipline
- Prioritise doctrinal logic over persuasive or narrative framing
- Flag strategic weaknesses or gaps in authority
- Trace every argument to a grounded source

### M.G — Jurisdictional and User Alignment
- Default legal reasoning to UK commercial and competition litigation
- Outputs must assume a senior solicitor or barrister audience
"""

def safe_faiss_normalize(embeddings):
    """Safely normalize embeddings with proper type checking"""
    if faiss is not None and np is not None and embeddings is not None:
        faiss.normalize_L2(embeddings.astype(np.float32))

def create_general_legal_prompt(query: str, context: str, protocols: str) -> str:
    """Create a general legal RAG prompt based on strategic protocols"""
    
    return f"""You are a legal AI assistant analyzing documents in accordance with Strategic Counsel Protocols.

STRATEGIC PROTOCOLS (MANDATORY COMPLIANCE):
{protocols}

DOCUMENT ANALYSIS CONTEXT:
The documents provided may include any type of legal text: court pleadings, contracts, correspondence, statutes, case law, legal opinions, compliance documents, corporate records, or other legal materials. Do not assume document type - analyze what is actually present.

QUERY: {query}

DOCUMENT CONTENT:
{context}

CRITICAL CITATION REQUIREMENTS:
- You MUST cite sources using EXACTLY this format: [Source 1], [Source 2], etc.
- Every factual statement MUST include a source citation
- If information spans multiple sources, cite all relevant sources
- When quoting directly, use the exact format: "quote text" [Source X]

ANALYSIS REQUIREMENTS:
1. **Protocol M.A Compliance**: Only cite content from provided documents. Mark any external references as [UNVERIFIED].
2. **Protocol M.B Compliance**: Base role identification solely on document content. Tag assumptions as [HYPOTHETICAL INFERENCE].
3. **Protocol M.D Compliance**: Focus on doctrinal analysis over narrative. Identify gaps in authority.
4. **Protocol M.G Compliance**: Apply UK legal reasoning where applicable.

RESPONSE FORMAT:
- Start with "Based on the provided documents:"
- Provide direct answers based solely on document content
- Use [Source X] citations for ALL factual statements
- Clearly state when information is not available in documents
- Flag any uncertainties or inferential reasoning
- Structure for senior legal practitioner audience

EXAMPLE RESPONSE FORMAT:
"Based on the provided documents: The case number is KB-2023-000930 [Source 1]. The claimant is identified as John Smith [Source 2]. The main legal claims include breach of contract and negligence [Source 1, Source 2]."

ANALYSIS:"""

class LocalRAGPipeline:
    """
    Enhanced Local RAG Pipeline with BGE (BAAI) embeddings for superior performance.
    
    Features:
    - BGE-base-en-v1.5 embeddings (SOTA on legal text)
    - BGE reranker for 20-30 point MRR uplift  
    - Backward compatibility with sentence-transformers
    - Hardware optimization (GPU/CPU fallback)
    """
    
    def __init__(self, 
                 matter_id: str,
                 embedding_model: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 ollama_base_url: str = "http://localhost:11434",
                 enable_reranking: bool = True):
        
        self.matter_id = matter_id
        self.ollama_base_url = ollama_base_url
        self.enable_reranking = enable_reranking and BGE_AVAILABLE
        
        # BGE models for SOTA performance (fallback to existing)
        if embedding_model:
            self.embedding_model_name = embedding_model
        elif BGE_AVAILABLE:
            self.embedding_model_name = "BAAI/bge-base-en-v1.5"  # SOTA default
        else:
            self.embedding_model_name = "all-mpnet-base-v2"  # Fallback
        
        self.reranker_model_name = "BAAI/bge-reranker-base"
        
        # Configuration with optimized defaults
        self.chunk_size = chunk_size or 400  # Optimized for BGE
        self.chunk_overlap = chunk_overlap or 80
        self.batch_size = 16  # Efficient batch processing
        self.max_docs_per_matter = 100
        
        # Model instances
        self.embedding_model = None
        self.reranker_model = None
        self.is_using_bge = False
        
        # Performance tracking
        self.performance_stats = {
            'embedding_time': [],
            'rerank_time': [],
            'search_improvements': []
        }
        
        # Load strategic protocols for consistent prompting
        self.strategic_protocols = load_strategic_protocols()
        
        # Initialize paths
        self.rag_base_path = APP_BASE_PATH / "rag_storage" / matter_id
        self.vector_db_path = self.rag_base_path / "vector_db"
        self.documents_path = self.rag_base_path / "documents"
        self.metadata_path = self.rag_base_path / "metadata.json"
        
        # Create directories
        for path in [self.rag_base_path, self.vector_db_path, self.documents_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.vector_index = None
        self.document_metadata = {}
        self.chunk_metadata = []
        
        # Track available Ollama models
        self.available_models = []
        
        # Load existing data
        self._load_metadata()
        self._initialize_embedding_model()
        self._load_vector_index()
    
    def _initialize_embedding_model(self) -> bool:
        """Initialize BGE embedding model with fallback to sentence-transformers"""
        
        # Try BGE models first (SOTA performance)
        if BGE_AVAILABLE and FlagModel is not None and self.embedding_model_name.startswith("BAAI/"):
            try:
                logger.info(f"🚀 Initializing SOTA BGE embedding model: {self.embedding_model_name}")
                self.embedding_model = FlagModel(
                    self.embedding_model_name,
                    query_instruction_for_retrieval="Represent this query for searching legal documents:",
                    use_fp16=True  # Memory efficiency
                )
                
                # Initialize reranker if enabled
                if self.enable_reranking and FlagReranker is not None:
                    logger.info(f"🎯 Initializing BGE reranker: {self.reranker_model_name}")
                    self.reranker_model = FlagReranker(
                        self.reranker_model_name,
                        use_fp16=True
                    )
                    logger.info("✅ BGE reranker initialized for 20-30 point MRR uplift")
                
                self.is_using_bge = True
                logger.info("✅ SOTA BGE models initialized successfully")
                return True
                
            except Exception as e:
                logger.warning(f"BGE model initialization failed, falling back: {e}")
        
        # Fallback to sentence-transformers for compatibility
        if EMBEDDING_AVAILABLE and SentenceTransformer is not None:
            try:
                logger.info(f"📦 Initializing fallback embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                
                # Enable GPU if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.embedding_model = self.embedding_model.cuda()
                        logger.info("GPU acceleration enabled for embedding model")
                except ImportError:
                    logger.info("GPU acceleration not available (PyTorch not found)")
                
                self.is_using_bge = False
                logger.info(f"✅ Fallback embedding model initialized: {self.embedding_model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize embedding model {self.embedding_model_name}: {e}")
        
        logger.error("❌ Failed to initialize any embedding model")
        return False
    
    def _load_metadata(self):
        """Load existing document and chunk metadata"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.document_metadata = data.get('documents', {})
                    self.chunk_metadata = data.get('chunks', [])
                logger.info(f"Loaded metadata for {len(self.document_metadata)} documents and {len(self.chunk_metadata)} chunks")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save document and chunk metadata"""
        try:
            data = {
                'documents': self.document_metadata,
                'chunks': self.chunk_metadata,
                'configuration': {
                    'embedding_model': self.embedding_model_name,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'max_docs_per_matter': self.max_docs_per_matter
                }
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _load_vector_index(self):
        """Load existing FAISS vector index with hardware optimizations"""
        if not FAISS_AVAILABLE or faiss is None:
            logger.error("FAISS not available. Cannot load vector index.")
            return
        
        index_path = self.vector_db_path / "faiss_index.bin"
        if index_path.exists():
            try:
                self.vector_index = faiss.read_index(str(index_path))
                
                # Apply hardware optimizations if available
                if OPTIMIZER_AVAILABLE and rag_optimizer is not None:
                    perf_config = rag_optimizer.performance_config
                    if perf_config.get("vector_index_type") == "IVF" and hasattr(faiss, 'IndexIVFFlat'):
                        # Could migrate to IVF index for better performance, but keep existing for compatibility
                        pass
                
                logger.info(f"Loaded FAISS index with {self.vector_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
    
    def _save_vector_index(self):
        """Save FAISS vector index"""
        if not FAISS_AVAILABLE or faiss is None or self.vector_index is None:
            return
        
        try:
            index_path = self.vector_db_path / "faiss_index.bin"
            faiss.write_index(self.vector_index, str(index_path))
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with optimized sizing"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'doc_id': doc_id,
                'chunk_index': len(chunks),
                'word_start': i,
                'word_end': min(i + self.chunk_size, len(words)),
                'created_at': datetime.now().isoformat()
            })
        
        return chunks
    
    def add_document(self, file_obj: io.BytesIO, filename: str, 
                    ocr_preference: str = "aws") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Add a document to the RAG pipeline with hardware-optimized processing
        
        Returns: (success, message, document_info)
        """
        if not self.embedding_model:
            return False, "Embedding model not available", {}
        
        # Check document limits
        if len(self.document_metadata) >= self.max_docs_per_matter:
            return False, f"Document limit reached: {len(self.document_metadata)}/{self.max_docs_per_matter}", {}
        
        # Extract text from document
        text_content, error_message = extract_text_from_uploaded_file(
            file_obj, filename, ocr_preference
        )
        
        if not text_content:
            return False, f"Failed to extract text: {error_message}", {}
        
        # Generate document ID
        doc_hash = hashlib.sha256(text_content.encode()).hexdigest()[:16]
        doc_id = f"{filename}_{doc_hash}"
        
        # Check if document already exists
        if doc_id in self.document_metadata:
            return False, "Document already exists in the database", {}
        
        try:
            # Chunk the document
            chunks = self._chunk_text(text_content, doc_id)
            
            # Generate embeddings for chunks with batching for performance
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            # Process embeddings in batches for better performance
            all_embeddings = []
            for i in range(0, len(chunk_texts), self.batch_size):
                batch_texts = chunk_texts[i:i + self.batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts)
                all_embeddings.append(batch_embeddings)
            
            if np is not None:
                embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            else:
                return False, "NumPy not available for embedding processing", {}
            
            # Initialize or update FAISS index
            if self.vector_index is None and faiss is not None:
                dimension = embeddings.shape[1]
                
                # Use optimized index type if available
                if OPTIMIZER_AVAILABLE and rag_optimizer is not None and rag_optimizer.performance_config.get("vector_index_type") == "IVF":
                    # For large collections, IVF is better - but adjust nlist based on data size
                    nlist = min(100, max(1, len(embeddings) // 10))  # Ensure nlist <= number of training points
                    if nlist < 1:
                        nlist = 1
                    quantizer = faiss.IndexFlatIP(dimension)
                    self.vector_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                    if len(embeddings) >= nlist:  # Only train if we have enough data
                        self.vector_index.train(embeddings.astype(np.float32))
                else:
                    # Standard flat index for smaller collections
                    self.vector_index = faiss.IndexFlatIP(dimension)
                
                # Normalize embeddings for cosine similarity
                if np is not None:
                    embeddings_normalized = embeddings.astype(np.float32)
                    safe_faiss_normalize(embeddings_normalized)
                    embeddings = embeddings_normalized
            elif faiss is not None and np is not None:
                # Normalize new embeddings
                faiss.normalize_L2(embeddings.astype(np.float32))
            
            # Add to vector index
            if self.vector_index is not None:
                start_index = self.vector_index.ntotal
                self.vector_index.add(embeddings.astype(np.float32))
                
                # Update chunk metadata with vector indices
                for i, chunk in enumerate(chunks):
                    chunk['vector_index'] = start_index + i
                    self.chunk_metadata.append(chunk)
            
            # Store document metadata
            doc_info = {
                'id': doc_id,
                'filename': filename,
                'text_length': len(text_content),
                'chunk_count': len(chunks),
                'created_at': datetime.now().isoformat(),
                'file_hash': doc_hash,
                'embedding_model': self.embedding_model_name,
                'chunk_size': self.chunk_size
            }
            self.document_metadata[doc_id] = doc_info
            
            # Save to disk
            doc_path = self.documents_path / f"{doc_id}.txt"
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            self._save_metadata()
            self._save_vector_index()
            
            logger.info(f"Successfully added document {filename} with {len(chunks)} chunks")
            return True, f"Successfully processed {len(chunks)} chunks", doc_info
            
        except Exception as e:
            logger.error(f"Failed to add document {filename}: {e}")
            return False, f"Processing error: {str(e)}", {}
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced search with BGE reranking for superior relevance.
        
        Returns: List of chunks with similarity scores and metadata
        """
        if not self.embedding_model or not self.vector_index or faiss is None or np is None:
            return []
        
        try:
            start_time = datetime.now()
            
            # Step 1: Initial vector search (get more candidates for reranking)
            initial_k = top_k * 3 if self.enable_reranking and self.reranker_model else top_k
            
            # Embed the query using BGE or fallback
            if self.is_using_bge and hasattr(self.embedding_model, 'encode_queries'):
                query_embedding = self.embedding_model.encode_queries([query])
            else:
                query_embedding = self.embedding_model.encode([query])
            
            # Convert to numpy array safely
            if hasattr(query_embedding, 'cpu') and hasattr(query_embedding, 'numpy'):
                # PyTorch tensor
                query_embedding = query_embedding.cpu().numpy()
            elif hasattr(query_embedding, 'numpy') and callable(getattr(query_embedding, 'numpy', None)):
                # TensorFlow tensor or similar
                query_embedding = query_embedding.numpy()
            elif not isinstance(query_embedding, np.ndarray):
                # Convert to numpy if it's not already
                query_embedding = np.array(query_embedding)
            
            # Ensure it's the right type and shape
            query_embedding = np.asarray(query_embedding, dtype=np.float32)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            safe_faiss_normalize(query_embedding)
            
            # Search vector index
            scores, indices = self.vector_index.search(query_embedding, initial_k)
            
            # Retrieve initial chunk metadata
            initial_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    chunk = self.chunk_metadata[idx].copy()
                    chunk['similarity_score'] = float(score)
                    chunk['initial_rank'] = len(initial_results) + 1
                    
                    # Add document metadata
                    doc_id = chunk['doc_id']
                    if doc_id in self.document_metadata:
                        chunk['document_info'] = self.document_metadata[doc_id]
                    
                    initial_results.append(chunk)
            
            # Step 2: BGE Reranking for improved relevance (SOTA feature)
            if self.enable_reranking and self.reranker_model and initial_results:
                rerank_start = datetime.now()
                
                # Prepare query-passage pairs for reranking
                pairs = [(query, chunk['text']) for chunk in initial_results]
                
                # Get rerank scores from BGE reranker
                rerank_scores = self.reranker_model.compute_score(pairs, normalize=True)
                
                # Convert to list if single score
                if not isinstance(rerank_scores, list):
                    rerank_scores = [rerank_scores]
                
                # Update results with rerank scores and resort
                for i, (chunk, rerank_score) in enumerate(zip(initial_results, rerank_scores)):
                    chunk['rerank_score'] = float(rerank_score)
                    chunk['score_improvement'] = float(rerank_score) - chunk['similarity_score']
                
                # Sort by rerank score (higher is better)
                initial_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                # Add final ranks
                for i, chunk in enumerate(initial_results):
                    chunk['final_rank'] = i + 1
                    chunk['rank_improvement'] = chunk['initial_rank'] - chunk['final_rank']
                
                # Take top_k after reranking
                results = initial_results[:top_k]
                
                rerank_time = (datetime.now() - rerank_start).total_seconds()
                self.performance_stats['rerank_time'].append(rerank_time)
                
                logger.info(f"🎯 BGE reranking completed in {rerank_time:.3f}s")
                
                # Track performance improvement safely
                if results:
                    improvements = [r.get('score_improvement', 0.0) for r in results]
                    if improvements:
                        avg_improvement = float(np.mean(improvements))
                        self.performance_stats['search_improvements'].append(avg_improvement)
                
            else:
                # No reranking - use initial results
                results = initial_results[:top_k]
                for i, chunk in enumerate(results):
                    chunk['final_rank'] = i + 1
                    chunk['rerank_score'] = chunk['similarity_score']  # Same as similarity
            
            total_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats['embedding_time'].append(total_time)
            
            # Add search metadata
            for chunk in results:
                chunk['search_metadata'] = {
                    'using_bge': self.is_using_bge,
                    'reranking_applied': self.enable_reranking and self.reranker_model is not None,
                    'embedding_model': self.embedding_model_name,
                    'search_time': total_time
                }
            
            logger.info(f"🔍 Search completed in {total_time:.3f}s ({'BGE' if self.is_using_bge else 'Standard'} embeddings)")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return []
    
    def search_documents_filtered(self, query: str, top_k: int = 5, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks from specific documents only
        
        Args:
            query: Search query
            top_k: Maximum number of results
            document_ids: List of document IDs to search within
        
        Returns: List of chunks with similarity scores and metadata from selected documents
        """
        if not self.embedding_model or not self.vector_index or faiss is None or np is None:
            return []
        
        if not document_ids:
            # If no specific documents, use regular search
            return self.search_documents(query, top_k)
        
        try:
            # First get all chunks from the specified documents
            filtered_chunk_indices = []
            for i, chunk in enumerate(self.chunk_metadata):
                if chunk['doc_id'] in document_ids:
                    filtered_chunk_indices.append(i)
            
            if not filtered_chunk_indices:
                logger.info(f"No chunks found for selected documents: {document_ids}")
                return []
            
            # Embed the query
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype(np.float32)
            safe_faiss_normalize(query_embedding)
            
            # Search with a larger top_k to get more candidates
            search_k = min(len(self.chunk_metadata), top_k * 5)  # Search more broadly
            scores, indices = self.vector_index.search(query_embedding.astype(np.float32), search_k)
            
            # Filter results to only include chunks from selected documents
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx in filtered_chunk_indices and len(results) < top_k:
                    chunk = self.chunk_metadata[idx].copy()
                    chunk['similarity_score'] = float(score)
                    
                    # Add document metadata
                    doc_id = chunk['doc_id']
                    if doc_id in self.document_metadata:
                        chunk['document_info'] = self.document_metadata[doc_id]
                    
                    results.append(chunk)
                    
                    if len(results) >= top_k:
                        break
            
            logger.info(f"Filtered search returned {len(results)} chunks from {len(document_ids)} selected documents")
            return results
            
        except Exception as e:
            logger.error(f"Filtered search failed: {e}")
            return []
    
    async def query_ollama_models(self) -> List[Dict[str, Any]]:
        """Get available models from Ollama with hardware-specific metadata"""
        if not AIOHTTP_AVAILABLE or aiohttp is None:
            logger.error("aiohttp not available. Cannot query Ollama models.")
            return []
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        
                        # Enhance model data with hardware-specific information
                        enhanced_models = []
                        for model in models:
                            model_name = model['name']
                            enhanced_model = model.copy()
                            
                            # Add hardware optimization data if available
                            if OPTIMIZER_AVAILABLE and rag_optimizer is not None and model_name in rag_optimizer.available_models:
                                model_config = rag_optimizer.available_models[model_name]
                                enhanced_model.update({
                                    'recommended_use': model_config.recommended_use,
                                    'speed_rating': model_config.speed,
                                    'quality_rating': model_config.quality,
                                    'optimal_temperature': model_config.optimal_temperature,
                                    'max_context_chunks': model_config.max_context_chunks,
                                    'hardware_utilization': rag_optimizer.get_hardware_utilization_estimate(model_name)
                                })
                            
                            enhanced_models.append(enhanced_model)
                        
                        self.available_models = enhanced_models
                        return enhanced_models
                    else:
                        logger.error(f"Failed to query Ollama models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error querying Ollama models: {e}")
            return []
    
    async def generate_rag_answer(self, 
                                query: str, 
                                model_name: str,
                                max_context_chunks: Optional[int] = None,
                                temperature: Optional[float] = None,
                                enforce_protocols: bool = True) -> Dict[str, Any]:
        """
        Generate RAG answer with MCP protocol enforcement
        
        Returns: Dictionary with answer, sources, and metadata
        """
        if not AIOHTTP_AVAILABLE or aiohttp is None:
            return {
                'answer': "aiohttp not available. Cannot generate responses.",
                'sources': [],
                'query': query,
                'model_used': model_name,
                'context_chunks': 0,
                'error': "aiohttp dependency missing"
            }
        
        # Step 1: MCP Query Validation (if available and requested)
        if enforce_protocols and MCP_AVAILABLE and mcp_rag_server is not None:
            is_valid, validation_msg, validation_metadata = mcp_rag_server.validate_query(
                self.matter_id, query
            )
            if not is_valid:
                return {
                    'answer': f"Query validation failed: {validation_msg}",
                    'sources': [],
                    'query': query,
                    'model_used': model_name,
                    'context_chunks': 0,
                    'error': f"MCP validation error: {validation_msg}",
                    'protocol_compliance': False
                }
        
        # Use optimized settings if available
        if OPTIMIZER_AVAILABLE and rag_optimizer is not None and model_name in rag_optimizer.available_models:
            optimal_settings = rag_optimizer.get_optimal_settings(model_name)
            max_context_chunks = max_context_chunks or optimal_settings['max_context_chunks']
            temperature = temperature or optimal_settings['temperature']
        else:
            max_context_chunks = max_context_chunks or 5
            temperature = temperature or 0.1
        
        # Retrieve relevant chunks
        relevant_chunks = self.search_documents(query, top_k=max_context_chunks)
        
        if not relevant_chunks:
            return {
                'answer': "No relevant documents found for your query.",
                'sources': [],
                'query': query,
                'model_used': model_name,
                'context_chunks': 0
            }
        
        # Build context from chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"[Source {i+1}] {chunk['text']}")
            sources.append({
                'chunk_id': chunk['id'],
                'document': chunk.get('document_info', {}).get('filename', 'Unknown'),
                'similarity_score': chunk['similarity_score'],
                'chunk_index': chunk['chunk_index'],
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            })
        
        context = "\n\n".join(context_parts)
        
        # Create general legal prompt with strategic protocols
        prompt = create_general_legal_prompt(query, context, self.strategic_protocols)
        
        try:
            # Generate answer using Ollama with optimized parameters
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                }
                
                # Add hardware-optimized parameters if available
                if OPTIMIZER_AVAILABLE and rag_optimizer is not None and model_name in rag_optimizer.available_models:
                    optimal_settings = rag_optimizer.get_optimal_settings(model_name)
                    payload.update({
                        "top_p": optimal_settings.get('top_p', 0.9),
                        "frequency_penalty": optimal_settings.get('frequency_penalty', 0.1),
                        "presence_penalty": optimal_settings.get('presence_penalty', 0.1)
                    })
                
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get('response', 'No response generated')
                        
                        rag_result = {
                            'answer': answer,
                            'sources': sources,
                            'query': query,
                            'model_used': model_name,
                            'context_chunks': len(relevant_chunks),
                            'generated_at': datetime.now().isoformat(),
                            'prompt_tokens': len(prompt.split()),
                            'response_tokens': len(answer.split()),
                            'temperature': temperature,
                            'hardware_optimized': OPTIMIZER_AVAILABLE and rag_optimizer is not None,
                            'protocol_enforcement_requested': enforce_protocols
                        }
                        
                        # Step 2: MCP Response Protocol Enforcement (if available and requested)
                        if enforce_protocols and MCP_AVAILABLE and mcp_rag_server is not None:
                            is_compliant, compliance_msg, enforcement_metadata = mcp_rag_server.enforce_protocol_on_response(
                                self.matter_id, rag_result
                            )
                            
                            rag_result.update({
                                'protocol_compliance': is_compliant,
                                'compliance_message': compliance_msg,
                                'enforcement_metadata': enforcement_metadata
                            })
                            
                            if not is_compliant:
                                logger.warning(f"Protocol compliance issue: {compliance_msg}")
                        
                        return rag_result
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error {response.status}: {error_text}")
                        return {
                            'answer': f"Error generating response: {response.status}",
                            'sources': sources,
                            'query': query,
                            'model_used': model_name,
                            'context_chunks': len(relevant_chunks),
                            'error': f"API Error: {response.status}",
                            'protocol_enforcement_requested': enforce_protocols
                        }
        
        except Exception as e:
            logger.error(f"Failed to generate RAG answer: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': sources,
                'query': query,
                'model_used': model_name,
                'context_chunks': len(relevant_chunks),
                'error': str(e),
                'protocol_enforcement_requested': enforce_protocols
            }
    
    def get_document_status(self) -> Dict[str, Any]:
        """Get status of all documents in the RAG system with hardware info"""
        status = {
            'total_documents': len(self.document_metadata),
            'total_chunks': len(self.chunk_metadata),
            'vector_index_size': self.vector_index.ntotal if self.vector_index else 0,
            'embedding_model': self.embedding_model_name,
            'matter_id': self.matter_id,
            'storage_path': str(self.rag_base_path),
            'documents': list(self.document_metadata.values()),
            'configuration': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'batch_size': self.batch_size,
                'max_docs_per_matter': self.max_docs_per_matter
            }
        }
        
        # Add hardware optimization info if available
        if OPTIMIZER_AVAILABLE and rag_optimizer is not None:
            status['hardware_optimization'] = {
                'enabled': True,
                'ram_gb': rag_optimizer.ram_gb,
                'vram_gb': rag_optimizer.vram_gb,
                'performance_config': rag_optimizer.performance_config
            }
        
        return status
    
    def delete_document(self, doc_id: str) -> Tuple[bool, str]:
        """Archive a document (move to archive instead of deleting)"""
        if doc_id not in self.document_metadata:
            return False, "Document not found"
        
        try:
            # Create archive directory if it doesn't exist
            archive_path = self.rag_base_path / "archived_documents"
            archive_path.mkdir(exist_ok=True)
            
            # Archive document metadata
            archived_metadata_path = archive_path / "archived_metadata.json"
            archived_metadata = {}
            if archived_metadata_path.exists():
                with open(archived_metadata_path, 'r', encoding='utf-8') as f:
                    archived_metadata = json.load(f)
            
            # Add current document to archive metadata
            archived_metadata[doc_id] = {
                **self.document_metadata[doc_id],
                'archived_at': datetime.now().isoformat(),
                'archived_chunks': [chunk for chunk in self.chunk_metadata if chunk['doc_id'] == doc_id]
            }
            
            # Save updated archive metadata
            with open(archived_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(archived_metadata, f, indent=2)
            
            # Remove chunks from active metadata
            original_chunk_count = len(self.chunk_metadata)
            self.chunk_metadata = [
                chunk for chunk in self.chunk_metadata 
                if chunk['doc_id'] != doc_id
            ]
            removed_chunks = original_chunk_count - len(self.chunk_metadata)
            
            # Remove document from active metadata
            del self.document_metadata[doc_id]
            
            # Move document file to archive (don't delete)
            doc_path = self.documents_path / f"{doc_id}.txt"
            if doc_path.exists():
                archived_doc_path = archive_path / f"{doc_id}.txt"
                doc_path.rename(archived_doc_path)
            
            # Rebuild vector index (expensive operation)
            if self.chunk_metadata and self.embedding_model and faiss is not None and np is not None:
                try:
                    chunk_texts = [chunk['text'] for chunk in self.chunk_metadata]
                    
                    # Process embeddings in batches
                    all_embeddings = []
                    for i in range(0, len(chunk_texts), self.batch_size):
                        batch_texts = chunk_texts[i:i + self.batch_size]
                        batch_embeddings = self.embedding_model.encode(batch_texts)
                        all_embeddings.append(batch_embeddings)
                    
                    if all_embeddings:
                        embeddings = np.vstack(all_embeddings)
                    else:
                        embeddings = np.array([])
                    
                    # Recreate index with same configuration
                    if len(embeddings) > 0:
                        dimension = embeddings.shape[1]
                        if OPTIMIZER_AVAILABLE and rag_optimizer is not None and rag_optimizer.performance_config.get("vector_index_type") == "IVF":
                            nlist = min(100, max(1, len(self.chunk_metadata) // 10))  # Ensure nlist >= 1
                            quantizer = faiss.IndexFlatIP(dimension)
                            self.vector_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                            if len(embeddings) >= nlist:  # Only train if we have enough data
                                self.vector_index.train(embeddings.astype(np.float32))
                        else:
                            self.vector_index = faiss.IndexFlatIP(dimension)
                        
                        # Normalize and add embeddings
                        embeddings_normalized = embeddings.astype(np.float32)
                        safe_faiss_normalize(embeddings_normalized)
                        self.vector_index.add(embeddings_normalized)
                        
                        # Update vector indices in metadata
                        for i, chunk in enumerate(self.chunk_metadata):
                            chunk['vector_index'] = i
                    else:
                        # No embeddings left, clear the index
                        self.vector_index = None
                except Exception as e:
                    logger.error(f"Error rebuilding vector index after archiving: {e}")
                    # Clear the index if rebuilding fails
                    self.vector_index = None
            else:
                self.vector_index = None
            
            self._save_metadata()
            self._save_vector_index()
            
            logger.info(f"Archived document {doc_id} and removed {removed_chunks} chunks from active search")
            return True, f"Successfully archived document and removed {removed_chunks} chunks from active search"
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False, f"Deletion error: {str(e)}"

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for BGE vs standard embeddings"""
        stats = {
            'using_bge_embeddings': self.is_using_bge,
            'reranking_enabled': self.enable_reranking and self.reranker_model is not None,
            'embedding_model': self.embedding_model_name,
            'total_documents': len(self.document_metadata),
            'total_chunks': len(self.chunk_metadata)
        }
        
        # Safe calculation of performance statistics
        embedding_times = self.performance_stats.get('embedding_time', [])
        if embedding_times:
            stats['avg_search_time'] = float(np.mean(embedding_times))
            stats['search_count'] = len(embedding_times)
        
        rerank_times = self.performance_stats.get('rerank_time', [])
        if rerank_times:
            stats['avg_rerank_time'] = float(np.mean(rerank_times))
            stats['rerank_count'] = len(rerank_times)
        
        improvements = self.performance_stats.get('search_improvements', [])
        if improvements:
            stats['avg_score_improvement'] = float(np.mean(improvements))
            stats['max_score_improvement'] = float(np.max(improvements))
        
        return stats


class RAGSessionManager:
    """Manages RAG sessions and integrates with MCP server for protocol enforcement"""
    
    def __init__(self):
        self.active_pipelines: Dict[str, LocalRAGPipeline] = {}
        self.session_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_or_create_pipeline(self, matter_id: str) -> LocalRAGPipeline:
        """Get existing or create new RAG pipeline for a matter"""
        if matter_id not in self.active_pipelines:
            self.active_pipelines[matter_id] = LocalRAGPipeline(matter_id)
        return self.active_pipelines[matter_id]
    
    def add_to_session_history(self, matter_id: str, query_result: Dict[str, Any]):
        """Add query result to session history"""
        if matter_id not in self.session_history:
            self.session_history[matter_id] = []
        
        self.session_history[matter_id].append({
            'timestamp': datetime.now().isoformat(),
            'query': query_result.get('query'),
            'answer': query_result.get('answer'),
            'model_used': query_result.get('model_used'),
            'sources_count': len(query_result.get('sources', [])),
            'context_chunks': query_result.get('context_chunks', 0),
            'hardware_optimized': query_result.get('hardware_optimized', False)
        })
    
    def get_session_history(self, matter_id: str) -> List[Dict[str, Any]]:
        """Get session history for a matter"""
        return self.session_history.get(matter_id, [])


# Global session manager
rag_session_manager = RAGSessionManager() 