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
    Local Retrieval-Augmented Generation Pipeline for Strategic Counsel
    
    Features:
    - Local document ingestion and chunking
    - Vector embeddings with FAISS storage
    - Multi-LLM support through Ollama
    - MCP server integration for protocol enforcement
    - Citation and provenance tracking
    - Hardware-optimized configuration
    """
    
    def __init__(self, 
                 matter_id: str,
                 embedding_model: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 ollama_base_url: str = "http://localhost:11434"):
        
        self.matter_id = matter_id
        self.ollama_base_url = ollama_base_url
        
        # Load strategic protocols for consistent prompting
        self.strategic_protocols = load_strategic_protocols()
        
        # Use optimized settings if available
        if OPTIMIZER_AVAILABLE and rag_optimizer is not None:
            self.embedding_model_name = embedding_model or rag_optimizer.embedding_config["model"]
            self.chunk_size = chunk_size or rag_optimizer.embedding_config["chunk_size"]
            self.chunk_overlap = chunk_overlap or rag_optimizer.embedding_config["chunk_overlap"]
            self.batch_size = rag_optimizer.embedding_config["batch_size"]
            self.max_docs_per_matter = rag_optimizer.embedding_config["max_docs_per_matter"]
        else:
            # Fallback to default settings
            self.embedding_model_name = embedding_model or "all-MiniLM-L6-v2"
            self.chunk_size = chunk_size or 500
            self.chunk_overlap = chunk_overlap or 50
            self.batch_size = 16
            self.max_docs_per_matter = 100
        
        # Initialize paths
        self.rag_base_path = APP_BASE_PATH / "rag_storage" / matter_id
        self.vector_db_path = self.rag_base_path / "vector_db"
        self.documents_path = self.rag_base_path / "documents"
        self.metadata_path = self.rag_base_path / "metadata.json"
        
        # Create directories
        for path in [self.rag_base_path, self.vector_db_path, self.documents_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_model = None
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
        """Initialize the local embedding model with hardware optimization"""
        if not EMBEDDING_AVAILABLE or SentenceTransformer is None:
            logger.error("sentence-transformers not available. Cannot initialize embedding model.")
            return False
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Hardware optimizations if available
            if OPTIMIZER_AVAILABLE and rag_optimizer is not None:
                # Enable GPU acceleration if available and beneficial
                if hasattr(self.embedding_model, 'encode') and rag_optimizer.performance_config.get("gpu_acceleration", False):
                    try:
                        import torch
                        if torch.cuda.is_available():
                            self.embedding_model = self.embedding_model.cuda()
                            logger.info(f"Enabled GPU acceleration for embedding model")
                    except ImportError:
                        logger.info("GPU acceleration not available (PyTorch not found)")
            
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {self.embedding_model_name}: {e}")
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
                    faiss.normalize_L2(embeddings.astype(np.float32))
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
        Search for relevant document chunks with hardware-optimized retrieval
        
        Returns: List of chunks with similarity scores and metadata
        """
        if not self.embedding_model or not self.vector_index or faiss is None or np is None:
            return []
        
        try:
            # Embed the query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding.astype(np.float32))
            
            # Search vector index
            scores, indices = self.vector_index.search(query_embedding.astype(np.float32), top_k)
            
            # Retrieve chunk metadata
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    chunk = self.chunk_metadata[idx].copy()
                    chunk['similarity_score'] = float(score)
                    
                    # Add document metadata
                    doc_id = chunk['doc_id']
                    if doc_id in self.document_metadata:
                        chunk['document_info'] = self.document_metadata[doc_id]
                    
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
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
        """Delete a document and its chunks from the system"""
        if doc_id not in self.document_metadata:
            return False, "Document not found"
        
        try:
            # Remove chunks from metadata
            original_chunk_count = len(self.chunk_metadata)
            self.chunk_metadata = [
                chunk for chunk in self.chunk_metadata 
                if chunk['doc_id'] != doc_id
            ]
            removed_chunks = original_chunk_count - len(self.chunk_metadata)
            
            # Remove document metadata
            del self.document_metadata[doc_id]
            
            # Remove document file
            doc_path = self.documents_path / f"{doc_id}.txt"
            if doc_path.exists():
                doc_path.unlink()
            
            # Rebuild vector index (expensive operation)
            if self.chunk_metadata and self.embedding_model and faiss is not None and np is not None:
                chunk_texts = [chunk['text'] for chunk in self.chunk_metadata]
                
                # Process embeddings in batches
                all_embeddings = []
                for i in range(0, len(chunk_texts), self.batch_size):
                    batch_texts = chunk_texts[i:i + self.batch_size]
                    batch_embeddings = self.embedding_model.encode(batch_texts)
                    all_embeddings.append(batch_embeddings)
                
                embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
                
                # Recreate index with same configuration
                dimension = embeddings.shape[1]
                if OPTIMIZER_AVAILABLE and rag_optimizer is not None and rag_optimizer.performance_config.get("vector_index_type") == "IVF":
                    nlist = min(100, len(self.chunk_metadata) // 10)  # Adjust nlist based on data size
                    quantizer = faiss.IndexFlatIP(dimension)
                    self.vector_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                    if len(embeddings) > 0:
                        self.vector_index.train(embeddings.astype(np.float32))
                else:
                    self.vector_index = faiss.IndexFlatIP(dimension)
                
                if len(embeddings) > 0:
                    faiss.normalize_L2(embeddings.astype(np.float32))
                    self.vector_index.add(embeddings.astype(np.float32))
                
                # Update vector indices in metadata
                for i, chunk in enumerate(self.chunk_metadata):
                    chunk['vector_index'] = i
            else:
                self.vector_index = None
            
            self._save_metadata()
            self._save_vector_index()
            
            logger.info(f"Deleted document {doc_id} and {removed_chunks} chunks")
            return True, f"Successfully deleted document and {removed_chunks} chunks"
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False, f"Deletion error: {str(e)}"


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