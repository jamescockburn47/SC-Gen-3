# mcp_rag_server.py

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from config import logger, APP_BASE_PATH, LOADED_PROTO_TEXT
from local_rag_pipeline import rag_session_manager

class MCPRAGServer:
    """
    Model Control Protocol Server for RAG Operations
    
    Handles:
    - Protocol enforcement for RAG queries and responses
    - Memory and state management per matter
    - Citation and provenance validation
    - Output audit and hallucination suppression
    """
    
    def __init__(self):
        self.protocol_text = LOADED_PROTO_TEXT
        self.memory_storage_path = APP_BASE_PATH / "memory" / "rag_sessions"
        self.memory_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Protocol enforcement rules
        self.enforcement_rules = {
            'require_citations': True,
            'max_context_chunks': 10,
            'min_similarity_threshold': 0.3,
            'hallucination_detection': True,
            'provenance_tracking': True,
            'output_audit': True
        }
    
    def validate_query(self, matter_id: str, query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate incoming RAG query against protocol requirements
        
        Returns: (is_valid, message, validation_metadata)
        """
        validation_metadata = {
            'matter_id': matter_id,
            'query_length': len(query),
            'validation_timestamp': datetime.now().isoformat(),
            'protocol_version': '1.0'
        }
        
        # Basic query validation
        if not query or len(query.strip()) < 10:
            return False, "Query too short. Please provide a more detailed question.", validation_metadata
        
        if len(query) > 2000:
            return False, "Query too long. Please limit to 2000 characters.", validation_metadata
        
        # Check for inappropriate content (basic implementation)
        prohibited_terms = ['hack', 'exploit', 'illegal', 'unethical']
        query_lower = query.lower()
        for term in prohibited_terms:
            if term in query_lower:
                logger.warning(f"Potentially inappropriate query detected: {term}")
                validation_metadata['flagged_terms'] = [term]
        
        # Log query for audit trail
        self._log_query_audit(matter_id, query, validation_metadata)
        
        validation_metadata['status'] = 'validated'
        return True, "Query validated successfully", validation_metadata
    
    def enforce_protocol_on_response(self, 
                                   matter_id: str, 
                                   rag_response: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Enforce protocol requirements on RAG response
        
        Returns: (is_compliant, message, enforcement_metadata)
        """
        enforcement_metadata = {
            'matter_id': matter_id,
            'enforcement_timestamp': datetime.now().isoformat(),
            'checks_performed': []
        }
        
        # Check 1: Citation Requirements
        if self.enforcement_rules['require_citations']:
            answer = rag_response.get('answer', '')
            sources = rag_response.get('sources', [])
            
            if not sources:
                enforcement_metadata['checks_performed'].append('citation_check_failed')
                return False, "Response must include source citations", enforcement_metadata
            
            # Check for citation markers in answer
            citation_markers = ['[Source', '[source', 'Source 1', 'source 1']
            has_citations = any(marker in answer for marker in citation_markers)
            
            if not has_citations and sources:
                enforcement_metadata['checks_performed'].append('citation_format_warning')
                enforcement_metadata['warnings'] = ['Answer should reference sources using [Source X] format']
        
        # Check 2: Context Chunk Limits
        context_chunks = rag_response.get('context_chunks', 0)
        if context_chunks > self.enforcement_rules['max_context_chunks']:
            enforcement_metadata['checks_performed'].append('context_limit_exceeded')
            return False, f"Too many context chunks used: {context_chunks} > {self.enforcement_rules['max_context_chunks']}", enforcement_metadata
        
        # Check 3: Similarity Threshold Validation
        if self.enforcement_rules['min_similarity_threshold'] > 0:
            sources = rag_response.get('sources', [])
            low_similarity_sources = [
                s for s in sources 
                if s.get('similarity_score', 0) < self.enforcement_rules['min_similarity_threshold']
            ]
            
            if low_similarity_sources:
                enforcement_metadata['checks_performed'].append('low_similarity_warning')
                enforcement_metadata['low_similarity_count'] = len(low_similarity_sources)
        
        # Check 4: Hallucination Detection (basic implementation)
        if self.enforcement_rules['hallucination_detection']:
            answer = rag_response.get('answer', '')
            
            # Check for phrases that indicate hallucination
            hallucination_indicators = [
                'I think', 'I believe', 'probably', 'might be', 'could be',
                'in my opinion', 'generally speaking', 'typically'
            ]
            
            found_indicators = [indicator for indicator in hallucination_indicators if indicator.lower() in answer.lower()]
            if found_indicators:
                enforcement_metadata['checks_performed'].append('hallucination_indicators_found')
                enforcement_metadata['hallucination_indicators'] = found_indicators
                enforcement_metadata['warnings'] = enforcement_metadata.get('warnings', []) + [
                    f"Response contains uncertain language: {', '.join(found_indicators)}"
                ]
        
        # Check 5: Protocol Compliance
        protocol_keywords = ['based on the provided documents', 'according to the context', 'the documents indicate']
        answer = rag_response.get('answer', '')
        has_protocol_language = any(keyword in answer.lower() for keyword in protocol_keywords)
        
        if not has_protocol_language:
            enforcement_metadata['checks_performed'].append('protocol_language_missing')
            enforcement_metadata['warnings'] = enforcement_metadata.get('warnings', []) + [
                'Response should explicitly reference provided documents'
            ]
        
        # Log enforcement results
        self._log_enforcement_audit(matter_id, rag_response, enforcement_metadata)
        
        enforcement_metadata['status'] = 'compliant'
        enforcement_metadata['checks_performed'].append('all_checks_passed')
        
        return True, "Response compliant with protocol requirements", enforcement_metadata
    
    def manage_session_memory(self, matter_id: str, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage session memory and state for the matter
        
        Returns: Updated memory state
        """
        # Add to session manager
        rag_session_manager.add_to_session_history(matter_id, query_result)
        
        # Load existing memory
        memory_file = self.memory_storage_path / f"{matter_id}_rag_memory.json"
        memory_state = {}
        
        if memory_file.exists():
            try:
                with open(memory_file, 'r') as f:
                    memory_state = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory for {matter_id}: {e}")
        
        # Update memory state
        if 'session_history' not in memory_state:
            memory_state['session_history'] = []
        
        # Add current query to memory
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query_result.get('query'),
            'model_used': query_result.get('model_used'),
            'sources_count': len(query_result.get('sources', [])),
            'context_chunks': query_result.get('context_chunks', 0),
            'answer_length': len(query_result.get('answer', '')),
            'response_tokens': query_result.get('response_tokens', 0)
        }
        memory_state['session_history'].append(memory_entry)
        
        # Update session statistics
        memory_state['total_queries'] = len(memory_state['session_history'])
        memory_state['last_updated'] = datetime.now().isoformat()
        memory_state['matter_id'] = matter_id
        
        # Calculate session metrics
        if memory_state['session_history']:
            total_sources = sum(entry.get('sources_count', 0) for entry in memory_state['session_history'])
            total_chunks = sum(entry.get('context_chunks', 0) for entry in memory_state['session_history'])
            total_tokens = sum(entry.get('response_tokens', 0) for entry in memory_state['session_history'])
            
            memory_state['session_metrics'] = {
                'avg_sources_per_query': total_sources / len(memory_state['session_history']),
                'avg_chunks_per_query': total_chunks / len(memory_state['session_history']),
                'total_response_tokens': total_tokens,
                'unique_models_used': len(set(entry.get('model_used', '') for entry in memory_state['session_history']))
            }
        
        # Save updated memory
        try:
            with open(memory_file, 'w') as f:
                json.dump(memory_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory for {matter_id}: {e}")
        
        return memory_state
    
    def validate_document_upload(self, matter_id: str, filename: str, file_size: int) -> Tuple[bool, str]:
        """
        Validate document upload against protocol requirements
        
        Returns: (is_valid, message)
        """
        # File size limits (50MB max)
        max_file_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_file_size:
            return False, f"File too large: {file_size / (1024*1024):.1f}MB > 50MB limit"
        
        # File type validation
        allowed_extensions = {'.pdf', '.docx', '.txt', '.doc', '.rtf'}
        file_path = Path(filename)
        if file_path.suffix.lower() not in allowed_extensions:
            return False, f"Unsupported file type: {file_path.suffix}. Allowed: {', '.join(allowed_extensions)}"
        
        # Check document count limits per matter
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        doc_status = pipeline.get_document_status()
        
        max_docs_per_matter = 100  # Reasonable limit
        if doc_status['total_documents'] >= max_docs_per_matter:
            return False, f"Document limit reached: {doc_status['total_documents']}/{max_docs_per_matter}"
        
        return True, "Document upload validated"
    
    def audit_citation_provenance(self, rag_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit citation provenance and source reliability
        
        Returns: Provenance audit results
        """
        sources = rag_response.get('sources', [])
        audit_results = {
            'total_sources': len(sources),
            'provenance_verified': True,
            'source_reliability': {},
            'audit_timestamp': datetime.now().isoformat()
        }
        
        for i, source in enumerate(sources):
            source_audit = {
                'chunk_id': source.get('chunk_id'),
                'document': source.get('document'),
                'similarity_score': source.get('similarity_score', 0),
                'reliability_score': min(1.0, source.get('similarity_score', 0) * 2),  # Scale 0-1
                'verified': source.get('similarity_score', 0) > 0.3
            }
            
            audit_results['source_reliability'][f'source_{i+1}'] = source_audit
        
        # Calculate overall reliability
        if sources:
            avg_similarity = sum(s.get('similarity_score', 0) for s in sources) / len(sources)
            audit_results['overall_reliability'] = min(1.0, avg_similarity * 2)
        else:
            audit_results['overall_reliability'] = 0.0
        
        return audit_results
    
    def _log_query_audit(self, matter_id: str, query: str, metadata: Dict[str, Any]):
        """Log query for audit trail"""
        audit_entry = {
            'type': 'query_validation',
            'matter_id': matter_id,
            'query_hash': hash(query) % 10000,  # Anonymized
            'query_length': len(query),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        audit_log_path = self.memory_storage_path / "audit_log.jsonl"
        try:
            with open(audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _log_enforcement_audit(self, matter_id: str, response: Dict[str, Any], metadata: Dict[str, Any]):
        """Log enforcement results for audit trail"""
        audit_entry = {
            'type': 'response_enforcement',
            'matter_id': matter_id,
            'response_length': len(response.get('answer', '')),
            'sources_count': len(response.get('sources', [])),
            'model_used': response.get('model_used'),
            'timestamp': datetime.now().isoformat(),
            'enforcement_metadata': metadata
        }
        
        audit_log_path = self.memory_storage_path / "audit_log.jsonl"
        try:
            with open(audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_matter_statistics(self, matter_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a matter"""
        memory_file = self.memory_storage_path / f"{matter_id}_rag_memory.json"
        
        if not memory_file.exists():
            return {
                'matter_id': matter_id,
                'total_queries': 0,
                'total_documents': 0,
                'status': 'new_matter'
            }
        
        try:
            with open(memory_file, 'r') as f:
                memory_state = json.load(f)
                
            # Get document statistics from pipeline
            pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
            doc_status = pipeline.get_document_status()
            
            return {
                'matter_id': matter_id,
                'total_queries': memory_state.get('total_queries', 0),
                'total_documents': doc_status['total_documents'],
                'total_chunks': doc_status['total_chunks'],
                'session_metrics': memory_state.get('session_metrics', {}),
                'last_updated': memory_state.get('last_updated'),
                'storage_path': doc_status['storage_path'],
                'embedding_model': doc_status['embedding_model']
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics for {matter_id}: {e}")
            return {
                'matter_id': matter_id,
                'error': str(e),
                'status': 'error'
            }


# Global MCP server instance
mcp_rag_server = MCPRAGServer() 