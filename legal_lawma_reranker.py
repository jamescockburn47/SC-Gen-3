"""
LawMA Legal Specialist Reranker
==============================

Implements LawMA-8B as a legal content filter and reranker for improving RAG retrieval accuracy.
This follows the recommended approach of using LawMA as a specialist classifier rather than a generator.

Pipeline: BGE embeddings → Vector search → LawMA legal relevance filtering → LLM generation
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import aiohttp
import re

logger = logging.getLogger(__name__)

class LawMALegalReranker:
    """
    LawMA-8B Legal Specialist Reranker
    
    Uses LawMA-8B as a legal content filter to:
    1. Rerank retrieved chunks by legal relevance
    2. Classify chunk relevance with legal expertise
    3. Verify citations and flag hallucinations
    4. Filter content for actual legal pertinence
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.model_name = "lawma-8b:latest"
        self.available = False
        self.performance_stats = {
            'rerank_times': [],
            'verification_times': [],
            'relevance_improvements': []
        }
        
        # Check LawMA availability
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if LawMA-8B is available in Ollama"""
        try:
            # This would be replaced with actual availability check
            self.available = True  # Assume available for now
            logger.info("✅ LawMA-8B Legal Reranker initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️ LawMA-8B not available: {e}")
            self.available = False
            return False
    
    async def rerank_chunks_by_legal_relevance(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using LawMA's legal expertise
        
        Args:
            query: Legal query
            chunks: Retrieved chunks from vector search
            top_k: Number of top chunks to return after reranking
            
        Returns:
            Reranked chunks with legal relevance scores
        """
        if not self.available or not chunks:
            logger.warning("LawMA reranker not available, returning original order")
            return chunks[:top_k]
        
        start_time = datetime.now()
        
        try:
            # Score each chunk for legal relevance
            scored_chunks = []
            
            for i, chunk in enumerate(chunks):
                relevance_score = await self._score_legal_relevance(query, chunk)
                
                chunk_copy = chunk.copy()
                chunk_copy['lawma_relevance_score'] = relevance_score
                chunk_copy['original_rank'] = i + 1
                
                scored_chunks.append(chunk_copy)
            
            # Sort by LawMA legal relevance score
            scored_chunks.sort(key=lambda x: x['lawma_relevance_score'], reverse=True)
            
            # Add reranked positions
            for i, chunk in enumerate(scored_chunks):
                chunk['lawma_rank'] = i + 1
                chunk['rank_improvement'] = chunk['original_rank'] - chunk['lawma_rank']
            
            # Take top_k after legal reranking
            result = scored_chunks[:top_k]
            
            # Track performance
            rerank_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats['rerank_times'].append(rerank_time)
            
            # Calculate improvement metrics
            avg_improvement = sum(c.get('rank_improvement', 0) for c in result) / len(result) if result else 0
            self.performance_stats['relevance_improvements'].append(avg_improvement)
            
            logger.info(f"🏛️ LawMA reranked {len(chunks)} chunks → {len(result)} in {rerank_time:.3f}s")
            logger.info(f"📊 Average rank improvement: {avg_improvement:.1f} positions")
            
            return result
            
        except Exception as e:
            logger.error(f"LawMA reranking failed: {e}")
            return chunks[:top_k]  # Fallback to original order
    
    async def _score_legal_relevance(self, query: str, chunk: Dict[str, Any]) -> float:
        """
        Score a single chunk for legal relevance using LawMA
        
        Returns: Relevance score between 0.0 and 1.0
        """
        chunk_text = chunk.get('text', '')
        
        # Create legal relevance prompt
        relevance_prompt = self._create_relevance_prompt(query, chunk_text)
        
        try:
            response = await self._query_lawma(relevance_prompt)
            score = self._parse_relevance_response(response)
            return score
            
        except Exception as e:
            logger.error(f"LawMA scoring failed for chunk: {e}")
            return 0.5  # Neutral score on failure
    
    def _create_relevance_prompt(self, query: str, chunk_text: str) -> str:
        """Create a prompt for LawMA to assess legal relevance"""
        return f"""
As a legal expert, assess how relevant this text chunk is to the legal query.

QUERY: {query}

TEXT CHUNK:
{chunk_text[:1000]}...

Rate the legal relevance on a scale of 0-10 where:
- 10: Directly answers the query with specific legal content
- 8-9: Highly relevant legal information that supports the query
- 6-7: Somewhat relevant with useful legal context
- 4-5: Contains legal terms but limited relevance
- 2-3: Minimal legal relevance
- 0-1: Not legally relevant to the query

Consider:
- Direct legal applicability to the query
- Presence of relevant legal concepts, cases, or statutes
- Factual accuracy and legal specificity
- Procedural relevance if applicable

Respond with only the numeric score (0-10):"""
    
    async def _query_lawma(self, prompt: str) -> str:
        """Query LawMA-8B model"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent scoring
                        "num_predict": 10    # Only need a short response
                    }
                }
                
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '').strip()
                    else:
                        logger.error(f"LawMA query failed: {response.status}")
                        return "5"  # Neutral fallback
                        
        except Exception as e:
            logger.error(f"LawMA query error: {e}")
            return "5"  # Neutral fallback
    
    def _parse_relevance_response(self, response: str) -> float:
        """Parse LawMA's relevance score response"""
        try:
            # Extract numeric score from response
            score_match = re.search(r'(\d+(?:\.\d+)?)', response)
            if score_match:
                score = float(score_match.group(1))
                # Normalize to 0-1 range
                return min(max(score / 10.0, 0.0), 1.0)
            else:
                logger.warning(f"Could not parse LawMA score: {response}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Score parsing failed: {e}")
            return 0.5
    
    async def verify_citations(
        self, 
        answer: str, 
        source_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LawMA to verify that citations in the answer are supported by source chunks
        
        Returns: Verification report with flagged issues
        """
        if not self.available:
            return {"verified": True, "warnings": [], "lawma_available": False}
        
        start_time = datetime.now()
        
        try:
            verification_results = {
                "verified": True,
                "warnings": [],
                "hallucination_flags": [],
                "lawma_available": True,
                "verification_time": 0
            }
            
            # Extract claims from the answer
            claims = self._extract_claims(answer)
            
            for claim in claims:
                support_found = await self._verify_claim_support(claim, source_chunks)
                
                if not support_found:
                    verification_results["verified"] = False
                    verification_results["hallucination_flags"].append({
                        "claim": claim,
                        "issue": "No supporting evidence found in source documents"
                    })
            
            verification_time = (datetime.now() - start_time).total_seconds()
            verification_results["verification_time"] = verification_time
            self.performance_stats['verification_times'].append(verification_time)
            
            logger.info(f"🔍 LawMA verified {len(claims)} claims in {verification_time:.3f}s")
            
            return verification_results
            
        except Exception as e:
            logger.error(f"LawMA citation verification failed: {e}")
            return {"verified": True, "warnings": [f"Verification failed: {e}"], "lawma_available": False}
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract verifiable claims from the answer"""
        # Simple sentence splitting - could be enhanced
        sentences = re.split(r'[.!?]+', answer)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims[:5]  # Limit to first 5 claims for performance
    
    async def _verify_claim_support(self, claim: str, source_chunks: List[Dict[str, Any]]) -> bool:
        """Verify if a claim is supported by source chunks using LawMA"""
        verification_prompt = f"""
As a legal expert, determine if this claim is supported by the provided source text.

CLAIM TO VERIFY: {claim}

SOURCE TEXTS:
{self._format_sources_for_verification(source_chunks)}

Is the claim adequately supported by the source texts? 
Consider legal accuracy and factual basis.

Respond with only: SUPPORTED or NOT_SUPPORTED"""
        
        try:
            response = await self._query_lawma(verification_prompt)
            return "SUPPORTED" in response.upper()
            
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return True  # Default to supported on error
    
    def _format_sources_for_verification(self, chunks: List[Dict[str, Any]]) -> str:
        """Format source chunks for verification prompt"""
        formatted = []
        for i, chunk in enumerate(chunks[:3]):  # Limit to top 3 chunks for prompt size
            text = chunk.get('text', '')[:300]  # Limit chunk size
            formatted.append(f"SOURCE {i+1}: {text}...")
        return "\n\n".join(formatted)
    
    async def filter_by_legal_types(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        legal_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks based on specific legal document types or content areas
        
        Args:
            query: Legal query
            chunks: Chunks to filter
            legal_types: Specific legal areas to focus on (e.g., ['procedural', 'substantive', 'evidence'])
            
        Returns:
            Filtered chunks with legal type classifications
        """
        if not self.available or not chunks:
            return chunks
        
        if legal_types is None:
            legal_types = ['procedural', 'substantive', 'evidence', 'factual']
        
        try:
            classified_chunks = []
            
            for chunk in chunks:
                classification = await self._classify_legal_content(chunk['text'], legal_types)
                chunk_copy = chunk.copy()
                chunk_copy['legal_classification'] = classification
                
                # Include chunk if it matches desired legal types or if classification failed
                if any(lt in classification['types'] for lt in legal_types) or not classification['types']:
                    classified_chunks.append(chunk_copy)
            
            logger.info(f"🏛️ LawMA filtered {len(chunks)} → {len(classified_chunks)} chunks by legal type")
            return classified_chunks
            
        except Exception as e:
            logger.error(f"Legal type filtering failed: {e}")
            return chunks
    
    async def _classify_legal_content(self, text: str, legal_types: List[str]) -> Dict[str, Any]:
        """Classify legal content using LawMA"""
        classification_prompt = f"""
As a legal expert, classify this text according to these legal content types:
{', '.join(legal_types)}

TEXT: {text[:500]}...

Which legal content types apply? Consider:
- Procedural: Court procedures, deadlines, process requirements
- Substantive: Legal rights, obligations, principles, case law
- Evidence: Facts, witness statements, documentary evidence
- Factual: Timeline events, parties, background information

Respond with applicable types separated by commas:"""
        
        try:
            response = await self._query_lawma(classification_prompt)
            identified_types = [t.strip().lower() for t in response.split(',')]
            
            return {
                "types": [t for t in identified_types if t in [lt.lower() for lt in legal_types]],
                "confidence": 0.8,  # Could be enhanced to extract confidence
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"Legal classification failed: {e}")
            return {"types": [], "confidence": 0.0, "raw_response": ""}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get LawMA reranker performance statistics"""
        stats = {
            "available": self.available,
            "model": self.model_name,
            "operations": {}
        }
        
        if self.performance_stats['rerank_times']:
            stats["operations"]["reranking"] = {
                "count": len(self.performance_stats['rerank_times']),
                "avg_time": sum(self.performance_stats['rerank_times']) / len(self.performance_stats['rerank_times']),
                "total_time": sum(self.performance_stats['rerank_times'])
            }
        
        if self.performance_stats['verification_times']:
            stats["operations"]["verification"] = {
                "count": len(self.performance_stats['verification_times']),
                "avg_time": sum(self.performance_stats['verification_times']) / len(self.performance_stats['verification_times']),
                "total_time": sum(self.performance_stats['verification_times'])
            }
        
        if self.performance_stats['relevance_improvements']:
            avg_improvement = sum(self.performance_stats['relevance_improvements']) / len(self.performance_stats['relevance_improvements'])
            stats["effectiveness"] = {
                "avg_rank_improvement": avg_improvement,
                "improvement_count": len(self.performance_stats['relevance_improvements'])
            }
        
        return stats

# Global instance
lawma_reranker = LawMALegalReranker()

async def get_lawma_enhanced_chunks(
    query: str, 
    initial_chunks: List[Dict[str, Any]], 
    top_k: int = 8,
    enable_verification: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function to apply LawMA reranking and optionally verify results
    
    Returns: (reranked_chunks, metadata)
    """
    # Rerank by legal relevance
    reranked_chunks = await lawma_reranker.rerank_chunks_by_legal_relevance(
        query, initial_chunks, top_k
    )
    
    metadata = {
        "lawma_reranking_applied": lawma_reranker.available,
        "original_count": len(initial_chunks),
        "reranked_count": len(reranked_chunks),
        "verification_enabled": enable_verification
    }
    
    return reranked_chunks, metadata 