# multi_agent_rag_orchestrator.py - DISABLED VERSION

"""
Multi-agent RAG orchestrator - TEMPORARILY DISABLED
Falls back to single-agent RAG to prevent hallucination issues
"""

import logging
from typing import Dict, Any
from local_rag_pipeline import rag_session_manager

logger = logging.getLogger(__name__)

class MultiAgentRAGOrchestrator:
    """
    DISABLED: Multi-agent system with hallucination issues
    Falls back to reliable single-agent RAG
    """
    
    def __init__(self, matter_id: str):
        self.matter_id = matter_id
        self.rag_pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        logger.warning("Multi-agent RAG system disabled - using single-agent fallback")
    
    async def process_query(self, query: str, max_context_chunks: int = 10) -> Dict[str, Any]:
        """
        Fallback to single-agent RAG processing
        """
        logger.info("Using single-agent fallback due to multi-agent system being disabled")
        
        # Use the reliable single-agent approach
        result = await self.rag_pipeline.generate_rag_answer(
            query=query,
            model_name="phi3:latest",  # Use fast, reliable model
            max_context_chunks=max_context_chunks,
            temperature=0.0,  # Deterministic
            enforce_protocols=True
        )
        
        # Format to match expected multi-agent output
        return {
            'answer': result.get('answer', 'No answer generated'),
            'agent_results': {},  # Empty since no multi-agent processing
            'agents_used': ['phi3:latest (single-agent fallback)'],
            'execution_time': 0,
            'confidence': 0.8,  # Fixed confidence for single-agent
            'sources': result.get('sources', []),
            'metadata': {
                'mode': 'single_agent_fallback',
                'reason': 'multi_agent_disabled_due_to_hallucination_issues',
                'model_used': 'phi3:latest'
            },
            'task_breakdown': {
                'single_agent_analysis': {
                    'model': 'phi3:latest',
                    'confidence': 0.8,
                    'execution_time': 0,
                    'success': True,
                    'key_findings': ['Single-agent analysis completed']
                }
            }
        }

# Global orchestrator instances per matter
_orchestrator_instances: Dict[str, MultiAgentRAGOrchestrator] = {}

def get_orchestrator(matter_id: str) -> MultiAgentRAGOrchestrator:
    """Get or create orchestrator instance for a matter"""
    if matter_id not in _orchestrator_instances:
        _orchestrator_instances[matter_id] = MultiAgentRAGOrchestrator(matter_id)
    return _orchestrator_instances[matter_id]
