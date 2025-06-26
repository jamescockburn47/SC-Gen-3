#!/usr/bin/env python3
"""
Temporarily disable multi-agent RAG system to use fixed single-agent approach
"""

import os
import shutil
from pathlib import Path

def disable_multiagent_rag():
    """Disable the multi-agent system temporarily"""
    
    print("🔧 Disabling Multi-Agent RAG System")
    print("=" * 40)
    
    # Backup the original multi-agent file
    multi_agent_file = Path("multi_agent_rag_orchestrator.py")
    backup_file = Path("multi_agent_rag_orchestrator.py.backup")
    
    if multi_agent_file.exists() and not backup_file.exists():
        print("📦 Creating backup of multi_agent_rag_orchestrator.py")
        shutil.copy2(multi_agent_file, backup_file)
    
    # Create a simple replacement that falls back to single-agent
    replacement_content = '''# multi_agent_rag_orchestrator.py - DISABLED VERSION

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
'''
    
    # Write the replacement file
    with open(multi_agent_file, 'w') as f:
        f.write(replacement_content)
    
    print("✅ Multi-agent system disabled")
    print("✅ Single-agent fallback active")
    print()
    print("Now the RAG system will use:")
    print("- Single model (phi3:latest) for reliability")
    print("- Strict prompting to prevent hallucination")
    print("- Direct document extraction only")
    print()
    print("To test: python3 test_rag_hallucination_fix.py")

def restore_multiagent_rag():
    """Restore the original multi-agent system"""
    
    print("🔄 Restoring Multi-Agent RAG System")
    print("=" * 40)
    
    multi_agent_file = Path("multi_agent_rag_orchestrator.py")
    backup_file = Path("multi_agent_rag_orchestrator.py.backup")
    
    if backup_file.exists():
        print("📦 Restoring from backup")
        shutil.copy2(backup_file, multi_agent_file)
        print("✅ Multi-agent system restored")
    else:
        print("❌ No backup file found")
        print("   The original file may not have been backed up")

def check_status():
    """Check current status of multi-agent system"""
    
    multi_agent_file = Path("multi_agent_rag_orchestrator.py")
    backup_file = Path("multi_agent_rag_orchestrator.py.backup")
    
    print("📊 Multi-Agent RAG Status")
    print("=" * 30)
    
    if multi_agent_file.exists():
        with open(multi_agent_file, 'r') as f:
            content = f.read()
            if "DISABLED VERSION" in content:
                print("🔴 Status: DISABLED (single-agent fallback active)")
                print("   Reason: Preventing hallucination issues")
                print("   Using: phi3:latest with strict prompting")
            else:
                print("🟢 Status: ACTIVE (multi-agent system running)")
                print("   Warning: May produce template/placeholder responses")
    
    if backup_file.exists():
        print("📦 Backup: Available")
        print("   Use 'restore' to enable multi-agent system again")
    else:
        print("📦 Backup: Not found")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 disable_multiagent_rag.py disable    # Disable multi-agent system")
        print("  python3 disable_multiagent_rag.py restore    # Restore multi-agent system")
        print("  python3 disable_multiagent_rag.py status     # Check current status")
        return
    
    command = sys.argv[1].lower()
    
    if command == "disable":
        disable_multiagent_rag()
    elif command == "restore":
        restore_multiagent_rag()
    elif command == "status":
        check_status()
    else:
        print(f"Unknown command: {command}")
        print("Use: disable, restore, or status")

if __name__ == "__main__":
    main() 