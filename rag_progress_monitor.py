import time
import asyncio
import threading
from typing import Dict, Any, Optional, Callable
import streamlit as st
from datetime import datetime, timedelta
import aiohttp
from config import logger

class RAGProgressMonitor:
    """Real-time progress monitoring for RAG queries with streaming support"""
    
    def __init__(self):
        self.active_queries: Dict[str, Dict[str, Any]] = {}
        self.query_history: list = []
        
    def start_query_monitoring(self, query_id: str, query: str, model: str) -> None:
        """Start monitoring a new query"""
        self.active_queries[query_id] = {
            'query': query,
            'model': model,
            'start_time': datetime.now(),
            'status': 'starting',
            'progress': 0,
            'current_stage': 'Initializing...',
            'chunks_retrieved': 0,
            'tokens_generated': 0,
            'estimated_completion': None,
            'warnings': []
        }
    
    def update_query_progress(self, query_id: str, stage: str, progress: int, 
                            chunks: int = 0, tokens: int = 0, warnings: list = None):
        """Update progress for an active query"""
        if query_id in self.active_queries:
            query_info = self.active_queries[query_id]
            query_info.update({
                'current_stage': stage,
                'progress': progress,
                'chunks_retrieved': chunks,
                'tokens_generated': tokens,
                'status': 'running' if progress < 100 else 'completed'
            })
            
            if warnings:
                query_info['warnings'].extend(warnings)
            
            # Estimate completion time
            elapsed = datetime.now() - query_info['start_time']
            if progress > 10:
                total_estimated = elapsed * (100 / progress)
                query_info['estimated_completion'] = query_info['start_time'] + total_estimated
    
    def complete_query(self, query_id: str, success: bool, result: Dict[str, Any] = None):
        """Mark query as completed"""
        if query_id in self.active_queries:
            query_info = self.active_queries[query_id]
            query_info.update({
                'status': 'completed' if success else 'failed',
                'progress': 100,
                'end_time': datetime.now(),
                'result': result
            })
            
            # Move to history
            self.query_history.append(query_info.copy())
            if len(self.query_history) > 50:  # Keep last 50 queries
                self.query_history.pop(0)
            
            # Remove from active
            del self.active_queries[query_id]
    
    def get_query_status(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a query"""
        return self.active_queries.get(query_id)
    
    def display_progress_ui(self, query_id: str):
        """Display real-time progress in Streamlit UI"""
        if query_id not in self.active_queries:
            return
        
        query_info = self.active_queries[query_id]
        
        # Progress bar
        progress_bar = st.progress(query_info['progress'] / 100)
        
        # Status information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", query_info['current_stage'])
            
        with col2:
            elapsed = datetime.now() - query_info['start_time']
            st.metric("Elapsed", f"{elapsed.seconds}s")
            
        with col3:
            if query_info['estimated_completion']:
                remaining = query_info['estimated_completion'] - datetime.now()
                st.metric("ETA", f"{max(0, remaining.seconds)}s")
        
        # Detailed information
        st.write(f"**Model:** {query_info['model']}")
        st.write(f"**Chunks Retrieved:** {query_info['chunks_retrieved']}")
        st.write(f"**Tokens Generated:** {query_info['tokens_generated']}")
        
        # Warnings
        if query_info['warnings']:
            st.warning("⚠️ " + "; ".join(query_info['warnings']))

class StreamingRAGGenerator:
    """Streaming RAG response generator with real-time progress"""
    
    def __init__(self, monitor: RAGProgressMonitor):
        self.monitor = monitor
        self.ollama_base_url = "http://localhost:11434"
    
    async def generate_streaming_response(self, query_id: str, prompt: str, 
                                        model: str, temperature: float = 0.1) -> AsyncIterator[str]:
        """Generate streaming response with progress updates"""
        
        try:
            self.monitor.update_query_progress(query_id, "Connecting to LLM...", 20)
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": True  # Enable streaming
                }
                
                self.monitor.update_query_progress(query_id, "Generating response...", 30)
                
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        self.monitor.update_query_progress(
                            query_id, f"Error: {response.status}", 100, 
                            warnings=[f"HTTP {response.status}: {error_text}"]
                        )
                        return
                    
                    full_response = ""
                    tokens_count = 0
                    
                    async for line in response.content:
                        if line:
                            try:
                                import json
                                data = json.loads(line.decode('utf-8'))
                                
                                if 'response' in data:
                                    chunk = data['response']
                                    full_response += chunk
                                    tokens_count += len(chunk.split())
                                    
                                    # Update progress based on tokens generated
                                    progress = min(90, 30 + (tokens_count // 10))
                                    self.monitor.update_query_progress(
                                        query_id, "Generating response...", progress,
                                        tokens=tokens_count
                                    )
                                    
                                    yield chunk
                                
                                if data.get('done', False):
                                    self.monitor.update_query_progress(
                                        query_id, "Response complete", 100,
                                        tokens=tokens_count
                                    )
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                
        except asyncio.TimeoutError:
            self.monitor.update_query_progress(
                query_id, "Timeout", 100,
                warnings=["Request timed out after 5 minutes"]
            )
        except Exception as e:
            self.monitor.update_query_progress(
                query_id, f"Error: {str(e)}", 100,
                warnings=[str(e)]
            )

class RAGTestingSuite:
    """Comprehensive testing suite for RAG system"""
    
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.test_queries = [
            "What is this case about?",
            "Who are the parties involved?",
            "What are the key dates mentioned?",
            "Summarize the main legal issues",
            "What evidence is mentioned?"
        ]
    
    async def run_quick_test(self, model: str = "phi3:latest") -> Dict[str, Any]:
        """Run a quick test query to verify the system is working"""
        
        test_query = "What is the main topic of these documents?"
        
        logger.info(f"Running quick test with model: {model}")
        
        try:
            # Test document search
            search_results = self.pipeline.search_documents(test_query, top_k=3)
            
            if not search_results:
                return {
                    'success': False,
                    'error': 'No search results found',
                    'stage': 'document_search'
                }
            
            # Test LLM generation
            start_time = time.time()
            result = await self.pipeline.generate_rag_answer(
                test_query, model, max_context_chunks=3, temperature=0.1
            )
            end_time = time.time()
            
            return {
                'success': True,
                'response_time': end_time - start_time,
                'chunks_found': len(search_results),
                'answer_length': len(result.get('answer', '')),
                'model_used': model,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Quick test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'llm_generation'
            }
    
    async def run_model_benchmark(self) -> Dict[str, Dict[str, Any]]:
        """Test all available models with a simple query"""
        
        benchmark_results = {}
        test_query = "Summarize the key points from these documents in one paragraph."
        
        # Get available models
        models = await self.pipeline.query_ollama_models()
        
        for model_info in models:
            model_name = model_info['name']
            logger.info(f"Testing model: {model_name}")
            
            try:
                start_time = time.time()
                result = await self.pipeline.generate_rag_answer(
                    test_query, model_name, max_context_chunks=2, temperature=0.1
                )
                end_time = time.time()
                
                benchmark_results[model_name] = {
                    'success': True,
                    'response_time': end_time - start_time,
                    'answer_length': len(result.get('answer', '')),
                    'context_chunks': result.get('context_chunks', 0),
                    'answer_preview': result.get('answer', '')[:200] + "..."
                }
                
            except Exception as e:
                benchmark_results[model_name] = {
                    'success': False,
                    'error': str(e),
                    'response_time': None
                }
        
        return benchmark_results
    
    def display_test_results(self, results: Dict[str, Any]):
        """Display test results in Streamlit"""
        
        if results['success']:
            st.success("✅ RAG System Test Passed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Time", f"{results['response_time']:.1f}s")
            with col2:
                st.metric("Chunks Found", results['chunks_found'])
            with col3:
                st.metric("Answer Length", f"{results['answer_length']} chars")
            
            with st.expander("Test Response"):
                st.write(results['result']['answer'])
                
        else:
            st.error(f"❌ RAG System Test Failed: {results['error']}")
            st.info(f"Failed at stage: {results.get('stage', 'unknown')}")

# Global instances
rag_monitor = RAGProgressMonitor()
streaming_generator = StreamingRAGGenerator(rag_monitor)

def get_rag_monitor() -> RAGProgressMonitor:
    """Get the global RAG monitor instance"""
    return rag_monitor

def get_streaming_generator() -> StreamingRAGGenerator:
    """Get the global streaming generator instance"""
    return streaming_generator 