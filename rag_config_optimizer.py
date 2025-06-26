# rag_config_optimizer.py
"""
RAG System Configuration Optimizer for High-Performance Hardware
Optimized for: 8GB VRAM + 64GB RAM + 5 Ollama models
"""

import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    size_gb: float
    parameters: str
    speed: str  # "fast", "balanced", "slow"
    quality: str  # "high", "medium", "low"
    recommended_use: str
    max_context_chunks: int
    optimal_temperature: float

class RAGOptimizer:
    """Optimized RAG configuration for user's hardware setup"""
    
    def __init__(self):
        # Hardware specs
        self.vram_gb = 8
        self.ram_gb = 64
        
        # Detected models with optimization metadata
        self.available_models = {
                    "deepseek-llm:7b": ModelConfig(
            name="deepseek-llm:7b",
                size_gb=38,
                parameters="67B",
                speed="slow",
                quality="high",
                recommended_use="Complex legal analysis, detailed research, comprehensive contract review",
                max_context_chunks=8,  # Larger model can handle more context
                optimal_temperature=0.1  # Conservative for legal work
            ),
            "mixtral:latest": ModelConfig(
                name="mixtral:latest", 
                size_gb=26,
                parameters="46.7B",
                speed="balanced",
                quality="high",
                recommended_use="Legal reasoning, contract analysis, case law research",
                max_context_chunks=7,
                optimal_temperature=0.15
            ),
            "deepseek-llm:7b": ModelConfig(
                name="deepseek-llm:7b",
                size_gb=4,
                parameters="7B", 
                speed="fast",
                quality="medium",
                recommended_use="Quick summaries, document categorization, first-pass analysis",
                max_context_chunks=6,
                optimal_temperature=0.2
            ),
            "mistral:latest": ModelConfig(
                name="mistral:latest",
                size_gb=4.1,
                parameters="7B",
                speed="fast", 
                quality="medium",
                recommended_use="General Q&A, document summaries, client communications",
                max_context_chunks=6,
                optimal_temperature=0.2
            ),
            "phi3:latest": ModelConfig(
                name="phi3:latest",
                size_gb=2.2,
                parameters="3.8B",
                speed="fast",
                quality="medium",
                recommended_use="Quick queries, testing, rapid document scanning",
                max_context_chunks=5,
                optimal_temperature=0.25
            )
        }
        
        # Optimized embedding configuration for high-end hardware
        self.embedding_config = {
            "model": "all-mpnet-base-v2",  # Higher quality embedding (768 dim vs 384)
            "chunk_size": 600,  # Larger chunks for better context with abundant RAM
            "chunk_overlap": 75,  # More overlap for better retrieval
            "max_docs_per_matter": 200,  # Increased limit for high RAM
            "batch_size": 32  # Larger batches for faster processing
        }
        
        # Performance optimizations for your hardware
        self.performance_config = {
            "parallel_processing": True,
            "gpu_acceleration": True,  # Can leverage VRAM for embeddings
            "memory_optimization": False,  # Disable with 64GB RAM
            "preload_models": ["phi3:latest", "deepseek-llm:7b"],  # Keep fast models warm
            "max_concurrent_queries": 3,  # Can handle multiple with your RAM
            "vector_index_type": "IVF",  # More advanced indexing with sufficient resources
            "cache_embeddings": True  # Cache with abundant RAM
        }
    
    def get_recommended_model_by_use_case(self, use_case: str) -> str:
        """Get the best model for specific use cases"""
        recommendations = {
            "complex_analysis": "deepseek-llm:7b",
            "contract_review": "mixtral:latest", 
            "legal_research": "mixtral:latest",
            "document_summary": "deepseek-llm:7b",
            "quick_query": "phi3:latest",
            "general_qa": "mistral:latest",
            "case_law_analysis": "deepseek-llm:7b",
            "client_communication": "mistral:latest",
            "document_categorization": "deepseek-llm:7b",
            "compliance_check": "mixtral:latest"
        }
        return recommendations.get(use_case, "mistral:latest")
    
    def get_optimal_settings(self, model_name: str) -> Dict:
        """Get optimal settings for a specific model"""
        if model_name not in self.available_models:
            model_name = "mistral:latest"  # Fallback
            
        model = self.available_models[model_name]
        
        return {
            "model_name": model_name,
            "max_context_chunks": model.max_context_chunks,
            "temperature": model.optimal_temperature,
            "max_tokens": 4000 if model.size_gb > 20 else 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
    
    def get_model_categories(self) -> Dict[str, List[str]]:
        """Categorize models by use case for UI"""
        return {
            "🚀 High Performance (Complex Analysis)": [
                "deepseek-llm:7b",
                "mixtral:latest"
            ],
            "⚡ Balanced (General Use)": [
                "deepseek-llm:7b", 
                "mistral:latest"
            ],
            "🏃 Fast (Quick Queries)": [
                "phi3:latest"
            ]
        }
    
    def get_hardware_utilization_estimate(self, model_name: str) -> Dict:
        """Estimate resource usage for a model"""
        model = self.available_models.get(model_name)
        if not model:
            return {}
            
        # Rough estimates based on model size
        vram_usage = min(model.size_gb * 0.3, self.vram_gb * 0.8)  # Conservative VRAM estimate
        ram_usage = model.size_gb + 2  # Model + overhead
        
        return {
            "estimated_vram_gb": round(vram_usage, 1),
            "estimated_ram_gb": round(ram_usage, 1),
            "vram_utilization_percent": round((vram_usage / self.vram_gb) * 100, 1),
            "ram_utilization_percent": round((ram_usage / self.ram_gb) * 100, 1),
            "performance_rating": "Excellent" if ram_usage < 32 else "Good"
        }
    
    def get_concurrent_model_recommendations(self) -> List[Tuple[str, str]]:
        """Models that can run concurrently with your hardware"""
        return [
            ("phi3:latest", "deepseek-llm:7b"),  # Fast combo for multi-tasking
            ("mistral:latest", "phi3:latest"),   # Balanced + fast
            # Single large model usage
            ("deepseek-llm:7b", "mistral:latest"),  # Balanced pair
            ("mixtral:latest", "phi3:latest")    # Complex + fast backup
        ]

    def export_environment_config(self) -> str:
        """Generate optimized environment variables"""
        config = f"""
# Optimized RAG Configuration for High-End Hardware
# Hardware: 8GB VRAM, 64GB RAM, 5 Ollama Models

# Ollama Configuration
export OLLAMA_NUM_PARALLEL=3
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=300  # Keep models loaded longer

# RAG System Settings
export RAG_EMBEDDING_MODEL="{self.embedding_config['model']}"
export RAG_CHUNK_SIZE="{self.embedding_config['chunk_size']}"
export RAG_CHUNK_OVERLAP="{self.embedding_config['chunk_overlap']}"
export RAG_MAX_DOCS_PER_MATTER="{self.embedding_config['max_docs_per_matter']}"
export RAG_BATCH_SIZE="{self.embedding_config['batch_size']}"

# Performance Optimizations
export RAG_ENABLE_GPU_ACCELERATION="true"
export RAG_CACHE_EMBEDDINGS="true"
export RAG_MAX_CONCURRENT_QUERIES="{self.performance_config['max_concurrent_queries']}"
export RAG_PRELOAD_MODELS="phi3:latest,deepseek-llm:7b"

# Vector Database Settings
export FAISS_USE_GPU="false"  # CPU FAISS is sufficient
export FAISS_INDEX_TYPE="IVF"
export FAISS_NLIST="100"

# MCP Server Settings  
export MCP_MAX_CONTEXT_CHUNKS="8"
export MCP_ENABLE_ADVANCED_VALIDATION="true"
export MCP_CACHE_VALIDATION_RESULTS="true"
        """.strip()
        
        return config

# Global optimizer instance
rag_optimizer = RAGOptimizer() 