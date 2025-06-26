#!/bin/bash

# RAG System Setup Script - Optimized for 8GB VRAM + 64GB RAM + 5 Ollama Models
# Strategic Counsel RAG System Hardware-Optimized Installation

echo "🚀 Setting up RAG System for Strategic Counsel"
echo "Hardware: 8GB VRAM + 64GB RAM"
echo "Models: deepseek-llm:67b, mixtral:latest, deepseek-llm:7b, mistral:latest, phi3:latest"
echo ""

# Check current directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the Strategic Counsel directory (SC-Gen-3)"
    exit 1
fi

# Check if Ollama is running
echo "🔍 Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
    MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | grep -o '"name"' | wc -l)
    echo "📊 Found $MODEL_COUNT models available"
else
    echo "❌ Ollama is not running. Please start Ollama first:"
    echo "   ollama serve"
    exit 1
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies for RAG system..."

# Core RAG dependencies
pip install faiss-cpu==1.7.4
pip install sentence-transformers==2.2.2
pip install aiohttp==3.9.3
pip install numpy==1.24.3

# Check installations
echo ""
echo "🧪 Testing installations..."

python3 -c "
try:
    import faiss
    print('✅ FAISS installed successfully')
except ImportError as e:
    print(f'❌ FAISS installation failed: {e}')

try:
    import sentence_transformers
    print('✅ sentence-transformers installed successfully')
except ImportError as e:
    print(f'❌ sentence-transformers installation failed: {e}')

try:
    import aiohttp
    print('✅ aiohttp installed successfully')
except ImportError as e:
    print(f'❌ aiohttp installation failed: {e}')

try:
    import numpy
    print('✅ numpy installed successfully')
except ImportError as e:
    print(f'❌ numpy installation failed: {e}')
"

# Generate optimized configuration
echo ""
echo "⚙️ Generating optimized configuration..."

python3 -c "
from rag_config_optimizer import rag_optimizer
config = rag_optimizer.export_environment_config()
with open('rag_optimized_config.env', 'w') as f:
    f.write(config)
print('✅ Generated rag_optimized_config.env')
"

# Apply optimized settings
echo ""
echo "🎯 Applying hardware optimizations..."
source rag_optimized_config.env

# Test Ollama connectivity and model enumeration
echo ""
echo "🔗 Testing Ollama model connectivity..."

python3 -c "
import asyncio
import sys
import os
sys.path.append(os.getcwd())

async def test_models():
    try:
        from local_rag_pipeline import LocalRAGPipeline
        pipeline = LocalRAGPipeline('test_matter')
        models = await pipeline.query_ollama_models()
        
        print(f'✅ Successfully connected to Ollama')
        print(f'📊 Found {len(models)} models:')
        for model in models:
            name = model['name']
            size_mb = model.get('size', 0) / 1024 / 1024
            print(f'   • {name} ({size_mb:.1f} MB)')
            if 'recommended_use' in model:
                print(f'     └─ Use case: {model[\"recommended_use\"]}')
                print(f'     └─ Speed: {model.get(\"speed_rating\", \"N/A\")}, Quality: {model.get(\"quality_rating\", \"N/A\")}')
    except Exception as e:
        print(f'❌ Error testing models: {e}')

asyncio.run(test_models())
"

# Create directories for RAG storage
echo ""
echo "📁 Creating RAG storage directories..."
mkdir -p rag_storage
mkdir -p memory/rag_sessions
echo "✅ Storage directories created"

# Performance recommendations
echo ""
echo "🎯 HARDWARE-OPTIMIZED CONFIGURATION COMPLETE!"
echo ""
echo "🚀 PERFORMANCE RECOMMENDATIONS:"
echo ""
echo "📊 Model Selection by Use Case:"
echo "   • Complex Legal Analysis:    deepseek-llm:67b (38GB, highest quality)"
echo "   • Contract Review:           mixtral:latest (26GB, excellent reasoning)"
echo "   • Quick Summaries:           deepseek-llm:7b (4GB, fast)"
echo "   • General Questions:         mistral:latest (4.1GB, balanced)"
echo "   • Testing/Development:       phi3:latest (2.2GB, very fast)"
echo ""
echo "⚡ Optimized Settings Applied:"
echo "   • Embedding Model:           all-mpnet-base-v2 (higher quality)"
echo "   • Chunk Size:                600 words (larger for better context)"
echo "   • Chunk Overlap:             75 words (more overlap for accuracy)"
echo "   • Batch Processing:          32 chunks at once"
echo "   • Max Documents:             200 per matter (increased limit)"
echo "   • Concurrent Queries:        3 simultaneous"
echo ""
echo "💾 Memory Usage Estimates:"
echo "   • deepseek-llm:67b:          ~40GB RAM (excellent fit)"
echo "   • mixtral:latest:            ~28GB RAM (excellent fit)"
echo "   • deepseek-llm:7b:           ~6GB RAM (very fast)"
echo "   • mistral:latest:            ~6GB RAM (very fast)"
echo "   • phi3:latest:               ~4GB RAM (lightning fast)"
echo ""
echo "🚦 Starting Strategic Counsel..."
echo "   Run: streamlit run app.py"
echo "   Navigate to: 📚 Document RAG tab"
echo ""
echo "✅ RAG System Setup Complete!"
echo ""
echo "📖 Quick Start:"
echo "   1. Start Strategic Counsel: streamlit run app.py"
echo "   2. Go to '📚 Document RAG' tab"
echo "   3. Upload legal documents (PDF/DOCX/TXT)"
echo "   4. Select model (mixtral recommended for legal work)"
echo "   5. Ask questions about your documents"
echo ""
echo "🛠️ Troubleshooting:"
echo "   • If models don't appear: Check 'ollama serve' is running"
echo "   • If embeddings fail: First query downloads model (~400MB)"
echo "   • If slow performance: Use phi3 for testing, mixtral for production"
echo "   • Memory issues: Close other applications, use smaller models"
echo ""
echo "💡 Pro Tips:"
echo "   • Use deepseek-llm:67b for complex contract analysis"
echo "   • Use mixtral for legal reasoning and case law"
echo "   • Use deepseek-llm:7b for quick document summaries"
echo "   • Check 'System Information' panel for real-time stats"
echo "" 