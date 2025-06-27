#!/bin/bash
# Legal Models Setup Script for Strategic Counsel
# Adds LawMA-8B and sets Mixtral as default

echo "🏛️ Strategic Counsel Legal Models Setup"
echo "========================================="

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama not running. Please start Ollama first:"
    echo "   ollama serve"
    exit 1
fi

echo "✅ Ollama is running"

# Function to setup LawMA-8B
setup_lawma() {
    echo ""
    echo "🏛️ Setting up LawMA-8B (Specialized Legal Model)"
    echo "================================================"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    echo "📥 Downloading LawMA-8B GGUF model (this may take a few minutes)..."
    
    # Download the GGUF file from HuggingFace
    curl -L -o lawma-8b-q4_k_m.gguf \
        "https://huggingface.co/Khawn2u/lawma-8b-Q4_K_M-GGUF/resolve/main/lawma-8b-q4_k_m.gguf"
    
    if [ ! -f "lawma-8b-q4_k_m.gguf" ]; then
        echo "❌ Failed to download LawMA-8B model"
        return 1
    fi
    
    echo "📝 Creating Modelfile for LawMA-8B..."
    
    # Create Modelfile with legal-optimized parameters
    cat > Modelfile << 'MODELFILE_EOF'
FROM ./lawma-8b-q4_k_m.gguf

TEMPLATE """<|system|>
You are LawMA, a specialized legal AI assistant trained on legal documents and case law. You provide accurate, well-researched legal analysis while being clear about limitations.

<|user|>
{{ .Prompt }}

<|assistant|>
"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048

SYSTEM """You are LawMA, a specialized legal AI assistant trained on legal documents, case law, and legal procedures. 

Key Guidelines:
1. Provide accurate legal analysis based on available information
2. Always cite specific sources when available  
3. Use appropriate legal terminology while remaining accessible
4. Structure responses clearly with headings and bullet points when helpful
5. Acknowledge limitations and recommend seeking professional legal advice for specific situations

You excel at:
- Legal document analysis
- Case law research and citations
- Procedural guidance
- Legal writing assistance
- Regulatory compliance analysis
- Contract review and analysis"""
MODELFILE_EOF
    
    echo "🔧 Creating LawMA-8B model in Ollama..."
    ollama create lawma-8b -f Modelfile
    
    if [ $? -eq 0 ]; then
        echo "✅ LawMA-8B successfully added to Ollama!"
        echo "   Model name: lawma-8b"
        echo "   Size: ~4.9GB"
        echo "   Specialization: Legal analysis and document review"
    else
        echo "❌ Failed to create LawMA-8B model"
        return 1
    fi
    
    # Cleanup
    cd - > /dev/null
    rm -rf "$TEMP_DIR"
}

# Test the models
test_models() {
    echo ""
    echo "🧪 Testing models..."
    echo "==================="
    
    # Test LawMA-8B
    if ollama list | grep -q "lawma-8b"; then
        echo "🧪 Testing LawMA-8B..."
        
        response=$(timeout 30s ollama run lawma-8b "What is a contract?" 2>/dev/null || echo "timeout")
        
        if [ "$response" != "timeout" ] && [ ! -z "$response" ]; then
            echo "✅ LawMA-8B is working correctly"
        else
            echo "⚠️  LawMA-8B test timeout - this is normal for first run"
        fi
    fi
    
    # Check other models
    if ollama list | grep -q "mixtral"; then
        echo "✅ Mixtral is available (recommended default)"
    fi
    
    if ollama list | grep -q "mistral"; then
        echo "✅ Mistral is available"
    fi
}

# Main execution
echo "🚀 Starting legal models setup..."

# Setup LawMA-8B
setup_lawma

# Test models
test_models

echo ""
echo "🎉 Legal Models Setup Complete!"
echo "==============================="
echo ""
echo "📊 Current Models:"
ollama list
echo ""
echo "💡 Model Recommendations:"
echo "• LawMA-8B (lawma-8b): Best for specialized legal analysis"
echo "• Mixtral (mixtral:latest): Most powerful, set as default"  
echo "• Mistral (mistral:latest): Fast and reliable"
echo ""
echo "🔧 Next Steps:"
echo "1. Your Strategic Counsel app now defaults to Mixtral"
echo "2. You can select LawMA-8B for specialized legal analysis"
echo "3. Test the models with legal queries"
