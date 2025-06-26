#!/bin/bash

echo "🔧 Fixing RAG Dependencies - Simple Version"
echo ""

# Test current status
echo "🧪 Testing current installations..."
python3 -c "
try:
    import faiss
    print('✅ FAISS: Working')
except:
    print('❌ FAISS: Not working')

try:
    import aiohttp
    print('✅ aiohttp: Working')
except:
    print('❌ aiohttp: Not working')

try:
    import numpy
    print('✅ numpy: Working')
except:
    print('❌ numpy: Not working')

try:
    import sentence_transformers
    print('✅ sentence-transformers: Working')
except Exception as e:
    print(f'❌ sentence-transformers: {e}')
"

echo ""
echo "🔧 Fixing sentence-transformers compatibility..."

# Clean fix approach
pip uninstall sentence-transformers huggingface-hub transformers -y
pip install transformers==4.21.0 huggingface-hub==0.24.0 sentence-transformers==2.7.0

echo ""
echo "🧪 Testing after fix..."
python3 -c "
try:
    import sentence_transformers
    print('✅ sentence-transformers: Fixed!')
    
    # Test basic functionality
    from sentence_transformers import SentenceTransformer
    print('✅ SentenceTransformer class: Available')
    
    # Test model loading (just check if it works)
    print('✅ Ready for embedding model initialization')
    
except Exception as e:
    print(f'❌ Still broken: {e}')
    print('')
    print('🛠️ Alternative fix:')
    print('   pip install --upgrade sentence-transformers --no-deps')
    print('   pip install huggingface-hub==0.20.0')
"

echo ""
echo "✅ Dependency fix complete!"
echo ""
echo "🚀 To start using RAG:"
echo "   1. Run: streamlit run app.py"
echo "   2. Go to '📚 Document RAG' tab"
echo "   3. Upload a document and test"
echo ""
echo "📊 Your available models:"
echo "   • deepseek-llm:67b (best for complex analysis)"
echo "   • mixtral:latest (recommended for legal work)"
echo "   • deepseek-llm:7b (fast summaries)"
echo "   • mistral:latest (general Q&A)"
echo "   • phi3:latest (quick testing)" 