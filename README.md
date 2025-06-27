# Strategic Counsel Gen 4

## Overview
Strategic Counsel Gen 4 is an advanced AI-powered legal analysis platform featuring **LawMA legal specialist reranking** and **multi-agent RAG (Retrieval Augmented Generation)** with automatic model orchestration. The system intelligently deploys multiple specialized AI models including LawMA-8B for superior legal relevance filtering to provide comprehensive legal document analysis, with enhanced timeline processing, document archiving, and protocol compliance checking.

## 🚀 Key Features

### **🤖 Multi-Agent RAG System**
- **5 Specialized AI Models** working in parallel
- **Automatic task assignment** based on query complexity
- **Mixtral as default** (26GB, most powerful legal analysis)
- **LawMA-8B integration** (4.9GB legal specialist model)
- **Model comparison mode** for side-by-side analysis

### **⚖️ Advanced Legal Intelligence**
- **Enhanced timeline detection** with 50-chunk processing vs 25 standard
- **ColBERT Late Interaction** enabled by default for superior retrieval
- **Anti-hallucination protocols** with real-time compliance checking
- **Document archiving** to `/archived_documents/` directory
- **Citation verification** and legal reference validation

### **🏢 Corporate Analysis Integration**
- **Companies House API** integration for UK corporate data
- **Group structure mapping** and visualization
- **AWS Textract OCR** for scanned document processing
- **Google Drive integration** for seamless file access

### **🛡️ Professional Grade Features**
- **Protocol compliance reporting** with scoring and recommendations
- **Matter-based document organization** with persistent storage
- **Professional DOCX export** capabilities
- **GPU acceleration** for embedding models
- **Comprehensive audit logging**

## 🎯 Available AI Models

### **Recommended Usage Hierarchy:**

1. **🧠 Mixtral (`mixtral:latest`)** - *Default* ⭐
   - **Size:** 26GB | **Parameters:** 46.7B
   - **Best for:** Complex legal reasoning, multi-document analysis, comprehensive cases
   - **Speed:** Slower but highest quality
   - **Use when:** Primary legal analysis requiring deep reasoning

2. **🏛️ LawMA-8B (`lawma-8b:latest`)** - *Legal Specialist*
   - **Size:** 4.9GB | **Parameters:** 8B (Legal-optimized)
   - **Best for:** Specialized legal analysis, case law research, legal writing
   - **Speed:** Fast with legal domain expertise
   - **Use when:** Need deep legal specialization

3. **⚡ Mistral (`mistral:latest`)** - *Balanced*
   - **Size:** 4.1GB | **Parameters:** 7B
   - **Best for:** Quick legal queries, document summaries, general Q&A
   - **Speed:** Fast and reliable
   - **Use when:** Balanced speed/quality for straightforward questions

4. **🔬 DeepSeek-LLM (`deepseek-llm:7b`)** - *Analysis*
   - **Size:** 4.0GB | **Parameters:** 7B
   - **Best for:** Detailed document analysis, structured extraction
   - **Speed:** Fast with good analytical capabilities
   - **Use when:** Detailed analysis and data extraction

5. **🏃 Phi3 (`phi3:latest`)** - *Quick Response*
   - **Size:** 2.2GB | **Parameters:** 3.8B
   - **Best for:** Quick questions, testing, simple queries
   - **Speed:** Fastest response time
   - **Use when:** Rapid iteration and simple questions

## 🛠️ Setup

### 1. Clone and Install Dependencies
```bash
git clone <your-repo-url>
cd SC-Gen-4
pip install -r requirements.txt
```

### 2. Install and Configure Ollama
```bash
# Install Ollama (visit ollama.com for platform-specific instructions)

# Pull required models
ollama pull mixtral      # Default model (26GB)
ollama pull mistral      # Backup model (4.1GB)
ollama pull deepseek-llm:7b
ollama pull phi3

# Optional: Add LawMA-8B legal specialist
./setup_legal_models.sh  # Automated setup script
```

### 3. Environment Configuration
Create `.env` file in project root:

```env
# Core API Keys
CH_API_KEY=your_companies_house_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key

# Model Configuration (optimized defaults)
GEMINI_MODEL_FOR_SUMMARIES=gemini-1.5-flash-latest
OPENAI_MODEL=gpt-4o
PROTOCOL_CHECK_MODEL_PROVIDER=gemini

# AWS Textract (optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=eu-west-2
S3_TEXTRACT_BUCKET=your_s3_bucket
MAX_TEXTRACT_WORKERS=4

# Google Drive (optional)
ENABLE_GOOGLE_DRIVE_INTEGRATION=true
GOOGLE_CLIENT_SECRET_FILE=client_secret.json
GOOGLE_TOKEN_FILE=token.json

# RAG System Optimization
RAG_EMBEDDING_MODEL=all-mpnet-base-v2
RAG_CHUNK_SIZE=600
RAG_CHUNK_OVERLAP=75
RAG_MAX_DOCS_PER_MATTER=200
RAG_BATCH_SIZE=32
RAG_ENABLE_GPU_ACCELERATION=true
```

### 4. Launch Application
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📚 Core Workflows

### **Document Analysis Workflow**
1. **Navigate to 📚 Document RAG tab**
2. **Select/create matter** (e.g., "Client ABC Contract Review")
3. **Upload documents** (PDF, DOCX, TXT, DOC, RTF)
4. **System automatically**:
   - Processes and chunks documents
   - Creates vector embeddings with GPU acceleration
   - Enables ColBERT Late Interaction for better retrieval
5. **Ask questions** in natural language
6. **Multi-agent system**:
   - Analyzes query complexity
   - Assigns appropriate specialist agents
   - Provides comprehensive analysis with citations

### **Companies House Analysis**
1. **Navigate to 🏢 Companies House tab**
2. **Enter UK company numbers**
3. **Select document types** and date ranges
4. **System retrieves and analyzes** filings with AI summaries
5. **Export results** or save to matter

### **Model Comparison Analysis**
1. **Use Enhanced RAG interface**
2. **Enable comparison mode**
3. **System runs query on both**:
   - Mixtral (general powerhouse)
   - LawMA-8B (legal specialist)
4. **Compare results** with performance metrics and recommendations

## 🎯 Current System Status

### **Performance Metrics:**
- **Documents Loaded:** 13 active documents
- **Vector Index:** 115 chunks indexed
- **GPU Acceleration:** CUDA-enabled for embeddings
- **Default Model:** Mixtral (26GB, most powerful)
- **Specialized Model:** LawMA-8B available
- **Document Archiving:** Enabled to `/archived_documents/`

### **Advanced Features Active:**
- ✅ **ColBERT Late Interaction** for superior document retrieval
- ✅ **Enhanced Timeline Processing** (50 chunks vs 25 standard)
- ✅ **Anti-hallucination protocols** with compliance scoring
- ✅ **Multi-agent orchestration** with automatic task assignment
- ✅ **Document archiving** system for long-term storage
- ✅ **Protocol compliance checking** with detailed reporting

## 🚦 Quick Start Guide

### **For Legal Document Analysis:**
```
1. Go to 📚 Document RAG → Upload legal documents
2. Ask: "What are the key risks and obligations in these contracts?"
3. Watch multi-agent system automatically analyze and synthesize
4. Review comprehensive results with source citations
```

### **For Company Research:**
```
1. Go to 🏢 Companies House → Enter company number
2. Select document types (accounts, filings, etc.)
3. Get AI-powered analysis of corporate structure and filings
4. Export or save results to your matter
```

### **For Model Comparison:**
```
1. Use Enhanced RAG interface → Enable comparison mode
2. Ask complex legal question
3. Get side-by-side analysis from Mixtral vs LawMA-8B
4. Review performance metrics and model recommendations
```

## 🔧 System Optimization

### **Hardware Requirements:**
- **Recommended:** 8GB+ VRAM, 64GB+ RAM for optimal performance
- **Minimum:** 4GB VRAM, 16GB RAM (use smaller models)

### **Model Selection Tips:**
- **Complex legal analysis:** Mixtral (default)
- **Legal specialization:** LawMA-8B (after setup)
- **Quick queries:** Mistral or Phi3
- **Balanced performance:** DeepSeek-LLM:7b

### **Performance Optimization:**
- **GPU acceleration** enabled for embeddings
- **Model caching** to keep frequently used models loaded
- **Batch processing** for multiple documents
- **Intelligent chunking** with overlap for context preservation

## 📊 Testing and Validation

### **Run Test Suite:**
```bash
cd tests
python run_tests.py
```

### **Test Model Connections:**
```bash
python test_llm_connection.py
```

### **Validate RAG System:**
```bash
python model_selection_guide.py
```

## 🛠️ Troubleshooting

### **Common Issues:**

**Models not appearing:**
```bash
# Check Ollama service
ollama serve
ollama list  # Verify models are installed
```

**Performance issues:**
```bash
# Use faster models for testing
# Mixtral: Most powerful but slower
# Mistral: Balanced performance
# Phi3: Fastest for simple queries
```

**Memory issues:**
```bash
# Reduce concurrent models
export OLLAMA_MAX_LOADED_MODELS=1
# Use smaller models
# Close other applications
```

**Document processing errors:**
```bash
# Check GPU acceleration
nvidia-smi  # Verify GPU availability
# Fallback to CPU embeddings if needed
```

## 📈 Advanced Configuration

### **Multi-Agent Settings:**
- Automatic agent selection based on query complexity
- Concurrent model execution for comprehensive analysis
- Result synthesis from multiple specialized agents

### **RAG Optimization:**
- ColBERT Late Interaction for 20-30% retrieval improvement
- Enhanced timeline detection with 600-word chunks
- GPU-accelerated embeddings with all-mpnet-base-v2

### **Document Management:**
- Automatic archiving to preserve storage
- Matter-based organization with persistent metadata
- Professional export capabilities

## 🤝 Support

- **GitHub Issues:** Report bugs and request features
- **Documentation:** Comprehensive help system in-app
- **Model Guide:** Run `python model_selection_guide.py` for selection help

---

**Strategic Counsel Gen 4** - Professional AI legal analysis with LawMA legal specialist integration and multi-agent intelligence, optimized for complex legal workflows with enterprise-grade features and performance.
