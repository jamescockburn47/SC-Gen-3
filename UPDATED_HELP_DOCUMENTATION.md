# Strategic Counsel Gen 3 - Complete User Guide

## 🎯 System Overview

**Strategic Counsel Gen 3** is a professional AI-powered legal analysis platform featuring a **revolutionary multi-agent RAG system** that automatically orchestrates specialized AI models for comprehensive legal document analysis.

### Current System Status
- **Default Model:** Mixtral (26GB, most powerful)
- **Legal Specialist:** LawMA-8B (4.9GB, available via setup)
- **Documents Loaded:** 13 active documents
- **Vector Index:** 115 chunks with GPU acceleration
- **Enhanced Features:** ColBERT Late Interaction, Timeline Processing, Document Archiving

## 🤖 Your AI Legal Team

### **Primary Analysis Engine**
**🧠 Mixtral (`mixtral:latest`) - Default Model ⭐**
- **Size:** 26GB | **Parameters:** 46.7B
- **Role:** Primary legal analysis engine
- **Best for:** Complex legal reasoning, multi-document analysis, comprehensive case review
- **Speed:** Slower but highest quality output
- **Status:** ✅ Active as default model

### **Legal Domain Specialist**
**🏛️ LawMA-8B (`lawma-8b:latest`) - Legal Specialist**
- **Size:** 4.9GB | **Parameters:** 8B (Legal-optimized)
- **Role:** Specialized legal domain expert
- **Best for:** Legal document analysis, case law research, legal writing assistance
- **Speed:** Fast with deep legal specialization
- **Setup:** Run `./setup_legal_models.sh`

### **Supporting Models**
**⚡ Mistral (`mistral:latest`) - Balanced Performance**
- **Size:** 4.1GB | **Speed:** Fast and reliable
- **Best for:** Quick legal queries, document summaries, general Q&A

**🔬 DeepSeek-LLM (`deepseek-llm:7b`) - Analysis Specialist**
- **Size:** 4.0GB | **Speed:** Fast analytical capabilities
- **Best for:** Detailed document analysis, structured data extraction

**🏃 Phi3 (`phi3:latest`) - Quick Response**
- **Size:** 2.2GB | **Speed:** Fastest response time
- **Best for:** Quick questions, testing, simple queries

## 🚀 Getting Started

### Step 1: Document Analysis Workflow
1. **Navigate to 📚 Document RAG tab**
2. **Select/create matter** (e.g., "Client ABC Contract Review")
3. **Upload documents** (PDF, DOCX, TXT, DOC, RTF)
4. **System automatically:**
   - Processes and chunks documents
   - Creates vector embeddings with GPU acceleration
   - Enables ColBERT Late Interaction for better retrieval
5. **Ask questions** in natural language
6. **Multi-agent system:**
   - Analyzes query complexity
   - Assigns appropriate specialist agents
   - Provides comprehensive analysis with citations

### Step 2: Companies House Analysis
1. **Navigate to 🏢 Companies House tab**
2. **Enter UK company numbers**
3. **Select document types** and date ranges
4. **System retrieves and analyzes** filings with AI summaries
5. **Export results** or save to matter

### Step 3: Model Comparison
1. **Use Enhanced RAG interface**
2. **Enable comparison mode**
3. **System runs query on both:**
   - Mixtral (general powerhouse)
   - LawMA-8B (legal specialist)
4. **Compare results** with performance metrics

## 🛡️ Advanced Features

### **ColBERT Late Interaction Enhancement**
- **20-30% improvement** in document retrieval accuracy
- **Context-aware embeddings** for better legal term matching
- **Granular similarity computation** between query and document tokens
- **Status:** ✅ Active by default for all queries

### **Enhanced Timeline Processing**
- **Automatic detection** of chronological queries
- **50 chunks** retrieved vs 25 standard (doubled context)
- **600-word chunks** vs 400 standard (better context preservation)
- **Temporal relationship analysis** between events

### **Anti-Hallucination Protocol System**
- **Real-time compliance monitoring** against professional standards
- **Citation requirements:** Mandatory [Source X] format
- **Quality assurance checks:** Citation coverage, protocol language, document grounding
- **Compliance scoring:** 0-1 scale with detailed recommendations

### **Document Archiving System**
- **Automatic archiving** to `/archived_documents/` directory
- **Long-term storage** for completed matters
- **Metadata preservation** and searchability
- **Audit trail maintenance** for professional standards

## 📚 Core Workflows

### **Legal Document Analysis**
**Example Questions:**
- "What are the key risks and obligations in these contracts?"
- "Who are the parties and their responsibilities?"
- "Extract all important dates and deadlines"
- "Provide a comprehensive legal analysis"

**Process:**
1. System analyzes query complexity
2. Selects appropriate model(s) automatically
3. Retrieves relevant chunks with ColBERT enhancement
4. Generates analysis with proper citations
5. Provides compliance report and recommendations

### **Timeline Analysis**
**Example Questions:**
- "What happened chronologically in this case?"
- "Provide a timeline of events"
- "When did each party fulfill their obligations?"

**Enhanced Processing:**
- Automatic detection triggers 50-chunk processing
- Larger context windows for better chronological understanding
- Temporal relationship analysis
- Date extraction and verification

### **Model Comparison Analysis**
**When to Use:**
- Complex legal questions requiring multiple perspectives
- Quality assurance for critical analysis
- Comparing general vs. specialized legal expertise

**Process:**
1. Both models analyze simultaneously
2. Side-by-side result comparison
3. Performance metrics analysis
4. Intelligent recommendations based on query type

## 🔧 System Optimization

### **Hardware Requirements**
**Recommended:**
- 8GB+ VRAM (GPU acceleration)
- 64GB+ RAM (for Mixtral)
- 100GB+ free storage
- CUDA-compatible GPU

**Minimum:**
- 4GB VRAM (basic GPU acceleration)
- 16GB+ RAM (smaller models only)
- 50GB+ free storage
- CPU fallback available

### **Model Selection Strategy**
- **Complex legal analysis:** Mixtral (default)
- **Legal specialization:** LawMA-8B (after setup)
- **Quick queries:** Mistral or Phi3
- **Balanced performance:** DeepSeek-LLM:7b

### **Performance Optimization**
- **GPU acceleration** enabled for embeddings
- **Model caching** to keep frequently used models loaded
- **Batch processing** for multiple documents
- **Intelligent chunking** with overlap for context preservation

## 🛠️ Setup and Installation

### **1. Basic Setup**
```bash
git clone <your-repo-url>
cd SC-Gen-3
pip install -r requirements.txt
```

### **2. Install Ollama Models**
```bash
# Essential models
ollama pull mixtral      # Default model (26GB)
ollama pull mistral      # Backup model (4.1GB)
ollama pull deepseek-llm:7b
ollama pull phi3

# Optional: Legal specialist
./setup_legal_models.sh  # Automated LawMA-8B setup
```

### **3. Environment Configuration**
Create `.env` file:
```env
# Core API Keys
CH_API_KEY=your_companies_house_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key

# RAG System Optimization
RAG_EMBEDDING_MODEL=all-mpnet-base-v2
RAG_CHUNK_SIZE=600
RAG_CHUNK_OVERLAP=75
RAG_ENABLE_GPU_ACCELERATION=true
```

### **4. Launch Application**
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔧 Troubleshooting

### **Model Issues**
```bash
# Check Ollama service
ollama serve
ollama list

# Restart if needed
pkill ollama
ollama serve
```

### **Performance Issues**
```bash
# Check GPU memory
nvidia-smi

# Reduce concurrent models
export OLLAMA_MAX_LOADED_MODELS=1

# Use smaller models for testing
# Phi3: Fastest, Mistral: Balanced, Mixtral: Most powerful
```

### **Document Processing**
```bash
# Check GPU acceleration
python -c "import torch; print(torch.cuda.is_available())"

# Verify embeddings
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

## 📊 Testing and Validation

### **Test Model Connections**
```bash
python test_llm_connection.py
```

### **Model Selection Guide**
```bash
python model_selection_guide.py
```

### **System Status Check**
```bash
# Check available models
ollama list

# Test embeddings
python -c "from local_rag_pipeline import rag_session_manager; print('RAG system available')"
```

## 🎯 Professional Features

### **Legal Practice Areas**
- Commercial Litigation
- Employment Law
- Personal Injury
- Family Law
- Criminal Law
- Property/Real Estate
- Regulatory/Compliance
- Insolvency/Restructuring

### **Quality Assurance**
- Real-time protocol compliance monitoring
- Professional citation standards
- Anti-hallucination detection
- Quality scoring and recommendations

### **Export Capabilities**
- Professional DOCX reports
- Compliance documentation
- Executive summaries
- Citation-formatted analysis

## 📈 Current System Metrics

- **Documents Loaded:** 13 active documents
- **Vector Index:** 115 chunks indexed
- **GPU Acceleration:** CUDA-enabled
- **Default Model:** Mixtral (26GB, most powerful)
- **Legal Specialist:** LawMA-8B available
- **Document Archiving:** Active to `/archived_documents/`
- **ColBERT Enhancement:** Active for superior retrieval
- **Timeline Processing:** Enhanced 50-chunk analysis
- **Protocol Compliance:** Real-time monitoring active

---

**Strategic Counsel Gen 3** - Professional AI legal analysis with multi-agent intelligence, optimized for complex legal workflows with enterprise-grade features and performance.
