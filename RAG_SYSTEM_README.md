# Strategic Counsel RAG System

## Overview

The **Document RAG (Retrieval-Augmented Generation)** system in Strategic Counsel provides powerful local document analysis capabilities. It replaces the previous "Case Timeline" feature with a comprehensive AI-powered document query system.

## Key Features

### 🏠 **Fully Local Processing**
- All document processing happens locally on your machine
- No data leaves your system unless explicitly configured
- Local vector database (FAISS) for fast similarity search
- Local embedding models via sentence-transformers

### 🤖 **Multi-LLM Support**
- Integration with Ollama for local LLM inference
- Support for multiple models: Mixtral, Mistral, Phi3, DeepSeek, Llama2, and more
- Model selection per query for optimal results
- No OpenAI dependency for document queries

### 🛡️ **Protocol Enforcement (MCP Server)**
- Automated protocol compliance checking
- Citation and provenance validation
- Hallucination detection and suppression
- Output audit trail and session memory
- Query validation and response filtering

### 📚 **Advanced Document Management**
- Support for PDF, DOCX, TXT, DOC, RTF formats
- Intelligent text chunking with overlap
- OCR integration (AWS Textract or Google Drive)
- Document versioning and metadata tracking
- Automatic deduplication

### 🔍 **Intelligent Querying**
- Natural language questions about your documents
- Source citation with similarity scores
- Context chunk preview and full text access
- Advanced search parameters (temperature, chunk count)
- Session history and query analytics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Strategic Counsel UI                     │
│                  (Streamlit Interface)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP RAG Server                            │
│           (Protocol & Memory Enforcement)                   │
│  • Query Validation    • Response Audit                     │
│  • Citation Checking   • Session Memory                     │
│  • Protocol Compliance • Provenance Tracking               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Local RAG Pipeline                           │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │  Document       │  │  Vector Search  │  │  LLM         ││
│  │  Processing     │  │  (FAISS)        │  │  (Ollama)    ││
│  │                 │  │                 │  │              ││
│  │ • Text Extract  │  │ • Embeddings    │  │ • Local      ││
│  │ • Chunking      │  │ • Similarity    │  │ • Multi-model││
│  │ • OCR (opt)     │  │ • Indexing      │  │ • No API keys││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Local Storage                               │
│  • Vector Database (FAISS)                                 │
│  • Document Store                                          │
│  • Session Memory                                          │
│  • Audit Logs                                              │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The RAG system requires these additional packages:
- `faiss-cpu==1.7.4` - Vector database
- `sentence-transformers==2.2.2` - Embedding models
- `aiohttp==3.9.3` - Async HTTP for Ollama
- `numpy==1.24.3` - Vector operations

### 2. Install and Setup Ollama

#### Download Ollama
Visit [ollama.com](https://ollama.com) and download for your platform.

#### Install Local Models
```bash
# Essential models for legal work
ollama pull mixtral          # Best for complex legal analysis
ollama pull mistral          # Good balance of speed/quality
ollama pull phi3             # Fast, lightweight
ollama pull deepseek-coder   # Good for structured data
ollama pull llama2           # Fallback option

# Start Ollama service
ollama serve
```

#### Verify Ollama is Running
```bash
curl http://localhost:11434/api/tags
```

### 3. Configure OCR (Optional)

For PDF text extraction, configure one of these:

#### AWS Textract (Recommended)
```bash
# Set environment variables
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="eu-west-2"
export S3_TEXTRACT_BUCKET="your_bucket"
```

#### Google Drive OCR
```bash
# Set environment variables
export ENABLE_GOOGLE_DRIVE_INTEGRATION="true"
export GOOGLE_CLIENT_SECRET_FILE="path/to/client_secret.json"
export GOOGLE_TOKEN_FILE="path/to/google_token.json"
```

## Usage Guide

### 1. Accessing the RAG System

1. Start Strategic Counsel: `streamlit run app.py`
2. Navigate to the **"📚 Document RAG"** tab
3. Select your matter/topic in the sidebar

### 2. Document Upload & Processing

#### Upload Documents
1. Click **"📁 Upload New Documents"**
2. Select PDF, DOCX, TXT, DOC, or RTF files
3. Click **"🔄 Process and Index Documents"**
4. Monitor the progress bar as documents are processed

#### Document Processing Details
- **Text Extraction**: Automatic OCR for PDFs if configured
- **Chunking**: Documents split into ~500 word chunks with 50-word overlap
- **Embedding**: Each chunk converted to vector embeddings
- **Indexing**: Vectors stored in local FAISS database
- **Metadata**: Document info, timestamps, and chunk positions tracked

### 3. Querying Documents

#### Basic Query
1. Ensure documents are uploaded and processed
2. Select an Ollama model from the dropdown
3. Type your question in natural language
4. Click **"🧠 Generate Answer"**

#### Advanced Settings
- **Max Context Chunks**: Number of relevant chunks to use (1-10)
- **Temperature**: Controls response randomness (0.0-1.0)
- **Model Selection**: Choose optimal model for your query type

#### Example Queries
```
"What are the main financial risks mentioned in these documents?"
"Summarize the key legal obligations from the contracts"
"What deadlines or important dates are mentioned?"
"Who are the key parties involved in these agreements?"
"What are the potential liabilities discussed?"
```

### 4. Understanding Results

#### AI Response
- **Answer**: AI-generated response based on document content
- **Protocol Compliance**: MCP server validation status
- **Citations**: Source references with similarity scores

#### Source Citations
- **Document Name**: Source file for each piece of information
- **Similarity Score**: How relevant the chunk is (0-1)
- **Chunk Preview**: First 200 characters of the source text
- **Full Chunk Access**: View complete text section

#### Metadata
- **Context Chunks**: Number of document sections used
- **Token Counts**: Prompt and response token usage
- **Model Used**: Which Ollama model generated the response
- **Timestamp**: When the query was processed

### 5. Session Management

#### Session History
- **Query Tracking**: All questions and answers stored per matter
- **Model Analytics**: Track which models perform best
- **Usage Statistics**: Token counts, sources used, etc.
- **Export Options**: Download session data as CSV/JSON

#### Matter Organization
- Each matter/topic maintains separate document collections
- Independent vector databases per matter
- Isolated session histories
- Cross-matter privacy protection

## Advanced Features

### Protocol Enforcement (MCP Server)

The MCP Server provides several layers of validation:

#### Query Validation
- Minimum/maximum length requirements
- Content appropriateness checking
- Matter-specific context validation
- Audit trail logging

#### Response Compliance
- **Citation Requirements**: Ensures all answers reference sources
- **Hallucination Detection**: Flags uncertain language
- **Protocol Language**: Ensures document-grounded responses
- **Context Limits**: Prevents overuse of irrelevant chunks

#### Citation Audit
- **Provenance Tracking**: Full source chain for each fact
- **Reliability Scoring**: Confidence metrics for citations
- **Verification Status**: Automatic source validation
- **Audit Logging**: Complete compliance trail

### Performance Optimization

#### Embedding Model Selection
- **Default**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Alternative**: `all-mpnet-base-v2` (768 dimensions, accurate)
- **Multilingual**: `paraphrase-multilingual-MiniLM-L12-v2`

#### Vector Database Tuning
- **Index Type**: FAISS IndexFlatIP (cosine similarity)
- **Normalization**: L2 normalization for embeddings
- **Chunk Size**: 500 words with 50-word overlap
- **Max Documents**: 100 per matter (configurable)

#### Model Recommendations by Use Case

| Use Case | Recommended Model | Reasoning |
|----------|------------------|-----------|
| Contract Analysis | Mixtral | Best legal reasoning |
| Quick Summaries | Phi3 | Fast, efficient |
| Technical Documents | DeepSeek-Coder | Structured data handling |
| General Q&A | Mistral | Good balance |
| Complex Reasoning | Mixtral | Most capable |

## File Structure

```
SC-Gen-3/
├── local_rag_pipeline.py          # Core RAG implementation
├── mcp_rag_server.py              # Protocol enforcement server
├── rag_storage/                   # Per-matter storage
│   └── {matter_id}/
│       ├── vector_db/             # FAISS index files
│       ├── documents/             # Original document text
│       └── metadata.json          # Document/chunk metadata
├── memory/
│   └── rag_sessions/              # Session memory & audit logs
│       ├── {matter_id}_rag_memory.json
│       └── audit_log.jsonl
└── static/                        # UI assets
```

## Configuration Options

### Environment Variables

```bash
# Ollama Configuration
OLLAMA_BASE_URL="http://localhost:11434"  # Default Ollama endpoint

# RAG System Settings
RAG_CHUNK_SIZE="500"                       # Words per chunk
RAG_CHUNK_OVERLAP="50"                     # Overlap between chunks
RAG_MAX_DOCS_PER_MATTER="100"             # Document limit per matter
RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"    # Default embedding model

# MCP Server Settings
MCP_REQUIRE_CITATIONS="true"               # Enforce citation requirements
MCP_MAX_CONTEXT_CHUNKS="10"               # Maximum chunks per query
MCP_MIN_SIMILARITY_THRESHOLD="0.3"        # Minimum relevance score
MCP_HALLUCINATION_DETECTION="true"        # Enable hallucination detection

# Protocol Enforcement
PROTOCOL_CHECK_MODEL_PROVIDER="gemini"    # Protocol check model
PROTOCOL_COMPLIANCE_STRICT="false"        # Strict mode enforcement
```

### Advanced Configuration

#### Custom Embedding Models

```python
# In local_rag_pipeline.py, modify:
def __init__(self, matter_id: str, embedding_model: str = "your-model-name"):
    # Your custom embedding model
    self.embedding_model_name = embedding_model
```

#### Custom Chunking Strategy

```python
# Modify _chunk_text method for different chunking:
def _chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
    # Implement sentence-based, paragraph-based, or semantic chunking
```

#### Custom Vector Database

```python
# Alternative to FAISS (e.g., Qdrant, Chroma):
# Replace FAISS implementation in _load_vector_index and _save_vector_index
```

## Troubleshooting

### Common Issues

#### 1. "RAG system not available" Error
**Cause**: Missing dependencies or import errors
**Solution**: 
```bash
pip install faiss-cpu sentence-transformers aiohttp numpy
```

#### 2. "No Ollama models available"
**Cause**: Ollama not running or no models installed
**Solution**:
```bash
ollama serve
ollama pull mixtral
```

#### 3. "Embedding model not available"
**Cause**: sentence-transformers not installed or model download failed
**Solution**:
```bash
pip install sentence-transformers
# First query will download the model automatically
```

#### 4. OCR Not Working
**Cause**: AWS/Google credentials not configured
**Solution**: Set up credentials as per Prerequisites section

#### 5. "Vector index failed to load"
**Cause**: Corrupted FAISS index or version mismatch
**Solution**: Delete `rag_storage/{matter_id}/vector_db/` folder and reprocess documents

### Performance Issues

#### Slow Document Processing
- **Reduce chunk size**: Set `RAG_CHUNK_SIZE="300"`
- **Disable OCR**: Use `ocr_method="none"` for text documents
- **Smaller embedding model**: Use `all-MiniLM-L6-v2` instead of larger models

#### Slow Query Response
- **Reduce context chunks**: Lower `max_chunks` setting
- **Use faster model**: Switch to Phi3 instead of Mixtral
- **Check Ollama performance**: Monitor `ollama ps` for resource usage

#### Memory Usage
- **Vector database size**: Grows with number of documents/chunks
- **Embedding model**: Loaded in memory (~400MB for MiniLM)
- **Ollama models**: Each model requires 4-40GB depending on size

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("strategic_counsel_app").setLevel(logging.DEBUG)
```

Check logs in:
- `logs/` directory for application logs
- `memory/rag_sessions/audit_log.jsonl` for MCP audit trail

## Security Considerations

### Data Privacy
- **Local Processing**: All data remains on your machine
- **No External APIs**: Ollama runs locally, no cloud dependencies
- **Matter Isolation**: Each matter has separate storage and memory
- **Audit Trail**: Complete logging of all operations

### File Security
- Documents stored as plain text in `rag_storage/`
- Vector embeddings are not human-readable
- Session memory contains query summaries, not full text
- Automatic cleanup of temporary files

### Network Security
- Only local connections to Ollama (localhost:11434)
- No outbound connections except for model downloads
- Optional: Firewall rules to restrict Ollama access

## API Integration

### Programmatic Access

```python
from local_rag_pipeline import rag_session_manager
from mcp_rag_server import mcp_rag_server
import asyncio

# Initialize pipeline
pipeline = rag_session_manager.get_or_create_pipeline("my_matter")

# Add document
with open("document.pdf", "rb") as f:
    success, message, doc_info = pipeline.add_document(f, "document.pdf")

# Query documents
result = asyncio.run(pipeline.generate_rag_answer(
    "What are the key points?", 
    "mixtral", 
    max_context_chunks=5
))

print(result['answer'])
```

### REST API (Future Enhancement)

A REST API wrapper could be added for external integrations:

```python
# Potential endpoints:
POST /api/v1/matters/{matter_id}/documents    # Upload document
GET  /api/v1/matters/{matter_id}/documents    # List documents
POST /api/v1/matters/{matter_id}/query        # Query documents
GET  /api/v1/matters/{matter_id}/history      # Get session history
```

## Contributing

### Development Setup

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black isort mypy
   ```
3. Run tests:
   ```bash
   pytest tests/
   ```

### Adding New Features

#### New Embedding Models
1. Add model to `sentence-transformers` initialization
2. Update configuration options
3. Add performance benchmarks

#### New Vector Databases
1. Implement database interface in `local_rag_pipeline.py`
2. Add configuration options
3. Ensure compatibility with existing data

#### New LLM Providers
1. Add provider interface to `local_rag_pipeline.py`
2. Update model selection UI
3. Add provider-specific configuration

### Code Quality

- **Type Hints**: All functions should have type annotations
- **Error Handling**: Comprehensive try-catch blocks
- **Logging**: Use the structured logging system
- **Documentation**: Update this README for new features

## Performance Benchmarks

### Hardware Requirements

| Component | Minimum | Recommended | High Performance |
|-----------|---------|-------------|------------------|
| RAM | 8GB | 16GB | 32GB+ |
| Storage | 10GB free | 50GB free | 100GB+ SSD |
| CPU | 4 cores | 8 cores | 16+ cores |
| GPU | None | None | Optional for large models |

### Processing Speed

| Operation | Time (avg) | Notes |
|-----------|------------|-------|
| Document upload (10MB PDF) | 30-60s | Includes OCR + embedding |
| Text document (1MB) | 5-10s | No OCR required |
| Query response | 5-15s | Depends on model size |
| Vector search | <1s | FAISS is very fast |

### Scalability Limits

| Metric | Limit | Notes |
|--------|-------|-------|
| Documents per matter | 100 | Configurable limit |
| Total text size | 100MB | Per matter recommended |
| Chunks per matter | 10,000 | Performance degrades after |
| Concurrent queries | 5 | Limited by Ollama |

## License

This RAG system is part of Strategic Counsel and follows the same licensing terms as the main application.

## Support

For issues specific to the RAG system:

1. **Check Prerequisites**: Ensure Ollama and dependencies are installed
2. **Review Logs**: Check application logs and audit trail
3. **Consult Troubleshooting**: Follow the troubleshooting guide above
4. **Report Issues**: Include logs, configuration, and steps to reproduce

For general Strategic Counsel support, refer to the main application documentation. 