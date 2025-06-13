# Strategic Counsel Gen 3

## Overview
Strategic Counsel Gen 3 is an advanced legal and corporate analysis platform that leverages AI (OpenAI, Gemini), AWS Textract OCR, and Google Drive integration to automate the extraction, summarization, and verification of information from Companies House filings, court dockets, and other legal documents. The app is built with Python and Streamlit, providing an interactive, user-friendly interface for legal professionals.

## Features
- **AI-Powered Summarization & Analysis**: Uses OpenAI and Gemini models for document analysis, summaries, and protocol compliance checks.
- **Companies House Integration**: Retrieves and analyzes filings, including scanned PDFs with OCR fallback.
- **AWS Textract OCR**: Automatically processes image-based PDFs for text extraction.
- **Google Drive Integration**: Browse, select, and process files directly from your Google Drive.
- **Case Timeline Visualization**: Upload and visualize court dockets and case events.
- **Citation Verification**: Checks and verifies legal citations against uploaded documents and public sources.
- **Protocol Compliance**: Automatically checks AI outputs against strategic protocols and flags non-compliance.
- **Comprehensive Test Suite**: Includes unit and integration tests with coverage reporting.

## Core Functions

**AI Summarization & Protocol Compliance (`ai_utils.py`)**
- Provides rigorous, objective AI-powered summaries of legal and financial documents using OpenAI or Gemini models.
- Enforces a strict, factual prompt structure for extracting key financials, governance, risks, and events.
- Supports chunking and aggregation for large documents.
- Includes protocol compliance checks: every AI output is compared against a master protocol file to ensure professional standards and flag non-compliance.

**Document & Web Intake, Extraction, and Integration (`app_utils.py`)**
- Handles file uploads (PDF, DOCX, TXT) and URL ingestion, extracting text using PyPDF2, pdfminer, python-docx, and BeautifulSoup4.
- Summarizes uploaded or fetched content for quick review and context injection.
- Integrates with Google Drive for direct file selection and processing.
- Provides utility functions for extracting legal citations and verifying them against uploaded files or public sources.

**Group Structure Analysis (`group_structure_utils.py`)**
- Orchestrates the staged analysis of UK company group structures using Companies House data.
- Fetches company profiles, identifies parent/subsidiary relationships, and processes filings across multiple years.
- Prioritizes structured data (JSON, XHTML/XML) but falls back to PDF extraction and OCR when needed.
- Extracts, deduplicates, and visualizes group hierarchies, timelines, and subsidiary evolution.
- Integrates with AI summarization for objective, technical summaries of filings.

**Companies House API Integration (`ch_api_utils.py`)**
- Interfaces with the Companies House Public Data API to retrieve company profiles, filings metadata, and document content.
- Handles pagination, category filtering, and robust error handling for large-scale data pulls.
- Supports multi-format document retrieval (JSON, XHTML, PDF) and metadata extraction for downstream analysis.

**OCR & AWS Textract Integration**
- Enables advanced text extraction from scanned/image-based PDFs using AWS Textract.
- Runs OCR in parallel for large batches, with fallback logic if standard extraction fails.
- Integrates seamlessly with group structure and document analysis workflows.

**Session & Context Management**
- Maintains topic-based workspaces, session digests, and persistent memories for each matter.
- Caches document summaries and analysis results to improve performance and reduce costs.
- Provides dynamic context injection for AI consultations, ensuring highly relevant and tailored outputs.

## Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd SC_Gen3
```

### 2. Install Dependencies
#### Using pip
```bash
pip install -r requirements.txt
```
#### Or use the setup script (Unix)
```bash
./setup.sh
```
#### Windows Notes
- You may need to install Microsoft Visual Studio Build Tools if pandas or other packages are built from source: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- You can use `init_SC_Gen3_generic.ps1` or `setup_and_launch_SC_Gen3.bat` for automated setup on Windows.

### 3. Environment Variables
Create a `.env` file in the project root with the following keys (see `config.py` for all options):

```
CH_API_KEY=your_companies_house_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_FOR_SUMMARIES=gemini-1.5-flash-latest
OPENAI_MODEL=gpt-4o
GEMINI_MODEL_FOR_PROTOCOL_CHECK=gemini-1.5-pro
PROTOCOL_CHECK_MODEL_PROVIDER=gemini
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=eu-west-2
S3_TEXTRACT_BUCKET=your_s3_bucket
MAX_TEXTRACT_WORKERS=4
ENABLE_GOOGLE_DRIVE_INTEGRATION=true
GOOGLE_CLIENT_SECRET_FILE=client_secret.json
GOOGLE_TOKEN_FILE=token.json
```

- See `config.py` for additional/optional variables (logging, retries, etc).
- If using Google Drive, provide OAuth credentials and follow the first-time authorization prompt.

### 4. Run the Application
```bash
streamlit run app.py
```

## Testing

### 1. Install Test Dependencies
```bash
pip install -r tests/requirements-test.txt
```

### 2. Run All Tests with Coverage
```bash
cd tests
python run_tests.py
```
- Coverage reports are generated in the `coverage_html/` directory and as `coverage.xml`.
- You can also run individual tests with `pytest`.

## Usage Highlights
- **OCR**: Enable AWS Textract in the Group Structure tab to process scanned PDFs. The app will fallback to OCR if embedded text is missing.
- **Google Drive**: Set up integration and select files from the sidebar.
- **Case Timeline**: Upload CSV, JSON, or PDF dockets to visualize events.
- **Citation Verification**: Unverified citations are flagged and can be manually resolved.
- **Protocol Compliance**: Toggle auto-checks in the sidebar; view detailed reports for each response.

## Troubleshooting
- **Dependency Issues**: Ensure all dependencies are installed with the correct versions. Use the provided setup scripts for your OS.
- **Model/API Errors**: Verify your API keys and model names. Use helper functions in `config.py` to check model availability.
- **OCR Fails**: Check AWS credentials and S3 bucket configuration. Reduce `MAX_TEXTRACT_WORKERS` if you hit rate limits.
- **Google Drive Auth**: Ensure your OAuth credentials are correct and follow the browser prompt on first use.

## Contributing
Pull requests and issues are welcome! Please ensure all tests pass and follow the existing code style. For major changes, open an issue to discuss your proposal first.

## Support
For help, open an issue on GitHub or contact the maintainer.

---
Refer to this README whenever configuring a new environment or troubleshooting setup. For advanced configuration, see `config.py` and the in-app instructions.
