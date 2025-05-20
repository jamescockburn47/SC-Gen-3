# Strategic Counsel Gen 3

## Setup Overview

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Create a `.env` file** in the project root containing the required environment variables.
3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Required Environment Variables

The application loads configuration from a `.env` file or the host environment. At a minimum, set the following keys:

- `CH_API_KEY` – Companies House API key for retrieving filings.
- `OPENAI_API_KEY` – API key for GPT models (used for summarisation and drafting).
- `GEMINI_API_KEY` – API key for Google Gemini models.
- `GEMINI_MODEL_FOR_SUMMARIES` – override the Gemini model used for summaries (default `gemini-1.5-flash-latest`).
- `OPENAI_MODEL` – default OpenAI model for analysis tasks and fallback summaries.
-   If you encounter errors when using a newer model name like `gpt-4.1`, run
    the helper `check_openai_model()` from `config.py` or `openai.models.list()`
    to verify the model is available to your API key.
- `GEMINI_MODEL_FOR_PROTOCOL_CHECK` – Gemini model used when checking responses against the strategic protocols.
- `PROTOCOL_CHECK_MODEL_PROVIDER` – choose `gemini` (default) or `openai` for protocol compliance checks.
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` – credentials used when AWS Textract OCR is enabled.
- `AWS_DEFAULT_REGION` – AWS region (for Textract), for example `eu-west-2`.
- `S3_TEXTRACT_BUCKET` – S3 bucket name used to temporarily store PDFs when sending them to Textract.
- `MAX_TEXTRACT_WORKERS` – number of concurrent Textract OCR workers (default `4`).

Gemini is the preferred model for generating summaries. If an OpenAI API key is provided, GPT models are used for analysis tasks and can power protocol checks when `PROTOCOL_CHECK_MODEL_PROVIDER` is set to `openai`.

Other optional variables (such as logging level or API retry options) can be defined as needed. See `config.py` for the full list.

## Enabling OCR with AWS Textract

Within the **Group Structure** tab of the application there is a checkbox labelled **"Use AWS Textract for PDF OCR"**. When checked, the system attempts to initialise AWS Textract using the credentials above. Scanned or image-based PDFs from Companies House will then be sent to Textract for optical character recognition before analysis.

If pdfminer extracts little or no text from a PDF, the system now automatically falls back to Textract whenever it is enabled. This means even filings that aren't explicitly flagged for OCR will still be processed when embedded text is missing.

Textract calls can now run in parallel to speed up large batches. The default maximum number of concurrent OCR workers is controlled by the `MAX_TEXTRACT_WORKERS` environment variable (default `4`). Reduce this value if you hit AWS rate limits.

Without OCR, many Companies House PDF filings cannot be parsed, meaning group-structure analysis may miss critical information contained in scanned documents. If OCR fails to initialise or the checkbox is left unchecked, only PDFs containing embedded text are analysed.

When subsidiaries are listed after analysis the application now aggregates entries from all retrieved years into a single deduplicated list. Previously only the most recent year's subsidiaries were shown.

Companies House may label some group accounts filings as "legacy". The system now automatically highlights these documents when presenting the list of available filings.


Refer back to this README whenever configuring a new environment or troubleshooting OCR setup.

## Case Timeline

The **📅 Case Timeline** tab lets you upload court docket files in CSV, JSON or PDF format. Dates and descriptions are extracted and displayed chronologically. Long descriptions are summarised using the same AI routines as the Companies House analysis. If the optional `streamlit_timeline` component is installed the events are shown on an interactive timeline, otherwise a simple table is displayed.

## Citation Verification

After each consultation the app scans the AI response for case names and statute titles. It then checks any uploaded documents for matching text and falls back to searching trusted public sources such as Bailii or Casemine. Citations that cannot be located are tagged with `[UNVERIFIED]` and a warning is shown asking you to provide the original document or a direct link. A form appears letting you enter a URL or reference for each citation and these links are stored in `verified_sources.json` for later checking. Verified citations are cached in the same file to speed up later runs.

## Protocol Compliance Check

Every AI response is automatically checked against the Strategic Protocols. Any non-compliance is flagged directly below the output with an expander showing the full report. You can disable this behaviour with the **Auto-check after each response** toggle in the sidebar and rerun the check manually at any time.
