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
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` – credentials used when AWS Textract OCR is enabled.
- `AWS_DEFAULT_REGION` – AWS region (for Textract), for example `eu-west-2`.
- `S3_TEXTRACT_BUCKET` – S3 bucket name used to temporarily store PDFs when sending them to Textract.

Other optional variables (such as logging level or API retry options) can be defined as needed. See `config.py` for the full list.

## Enabling OCR with AWS Textract

Within the **Group Structure** tab of the application there is a checkbox labelled **"Use AWS Textract for PDF OCR"**. When checked, the system attempts to initialise AWS Textract using the credentials above. Scanned or image-based PDFs from Companies House will then be sent to Textract for optical character recognition before analysis.

Without OCR, many Companies House PDF filings cannot be parsed, meaning group-structure analysis may miss critical information contained in scanned documents. If OCR fails to initialise or the checkbox is left unchecked, only PDFs containing embedded text are analysed.

When subsidiaries are listed after analysis the application now aggregates entries from all retrieved years into a single deduplicated list. Previously only the most recent year's subsidiaries were shown.

Refer back to this README whenever configuring a new environment or troubleshooting OCR setup.

## Citation Verification

After each consultation the app scans the AI response for case names and statute titles. It then checks any uploaded documents for matching text and falls back to searching trusted public sources such as Bailii or Casemine. Citations that cannot be located are tagged with `[UNVERIFIED]` and a warning is shown asking you to provide the original document or a direct link. Verified citations are cached in `verified_sources.json` to speed up later runs.

## Protocol Compliance Check

Every AI response is automatically checked against the Strategic Protocols. Any non-compliance is flagged directly below the output with an expander showing the full report. You can disable this behaviour with the **Auto-check after each response** toggle in the sidebar and rerun the check manually at any time.
