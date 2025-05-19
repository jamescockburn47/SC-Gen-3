# cross_exam_utils.py
"""Utilities for analysing witness transcripts and planning cross-examination.

This module extracts text from uploaded transcript files using
``text_extraction_utils`` and leverages ``ai_utils`` to produce a concise
summary of each witness statement together with a bullet point outline of
potential cross-examination questions.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Dict, Any

import logging

import config
from text_extraction_utils import extract_text_from_document
from ai_utils import (
    gemini_summarise_ch_docs,
    gpt_summarise_ch_docs,
    _gemini_generate_content_with_retry_and_tokens,
    DEFAULT_GEMINI_SAFETY_SETTINGS,
    genai_sdk,
)

logger = logging.getLogger(__name__)


def _read_file_content(uploaded_file: Any) -> Tuple[bytes, str]:
    """Return file bytes and extension (lower case) from a Streamlit UploadedFile."""
    file_bytes = uploaded_file.getvalue()
    name = getattr(uploaded_file, "name", "uploaded")
    ext = Path(name).suffix.lower().lstrip(".")
    return file_bytes, ext


def extract_transcript_text(uploaded_file: Any) -> Tuple[str, str | None]:
    """Extract raw text from an uploaded transcript file.

    Uses ``extract_text_from_document`` for PDFs and XHTML/XML. DOCX and TXT are
    handled with simple fallbacks.
    Returns a tuple ``(text, error_message)``.
    """
    file_bytes, ext = _read_file_content(uploaded_file)

    if ext == "pdf":
        text, _pages, err = extract_text_from_document(file_bytes, "pdf", uploaded_file.name)
        return text, err
    if ext in {"xhtml", "xml"}:
        text, _pages, err = extract_text_from_document(file_bytes.decode("utf-8", "ignore"), "xhtml", uploaded_file.name)
        return text, err
    if ext == "docx":
        try:
            from docx import Document
        except Exception as e:  # pragma: no cover - dependency missing
            logger.error("python-docx not available: %s", e)
            return "", "DOCX processing library not available"
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text)
        return text, None
    if ext in {"txt", "text"}:
        try:
            return file_bytes.decode("utf-8", "ignore"), None
        except Exception as e:
            logger.error("Failed to decode text file %s: %s", uploaded_file.name, e)
            return "", str(e)

    return "", f"Unsupported file type: {ext}"


def summarise_statement(text: str, identifier: str) -> str:
    """Return a concise summary of ``text`` using available AI models."""
    if config.GEMINI_API_KEY and genai_sdk:
        summary, _p, _c = gemini_summarise_ch_docs(text, identifier)
    else:
        summary, _p, _c = gpt_summarise_ch_docs(text, identifier)
    return summary


def generate_question_outline(text: str, identifier: str) -> str:
    """Generate a bullet point outline of cross‑examination questions."""
    prompt = (
        "You are an experienced UK barrister preparing to cross-examine a witness. "
        "Read the witness statement below and provide a concise bullet point outline "
        "of potential cross-examination questions focusing on credibility, inconsistencies "
        "and key facts. Limit to 12 points.\n\nWITNESS STATEMENT:\n---\n"
        f"{text}\n---\n\nCROSS-EXAMINATION QUESTION OUTLINE:" 
    )

    if config.GEMINI_API_KEY and genai_sdk:
        model, err = config.get_gemini_model(config.GEMINI_MODEL_DEFAULT)
        if not model:
            logger.error("Gemini model init failed: %s", err)
            return f"Error: {err}"
        gen_config = genai_sdk.types.GenerationConfig(temperature=0.3, max_output_tokens=800)
        outline, _p, _c = _gemini_generate_content_with_retry_and_tokens(
            model,
            [prompt],
            gen_config,
            DEFAULT_GEMINI_SAFETY_SETTINGS,
            identifier,
            "CrossExamOutline",
        )
        return outline

    openai_client = config.get_openai_client()
    if not openai_client:
        return "Error: No AI service available"

    response = openai_client.chat.completions.create(
        model=config.OPENAI_MODEL_DEFAULT,
        temperature=0.3,
        messages=[{"role": "system", "content": "You are an experienced UK barrister."}, {"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def analyse_uploaded_transcript(uploaded_file: Any) -> Dict[str, str | None]:
    """Extract text and return summary and question outline for ``uploaded_file``."""
    text, err = extract_transcript_text(uploaded_file)
    if not text:
        return {"error": err or "No text extracted"}

    summary = summarise_statement(text, uploaded_file.name)
    outline = generate_question_outline(text, uploaded_file.name)
    return {"summary": summary, "questions": outline, "error": err}
