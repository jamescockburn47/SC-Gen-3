import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from dateutil import parser as dateparser

from text_extraction_utils import pdfminer_extract
from ai_utils import (
    MIN_CHARS_FOR_AI_SUMMARY,
    gpt_summarise_ch_docs,
    gemini_summarise_ch_docs,
)

logger = logging.getLogger(__name__)

Event = Dict[str, str]


def _parse_date(date_str: str) -> str:
    """Return ISO formatted date string if parsable, else empty string."""
    if not date_str:
        return ""
    try:
        dt = dateparser.parse(date_str, dayfirst=True, fuzzy=True)
        if dt:
            return dt.date().isoformat()
    except Exception:
        pass
    return ""


def _summarise_if_long(text: str) -> str:
    """Use AI summarisation for long descriptions."""
    if not text or len(text) < MIN_CHARS_FOR_AI_SUMMARY:
        return text.strip() if text else ""

    try:
        summary, _, _ = gemini_summarise_ch_docs(text, "docket_event")
        if summary and not summary.lower().startswith("error"):
            return summary.strip()
    except Exception as e:
        logger.warning("Gemini summarisation failed: %s", e)

    try:
        summary, _, _ = gpt_summarise_ch_docs(text, "docket_event")
        if summary and not summary.lower().startswith("error"):
            return summary.strip()
    except Exception as e:
        logger.warning("GPT summarisation failed: %s", e)
    return text.strip()


def parse_csv(path: Path) -> List[Event]:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = next((cols_lower[c] for c in cols_lower if "date" in c), df.columns[0])
    desc_col = next(
        (cols_lower[c] for c in cols_lower if any(word in c for word in ["description", "event", "entry", "detail"])),
        df.columns[1] if len(df.columns) > 1 else df.columns[0],
    )
    events: List[Event] = []
    for _, row in df.iterrows():
        date_val = _parse_date(str(row.get(date_col, "")))
        desc = str(row.get(desc_col, "")).strip()
        events.append({"date": date_val, "description": _summarise_if_long(desc)})
    return events


def parse_json(path: Path) -> List[Event]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("events") or list(data.values())

    events: List[Event] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                date_val = _parse_date(str(item.get("date") or item.get("event_date") or ""))
                desc = str(item.get("description") or item.get("event") or item.get("detail") or item)
                events.append({"date": date_val, "description": _summarise_if_long(desc)})
    return events


def parse_pdf(path: Path) -> List[Event]:
    text = pdfminer_extract(str(path))
    events: List[Event] = []
    date_re = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})")
    for line in text.splitlines():
        m = date_re.search(line)
        if m:
            date_val = _parse_date(m.group(1))
            desc = line[m.end():].strip() or line.strip()
            events.append({"date": date_val, "description": _summarise_if_long(desc)})
    return events


def parse_docket_file(file_path: Union[str, Path]) -> List[Event]:
    """Parse a docket CSV, JSON or PDF file into a list of events."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return parse_csv(path)
    if ext == ".json":
        return parse_json(path)
    if ext == ".pdf":
        return parse_pdf(path)
    raise ValueError(f"Unsupported docket file type: {ext}")
