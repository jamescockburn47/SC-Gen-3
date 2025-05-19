"""Utility functions for settlement estimation."""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

# Default template rows for damages categories
DAMAGE_TEMPLATES = {
    "economic": [
        {"item": "Loss of earnings", "claimed": 0.0},
        {"item": "Medical expenses", "claimed": 0.0},
        {"item": "Property damage", "claimed": 0.0},
    ],
    "non_economic": [
        {"item": "Pain and suffering", "claimed": 0.0},
        {"item": "Loss of amenity", "claimed": 0.0},
    ],
    "punitive": [
        {"item": "Exemplary damages", "claimed": 0.0},
    ],
}


def build_template_dataframe() -> pd.DataFrame:
    """Return a default editable DataFrame for settlement calculations."""
    rows = []
    for category, items in DAMAGE_TEMPLATES.items():
        for entry in items:
            rows.append(
                {
                    "category": category,
                    "item": entry["item"],
                    "claimed": entry.get("claimed", 0.0),
                    "liability_pct": 100.0,
                    "costs": 0.0,
                }
            )
    return pd.DataFrame(rows)


def calculate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate estimated settlement ranges for each row."""
    calc_df = df.copy()
    calc_df["claimed"] = pd.to_numeric(calc_df["claimed"], errors="coerce").fillna(0.0)
    calc_df["liability_pct"] = pd.to_numeric(calc_df["liability_pct"], errors="coerce").fillna(0.0)
    calc_df["costs"] = pd.to_numeric(calc_df["costs"], errors="coerce").fillna(0.0)
    calc_df["estimated_recovery"] = calc_df["claimed"] * calc_df["liability_pct"] / 100.0
    calc_df["lower_range"] = calc_df["estimated_recovery"] * 0.75
    calc_df["upper_range"] = calc_df["estimated_recovery"] * 1.25
    return calc_df


def generate_ai_commentary(df: pd.DataFrame) -> str:
    """Use an available AI model to provide short risk commentary."""
    try:
        from config import LOADED_PROTO_TEXT, OPENAI_MODEL_DEFAULT, get_openai_client
    except Exception as e:  # pragma: no cover - config missing
        logger.error("Config import failed for AI commentary: %s", e)
        return "AI commentary unavailable"

    client = get_openai_client()
    if not client:
        return "AI commentary unavailable"

    csv_preview = df.to_csv(index=False)
    prompt = (
        f"{LOADED_PROTO_TEXT}\n\nProvide a concise risk analysis of the following settlement data:\n{csv_preview}\n"
    )
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_DEFAULT,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("AI commentary generation failed: %s", e, exc_info=True)
        return f"Error generating commentary: {e}"


def export_dataframe(df: pd.DataFrame, path: Path, fmt: str) -> Path:
    """Export DataFrame to CSV or PDF."""
    fmt = fmt.lower()
    path = Path(path)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "pdf":
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

            doc = SimpleDocTemplate(str(path), pagesize=letter)
            data = [list(df.columns)] + df.astype(str).values.tolist()
            table = Table(data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ]
                )
            )
            doc.build([table])
        except Exception as e:
            logger.error("PDF export failed: %s", e, exc_info=True)
            raise
    else:
        raise ValueError("Unsupported export format")
    return path
