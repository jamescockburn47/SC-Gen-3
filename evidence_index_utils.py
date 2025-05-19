# evidence_index_utils.py
"""Utilities for ingesting evidence documents and performing semantic search."""

from __future__ import annotations

import json
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - handled at runtime
    faiss = None  # type: ignore

from config import logger, get_openai_client
import app_utils

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

class EvidenceIndex:
    """Simple FAISS-backed evidence index."""

    def __init__(self, dir_path: str | Path):
        self.dir_path = Path(dir_path)
        self.index_file = self.dir_path / "faiss.index"
        self.meta_file = self.dir_path / "metadata.json"
        self.docs_dir = self.dir_path / "docs"
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        self.metadata: List[Dict] = []
        self.index = None
        self._load()

    def _load(self) -> None:
        if faiss is None:
            logger.error("faiss library not available. Evidence index disabled.")
            self.index = None
            self.metadata = []
            return
        if self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.index = faiss.IndexFlatL2(EMBED_DIM)
        else:
            self.index = faiss.IndexFlatL2(EMBED_DIM)
        if self.meta_file.exists():
            try:
                self.metadata = json.loads(self.meta_file.read_text())
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.metadata = []

    def _save(self) -> None:
        if self.index is None:
            return
        try:
            faiss.write_index(self.index, str(self.index_file))
            self.meta_file.write_text(json.dumps(self.metadata, indent=2))
        except Exception as e:
            logger.error(f"Error saving evidence index: {e}")

    def _embed(self, text: str) -> np.ndarray:
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client not available for embeddings")
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        vec = np.asarray(resp.data[0].embedding, dtype="float32")
        return vec

    def ingest_uploaded_file(
        self,
        uploaded_file,
        doc_type: str,
        tags: List[str] | None = None,
    ) -> Dict:
        """Extract text, summarise and index an uploaded file."""
        if faiss is None:
            raise RuntimeError("faiss library not available")
        tags = tags or []
        file_bytes = uploaded_file.getvalue()
        text, err = app_utils.extract_text_from_uploaded_file(io.BytesIO(file_bytes), uploaded_file.name)
        if err:
            logger.warning(f"{uploaded_file.name}: {err}")
        if not text:
            text = "No extractable text."
        title, summary = app_utils.summarise_with_title(text, "", uploaded_file.name)
        embedding = self._embed(summary)
        self.index.add(np.expand_dims(embedding, 0))
        doc_path = self.docs_dir / uploaded_file.name
        try:
            doc_path.write_bytes(file_bytes)
        except Exception as e:
            logger.error(f"Failed to save uploaded file {uploaded_file.name}: {e}")
        meta = {
            "path": str(doc_path),
            "file_name": uploaded_file.name,
            "doc_type": doc_type,
            "tags": tags,
            "title": title,
            "summary": summary,
        }
        self.metadata.append(meta)
        self._save()
        return meta

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Tuple[float, Dict]]:
        if faiss is None or self.index is None or self.index.ntotal == 0:
            return []
        embedding = self._embed(query)
        distances, indices = self.index.search(np.expand_dims(embedding, 0), min(top_k, self.index.ntotal))
        results: List[Tuple[float, Dict]] = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata[idx]
            if doc_type and meta.get("doc_type") != doc_type:
                continue
            if tags and not set(tags).issubset(set(meta.get("tags", []))):
                continue
            results.append((dist, meta))
        return results
