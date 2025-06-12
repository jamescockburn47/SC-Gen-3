from __future__ import annotations

import io
from typing import List, Dict, Optional

from config import logger, get_google_drive_service

try:
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
except Exception:  # pragma: no cover - library may not be available
    MediaIoBaseDownload = None  # type: ignore
    MediaIoBaseUpload = None  # type: ignore


def list_files(service, query: str | None = None) -> List[Dict[str, str]]:
    """Return a list of files matching the given query."""
    try:
        results = service.files().list(q=query, fields="files(id,name,mimeType)").execute()
        return results.get("files", [])
    except Exception as e:  # pragma: no cover - external service
        logger.error(f"Drive list failed: {e}")
        return []


def download_file_bytes(service, file_id: str) -> Optional[bytes]:
    """Download a file's bytes from Drive."""
    if not MediaIoBaseDownload:
        logger.error("googleapiclient not available for download")
        return None
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()
        return fh.getvalue()
    except Exception as e:  # pragma: no cover - external service
        logger.error(f"Drive download failed for {file_id}: {e}")
        return None


def upload_pdf_for_ocr(service, pdf_bytes: bytes, filename: str) -> Optional[str]:
    """Upload a PDF and convert to Google Doc for OCR. Returns new file ID."""
    if not MediaIoBaseUpload:
        logger.error("googleapiclient not available for upload")
        return None
    try:
        media = MediaIoBaseUpload(io.BytesIO(pdf_bytes), mimetype="application/pdf", resumable=False)
        body = {"name": filename, "mimeType": "application/vnd.google-apps.document"}
        file = service.files().create(media_body=media, body=body, fields="id").execute()
        return file.get("id")
    except Exception as e:  # pragma: no cover - external service
        logger.error(f"Upload for OCR failed {filename}: {e}")
        return None


def extract_text_from_google_doc(service, file_id: str) -> Optional[str]:
    """Export a Google Doc as plain text."""
    try:
        data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    except Exception as e:  # pragma: no cover - external service
        logger.error(f"Export text failed for {file_id}: {e}")
        return None
