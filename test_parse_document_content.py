import unittest
import logging
from unittest.mock import patch

from group_structure_utils import _parse_document_content

class TestParseDocumentContentOCR(unittest.TestCase):
    def test_pdf_without_text_triggers_ocr_even_when_not_priority(self):
        doc_content = {"pdf": b"dummy"}
        fetched_types = ["pdf"]
        metadata = {"description": "some filing", "date": "2023-01-01"}
        company_no = "12345678"

        def fake_ocr(pdf_bytes, log_id):
            return "OCR TEXT", 1, None

        def fake_extract(doc_content_input, content_type_input, company_no_for_logging, ocr_handler=None):
            # Simulate no text extracted by pdfminer so OCR handler is called if provided
            if content_type_input == "pdf" and ocr_handler:
                return ocr_handler(doc_content_input, company_no_for_logging)
            return "", 0, None

        with patch("group_structure_utils.extract_text_from_document", side_effect=fake_extract):
            result = _parse_document_content(doc_content, fetched_types, company_no, metadata, logging.getLogger("test"), fake_ocr, False)

        self.assertEqual(result["text_content"], "OCR TEXT")
        self.assertEqual(result["pages_ocrd"], 1)

if __name__ == "__main__":
    unittest.main()
