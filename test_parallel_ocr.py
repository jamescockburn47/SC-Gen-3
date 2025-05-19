import unittest
import time
from pathlib import Path
import tempfile
from unittest.mock import patch
import importlib.util
import sys
import types

class TestParallelOCR(unittest.TestCase):
    def test_parallel_textract_processing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = []
            for i in range(3):
                p = Path(tmpdir) / f"doc{i}.pdf"
                content = f"text{i}".encode()
                p.write_bytes(content)
                docs.append({
                    "local_path": p,
                    "saved_content_type": "pdf",
                    "original_metadata": {"transaction_id": f"tx{i}"}
                })

            sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
            openai_mod = types.ModuleType('openai'); openai_mod.OpenAI = object
            sys.modules.setdefault('openai', openai_mod)
            requests_mod = types.ModuleType('requests'); requests_mod.Session = object
            sys.modules.setdefault('requests', requests_mod)
            sys.modules.setdefault('google', types.ModuleType('google'))
            sys.modules.setdefault('google.generativeai', types.ModuleType('google.generativeai'))
            sys.modules.setdefault('bs4', types.ModuleType('bs4'))
            pdf_mod = types.ModuleType('PyPDF2'); pdf_mod.errors = types.SimpleNamespace(PdfReadWarning=Warning)
            sys.modules.setdefault('PyPDF2', pdf_mod)
            sys.modules.setdefault('PyPDF2.errors', pdf_mod.errors)
            text_ex_mod = types.ModuleType('text_extraction_utils')
            text_ex_mod.extract_text_from_document = lambda *a, **k: ("",0,None)
            text_ex_mod.OCRHandlerType = object
            sys.modules.setdefault('text_extraction_utils', text_ex_mod)
            ai_utils_mod = types.ModuleType('ai_utils')
            ai_utils_mod.gpt_summarise_ch_docs = lambda *a, **k: ""
            ai_utils_mod.gemini_summarise_ch_docs = lambda *a, **k: ""
            sys.modules.setdefault('ai_utils', ai_utils_mod)

            spec = importlib.util.spec_from_file_location('ch_pipeline', Path(__file__).resolve().parent / 'ch_pipeline.py')
            ch_pipeline_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ch_pipeline_mod)

            pipeline = ch_pipeline_mod.CompanyHouseDocumentPipeline(
                company_number="12345678",
                ch_api_key="dummy",
                use_textract_for_ocr_if_available=True,
            )

            def fake_ocr(pdf_bytes, log_id):
                time.sleep(0.2)
                return pdf_bytes.decode(), 1, None

            def fake_extract(*args, **kwargs):
                ocr_handler = kwargs.get("ocr_handler")
                pdf_bytes = kwargs.get("doc_content_input")
                company_no = kwargs.get("company_no_for_logging", "n")
                return ocr_handler(pdf_bytes, company_no)

            with patch.object(ch_pipeline_mod, "perform_textract_ocr", side_effect=fake_ocr), \
                 patch.object(ch_pipeline_mod, "extract_text_from_document", side_effect=fake_extract):
                start = time.time()
                results = pipeline._process_documents(docs, max_workers=3)
                duration = time.time() - start

            self.assertLess(duration, 0.5)
            for idx, res in enumerate(results):
                self.assertEqual(res["extracted_text"], f"text{idx}")
                self.assertEqual(res["num_pages_ocrd"], 1)
                self.assertIsNone(res["extraction_error"])

if __name__ == "__main__":
    unittest.main()
