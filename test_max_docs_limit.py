import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch
import sys
import types

class TestMaxDocsLimit(unittest.TestCase):
    def setUp(self):
        # Stub external modules required by ch_pipeline and config
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
        ai_utils_mod.gpt_summarise_ch_docs = lambda *a, **k: ("",0,0)
        ai_utils_mod.gemini_summarise_ch_docs = lambda *a, **k: ("",0,0)
        sys.modules.setdefault('ai_utils', ai_utils_mod)

        spec = importlib.util.spec_from_file_location("ch_pipeline", Path(__file__).resolve().parent / "ch_pipeline.py")
        self.ch_pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ch_pipeline)

    def test_single_pipeline_limit(self):
        pipeline = self.ch_pipeline.CompanyHouseDocumentPipeline(
            company_number="12345678",
            ch_api_key="dummy"
        )
        docs = [{"transaction_id": f"tx{i}", "date": "2024-01-01"} for i in range(5)]
        with patch.object(self.ch_pipeline, "get_relevant_filings_metadata", return_value=(docs, {}, None)), \
             patch.object(pipeline, "_download_filings", return_value=[] ) as mock_dl, \
             patch.object(pipeline, "_process_documents", return_value=[]):
            self.ch_pipeline.config.MAX_DOCS_TO_PROCESS_PER_COMPANY = 3
            pipeline.run()
            self.assertEqual(len(mock_dl.call_args[0][0]), 3)

    def test_batch_limit(self):
        filings = [{"transaction_id": f"tx{i}", "date": "2024-01-01", "category": "accounts"} for i in range(4)]
        with patch.object(self.ch_pipeline, "_fetch_document_content_from_ch", return_value=({"json": "{}"}, ["json"], None)), \
             patch.object(self.ch_pipeline, "_save_raw_document_content", return_value=Path("/tmp/d")), \
             patch.object(self.ch_pipeline, "extract_text_from_document", return_value=("text", 0, None)), \
             patch.object(self.ch_pipeline, "gemini_summarise_ch_docs", return_value=("summary", 0, 0)), \
             patch.object(self.ch_pipeline, "gpt_summarise_ch_docs", return_value=("summary", 0, 0)):
            self.ch_pipeline.config.MAX_DOCS_TO_PROCESS_PER_COMPANY = 2
            _, metrics = self.ch_pipeline.run_batch_company_analysis(
                company_numbers_list=["123"],
                selected_filings_metadata_by_company={"123": filings},
                company_profiles_map={"123": {"company_name": "Test"}},
                ch_api_key_batch="dummy",
                model_prices_gbp={},
                use_textract_ocr=False,
                textract_workers=1
            )
            self.assertEqual(metrics.get("total_documents_analyzed"), 2)

if __name__ == "__main__":
    unittest.main()
