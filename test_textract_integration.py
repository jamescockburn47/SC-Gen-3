import os
import sys
import unittest
import tempfile
import importlib.util
from pathlib import Path
import types


class TestConfigImport(unittest.TestCase):
    """Verify that ch_pipeline loads config relative to its file."""

    def test_config_loads_from_other_directory(self):
        root_dir = Path(__file__).resolve().parent
        pipeline_path = root_dir / "ch_pipeline.py"
        orig_cwd = os.getcwd()
        orig_sys_path = sys.path.copy()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                sys.path = [p for p in orig_sys_path if Path(p).resolve() != root_dir]
                # Stub external modules required by config.py and ch_pipeline
                sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
                openai_mod = types.ModuleType('openai'); openai_mod.OpenAI = object
                sys.modules.setdefault('openai', openai_mod)
                requests_mod = types.ModuleType('requests'); requests_mod.Session = object
                sys.modules.setdefault('requests', requests_mod)
                sys.modules.setdefault('google', types.ModuleType('google'))
                sys.modules.setdefault('google.generativeai', types.ModuleType('google.generativeai'))
                pypdf2_mod = types.ModuleType('PyPDF2'); pypdf2_mod.errors = types.SimpleNamespace(PdfReadWarning=Warning)
                sys.modules.setdefault('PyPDF2', pypdf2_mod)
                sys.modules.setdefault('PyPDF2.errors', pypdf2_mod.errors)
                sys.modules.setdefault('bs4', types.ModuleType('bs4'))
                text_ex_mod = types.ModuleType('text_extraction_utils')
                text_ex_mod.extract_text_from_document = lambda *a, **k: ("",0,None)
                text_ex_mod.OCRHandlerType = object
                sys.modules.setdefault('text_extraction_utils', text_ex_mod)
                ai_utils_mod = types.ModuleType('ai_utils')
                ai_utils_mod.gpt_summarise_ch_docs = lambda *a, **k: ""
                ai_utils_mod.gemini_summarise_ch_docs = lambda *a, **k: ""
                sys.modules.setdefault('ai_utils', ai_utils_mod)
                aws_textract_mod = types.ModuleType('aws_textract_utils')
                aws_textract_mod.perform_textract_ocr = lambda *a, **k: ("",0,None)
                aws_textract_mod.get_textract_cost_estimation = lambda *a, **k: 0
                aws_textract_mod._initialize_aws_clients = lambda *a, **k: None
                sys.modules.setdefault('aws_textract_utils', aws_textract_mod)
                ch_api_utils_mod = types.ModuleType('ch_api_utils')
                ch_api_utils_mod.get_ch_documents_metadata = lambda *a, **k: {}
                ch_api_utils_mod._fetch_document_content_from_ch = lambda *a, **k: b""
                ch_api_utils_mod.get_company_profile = lambda *a, **k: {}
                sys.modules.setdefault('ch_api_utils', ch_api_utils_mod)

                spec = importlib.util.spec_from_file_location("ch_pipeline", pipeline_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[arg-type]
                self.assertTrue(hasattr(module, "config"))
                self.assertTrue(str(module.config.__file__).endswith("config.py"))
        finally:
            os.chdir(orig_cwd)
            sys.path = orig_sys_path


if __name__ == "__main__":
    unittest.main()
