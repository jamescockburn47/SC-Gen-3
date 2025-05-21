import unittest
from unittest.mock import MagicMock
import importlib
import sys
import types


class TestGoogleDriveUtils(unittest.TestCase):
    def setUp(self):
        sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
        openai_mod = types.ModuleType('openai'); openai_mod.OpenAI = object
        sys.modules.setdefault('openai', openai_mod)
        req_mod = types.ModuleType('requests'); req_mod.Session = object
        sys.modules.setdefault('requests', req_mod)
        sys.modules.setdefault('google', types.ModuleType('google'))
        sys.modules.setdefault('google.generativeai', types.ModuleType('google.generativeai'))
        sys.modules.setdefault('bs4', types.ModuleType('bs4'))
        pdf_mod = types.ModuleType('PyPDF2'); pdf_mod.errors = types.SimpleNamespace(PdfReadWarning=Warning)
        sys.modules.setdefault('PyPDF2', pdf_mod); sys.modules.setdefault('PyPDF2.errors', pdf_mod.errors)

        import google_drive_utils as gdu_mod
        self.gdu = importlib.reload(gdu_mod)
    def test_list_files_handles_error(self):
        service = MagicMock()
        service.files().list().execute.side_effect = Exception('fail')
        files = self.gdu.list_files(service)
        self.assertEqual(files, [])

    def test_extract_text_from_google_doc(self):
        service = MagicMock()
        service.files().export().execute.return_value = b"hello"
        text = self.gdu.extract_text_from_google_doc(service, '1')
        self.assertEqual(text, 'hello')

    def test_download_bytes(self):
        service = MagicMock()
        request = object()
        service.files().get_media.return_value = request
        chunk = MagicMock()
        chunk.next_chunk.side_effect = [(None, True)]
        self.gdu.MediaIoBaseDownload = MagicMock(return_value=chunk)
        result = self.gdu.download_file_bytes(service, 'x')
        self.assertEqual(result, b"")

if __name__ == '__main__':
    unittest.main()
