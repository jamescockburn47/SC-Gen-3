import unittest
from unittest.mock import patch, MagicMock

class TestTextractIntegration(unittest.TestCase):

    @patch('aws_textract_utils.perform_textract_ocr')
    def test_extract_text_from_uploaded_file_textract(self, mock_perform_textract_ocr):
        # Mock the return value of perform_textract_ocr
        mock_perform_textract_ocr.return_value = ("Extracted text from PDF", 1, None)
        
        with patch('boto3.client') as mock_boto_client:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    mock_textract = MagicMock()
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")
            mock_boto_client.return_value = mock_textract
            mock_textract.detect_document_text.return_value = {'Blocks': [{'Text': 'Extracted text from PDF'}]}  # Mocking Textract response
            
            from app_utils import extract_text_from_uploaded_file
            from io import BytesIO
            
            # Simulate a PDF file object
            pdf_file = BytesIO(b'%PDF-1.4...')  # Mock PDF bytes
            file_name = 'test_document.pdf'
            
            print("Calling extract_text_from_uploaded_file...")
            text_content, error_message = extract_text_from_uploaded_file(pdf_file, file_name)
            
            print(f"Text content: {text_content}, Error message: {error_message}")
            self.assertIsNone(error_message)
            self.assertEqual(text_content, "Extracted text from PDF")

    @patch('aws_textract_utils.perform_textract_ocr')
    def test_run_ch_document_pipeline_for_company_textract(self, mock_perform_textract_ocr):
        # Mock the return value of perform_textract_ocr
        mock_perform_textract_ocr.return_value = ("Extracted text from PDF", 1, None)
        
        from ch_pipeline import run_ch_document_pipeline_for_company
        
        # Mock parameters
        company_no = '12345678'
        ch_api_key = 'test_api_key'
        selected_company_filings = [{'type': 'pdf', 'transaction_id': 'txn123', 'date': '2025-05-14'}]
        company_profile = None
        scratch_dir = '/mock/scratch/dir'
        filter_keywords = None
        use_textract = True
        
        result = run_ch_document_pipeline_for_company(company_no, ch_api_key, selected_company_filings, company_profile, scratch_dir, filter_keywords, use_textract)
        
        # Check that the result is as expected
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
