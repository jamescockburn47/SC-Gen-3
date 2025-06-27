"""
Enhanced PDF Reader for Legal Documents
=======================================

Advanced PDF text extraction using pdfplumber with OCR fallback.
Maintains compatibility with existing Strategic Counsel OCR preferences.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import io
from pathlib import Path

# PDF processing imports with fallbacks
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    PyPDF2 = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    pdfminer_extract = None

# OCR imports (optional)
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    Image = None

from legal_rag import logger

class LegalPDFReader:
    """
    Enhanced PDF reader specifically designed for legal documents.
    
    Features:
    - pdfplumber for better table and layout preservation
    - Multiple extraction methods with fallbacks
    - OCR for scanned documents
    - Legal document structure preservation
    - Compatibility with existing Strategic Counsel system
    """
    
    def __init__(self, prefer_ocr: bool = False, ocr_preference: str = "local"):
        """
        Initialize PDF reader.
        
        Args:
            prefer_ocr: Whether to prefer OCR even for text PDFs
            ocr_preference: OCR preference ('local', 'aws', 'tesseract')
        """
        self.prefer_ocr = prefer_ocr
        self.ocr_preference = ocr_preference
        
        # Track available methods
        self.available_methods = {
            'pdfplumber': PDFPLUMBER_AVAILABLE,
            'pypdf2': PYPDF2_AVAILABLE,
            'pdfminer': PDFMINER_AVAILABLE,
            'tesseract': TESSERACT_AVAILABLE
        }
        
        logger.info(f"PDF Reader initialized - Available methods: {self.available_methods}")
    
    def extract_with_pdfplumber(self, file_obj: io.BytesIO) -> Tuple[str, Optional[str]]:
        """
        Extract text using pdfplumber (best for legal documents with tables).
        
        Args:
            file_obj: PDF file object
            
        Returns:
            Tuple of (extracted_text, error_message)
        """
        if not PDFPLUMBER_AVAILABLE:
            return "", "pdfplumber not available"
        
        try:
            file_obj.seek(0)
            extracted_text = ""
            
            with pdfplumber.open(file_obj) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text while preserving structure
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Add page marker for legal citations
                        extracted_text += f"\n[Page {page_num}]\n{page_text}\n"
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            for table_num, table in enumerate(tables, 1):
                                extracted_text += f"\n[Table {table_num} on Page {page_num}]\n"
                                for row in table:
                                    if row:
                                        row_text = " | ".join([cell or "" for cell in row])
                                        extracted_text += f"{row_text}\n"
                                extracted_text += "\n"
            
            if not extracted_text.strip():
                return "", "No text extracted with pdfplumber"
            
            logger.info(f"✅ pdfplumber extracted {len(extracted_text)} characters")
            return extracted_text.strip(), None
            
        except Exception as e:
            error_msg = f"pdfplumber extraction failed: {str(e)}"
            logger.warning(error_msg)
            return "", error_msg
    
    def extract_with_pypdf2(self, file_obj: io.BytesIO) -> Tuple[str, Optional[str]]:
        """
        Extract text using PyPDF2 (fallback method).
        
        Args:
            file_obj: PDF file object
            
        Returns:
            Tuple of (extracted_text, error_message)
        """
        if not PYPDF2_AVAILABLE:
            return "", "PyPDF2 not available"
        
        try:
            file_obj.seek(0)
            pdf_reader = PyPDF2.PdfReader(file_obj)
            extracted_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    extracted_text += f"\n[Page {page_num}]\n{page_text}\n"
            
            if not extracted_text.strip():
                return "", "No text extracted with PyPDF2"
            
            logger.info(f"📦 PyPDF2 extracted {len(extracted_text)} characters")
            return extracted_text.strip(), None
            
        except Exception as e:
            error_msg = f"PyPDF2 extraction failed: {str(e)}"
            logger.warning(error_msg)
            return "", error_msg
    
    def extract_with_pdfminer(self, file_obj: io.BytesIO) -> Tuple[str, Optional[str]]:
        """
        Extract text using pdfminer (another fallback).
        
        Args:
            file_obj: PDF file object
            
        Returns:
            Tuple of (extracted_text, error_message)
        """
        if not PDFMINER_AVAILABLE:
            return "", "pdfminer not available"
        
        try:
            file_obj.seek(0)
            extracted_text = pdfminer_extract(file_obj)
            
            if not extracted_text.strip():
                return "", "No text extracted with pdfminer"
            
            logger.info(f"⚙️ pdfminer extracted {len(extracted_text)} characters")
            return extracted_text.strip(), None
            
        except Exception as e:
            error_msg = f"pdfminer extraction failed: {str(e)}"
            logger.warning(error_msg)
            return "", error_msg
    
    def extract_with_ocr(self, file_obj: io.BytesIO) -> Tuple[str, Optional[str]]:
        """
        Extract text using OCR (for scanned documents).
        
        Args:
            file_obj: PDF file object
            
        Returns:
            Tuple of (extracted_text, error_message)
        """
        # This is a placeholder for OCR integration
        # In practice, this would integrate with:
        # - AWS Textract (if ocr_preference == "aws")
        # - Local Tesseract (if ocr_preference == "tesseract")
        # - Existing Strategic Counsel OCR system
        
        if self.ocr_preference == "aws":
            return self._extract_with_aws_textract(file_obj)
        elif self.ocr_preference == "tesseract":
            return self._extract_with_tesseract(file_obj)
        else:
            return self._extract_with_existing_ocr(file_obj)
    
    def _extract_with_aws_textract(self, file_obj: io.BytesIO) -> Tuple[str, Optional[str]]:
        """Integrate with existing AWS Textract functionality."""
        try:
            # Import existing Strategic Counsel AWS utilities
            from app_utils import extract_text_from_uploaded_file
            
            file_obj.seek(0)
            text, error = extract_text_from_uploaded_file(file_obj, "document.pdf", "aws")
            
            if text:
                logger.info(f"🔍 AWS Textract extracted {len(text)} characters")
                return text, None
            else:
                return "", error or "AWS Textract failed"
                
        except ImportError:
            return "", "AWS Textract not available (app_utils not found)"
        except Exception as e:
            return "", f"AWS Textract failed: {str(e)}"
    
    def _extract_with_tesseract(self, file_obj: io.BytesIO) -> Tuple[str, Optional[str]]:
        """Extract using local Tesseract OCR."""
        if not TESSERACT_AVAILABLE:
            return "", "Tesseract not available"
        
        try:
            # This would implement PDF to image conversion + Tesseract
            # Placeholder for now
            return "", "Tesseract OCR not yet implemented"
            
        except Exception as e:
            return "", f"Tesseract OCR failed: {str(e)}"
    
    def _extract_with_existing_ocr(self, file_obj: io.BytesIO) -> Tuple[str, Optional[str]]:
        """Use existing Strategic Counsel OCR system."""
        try:
            # Import existing Strategic Counsel OCR functionality
            from app_utils import extract_text_from_uploaded_file
            
            file_obj.seek(0)
            text, error = extract_text_from_uploaded_file(file_obj, "document.pdf", "local")
            
            if text:
                logger.info(f"🏠 Local OCR extracted {len(text)} characters")
                return text, None
            else:
                return "", error or "Local OCR failed"
                
        except ImportError:
            return "", "Existing OCR not available (app_utils not found)"
        except Exception as e:
            return "", f"Existing OCR failed: {str(e)}"


def extract_text_with_fallback(
    file_obj: io.BytesIO,
    filename: str = "document.pdf",
    ocr_preference: str = "local",
    prefer_ocr: bool = False
) -> Tuple[str, Optional[str]]:
    """
    Extract text from PDF with multiple fallback methods.
    
    This is the main function that should be used for PDF text extraction.
    Maintains compatibility with existing Strategic Counsel system.
    
    Args:
        file_obj: PDF file object
        filename: Original filename
        ocr_preference: OCR preference ('local', 'aws', 'tesseract')
        prefer_ocr: Whether to prefer OCR even for text PDFs
        
    Returns:
        Tuple of (extracted_text, error_message)
    """
    reader = LegalPDFReader(prefer_ocr=prefer_ocr, ocr_preference=ocr_preference)
    
    # Define extraction methods in order of preference
    extraction_methods = []
    
    if prefer_ocr:
        # OCR first if preferred
        extraction_methods.append(("OCR", reader.extract_with_ocr))
        extraction_methods.append(("pdfplumber", reader.extract_with_pdfplumber))
        extraction_methods.append(("PyPDF2", reader.extract_with_pypdf2))
        extraction_methods.append(("pdfminer", reader.extract_with_pdfminer))
    else:
        # Text extraction first, OCR as fallback
        extraction_methods.append(("pdfplumber", reader.extract_with_pdfplumber))
        extraction_methods.append(("PyPDF2", reader.extract_with_pypdf2))
        extraction_methods.append(("pdfminer", reader.extract_with_pdfminer))
        extraction_methods.append(("OCR", reader.extract_with_ocr))
    
    # Try each method until one succeeds
    last_error = None
    for method_name, method_func in extraction_methods:
        try:
            text, error = method_func(file_obj)
            
            if text and text.strip():
                logger.info(f"✅ Successfully extracted text using {method_name}")
                return text, None
            elif error:
                last_error = error
                logger.debug(f"❌ {method_name} failed: {error}")
                
        except Exception as e:
            last_error = f"{method_name} exception: {str(e)}"
            logger.debug(last_error)
    
    # All methods failed
    error_msg = f"All extraction methods failed for {filename}. Last error: {last_error}"
    logger.error(error_msg)
    return "", error_msg


# Backward compatibility function
def extract_text_from_pdf(file_obj: io.BytesIO, ocr_preference: str = "local") -> Tuple[str, str]:
    """
    Backward compatibility function for existing Strategic Counsel system.
    
    Returns:
        Tuple of (text, error_message) - matches existing interface
    """
    text, error = extract_text_with_fallback(file_obj, ocr_preference=ocr_preference)
    return text, error or "" 