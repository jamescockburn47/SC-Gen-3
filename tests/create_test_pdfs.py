from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pathlib import Path
import os

def create_sample_pdf(output_path, content, num_pages=1):
    """Create a sample PDF file with the given content"""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    
    for page in range(num_pages):
        # Add page number
        c.drawString(100, height - 100, f"Page {page + 1}")
        
        # Add content
        y = height - 150
        for line in content:
            c.drawString(100, y, line)
            y -= 20
        
        c.showPage()
    
    c.save()

def main():
    # Create test data directory if it doesn't exist
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Create sample.pdf (basic test file)
    sample_content = [
        "Sample Document for Testing",
        "This is a test document for OCR processing.",
        "It contains multiple lines of text.",
        "Testing AWS Textract and Tesseract OCR.",
        "End of sample document."
    ]
    create_sample_pdf(test_data_dir / "sample.pdf", sample_content)
    
    # Create sample2.pdf (different content)
    sample2_content = [
        "Second Sample Document",
        "This is another test document.",
        "It has different content for testing.",
        "Testing multiple document processing.",
        "End of second sample."
    ]
    create_sample_pdf(test_data_dir / "sample2.pdf", sample2_content)
    
    # Create sample3.pdf (more content)
    sample3_content = [
        "Third Sample Document",
        "This is a third test document.",
        "It contains more text for testing.",
        "Testing document comparison.",
        "End of third sample."
    ]
    create_sample_pdf(test_data_dir / "sample3.pdf", sample3_content)
    
    # Create large_sample.pdf (multiple pages)
    large_content = [
        "Large Sample Document",
        "This is a large test document with multiple pages.",
        "It contains more text for testing large document processing.",
        "Testing performance with larger files.",
        "This document has multiple pages to test pagination.",
        "Testing AWS Textract with larger documents.",
        "End of large sample document."
    ]
    create_sample_pdf(test_data_dir / "large_sample.pdf", large_content, num_pages=5)

if __name__ == "__main__":
    main() 