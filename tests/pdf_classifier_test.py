import fitz
import importlib_resources
from fitz import Document

from pdf2data.pdf_classifier import PDF_Classifier


def test_pdf_classifier():
    """Test the PDF_Classifier class"""
    text_file_path: str = str(
        importlib_resources.files("pdf2data") / "resources" / "test.pdf"
    )
    scanned_file_path: str = str(
        importlib_resources.files("pdf2data") / "resources" / "test_scanned.pdf"
    )
    text_doc: Document = fitz.open(text_file_path)
    scanned_doc: Document = fitz.open(scanned_file_path)
    classifier_text: PDF_Classifier = PDF_Classifier(document=text_doc)
    classifier_scanned: PDF_Classifier = PDF_Classifier(document=scanned_doc)
    assert classifier_text.has_text() is True
    assert classifier_scanned.has_text() is False
