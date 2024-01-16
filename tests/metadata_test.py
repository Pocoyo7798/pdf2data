from pdf2data.metadata import Metadata
import importlib_resources

def test_pdf_file():
    file_path: str = str(
        importlib_resources.files("pdf2data") / "resources" / "test.pdf"
    )
    metadata: Metadata = Metadata(file_path=file_path)
    metadata.update()
    assert metadata.doi != None

def test_cermxml_file():
    file_path: str = str(
        importlib_resources.files("pdf2data") / "resources" / "test.cermxml"
    )
    metadata: Metadata = Metadata(file_path=file_path)
    metadata.update()
    assert metadata.journal == ["Applied Catalysis A: General"]