import importlib_resources
import pytest
import fitz

from pdf2data.mask import LayoutParser, TableStructureParser


def test_layoutparser():
    file_path: str = str(
        importlib_resources.files("pdf2data") / "resources" / "test.pdf"
    )
    mask1 = LayoutParser(model="this is not a model")
    with pytest.raises(AttributeError):
        mask1.model_post_init(None)
    mask2 = LayoutParser(
        model="PubLayNet_faster_rcnn_R_50_FPN_3x",
        table_model="microsoft/table-transformer-detection",
    )
    mask2.model_post_init(None)
    layout = mask2.get_layout(file_path, 0.8)
    assert layout["boxes"] != 0
    assert layout["scores"] != 0
    assert layout["types"] != 0
    assert layout["page_type"] != 0

def test_table_structure():
    parser = TableStructureParser(model="microsoft/table-transformer-structure-recognition")
    parser.model_post_init(None)
    file_path: str = str(
        importlib_resources.files("pdf2data") / "resources" / "test.pdf"
    )
    document = fitz.open(file_path)
    page_index = 2
    table_box = [30.076650027310833, 86.22508175565758, 563.0077171396165, 156.38635419585557]
    table_structure = parser.get_table_structure(document, page_index, table_box)
    assert len(table_structure["collumns"]) > 4
