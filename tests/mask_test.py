from pdf2data.mask import LayoutParser
import pytest

import importlib_resources

def layoutparser_test():
    file_path: str = str(
        importlib_resources.files("pdf2data") / "resources" / "test.pdf"
    )
    mask1 = LayoutParser(model="this is not a model")
    with pytest.raises(AttributeError):
        mask1.model_post_init(None)
    mask2 = LayoutParser(model="PubLayNet_faster_rcnn_R_50_FPN_3x", table_model="microsoft/table-transformer-detection")
    mask2.model_post_init(None)
    layout = mask2.get_layout(file_path, 0.8)
    assert layout["boxes"] != 0
    assert layout["scores"] != 0
    assert layout["types"] != 0
    assert layout["page_type"] != 0
