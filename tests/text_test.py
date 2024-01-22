import importlib_resources

from pdf2data.mask import LayoutParser
from pdf2data.text import TextFileGenerator, TextExtractor

def test_file_generator():
    input_folder = str(
        importlib_resources.files("pdf2data") / "resources"
    )
    generator = TextFileGenerator(input_folder=input_folder)
    generator.model_post_init(None)
    generator.pdf_to_cermxml()
    generator.pdf_to_miner("txt")
    generator.pdf_to_muhtml()

def test_text_extractor():
    file_path1 = str(
        importlib_resources.files("pdf2data") / "resources" / "test.pdf"
    )
    file_path2 = str(
        importlib_resources.files("pdf2data") / "resources" / "test.cermxml"
    )
    file_path3 = str(
        importlib_resources.files("pdf2data") / "resources" / "test.txt"
    )
    output_folder = str(
        importlib_resources.files("pdf2data") / "resources"
    )
    mask = LayoutParser(model="PubLayNet_faster_rcnn_R_50_FPN_3x")
    mask.model_post_init(None)
    layout = mask.get_layout(file_path1, 0.5)
    extractor1 = TextExtractor(input_file=file_path1, output_folder=output_folder)
    extractor2 = TextExtractor(input_file=file_path2, output_folder=output_folder)
    extractor3 = TextExtractor(input_file=file_path3, output_folder=output_folder)
    text1 = extractor1.extract_layoutparser("test_layoutparser", layout)
    text2 = extractor2.extract_cermine("test_cermine")
    text3 = extractor3.extract_txt("test_minersix")
    assert len(text1["Text"]) != 0
    assert len(text2["Type"]) != 0
    assert len(text3["Text"]) != 0


    