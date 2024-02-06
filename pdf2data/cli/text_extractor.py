import os
from typing import Any, Dict, List, Optional

import click

from pdf2data.mask import LayoutParser
from pdf2data.support import get_doc_list
from pdf2data.text import TextExtractor, TextFileGenerator


@click.command()
@click.argument("input_folder", type=str)
@click.option(
    "--output_folder",
    default=None,
    help="Output folder path",
)
@click.option(
    "--type",
    default="layoutparser",
    help="type of the text extractor, available: ['layoutparser', 'cermine', 'minersix']",
)
@click.option(
    "--layout_model",
    default="PubLayNet_mask_rcnn_X_101_32x8d_FPN_3x",
    help="model to use to detect the document layout",
)
@click.option(
    "--model_threshold",
    default=0.8,
    help="layout model threshold",
)
def text_extractor(input_folder: str, output_folder: Optional[str], type: str, layout_model: str, model_threshold: float) -> None:
    if output_folder is None:
        output_folder = input_folder
    elif os.path.isdir(output_folder) is False:
        os.mkdir(output_folder)
    possible_types = set(["layoutparser", "cermine", "minersix"])
    if type not in possible_types:
        raise AttributeError(
            f"{type} is not a available type, try one of the following: {possible_types}"
        )
    if type == "layoutparser":
        mask: LayoutParser = LayoutParser(
            model=layout_model, model_threshold=model_threshold
        )
        mask.model_post_init(None)
        docs: List[str] = get_doc_list(input_folder, ".pdf")
    else:
        generator: TextFileGenerator = TextFileGenerator(
            input_folder=input_folder, output_folder=output_folder
        )
        generator.model_post_init(None)
        if type == "cermine":
            generator.pdf_to_cermxml()
            docs = get_doc_list(output_folder, ".cermxml")
        elif type == "minersix":
            generator.pdf_to_miner("txt")
            docs = get_doc_list(output_folder, ".txt")
    total_docs: int = len(docs)
    doc_number: int = 1
    for doc in docs:
        print(doc)
        print(f'{doc_number}//{total_docs} processed')
        doc_number += 1
        file_name: str = os.path.splitext(doc)[0]
        if type == "layoutparser":
            file_path: str = f"{input_folder}/{doc}"
            extractor: TextExtractor = TextExtractor(
                input_file=file_path, output_folder=output_folder
            )
            layout: Dict[str, Any] = mask.get_layout(file_path, 0.5)
            extractor.extract_layoutparser(f"{file_name}_text", layout)
        else:
            file_path = f"{output_folder}/{doc}"
            extractor = TextExtractor(input_file=file_path, output_folder=output_folder)
            if type == "cermine":
                extractor.extract_cermine(f"{file_name}_text")
            elif type == "minersix":
                extractor.extract_txt(f"{file_name}_text")


def main():
    text_extractor()


if __name__ == "__main__":
    main()
