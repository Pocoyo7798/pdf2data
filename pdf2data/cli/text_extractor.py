import os
from typing import Any, Dict, List, Optional

import click

#from pdf2data.mask import LayoutParser
#from pdf2data.support import get_doc_list
#from pdf2data.text import TextExtractor, TextFileGenerator


@click.command()
@click.argument("input_folder", type=str)
@click.argument("output_folder", type=str)
@click.option(
    "--pipeline",
    default="MinerU", #NotDefined, MinerU, Docling, PaddlePPStructure
    help="Define the pipeline to be used",
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
@click.option(
    "--device_type",
    default="cpu",
    help="device type to run the code. Ex: 'cuda'. cpu, etc...",
)
def text_extractor(input_folder: str, output_folder: Optional[str], pipeline: str,  type: str, layout_model: str, model_threshold: float, device_type: str) -> None:
    if output_folder is None:
        output_folder = input_folder
    elif os.path.isdir(output_folder) is False:
        os.mkdir(output_folder)
    possible_types = set(["layoutparser", "cermine", "minersix"])
    if type not in possible_types:
        raise AttributeError(
            f"{type} is not a available type, try one of the following: {possible_types}"
        )
    if pipeline == "NotDefined":
        pass
        """mask: LayoutParser = LayoutParser(
            model=layout_model, model_threshold=model_threshold, device_type=device_type
        )
        mask.model_post_init(None)
        docs: List[str] = get_doc_list(input_folder, ".pdf")
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
                    extractor.extract_txt(f"{file_name}_text")"""
    elif pipeline == "MinerU":
        from pdf2data.mineru import MinerU
        miner_pipeline: MinerU = MinerU(
            input_folder=input_folder,
            output_folder=output_folder,
            extract_tables=False,
            extract_figures=False)
        miner_pipeline.pdf_transform()
    elif pipeline == "Docling":
        from pdf2data.docling import Docling
        docling_pipeline: Docling = Docling(
            input_folder=input_folder,
            output_folder=output_folder,
            extract_tables=False,
            extract_figures=False)
        docling_pipeline.pdf_transform()
    elif pipeline in ["PaddlePPStructure", "PaddleVL"]:
        from pdf2data.padle_pipeline import PaddlePPStructure
        paddle_pipeline: PaddlePPStructure = PaddlePPStructure(
            extractor_name=pipeline,
            input_folder=input_folder,
            output_folder=output_folder,
            extract_tables=False,
            extract_figures=False)
        paddle_pipeline.pdf_transform()
    


def main():
    text_extractor()


if __name__ == "__main__":
    main()
