import os
from typing import Any, Dict, List, Optional

import click


@click.command()
@click.argument("input_folder", type=str)
@click.argument("output_folder", type=str)
@click.option(
    "--pipeline",
    default="MinerU", #NotDefined, MinerU, Docling, PaddlePPStructure
    help="Define the pipeline to be used",
)
@click.option(
    "--layout_model",
    default="PP-DocLayout-L",
    help="model to use to detect the document layout",
)
@click.option(
    "--model_threshold",
    default=0.7,
    help="layout model threshold",
)
@click.option(
    "--device",
    default="cpu",
    help="device type to run the code. Ex: 'cuda'. cpu, etc...",
)
def text_extractor(input_folder: str, output_folder: Optional[str], pipeline: str, layout_model: str, model_threshold: float, device: str) -> None:
    if output_folder is None:
        output_folder = input_folder
    elif os.path.isdir(output_folder) is False:
        os.mkdir(output_folder)
    if pipeline == "NotDefined":
        from pdf2data.pdf2data_pipeline import PDF2Data
        pdf2data_pipeline: PDF2Data = PDF2Data(layout_model=layout_model, 
                                               layout_model_threshold=model_threshold,  
                                               device=device, 
                                               input_folder=input_folder, 
                                               output_folder=output_folder,
                                               extract_tables=False,
                                               extract_figures=False)
        pdf2data_pipeline.pdf_transform()
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
    elif pipeline == "MinerUVL":
        from pdf2data.mineru_vlm import MinerUVLM
        mineru_vlm_pipeline: MinerUVLM = MinerUVLM(
            input_folder=input_folder,
            output_folder=output_folder,
            extract_tables=False,
            extract_figures=False)
        mineru_vlm_pipeline.pdf_transform()
    


def main():
    text_extractor()


if __name__ == "__main__":
    main()
