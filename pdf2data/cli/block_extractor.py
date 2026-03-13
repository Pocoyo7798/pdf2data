import os
from typing import List, Optional

import click


@click.command()
@click.argument("input_folder", type=str)
@click.argument("output_folder", type=str)
@click.option(
    "--pipeline",
    default="MinerU", #NotDefined, MinerU
    help="Define the pipeline to be used",
)
@click.option(
    "--layout_model",
    default="DocLayout-YOLO-DocStructBench",
    help="model used to determine the overall layout",
)
@click.option(
    "--table_model",
    default=None,
    help="model used to identify the tables",
)
@click.option(
    "--layout_model_threshold",
    default=0.6,
    help="threshold of the layout model",
)
@click.option(
    "--table_model_threshold",
    default=0.6,
    help="threshold of the table detection model",
)
@click.option(
    "--struct_model",
    default="microsoft/table-structure-recognition-v1.1-all",
    help="table structure detection model",
)
@click.option(
    "--device",
    default="cpu",
    help="device to run the mask models",
)
def block_extractor(
    input_folder: str,
    output_folder: str,
    pipeline: Optional[str],
    layout_model: str,
    table_model: str,
    layout_model_threshold: float,
    table_model_threshold: float,
    struct_model: str,
    device: str,
) -> None:
    if pipeline == "NotDefined":
        from pdf2data.pdf2data_pipeline import PDF2Data
        pdf2data_pipeline: PDF2Data = PDF2Data(layout_model=layout_model, 
                                                layout_model_threshold=layout_model_threshold,
                                                table_model=table_model,
                                                table_model_threshold=table_model_threshold,
                                                table_structure_model=struct_model,
                                                device=device, 
                                                input_folder=input_folder, 
                                                output_folder=output_folder,
                                                extract_text=False,
                                                extract_equations=False)
        pdf2data_pipeline.pdf_transform()
        """file_list: List[str] = get_doc_list(input_folder, "pdf")
        extractor: BlockExtractor = BlockExtractor(
            ocr_model=ocr_model,
            word_detection_model=word_detection_model,
            word_detection_threshold=word_detection_model_threshold,
            extract_tables=extract_tables,
            extract_figures=extract_figures,
            correct_struct=correct_struct,
            table_zoom=table_zoom,
            cell_zoom=cell_zoom,
            figure_zoom=figure_zoom,
            x_table_corr=x_table_corr,
            y_table_corr=y_table_corr,
            iou_lines=iou_lines,
            iou_struct=iou_struct,
            word_factor=word_factor,
            word_iou=word_iou,
            structure_model=struct_model,
            struct_model_threshold=struct_model_threshold,
            reconstructor_type=reconstructor_type,
            brightness=brightness,
            contrast=contrast,
            letter_ratio=letter_ratio
        )
        mask: LayoutParser = LayoutParser(
            model=layout_model,
            model_threshold=layout_model_threshold,
            table_model=table_model,
            table_model_threshold=table_model_threshold,
            device_type=device
        )
        mask.model_post_init(None)
        total_docs: int = len(file_list)
        doc_number: int = 1
        for file in file_list:
            print(f'{doc_number}//{total_docs} processed')
            print(file)
            file_path: str = input_folder + "/" + file
            layout = mask.get_layout(file_path)
            extractor.get_blocks(file_path, layout, output_folder)
            doc_number += 1"""
    elif pipeline == "MinerU":
        from pdf2data.mineru import MinerU
        miner_pipeline: MinerU = MinerU(
            input_folder=input_folder,
            output_folder=output_folder,
            extract_equations=False,
            extract_text=False)
        miner_pipeline.pdf_transform()
    elif pipeline == "Docling":
        from pdf2data.docling import Docling
        docling_pipeline: Docling = Docling(
            input_folder=input_folder,
            output_folder=output_folder,
            extract_equations=False,
            extract_text=False)
        docling_pipeline.pdf_transform()
    elif pipeline in ["PaddlePPStructure", "PaddleVL"]:
        from pdf2data.padle_pipeline import PaddlePPStructure
        paddle_pipeline: PaddlePPStructure = PaddlePPStructure(
            extractor_name=pipeline,
            input_folder=input_folder,
            output_folder=output_folder,
            extract_equations=False,
            extract_text=False)
        paddle_pipeline.pdf_transform()
    elif pipeline == "MinerUVL":
        from pdf2data.mineru_vlm import MinerUVLM
        mineru_vlm_pipeline: MinerUVLM = MinerUVLM(
            input_folder=input_folder,
            output_folder=output_folder,
            extract_equations=False,
            extract_text=False)
        mineru_vlm_pipeline.pdf_transform()
    
    
            

def main():
    block_extractor()

"""@click.option(
    "--table_model",
    default="TableBank_faster_rcnn_R_101_FPN_3x",
    help="model used to identify the tables",
)"""

if __name__ == "__main__":
    main()
