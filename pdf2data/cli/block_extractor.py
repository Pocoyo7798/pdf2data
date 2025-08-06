import os
from typing import List

import click

from pdf2data.block import BlockExtractor
from pdf2data.mask import LayoutParser
from pdf2data.support import get_doc_list


@click.command()
@click.argument("input_folder", type=str)
@click.argument("output_folder", type=str)
@click.option(
    "--layout_model",
    default="DocLayout-YOLO-DocStructBench",
    help="model used to determine the overall layout",
)
@click.option(
    "--layout_model_threshold",
    default=0.7,
    help="threshold of the layout model",
)
@click.option(
    "--table_model",
    default=None,
    help="model used to identify the tables",
)
@click.option(
    "--table_model_threshold",
    default=0.5,
    help="threshold of the table detection model",
)
@click.option(
    "--extract_tables",
    default=True,
    help="True to extract tables, False otherwise",
)
@click.option(
    "--extract_figures",
    default=True,
    help="True to extract figures, False otherwise",
)
@click.option(
    "--correct_struct",
    default=True,
    help="True to to correct the table structure using the words position, False otherwise",
)
@click.option(
    "--table_zoom",
    default=1.5,
    help="zoom of the image containing the table",
)
@click.option(
    "--figure_zoom",
    default=1,
    help="zoom of the figures extracted",
)
@click.option(
    "--x_table_corr",
    default=0.015,
    help="factor correct the table coordinates in the x axis",
)
@click.option(
    "--y_table_corr",
    default=0.015,
    help="factor correct the table coordinates in the y axis",
)
@click.option(
    "--iou_lines",
    default=0.2,
    help="iou value to supress collumns and rows",
)
@click.option(
    "--iou_struct",
    default=0.02,
    help="minimum iou between table and row/collumn to consider the row/collumn as correct",
)
@click.option(
    "--word_factor",
    default=1.0,
    help="factor used to determine the maximum distance to consider two different words as a single table entry",
)
@click.option(
    "--word_iou",
    default=0.02,
    help="iou value to consider that a word is inside a specific table entry",
)
@click.option(
    "--struct_model",
    default="microsoft/table-structure-recognition-v1.1-all",
    help="table structure detection model threshold",
)
@click.option(
    "--struct_model_threshold",
    default=0.3,
    help="table structure detection model threshold",
)
@click.option(
    "--reconstructor_type",
    default="entry_by_entry",
    help="type of reconstructor used, options: 'entry_by_entry' or 'word_by_word'",
)
@click.option(
    "--brightness",
    default=1.0,
    help="brightness factor of the Table image",
)
@click.option(
    "--contrast",
    default=1.1,
    help="contrast factor of the Table image",
)
@click.option(
    "--device",
    default="cpu",
    help="device to run the mask models",
)
@click.option(
    "--letter_ratio",
    default=3.0,
    help="minimum ratio between letter and ratio to consider a column as a row index or a row as a collumn header",
)
def block_extractor(
    input_folder: str,
    output_folder: str,
    layout_model: str,
    table_model: str,
    layout_model_threshold: float,
    table_model_threshold: float,
    extract_tables: bool,
    extract_figures: bool,
    correct_struct: bool,
    table_zoom: float,
    figure_zoom: float,
    x_table_corr: float,
    y_table_corr: float,
    iou_lines: float,
    iou_struct: float,
    word_factor: float,
    word_iou: float,
    struct_model: str,
    struct_model_threshold: float,
    reconstructor_type: str,
    brightness: float,
    contrast: float,
    device: str,
    letter_ratio: float,
) -> None:
    if os.path.isdir(output_folder) is False:
        os.mkdir(output_folder)
    file_list: List[str] = get_doc_list(input_folder, "pdf")
    extractor: BlockExtractor = BlockExtractor(
        extract_tables=extract_tables,
        extract_figures=extract_figures,
        correct_struct=correct_struct,
        table_zoom=table_zoom,
        figure_zoom=figure_zoom,
        x_table_corr=x_table_corr,
        y_table_corr=y_table_corr,
        iou_lines=iou_lines,
        iou_struct=iou_struct,
        word_factor=word_factor,
        word_iou=word_iou,
        structure_model= struct_model,
        struct_model_threshold=struct_model_threshold,
        reconstructor_type=reconstructor_type,
        brightness=brightness,
        contrast=contrast,
        letter_ratio=letter_ratio
    )
    extractor.model_post_init(None)
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
        file_path: str = input_folder + "/" + file
        layout = mask.get_layout(file_path)
        extractor.get_blocks(file_path, layout, output_folder)
        doc_number += 1

def main():
    block_extractor()

"""@click.option(
    "--table_model",
    default="TableBank_faster_rcnn_R_101_FPN_3x",
    help="model used to identify the tables",
)"""

if __name__ == "__main__":
    main()
