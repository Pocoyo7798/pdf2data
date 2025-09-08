import os
from typing import Any, Dict, List
import json
import shutil

import click

from pdf2data.block import BlockExtractor
from pdf2data.mask import LayoutParser
from pdf2data.support import get_doc_list
from pdf2data.text import TextExtractor, TextFileGenerator
from pdf2data.references import References
from pdf2data.metadata import Metadata

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
    "--text_extractor_type",
    default="layoutparser",
    help="type of the text extractor, available: ['layoutparser', 'cermine', 'minersix']",
)
@click.option(
    "--reference_extractor_type",
    default="anystyle",
    help="type of the reference extractor, available: ['anystyle', 'cermine']",
)
@click.option(
    "--metadata_extractor_type",
    default="pdf2doi",
    help="type of the metadata extractor, available: ['pdf2doi', 'cermine']",
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
    default=False,
    help="True to to correct the table structure using the words position, False otherwise",
)
@click.option(
    "--table_zoom",
    default=1.5,
    help="zoom of the image containing the table",
)
@click.option(
    "--figure_zoom",
    default=3,
    help="zoom of the figures extracted",
)
@click.option(
    "--x_table_corr",
    default=0.01,
    help="factor correct the table coordinates in the x axis",
)
@click.option(
    "--y_table_corr",
    default=0.01,
    help="factor correct the table coordinates in the y axis",
)
@click.option(
    "--iou_lines",
    default=0.5,
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
    default=0.00001,
    help="iou value to consider that a word is inside a specific table entry",
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
    default=4.0,
    help="minimum ratio between letter and ratio to consider a column as a row index or a row as a collumn header",
)
def pdf2data(input_folder: str,
    output_folder: str,
    layout_model: str,
    table_model: str,
    layout_model_threshold: float,
    table_model_threshold: float,
    text_extractor_type: float,
    reference_extractor_type: float,
    metadata_extractor_type: float,
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
    possible_types = set(["layoutparser", "cermine", "minersix"])
    if text_extractor_type not in possible_types:
        raise AttributeError(
            f"{type} is not a available type, try one of the following: {possible_types}"
        )
    if text_extractor_type == "layoutparser":
        text_extension: str = ".pdf"
    else:
        generator: TextFileGenerator = TextFileGenerator(
            input_folder=input_folder, output_folder=output_folder
        )
        generator.model_post_init(None)
        if text_extractor_type == "cermine":
            generator.pdf_to_cermxml()
            text_extension = ".cermxml"
        elif text_extractor_type == "minersix":
            generator.pdf_to_miner("txt")
            text_extension = ".txt"
    if reference_extractor_type == "anystyle":
        reference_extension: str = ".pdf"
    elif reference_extractor_type == "cermine":
        if text_extractor_type != "cermine":
            raise ValueError("Can only use reference_extractor as 'cermine' if the text_extrator is 'cermine'")
        else:
            reference_extension: str = ".cermxml"
    if metadata_extractor_type == "pdf2doi":
        metadata_extension: str = ".pdf"
    elif metadata_extractor_type == "cermine":
        if text_extractor_type != "cermine":
            raise ValueError("Can only use metadata_extractor as 'cermine' if the text_extrator is 'cermine'")
        else:
            metadata_extension: str = ".cermxml"
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
        print(file)
        doc_number += 1
        file_name = os.path.splitext(file)[0]
        file_folder = output_folder + "/" + file_name

        if os.path.isdir(file_folder) is False:
            os.mkdir(file_folder)
        file_path = input_folder + "/" + file
        layout: Dict[str, Any] = mask.get_layout(file_path)
        text_doc = file_name + text_extension
        text_path: str = f"{output_folder}/{text_doc}"
        text_extractor: TextExtractor = TextExtractor(
                input_file=text_path, output_folder=file_folder
            )
        if text_extractor_type == "layoutparser":
            text_extractor.extract_layoutparser(f"{file_name}_text", layout)
        else:
            if text_extractor_type == "cermine":
                text_extractor.extract_cermine(f"{file_name}_text")
            elif text_extractor_type == "minersix":
                text_extractor.extract_txt(f"{file_name}_text")
        reference_path = input_folder + "/" + file_name + reference_extension
        references: References = References(
            file_path=reference_path, output_folder=file_folder
        )
        references.generate_reference_file()
        metadata_path = input_folder + "/" + file_name + metadata_extension
        metadata: Metadata = Metadata(file_path=metadata_path)
        metadata.update()
        doi: str = metadata.doi
        metadata_dict: Dict[str, Any] = metadata.__dict__
        del metadata_dict["file_path"]
        json_metadata = json.dumps(metadata_dict, indent=4)
        with open(f"{file_folder}/{file_name}_metadata.json", "w") as j:
            # convert the dictionary into a json variable
            j.write(json_metadata)
        extractor.get_blocks(file_path, layout, file_folder, doi=doi)
        if text_extractor_type in set(["cermine", "minersix"]):
            shutil.move(
                text_path, file_folder + "/" + file_name + text_extension
            )
        shutil.move(
                file_path, file_folder + "/" + file
            )

def main():
    pdf2data()


if __name__ == "__main__":
    main() 