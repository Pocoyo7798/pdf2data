import json
import os

import click

from pdf2data.evaluator import Evaluator
from pdf2data.support import get_doc_list

@click.command()
@click.argument("ref_folder", type=str)
@click.argument("results_folder", type=str)
@click.argument("eval_file_path", type=str)
@click.argument("type", type=str)
@click.option(
    "--string_threshold",
    default=0.85,
    help="similarity threshold between strings",
)
@click.option(
    "--box_threshold",
    default=0.6,
    help="similarity threshold between strings",
)

def evaluator(ref_folder: str, results_folder: str, eval_file_path: str, type: str, string_threshold: float, box_threshold: float):
    available_types: set = set(["metadata", "text", "blocks", "table_detection"])
    if type not in available_types:
        raise AttributeError(f"The specified type is not availabe, the available types are {available_types}")
    evaluator = Evaluator(ref_folder=ref_folder, result_folder=results_folder, eval_file_path=eval_file_path, string_similarity=string_threshold, iou_threshold=box_threshold)
    if type == "metadata":
        evaluator.eval_metadata()
    elif type == "text":
        evaluator.eval_text()
    elif type == "blocks":
        evaluator.eval_blocks()
    elif type == "table_detection":
        evaluator.eval_table_detector()
    

def main():
    evaluator()


if __name__ == "__main__":
    main()
