import os
from typing import Any, Dict, List
import json
from pdf2data.support import get_doc_list

import click

from pdf2data.keywords import BlockFinder

@click.command()
@click.argument("input_folder", type=str)
@click.argument("output_folder", type=str)
@click.argument("keywords_file", type=str)
@click.option(
    "--generic_file",
    default=None,
    help="file containing generic keywords",
)
@click.option(
    "--find_tables",
    default=True,
    help="True to look for tables, False otherwise.",
)
@click.option(
    "--find_figures",
    default=False,
    help="True to look for figures, False otherwise.",
)
def block_finder(input_folder: str, output_folder: str, keywords_file:str, generic_file: str, find_tables: bool, find_figures: bool):
    if os.path.isdir(output_folder) is False:
        os.mkdir(output_folder)
    finder: BlockFinder = BlockFinder(keywords_file_path=keywords_file, generic_keywords_file_path=generic_file)
    finder.model_post_init(None)
    doc_list: List[str] = get_doc_list(input_folder, "")
    for doc in doc_list:
        print(doc)
        blocks_path: str = f"{input_folder}/{doc}/{doc}_blocks.json"
        blocks = finder.find(blocks_path, tables=find_tables, figures=find_figures)
        blocks_json = json.dumps(blocks, indent=4)
        results_path = f"{output_folder}/{doc}_found_blocks.json"
        with open(results_path, "w") as f:
            f.write(blocks_json)

def main():
    block_finder()


if __name__ == "__main__":
    main() 