import os
from typing import Any, Dict, List
import json
from pdf2data.support import get_doc_list

import click

from pdf2data.keywords import TextFinder

@click.command()
@click.argument("input_folder", type=str)
@click.argument("output_folder", type=str)
@click.argument("keywords_file", type=str)
@click.option(
    "--word_count_threshold",
    default=6,
    help="file containing generic keywords",
)
@click.option(
    "--find_paragraphs",
    default=True,
    help="True to look for paragraph type text, False otherwise.",
)
@click.option(
    "--find_section_headers",
    default=True,
    help="True to look for section headers type text, False otherwise.",
)
@click.option(
    "--count_duplicates",
    default=False,
    help="True to consider duplicates in the word count",
)

def text_finder(input_folder: str, output_folder: str, keywords_file: str, word_count_threshold: int,find_paragraphs: bool, find_section_headers:bool, count_duplicates:bool) -> None:
    if os.path.isdir(output_folder) is False:
        os.mkdir(output_folder)
    finder: TextFinder = TextFinder(keywords_file_path=keywords_file)
    doc_list: List[str] = get_doc_list(input_folder, "")
    results_path = f"{output_folder}/found_texts.txt"
    name_path = f"{output_folder}/found_texts_doc_names.txt"
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(name_path):
        os.remove(name_path)
    for doc in doc_list:
        text_path: str = f"{input_folder}/{doc}/{doc}_content.json"
        results: Dict[str, Any] = finder.find(text_path, word_count_threshold, paragraph=find_paragraphs, section_header=find_section_headers, count_duplicates=count_duplicates)
        text_list = results["text"]
        for text in text_list:
            with open(name_path, "a") as f:
                f.write(doc + "\n")
            with open(results_path, "a") as f:
                f.write(text + "\n")

def main():
    text_finder()


if __name__ == "__main__":
    main() 