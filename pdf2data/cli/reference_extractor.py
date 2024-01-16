import os
from typing import List

import click

from pdf2data.references import References
from pdf2data.support import get_doc_list


@click.command()
@click.argument("input_path", type=str)
@click.argument("output_folder", type=str)
def reference_extractor(input_path: str, output_folder: str) -> None:
    if os.path.isfile(input_path):
        file_list: List[str] = [input_path]
        input_folder: str = ""
    else:
        file_list = get_doc_list(input_path, "pdf") + get_doc_list(
            input_path, "cermxml"
        )
        input_folder = input_path + "/"
    for file in file_list:
        file_path: str = f"{input_folder}{file}"
        references: References = References(
            file_path=file_path, output_folder=output_folder
        )
        references.generate_reference_file()


def main():
    reference_extractor()


if __name__ == "__main__":
    main()
