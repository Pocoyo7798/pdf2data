import json
import os
from typing import Any, Dict, List, Optional

import click

from pdf2data.metadata2 import Metadata
from pdf2data.support2 import get_doc_list


@click.command()
@click.argument("path", type=str)
@click.option(
    "--output_folder",
    default=None,
    help="Output folder path",
)
def metadata_finder(path: str, output_folder: Optional[str]) -> None:
    """Create a json file containing the metadata for each file in path

    Parameters
    ----------
    path : str
        path to the file or folder to be processed
    output_folder : Optional[str]
        Output folder path to save the genereate json file containing the metdata
    """
    if os.path.isfile(path):
        file_list: List[str] = [path]
        input_folder: str = ""
    else:
        file_list = get_doc_list(path, "pdf") + get_doc_list(path, "cermxml")
        input_folder = path + "/"
    if output_folder is None:
        output_path = ""
    else:
        output_path = output_folder + "/"
    for file in file_list:
        file_path: str = f"{input_folder}{file}"
        file_name: str = os.path.splitext(file)[0]
        metadata: Metadata = Metadata(file_path=file_path)
        metadata.update()
    metadata_dict: Dict[str, Any] = metadata.__dict__
    del metadata_dict["file_path"]
    json_metadata = json.dumps(metadata_dict, indent=4)
    with open(f"{output_path}{file_name}_metadata.json", "w") as j:
        # convert the dictionary into a json variable
        j.write(json_metadata)


def main():
    metadata_finder()


if __name__ == "__main__":
    main()
