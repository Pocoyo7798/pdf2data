import os
from typing import Any, Dict, List
import json
import time

import click
from pdf2data.upgrade import Upgrader


@click.command()
@click.argument("input_folder", type=str)
@click.option(
    "--correct_unicodes",
    default=True,
    help="True to correct unicodes, False otherwise",
)
@click.option(
    "--merge_figures",
    default=True,
    help="True to merge close figures, False otherwise",
)
@click.option(
    "--all_documents",
    default=True,
    help="True to upgrade all documents, False to upgrade partial documents",
)
@click.option(
    "--distance_threshold",
    default=25.0,
    help="distance threshold to merge figures, in pixels",
)
def upgrader(input_folder: str,
                correct_unicodes: bool,
                merge_figures: bool,
                all_documents: bool,
                distance_threshold: float
) -> None:
    upgrader = Upgrader(
        correct_unicodes=correct_unicodes,
        merge_figures=merge_figures,
        all_documents=all_documents,
        distance_threshold=distance_threshold
    )
    upgrader.upgrade(input_folder)

def main():
    upgrader()


if __name__ == "__main__":
    main() 