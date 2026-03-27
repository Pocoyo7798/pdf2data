import os
from typing import Any, Dict, List
import json
import time

import click
from pdf2data.table_extractor import TableExtractor


@click.command()
@click.argument("input_file", type=str)
@click.argument("output_file", type=str)
@click.option(
    "--table_type",
    default="characterization",
    help="type of data to extract (characterization, synthesis, or all)",
)

def extract(input_file: str,
                output_file: str,
                table_type: str,
) -> None:
    extractor = TableExtractor(table_type=table_type)
    extractor.extract_tables(input_file, output_file)

def main():
    extract()


if __name__ == "__main__":
    main() 