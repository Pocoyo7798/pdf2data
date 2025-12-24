import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy
import subprocess
from pylatexenc.latex2text import LatexNodes2Text

import fitz
import PyPDF2
from sympy import content
import tensorflow as tf
from pydantic import BaseModel, PrivateAttr
import easyocr
from support import html_table_to_list

class Table(BaseModel):
    type: str = "Table"
    filepath: Optional[str] = None
    number: Optional[int] = None
    caption: Optional[str] = None
    block: Optional[List[List[str]]] = None
    footnotes: Optional[str] = None
    column_headers: Optional[List[int]] = None
    row_indexes: Optional[List[int]] = None
    page: Optional[int] = None
    box: Optional[List[float]] = None

class Figure(BaseModel):
    type: str = "Figure"
    filepath: Optional[str] = None
    number: Optional[int] = None
    caption: Optional[str] = None
    footnotes: Optional[str] = None
    page: Optional[int] = None    
    box: Optional[List[float]] = None

class Pipeline(BaseModel):
    input_folder: str
    output_folder: str
    letter_ratio: float = 3
    latex_parser: LatexNodes2Text = PrivateAttr(default=LatexNodes2Text())

    def find_column_headers(self, table_block: List[List[str]]) -> List[int]:
        """find the collumn headers as rows that do not have numbers"""
        if len(table_block) == 0:
            pass
        elif len(table_block[0]) == 0:
            pass
        else:
            collumn_headers: List[int] = []
            find_number: bool = True
            for row_number in range(len(table_block)):
                if find_number is False:
                    collumn_headers.append(row_number - 1)
                find_number = False
                for entry in table_block[row_number]:
                    if entry == "":
                        digits: int = 0
                        letters: int = 0
                    else:
                        digits = len(re.findall("[1-9]", entry))
                        letters = len(re.findall("[a-zA-Z]", entry))
                    # Verify if the entry as any letter
                    if digits > self.letter_ratio * letters:
                        find_number = True
                        break
            return collumn_headers

    def find_row_indexes(self, table_block: List[List[str]]) -> List[int]:
        """find the row indexes by finding collumns without entries with three times more digits then letters

        Parameters
        ----------
        max_rows : int, optional
            maximum rows to be considered, by default 2
        """
        row_indexes: List[int] = []
        find_number: bool = True
        if len(table_block) == 0:
            pass
        elif len(table_block) == 0:
            pass
        else:
            max_rows: int = min(len(table_block[0]), max_rows)
            for collumn_number in range(max_rows):
                find_number = False
                for row in table_block:
                    if row[collumn_number] == "":
                        digits: int = 0
                        letters: int = 0
                    else:
                        # test = re.search('[a-zA-Z]', row[collumn_number])
                        digits = len(re.findall("[1-9]", row[collumn_number]))
                        letters = len(re.findall("[a-zA-Z]", row[collumn_number]))
                        # print(f'{row[collumn_number]} presents {digits} digits and {letters} letters')
                    # Verify if the entry as any letter
                    # if test is None:
                    if digits > self.letter_ratio * letters:
                        find_number = True
                        break
                if find_number is False:
                    row_indexes.append(collumn_number)
            return row_indexes

class MinerU(Pipeline):
    folder_list: List[str] = PrivateAttr(default=[])
    
    def extract_pdfs(self) -> None:
        subprocess.run(
            [
                "mineru",
                "-p",
                self.input_folder,
                "-o",
                self.output_folder,
            ]
        )
        self.folder_list = os.listdir(self.output_folder)
    
    def generate_table_block(self, 
                             initial_block: Dict[str, Any],
                             number: int,
                             doc_name: str,
                             image_folder: str, 
                             document_folder: str) -> Dict[str, Any]:
        table_object = Table()
        table_object.number = number
        table_object.caption = self.latex_parser.latex_to_text(" ".join(initial_block["table_caption"]))
        table_object.block = html_table_to_list(initial_block["table_body"])
        table_object.footnotes = self.latex_parser.latex_to_text(" ".join(initial_block["table_footnote"]))
        table_object.column_headers = self.find_column_headers(table_object.block)
        table_object.row_indexes = self.find_row_indexes(table_object.block)
        table_object.page = initial_block["page_idx"] + 1
        table_object.box = initial_block["bbox"]
        shutil.move(
            os.path.join(document_folder, initial_block["img_path"]),
            os.path.join(f"{doc_name}_images", f"Table{number}.png"),
        )
        table_object.filepath = os.path.join(image_folder, f"{doc_name}_images", f"Table{number}.png")
        return table_object.model_dump()
    
    def generate_figure_block(self, number: int,
                              initial_block: Dict[str, Any],
                              doc_name: str,
                              image_folder: str,
                              document_folder: str) -> Dict[str, Any]:
        figure_object = Figure()
        figure_object.number = number
        figure_object.caption = self.latex_parser.latex_to_text(" ".join(initial_block["image_caption"]))
        figure_object.number = number
        figure_object.page = initial_block["page_idx"] + 1
        figure_object.box = initial_block["bbox"]
        shutil.move(
            os.path.join(document_folder, initial_block["img_path"]),
            os.path.join(f"{doc_name}_images", f"Figure{number}.png"),
        )
        figure_object.filepath = os.path.join(image_folder, f"{doc_name}_images", f"Figure{number}.png")
        return figure_object.model_dump()
    
    def generate_blocks_from_folder(self) -> None:
        if self.folder_list == []:
            raise ValueError("Folder list is empty. Please run extract_pdfs() first.")
        for folder in self.folder_list:
            blocks_info: Dict[str, List[Dict[str, Any]]] = {"blocks": [], "amount": 0}
            figure_amount: int = 0
            table_amount: int = 0
            doc_file_path: str = os.path.join(self.output_folder, folder, f"{folder}_content_list.json")
            with open(doc_file_path, "r") as doc_file:
                content_list: List[Dict[str, Any]] = json.load(doc_file)
            image_folder_path = os.path.join(self.output_folder, f"{folder}_images")
            if not os.path.exists(image_folder_path):
                os.makedirs(image_folder_path)
            for content in content_list:
                if content["type"] == "table":
                    table_amount += 1
                    table_block = self.generate_table_block(
                        content,
                        table_amount,
                        folder,
                        image_folder_path,
                        os.path.join(self.input_folder, folder),
                    )
                    blocks_info["blocks"].append(table_block)
                elif content["type"] == "image":
                    figure_amount += 1
                    figure_block = self.generate_figure_block(
                        figure_amount,
                        content,
                        folder,
                        image_folder_path,
                        os.path.join(self.input_folder, folder),
                    )
                    blocks_info["blocks"].append(figure_block)
            blocks_info["amount"] = table_amount + figure_amount
            with open(os.path.join(self.output_folder, f"{folder}blocks.json"), "w") as f:
                json.dump(blocks_info, f, indent=4)
            