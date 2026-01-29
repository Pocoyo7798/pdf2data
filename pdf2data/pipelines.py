import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from weakref import ref
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
from pdf2data.support import html_table_to_list
from docling.document_converter import DocumentConverter
from paddleocr import PPStructureV3


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

class Text(BaseModel):
    type: str = "paragraph" #paragraphs, titles
    content: Optional[str] = None
    page: Optional[int] = None
    box: Optional[List[float]] = None

class Equation(BaseModel):
    type: str = "equation"
    filepath: Optional[str] = None
    number: Optional[int] = None
    content: Optional[str] = None
    page: Optional[int] = None
    box: Optional[List[float]] = None

class Pipeline(BaseModel):
    input_folder: str
    output_folder: str
    letter_ratio: float = 3
    extract_tables: bool = True
    extract_figures: bool = True
    extract_text: bool = True
    extract_equations: bool = True
    extract_references: bool = False
    _latex_parser: LatexNodes2Text = PrivateAttr(default=LatexNodes2Text())
    
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

    def find_row_indexes(self, table_block: List[List[str]], max_rows: int = 2) -> List[int]:
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
    
    def snap_figure(self, image_folder_path: str, page, file_path:str, box: List[float], number: int, doc_name: str, block_type) -> Dict[str, Any]:
        # Open the PDF and extract the formula region
        pdf_document = fitz.open(file_path)
        page = pdf_document[page - 1]  # Pages are 0-indexed in fitz
        
        # Create a rectangle from the box coordinates [l, t, r, b]
        rect = fitz.Rect(box[0], box[1], 
                        box[2], box[3])
        
        # Normalize and clip the rectangle to page bounds
        rect.normalize()  # Ensures coordinates are in correct order
        page_rect = page.rect
        rect = rect & page_rect  # Intersect with page bounds
        
        # Render the page region as a pixmap (image)
        mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better quality
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        # Save the image
        image_path = os.path.join(image_folder_path, f"{block_type}_{number}.png")
        pix.save(image_path)
        pdf_document.close()
        return os.path.join(f"{doc_name}_images", f"{block_type}_{number}.png")

    def correct_table_structure(self, table_list: List[List[str]]) -> List[List[str]]:
        row_length: int = max(len(row) for row in table_list)
        corrected_table: List[List[str]] = []
        i: int = 0
        for row in table_list:
            if len(row) < row_length and i == 0:
                row += [""] * (row_length - len(row))
            elif len(row) < row_length:
                j = len(row)
                while j < len(corrected_table[i - 1]):
                    row += [corrected_table[i - 1][j]]
                    j += 1
            i += 1
            corrected_table.append(row)
        return corrected_table

class MinerU(Pipeline):
    pdf_copies_folder: Optional[str] = None
    _folder_list: List[str] = PrivateAttr(default=[])
    _reference_list: List[str] = PrivateAttr(default=[])
    
    def generate_text(self, text_list: List[Dict[str, Any]]) -> str:
        final_text = ""
        for text in text_list:
            if text["type"] == "equation_inline":
                final_text += self._latex_parser.latex_to_text(text["content"]).replace(" ", "")
            else:
                final_text += text["content"]
        return final_text
    
    def extract_pdfs(self) -> None:
        print(os.listdir(self.pdf_copies_folder))
        subprocess.run(
            [
                "mineru",
                "-p",
                self.pdf_copies_folder,
                "-o",
                self.output_folder,
                "-l",
                "en",
            ]
        )
    
    def generate_table_block(self, 
                             initial_block: Dict[str, Any],
                             number: int,
                             doc_name: str,
                             image_folder: str, 
                             document_folder: str,
                             page_number: int) -> Dict[str, Any]:
        table_object = Table()
        content = initial_block["content"] 
        table_object.number = number
        table_object.caption = self.generate_text(content["table_caption"])
        try:
            table_object.block = html_table_to_list(content["html"])
        except KeyError:
            return None
        table_object.block = self.correct_table_structure(table_object.block)
        table_object.footnotes = self.generate_text(content["table_footnote"])
        table_object.column_headers = self.find_column_headers(table_object.block)
        table_object.row_indexes = self.find_row_indexes(table_object.block)
        table_object.page = page_number
        table_object.box = initial_block["bbox"]
        if content["image_source"]["path"] == "images/":
            return None
        shutil.move(
            os.path.join(document_folder, f"{content['image_source']['path']}"),
            os.path.join(image_folder, f"Table_{number}.png"),
        )
        table_object.filepath = os.path.join(f"{doc_name}_images", f"Table{number}.png")
        return table_object.model_dump()
    
    def generate_figure_block(self, number: int,
                              initial_block: Dict[str, Any],
                              doc_name: str,
                              image_folder: str,
                              document_folder: str,
                              page_number:int) -> Dict[str, Any]:
        figure_object = Figure()
        content = initial_block["content"]
        figure_object.number = number
        figure_object.caption = self.generate_text(content["image_caption"])
        figure_object.number = number
        figure_object.page = page_number
        figure_object.box = initial_block["bbox"]
        shutil.move(
            os.path.join(document_folder, content["image_source"]["path"]),
            os.path.join(image_folder, f"Figure_{number}.png"),
        )
        figure_object.filepath = os.path.join(f"{doc_name}_images", f"Figure_{number}.png")
        return figure_object.model_dump()
    
    def generate_text_block(self, 
                             initial_block: Dict[str, Any], page_number: int) -> Dict[str, Any]: 
        text_object = Text()
        if initial_block["type"] == "paragraph":
            text_object.type = "paragraph"
        elif initial_block["type"] == "title":
            text_object.type = "section_header"
        text_content = initial_block["content"][f"{initial_block['type']}_content"]
        text_object.content = self.generate_text(text_content)
        text_object.page = page_number
        text_object.box = initial_block["bbox"]
        return text_object.model_dump()
    
    def generate_equation_block(self, number:int,
                            initial_block: Dict[str, Any], 
                            doc_name: str,
                            image_folder: str,
                            document_folder: str,
                            page_number: int) -> Dict[str, Any]:
        equation_object = Equation()
        equation_object.content = self._latex_parser.latex_to_text(initial_block["content"]["math_content"]).replace(" ", "")
        equation_object.page = page_number
        equation_object.box = initial_block["bbox"]
        equation_object.number = number
        shutil.move(
            os.path.join(document_folder, initial_block["content"]["image_source"]["path"]),
            os.path.join(image_folder, f"Equation_{number}.png"),
        )
        equation_object.filepath = os.path.join(f"{doc_name}_images", f"Equation_{number}.png")
        return equation_object.model_dump()

    def update_references(self, content: Dict[str, Any]) -> None:
        try:
            if content["content"]["list_type"] == "reference_list":
                pass
        except KeyError:
            return None
        for item in content["content"]["list_items"]:
            self._reference_list.append(self.generate_text(item["item_content"]))

    def generate_blocks_from_folder(self) -> None:
        if self._folder_list == []:
            raise ValueError("Folder list is empty. Please run extract_pdfs() first.")
        total_docs: int = len(self._folder_list)
        doc_number: int = 1
        for folder in self._folder_list:
            print(folder)
            print(f'{doc_number}//{total_docs} processed')
            doc_number += 1
            blocks_info: Dict[str, List[Dict[str, Any]]] = {"blocks": []}
            figure_amount: int = 0
            table_amount: int = 0
            equation_amount: int = 0
            results_folder_list = os.listdir(self.output_folder + f'/{folder}')
            if "auto" in results_folder_list:
                process_type = "auto"
            elif "hybrid_auto" in results_folder_list:
                process_type = "hybrid_auto"
            else:
                raise ValueError("Did not found a valid folder type in the MinerU output.")
            miner_results_folder = os.path.join(self.output_folder, f"{folder}/{process_type}")
            doc_file_path: str = os.path.join(miner_results_folder, f"{folder}_content_list_v2.json")
            with open(doc_file_path, "r") as doc_file:
                content_list: List[Dict[str, Any]] = json.load(doc_file)
            if self.extract_tables and self.extract_figures and self.extract_text:
                image_folder_path = os.path.join(self.output_folder, folder, f"{folder}_images")
            else:
                image_folder_path = os.path.join(self.output_folder, f"{folder}_images")
            if not os.path.exists(image_folder_path):
                os.makedirs(image_folder_path)
            page_number = 0
            self._reference_list = []
            for page in content_list:
                page_number += 1
                for content in page:
                    if content["type"] == "table" and self.extract_tables:
                        table_amount += 1
                        table_block = self.generate_table_block(
                            content,
                            table_amount,
                            folder,
                            image_folder_path,
                            miner_results_folder,
                            page_number
                        )
                        if table_block is not None:
                            blocks_info["blocks"].append(table_block)
                    elif content["type"] == "image" and self.extract_figures:
                        figure_amount += 1
                        figure_block = self.generate_figure_block(
                            figure_amount,
                            content,
                            folder,
                            image_folder_path,
                            miner_results_folder,
                            page_number
                        )
                        blocks_info["blocks"].append(figure_block)
                    elif content["type"] in ["paragraph", "title"] and self.extract_text:
                        text_block = self.generate_text_block(
                            content,
                            page_number
                        )
                        blocks_info["blocks"].append(text_block)
                    elif content["type"] == "equation_interline" and self.extract_equations:
                        equation_amount += 1
                        equation_block = self.generate_equation_block(
                            equation_amount,
                            content,
                            folder,
                            image_folder_path,
                            miner_results_folder,
                            page_number
                        )
                        blocks_info["blocks"].append(equation_block)
                    elif content["type"] == "list" and self.extract_references:
                        self.update_references(content)
            if self.extract_tables and self.extract_figures and self.extract_text:
                with open(os.path.join(f"{self.output_folder}", folder, f"{folder}_content.json"), "w") as f:
                    json.dump(blocks_info, f, indent=4)
                if self.extract_references:
                    with open(os.path.join(f"{self.output_folder}", folder, f"{folder}_references.txt"), "w") as r:
                        for reference in self._reference_list:
                            r.write(reference + "\n")
                # Delete the miner results folder and its contents
                if os.path.exists(miner_results_folder):
                    shutil.rmtree(miner_results_folder)
            else:
                with open(os.path.join(f"{self.output_folder}", f"{folder}_content.json"), "w") as f:
                    json.dump(blocks_info, f, indent=4)
                # Delete the miner results folder and its contents
                if os.path.exists(os.path.join(self.output_folder, folder)):
                    shutil.rmtree(os.path.join(self.output_folder, folder))
    
    def pdf_transform(self) -> None:
        files_list = os.listdir(self.input_folder)
        self.pdf_copies_folder = os.path.join(self.input_folder, "pdf_copies")
        os.makedirs(self.pdf_copies_folder, exist_ok=True)
        total_size = 0
        print(files_list)
        for file_name in files_list:
            if file_name.endswith('.pdf'):
                file_path = os.path.join(self.input_folder, file_name)
                file_size = os.path.getsize(file_path) / 1024 / 1024
                print(total_size)
                if total_size + file_size <= 100:  # 30 MB in bytes
                    new_file_path = os.path.join(self.pdf_copies_folder, file_name)
                    shutil.copy(file_path, new_file_path)
                    total_size += file_size
                else:
                    self.extract_pdfs()
                    # Delete all files in the pdf_copies_folder
                    for copied_file in os.listdir(self.pdf_copies_folder):
                        copied_file_path = os.path.join(self.pdf_copies_folder, copied_file)
                        if os.path.isfile(copied_file_path):
                            os.remove(copied_file_path)
                    new_file_path = os.path.join(self.pdf_copies_folder, file_name)
                    shutil.copy(file_path, new_file_path)
                    total_size += file_size
        self.extract_pdfs()
        if os.path.exists(self.pdf_copies_folder):
                    shutil.rmtree(self.pdf_copies_folder)
        self._folder_list = os.listdir(self.output_folder)
        self.generate_blocks_from_folder()


class Docling(Pipeline):
    _converter: DocumentConverter = PrivateAttr(default=DocumentConverter())
    _table_dict: Dict[str, Any] = PrivateAttr(default={})
    _text_dict: Dict[str, Any] = PrivateAttr(default={})
    _figure_dict: Dict[str, Any] = PrivateAttr(default={})
    _groups_dict: Dict[str, Any] = PrivateAttr(default={})
    _page_dict_height: Dict[str, Any] = PrivateAttr(default={})

    def correct_boxes(self, l, t, r, b, page_height, origin):
        if origin == "BOTTOMLEFT":
            corrected_t = page_height - b
            corrected_b = page_height - t
            return [l, corrected_t, r, corrected_b]
        elif origin == "TOPLEFT":
            return [l, t, r, b]

    def generate_blocks_dicts(self, document_dict: Dict[str, Any]) -> None:
        text_list = document_dict["texts"]
        table_list = document_dict["tables"]
        figure_list = document_dict["pictures"]
        groups_list = document_dict["groups"]
        text_dict = {}
        table_dict = {}
        figure_dict = {}
        for text in text_list:
            ref = text["self_ref"]
            text_dict[ref] = text
        for table in table_list:
            ref = table["self_ref"]
            table_dict[ref] = table
        for figure in figure_list:
            ref = figure["self_ref"]
            figure_dict[ref] = figure
        for group in groups_list:
            ref = group["self_ref"]
            self._groups_dict[ref] = group
        self._text_dict = text_dict
        self._table_dict = table_dict
        self._figure_dict = figure_dict
        self._page_dict_height = document_dict["pages"]

    def get_text_from_list(self, text_list: List[Dict[str, Any]]) -> str:
        final_text = ""
        for text in text_list:
            text_dict = self._text_dict[text["$ref"]]
            final_text += text_dict["orig"]
        return final_text
    
    def get_table_from_cells(self, cell_list: List[Dict[str, Any]]) -> List[List[str]]:
        table_block = []
        column_headers = []
        row_indexes = []
        max_row = 0
        max_col = 0
        for cell in cell_list:
            if cell["end_row_offset_idx"] - 1 > max_row:
                max_row = cell["end_row_offset_idx"] - 1
            if cell["end_col_offset_idx"] - 1 > max_col:
                max_col = cell["end_col_offset_idx"] - 1
        for _ in range(max_col+ 1):
            table_block.append([""] * (max_row + 1))
        for cell in cell_list:
            cell_text = cell["text"]
            for row in range(cell["start_row_offset_idx"], cell["end_row_offset_idx"]):
                for col in range(cell["start_col_offset_idx"], cell["end_col_offset_idx"]):
                    table_block[col][row] = cell_text
                    if cell["column_header"] and row not in column_headers:
                        column_headers.append(row)
                    if cell["row_header"] and col not in row_indexes:
                        row_indexes.append(col)
        return table_block, column_headers, row_indexes

    def generate_formula_block(self, formula_dict: Dict[str, Any], image_folder_path: str, file_path: str, number: int, doc_name: str) -> Dict[str, Any]:
        formula_object = Equation()
        formula_object.content = formula_dict["orig"]
        formula_object.page = formula_dict["prov"][0]["page_no"]
        formula_object.number = number
        formula_object.box = self.correct_boxes(formula_dict["prov"][0]["bbox"]["l"],
                                               formula_dict["prov"][0]["bbox"]["t"],
                                               formula_dict["prov"][0]["bbox"]["r"],
                                               formula_dict["prov"][0]["bbox"]["b"],
                                               self._page_dict_height[str(formula_object.page)]["size"]["height"],
                                               formula_dict["prov"][0]["bbox"]["coord_origin"])
        formula_object.filepath = self.snap_figure(image_folder_path,
                                                  formula_object.page,
                                                  file_path,
                                                  formula_object.box,
                                                  number,
                                                  doc_name,
                                                  "Equation")
        return formula_object.model_dump()

    def generate_text_block(self, 
                             ref: str,
                             image_folder_path: str,
                             file_path: str,
                             number: int,
                             doc_name: str) -> Dict[str, Any]:
        text_dict: Dict[str, Any] = self._text_dict[ref]
        text_object = Text()
        text_object.content = text_dict["orig"]
        if text_dict["label"] == "text" and self.extract_text:
            text_object.type = "paragraph"
        elif text_dict["label"] == "section_header" and self.extract_text:
            text_object.type = "section_header"
        elif text_dict["label"] == "formula" and self.extract_equations:
            return self.generate_formula_block(text_dict, image_folder_path, file_path, number, doc_name)
        else:
            return {}
        text_object.page = text_dict["prov"][0]["page_no"]
        text_object.box = self.correct_boxes(text_dict["prov"][0]["bbox"]["l"],
                                             text_dict["prov"][0]["bbox"]["t"],
                                             text_dict["prov"][0]["bbox"]["r"],
                                             text_dict["prov"][0]["bbox"]["b"], 
                                             self._page_dict_height[str(text_object.page)]["size"]["height"],
                                             text_dict["prov"][0]["bbox"]["coord_origin"]
                                            )
        return text_object.model_dump()
    
    def generate_figure_block(self,
                             ref: str,
                             image_folder_path: str,
                             file_path: str,
                             number: int,
                             doc_name: str) -> Dict[str, Any]:
        figure_dict: Dict[str, Any] = self._figure_dict[ref]
        figure_object = Figure()
        figure_object.caption = self.get_text_from_list(figure_dict["captions"])
        figure_object.footnotes = self.get_text_from_list(figure_dict["footnotes"])
        figure_object.page = figure_dict["prov"][0]["page_no"]
        figure_object.number = number
        figure_object.box = self.correct_boxes(figure_dict["prov"][0]["bbox"]["l"],
                                               figure_dict["prov"][0]["bbox"]["t"],
                                               figure_dict["prov"][0]["bbox"]["r"],
                                               figure_dict["prov"][0]["bbox"]["b"],
                                               self._page_dict_height[str(figure_object.page)]["size"]["height"],
                                               figure_dict["prov"][0]["bbox"]["coord_origin"])
        figure_object.filepath = self.snap_figure(image_folder_path,
                                                figure_object.page,
                                                file_path,
                                                figure_object.box,
                                                number,
                                                doc_name,
                                                "Figure")
        return figure_object.model_dump()
    
    def generate_table_block(self,
                             ref: str,
                             image_folder_path: str,
                             file_path: str,
                             number: int,
                             doc_name: str) -> Dict[str, Any]:
        table_dict: Dict[str, Any] = self._table_dict[ref]
        table_object = Table()
        table_object.caption = self.get_text_from_list(table_dict["captions"])
        table_object.footnotes = self.get_text_from_list(table_dict["footnotes"])
        table_object.page = table_dict["prov"][0]["page_no"]
        table_object.number = number
        table_object.box = self.correct_boxes(table_dict["prov"][0]["bbox"]["l"],
                                              table_dict["prov"][0]["bbox"]["t"],
                                              table_dict["prov"][0]["bbox"]["r"],
                                              table_dict["prov"][0]["bbox"]["b"],
                                              self._page_dict_height[str(table_object.page)]["size"]["height"],
                                              table_dict["prov"][0]["bbox"]["coord_origin"])
        table_object.block, table_object.column_headers, table_object.row_indexes = self.get_table_from_cells(table_dict["data"]["table_cells"])
        table_object.filepath = self.snap_figure(image_folder_path,
                                                 table_object.page,
                                                 file_path,
                                                 table_object.box,
                                                 number,
                                                 doc_name,
                                                 "Table")
        return table_object.model_dump()

    def generate_blocks_from_dict(self, document_dict: Dict[str, Any], results_folder: str, image_folder_path: str, file_path: str, file_name) -> None:
        document_body: List[Dict] = document_dict["body"]["children"]
        blocks_info: Dict[str, List[Dict[str, Any]]] = {"blocks": []}
        figure_amount: int = 1
        table_amount: int = 1
        equation_amount: int = 1
        get_references: bool = False
        reference_list: List[str] = []
        for content in document_body:
            reference = content["$ref"]
            block_data: Dict[str, Any] = {}
            if "texts" in reference and self.extract_text:
                text_block = self.generate_text_block(
                        reference,
                        image_folder_path,
                        file_path,
                        equation_amount,
                        file_name
                    )
                if text_block == {}:
                    pass
                elif text_block["type"] == "equation":
                    equation_amount += 1
                elif  text_block["content"].lower() in ["references", "bibliography", "references list", "literature cited", "works cited"]:
                    get_references = True
                block_data = text_block
            elif "pictures" in reference and self.extract_figures:
                image_block = self.generate_figure_block(
                        reference,
                        image_folder_path,
                        file_path,
                        figure_amount,
                        file_name
                    )
                figure_amount += 1
                block_data = image_block
            elif "tables" in reference and self.extract_tables:
                table_block = self.generate_table_block(
                        reference,
                        image_folder_path,
                        file_path,
                        table_amount,
                        file_name
                    )
                table_amount += 1
                block_data = table_block
            elif "groups" in reference and get_references:
                group_dict: Dict[str, Any] = self._groups_dict[reference]
                for item in group_dict["children"]:
                    text_reference = item["$ref"]
                    text = self._text_dict[text_reference]["orig"]
                    reference_list.append(text)
            if block_data != {}:
                blocks_info["blocks"].append(block_data)
        block_info_json = json.dumps(blocks_info, indent=4)
        with open(results_folder + "/" + f"{file_name}_content.json", "w") as f:
            f.write(block_info_json)
        if self.extract_references:
            with open(os.path.join(results_folder, f"{file_name}_references.txt"), "w") as r:
                for reference in reference_list:
                    r.write(reference + "\n")
    
    def pdf_transform(self) -> None:
        files_list = os.listdir(self.input_folder)
        number = 1
        for file in files_list:
            print(f"{number}//{len(files_list)} processed")
            if file.endswith('.pdf'):
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(self.input_folder, file)
                results = self._converter.convert(file_path)
                document_dict = results.document.export_to_dict()
                if self.extract_tables and self.extract_figures and self.extract_text:
                    results_folder = os.path.join(self.output_folder, file_name)
                else:
                    results_folder = self.output_folder
                image_folder_path = os.path.join(results_folder, f"{file_name}_images")
                if not os.path.exists(image_folder_path):
                    os.makedirs(image_folder_path)
                self.generate_blocks_dicts(document_dict)
                self.generate_blocks_from_dict(document_dict, results_folder, image_folder_path, file_path, file_name)
            number += 1

class Paddle(Pipeline):
    _converter: PPStructureV3 = PrivateAttr(default=PPStructureV3())

    def generate_text_block(self, 
                             block: Dict[str, Any], page_number: int) -> Dict[str, Any]: 
        text_object = Text()
        if block["type"] in ["paragraph", "abstract"]:
            text_object.type = "paragraph"
        else:
            text_object.type = "section_header"
        text_object.content = block["block_content"]
        text_object.page = page_number
        text_object.box = block["block_bbox"]
        return text_object.model_dump()
    
    def genera_equation_block(self,
                             block: Dict[str, Any],
                             image_folder_path: str,
                             file_path:str,
                             number: int,
                             doc_name: str) -> Dict[str, Any]:
        equation_object = Equation()
        equation_object.content = self._latex_parser.latex_to_text(block["block_content"])
        equation_object.page = block["page_number"]
        equation_object.box = block["block_bbox"]
        equation_object.number = number
        equation_object.filepath = self.snap_figure(image_folder_path,
                                                   equation_object.page,
                                                   file_path,
                                                   equation_object.box,
                                                   number,
                                                   doc_name,
                                                   "Equation")
        return equation_object.model_dump()

    def generate_blocks_from_dict(self, output:Any, results_folder: str, image_folder_path: str, file_path: str, file_name) -> None:
        blocks_info: Dict[str, List[Dict[str, Any]]] = {"blocks": []}
        figure_amount: int = 1
        table_amount: int = 1
        equation_amount: int = 1
        get_references: bool = False
        reference_list: List[str] = []
        page_number = 1
        for res in output:
            document_dict: Dict[str, Any] = res.json
            blocks_list = document_dict[res]['parsing_res_list']
            for block in blocks_list:
                block_data: Dict[str, Any] = {}
                if block['type'] in ["paragraph", "paragraph_title", "doc_title", "abstract"] and self.extract_text:
                    text_block = self.generate_text_block(
                            block, page_number
                        )
                    block_data = text_block
                elif block['type'] == 'formula' and self.extract_equations:
                    equation_block = self.genera_equation_block(
                            block,
                            image_folder_path,
                            file_path,
                            equation_amount,
                            file_name
                        )
                    equation_amount += 1
                    block_data = equation_block
                elif block['type'] == 'table' and self.extract_tables:
                    table_object = Table()
                    table_object.number = table_amount
                    table_object.block = block['data']
                    table_object.page = page_number
                    table_object.box = block['box']
                    table_amount += 1
                    block_data = table_object.model_dump()
                if block_data != {}:
                    blocks_info["blocks"].append(block_data)
            page_number += 1
        block_info_json = json.dumps(blocks_info, indent=4)
        with open(results_folder + "/" + f"{file_name}_content.json", "w") as f:
            f.write(block_info_json)

    def pdf_transform(self) -> None:
        files_list = os.listdir(self.input_folder)
        number = 1
        for file in files_list:
            print(f"{number}//{len(files_list)} processed")
            if file.endswith('.pdf'):
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(self.input_folder, file)
                results = self._converter.predict(file_path)
                if self.extract_tables and self.extract_figures and self.extract_text:
                    results_folder = os.path.join(self.output_folder, file_name)
                else:
                    results_folder = self.output_folder
                image_folder_path = os.path.join(results_folder, f"{file_name}_images")
                if not os.path.exists(image_folder_path):
                    os.makedirs(image_folder_path)
            self.generate_blocks_from_dict(results, results_folder, image_folder_path, file_path, file_name)
            number += 1
# -*- coding: utf-8 -*-