from typing import Any, Dict, List, Optional

import fitz
from torch import device
from pdf2data.pipeline import Pipeline, Table, Figure, Text, Equation
from pdf2data.mask import LayoutParser, TableStructureParser
import os
import shutil
from pydantic import BaseModel, PrivateAttr
import PyPDF2
import json
from pdf2data.support import (box_corretor, find_legend, iou, order_horizontal,
                              order_vertical, word_horiz_box_corrector,
                              words_from_line, Latex2Table)

class TableReconstructor(BaseModel):
    iou_threshold: float

    def entry_by_entry(
        self, words_dict: Dict[str, List[Any]], table_dict: Dict[str, List[List[float]]]
    ) -> List[List[str]]:
        """reconstruct the table going entry by entry verifying every word that intercept it

        Parameters
        ----------
        words_dict : Dict[str, List[Any]]
            dictionary containing the words and their coordinates
        table_dict : Dict[str, List[List[float]]]
            dictionary containing the lists of rows and collumns coordinates

        Returns
        -------
        List[List[str]]
            a list of list representing all the table entries
        """
        words: List[str] = words_dict["words"]
        boxes: List[List[float]] = words_dict["boxes"]
        rows: List[List[float]] = table_dict["rows"]
        collumns: List[List[float]] = table_dict["collumns"]
        # Create a table with the desired amount of lines and collumns
        table_entries: List[List[str]] = [
            ["" for i in range(len(collumns))] for j in range(len(rows))
        ]
        table_improvement_possibility: bool = True
        while table_improvement_possibility is True:
            initial_size: int = len(words)
            # print(initial_size)
            for i in range(len(rows)):
                for j in range(len(collumns)):
                    # Determine each table entry coords
                    intersect: List[float] = [
                        collumns[j][0],
                        rows[i][1],
                        collumns[j][2],
                        rows[i][3],
                    ]
                    b: int = 0
                    # It is False while no word match a table entry
                    word_test: bool = False
                    while b < len(boxes) and word_test is False:
                        # Verify if the coorde intesects enough the table entry
                        if iou(intersect, boxes[b]) > self.iou_threshold:
                            word_test = True
                            if table_entries[i][j] != "":
                                table_entries[i][j] = (
                                    table_entries[i][j] + " " + words[b]
                                )
                            else:
                                table_entries[i][j] = words[b]
                            del words[b]
                            del boxes[b]
                        else:
                            b = b + 1
            # print(initial_size)
            if len(words) == 0 or len(words) == initial_size:
                table_improvement_possibility = False
        return table_entries

class PDF2Data(Pipeline):
    layout_model: str
    layout_model_threshold: float
    table_model: Optional[str] =  None
    table_model_threshold: float = 0.7
    table_structure_model: str= "microsoft/table-structure-recognition-v1.1-all"
    device: str
    _mask: Optional[LayoutParser] = PrivateAttr(default=None)
    _structure_mask: Optional[TableStructureParser] = PrivateAttr(default=None)
    _table_reconstructor: Optional[TableReconstructor] = PrivateAttr(default=None)

    def model_post_init(self, context):
        self._mask = LayoutParser(
            model=self.layout_model,
            model_threshold=self.layout_model_threshold,
            table_model=self.table_model,
            table_model_threshold=self.table_model_threshold,
            device_type=self.device
        )
        if self.extract_tables:
            self._structure_mask = TableStructureParser(
                    model=self.table_structure_model
                )
            self._table_reconstructor = TableReconstructor(iou_threshold=0.00001)

    def box_corretor(
        self,
        pdf_size: List[float],
        box: List[float],
        x_corrector: float = 0.01,
        y_corrector: float = 0.01,
    ) -> List[float]:
        """increase the box size depending on the page size

        Parameters
        ----------
        pdf_size : List[float]
            size of the pdf page size
        box : List[float]
            inital box coordinates
        x_corrector : float, optional
            corrector of the x axis, by default 0
        y_corrector : float, optional
            corrector of the y axis, by default 0

        Returns
        -------
        List[float]
            a list with the corrected coordinates
        """
        x_1: float = max(
            pdf_size[0],
            int(float(box[0]) - x_corrector * (float(pdf_size[2] - pdf_size[0]))),
        )
        y_1: float = max(
            pdf_size[1],
            int(float(box[1]) - y_corrector * (float(pdf_size[3] - pdf_size[1]))),
        )
        x_2: float = min(
            pdf_size[2],
            int(float(box[2]) + x_corrector * (float(pdf_size[2] - pdf_size[0]))),
        )
        y_2: float = min(
            pdf_size[3],
            int(float(box[3]) + y_corrector * (float(pdf_size[3] - pdf_size[1]))),
        )
        return x_1, y_1, x_2, y_2
    
    def get_string_from_box(
        self,
        page: Any,
        box_coords: List[float],
        page_size: List[float],
        x_corrector_value: float = 0.01,
        y_corrector_value: float = 0.01,
    ) -> str:
        """retrieve the text inside a box

        Parameters
        ----------
        page : Any
            pdf page
        box_coords : List[float]
            box coordinates
        page_size : List[float]
            size of the pdf size
        x_corrector_value : float, optional
            corrector of the x axis, by default 0.01
        y_corrector_value : float, optional
            corrector of the y axis, by default 0.005

        Returns
        -------
        str
            the text string isnde the box
        """
        # Correct the tablle coordinates acording the the page size
        x_1, y_1, x_2, y_2 = self.box_corretor(
            page_size,
            box_coords,
            x_corrector=x_corrector_value,
            y_corrector=y_corrector_value,
        )
        table_rect: Any = fitz.Rect(x_1, y_1, x_2, y_2)
        # Retrive the text inside the box
        text: str = page.get_text(
            clip=table_rect,
        )
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("- ", " ")
        text = text.replace("\u2010 ", " ")
        text = text.replace("\u00a0", " ")
        text = text.replace(" -", "")
        text = text.replace("  ", " ")
        text = text.replace("   ", " ")
        text = text.replace("    ", " ")
        return text
    
    def get_words_pymupdf(
        self, page_size: List[float], page: Any, table_coords: List[List[float]]
    ) -> Dict[str, List[Any]]:
        """get words from a table using pymupdf

        Parameters
        ----------
        page_size : List[float]
            dimensions of the page
        page : Any
            page where the table is
        table_coords : List[List[float]]
            table coordinates

        Returns
        -------
        Dict[str, List[Any]]
            a dictionary containing the merged words, their respective coordinates and the table size
        """
        # Correct the table coordinates acording the the page size
        x_1 = table_coords[0]
        y_1 = table_coords[1]
        x_2 = table_coords[2]
        y_2 = table_coords[3]
        table_rect: fitz.Rect = fitz.Rect(x_1, y_1, x_2, y_2)
        # Retrive the text dict only in the table
        block_list: Any = page.get_text(
            "dict",
            clip=table_rect,
        )
        words_list: List[str] = []
        box_list: List[List[float]] = []
        page_area: float = abs(
            (page_size[2] - page_size[0]) * (page_size[3] - page_size[1])
        )
        threshold: float = (
            float(abs(page_size[2] - page_size[0]))
        )
        # Go through all dict entries
        for block in block_list["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    words_info: Dict[str, List[Any]] = words_from_line(
                        line, threshold, page_area
                    )
                    words_list = words_list + words_info["words"]
                    box_list = box_list + words_info["boxes"]
        table_coords = [x_1, y_1, x_2, y_2]
        dict: Dict[str, List[Any]] = word_horiz_box_corrector(
            words_list, box_list, table_coords
        )
        return dict

    def generate_text_block(self, doc_page: Any, page: int, block_coords: List[float], block_type: str, page_size: List[float]) -> Dict[str, Any]:
        text_block = Text()
        if block_type == "Text":
            text_block.type = "paragraph"
        elif block_type == "Title":
            text_block.type = "section_header"
        text_block.box = block_coords
        text_block.page = page
        text_block.content = self.get_string_from_box(doc_page, block_coords, page_size)
        return text_block.model_dump()

    def generate_equation_block(self, doc_page: Any, page: int, block_coords: List[float], page_size: List[float], image_folder_path: str, file_path: str, file_name: str, equation_amount: int) -> Dict[str, Any]:
        equation_block = Equation()
        equation_block.box = block_coords
        equation_block.page = page
        equation_block.number = equation_amount
        x_1, y_1, x_2, y_2 = self.box_corretor(
            page_size,
            block_coords,
            x_corrector=0.01,
            y_corrector=0.002,
        )
        corrected_blocks = [x_1, y_1, x_2, y_2]
        equation_block.content = self.get_string_from_box(doc_page, corrected_blocks, page_size)
        equation_block.filepath = self.snap_figure(image_folder_path, page, file_path, corrected_blocks, equation_amount, file_name, "equation")
        return equation_block.model_dump()

    def generate_figure_block(self, doc_page: Any, page: int, index, block_coords: List[float], page_size: List[float], image_folder_path: str, file_path: str, file_name: str, figure_amount: int, blocks_types_list: List[str], blocks_coords_list: List[List[float]]) -> Dict[str, Any]:
        figure_object = Figure()
        figure_object.box = block_coords
        figure_object.page = page
        figure_object.number = figure_amount
        x_1, y_1, x_2, y_2 = self.box_corretor(
            page_size,
            block_coords,
            x_corrector=0.01,
            y_corrector=0.01,
        )
        corrected_blocks = [x_1, y_1, x_2, y_2]
        figure_object.filepath = self.snap_figure(image_folder_path, page, file_path, corrected_blocks, figure_amount, file_name, "figure")
        if index < 1 and len(blocks_types_list) > 1:
            figure_object.caption = ""
            if blocks_types_list[index + 1] == "Figure Caption":
                figure_object.caption = self.get_string_from_box(doc_page, blocks_coords_list[index + 1], page_size)
        elif index >= len(blocks_types_list) - 1:
            figure_object.caption = ""
            if blocks_types_list[index - 1] == "Figure Caption":
                figure_object.caption = self.get_string_from_box(doc_page, blocks_coords_list[index - 1], page_size)
        else:
            if blocks_types_list[index - 1] == "Figure Caption":
                figure_object.caption = self.get_string_from_box(doc_page, blocks_coords_list[index - 1], page_size)
            elif blocks_types_list[index + 1] == "Figure Caption":
                figure_object.caption = self.get_string_from_box(doc_page, blocks_coords_list[index + 1], page_size)
        if index >= len(blocks_types_list) - 1:
            figure_object.footnotes = ""
        else:
            if blocks_types_list[index + 1] == "Table Footnote":
                figure_object.footnotes = self.get_string_from_box(doc_page, blocks_coords_list[index + 1], page_size)
        return figure_object.model_dump()

    def generate_table_block(self, doc_page: Any, page: int, index, block_coords: List[float], page_size: List[float], image_folder_path: str, file_path: str, file_name: str, table_amount: int, blocks_types_list: List[str], blocks_coords_list: List[List[float]]) -> Dict[str, Any]:
        table_object = Table()
        table_object.box = block_coords
        table_object.page = page
        table_object.number = table_amount
        x_1, y_1, x_2, y_2 = self.box_corretor(
            page_size,
            block_coords,
            x_corrector=0.01,
            y_corrector=0.01,
        )
        corrected_coords = [x_1, y_1, x_2, y_2]
        table_structure = (
                self._structure_mask.get_table_structure(
                doc_page, block_coords, corrected_coords
                ))
        entries: Dict[str, List[Any]] = self.get_words_pymupdf(page_size, doc_page, corrected_coords)
        table_object.block = self._table_reconstructor.entry_by_entry(
            entries, table_structure
            )
        table_object.column_headers = self.find_column_headers(table_object.block)
        table_object.row_indexes = self.find_row_indexes(table_object.block)
        table_object.filepath = self.snap_figure(image_folder_path, page, file_path, corrected_coords, table_amount, file_name, "table")
        if index >= len(blocks_types_list) - 1:
            table_object.caption = ""
            if blocks_types_list[index - 1] == "Table Caption":
                table_object.caption = self.get_string_from_box(doc_page, blocks_coords_list[index - 1], page_size)
        elif index < 1 and len(blocks_types_list) > 1:
            table_object.caption = ""
            if blocks_types_list[index + 1] == "Table Caption":
                table_object.caption = self.get_string_from_box(doc_page, blocks_coords_list[index + 1], page_size)
        else:
            if blocks_types_list[index - 1] == "Table Caption":
                table_object.caption = self.get_string_from_box(doc_page, blocks_coords_list[index - 1], page_size)
            elif blocks_types_list[index + 1] == "Table Caption":
                table_object.caption = self.get_string_from_box(doc_page, blocks_coords_list[index + 1], page_size)
        if index >= len(blocks_types_list) - 1:
            table_object.footnotes = ""
        else:
            if blocks_types_list[index + 1] == "Table Footnote":
                table_object.footnotes = self.get_string_from_box(doc_page, blocks_coords_list[index + 1], page_size)
        return table_object.model_dump()

    def generate_blocks_from_dict(self, doc_layout: Dict[str, Any], results_folder: str, image_folder_path: str, file_path: str, file_name: str) -> None:
        figure_amount: int = 1
        table_amount: int = 1
        equation_amount: int = 1
        reference_list: List[str] = []
        page_number: int = 1
        blocks_info: Dict[str, List[Dict[str, Any]]] = {"blocks": []}
        document = fitz.open(file_path)
        pdf = PyPDF2.PdfReader(file_path)
        end_extraction = False
        for page in doc_layout["types"]:
            i = 0
            doc_page = document[page_number - 1]
            page_size = pdf.pages[page_number - 1].mediabox
            for block_type in page:
                box_coords = doc_layout["boxes"][page_number-1][i]
                block_data: Dict[str, Any] = {}
                if block_type in ["Text", "Title"] and self.extract_text:
                    block_data = self.generate_text_block(doc_page, page_number, box_coords, block_type, page_size)
                    if block_data["content"].lower().strip() in ["references", "bibliography", "reference"] and self.layout_model == "DocLayout-YOLO-DocStructBench":
                        end_extraction = True
                        break
                elif block_type == "Equation" and self.extract_equations:
                    block_data = self.generate_equation_block(doc_page, page_number, box_coords, page_size, image_folder_path, file_path, file_name, equation_amount)
                    equation_amount += 1
                elif block_type == "Figure" and self.extract_figures:
                    block_data = self.generate_figure_block(doc_page, page_number, i, box_coords, page_size, image_folder_path, file_path, file_name, figure_amount, page, doc_layout["boxes"][page_number-1])
                    figure_amount += 1
                elif block_type == "Table" and self.extract_tables:
                    block_data = self.generate_table_block(doc_page, page_number, i, box_coords, page_size, image_folder_path, file_path, file_name, table_amount, page, doc_layout["boxes"][page_number-1])
                    table_amount += 1
                elif block_type == "Reference" and self.extract_references:
                    reference_list.extend(self.get_string_from_box(doc_page, box_coords, page_size).split("\n"))
                if block_data != {}:
                    blocks_info["blocks"].append(block_data)
                i += 1
            page_number += 1
            if end_extraction:
                break
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
                doc_layout: Dict[str, Any] = self._mask.get_layout(file_path)
                if self.extract_tables and self.extract_figures and self.extract_text:
                    results_folder = os.path.join(self.output_folder, file_name)
                else:
                    results_folder = self.output_folder
                image_folder_path = os.path.join(results_folder, f"{file_name}_images")
                if not os.path.exists(image_folder_path):
                    os.makedirs(image_folder_path)
                print(file_path)
                self.generate_blocks_from_dict(doc_layout, results_folder, image_folder_path, file_path, file_name)
            number += 1