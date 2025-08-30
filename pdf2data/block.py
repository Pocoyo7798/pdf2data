import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy

import fitz
import PyPDF2
import tensorflow as tf
from pydantic import BaseModel, PrivateAttr
import easyocr

from pdf2data.mask import TableStructureParser, Table2Latex
from pdf2data.support import (box_corretor, find_legend, iou, order_horizontal,
                              order_vertical, word_horiz_box_corrector,
                              words_from_line, Latex2Table)
from paddleocr import TableCellsDetection, TextDetection, PaddleOCR


class TableWords(BaseModel):
    word_proximity_factor: float = 0.01
    x_corrector_value: float = 0.016
    y_corrector_value: float = 0.005
    iou_horizontal: float = 0.3
    iou_vertical: float = 0.05
    word_detection_threshold: float = 0.3
    table_zoom: float = 1.0
    cell_zoom: float = 1.0
    ocr_model : Optional[str] =  None
    word_detection_model : Optional[str] =  None
    _ocr_model: Optional[Any] = PrivateAttr(default=None)
    _word_detection_model: Optional[Any] = PrivateAttr(default=None)
    _existing_ocr_models: set = PrivateAttr(
        default=set(["easy_ocr", "padle_ocr"])
    )
    _existing_detection_models: set = PrivateAttr(
        default=set(["RT-DETR-L_wireless_table_cell_det", "RT-DETR-L_wired_table_cell_det"])
    )

    def model_post_init(self, __context: Any) -> None:
        if self.ocr_model is None:
            pass
        elif self.ocr_model not in self._existing_ocr_models:
            raise AttributeError("The ocr model provided is not valid.")
        elif self.ocr_model == "padle_ocr":
            self._ocr_model = PaddleOCR(
                lang="en", # Specify French recognition model with the lang parameter,
                use_doc_orientation_classify=False, # Disable document orientation classification model
                use_doc_unwarping=False, # Disable text image unwarping model
                use_textline_orientation=False, # Disable text line orientation classification model
            )
        elif self.ocr_model == "easy_ocr":
            self._ocr_model = easyocr.Reader(['en'])
        if self.word_detection_model is None:
            pass
        elif self.word_detection_model not in self._existing_detection_models:
            raise AttributeError("The word detection model provided is not valid.")
        elif self.word_detection_model in set(["RT-DETR-L_wireless_table_cell_det", "RT-DETR-L_wired_table_cell_det"]):
            self._word_detection_model = TableCellsDetection(model_name=self.word_detection_model)

    def get_cells_by_detr_cell(self, page: Any, table_coords: List[List[float]]) -> List[List[float]]:
        """detect the table cell boxes using padl padle cell detection models

        Returns
        -------
        List[List[float]]
            A list containing the boxes from all the table cells found
        """
        x_1 = table_coords[0]
        y_1 = table_coords[1]
        x_2 = table_coords[2]
        y_2 = table_coords[3]
        table_rect: fitz.Rect = fitz.Rect(x_1, y_1, x_2, y_2)
        mat: fitz.Matrix = fitz.Matrix(self.table_zoom, self.table_zoom)
        image: Any = page.get_pixmap(matrix=mat, clip=table_rect)
        pillow_image = Image.frombytes("RGB", [image.width, image.height], image.samples)
        width, height = pillow_image.size
        predictions = self._word_detection_model.predict(numpy.array(pillow_image), threshold=self.word_detection_threshold, batch_size=1)
        cells: List[Dict[str, Any]] = []
        cell_list: List[List[float]] = []
        for prediction in predictions:
            cells.extend(prediction["boxes"])
        for cell in cells:
            if cell["label"] != "cell":
                pass
            else:
                x1: float = table_coords[0] + cell["coordinate"][0] / width * float(table_coords[2] - table_coords[0])
                y1: float = table_coords[1] + cell["coordinate"][1] / height * float(table_coords[3] - table_coords[1])
                x2: float = table_coords[0] + cell["coordinate"][2] / width * float(table_coords[2] - table_coords[0])
                y2: float = table_coords[1] + cell["coordinate"][3] / height * float(table_coords[3] - table_coords[1])
                cell_box = [x1, y1, x2, y2]
                cell_list.append(cell_box)
        return cell_list

    def get_cells_by_structure(self, table_structure: Optional[Dict[str, List[List[float]]]] = None) -> List[List[float]]:
        """create all the table cell boxes from the table rows and collumns coordinates

        Returns
        -------
        List[List[float]]
            A list containing the boxes from all the table cells found
        """
        rows: List[List[float]] = table_structure["rows"]
        columns: List[List[float]] = table_structure["collumns"]
        cell_list: List[List[float]] = []
        for row in rows:
            for column in columns:
                cell: List[float] = [
                        column[0],
                        row[1],
                        column[2],
                        row[3],
                    ]
                cell_list.append(cell)
        return cell_list
    
    def get_word_dict_easyocr(self, page: Any, table_coords: List[float], word_boxes: List[List[float]]):
        words_list: List[str] = []
        i = 0
        for box in word_boxes:
            x_1 = box[0]
            y_1 = box[1]
            x_2 = box[2]
            y_2 = box[3]
            cell_rect: fitz.Rect = fitz.Rect(x_1, y_1, x_2, y_2)
            mat: fitz.Matrix = fitz.Matrix(self.cell_zoom, self.cell_zoom)
            image: Any = page.get_pixmap(matrix=mat, clip=cell_rect)
            pillow_image = Image.frombytes("RGB", [image.width, image.height], image.samples)
            #pillow_image.save(f"table_{table_coords[0]}_{table_coords[1]}_{table_coords[2]}_{table_coords[3]}_cell_{i}.png")
            cell_text_output = self._ocr_model.readtext(numpy.array(pillow_image))
            text_list = []
            for text in cell_text_output:
                 text_list.extend(text[1])
            words_list.append(" ".join(text_list))
            i += 1
        return {"words": words_list, "boxes": word_boxes, "table_box": table_coords}

    def get_word_dict_padle(self, page: Any, table_coords: List[float], word_boxes: List[List[float]]):
        words_list: List[str] = []
        i = 0
        for box in word_boxes:
            x_1 = box[0]
            y_1 = box[1]
            x_2 = box[2]
            y_2 = box[3]
            cell_rect: fitz.Rect = fitz.Rect(x_1, y_1, x_2, y_2)
            mat: fitz.Matrix = fitz.Matrix(self.cell_zoom, self.cell_zoom)
            image: Any = page.get_pixmap(matrix=mat, clip=cell_rect)
            pillow_image = Image.frombytes("RGB", [image.width, image.height], image.samples)
            #pillow_image.save(f"table_{table_coords[0]}_{table_coords[1]}_{table_coords[2]}_{table_coords[3]}_cell_{i}.png")
            cell_text_output = self._ocr_model.predict(numpy.array(pillow_image))
            text_list = []
            for text in cell_text_output:
                 text_list.extend(text["rec_texts"])
            words_list.append(" ".join(text_list))
            i += 1
        return {"words": words_list, "boxes": word_boxes, "table_box": table_coords}
    
    def get_words_ocr(
        self, page_size: List[float], page: Any, table_coords: List[float], table_structure: Optional[Dict[str, List[List[float]]]] = None
    ) -> Dict[str, List[Any]]:
        """get words from a table using ocr

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
        if self.word_detection_model is None:
            word_boxes = self.get_cells_by_structure(table_structure)
        elif self.word_detection_model == "RT-DETR-L_wireless_table_cell_det":
            word_boxes = self.get_cells_by_detr_cell(page, table_coords)
        if self.ocr_model == "padle_ocr":
            dict = self.get_word_dict_padle(page, table_coords, word_boxes)
        elif self.ocr_model == "easy_ocr":
            dict = self.get_word_dict_easyocr(page, table_coords, word_boxes)
        return dict
    
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
            float(abs(page_size[2] - page_size[0])) * self.word_proximity_factor
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

    def get_words(
        self, page_size: List[float], page: Any, table_coords: List[List[float]], table_structure: Optional[Dict[str, List[List[float]]]] = None
    ) -> Dict[str, List[Any]]:
        """get words from a table

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
        if self.ocr_model is not None:
            dict: Dict[str, List[Any]] = self.get_words_ocr(page_size, page, table_coords, table_structure)
        else:
            dict = self.get_words_pymupdf(page_size, page, table_coords)
        return dict

    def table_struture_with_boxes(
        self, boxes: List[List[float]], table_coords: List[float], page
    ) -> Dict[str, List[List[float]]]:
        """generate the table structure from the word coordinates

        Parameters
        ----------
        boxes : List[List[float]]
            list of word boxes
        table_coords : List[float]
            table coordinates

        Returns
        -------
        Dict[str, List[List[float]]]
            a dictionary containing the table rows and collumns coordinates
        """
        horizontal_boxes: List[List[float]] = []
        vertical_boxes: List[List[float]] = []
        if self.ocr_model is None and self.word_detection_model is not None:
            self.get_cells_by_detr_cell(page, table_coords)
        for box in boxes:
            # Generate the coordenates of words collumns and lines
            x1_horizontal, x1_vertical = float(table_coords[0]), float(box[0])
            y1_horizontal, y1_vertical = float(box[1]), float(table_coords[1])
            x2_horizontal, x2_vertical = float(table_coords[2]), float(box[2])
            y2_horizontal, y2_vertical = float(box[3]), float(table_coords[3])
            horizontal_boxes.append(
                [x1_horizontal, y1_horizontal, x2_horizontal, y2_horizontal]
            )
            vertical_boxes.append([x1_vertical, y1_vertical, x2_vertical, y2_vertical])
        if len(horizontal_boxes) > 0:
            probabilities_horizontal: List[float] = []
            # Create the list of higher probability of being a line
            for box in horizontal_boxes:
                # This way smaller rectangles are more important
                probabilities_horizontal.append(1 / (box[3] - box[1]))
            horiz_supressed_index: List[int] = tf.image.non_max_suppression(
                horizontal_boxes,
                probabilities_horizontal,
                max_output_size=1000,
                iou_threshold=self.iou_horizontal,
                score_threshold=float("-inf"),
                name=None,
            )
            horizontal_lines: List[List[float]] = []
            for index in horiz_supressed_index:
                horizontal_lines.append(horizontal_boxes[index])
        else:
            horizontal_lines = horizontal_boxes
        if len(vertical_boxes) > 0:
            probabilities_vertical: List[float] = []
            # Create the list of higher probability of being a collumn
            for box in vertical_boxes:
                # This way smaller rectangles are more important
                probabilities_vertical.append(1 / (box[2] - box[0]))
            vert_supressed_index: List[int] = tf.image.non_max_suppression(
                vertical_boxes,
                probabilities_vertical,
                max_output_size=1000,
                iou_threshold=self.iou_vertical,
                score_threshold=float("-inf"),
                name=None,
            )
            vertical_lines: List[List[float]] = []
            for index in vert_supressed_index:
                vertical_lines.append(vertical_boxes[index])
        else:
            vertical_lines = vertical_boxes
        # Order the index from down to bottom
        ordered_rows: List[List[float]] = order_horizontal(horizontal_lines)
        # Order the indexes from left to right
        ordered_collumns: List[List[float]] = order_vertical(vertical_lines)
        return {"rows": ordered_rows, "collumns": ordered_collumns}


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

    def word_by_word(
        self, words_dict: Dict[str, List[Any]], table_dict: Dict[str, List[List[float]]]
    ) -> List[List[str]]:
        """reconstruct the table going word by word and verifying the first entry that intercepts it

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
        b: int = 0
        for word in words:
            found_entry: bool = False
            for i in range(len(rows)):
                for j in range(len(collumns)):
                    # Determine each table entry coords
                    intersect: float = [
                        collumns[j][0],
                        rows[i][1],
                        collumns[j][2],
                        rows[i][3],
                    ]
                    # Verify if the coorde intesects enough the table entry
                    if iou(intersect, boxes[b]) > self.iou_threshold:
                        if table_entries[i][j] != "":
                            table_entries[i][j] = table_entries[i][j] + " " + word
                        else:
                            table_entries[i][j] = word
                        found_entry = True
                        break
                if found_entry is True:
                    break
            b = b + 1
        return table_entries


class Table(BaseModel):
    page: int
    name: str
    block: List[List[str]]
    type: str = "Table"
    collumn_headers: List[int] = []
    row_indexes: List[int] = []
    number: int = 0
    legend: str = ""
    box: List[float] = []
    letter_ratio: float = 3

    def find_collumn_headers(self) -> None:
        """find the collumn headers as rows that do not have numbers"""
        if len(self.block) == 0:
            pass
        elif len(self.block[0]) == 0:
            pass
        else:
            collumn_headers: List[int] = []
            find_number: bool = True
            for row_number in range(len(self.block)):
                if find_number is False:
                    collumn_headers.append(row_number - 1)
                find_number = False
                for entry in self.block[row_number]:
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
            self.collumn_headers = collumn_headers

    def find_row_indexes(self, max_rows: int = 2) -> None:
        """find the row indexes by finding collumns without entries with three times more digits then letters

        Parameters
        ----------
        max_rows : int, optional
            maximum rows to be considered, by default 2
        """
        row_indexes: List[int] = []
        find_number: bool = True
        if len(self.block) == 0:
            pass
        elif len(self.block) == 0:
            pass
        else:
            max_rows: int = min(len(self.block[0]), max_rows)
            for collumn_number in range(max_rows):
                find_number = False
                for row in self.block:
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
            self.row_indexes = row_indexes

    def create_dict(
        self,
        page: Any,
        page_size: List[float],
        layout_boxes: List[List[float]],
        layout_types: List[str],
        index: int,
    ) -> Dict[str, Any]:
        """generates a dictionary describing the table object

        Parameters
        ----------
        page : Any
            pdf page to be considered
        page_size : List[float]
            size of the page
        boxes : List[List[float]]
            list of boxes of the page layout
        types : List[str]
            list of the types of the page layout
        index : int
            position of the table or figure in the layout list

        Returns
        -------
        Dict[str, Any]
            a dictionary with the page number, table entries, type, collumn headers, row indexes, table number, legend and table coordinates
        """
        i_vertical: int = 1
        # Go through all entries in the table
        self.find_collumn_headers()
        self.find_row_indexes()
        self.legend = find_legend(
            page, page_size, layout_boxes, layout_types, index, type=self.type
        )
        for i_horizontal in range(len(self.block)):
            j_horizontal: int = 0
            for j_vertical in range(len(self.block[i_horizontal])):
                if i_vertical < len(self.block):
                    # Verify if the entry is empty
                    if self.block[i_vertical][j_vertical] == "":
                        # New entry is the one above
                        new_entry_vert: str = self.block[i_vertical - 1][j_vertical]
                        if (
                            re.search("[a-zA-Z]", new_entry_vert) is not None
                            or j_vertical == 0
                        ):
                            self.block[i_vertical][j_vertical] = new_entry_vert
                if j_horizontal < len(self.block[i_horizontal]) and j_horizontal > 0:
                    if self.block[i_horizontal][j_horizontal] == "":
                        # New entry is the one on the left
                        new_entry_horiz = self.block[i_horizontal][j_horizontal - 1]
                        self.block[i_horizontal][j_horizontal] = new_entry_horiz
                j_horizontal = j_horizontal + 1
            i_vertical = i_vertical + 1
        image_rect: fitz.Rect = fitz.Rect(
            self.box[0], self.box[1], self.box[2], self.box[3]
        )
        mat: fitz.Matrix = fitz.Matrix(3, 3)
        # Get Image from the Rectangle
        image: Any = page.get_pixmap(matrix=mat, clip=image_rect)
        # Save as Tiff
        image.pil_save(self.name, format="TIFF")
        result = self.__dict__
        del result["letter_ratio"]
        return result


class Figure(BaseModel):
    page: int
    name: str
    type: str = "Figure"
    number: int = 0
    legend: str = ""
    box: List[float] = []

    def create_dict(
        self,
        page: Any,
        page_size: List[float],
        layout_boxes: List[List[float]],
        layout_types: List[str],
        index: int,
        zoom: float = 1,
    ) -> Dict[str, Any]:
        """generates a dictonary describing the figure object

        Parameters
        ----------
        page : Any
            pdf page to be considered
        page_size : List[float]
            size of the oage
        boxes : List[List[float]]
            list of boxes of the page layout
        types : List[str]
            list of the types of the page layout
        index : int
            position of the table or figure in the layout list
        zoom : float, optional
            image zoom, by default 1

        Returns
        -------
        Dict[str, Any]
            a dictionary with the page number, figure name, type, figure number, legend and figure coordinates
        """
        self.legend = find_legend(
            page, page_size, layout_boxes, layout_types, index, type=self.type
        )
        image_rect: fitz.Rect = fitz.Rect(
            self.box[0], self.box[1], self.box[2], self.box[3]
        )
        mat: fitz.Matrix = fitz.Matrix(3, 3)
        # Get Image from the Rectangle
        image: Any = page.get_pixmap(matrix=mat, clip=image_rect)
        # Save as Tiff
        image.pil_save(self.name, format="TIFF")
        return self.__dict__


class BlockExtractor(BaseModel):
    extract_tables: bool = True
    extract_figures: bool = True
    correct_struct: bool = True
    table_zoom: float = 1.5
    figure_zoom: float = 1
    x_table_corr: float = 0.015
    y_table_corr: float = 0.015
    iou_lines: float = 0.05
    iou_struct: float = 0.02
    word_factor: float = 1
    word_iou: float = 0.04
    brightness: float = 1
    contrast: float = 1
    letter_ratio: float = 4
    reconstructor_type: str = "entry_by_entry"
    structure_model: Optional[str] = "microsoft/table-transformer-structure-recognition"
    struct_model_threshold: float = 0.3
    ocr_model : Optional[str] =  None
    word_detection_model : Optional[str] =  None
    word_detection_threshold: float = 0.3
    cell_zoom: float = 1.0
    _structure_parser: Optional[TableStructureParser] = PrivateAttr(default=None)
    _word_extractor: Optional[TableWords] = PrivateAttr(default=None)
    _reconstructor: TableReconstructor = PrivateAttr(default=None)
    _table2latex_model: Table2Latex = PrivateAttr(default=None)
    _latex_parser: Latex2Table = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.structure_model in set(["microsoft/table-transformer-structure-recognition", "microsoft/table-structure-recognition-v1.1-all", "RT-DETR-L_wireless_table_cell_det", "RT-DETR-L_wired_table_cell_det"]):
            self._reconstructor = TableReconstructor(iou_threshold=self.word_iou)
            self._structure_parser = TableStructureParser(
                model=self.structure_model,
                model_threshold=self.struct_model_threshold,
                zoom=self.table_zoom,
                iou_lines=self.iou_lines,
                iou_struct=self.iou_struct,
                brightness=self.brightness,
                contrast = self.contrast
            )
            self._word_extractor = TableWords(word_proximity_factor=self.word_factor, ocr_model=self.ocr_model, word_detection_model=self.word_detection_model, word_detection_threshold=self.word_detection_threshold, cell_zoom=self.cell_zoom)
        elif self.structure_model == "U4R/StructTable-InternVL2-1B":
            self._table2latex_model = Table2Latex(model_name=self.structure_model,
                zoom=self.table_zoom,)
            self._latex_parser = Latex2Table()
        else:
            self._reconstructor = TableReconstructor(iou_threshold=self.word_iou)
            self._word_extractor = TableWords(word_proximity_factor=self.word_factor, ocr_model=self.ocr_model, word_detection_model=self.word_detection_model, word_detection_threshold=self.word_detection_threshold, cell_zoom=self.cell_zoom)

    def generate_block_structure_detection(self, document, j, initial_box, corrected_box, pdf_size, page):
        if self.structure_model is None:
            entries: Dict[str, List[Any]] = self._word_extractor.get_words(
                            pdf_size, page, corrected_box
                        )
            if self.word_detection_model is not None:
                word_boxes = self._word_extractor.get_cells_by_detr_cell(page, corrected_box)
            else:
                word_boxes = entries["boxes"]
            table_structure: Dict[str, List[List[float]]] = self._word_extractor.table_struture_with_boxes(
                                word_boxes, entries["table_box"], page)
        elif self.correct_struct is True:
            if self.word_detection_model is None and self.ocr_model is not None:
                raise AttributeError("You need to pass a word detection model to use ocr to correct the structure")
            entries: Dict[str, List[Any]] = self._word_extractor.get_words(
                            pdf_size, page, corrected_box
                        )
            word_structure: Dict[str, List[List[float]]] = self._word_extractor.table_struture_with_boxes(
                                entries["boxes"], entries["table_box"], page
                            )
            table_structure: Dict[
                str, List[List[float]]
                ] = self._structure_parser.get_table_structure(
                document, j, initial_box, corrected_box, word_boxes=word_structure
                )
        else:
            table_structure = (
                self._structure_parser.get_table_structure(
                document, j, initial_box, corrected_box
                ))
            entries: Dict[str, List[Any]] = self._word_extractor.get_words(
                            pdf_size, page, corrected_box, table_structure=table_structure
                        )
        if self.reconstructor_type == "entry_by_entry":
            block: List[List[str]] = self._reconstructor.entry_by_entry(
            entries, table_structure
            )
        elif self.reconstructor_type == "word_by_word":
            block = self._reconstructor.word_by_word(
            entries, table_structure
            )
        else:
            raise AttributeError(
                "The reconstructor_type is not valid. Try 'entry_by_entry' or 'word_by_word'"
            )
        return block

    def generate_block_llm(self, 
        pdf: Any,
        page_index: int,
        table_coords: List[List[float]]):
        table = self._table2latex_model.generate_latex(pdf, page_index, table_coords)
        block = self._latex_parser.extract_latex_table(table)
        return block

    def get_blocks(
        self, file_path: str, layout: Dict[str, Any], output_folder: str, doi: str = ""
    ) -> List[str]:
        """Generate a json file, containing info about images and tables and a folder containing the figures found

        Parameters
        ----------
        file_path : str
            path to the pdf file
        layout : Dict[str, Any]
            layout information about the pdf file
        output_folder : str
            output folder path
        doi : str, optional
           doi of the scientific paper, by default ""

        Returns
        -------
        List[str]
            a list containing all the legends found

        Raises
        ------
        AttributeError
            if the file is not in pdf
        Warning
            if both extract_figures and extract_tables is False
        AttributeError
            if the reconstructor_type is not 'entry_by_entry' or 'word_by_word'
        """
        extension = os.path.splitext(file_path)[1]
        if extension != ".pdf":
            raise AttributeError("The file provided is not in pdf format")
        pdf = PyPDF2.PdfReader(file_path)
        document = fitz.open(file_path)
        block_list: List[Dict[str, Any]] = []
        check: List[str] = ["Table", "Table and Image", "Image"]
        table_number: int = 0
        image_number: int = 0
        legends: List[str] = []
        file_name: str = Path(file_path).stem
        if self.extract_tables is False and self.extract_figures is False:
            raise Warning("There is no indication to extract blocks")
        if (
            self.extract_figures is True
            and os.path.exists(f"{output_folder}/{file_name}_images") is False
        ):
            os.mkdir(f"{output_folder}/{file_name}_images")
        for j in range(len(layout["boxes"])):
            pdf_page: Any = pdf.pages[j]
            pdf_size: List[float] = pdf_page.mediabox
            page: Any = document[j]
            if layout["page_type"][j] in check:
                page_boxes: List[List[float]] = layout["boxes"][j]
                page_types: List[str] = layout["types"][j]
                for i in range(len(page_boxes)):
                    if page_types[i] == "Table" and self.extract_tables is True:
                        table_number: int = table_number + 1
                        table_name: str = f"Table{table_number}.tiff"
                        initial_box: List[float] = page_boxes[i]
                        x_1, y_1, x_2, y_2 = box_corretor(
                            pdf_size,
                            initial_box,
                            x_corrector=self.x_table_corr,
                            y_corrector=self.y_table_corr,
                        )
                        corrected_box: List[float] = [x_1, y_1, x_2, y_2]
                        if self.structure_model in set(["microsoft/table-transformer-structure-recognition", "microsoft/table-structure-recognition-v1.1-all", "RT-DETR-L_wireless_table_cell_det", "RT-DETR-L_wired_table_cell_det", None]):
                            block = self.generate_block_structure_detection(document, j, initial_box, corrected_box, pdf_size, page)
                        elif self.structure_model == "U4R/StructTable-InternVL2-1B":
                            block = self.generate_block_llm(document, j, corrected_box)
                        table: Table = Table(name=table_name,
                            page=j + 1, block=block, number=table_number, box=initial_box, letter_ratio=self.letter_ratio
                        )
                        block_dict: Dict[str, Any] = table.create_dict(
                            page, pdf_size, page_boxes, page_types, i
                        )
                        legends.append(table.legend)
                        block_list.append(block_dict)
                        shutil.move(
                            table_name,
                            f"{output_folder}/{file_name}_images/" + table_name,
                        )
                    elif page_types[i] == "Figure" and self.extract_figures is True:
                        box: List[float] = page_boxes[i]
                        image_number: int = image_number + 1
                        image_name: str = f"Figure{image_number}.tiff"
                        figure: Figure = Figure(
                            page=j + 1, name=image_name, number=image_number, box=box
                        )
                        block_dict: Dict[str, Any] = figure.create_dict(
                            page,
                            pdf_size,
                            page_boxes,
                            page_types,
                            i,
                            zoom=self.figure_zoom,
                        )
                        legends.append(figure.legend)
                        block_list.append(block_dict)
                        shutil.move(
                            image_name,
                            f"{output_folder}/{file_name}_images/" + image_name,
                        )
        number = image_number + table_number
        results_dict: Dict[str, Any] = {
            "blocks": block_list,
            "amount": number,
            "doi": doi,
        }
        json_results: Any = json.dumps(results_dict, indent=4)
        with open(f"{output_folder}/{file_name}_blocks.json", "w") as j:
            j.write(json_results)
        document.close()
        return legends
