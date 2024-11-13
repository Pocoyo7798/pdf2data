import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import PyPDF2
import tensorflow as tf
from pydantic import BaseModel, PrivateAttr

from pdf2data.mask import TableStructureParser
from pdf2data.support import (box_corretor, find_legend, iou, order_horizontal,
                              order_vertical, word_horiz_box_corrector,
                              words_from_line)


class TableWords(BaseModel):
    word_proximity_factor: float = 0.01
    x_corrector_value: float = 0.016
    y_corrector_value: float = 0.005
    iou_horizontal: float = 0.3
    iou_vertical: float = 0.05

    def get_words(
        self, page_size: List[float], page: Any, table_coords: List[List[float]]
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
        # Correct the table coordinates acording the the page size
        x_1, y_1, x_2, y_2 = box_corretor(
            page_size,
            table_coords,
            x_corrector=self.x_corrector_value,
            y_corrector=self.y_corrector_value,
        )
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

    def table_struture_with_boxes(
        self, boxes: List[List[float]], table_coords: List[float]
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
        probabilities_horizontal: List[float] = []
        # Create the list of higher probability of being a line
        for box in horizontal_boxes:
            # This way smaller rectangles are more important
            probabilities_horizontal.append(1 / (box[3] - box[1]))
        probabilities_vertical: List[float] = []
        # Create the list of higher probability of being a collumn
        for box in vertical_boxes:
            # This way smaller rectangles are more important
            probabilities_vertical.append(1 / (box[2] - box[0]))
        horiz_supressed_index: List[int] = tf.image.non_max_suppression(
            horizontal_boxes,
            probabilities_horizontal,
            max_output_size=1000,
            iou_threshold=self.iou_horizontal,
            score_threshold=float("-inf"),
            name=None,
        )
        vert_supressed_index: List[int] = tf.image.non_max_suppression(
            vertical_boxes,
            probabilities_vertical,
            max_output_size=1000,
            iou_threshold=self.iou_vertical,
            score_threshold=float("-inf"),
            name=None,
        )
        vertical_lines: List[List[float]] = []
        horizontal_lines: List[List[float]] = []
        for index in horiz_supressed_index:
            horizontal_lines.append(horizontal_boxes[index])
        for index in vert_supressed_index:
            vertical_lines.append(vertical_boxes[index])
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
            size of the oage
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
        mat: fitz.Matrix = fitz.Matrix(1, 1)
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
        mat: fitz.Matrix = fitz.Matrix(zoom, zoom)
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
    structure_model: str = "microsoft/table-transformer-structure-recognition"
    struct_model_threshold: float = 0.3
    _structure_parser: Optional[TableStructureParser] = PrivateAttr(default=None)
    _word_extractor: Optional[TableWords] = PrivateAttr(default=None)
    _reconstructor: TableReconstructor = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._reconstructor = TableReconstructor(iou_threshold=self.word_iou)
        self._structure_parser = TableStructureParser(
            model=self.structure_model,
            model_threshold=self.struct_model_threshold,
            x_corrector_value=self.x_table_corr,
            y_corrector_value=self.y_table_corr,
            zoom=self.table_zoom,
            iou_lines=self.iou_lines,
            iou_struct=self.iou_struct,
            brightness=self.brightness,
            contrast = self.contrast
        )
        self._word_extractor = TableWords(word_proximity_factor=self.word_factor)
        self._structure_parser.model_post_init(None)

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
                        box: List[float] = page_boxes[i]
                        entries: Dict[str, List[Any]] = self._word_extractor.get_words(
                            pdf_size, page, box
                        )
                        if self.correct_struct is True:
                            word_structure: Dict[
                                str, List[List[float]]
                            ] = self._word_extractor.table_struture_with_boxes(
                                entries["boxes"], entries["table_box"]
                            )
                            table_structure: Dict[
                                str, List[List[float]]
                            ] = self._structure_parser.get_table_structure(
                                document, j, page_boxes[i], word_boxes=word_structure
                            )
                        else:
                            table_structure = (
                                self._structure_parser.get_table_structure(
                                    document, j, box
                                )
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
                        table: Table = Table(name=table_name,
                            page=j + 1, block=block, number=table_number, box=box, letter_ratio=self.letter_ratio
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
