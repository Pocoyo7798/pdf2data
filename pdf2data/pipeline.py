import os
import re
from typing import Any, Dict, List, Optional
from pylatexenc.latex2text import LatexNodes2Text

import fitz
from pydantic import BaseModel, PrivateAttr
from bs4 import BeautifulSoup


class Table(BaseModel):
    type: str = "Table"
    filepath: Optional[str] = None
    number: Optional[int] = None
    caption: str = ""
    block: Optional[List[List[str]]] = None
    # Per-cell coordinates aligned with `block` rows and columns.
    cell_boxes: Optional[List[List[Optional[List[float]]]]] = None
    caption_box: Optional[List[float]] = None
    footnotes: Optional[str] = None
    column_headers: Optional[List[int]] = None
    row_indexes: Optional[List[int]] = None
    page: Optional[int] = None
    box: Optional[List[float]] = None

class Figure(BaseModel):
    type: str = "Figure"
    filepath: Optional[str] = None
    number: Optional[int] = None
    caption: str = ""
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

    def get_uniform_cell_boxes(
        self,
        table_box: Optional[List[float]],
        table_block: Optional[List[List[str]]],
    ) -> Optional[List[List[Optional[List[float]]]]]:
        """Approximate cell boxes by splitting the table box into a regular grid."""
        if table_box is None or table_block is None or len(table_block) == 0:
            return None
        if len(table_box) < 4:
            return None

        row_count = len(table_block)
        col_count = max((len(row) for row in table_block), default=0)
        if row_count == 0 or col_count == 0:
            return None

        x1, y1, x2, y2 = [float(v) for v in table_box[:4]]
        row_height = (y2 - y1) / row_count
        col_width = (x2 - x1) / col_count

        cell_boxes: List[List[Optional[List[float]]]] = []
        for row_index, row in enumerate(table_block):
            row_boxes: List[Optional[List[float]]] = []
            cell_count = len(row)
            for col_index in range(cell_count):
                row_boxes.append(
                    [
                        x1 + col_index * col_width,
                        y1 + row_index * row_height,
                        x1 + (col_index + 1) * col_width,
                        y1 + (row_index + 1) * row_height,
                    ]
                )
            cell_boxes.append(row_boxes)
        return cell_boxes

    def get_cell_boxes_from_structure(
        self,
        rows: List[List[float]],
        columns: List[List[float]],
        table_block: Optional[List[List[str]]] = None,
    ) -> Optional[List[List[Optional[List[float]]]]]:
        """Build per-cell boxes from detected table row/column boxes."""
        if len(rows) == 0 or len(columns) == 0:
            return None

        max_rows = len(rows)
        max_cols = len(columns)
        if table_block is None:
            target_rows = max_rows
            row_lengths = [max_cols] * max_rows
        else:
            target_rows = min(len(table_block), max_rows)
            row_lengths = [min(len(row), max_cols) for row in table_block[:target_rows]]

        cell_boxes: List[List[Optional[List[float]]]] = []
        for row_index in range(target_rows):
            row_boxes: List[Optional[List[float]]] = []
            for col_index in range(row_lengths[row_index]):
                row_box = rows[row_index]
                col_box = columns[col_index]
                row_boxes.append(
                    [
                        float(col_box[0]),
                        float(row_box[1]),
                        float(col_box[2]),
                        float(row_box[3]),
                    ]
                )
            cell_boxes.append(row_boxes)
        return cell_boxes

    def merge_boxes(self, boxes: List[List[float]]) -> Optional[List[float]]:
        """Return the enclosing box that contains all provided boxes."""
        valid_boxes = [box for box in boxes if box is not None and len(box) >= 4]
        if len(valid_boxes) == 0:
            return None
        return [
            min(float(box[0]) for box in valid_boxes),
            min(float(box[1]) for box in valid_boxes),
            max(float(box[2]) for box in valid_boxes),
            max(float(box[3]) for box in valid_boxes),
        ]
    
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

    def correct_box_size(self, box_size: List[float], page_size: tuple, file_path: str, page: int) -> List[float]:
        """correct the box size if it is out of the page bounds"""
        page_width, page_height = page_size
        pdf_document = fitz.open(file_path)
        page = pdf_document[page - 1]  # Pages are 0-indexed in fitz
        real_page_rect = page.rect
        real_width = real_page_rect.width
        real_height = real_page_rect.height
        new_box = [box_size[0] * real_width / page_width, box_size[1] * real_height / page_height,
                            box_size[2] * real_width / page_width, box_size[3] * real_height / page_height]
        pdf_document.close()
        return new_box

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
        if self.find_legend_in_row(table_list[0]):
            table_list = table_list[1:]
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
            for j in range(len(row)):
                row[j] = self._latex_parser.latex_to_text(row[j])
            corrected_table.append(row)
        return corrected_table
    
    def find_legend_in_row(self, row: List[str]) -> Optional[str]:
        caption_keywords = ["table", "tab.", "table."]
        same_entry = True
        entry_string = None
        for entry in row:
            if entry_string is None:
                entry_string = entry
            elif entry_string != entry:
                same_entry = False
                break
        legend_found = False
        if same_entry is True:
            for keyword in caption_keywords:
                if keyword in entry_string.lower():
                    legend_found = True
                    break
        return legend_found

    def html_table_to_list(self, html):
        if html == "":
            return [[]]
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        matrix = []          # final output
        rowspans = {}        # (row, col) → remaining rowspan cells

        for row_idx, row in enumerate(table.find_all("tr")):
            cols = []
            col_idx = 0

            # Fill in cells carried over by rowspan
            while (row_idx, col_idx) in rowspans:
                cols.append(rowspans[(row_idx, col_idx)])
                del rowspans[(row_idx, col_idx)]
                col_idx += 1

            for cell in row.find_all(["td", "th"]):
                value = cell.get_text(strip=True)
                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))

                # Add cell and all colspan duplicates
                for _ in range(colspan):
                    cols.append(value)

                # Store rowspan duplicates for future rows
                if rowspan > 1:
                    for rs in range(1, rowspan):
                        for cs in range(colspan):
                            rowspans[(row_idx + rs, col_idx + cs)] = value

                col_idx += colspan

            matrix.append(cols)

        return matrix
# -*- coding: utf-8 -*-