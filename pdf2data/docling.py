from typing import Any, Dict, List
from pdf2data.pipeline import Pipeline, Table, Figure, Text, Equation
from pydantic import  PrivateAttr
import json
import os
from docling.document_converter import DocumentConverter

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