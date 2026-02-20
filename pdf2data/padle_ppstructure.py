from typing import Any, Dict, List
from pdf2data.pipeline import Pipeline, Table, Figure, Text, Equation
from pydantic import PrivateAttr
import json
import os
from paddleocr import PPStructureV3

class PaddlePPStructure(Pipeline):
    _converter: PPStructureV3 = PrivateAttr(default=None)

    def model_post_init(self, context):
        self._converter = PPStructureV3(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_chart_recognition=False,)
    def generate_text_block(self, 
                             block: Dict[str, Any], page_number: int) -> Dict[str, Any]: 
        text_object = Text()
        if block["block_label"] in ["paragraph", "abstract", "text"]:
            text_object.type = "paragraph"
        else:
            text_object.type = "section_header"
        text_object.content = block["block_content"]
        text_object.page = page_number
        text_object.box = block["block_bbox"]
        return text_object.model_dump()
    
    def generate_equation_block(self,
                             block: Dict[str, Any],
                             image_folder_path: str,
                             file_path:str,
                             number: int,
                             doc_name: str,
                             page_number: int,
                             page_size: tuple) -> Dict[str, Any]:
        equation_object = Equation()
        equation_object.content = self._latex_parser.latex_to_text(block["block_content"])
        equation_object.page = page_number
        equation_object.box = block["block_bbox"]
        equation_object.number = number
        equation_object.filepath = self.snap_figure(image_folder_path,
                                                   equation_object.page,
                                                   file_path,
                                                   equation_object.box,
                                                   number,
                                                   doc_name,
                                                   "Equation",
                                                   page_size = page_size)
        return equation_object.model_dump()
    
    def generate_figure_block(self,
                             block: Dict[str, Any],
                             image_folder_path: str,
                             file_path:str,
                             number: int,
                             doc_name: str,
                             page_number: int,
                             page_size: tuple,
                             index: int,
                             blocks_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        figure_object = Figure()
        figure_object.page = page_number
        figure_object.box = block["block_bbox"]
        figure_object.number = number
        figure_object.filepath = self.snap_figure(image_folder_path,
                                                   figure_object.page,
                                                   file_path,
                                                   figure_object.box,
                                                   number,
                                                   doc_name,
                                                   "Figure",
                                                   page_size = page_size)
        if index < 1 or index >= len(blocks_list) - 1:
            figure_object.caption = ""
        elif blocks_list[index - 1]["block_label"] == "figure_title":
            figure_object.caption = blocks_list[index - 1]["block_content"]
        elif blocks_list[index + 1]["block_label"] == "figure_title":
            figure_object.caption = blocks_list[index + 1]["block_content"]
        else:            
            figure_object.caption = ""
        return figure_object.model_dump()
    
    def generate_table_block(self,
                             block: Dict[str, Any],
                             image_folder_path: str,
                             file_path:str,
                             number: int,
                             doc_name: str,
                             page_number: int,
                             page_size: tuple,
                             index: int,
                             blocks_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        table_object = Table()
        table_object.page = page_number
        table_object.box = block["block_bbox"]
        table_object.number = number
        table_object.filepath = self.snap_figure(image_folder_path,
                                                   table_object.page,
                                                   file_path,
                                                   table_object.box,
                                                   number,
                                                   doc_name,
                                                   "Table",
                                                   page_size = page_size)
        if index < 1 or index >= len(blocks_list) - 1:
            table_object.caption = ""
        elif blocks_list[index + 1]["block_label"] == "table_title":
            table_object.caption = blocks_list[index + 1]["block_content"]
        elif blocks_list[index - 1]["block_label"] == "table_title":
            table_object.caption = blocks_list[index - 1]["block_content"]
        else:            
            table_object.caption = ""
        table_object.block = self.html_table_to_list(block["block_content"])
        table_object.block = self.correct_table_structure(table_object.block)
        table_object.column_headers = self.find_column_headers(table_object.block)
        table_object.row_indexes = self.find_row_indexes(table_object.block)
        return table_object.model_dump()


    def generate_blocks_from_dict(self, output:Any, results_folder: str, image_folder_path: str, file_path: str, file_name) -> None:
        blocks_info: Dict[str, List[Dict[str, Any]]] = {"blocks": []}
        figure_amount: int = 1
        table_amount: int = 1
        equation_amount: int = 1
        reference_list: List[str] = []
        page_number = 1
        for res in output:
            document_dict: Dict[str, Any] = res.json
            page_width = document_dict["res"]["width"]
            page_height = document_dict["res"]["height"]
            page_size = (page_width, page_height)
            blocks_list = document_dict["res"]['parsing_res_list']
            index = 0
            for block in blocks_list:
                block_data: Dict[str, Any] = {}
                if block["block_label"] in ["paragraph", "paragraph_title", "doc_title", "abstract", "text"] and self.extract_text:
                    text_block = self.generate_text_block(
                            block, page_number
                        )
                    block_data = text_block
                elif block["block_label"] == "formula" and self.extract_equations:
                    equation_block = self.generate_equation_block(
                            block,
                            image_folder_path,
                            file_path,
                            equation_amount,
                            file_name,
                            page_number,
                            page_size,
                        )
                    equation_amount += 1
                    block_data = equation_block
                elif block["block_label"] in ["image", "chart"] and self.extract_figures:
                    figure_block = self.generate_figure_block(
                            block,
                            image_folder_path,
                            file_path,
                            figure_amount,
                            file_name,
                            page_number,
                            page_size,
                            index,
                            blocks_list,
                        )
                    figure_amount += 1
                    block_data = figure_block
                elif block["block_label"] == "table" and self.extract_tables:
                    table_block = self.generate_table_block(
                            block,
                            image_folder_path,
                            file_path,
                            table_amount,
                            file_name,
                            page_number,
                            page_size,
                            index,
                            blocks_list,
                        )
                    table_amount += 1
                    block_data = table_block
                elif self.extract_references and block["block_label"] == "reference":
                    references: List[str] = block["block_content"].split("\n")
                    reference_list.extend(references)
                if block_data != {}:
                    blocks_info["blocks"].append(block_data)
                index += 1
            page_number += 1
        block_info_json = json.dumps(blocks_info, indent=4)
        with open(results_folder + "/" + f"{file_name}_content.json", "w") as f:
            f.write(block_info_json)
        if self.extract_tables and self.extract_figures and self.extract_text and self.extract_references:
            with open(os.path.join(f"{results_folder}", f"{file_name}_references.txt"), "w") as r:
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
                results = self._converter.predict(input = file_path)
                if self.extract_tables and self.extract_figures and self.extract_text:
                    results_folder = os.path.join(self.output_folder, file_name)
                else:
                    results_folder = self.output_folder
                image_folder_path = os.path.join(results_folder, f"{file_name}_images")
                if not os.path.exists(image_folder_path):
                    os.makedirs(image_folder_path)
            self.generate_blocks_from_dict(results, results_folder, image_folder_path, file_path, file_name)
            number += 1
