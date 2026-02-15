from typing import Any, Dict, List
from pdf2data.pipeline import Pipeline, Table, Figure, Text, Equation
from pydantic import PrivateAttr
import json
import os
from paddleocr import PPStructureV3

class PaddleVLM(Pipeline):
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
