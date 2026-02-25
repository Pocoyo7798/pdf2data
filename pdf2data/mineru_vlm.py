from typing import Any, Dict, List, Optional
from pdf2data.pipeline import Pipeline, Table, Figure, Text, Equation
from pydantic import PrivateAttr
import json
import os
import fitz
from PIL import Image
import io
from mineru_vl_utils import MinerUClient
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

class MinerUVLM(Pipeline):

    def model_post_init(self, context):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B",
            dtype="auto", # use `torch_dtype` instead of `dtype` for transformers<4.56.0
            device_map="auto"
        )

        processor = AutoProcessor.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B",
            use_fast=True
        )

        self._converter = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor
        )

            
    def pdf_to_pillow_images(pdf_path, dpi=200):
        doc = fitz.open(pdf_path)
        images = []

        for page_number in range(len(doc)):
            page = doc[page_number]

            # Render page to a pixmap
            pix = page.get_pixmap(dpi=dpi)

            # Convert pixmap to bytes
            img_bytes = pix.tobytes("png")

            # Load into Pillow
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

        return images
    
    def generate_table_block(self, 
                             initial_block: Dict[str, Any],
                             number: int,
                             doc_name: str,
                             image_folder: str, 
                             document_folder: str,
                             page_number: int,
                             pdf_file_path: str,
                             page_size: Dict[str, Any],
                             blocks_list: List[Dict[str, Any]],
                             index: int) -> Dict[str, Any]:
        
        table_object = Table()
    
    def generate_figure_block(self, number: int,
                              initial_block: Dict[str, Any],
                              doc_name: str,
                              image_folder: str,
                              page_number:int,
                              pdf_file_path: str,
                              page_size: tuple,
                              blocks_list: List[Dict[str, Any]],
                              index: int) -> Dict[str, Any]:
        figure_object = Figure()
        figure_object.number = number
        figure_object.page = page_number
        figure_object.box = self.correct_box_size(initial_block['bbox'], page_size, pdf_file_path, page_number)
        figure_object.filepath =self.snap_figure(image_folder,
                                                   figure_object.page,
                                                   pdf_file_path,
                                                   figure_object.box,
                                                   number,
                                                   doc_name,
                                                   "Figure",)
        if index < 1 or index >= len(blocks_list) - 1:
            figure_object.caption = ""
            figure_object.footnotes = ""
        else:
            if blocks_list[index - 1]["type"] == "image_caption":
                figure_object.caption = blocks_list[index - 1]["content"]
            elif blocks_list[index + 1]["type"] == "image_caption":
                figure_object.caption = blocks_list[index + 1]["content"]
            if blocks_list[index + 1]["type"] == "image_footnote":
                figure_object.footnotes = blocks_list[index + 1]["content"]
            elif blocks_list[index + 2]["type"] == "image_footnote":
                figure_object.footnotes = blocks_list[index + 2]["content"]
        return figure_object.model_dump()
    
    def generate_text_block(self, 
                             initial_block: Dict[str, Any], page_number: int,
                             pdf_file_path: str, page_size: tuple) -> Dict[str, Any]: 
        text_object = Text()
        if initial_block["type"] == "title":
            text_object.type = "section_header"
        elif initial_block["type"] == "text":
            text_object.type = "paragraph"
        text_object.content = self._latex_parser(initial_block["content"])
        text_object.page = page_number
        text_object.box = self.correct_box_size(initial_block['bbox'], page_size, pdf_file_path, page_number)
        return text_object.model_dump()
    
    def generate_equation_block(self, number:int,
                            initial_block: Dict[str, Any], 
                            doc_name: str,
                            image_folder: str,
                            document_folder: str,
                            page_number: int,
                            pdf_file_path: str,
                            page_size: tuple) -> Dict[str, Any]:
        equation_object = Equation()
        equation_object.number = number
        equation_object.page = page_number
        equation_object.box = self.correct_box_size(initial_block['bbox'], page_size, pdf_file_path, page_number)
        equation_object.filepath = self.snap_figure(image_folder,
                                                   equation_object.page,
                                                   pdf_file_path,
                                                   equation_object.box,
                                                   number,
                                                   doc_name,
                                                   "Equation",)
        equation_object.content = self._latex_parser(initial_block["content"])
        return equation_object.model_dump()

    def update_references(self, content: Dict[str, Any]) -> None:
        pass

    def generate_blocks_from_folder(self, image_list :List[Any], results_folder, image_folder_path, file_path, file_name) -> None:
        blocks_info: Dict[str, List[Dict[str, Any]]] = {"blocks": []}
        figure_amount: int = 1
        table_amount: int = 1
        equation_amount: int = 1
        reference_list: List[str] = []
        page_number = 1
        page_size = (1, 1)
        for image in image_list:
            block_list = self._converter.two_step_extract(image)
            index = 0
            for block in block_list:
                if block['type'] == 'table' and self.extract_tables:
                    table_block = self.generate_table_block(block, table_amount, file_name, image_folder_path, page_number, file_path, page_size, index, block_list)
                    blocks_info["blocks"].append(table_block)
                    table_amount += 1
                elif block['type'] == 'image' and self.extract_figures:
                    figure_block = self.generate_figure_block(figure_amount, block, file_name, image_folder_path, page_number, file_path, page_size, index, block_list)
                    blocks_info["blocks"].append(figure_block)
                    figure_amount += 1
                elif block['type'] in ['text', "title"] and self.extract_text:
                    text_block = self.generate_text_block(block, page_number, file_path, page_size)
                    blocks_info["blocks"].append(text_block)
                elif block['type'] == 'equation' and self.extract_equations:
                    equation_block = self.generate_equation_block(equation_amount, block, file_name, image_folder_path, page_number, file_path, page_size)
                    blocks_info["blocks"].append(equation_block)
                    equation_amount += 1
                index += 1
            page_number += 1
        
    
    def pdf_transform(self) -> None:
        files_list = os.listdir(self.input_folder)
        total_docs: int = len(files_list)
        doc_number: int = 1
        for file in files_list:
            print(file)
            print(f'{doc_number}//{total_docs} processed')
            doc_number += 1
            if file.endswith('.pdf'):
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(self.input_folder, file)
                if self.extract_tables and self.extract_figures and self.extract_text:
                    results_folder = os.path.join(self.output_folder, file_name)
                else:
                    results_folder = self.output_folder
                image_folder_path = os.path.join(results_folder, f"{file_name}_images")
                if not os.path.exists(image_folder_path):
                    os.makedirs(image_folder_path)
                page_image_list = self.pdf_to_pillow_images(os.path.join(self.input_folder, file_name))
                self.generate_blocks_from_folder(page_image_list, results_folder, image_folder_path, file_path, file_name)
                
