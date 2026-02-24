from typing import Any, Dict, List, Optional
from pdf2data.pipeline import Pipeline, Table, Figure, Text, Equation
from pydantic import PrivateAttr
import subprocess
import json
import os
import shutil

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
                             page_number: int,
                             pdf_file_path: str,
                             page_size: Dict[str, Any]) -> Dict[str, Any]:
        
        table_object = Table()
        content = initial_block["content"] 
        table_object.number = number
        table_object.caption = self.generate_text(content["table_caption"])
        try:
            table_object.block = self.html_table_to_list(content["html"])
        except KeyError:
            return None
        old_table_block = table_object.block.copy()
        table_object.block = self.correct_table_structure(table_object.block)
        if len(old_table_block) != len(table_object.block):
            table_object.caption = old_table_block[0][0]
        table_object.footnotes = self.generate_text(content["table_footnote"])
        table_object.column_headers = self.find_column_headers(table_object.block)
        table_object.row_indexes = self.find_row_indexes(table_object.block)
        table_object.page = page_number
        table_object.box = self.correct_box_size(initial_block["bbox"], page_size, pdf_file_path, page_number)
        if content["image_source"]["path"] == "images/":
            return None
        shutil.move(
            os.path.join(document_folder, f"{content['image_source']['path']}"),
            os.path.join(image_folder, f"Table_{number}.png"),
        )
        table_object.filepath = os.path.join(f"{doc_name}_images", f"Table_{number}.png")
        return table_object.model_dump()
    
    def generate_figure_block(self, number: int,
                              initial_block: Dict[str, Any],
                              doc_name: str,
                              image_folder: str,
                              document_folder: str,
                              page_number:int,
                              pdf_file_path: str,
                              page_size: tuple) -> Dict[str, Any]:
        figure_object = Figure()
        content = initial_block["content"]
        figure_object.number = number
        figure_object.caption = self.generate_text(content["image_caption"])
        figure_object.number = number
        figure_object.page = page_number
        figure_object.box = self.correct_box_size(initial_block["bbox"], page_size, pdf_file_path, page_number)
        shutil.move(
            os.path.join(document_folder, content["image_source"]["path"]),
            os.path.join(image_folder, f"Figure_{number}.png"),
        )
        figure_object.filepath = os.path.join(f"{doc_name}_images", f"Figure_{number}.png")
        return figure_object.model_dump()
    
    def generate_text_block(self, 
                             initial_block: Dict[str, Any], page_number: int,
                             pdf_file_path: str, page_size: tuple) -> Dict[str, Any]: 
        text_object = Text()
        if initial_block["type"] == "paragraph":
            text_object.type = "paragraph"
        elif initial_block["type"] == "title":
            text_object.type = "section_header"
        text_content = initial_block["content"][f"{initial_block['type']}_content"]
        text_object.content = self.generate_text(text_content)
        text_object.page = page_number
        text_object.box = self.correct_box_size(initial_block["bbox"], page_size, pdf_file_path, page_number)
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
        equation_object.content = self._latex_parser.latex_to_text(initial_block["content"]["math_content"]).replace(" ", "")
        equation_object.page = page_number
        equation_object.box = self.correct_box_size(initial_block["bbox"], page_size, pdf_file_path, page_number)
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
            mildle_results_file_path = os.path.join(miner_results_folder, f"{folder}_middle.json")
            pdf_file_path = os.path.join(miner_results_folder, f"{folder}_origin.pdf")
            with open(doc_file_path, "r") as doc_file:
                content_list: List[Dict[str, Any]] = json.load(doc_file)
            with open(mildle_results_file_path, "r") as middle_file:
                middle_results_list: Dict[str, Any] = json.load(middle_file)["pdf_info"]
            if self.extract_tables and self.extract_figures and self.extract_text:
                image_folder_path = os.path.join(self.output_folder, folder, f"{folder}_images")
            else:
                image_folder_path = os.path.join(self.output_folder, f"{folder}_images")
            if not os.path.exists(image_folder_path):
                os.makedirs(image_folder_path)
            page_number = 0
            self._reference_list = []
            i = 0
            for page in content_list:
                page_size = tuple(middle_results_list[i]["page_size"])
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
                            page_number,
                            pdf_file_path,
                            page_size
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
                            page_number,
                            pdf_file_path,
                            page_size
                        )
                        blocks_info["blocks"].append(figure_block)
                    elif content["type"] in ["paragraph", "title"] and self.extract_text:
                        text_block = self.generate_text_block(
                            content,
                            page_number,
                            pdf_file_path,
                            page_size
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
                            page_number,
                            pdf_file_path,
                            page_size
                        )
                        blocks_info["blocks"].append(equation_block)
                    elif content["type"] == "list" and self.extract_references:
                        self.update_references(content)
                i += 1
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
                if total_size + file_size <= 50:  # 50 MB in bytes
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
