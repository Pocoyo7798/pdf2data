from typing import Any, Dict, List

from shapely import box
from pdf2data.pipeline import Pipeline, Table, Figure, Text, Equation
from pydantic import PrivateAttr
import json
import os
from paddleocr import PPStructureV3, PaddleOCRVL
import fitz
import subprocess
import importlib_resources
import time
import socket
import multiprocessing
import requests

def converter_worker(task_q, result_q, extractor_name):

    if extractor_name == "PaddlePPStructure":
        converter = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_chart_recognition=False,
        )
    elif extractor_name == "PaddleVL":
        while True:
            try:
                requests.get("http://127.0.0.1:8118/health", timeout=1)
                break
            except:
                time.sleep(1)
        converter = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url="http://127.0.0.1:8118/v1"
        )

    while True:
        file_path = task_q.get()

        if file_path is None:
            break

        try:
            result = converter.predict(file_path)

            # Convert to picklable structure
            result_q.put([r.json for r in result])

        except Exception as e:
            result_q.put(e)

class PaddlePPStructure(Pipeline):
    extractor_name: str = "PaddlePPStructure"
    _converter: PPStructureV3 = PrivateAttr(default=None)
    _max_attempts: int = PrivateAttr(default=3)
    _task_q: multiprocessing.Queue = PrivateAttr()
    _result_q: multiprocessing.Queue = PrivateAttr()
    _worker: multiprocessing.Process = PrivateAttr()
    _restart_lock: Any = PrivateAttr(default_factory=multiprocessing.Lock)

    def model_post_init(self, context):

        if self.extractor_name == "PaddleVL":
            self.close_server()

            config_file_path = str(
                importlib_resources.files("pdf2data") / "resources" / "vllm_config.yaml"
            )

            subprocess.Popen(
                [
                    "paddleocr",
                    "genai_server",
                    "--model_name",
                    "PaddleOCR-VL-1.5-0.9B",
                    "--backend",
                    "vllm",
                    "--port",
                    "8118",
                    "--backend_config",
                    config_file_path,
                ]
            )

            self.wait_for_port(8118)

        self._task_q = multiprocessing.Queue()
        self._result_q = multiprocessing.Queue()

        self._worker = multiprocessing.Process(
            target=converter_worker,
            args=(self._task_q, self._result_q, self.extractor_name),
        )

        self._worker.start()
    
    def start_worker(self):

        self._task_q = multiprocessing.Queue()
        self._result_q = multiprocessing.Queue()

        self._worker = multiprocessing.Process(
            target=converter_worker,
            args=(self._task_q, self._result_q, self.extractor_name),
        )

        self._worker.start()


    def restart_worker(self):

        print("Restarting worker...")

        try:
            self._task_q.put(None)
        except:
            pass

        if self._worker.is_alive():
            self._worker.terminate()

        self._worker.join()

        try:
            self._task_q.close()
            self._result_q.close()
        except:
            pass

        self.start_worker()


    def restart_server(self):

        print("Restarting PaddleOCR-VL server...")

        self.close_server()

        config_file_path = str(
            importlib_resources.files("pdf2data") / "resources" / "vllm_config.yaml"
        )

        subprocess.Popen([
            "paddleocr",
            "genai_server",
            "--model_name",
            "PaddleOCR-VL-1.5-0.9B",
            "--backend",
            "vllm",
            "--port",
            "8118",
            "--backend_config",
            config_file_path,
        ])

        self.wait_for_port(8118)

        # important: allow model loading
        time.sleep(6)

        print("Server restarted successfully")

    def close_server(self, port=8118):
        try:
            result = subprocess.check_output(["lsof", "-ti", f":{port}"]).decode().strip()
            for pid in result.split("\n"):
                subprocess.run(["kill", "-9", pid])
        except subprocess.CalledProcessError:
            pass  # nothing running on that port

    def predict_with_timeout(self, file_path, timeout=240):

        for attempt in range(self._max_attempts):

            try:

                # Wait if server restarting
                with self._restart_lock:
                    pass

                # Ensure server ready before request
                if self.extractor_name == "PaddleVL":
                    self.wait_for_port(8118)

                self._task_q.put(file_path)

                result = self._result_q.get(timeout=timeout)

                if isinstance(result, Exception):
                    raise result

                return result

            except Exception as e:

                print(f"Prediction failed (attempt {attempt+1}): {e}")

                if self.extractor_name == "PaddleVL":

                    with self._restart_lock:
                        self.restart_server()

                self.restart_worker()

        raise RuntimeError("Prediction failed after maximum retries")
    
    def wait_for_port(self, port, host="localhost", timeout=180):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                time.sleep(0.5)
        raise TimeoutError("Server did not start in time")


    def generate_text_block(self, 
                             block: Dict[str, Any], file_path: str, page_number: int, page_size: tuple) -> Dict[str, Any]: 
        text_object = Text()
        if block["block_label"] in ["paragraph", "abstract", "text"]:
            text_object.type = "paragraph"
        else:
            text_object.type = "section_header"
        text_object.content = block["block_content"]
        text_object.page = page_number
        text_object.box = self.correct_box_size(block["block_bbox"], page_size, file_path, page_number)
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
        equation_object.box = self.correct_box_size(block["block_bbox"], page_size, file_path, page_number)
        equation_object.number = number
        equation_object.filepath = self.snap_figure(image_folder_path,
                                                   equation_object.page,
                                                   file_path,
                                                   equation_object.box,
                                                   number,
                                                   doc_name,
                                                   "Equation",)
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
        figure_object.box = self.correct_box_size(block["block_bbox"], page_size, file_path, page_number)
        figure_object.number = number
        figure_object.filepath = self.snap_figure(image_folder_path,
                                                   figure_object.page,
                                                   file_path,
                                                   figure_object.box,
                                                   number,
                                                   doc_name,
                                                   "Figure",)
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
        table_object.box = self.correct_box_size(block["block_bbox"], page_size, file_path, page_number)
        table_object.number = number
        table_object.filepath = self.snap_figure(image_folder_path,
                                                   table_object.page,
                                                   file_path,
                                                   table_object.box,
                                                   number,
                                                   doc_name,
                                                   "Table")
        if index < 1 or index >= len(blocks_list) - 1:
            table_object.caption = ""
        elif blocks_list[index + 1]["block_label"] == "table_title":
            table_object.caption = blocks_list[index + 1]["block_content"]
        elif blocks_list[index - 1]["block_label"] == "table_title":
            table_object.caption = blocks_list[index - 1]["block_content"]
        else:            
            table_object.caption = ""
        table_object.block = self.html_table_to_list(block["block_content"])
        old_table_block = table_object.block.copy()
        table_object.block = self.correct_table_structure(table_object.block)
        if len(old_table_block) != len(table_object.block):
            table_object.caption = old_table_block[0][0]
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
            document_dict: Dict[str, Any] = res
            page_width = document_dict["res"]["width"]
            page_height = document_dict["res"]["height"]
            page_size = (page_width, page_height)
            blocks_list = document_dict["res"]['parsing_res_list']
            index = 0
            for block in blocks_list:
                block_data: Dict[str, Any] = {}
                if block["block_label"] in ["paragraph", "paragraph_title", "doc_title", "abstract", "text"] and self.extract_text:
                    text_block = self.generate_text_block(
                            block, file_path, page_number, page_size,
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
        self.wait_for_port(8118)
        for file in files_list:
            print(f"{number}//{len(files_list)} processed")
            if file.endswith('.pdf'):
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(self.input_folder, file)
                results = self.predict_with_timeout(file_path, timeout=180)
                if self.extract_tables and self.extract_figures and self.extract_text:
                    results_folder = os.path.join(self.output_folder, file_name)
                else:
                    results_folder = self.output_folder
                image_folder_path = os.path.join(results_folder, f"{file_name}_images")
                if not os.path.exists(image_folder_path):
                    os.makedirs(image_folder_path)
                self.generate_blocks_from_dict(results, results_folder, image_folder_path, file_path, file_name)
            number += 1
        self.close_server()