from typing import Any, Dict, List, Optional

from torch import device
from pdf2data.pipeline import Pipeline, Table, Figure, Text, Equation
from pdf2data.mask import LayoutParser
import os
import shutil
from pydantic import PrivateAttr

class PDF2Data(Pipeline):
    layout_model: str
    layout_model_threshold: float
    table_model: Optional[str]
    table_model_threshold: float
    device: str
    _mask: Optional[LayoutParser] = PrivateAttr(default=None)

    def model_post_init(self, context):
        self._mask = LayoutParser(
            model=self.layout_model,
            model_threshold=self.layout_model_threshold,
            table_model=self.table_model,
            table_model_threshold=self.table_model_threshold,
            device_type=self.device
        )
    
    def generate_blocks_from_dict(self, doc_layout: Dict[str, Any], results_folder: str, image_folder_path: str, file_path: str, file_name: str) -> None:
        figure_amount: int = 1
        table_amount: int = 1
        equation_amount: int = 1
        reference_list: List[str] = []
        print(doc_layout)

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
                self.generate_blocks_from_dict(doc_layout, results_folder, image_folder_path, file_path, file_name)
            number += 1