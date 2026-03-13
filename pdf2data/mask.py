import os
from typing import Any, Dict, List, Optional

import re
import cv2
import fitz
import PyPDF2
import tensorflow as tf
import torch
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageEnhance
from pydantic import BaseModel, PrivateAttr
from transformers import DetrImageProcessor, TableTransformerForObjectDetection, AutoModelForObjectDetection
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from torchvision import transforms
from paddleocr import LayoutDetection, TableCellsDetection
import numpy

from pdf2data.support import (block_organizer, box_corretor, iou,
                              order_horizontal, order_vertical, sobreposition, MaxResize, outputs_to_objects)


class LayoutParser(BaseModel):
    model: str
    model_threshold: float = 0.7
    table_model: Optional[str] = None
    table_model_threshold: float = 0.7
    model_path: Optional[str] = None
    table_model_path: Optional[str] = None
    device_type: str = "cpu"
    _model: Any = PrivateAttr(default=None)
    _table_model: Any = PrivateAttr(default=None)
    _existing_models: List[str] = PrivateAttr(
        default=set(
            [
                "microsoft/table-transformer-detection",
                "DocLayout-YOLO-DocStructBench",
                "PP-DocLayout-L"
            ]
        )
    )
    _labels: Any = PrivateAttr(default=None)
    _table_labels: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.model not in self._existing_models:
            raise AttributeError(
                f"The specified model is not available, the available, models are {self._existing_models}"
            )
        if (
            self.table_model not in self._existing_models
            and self.table_model is not None
        ):
            raise AttributeError(
                f"The specified table model is not available, the available, models are {self._existing_models}"
            )
        try:
            labels: Dict[int, str] = LAYOUT_PARSER_LABELS_REGISTRY[self.model]
            self._labels = labels
        except KeyError:
            raise AttributeError("Did not found the labels for the Layout Model")
        if self.model == "microsoft/table-transformer-detection":
            self._model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
        elif self.model == "DocLayout-YOLO-DocStructBench":
            filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
            self._model = YOLOv10(filepath)
        elif self.model == "PP-DocLayout-L":
            self._model = LayoutDetection(model_name="PP-DocLayout-L")
        if self.table_model is not None:
            try:
                labels: Dict[int, str] = LAYOUT_PARSER_LABELS_REGISTRY[self.table_model]
                self._table_labels = labels
            except KeyError:
                raise AttributeError("Did not found the labels for the Table Model")
            if self.table_model == "microsoft/table-transformer-detection":
                self._table_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
            elif self.table_model == "DocLayout-YOLO-DocStructBench":
                filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
                self._table_model = YOLOv10(filepath)
            elif self.table_model == "PP-DocLayout-L":
                self._table_model = LayoutDetection(model_name="PP-DocLayout-L")

    @staticmethod
    def generate_layout_doc_yolo(
        model: Any,
        page: Any,
        width: float,
        pdf_width: float,
        height: float,
        pdf_height: float,
        threshold: str,
        labels: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a PDF layout from a page image using DocLayout-YOLO-DocStructBench model

        Parameters
        ----------
        model : Any
            model loaded to detect the layout
        page : Any
            page to be analyzed
        width : float
            width of the image
        pdf_width : float
            width of the pdf
        height : float
            height of the image
        pdf_height : float
            height of the pdf

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the scores, boxes coordinates and types of each block found.
        """
        layout: List[Any] = model.predict(
            page,   # Image to predict
            imgsz=1024,        # Prediction image size
            conf=threshold,                  # Confidence threshold    # Device to use (e.g., 'cuda:0' or 'cpu')
            verbose=False
            )[0].boxes.data
        exist_figure: bool = False
        boxes: List[List[float]] = []
        scores: List[float] = []
        types: List[str] = []
        table_boxes: List[List[float]] = []
        table_scores: List[float] = []
        table_types: List[str] = []
        for entry in layout:
            # Retrieve the bounding box
            x1: float = entry[0].item() / width * pdf_width
            x2: float = entry[2].item() / width * pdf_width
            y1: float = entry[1].item() / height * pdf_height
            y2: float = entry[3].item() / height * pdf_height
            entry_type = labels[int(entry[5].item())]
            print(entry_type)
            if entry_type in TEXT_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry[4].item())
                types.append("Text")
            elif entry_type in TITLE_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry[4].item())
                types.append("Title")
            elif entry_type in TABLE_WORDS_REGISTRY:
                table_boxes.append([x1, y1, x2, y2])
                table_scores.append(entry[4].item())
                table_types.append("Table")
            elif entry_type in FIGURE_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry[4].item())
                types.append("Figure")
                exist_figure = True
            elif entry_type in FIGURE_CAPTIONS_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry[4].item())
                types.append("Figure Caption")
            elif entry_type in TABLE_CAPTIONS_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry[4].item())
                types.append("Table Caption")
            elif entry_type in TABLE_FOOTNOTE_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry[4].item())
                types.append("Table Footnote")
            elif entry_type in EQUATION_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry[4].item())
                types.append("Equation")
        return {
            "boxes": boxes,
            "scores": scores,
            "types": types,
            "table_boxes": table_boxes,
            "table_scores": table_scores,
            "table_type": table_types,
            "exist_figure": exist_figure,
        }
    
    @staticmethod
    def generate_layout_pp_doc_block(
        model: Any,
        page: Any,
        width: float,
        pdf_width: float,
        height: float,
        pdf_height: float,
        threshold: str,
    ) -> Dict[str, Any]:
        """Generate a PDF layout from a page image using "PP-DocLayout-L" model

        Parameters
        ----------
        model : Any
            model loaded to detect the layout
        page : Any
            page to be analyzed
        width : float
            width of the image
        pdf_width : float
            width of the pdf
        height : float
            height of the image
        pdf_height : float
            height of the pdf

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the scores, boxes coordinates and types of each block found.
        """
        layout: List[Any] = []
        predictions = model.predict(numpy.array(page), batch_size=1)
        for prediction in predictions:
            layout.extend(prediction["boxes"])
        exist_figure: bool = False
        boxes: List[List[float]] = []
        scores: List[float] = []
        types: List[str] = []
        table_boxes: List[List[float]] = []
        table_scores: List[float] = []
        table_types: List[str] = []
        for entry in layout:
            # Retrieve the bounding box
            x1: float = entry["coordinate"][0] / width * pdf_width
            x2: float = entry["coordinate"][2] / width * pdf_width
            y1: float = entry["coordinate"][1] / height * pdf_height
            y2: float = entry["coordinate"][3] / height * pdf_height
            entry_type = entry["label"]
            if entry["score"] < threshold:
                pass
            if entry_type in TEXT_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry["score"])
                types.append("Text")
            elif entry_type in TITLE_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry["score"])
                types.append("Title")
            elif entry_type in TABLE_WORDS_REGISTRY:
                table_boxes.append([x1, y1, x2, y2])
                table_scores.append(entry["score"])
                table_types.append("Table")
            elif entry_type in FIGURE_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry["score"])
                types.append("Figure")
                exist_figure = True
            elif entry_type in FIGURE_CAPTIONS_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry["score"])
                types.append("Figure Caption")
            elif entry_type in TABLE_CAPTIONS_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry["score"])
                types.append("Table Caption")
            elif entry_type in TABLE_FOOTNOTE_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry["score"])
                types.append("Table Footnote")
            elif entry_type in EQUATION_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry["score"])
                types.append("Equation")
            elif entry_type in REFERENCES_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry["score"])
                types.append("Reference")
        return {
            "boxes": boxes,
            "scores": scores,
            "types": types,
            "table_boxes": table_boxes,
            "table_scores": table_scores,
            "table_type": table_types,
            "exist_figure": exist_figure,
        }
    @staticmethod
    def generate_layout_tatr(
        model: Any,
        page: Any,
        width: float,
        pdf_width: float,
        height: float,
        pdf_height: float,
        threshold: str,
        label: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a PDF layout from a page image using microsoft TATR model

        Parameters
        ----------
        model : Any
            model loaded to detect the layout
        page : Any
            page to be analyzed
        width : float
            width of the image
        pdf_width : float
            width of the pdf
        height : float
            height of the image
        pdf_height : float
            height of the pdf
        threshold : str
            threshold value to not consider a positive detection

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the scores, boxes coordinates and types of each block found.
        """
        table_boxes: List[List[float]] = []
        table_scores: List[float] = []
        table_types: List[str] = []
        model.config.id2label
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        # Send the pixel values and pixel mask through the model
        image = page.convert("RGB")
        width, height = image.size
        detection_transform = transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        pixel_values = detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            outputs = model(pixel_values)
        objects = outputs_to_objects(outputs, image.size, label)
        for object in objects:
                x1: float = object["bbox"][0] / width * pdf_width
                x2: float = object["bbox"][2] / width * pdf_width
                y1: float = object["bbox"][1] / height * pdf_height
                y2: float = object["bbox"][3] / height * pdf_height
                entry_type = object["label"]
                if entry_type in TABLE_WORDS_REGISTRY and object["score"] > threshold:
                    table_boxes.append([x1, y1, x2, y2])
                    table_scores.append(object["score"])
                    table_types.append("Table")
        return {
            "boxes": [],
            "scores": [],
            "types": [],
            "table_boxes": table_boxes,
            "table_scores": table_scores,
            "table_type": table_types,
            "exist_figure": False,
        }

    def get_layout(self, pdf_path: str) -> Dict[str, Any]:
        """Get the dictionary with the layout of a pdf file

        Parameters
        ----------
        pdf_path : str
            path to the file

        Returns
        -------
        Dict[str, Any]
            A dictionary containing information about the bloccks coordinates, types, scores e page type

        Raises
        ------
        AttributeError
            if the the given is not in pdf
        """
        extension = os.path.splitext(pdf_path)[1]
        if extension != ".pdf":
            raise AttributeError("The file provided is not in pdf")
        exist_page: List[str] = []
        doc_types: List[List[str]] = []
        doc_boxes: List[List[List[float]]] = []
        doc_scores: List[List[float]] = []
        extracted_blocks: Dict[str, Any] = {}
        pdf = PyPDF2.PdfReader(pdf_path)
        page_list = convert_from_path(pdf_path, use_cropbox=True)
        i = 0
        for page in page_list:
            pdf_page = pdf.pages[i]
            pdf_size = pdf_page.cropbox
            pdf_width = float(pdf_size[2] - pdf_size[0])
            pdf_height = float(pdf_size[3] - pdf_size[1])
            width, height = page.size
            width = float(width)
            height = float(height)
            if self.model == "microsoft/table-transformer-detection":
                first_layout: Dict[str, Any] = LayoutParser.generate_layout_tatr(
                    self._model,
                    page,
                    width,
                    pdf_width,
                    height,
                    pdf_height,
                    self.model_threshold,
                    self._labels
                )
            elif self.model == "DocLayout-YOLO-DocStructBench":
                first_layout: Dict[str, Any] = LayoutParser.generate_layout_doc_yolo(
                    self._model,
                    page,
                    width,
                    pdf_width,
                    height,
                    pdf_height,
                    self.model_threshold,
                    self._labels
                )
            elif self.model == "PP-DocLayout-L":
                first_layout: Dict[str, Any] = LayoutParser.generate_layout_pp_doc_block(
                    self._model,
                    page,
                    width,
                    pdf_width,
                    height,
                    pdf_height,
                    self.model_threshold,
                )
            boxes: List[List[float]] = first_layout["boxes"]
            scores: List[float] = first_layout["scores"]
            types: List[str] = first_layout["types"]
            table_boxes1: List[List[float]] = first_layout["table_boxes"]
            table_scores1: List[float] = first_layout["table_scores"]
            table_types1: List[str] = first_layout["table_type"]
            exist_figures: bool = first_layout["exist_figure"]
            if self.table_model is not None:
                if self.table_model == "microsoft/table-transformer-detection":
                    sec_layout: Dict[str, Any] = LayoutParser.generate_layout_tatr(
                        self._table_model,
                        page,
                        width,
                        pdf_width,
                        height,
                        pdf_height,
                        self.table_model_threshold,
                        self._table_labels
                    )
                elif self.table_model == "DocLayout-YOLO-DocStructBench":
                    sec_layout: Dict[str, Any] = LayoutParser.generate_layout_doc_yolo(
                        self._table_model,
                        page,
                        width,
                        pdf_width,
                        height,
                        pdf_height,
                        self.table_model_threshold,
                        self._table_labels
                    )
                elif self.table_model == "PP-DocLayout-L":
                    sec_layout: Dict[str, Any] = LayoutParser.generate_layout_pp_doc_block(
                        self._table_model,
                        page,
                        width,
                        pdf_width,
                        height,
                        pdf_height,
                        self.model_threshold,
                    )
                table_boxes2: List[List[float]] = sec_layout["table_boxes"]
                table_scores2: List[float] = sec_layout["table_scores"]
                table_types2: List[str] = sec_layout["table_type"]
                boxes = boxes + table_boxes2
                scores = scores + table_scores2
                types = types + table_types2
            else:
                boxes = boxes + table_boxes1
                scores = scores + table_scores1
                types = types + table_types1
            horder_args = order_horizontal(boxes, "argument_list")
            horder_boxes = []
            horder_scores = []
            horder_types = []
            # Order the entries from top to bottom
            for index in horder_args:
                horder_boxes.append(boxes[index])
                horder_scores.append(scores[index])
                horder_types.append(types[index])
            vorder_args = block_organizer(horder_boxes, pdf_size)
            order_boxes = []
            order_scores = []
            order_types = []
            for index in vorder_args:
                order_boxes.append(horder_boxes[index])
                order_scores.append(horder_scores[index])
                order_types.append(horder_types[index])
            # Verify the existance of Tables and/or Images
            if len(table_boxes1) > 0 and exist_figures is True:
                exist_page.append("Table and Image")
            elif len(table_boxes1) > 0:
                exist_page.append("Table")
            elif exist_figures is True:
                exist_page.append("Image")
            else:
                exist_page.append("Only Text")
            doc_boxes.append(order_boxes)
            doc_scores.append(order_scores)
            doc_types.append(order_types)
            i = i + 1
        extracted_blocks["boxes"] = doc_boxes
        extracted_blocks["scores"] = doc_scores
        extracted_blocks["types"] = doc_types
        extracted_blocks["page_type"] = exist_page
        return extracted_blocks

class TableStructureParser(BaseModel):
    model: str
    model_threshold: float = 0.3
    zoom: float = 1.5
    iou_lines: float = 1.0
    iou_struct: float = 0.02
    iou_vert_words: float = 0.15
    brightness: float = 1.0
    contrast: float = 1.0
    _model: Any = PrivateAttr(default=None)
    _labels: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)
    _transform: Any = PrivateAttr(default=None)
    _existing_models: set = PrivateAttr(
        default=set(["microsoft/table-transformer-structure-recognition", "microsoft/table-structure-recognition-v1.1-all"])
    )

    def model_post_init(self, __context: Any) -> None:
        if self.model not in self._existing_models:
            raise AttributeError(
                f"The specified model is not available, the available, models are {self._existing_models}"
            )
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = TableTransformerForObjectDetection.from_pretrained(self.model)
            self._labels = LAYOUT_PARSER_LABELS_REGISTRY["microsoft/table-transformer-structure-recognition"]
            self._model.to(self._device)
            self._transform = transforms.Compose([
                MaxResize(1000),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def table_image_structure_tatr(
        self,
        file_path: str,
        table_coords_corrected: List[float],
        table_coords: List[float]
    ) -> Any:
        cropped_table = Image.open(file_path).convert("RGB")
        width, height = cropped_table.size
        pixel_values = self._transform(cropped_table).unsqueeze(0)
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            outputs = self._model(pixel_values)
        results: List[Dict[str, Any]] = outputs_to_objects(outputs, cropped_table.size, self._labels)
        # Transform a tensor in a list
        row_boxes: List[List[float]] = []
        column_boxes: List[List[float]] = []
        row_scores: List[float] = []
        column_scores: List[float] = []
        merge_cells_boxes: List[List[float]] = []
        for result in results:
            x1: float = float(table_coords_corrected[0]) + float(result["bbox"][0]) / float(width) * float(table_coords_corrected[2] - table_coords_corrected[0])
            y1: float = float(table_coords_corrected[1]) + float(result["bbox"][1]) / float(height) * float(table_coords_corrected[3] - table_coords_corrected[1])
            x2: float = float(table_coords_corrected[0]) + float(result["bbox"][2]) / float(width) * float(table_coords_corrected[2] - table_coords_corrected[0])
            y2: float = float(table_coords_corrected[1]) + float(result["bbox"][3]) / float(height) * float(table_coords_corrected[3] - table_coords_corrected[1])
            new_box = [x1, y1, x2, y2]
            if result["score"] < self.model_threshold:
                pass
            elif iou(new_box, table_coords) < self.iou_struct:
                pass
            elif result["label"] in set(["table row"]):
                row_boxes.append(new_box)
                row_scores.append(result["score"])
            elif result["label"] in set(["table column"]):
                column_boxes.append(new_box)
                column_scores.append(result["score"])
            elif result["label"] == "table spanning cell":
                merge_cells_boxes.append(new_box)
        return {"row_boxes": row_boxes, "row_scores": row_scores, "column_boxes": column_boxes, "column_scores": column_scores, "merged_cells_boxes": merge_cells_boxes}

    def get_table_structure(
        self,
        page: Any,
        table_coords: List[List[float]],
        corrected_table_coords: List[List[float]],
        word_boxes: Dict[str, List[List[float]]] = {},
    ) -> Dict[str, List[List[float]]]:
        """Get the table structure from a table in a pdf

        Parameters
        ----------
        pdf : Any
            pdf document
        page_index : int
            index of the page containing the table
        table_coords : List[List[float]]
            table coordinates
        word_boxes : Dict[str, List[List[float]]], optional
            dicionary with the rows and collumns obtained using the table words, by default {}

        Returns
        -------
        Dict[str, List[List[float]]]
            A dicitonary containing the rows a collumns ordered
        """
        # Correct the tablle coordinates acording the the page size
        table_rect: fitz.Rect = fitz.Rect(corrected_table_coords[0], corrected_table_coords[1], corrected_table_coords[2], corrected_table_coords[3])
        # apply zoom to increase resolution
        mat: fitz.Matrix = fitz.Matrix(self.zoom, self.zoom)
        image: Any = page.get_pixmap(matrix=mat, clip=table_rect)
        image.save("image.png")
        structure_dict: Dict[str, Any] = self.table_image_structure_tatr("image.png", corrected_table_coords, table_coords)
        rows: List[List[float]] = structure_dict["row_boxes"]
        collumns: List[List[float]] = structure_dict["column_boxes"]
        row_scores: List[float] = structure_dict["row_scores"]
        collumns_scores: List[float] = structure_dict["column_scores"]
        if len(rows) > 0:
            supressed_rows: List[KeyboardInterrupt] = tf.image.non_max_suppression(
                rows,
                row_scores,
                max_output_size=1000,
                iou_threshold=self.iou_lines,
                score_threshold=tf.constant(float("-inf"), dtype=tf.float32),
                name=None,
            )
            real_rows: List[List[float]] = []
            for index in supressed_rows:
                real_rows.append(rows[index])
        else:
            real_rows = rows
        if len(collumns) > 0:
            supressed_collumns: List[int] = tf.image.non_max_suppression(
                collumns,
                collumns_scores,
                max_output_size=1000,
                iou_threshold=self.iou_lines,
                score_threshold=tf.constant(float("-inf"), dtype=tf.float32),
                name=None,
            )
            real_collumns: List[List[float]] = []
            for index in supressed_collumns:
                real_collumns.append(collumns[index])
        else:
            real_collumns = collumns
        if "rows" in word_boxes.keys():
            word_rows: List[List[float]] = word_boxes["rows"]
        else:
            word_rows = []
        for row1 in word_rows:
            exists_row: bool = False
            for row2 in real_rows:
                if (
                    iou(row1, row2) > self.iou_vert_words
                    or sobreposition(row1, table_coords) < 0.5
                ):
                    exists_row = True
                    break
            if exists_row is False:
                real_rows.append(row1)
        # Order from top to bottom
        ordered_rows: List[List[float]] = order_horizontal(real_rows)
        # Order from left to right
        ordered_collumns: List[List[float]] = order_vertical(real_collumns)
        os.remove('image.png')
        return {"rows": ordered_rows, "collumns": ordered_collumns}


LAYOUT_PARSER_LABELS_REGISTRY: Dict[str, Dict[int, str]] = {
    "DocLayout-YOLO-DocStructBench": {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'},
    "microsoft/table-transformer-detection": {0: 'table', 1: 'table rotated', 2: 'no object'},
    "PP-DocLayout-L": {},
    "microsoft/table-transformer-structure-recognition": {0: 'table', 1: 'table column', 2: 'table row', 3: 'table column header', 4: 'table projected row header', 5: 'table spanning cell', 6: 'no object'}
}

LAYOUT_PARSER_CONFIG_REGISTRY: Dict[str, str] = {
    "PrimaLayout_mask_rcnn_R_50_FPN_3x": "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
    "PubLayNet_faster_rcnn_R_50_FPN_3x": "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    "PubLayNet_mask_rcnn_X_101_32x8d_FPN_3x": "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
    "TableBank_faster_rcnn_R_50_FPN_3x": "lp://TableBank/faster_rcnn_R_50_FPN_3x/config",
    "TableBank_faster_rcnn_R_101_FPN_3x": "lp://TableBank/faster_rcnn_R_101_FPN_3x/config",
}
TABLE_WORDS_REGISTRY: set = set(["TableRegion", "Table", "table", "table rotated"])
FIGURE_WORDS_REGISTRY: set = set(["ImageRegion", "Figure", "figure", "chart"])
TEXT_WORDS_REGISTRY: set = set(["TextRegion", "Text", "plain text", "text", "abstract"])
TITLE_WORDS_REGISTRY: set = set(["Title", "title", "doc_title", "paragraph_title"])
FIGURE_CAPTIONS_WORDS_REGISTRY: set = set(["figure_caption", "figure_caption", "figure_title", "chart_title"])
TABLE_CAPTIONS_WORDS_REGISTRY: set = set(["table_caption", "table_caption", "table_title"])
TABLE_FOOTNOTE_WORDS_REGISTRY: set = set(["table_footnote", "footnotes"])
EQUATION_WORDS_REGISTRY: set = set(["formula", "isolate_formula"])
REFERENCES_WORDS_REGISTRY: set = set(["reference"])
