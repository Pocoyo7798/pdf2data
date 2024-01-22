import os
from typing import Any, Dict, List, Optional

import layoutparser as lp
import PyPDF2
import torch
from pdf2image import convert_from_path
from pydantic import BaseModel, PrivateAttr
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from pdf2data.support import block_organizer, iou, order_horizontal


class LayoutParser(BaseModel):
    model: str
    model_threshold: float = 0.7
    table_model: Optional[str] = None
    table_model_threshold: float = 0.7
    model_path: Optional[str] = None
    table_model_path: Optional[str] = None
    _model: Any = PrivateAttr(default=None)
    _table_model: Any = PrivateAttr(default=None)
    _existing_models: List[str] = PrivateAttr(
        default=set(
            [
                "PrimaLayout_mask_rcnn_R_50_FPN_3x",
                "PubLayNet_faster_rcnn_R_50_FPN_3x",
                "PubLayNet_mask_rcnn_X_101_32x8d_FPN_3x",
                "TableBank_faster_rcnn_R_50_FPN_3x",
                "TableBank_faster_rcnn_R_101_FPN_3x",
                "microsoft/table-transformer-detection",
            ]
        )
    )

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
            layout_parser: bool = True
        except KeyError:
            layout_parser: bool = False
        if layout_parser is False:
            self._model = TableTransformerForObjectDetection.from_pretrained(self.model)
        elif self.model_path is not None:
            self._model = lp.Detectron2LayoutModel(
                f"{self.model_path}/config.yaml",
                f"{self.model_path}/model_final.pth",
                extra_config=[
                    "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                    self.model_threshold,
                ],
                label_map=labels,
            )
        else:
            model_config = LAYOUT_PARSER_CONFIG_REGISTRY[self.model]
            self._model = lp.Detectron2LayoutModel(
                config_path=model_config,  # In model catalog
                label_map=labels,  # In model`label_map`
                extra_config=[
                    "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                    self.model_threshold,
                ],  # Optional
            )
        if self.table_model is not None:
            try:
                labels: Dict[int, str] = LAYOUT_PARSER_LABELS_REGISTRY[self.table_model]
                layout_parser: bool = True
            except KeyError:
                layout_parser: bool = False
            if layout_parser is False:
                self._table_model = TableTransformerForObjectDetection.from_pretrained(
                    self.table_model
                )
            elif self.table_model_path is not None:
                self._model = lp.Detectron2LayoutModel(
                    f"{self.table_model_path}/config.yaml",
                    f"{self.table_model_path}/model_final.pth",
                    extra_config=[
                        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                        self.table_model_threshold,
                    ],
                    label_map=labels,
                )
            else:
                model_config = LAYOUT_PARSER_CONFIG_REGISTRY[self.table_model]
                self._model = lp.Detectron2LayoutModel(
                    config_path=model_config,  # In model catalog
                    label_map=labels,  # In model`label_map`
                    extra_config=[
                        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                        self.model_threshold,
                    ],  # Optional
                )

    @staticmethod
    def generate_layout_lp(
        model: Any,
        page: Any,
        width: float,
        pdf_width: float,
        height: float,
        pdf_height: float,
    ) -> Dict[str, Any]:
        """Generate a PDF layout from a page image using layout_parser models

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
        layout: Any = model.detect(page)
        exist_figure: bool = False
        boxes: List[List[float]] = []
        scores: List[float] = []
        types: List[str] = []
        table_boxes: List[List[float]] = []
        table_scores: List[float] = []
        table_types: List[str] = []
        for entry in layout:
            # Retrieve the bounding box
            x1: float = entry.block.x_1 / width * pdf_width
            x2: float = entry.block.x_2 / width * pdf_width
            y1: float = entry.block.y_1 / height * pdf_height
            y2: float = entry.block.y_2 / height * pdf_height
            if entry.type in TEXT_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry.score)
                types.append("Text")
            elif entry.type in TITLE_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry.score)
                types.append("Title")
            elif entry.type in TABLE_WORDS_REGISTRY:
                table_boxes.append([x1, y1, x2, y2])
                table_scores.append(entry.score)
                table_types.append("Table")
            elif entry.type in FIGURE_WORDS_REGISTRY:
                boxes.append([x1, y1, x2, y2])
                scores.append(entry.score)
                types.append("Figure")
                exist_figure = True
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
    def generate_layout_hf(
        model: Any,
        page: Any,
        width: float,
        pdf_width: float,
        height: float,
        pdf_height: float,
        threshold: str,
    ) -> Dict[str, Any]:
        """Generate a PDF layout from a page image using huggingface models

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
        types: List[str] = []
        feature_extractor = DetrImageProcessor()
        encoding = feature_extractor(page, return_tensors="pt")
        encoding.keys()
        # Send the pixel values and pixel mask through the model
        with torch.no_grad():
            outputs = model(**encoding)
        # Detect all tables in a image
        results = feature_extractor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(height, width)]
        )[0]
        # Verify if at least one table was detected
        boxes = results["boxes"].tolist()
        if len(boxes) != 0:
            for value in boxes:
                value[0] = value[0] / width * float(pdf_width)
                value[1] = value[1] / height * float(pdf_height)
                value[2] = value[2] / width * float(pdf_width)
                value[3] = value[3] / height * float(pdf_height)
                types.append("Table")
        table_scores = results["scores"].tolist()
        return {
            "boxes": [],
            "scores": [],
            "types": [],
            "table_boxes": boxes,
            "table_scores": table_scores,
            "table_type": types,
            "exist_figure": False,
        }

    def get_layout(self, pdf_path: str, iou_max: float) -> Dict[str, Any]:
        """Get the dictionary with the layout of a pdf file

        Parameters
        ----------
        pdf_path : str
            path to the file
        iou_max : float
            maximum iou to correct the table

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
                first_layout: Dict[str, Any] = LayoutParser.generate_layout_hf(
                    self._model,
                    page,
                    width,
                    height,
                    pdf_height,
                    pdf_width,
                    self.model_threshold,
                )
            else:
                first_layout = LayoutParser.generate_layout_lp(
                    self._model, page, width, height, pdf_height, pdf_width
                )
            boxes: List[List[float]] = first_layout["boxes"]
            scores: List[float] = first_layout["scores"]
            types: List[str] = first_layout["types"]
            table_boxes2: List[List[float]] = first_layout["table_boxes"]
            table_scores2: List[float] = first_layout["table_scores"]
            table_types2: List[str] = first_layout["table_type"]
            exist_figures: bool = first_layout["exist_figure"]
            if self.table_model is not None:
                if self.table_model == "microsoft/table-transformer-detection":
                    sec_layout: Dict[str, Any] = LayoutParser.generate_layout_hf(
                        self._table_model,
                        page,
                        width,
                        height,
                        pdf_height,
                        pdf_width,
                        self.model_threshold,
                    )
                else:
                    sec_layout = LayoutParser.generate_layout_lp(
                        self._table_model, page, width, height, pdf_height, pdf_width
                    )
                table_boxes1: List[List[float]] = sec_layout["table_boxes"]
                table_scores1: List[float] = sec_layout["table_scores"]
                table_types1: List[str] = sec_layout["table_type"]
                k = 0
                for table2 in table_boxes2:
                    area2: float = (table2[2] - table2[0]) * (table2[3] - table2[1])
                    j = 0
                    for table1 in table_boxes1:
                        area1: float = (table1[2] - table1[0]) * (table1[3] - table1[1])
                        if area2 < area1 and iou_max > iou(table2, table1) > 0.5:
                            table_boxes1[j] = table2
                            table_scores1[j] = table_scores2[k]
                            table_types1[j] = table_types2[k]
                        j = j + 1
                    k = k + 1
            else:
                table_boxes1 = table_boxes2
                table_scores1 = table_scores2
                table_types1 = table_types2
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


LAYOUT_PARSER_LABELS_REGISTRY: Dict[str, Dict[int, str]] = {
    "PrimaLayout_mask_rcnn_R_50_FPN_3x": {
        1: "TextRegion",
        2: "ImageRegion",
        3: "TableRegion",
        4: "MathsRegion",
        5: "SeparatorRegion",
        6: "OtherRegion",
    },
    "PubLayNet_faster_rcnn_R_50_FPN_3x": {
        0: "Text",
        1: "Title",
        2: "List",
        3: "Table",
        4: "Figure",
    },
    "PubLayNet_mask_rcnn_X_101_32x8d_FPN_3x": {
        0: "Text",
        1: "Title",
        2: "List",
        3: "Table",
        4: "Figure",
    },
    "TableBank_faster_rcnn_R_50_FPN_3x": {0: "Table"},
    "TableBank_faster_rcnn_R_101_FPN_3x": {0: "Table"},
}

LAYOUT_PARSER_CONFIG_REGISTRY: Dict[str, str] = {
    "PrimaLayout_mask_rcnn_R_50_FPN_3x": "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
    "PubLayNet_faster_rcnn_R_50_FPN_3x": "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    "PubLayNet_mask_rcnn_X_101_32x8d_FPN_3x": "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
    "TableBank_faster_rcnn_R_50_FPN_3x": "lp://TableBank/faster_rcnn_R_50_FPN_3x/config",
    "TableBank_faster_rcnn_R_101_FPN_3x": "lp://TableBank/faster_rcnn_R_101_FPN_3x/config",
}
TABLE_WORDS_REGISTRY: set = set(["TableRegion", "Table"])
FIGURE_WORDS_REGISTRY: set = set(["ImageRegion", "Figure"])
TEXT_WORDS_REGISTRY: set = set(["TextRegion", "Text"])
TITLE_WORDS_REGISTRY: set = set(["Title"])
