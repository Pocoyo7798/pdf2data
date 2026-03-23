from pydantic import BaseModel, PrivateAttr
import os
import json
from typing import List, Any
import re
import ftfy
from pyparsing import line
from itertools import groupby

from pdf2data import text

class Upgrader(BaseModel):
    correct_unicodes: bool = True
    merge_figures: bool = True
    all_documents: bool = True
    distance_threshold: float = 50.0
    _unicode_regex: re.Pattern = PrivateAttr(default=None)
    
    def model_post_init(self, context):
        self._unicode_regex = re.compile(
                                        "|".join(re.escape(k) for k in sorted(REPLACEMENTS, key=len, reverse=True))
                                    )
    
    def correct_unicodes_in_string(self, text: str) -> str:
        text = ftfy.fix_text(text)      # 1. Fix encoding/mojibake issues
        return self._unicode_regex.sub(lambda m: REPLACEMENTS[m.group(0)], text)  # 2. Handle remaining edge cases
    
    def correct_unicodes_in_blocks(self, blocks: List[Any]) -> List[Any]:
        for block in blocks:
            if block["type"] in ["paragraph", "section_header"]:
                block["content"] = self.correct_unicodes_in_string(block["content"])
            elif block["type"] == "Table":
                block["caption"] = self.correct_unicodes_in_string(block["caption"])
                for line in block["block"]:
                    for i, cell in enumerate(line):
                        line[i] = self.correct_unicodes_in_string(cell)
            elif block["type"] == "Figure":
                block["caption"] = self.correct_unicodes_in_string(block["caption"])
        return blocks
    
    def _box_distance(self, box1: list, box2: list) -> float:
        """Minimum distance between two [x1, y1, x2, y2] boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
        dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
        return (dx**2 + dy**2) ** 0.5

    def _merge_figure_group(self, figures: List[Any]) -> dict:
        """Merge a group of figure blocks into one."""
        captions = [f["caption"] for f in figures if f.get("caption")]
        
        # Merge bounding box
        all_boxes = [f["box"] for f in figures if f.get("box")]
        merged_box = [
            min(b[0] for b in all_boxes),  # x1
            min(b[1] for b in all_boxes),  # y1
            max(b[2] for b in all_boxes),  # x2
            max(b[3] for b in all_boxes),  # y2
        ]

        return {
            "type": "Figure",
            "filepath": [f["filepath"] for f in figures],
            "number": figures[0]["number"],
            "caption": captions[0] if captions else None,
            "footnotes": next((f["footnotes"] for f in figures if f.get("footnotes")), None),
            "page": figures[0]["page"],
            "box": merged_box,
        }

    def merge_close_figures(self, blocks: List[Any]) -> List[Any]:
        result = []
        i = 0

        while i < len(blocks):
            if blocks[i]["type"] != "Figure":
                result.append(blocks[i])
                i += 1
                continue

            # Start a new group with this figure
            group = [blocks[i]]
            j = i + 1

            while j < len(blocks):
                if blocks[j]["type"] != "Figure":
                    break  # Non-figure block interrupts the group
                
                candidate = blocks[j]
                same_page = candidate["page"] == group[-1]["page"]
                close_enough = self._box_distance(group[-1]["box"], candidate["box"]) <= self.distance_threshold

                if same_page and close_enough:
                    group.append(candidate)
                    j += 1
                else:
                    break

            # Validate: only merge if exactly one caption exists in the group
            captions = [f["caption"] for f in group if f.get("caption")]
            if len(group) > 1 and len(captions) == 1:
                result.append(self._merge_figure_group(group))
            else:
                result.extend(group)  # Don't merge, keep as-is

            i = j

        return result

    def upgrade_all(self, input_folder, folder_list, output_folder_path) -> None:
        for folder in folder_list:
            content_file_path = os.path.join(input_folder, folder, f"{folder}_content.json")
            with open(content_file_path, "r") as content_file:
                document_content = json.load(content_file)
            if self.correct_unicodes:
                document_content["blocks"] = self.correct_unicodes_in_blocks(document_content["blocks"])
            if self.merge_figures:
                document_content["blocks"] = self.merge_close_figures(document_content["blocks"])
            output_file_path = os.path.join(output_folder_path, folder, f"{folder}_content.json")
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w") as output_file:
                json.dump(document_content, output_file, indent=4)

    def upgrade_partial(self, input_folder, file_list, output_folder_path) -> None:
        for file in file_list:
            if file.endswith("_content.json"):
                content_file_path = os.path.join(input_folder, file)
                with open(content_file_path, "r") as content_file:
                    document_content = json.load(content_file)
                if self.correct_unicodes:
                    document_content["blocks"] = self.correct_unicodes_in_blocks(document_content["blocks"])
                if self.merge_figures:
                    document_content["blocks"] = self.merge_close_figures(document_content["blocks"])
                output_file_path = os.path.join(output_folder_path, file)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "w") as output_file:
                    json.dump(document_content, output_file, indent=4)

    def upgrade(self, input_folder) -> None:
        folder_list = os.listdir(input_folder)
        output_folder_path = input_folder + "_upgraded"
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        if self.all_documents:
            self.upgrade_all(input_folder, folder_list, output_folder_path)
        else:
            self.upgrade_partial(input_folder, folder_list, output_folder_path)

REPLACEMENTS = {
    "\ufb02": "fl",
    "\ufffd": "-",
    "\\xa": " ",
    "\ufb01": "fi",
    "\u25e6": "°",
    "\u00b0": "°",
    "\u2212": "-",
    "\u22c5": ".",
    "\uff65": ".",
    "\u00c5": "A°",
    "\u03bc": "u",
    "\u2013": "-",
    "\u2220": "<",
    "\u2219": ".",
    "\u00f8": "o",
    "\u0394\u03bd": "dv",
    "\u22ef": "...",
    "\u00b5": "u",
    "\u00b7": ".",
    "\\t": " ",
    "\u2010": "-",
    "\\uf053": "E",
    "g\u00a2": "g-",
    "\u00a2": "c",
    "\u00b1": "+/-",
    "\u2022": ".",
    "\ufb00": "ff",
    "\u03b3": "y",
    "\u03b8 ": "0",
    "\u00bc": "=",
    "\u00f0": "(",
    "\u00de": ")",
    "\u00a8": "",
    "\u00d7": "x",
    "\u201c": '"',
    "\u201d": '"',
    "\u2019\u2019": '"',
    "\u2019": '"',
    "\u2032": '"',
    "\u02da": "",
    "\u2460": "1",
    "\u2461": "2",
    "\u2462": "3",
    "\u00fe": "+",
    "\u03b8": "0",
    "\u2014": "-",
    "ACHTUNGTRENUNG": "",
    "  ": " ",
    ' ",': "",
    ' "]': "",
    "\\x02": "-",
    "\\x03": "-",
    "\\x01C": "°C",
    "\\u202f": " ",
    "\\u2002 \\u2009": " ",
    "\\u2009": " ",
    "\\x00": "-",
    "g\\x04": "g-",
    "\\x04": "<=",
    "\\x18": "",
    "g\\x01": "-",
    "\\x01": "",
    "𝜇": "u",
    "\u00f6": "o",
    "\ufb03": "fi",
    "\u2018": "",
    "\u00e6": "A°",
    "\u0394": "",
    "h\\x01": "-",
}