from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel


Box4 = List[float]


class EditTarget(TypedDict, total=False):
    kind: Literal["block", "tableCell", "tableCaption"]
    block_index: int
    row: int
    col: int
    caption_index: int
    page: int
    box: Box4


class JsonBoxEditor(BaseModel):
    """Utility action to visualize/edit JSON targets by bounding box."""

    data: Dict[str, Any]

    @staticmethod
    def to_box4(value: Any) -> Optional[Box4]:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            nums = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
        return nums

    @staticmethod
    def is_table_with_grid(block: Dict[str, Any]) -> bool:
        return str(block.get("type", "")).lower() == "table" and isinstance(block.get("block"), list)

    @staticmethod
    def get_table_dimensions(table_rows: List[Any]) -> Dict[str, int]:
        rows = len(table_rows)
        cols = 0
        for row in table_rows:
            if isinstance(row, list):
                cols = max(cols, len(row))
        return {"rows": rows, "cols": max(1, cols)}

    @classmethod
    def get_cell_boxes_matrix(
        cls,
        block: Dict[str, Any],
        row_count: int,
        col_count: int,
    ) -> List[List[Optional[Box4]]]:
        matrix: List[List[Optional[Box4]]] = [
            [None for _ in range(col_count)] for _ in range(row_count)
        ]

        raw = block.get("cell_boxes")
        if not isinstance(raw, list):
            return matrix

        # Preferred shape: cell_boxes[row][col] = [x1, y1, x2, y2]
        if any(isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], list) for entry in raw):
            for r in range(row_count):
                row = raw[r] if r < len(raw) and isinstance(raw[r], list) else []
                for c in range(col_count):
                    value = row[c] if c < len(row) else None
                    matrix[r][c] = cls.to_box4(value)
            return matrix

        # Legacy flat fallback: cell_boxes[idx] = [x1, y1, x2, y2]
        for r in range(row_count):
            for c in range(col_count):
                idx = r * col_count + c
                value = raw[idx] if idx < len(raw) else None
                matrix[r][c] = cls.to_box4(value)
        return matrix

    @classmethod
    def get_caption_boxes(cls, block: Dict[str, Any]) -> List[Box4]:
        singular = cls.to_box4(block.get("caption_box"))
        if singular is not None:
            return [singular]

        raw = block.get("caption_boxes")
        if raw is None:
            return []

        if isinstance(raw, list) and len(raw) == 4 and not isinstance(raw[0], list):
            single = cls.to_box4(raw)
            return [single] if single is not None else []

        if isinstance(raw, list):
            boxes: List[Box4] = []
            for entry in raw:
                box = cls.to_box4(entry)
                if box is not None:
                    boxes.append(box)
            return boxes

        return []

    @staticmethod
    def caption_position(block: Dict[str, Any]) -> Optional[str]:
        """Return caption position relative to table: above, below, overlap, or None."""
        table_box = JsonBoxEditor.to_box4(block.get("box"))
        caption_box = JsonBoxEditor.to_box4(block.get("caption_box"))
        if table_box is None or caption_box is None:
            return None

        table_mid_y = (table_box[1] + table_box[3]) / 2.0
        caption_mid_y = (caption_box[1] + caption_box[3]) / 2.0
        if caption_box[3] <= table_box[1] or caption_mid_y < table_mid_y:
            return "above"
        if caption_box[1] >= table_box[3] or caption_mid_y > table_mid_y:
            return "below"
        return "overlap"

    def blocks(self) -> List[Dict[str, Any]]:
        blocks = self.data.get("blocks", [])
        return blocks if isinstance(blocks, list) else []

    def list_targets(self, page: Optional[int] = None) -> List[EditTarget]:
        targets: List[EditTarget] = []
        for block_index, block in enumerate(self.blocks()):
            block_page = int(block.get("page", 1))
            if page is not None and block_page != page:
                continue

            box = self.to_box4(block.get("box"))
            if box is not None:
                targets.append(
                    {
                        "kind": "block",
                        "block_index": block_index,
                        "page": block_page,
                        "box": box,
                    }
                )

            if not self.is_table_with_grid(block):
                continue

            table_rows = block.get("block", [])
            dims = self.get_table_dimensions(table_rows if isinstance(table_rows, list) else [])
            matrix = self.get_cell_boxes_matrix(block, dims["rows"], dims["cols"])

            for row in range(dims["rows"]):
                for col in range(dims["cols"]):
                    cell_box = matrix[row][col]
                    if cell_box is None:
                        continue
                    targets.append(
                        {
                            "kind": "tableCell",
                            "block_index": block_index,
                            "row": row,
                            "col": col,
                            "page": block_page,
                            "box": cell_box,
                        }
                    )

            caption_boxes = self.get_caption_boxes(block)
            for caption_index, caption_box in enumerate(caption_boxes):
                targets.append(
                    {
                        "kind": "tableCaption",
                        "block_index": block_index,
                        "caption_index": caption_index,
                        "page": block_page,
                        "box": caption_box,
                    }
                )

        return targets

    def update_target(self, target: EditTarget, value: str) -> None:
        kind = target.get("kind")
        block_index = target.get("block_index")
        if kind is None or block_index is None:
            raise ValueError("Target must include 'kind' and 'block_index'.")

        blocks = self.blocks()
        if block_index < 0 or block_index >= len(blocks):
            raise IndexError("block_index out of range")
        block = blocks[block_index]

        if kind == "tableCell":
            row = target.get("row")
            col = target.get("col")
            if row is None or col is None:
                raise ValueError("tableCell target must include 'row' and 'col'.")
            table = block.get("block")
            if not isinstance(table, list):
                table = []
                block["block"] = table
            while len(table) <= row:
                table.append([])
            if not isinstance(table[row], list):
                table[row] = []
            while len(table[row]) <= col:
                table[row].append("")
            table[row][col] = value
            return

        if kind == "tableCaption":
            block["caption"] = value
            return

        # Generic text-like block
        block["content"] = value

    def to_canonical_content_json(self) -> Dict[str, Any]:
        metadata = self.data.get("metadata") if isinstance(self.data.get("metadata"), dict) else {}
        references = self.data.get("references") if isinstance(self.data.get("references"), list) else []
        blocks_input = self.blocks()

        blocks: List[Dict[str, Any]] = []
        optional_keys = [
            "filepath",
            "number",
            "caption",
            "footnotes",
            "block",
            "cell_boxes",
            "caption_box",
            "caption_boxes",
            "column_headers",
            "row_indexes",
        ]

        for block in blocks_input:
            box = self.to_box4(block.get("box"))
            if box is None:
                continue

            canonical: Dict[str, Any] = {
                "type": str(block.get("type", "paragraph")),
                "content": str(block.get("content", "")),
                "page": int(block.get("page", 1)),
                "box": box,
            }

            for key in optional_keys:
                if key in block:
                    canonical[key] = block[key]

            blocks.append(canonical)

        return {"metadata": metadata, "blocks": blocks, "references": references}
