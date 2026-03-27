from __future__ import annotations
 
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, PrivateAttr
import json
import os
 
 
@dataclass
class ColumnMatch:
    col_index: int
    raw_header: str
    registry_key: str
    score: float
    extracted_unit: Optional[str]
 
 
@dataclass
class CellValue:
    value: str
    unit: str
 
    def __repr__(self) -> str:
        return f"({self.value!r}, {self.unit!r})"
 
 
@dataclass
class ExtractedRow:
    raw_row: List[str]
    data: Dict[str, CellValue]
 
 
@dataclass
class TableExtractionResult:
    table_number: int
    caption: str
    column_matches: List[ColumnMatch]
    rows: List[ExtractedRow]
    transposed: bool = False   # True if block was transposed before extraction
 
 
class TableExtractor(BaseModel):
    """
    Extracts structured data from table blocks using a weighted keyword registry.
    """
    table_type: str = "characterization"
    _registry: Dict[str, Dict[str, Any]] = PrivateAttr(default=None)
    _unit_pattern: re.Pattern = PrivateAttr(default=None)
 
    def model_post_init(self, __context: Any) -> None:
        if self.table_type == "characterization":
            import importlib_resources
            file_path: str = str(
                importlib_resources.files("pdf2data") / "resources" / "zeolite_characterization.json"
            )
        else:
            raise ValueError(f"Unknown table_type: {self.table_type!r}")
        with open(file_path, "r") as f:
            self._registry = json.load(f)
        self._unit_pattern = re.compile(r"\(([^)]+)\)\s*$")
 
    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
 
    def extract_table(self, table: Dict[str, Any]) -> TableExtractionResult:
        """
        Extract structured data from a single table dict.
 
        Strategy
        --------
        1. Merge all header rows into one merged header list.
        2. Run column identification on the natural orientation.
        3. If fewer than half the columns matched, transpose the block and
           try again with the first row as header.
        4. Return whichever orientation matched more columns, emitting a
           UserWarning when both are weak.
 
        Args:
            table: dict with keys "block", "column_headers", "number", "caption".
 
        Returns:
            TableExtractionResult with a `transposed` flag.
        """
        block: List[List[str]] = table["block"]
        header_row_indices: List[int] = table.get("column_headers", [0])
 
        # ── natural orientation ────────────────────────────────────────────
        nat_matches, nat_rows = self._extract_orientation(block, header_row_indices)
 
        # Strict trigger: transpose only when 0 or 1 columns were matched
        if len(nat_matches) >= 2:
            return TableExtractionResult(
                table_number=table.get("number", -1),
                caption=table.get("caption", ""),
                column_matches=nat_matches,
                rows=nat_rows,
                transposed=False,
            )
 
        # ── transposed orientation ─────────────────────────────────────────
        t_block = self._transpose(block)
        t_matches, t_rows = self._extract_orientation(t_block, [0])
 
        use_transposed = len(t_matches) >= len(nat_matches)
 
        # Warn whenever both orientations are also weak (≤ 1 match each)
        if max(len(nat_matches), len(t_matches)) <= 1:
            chosen = "transposed" if use_transposed else "natural"
            warnings.warn(
                f"Table {table.get('number', '?')}: both orientations matched 0 or 1 "
                f"columns (natural={len(nat_matches)}, transposed={len(t_matches)}). "
                f"Using {chosen}.",
                UserWarning,
                stacklevel=2,
            )
 
        if use_transposed:
            return TableExtractionResult(
                table_number=table.get("number", -1),
                caption=table.get("caption", ""),
                column_matches=t_matches,
                rows=t_rows,
                transposed=True,
            )
        else:
            return TableExtractionResult(
                table_number=table.get("number", -1),
                caption=table.get("caption", ""),
                column_matches=nat_matches,
                rows=nat_rows,
                transposed=False,
            )
 
    def extract_tables(self, input_file: str, output_file: str) -> None:
        with open(input_file, "r") as f:
            data: Dict[str, Any] = json.load(f)
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, "a") as out_f:
            for document_name, document in data.items():
                if not document["blocks"]:
                    continue
                for table in document["blocks"]:
                    result = self.extract_table(table)
                    table_dict = {
                        "doc_name": document_name,
                        "filepath": table["filepath"],
                        "transposed": result.transposed,
                        "samples": [
                            {key: {"value": cv.value, "unit": cv.unit}
                             for key, cv in row.data.items()}
                            for row in result.rows
                        ],
                    }
                    out_f.write(json.dumps(table_dict) + "\n")
 
    # ------------------------------------------------------------------ #
    # Orientation helpers                                                  #
    # ------------------------------------------------------------------ #
 
    def _extract_orientation(
        self,
        block: List[List[str]],
        header_row_indices: List[int],
    ) -> Tuple[List[ColumnMatch], List[ExtractedRow]]:
        """Merge headers, identify columns, extract rows for one orientation."""
        merged_headers = self._merge_headers(block, header_row_indices)
        column_matches = self._identify_columns(merged_headers)
        rows = self._extract_rows(block, column_matches, header_row_indices)
        return column_matches, rows
 
    def _merge_headers(
        self,
        block: List[List[str]],
        header_row_indices: List[int],
    ) -> List[str]:
        """
        Merge one or more header rows into a single list of strings.
 
        Cells from each header row are joined with a space per column,
        skipping empty strings so sparse multi-row headers collapse cleanly.
 
        Example
        -------
        Row 0: ["",       "Surface area",  "Pore volume"   ]
        Row 1: ["Sample", "SBET (m2/g)",   "Vmeso (cm3/g)" ]
        →      ["Sample", "Surface area SBET (m2/g)", "Pore volume Vmeso (cm3/g)"]
        """
        if not header_row_indices:
            return []
        n_cols = max(len(block[i]) for i in header_row_indices)
        merged: List[str] = []
        for col in range(n_cols):
            parts = []
            for row_idx in header_row_indices:
                row = block[row_idx]
                cell = row[col].strip() if col < len(row) else ""
                if cell:
                    parts.append(cell)
            merged.append(" ".join(parts))
        return merged
 
    @staticmethod
    def _transpose(block: List[List[str]]) -> List[List[str]]:
        """
        Transpose a 2-D block (rows ↔ columns).
        Pads short rows with empty strings so the result is rectangular.
        """
        if not block:
            return []
        n_cols = max(len(row) for row in block)
        padded = [row + [""] * (n_cols - len(row)) for row in block]
        return [list(col) for col in zip(*padded)]
 
    # ------------------------------------------------------------------ #
    # Column identification                                                #
    # ------------------------------------------------------------------ #
 
    def _identify_columns(self, headers: List[str]) -> List[ColumnMatch]:
        """
        Walk the registry IN ORDER for each merged header.
        The FIRST entry that passes threshold + unit validation wins.
        """
        used_unique: set = set()
        matches: List[ColumnMatch] = []
 
        for col_idx, raw_header in enumerate(headers):
            header_text, header_unit = self._split_header(raw_header)
 
            for key, entry in self._registry.items():
                if entry.get("unique", True) and key in used_unique:
                    continue
                score = self._keyword_score(header_text, entry)
                if score < entry.get("threshold", 1.0):
                    continue
                if not self._unit_valid(header_unit, entry):
                    continue
                if entry.get("unique", True):
                    used_unique.add(key)
                matches.append(ColumnMatch(
                    col_index=col_idx,
                    raw_header=raw_header,
                    registry_key=key,
                    score=score,
                    extracted_unit=header_unit,
                ))
                break
 
        return matches
 
    # ------------------------------------------------------------------ #
    # Row extraction                                                       #
    # ------------------------------------------------------------------ #
 
    def _extract_rows(
        self,
        block: List[List[str]],
        column_matches: List[ColumnMatch],
        header_row_indices: List[int],
    ) -> List[ExtractedRow]:
        """
        Emit CellValue(value, unit) for each matched column in every data row.
        Unit = extracted header unit if present, else registry default_unit.
        """
        col_map: Dict[int, ColumnMatch] = {m.col_index: m for m in column_matches}
        header_set = set(header_row_indices)
        rows: List[ExtractedRow] = []
 
        for row_idx, row in enumerate(block):
            if row_idx in header_set:
                continue
            data: Dict[str, CellValue] = {}
            for col_idx, cell in enumerate(row):
                if col_idx in col_map:
                    match = col_map[col_idx]
                    unit = (
                        match.extracted_unit
                        if match.extracted_unit is not None
                        else self._registry.get(match.registry_key, {}).get("default_unit", "")
                    )
                    data[match.registry_key] = CellValue(value=cell.strip(), unit=unit)
            rows.append(ExtractedRow(raw_row=row, data=data))
 
        return rows
 
    # ------------------------------------------------------------------ #
    # Normalisation & scoring                                              #
    # ------------------------------------------------------------------ #
 
    def _normalise(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text
 
    def _split_header(self, raw: str) -> Tuple[str, Optional[str]]:
        raw_norm = self._normalise(raw)
        m = self._unit_pattern.search(raw_norm)
        if m:
            unit = m.group(1).strip()
            text = raw_norm[: m.start()].strip()
            return text, unit
        return raw_norm, None
 
    def _normalise_unit(self, unit: str) -> str:
        return self._normalise(unit).replace(" ", "")
 
    def _keyword_score(self, header_text: str, entry: Dict[str, Any]) -> float:
        limit_words: bool = entry.get("limit_words", False)
        score = 0.0
        for keyword, weight in entry["keywords"]:
            kw = self._normalise(keyword)
            if limit_words:
                pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
                if re.search(pattern, header_text):
                    score += weight
            else:
                if kw in header_text:
                    score += weight
        return score
 
    def _unit_valid(self, header_unit: Optional[str], entry: Dict[str, Any]) -> bool:
        if header_unit is None:
            return True
        accepted: List[str] = entry.get("units", [])
        if not accepted:
            return True
        norm_accepted = {self._normalise_unit(u) for u in accepted}
        return self._normalise_unit(header_unit) in norm_accepted