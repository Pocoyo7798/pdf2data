from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, PrivateAttr
import importlib_resources
import json
import os

@dataclass
class ColumnMatch:
    col_index: int
    raw_header: str
    registry_key: str
    score: float
    extracted_unit: Optional[str]   # unit parsed from the header; None if absent

@dataclass
class CellValue:
    value: str
    unit: str           # header unit if present, else entry's default_unit

    def __repr__(self) -> str:
        return f"({self.value!r}, {self.unit!r})"

@dataclass
class ExtractedRow:
    raw_row: List[str]
    data: Dict[str, CellValue]   # registry_key → (value, unit) tuple

@dataclass
class TableExtractionResult:
    table_number: int
    caption: str
    column_matches: List[ColumnMatch]
    rows: List[ExtractedRow]

class TableExtractor(BaseModel):
    """
    Responsible for extracting structured data from table blocks.
    
    Encapsulates the registry-based keyword matching and column/row extraction logic.
    """
    table_type: str = "characterization"  # for now we only have one type, but this allows future extensibility
    _registry: Dict[str, Dict[str, Any]] = PrivateAttr(default=None)
    _unit_pattern: re.Pattern = PrivateAttr(default=None)  # matches text within parentheses

    def model_post_init(self, __context: Any) -> None:
        if self.table_type == "characterization":
            file_path: str = str(
                importlib_resources.files("pdf2data") / "resources" / "zeolite_characterization.json"
            )
        else:
            raise ValueError(f"Unknown table_type: {self.table_type!r}")
        with open(file_path, "r") as f:
            self._registry = json.load(f)
        self._unit_pattern = re.compile(r"\(([^)]+)\)\s*$")

    def _normalise(self, text: str) -> str:
        """Lowercase, strip accents, collapse whitespace."""
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _split_header(self, raw: str) -> Tuple[str, Optional[str]]:
        """
        Split 'SBETa (m2/g)' → ('sbeta', 'm2/g').
        Returns (normalised_text, unit_string_or_None).

        Footnote letters attached to the word (e.g. 'Vmesob', 'SBETa') are NOT
        stripped — substring keyword matching handles them transparently.
        """
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
        """Sum weights of all matching keywords."""
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
        """
        Return True if the header unit is compatible with the entry.

        Rules:
        - Header HAS a unit  → it must appear in entry's "units" list.
                                If "units" is empty, any unit (or none) is accepted.
        - Header has NO unit → always valid; default_unit will be applied later.
        """
        if header_unit is None:
            return True   # no unit in header → default_unit takes over, always ok

        accepted: List[str] = entry.get("units", [])
        if not accepted:
            return True   # entry accepts any unit

        norm_accepted = {self._normalise_unit(u) for u in accepted}
        return self._normalise_unit(header_unit) in norm_accepted

    def extract_table(
        self,
        table: Dict[str, Any],
    ) -> TableExtractionResult:
        """
        Extract structured data from a single table dict.
        
        Args:
            table: Table dict with keys: "block" (list of rows), "column_headers" (header rows),
                   "number" (table ID), "caption" (table caption).
        
        Returns:
            TableExtractionResult with matched columns and extracted rows.
        """
        block: List[List[str]] = table["block"]
        header_row_indices: List[int] = table.get("column_headers", [0])
        column_header_row: int = header_row_indices[0] if header_row_indices else 0
        headers: List[str] = block[column_header_row]
        
        column_matches = self._identify_columns(headers)
        rows = self._extract_rows(block, column_matches, header_row_indices)
        
        return TableExtractionResult(
            table_number=table.get("number", -1),
            caption=table.get("caption", ""),
            column_matches=column_matches,
            rows=rows,
        )

    def extract_tables(
        self,
        input_file: str,
        output_file: str,
    ) -> List[TableExtractionResult]:
        with open(input_file, "r") as f:
            data: Dict[str, Any] = json.load(f)
        if os.path.exists(output_file):
            os.remove(output_file)
        for document_name in data.keys():
            if len(data[document_name]["blocks"]) == 0  :
                continue
            else:
                for table in data[document_name]["blocks"]:
                    result = self.extract_table(table)
                    table_dict = {"doc_name": document_name, "filepath": table["filepath"], "samples": []}
                    for row in result.rows:
                        final_dict = {key: (cv.value, cv.unit) for key, cv in row.data.items()}
                        table_dict["samples"].append(final_dict)
                    with open(output_file, "a") as f:
                        f.write(str(table_dict) + "\n")

    def _identify_columns(
        self,
        headers: List[str],
    ) -> List[ColumnMatch]:
        """
        Walk the registry IN ORDER for each header.
        The FIRST entry that passes threshold + unit validation wins.
        
        Args:
            headers: List of column header strings.
        
        Returns:
            List of ColumnMatch objects (may be shorter than headers).
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

                # First match wins
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

    def _extract_rows(
        self,
        block: List[List[str]],
        column_matches: List[ColumnMatch],
        header_row_indices: List[int],
    ) -> List[ExtractedRow]:
        """
        Iterate over non-header rows and emit CellValue(value, unit) per cell.
        The unit is the header-extracted unit when present, otherwise default_unit
        from the matching registry entry.
        
        Args:
            block: 2D list of cell strings.
            column_matches: List of ColumnMatch objects from _identify_columns.
            header_row_indices: Row indices that are headers (to skip).
        
        Returns:
            List of ExtractedRow objects.
        """
        col_map: Dict[int, ColumnMatch] = {m.col_index: m for m in column_matches}
        rows: List[ExtractedRow] = []

        for row_idx, row in enumerate(block):
            if row_idx in header_row_indices:
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