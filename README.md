# pdf2data
[![PyPI version](https://badge.fury.io/py/pdf2data-tools.svg)](https://pypi.org/project/pdf2data-tools/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Transforms PDF files into machine-readable JSON files. Extracts tables, figures, text blocks, metadata, and references from scientific papers and documents.

> **Note:** The repository is under active development for an article publication. Some errors are expected. Please report any issues on the [issues page](https://github.com/Pocoyo7798/pdf2data/issues).

## Installation

### From PyPI (recommended)

```bash
pip install pdf2data-tools
```

### With optional dependencies

```bash
# For the full PDF2Data pipeline (layout detection, OCR, etc.)
pip install pdf2data-tools[pdf2data_pipeline]
```

### From source (development)

```bash
conda create --name pdf2data python=3.10
conda activate pdf2data
git clone git@github.com:Pocoyo7798/pdf2data.git
cd pdf2data
pip install -e .
```

## Usage

### As a library

```python
from pdf2data.pdf2data_pipeline import PDF2Data

pipeline = PDF2Data(
    layout_model="DocLayout-YOLO-DocStructBench",
    input_folder="path/to/pdfs",
    output_folder="path/to/results",
)
```

### Command line

```bash
# Extract tables and figures
pdf2data_block path_to_folder path_to_results

# Extract text
pdf2data_text path_to_folder path_to_results

# Extract metadata
pdf2data_metadata path_to_folder path_to_results

# Extract references
pdf2data_references path_to_folder path_to_results
```

## Choosing a conversion pipeline

pdf2data does not lock you into a single PDF parser. The `pdf2data` command and its underlying pipeline classes support several conversion backends, selected with the `--pipeline` option:

| Pipeline value      | Backend                                   | Extra install                          |
| -------------------- | ------------------------------------------ | --------------------------------------- |
| `MinerU` (default)   | [MinerU](https://github.com/opendatalab/MinerU) pipeline backend | `pip install mineru`                    |
| `MinerUVL`           | MinerU's vision-language model backend     | `pip install mineru-vl-utils vllm transformers` |
| `Docling`            | [Docling](https://github.com/docling-project/docling) document converter | `pip install docling`                   |
| `PaddlePPStructure`  | PaddleOCR's PP-StructureV3 layout model    | `pip install pdf2data-tools[pdf2data_pipeline]` |
| `PaddleVL`           | PaddleOCR-VL, served over a local vLLM endpoint | `pip install pdf2data-tools[pdf2data_pipeline]` |

Each backend still produces the same `*_content.json` output shape (blocks, tables, figures, references, metadata), so downstream steps like keyword search and table-data extraction work the same regardless of which pipeline you pick.

### Command line

```bash
# Default: MinerU
pdf2data path/to/pdfs path/to/results --pipeline MinerU

# MinerU's VLM backend
pdf2data path/to/pdfs path/to/results --pipeline MinerUVL

# Docling
pdf2data path/to/pdfs path/to/results --pipeline Docling

# PaddleOCR PP-StructureV3
pdf2data path/to/pdfs path/to/results --pipeline PaddlePPStructure

# PaddleOCR-VL
pdf2data path/to/pdfs path/to/results --pipeline PaddleVL
```

### As a library

Each backend is exposed as its own `Pipeline` subclass, all sharing the same `input_folder` / `output_folder` / `extract_references` constructor arguments and the same `.pdf_transform()` entry point:

```python
# MinerU
from pdf2data.mineru import MinerU

pipeline = MinerU(
    input_folder="path/to/pdfs",
    output_folder="path/to/results",
    extract_references=True,
)
pipeline.pdf_transform()
```

```python
# MinerU (vision-language model backend)
from pdf2data.mineru_vlm import MinerUVLM

pipeline = MinerUVLM(
    input_folder="path/to/pdfs",
    output_folder="path/to/results",
    extract_references=True,
)
pipeline.pdf_transform()
```

```python
# Docling
from pdf2data.docling import Docling

pipeline = Docling(
    input_folder="path/to/pdfs",
    output_folder="path/to/results",
    extract_references=True,
)
pipeline.pdf_transform()
```

```python
# PaddleOCR PP-StructureV3 or PaddleOCR-VL
from pdf2data.padle_pipeline import PaddlePPStructure

pipeline = PaddlePPStructure(
    extractor_name="PaddlePPStructure",  # or "PaddleVL"
    input_folder="path/to/pdfs",
    output_folder="path/to/results",
    extract_references=True,
)
pipeline.pdf_transform()
```

> **Note:** `PaddleVL` expects a vLLM server for PaddleOCR-VL already running and reachable at `http://127.0.0.1:8118`.

## Finding text and tables by keyword

Once a PDF has been converted into a `*_content.json` file, you can search the extracted blocks for specific keywords instead of reading through every document by hand.

### Finding tables and figures containing keywords

`pdf2data_find_blocks` scans the `Table` and/or `Figure` blocks of every converted document for matches against a keyword list, and writes the matching blocks to a single JSON file.

```bash
pdf2data_find_blocks path/to/results path/to/search_results keywords.txt
```

- `keywords.txt` is a plain text file with one keyword (or regex-friendly phrase) per line.
- `--generic_file` optionally points to a fallback keyword list, used only when no document-specific keyword is found.
- `--find_tables` / `--find_figures` (default `True` / `False`) control which block types are searched.

```bash
pdf2data_find_blocks path/to/results path/to/search_results keywords.txt \
    --generic_file generic_keywords.txt \
    --find_tables True \
    --find_figures True
```

The output is a single `found_blocks.json` file mapping each document name to the list of matching blocks (plus its DOI, when available).

### Finding relevant text passages

`pdf2data_find_text` scores paragraphs and/or section headers against a weighted keyword list and keeps only the ones that pass a minimum score threshold.

```bash
pdf2data_find_text path/to/results path/to/search_results keywords.json
```

Here `keywords.json` is a dictionary mapping each keyword to an integer weight, e.g.:

```json
{
    "zeolite": 3,
    "synthesis": 2,
    "characterization": 1
}
```

Useful options:

```bash
pdf2data_find_text path/to/results path/to/search_results keywords.json \
    --word_count_threshold 6 \
    --find_paragraphs True \
    --find_section_headers True \
    --count_duplicates False
```

- `--word_count_threshold` is the minimum accumulated weight a text block needs to be kept.
- `--count_duplicates` controls whether the same keyword found multiple times in a block counts once or every time.

Results are written as two aligned files, `found_texts.txt` (the matching passages) and `found_texts_doc_names.txt` (the source document for each passage).

## Extracting structured data from tables

Once you've located the relevant tables (for example with `pdf2data_find_blocks`), `TableExtractor` turns each table block into structured, unit-aware data by matching column headers against a keyword registry.

```python
from pdf2data.table_extractor import TableExtractor

extractor = TableExtractor(table_type="characterization")
extractor.extract_tables("path/to/search_results/found_blocks.json", "path/to/extracted_tables.jsonl")
```

`extract_tables` reads a `found_blocks.json`-style file, extracts every table it contains, and appends one JSON object per table to the output file, with the sample values grouped by the registry key they matched and their associated unit.

You can also run the extraction on a single table dictionary and inspect the result directly:

```python
from pdf2data.table_extractor import TableExtractor

extractor = TableExtractor(table_type="characterization")
result = extractor.extract_table(table_dict)

print(result.column_matches)  # which columns were recognized, and with what confidence
print(result.rows)            # extracted (value, unit) pairs per row
print(result.transposed)      # True if the table had to be transposed to be read correctly
```

`TableExtractor` currently ships with a `characterization` registry (targeted at zeolite characterization data); passing any other `table_type` raises a `ValueError`.

## Update and Publish (PyPI)

Use this flow when you make changes and want to publish a new package version.

```bash
# 1) Bump version in pyproject.toml
# [project]
# version = "0.0.2"

# 2) (Optional) Keep __version__ in sync
# edit pdf2data/__init__.py

# 3) Install/reinstall build tools
python -m pip install --upgrade build twine

# 4) Clean previous artifacts
rm -rf dist build *.egg-info

# 5) Build package
python -m build

# 6) Validate distribution files
python -m twine check dist/*

# 7) Upload to PyPI
python -m twine upload dist/*
```

When prompted by `twine`:
- Username: `__token__`
- Password: your PyPI token (`pypi-...`)

Verify the release:

```bash
pip install --upgrade pdf2data-tools
pip show pdf2data-tools
```

## License

Apache Software License 2.0
