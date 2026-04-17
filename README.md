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
