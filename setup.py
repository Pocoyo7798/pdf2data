#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click<=8.3.1",
    "PyMuPDF<=1.26.7",
    "pylatexenc<=2.10",
    #"numpy<=2.1.3",
    "pydantic<=2.12.5",
    "beautifulsoup4<=4.14.3",
    "pdf2doi<=1.7",
    #"bibcure<=0.3.0",
    #"importlib_resources<=6.5.2",
    #"lxml<=6.0.0",
    #"transformers<=4.53.1",
    #"tensorflow<=2.20.0",
    #"cython<=3.0.8",
    #"effdet<=0.3.0",
    #"pillow<=11.3.0",
    "Levenshtein<= 0.27.3",
    "trieregex<=1.0.0",
    #"struct_eqtable<=0.3.3",
    #"pdfminer.six==20250506"
    #"easyocr<=1.7.2",
    #"docling<=2.68.0",
    "bibtexparser<=1.4.3",
    "pypdf>=3.1.0",
]

test_requirements = [
    "pytest>=3",
]

mineru_requirements = [

]

mineruvlm_requirements = [
]
pdf2data_pipeline_requirements = [
    "torch<=2.10.0",
    "opencv-python<=4.13.0.92",
    "tensorflow<=2.20.0",
    "doclayout_yolo<=0.0.4",
    "pdf2image<=1.17.0",
    "paddleocr<=3.4.0",
    "paddlepaddle<=3.3.0"
]

docling_requirements = [

]

paddle_structure_requirements = [

]
paddle_vl_requirements = [
    
]



setup(
    author="Daniel Pereira Costa",
    author_email="daniel.pereira.costa@tecnico.ulisboa.pt",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Trnsforms pdf files into machine readable json files",
    entry_points={
        "console_scripts": [
            "pdf2data_metadata=pdf2data.cli.metadata_finder:main",
            "pdf2data_references=pdf2data.cli.reference_extractor:main",
            "pdf2data_text=pdf2data.cli.text_extractor:main",
            "pdf2data_block=pdf2data.cli.block_extractor:main",
            "pdf2data_eval=pdf2data.cli.evaluator:main",
            "pdf2data_detect_image=pdf2data.cli.table_detector:main",
            "pdf2data=pdf2data.cli.pdf2data:main",
            "pdf2data_find_blocks=pdf2data.cli.block_finder:main",
            "pdf2data_find_text=pdf2data.cli.text_finder:main"
        ],
    },
    install_requires=requirements,
    extras_require={
        "test": test_requirements,
        "pdf2data_pipeline": pdf2data_pipeline_requirements,
        "mineru": mineru_requirements,
        "mineruvlm": mineruvlm_requirements,
        "docling": docling_requirements,
        "paddle_structure": paddle_structure_requirements,
        "paddle_vl": paddle_vl_requirements,
    },
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pdf2data",
    name="pdf2data",
    packages=find_packages(include=["pdf2data", "pdf2data.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/pocoyo7798/pdf2data",
    version="0.0.1",
    zip_safe=False,
)
