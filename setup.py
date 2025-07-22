#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click<=8.1.7",
    "numpy<=2.1.3",
    "pydantic<=2.11.7",
    "beautifulsoup4<=4.13.4",
    "pdf2doi<=1.7",
    "bibcure<=0.3.0",
    "importlib_resources<=6.5.2",
    "lxml<=6.0.0",
    "transformers<=4.53.1",
    "tensorflow<=2.19.0",
    "cython<=3.0.8",
    "effdet<=0.3.0",
    "pillow>=11.3.0",
    "levenshtein<=0.27.1",
    "trieregex<=1.0.0",
    "doclayout_yolo<=0.0.4",
    "lmdeploy<=0.9.1",
    "struct_eqtable<=0.3.3"
]

test_requirements = [
    "pytest>=3",
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
