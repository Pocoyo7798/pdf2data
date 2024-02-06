#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click<=8.1.7",
    "numpy>=1.24.3",
    "pydantic<2.0",
    "beautifulsoup4>=4.12.3",
    "pdf2doi>=1.5",
    "bibcure>=0.3.0",
    "importlib_resources>=6.1.1",
    "lxml>=5.1.0",
    "transformers>=4.36.2",
    "layoutparser>=0.3.4",
    "tensorflow>=2.13.1",
    "layoutparser[layoutmodels]",
    "cython<=3.0.8",
    "effdet<=0.3.0",
    "pillow<=9.1.0",
    "levenshtein<=0.24.0",
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
            "pdf2data=pdf2data.cli.cli:main",
            "pdf2data_metadata=pdf2data.cli.metadata_finder:main",
            "pdf2data_references=pdf2data.cli.reference_extractor:main",
            "pdf2data_text=pdf2data.cli.text_extractor:main",
            "pdf2data_block=pdf2data.cli.block_extractor:main",
            "pdf2data_eval=pdf2data.cli.evaluator:main",
            "pdf2data_detect_image=pdf2data.cli.table_detector:main"
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
