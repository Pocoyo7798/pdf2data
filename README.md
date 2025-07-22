========
pdf2data
========


.. image:: https://img.shields.io/pypi/v/pdf2data.svg
        :target: https://pypi.python.org/pypi/pdf2data

.. image:: https://img.shields.io/travis/pocoyo7798/pdf2data.svg
        :target: https://travis-ci.com/pocoyo7798/pdf2data

.. image:: https://readthedocs.org/projects/pdf2data/badge/?version=latest
        :target: https://pdf2data.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Trnsforms pdf files into machine readable json files
The repository is under transformation for a article publication some erros are expected. Report everything you find on the issues page, please.


* Free software: Apache Software License 2.0
* Documentation: https://pdf2data.readthedocs.io.


Installation
--------

```bash
conda create --name pdf2data python=3.10
conda activate pdf2data
git clone git@github.com:Pocoyo7798/pdf2data.git
cd zs4procext
pip install -e .
```

Run the tool
-------
#Extract Tables and Figures
```bash
pdf2data_block path_to_folder path_to_results
```
