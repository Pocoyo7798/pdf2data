# pdf2data
Transforms pdf files into machine readable json files
The repository is under transformation for a article publication some erros are expected. Report everything you find on the issues page, please.

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
