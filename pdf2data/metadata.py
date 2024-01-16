import os
import subprocess
from typing import Any, Dict, List, Optional

import bibtexparser
import pdf2doi
from bs4 import BeautifulSoup as bs
from pydantic import BaseModel

from pdf2data.support import find_term_in_list, list_into_bs_format


class Metadata(BaseModel):
    file_path: str
    title: Optional[str] = None
    doi: Optional[str] = None
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    year: Optional[str] = None
    journal: Optional[str] = None

    def get_cerm(self) -> None:
        """Updates a metadata class object from a .cermxml file

        Raises
        ------
        AttributeError
            The file path does not refer to a .cermxml file
        """
        with open(self.file_path, "r") as file:
            content = file.readlines()
        extension = os.path.splitext(self.file_path)[1]
        if extension != ".cermxml":
            raise AttributeError(
                f"A .cermxml file should be provided instead of a .{extension} file to use this method"
            )
        content: str = "".join(content)
        bs_content: Any = bs(content, "lxml")
        article_metadata: List[str] = bs_content.find_all("article-meta")
        # Convert the list with bs4 objects into a single bs4 object
        article_metadata: Any = list_into_bs_format(article_metadata)
        journal_metadata: List[str] = bs_content.find_all("journal-meta")
        journal_metadata: Any = list_into_bs_format(journal_metadata)
        title: List[str] = find_term_in_list(article_metadata, "article-title")
        doi: List[str] = find_term_in_list(article_metadata, "article-id")
        authors: List[str] = find_term_in_list(article_metadata, "string-name")
        keywords: List[str] = find_term_in_list(article_metadata, "kwd")
        pub_date: List[str] = article_metadata.find_all("pub-date")
        pub_date: List[str] = list_into_bs_format(pub_date)
        year: List[str] = find_term_in_list(pub_date, "year")
        journal: List[str] = find_term_in_list(journal_metadata, "journal-title")
        if doi == ["Nothing Found"]:
            pdf2doi.config.set("verbose", False)
            identifier = pdf2doi.pdf2doi(self.file_path)
            if bool(identifier) is True:
                doi = [identifier["identifier"]]
        self.title = title
        self.doi = doi
        self.authors = authors
        self.keywords = keywords
        self.year = year
        self.journal = journal

    def get_api(self) -> None:
        """Creata a class object from a .pdf file (file_path)"""
        # Run command to get pdf title using pdftitle
        title_output: Any = subprocess.check_output(
            ["pdftitle", "-p", self.file_path, "--replace-missing-char", '" "', "-t"],
        )
        # Transform the output into a string
        title_string: str = title_output.decode("utf-8")
        # remove \n from string
        title: List[str] = [title_string.replace("\n", "")]
        doi: List[str] = ["Nothing Found"]
        authors: List[str] = ["Nothing Found"]
        keywords: List[str] = ["Nothing Found"]
        year: List[str] = ["Nothing Found"]
        journal: List[str] = ["Nothing Found"]
        # Set the pdf2doi configurations
        pdf2doi.config.set("verbose", False)
        # Obtain a dictionary with possible identifiers
        identifier: Any = pdf2doi.pdf2doi(self.file_path)
        # Verify if the dictionaty is not blank
        if bool(identifier) is True:
            # Retrieve DOI
            doi = [identifier["identifier"]]
        # Get DOI metadata from dx.doi.org
        try:
            pdf_info: Dict[str, Any] = subprocess.check_output(
                [
                    'curl -LH "Accept: application/x-bibtex" "http://dx.doi.org/'
                    + doi[0]
                    + '"'
                ],
                shell=True,
                text=True,
                input="y",
            )
            with open("bibtext.bib", "w") as bibfile:
                bibfile.write(pdf_info)
            with open("bibtext.bib") as bibfile:
                pdf_info_bib = bibtexparser.load(bibfile)
            # Delete the .bib file
            os.remove("bibtext.bib")
            # Verify if there is any entry
            if len(pdf_info_bib.entries) > 0:
                pdf_info_dic = pdf_info_bib.entries[0]
                # Verify if 'title' is a valid key
                if "title" in pdf_info_dic.keys():
                    # Verify if the new title has more chars then the previous one
                    if len(pdf_info_dic["title"].split()) > len(title[0].split()):
                        title = [pdf_info_dic["title"]]
                if "author" in pdf_info_dic.keys():
                    # Split the output after each " and "
                    authors = pdf_info_dic["author"].split(" and ")
                if "year" in pdf_info_dic.keys():
                    year = [pdf_info_dic["year"]]
                if "journal" in pdf_info_dic.keys():
                    journal = [pdf_info_dic["journal"]]
        except Exception:
            print(f"The follwing doi raise an error: {doi[0]}")
        metadata_values: set = set([title[0], doi[0], authors[0], year[0], journal[0]])
        if "Nothing Found" in metadata_values:
            try:
                pdf_info = subprocess.check_output(
                    ["title2bib", title[0]], text=True, input="y"
                )
                with open("bibtext.bib", "w") as bibfile:
                    bibfile.write(pdf_info)
                with open("bibtext.bib") as bibfile:
                    pdf_info_bib = bibtexparser.load(bibfile)
                os.remove("bibtext.bib")
                if len(pdf_info_bib.entries) > 0:
                    pdf_info_dic = pdf_info_bib.entries[0]
                    if doi == ["Nothing Found"] and "doi" in pdf_info_dic.keys():
                        doi = [pdf_info_dic["doi"]]
                    if authors == ["Nothing Found"] and "author" in pdf_info_dic.keys():
                        authors = pdf_info_dic["author"].split(" and ")
                    if year == ["Nothing Found"] and "year" in pdf_info_dic.keys():
                        year = [pdf_info_dic["year"]]
                    if (
                        journal == ["Nothing Found"]
                        and "journal" in pdf_info_dic.keys()
                    ):
                        journal = [pdf_info_dic["journal"]]
            except Exception:
                pass
        self.title = title
        self.doi = doi
        self.authors = authors
        self.keywords = keywords
        self.year = year
        self.journal = journal

    def update(self):
        extension = os.path.splitext(self.file_path)[1]
        if extension == ".pdf":
            self.get_api()
        elif extension == ".cermxml":
            self.get_cerm()
        else:
            raise AttributeError(
                "The file provided is not valid, choose a file in .pdf or .cermxml format"
            )
