import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

from bs4 import BeautifulSoup as bs
from pydantic import BaseModel

from pdf2data.support import find_authors_cerm, find_term_in_list


class References(BaseModel):
    file_path: str
    output_folder: str

    def anystyle_pdf_references(self) -> None:
        """Generate a Json file containing a reference froma pdf file"""
        # Runs the commando to use Anystyle in the command prompt
        subprocess.run(
            [
                "anystyle",
                "-f",
                "json",
                "find",
                "--no-layout",
                self.file_path,
                self.output_folder,
            ]
        )

    def get_references_list_cerm(self) -> List[str]:
        """Get all references from a .cermxml file

        Returns
        -------
        List[str]
            List of all references found inside a .cermxml file
        """
        with open(self.file_path, "r") as file:
            # Read each line in the file, readlines() returns a list of lines
            content: List[str] = file.readlines()

        # Combine the lines in the list into a string
        content_str: str = "".join(content)
        # parse the XML file
        bs_content: Any = bs(content_str, "lxml")
        # Find all the references in the file
        all_references: List[StopIteration] = bs_content.find_all("ref")
        if len(all_references) > 0:
            return all_references
        else:
            print("No references have been found")
            return []

    def get_file_references_cerm(self) -> List[Dict[str, Any]]:
        """Get all references inside a .cermxml file

        Returns
        -------
        List[Dict[str, Any]]
            List containing all references as dictionaries
        """
        reference_list: List[str] = self.get_references_list_cerm()
        final_list: List[Dict[str, Any]] = []
        citation_number: int = 1
        for reference in reference_list:
            reference_dic: Dict[str, Any] = {}
            # add an entry to the dictionary
            reference_dic["citation-number"] = citation_number
            reference_dic["author"] = find_authors_cerm(reference)
            reference_dic["title"] = find_term_in_list(reference, "article-title")
            reference_dic["volume"] = find_term_in_list(reference, "volume")
            reference_dic["date"] = find_term_in_list(reference, "year")
            reference_dic["pages"] = find_term_in_list(reference, "fpage")
            reference_dic["container-title"] = find_term_in_list(reference, "source")
            citation_number = citation_number + 1
            # add the final dictionary to the final list
            final_list.append(reference_dic)
        return final_list

    def cermine_cermxml_references(self) -> None:
        """Generate a reference json file from a .cermxml file"""
        full_name: str = os.path.basename(self.file_path)
        file_name: str = os.path.splitext(full_name)[0]
        references_list: List[Dict[str, Any]] = self.get_file_references_cerm()
        if len(references_list) != 0:
            with open(f"{file_name}_references.json", "w") as j:
                # convert the dictionary into a json variable
                json_references = json.dumps(references_list, indent=4)
                j.write(json_references)
            shutil.move(
                file_name + ".json", self.output_folder + "/" + file_name + ".json"
            )

    def generate_reference_file(self) -> None:
        """Generate a reference json file from a .cermxml or .pdf file

        Raises
        ------
        AttributeError
            If the file provided is not a .pdf or .cermxml file
        """
        extension = os.path.splitext(self.file_path)[1]
        if extension == ".pdf":
            self.anystyle_pdf_references()
        elif extension == ".cermxml":
            self.cermine_cermxml_references()
        else:
            raise AttributeError(
                "The file provided is not valid, choose a file in .pdf or .cermxml format"
            )
