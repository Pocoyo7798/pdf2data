import json
import os
import shutil
import subprocess
from typing import Any, Dict, List

from bs4 import BeautifulSoup as bs
from pydantic import BaseModel
import ast

from pdf2data.support import find_authors_cerm, find_term_in_list


class References(BaseModel):
    file_path: str
    output_folder: str

    def anystyle_pdf_references(self) -> None:
        """Generate a Json file containing a reference froma pdf file"""
        # Runs the commando to use Anystyle in the command prompt
        result = subprocess.run(
            [
                "anystyle",
                "-f",
                "json",
                "parse",
                self.file_path,
            ],
            capture_output=True,
            text=True,
        )
        """with open(os.path.join(self.output_folder, "anystyle_output.txt"), "w") as f:
            f.write(result.stdout)"""
        result_list = ast.literal_eval(result.stdout.replace('null', 'None').replace('true', 'True').replace('false', 'False'))
        return result_list


    def generate_reference_list(self) -> None:
        """Generate a reference json file from a .cermxml or .pdf file

        Raises
        ------
        AttributeError
            If the file provided is not a .pdf or .cermxml file
        """
        extension = os.path.splitext(self.file_path)[1]
        if extension == ".txt":
            results = self.anystyle_pdf_references()
        else:
            raise AttributeError(
                "The file provided is not valid, choose a file in .pdf or .cermxml format"
            )
        return results