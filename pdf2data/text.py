import subprocess
import sys
import shutil
import fitz
import os
from pdfminer.converter import TextConverter, XMLConverter, HTMLConverter
from io import BytesIO
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from typing import Container
from bs4 import BeautifulSoup as bs
import json
import PyPDF2
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from pdf2data.support import get_doc_list, remove_pdf_images, convert_pdfminersix, verify_string_in_list, get_string_from_box

class TextFileGenerator(BaseModel):
    input_folder: str
    output_folder: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.output_folder is None:
            self.output_folder = self.input_folder
        doc_list = get_doc_list(self.input_folder, ".pdf")
        if len(doc_list) == 0:
            raise AttributeError("The input folder do not conatin any pdf file")
    
    def pdf_to_cermxml(self) -> None:
        """Generate .cermxml files from pdf files
        """
        # runs the Cermine command code to obtain the CERXML file
        try:
            subprocess.run(
                [
                    "java",
                    "-cp",
                    "cermine-impl-1.13-jar-with-dependencies.jar",
                    "pl.edu.icm.cermine.ContentExtractor",
                    "-path",
                    self.input_folder,
                    "-outputs",
                    "jats",
                ]
            )
        except:
            subprocess.run(["wget", "path_to_cermine_file"])
            subprocess.run(
                [
                    "java",
                    "-cp",
                    "cermine-impl-1.13-jar-with-dependencies.jar",
                    "pl.edu.icm.cermine.ContentExtractor",
                    "-path",
                    self.input_folder,
                    "-outputs",
                    "jats",
                ]
            )
        converted_doc_list: List[str] = get_doc_list(self.input_folder, ".cermxml")
        # move the final file to the output paste
        for document in converted_doc_list:
            shutil.move(self.input_folder + "/" + document, self.output_folder + "/" + document)
    
    def pdf_to_muhtml(self) -> None:
        """Generate .html files from pdf files in pymupdf html format
        """
        initial_doc_list: List[str] = get_doc_list(self.input_folder,".pdf")
        for doc in initial_doc_list:
            # remove the file type extension
            document_name: str = os.path.splitext(doc)[0]
            document: Any = fitz.open(self.input_folder + "/" + doc)
            document_no_image: Any = remove_pdf_images(document)
            # create an HTML file in writting mode
            with open(f"{document_name}.html", "w") as html:
                for page in document_no_image:
                    text = page.get_text("html")
                    html.write(text)
            document.close()
            # move the final file to the output paste
            shutil.move(
                document_name + ".html", self.output_folder + "/" + document_name +
                ".html")

    def pdf_to_miner(self, file_format: str) -> None:
        """Generate a .txt, .html or .xml file from a pdf using pdf minex six

        Parameters
        ----------
        file_format : str
            desired file format

        Raises
        ------
        AttributeError
            if the wanted format is not .txt, .html or .xml 
        """
        valid_formats: set = set(['txt', 'html', 'xml'])
        if file_format not in valid_formats:
            raise AttributeError(f'the file format is not valid. Only file format in this list are valid:{valid_formats}')
        initial_doc_list: List[str] = get_doc_list(self.input_folder,".pdf")
        for doc in initial_doc_list:
            document_name: str = os.path.splitext(doc)[0]
            if file_format == 'txt':
                text: str = convert_pdfminersix(self.input_folder + "/" + doc)
            else:
                text = convert_pdfminersix(self.input_folder + "/" + doc, format = file_format)
            # create a file in writting mode
            with open(f"{document_name}.{file_format}", "w") as file:
                file.write(text)
            # move the final file to the output paste
            shutil.move(document_name + "." + file_format, self.output_folder + "/" + document_name
                    + "." + file_format)


class TextExtractor(BaseModel):
    input_file: str
    output_folder: str = None


    def extract_cermine(self, file_name: str) -> Dict[str, List[str]]:
        """Create a Json file with the paragraphs from a cermxml file

        Parameters
        ----------
        file_name : str
            name of the file created

        Returns
        -------
        Dict[str, List[str]]
            A dictionary containing all text blocks and their types
        Raises
        ------
        AttributeError
            if the file is not in cermxml format
        """
        extension: str = os.path.splitext(self.input_file)[1]
        if extension != ".cermxml":
            raise AttributeError("This method only works for .cermxml files")
        with open(self.input_file, 'r') as f:
            text_lines: List[str] = list(f)
        extract: bool = False
        block_dict: Dict[str, List[str]] = {}
        text_list_final: List[str] = []
        type_list_final: List[str] = []
        for line in text_lines:
            if '<title>' in line:
                extract = True
                block: str = line
                type_list_final.append('section_header')
            elif '<p>' in line:
                extract = True
                block = line
                type_list_final.append('paragraph')
            elif '<xref ref-type="bibr"' in line:
                bs_ref: Any = bs(line, features='lxml')
                ref: str = bs_ref.getText()
                ref_string: str = f'%ref%({ref})'
                ref_string = ref_string.replace('\n', '')
                block = block + ref_string
            elif extract is True:
                block = block + line
            if '</title>' in line:
                extract = False
                bs_block = bs(block, features='lxml')
                text: str = bs_block.getText()
                text = text.replace('\n\n', ' ')
                text = text.replace('\n', ' ')
                text = text.replace('-  ', '')
                text_list_final.append(text)
            elif '</p>' in line:
                extract = False
                bs_block = bs(block, features='lxml')
                text = bs_block.getText()
                text = text.replace('\n\n', ' ')
                text = text.replace('\n', ' ')
                text = text.replace('-  ', '')
                text = text.replace('         ', '')
                text = text.replace('        ', '')
                text_list_final.append(text)
        block_dict['Text'] = text_list_final
        block_dict['Type'] = type_list_final
        with open(f"{self.output_folder}/{file_name}.json", "w") as j:
            # convert the dictionary into a json variable
            json_text = json.dumps(block_dict, indent=4)
            j.write(json_text)
        return block_dict
    
    def extract_txt(self, file_name: str, strings_to_remove: List[str]=[]) -> Dict[str, List[str]]:
        """Create a Json file with the paragraphs from a txt file
        Parameters
        ----------
        file_name : str
            name of the file created
        strings_to_remove : List[str], optional
            list of string to be removed from the final output, by default []

        Returns
        -------
        Dict[str, List[str]]
            A dictionary containing all text blocks and their types

        Raises
        ------
        AttributeError
            if the file is not in txt format
        """
        extension: str = os.path.splitext(self.input_file)[1]
        if extension != ".txt":
            raise AttributeError("This method only works for .txt files")
        with open(self.input_file, 'r') as f:
            text_lines: List[str] = f.read()
        text_list = text_lines.split("\n\n")
        new_text_list: List[str] = []
        type_list: List[str] = []
        block_dict: Dict[str, List[str]] = {}
        i = 0
        reference_list: List[str] = ['References', 'Reference', 'Bibliography']
        for line in text_list:
            if verify_string_in_list(line, strings_to_remove) is True:
                del text_list[i]
            else:
                line: str = line.replace('\n', '')
                line = line.replace('\t', '')
                line = line.replace('\f', '')
                line = line.replace('  ', '')
                new_text_list.append(line)
                type_list.append('paragraph')
                i = i + 1
            if line in reference_list:
                break
        block_dict['Text'] = new_text_list
        block_dict['Type'] = type_list
        with open(f"{self.output_folder}/{file_name}.json", "w") as j:
            # convert the dictionary into a json variable
            json_text = json.dumps(block_dict, indent=4)
            j.write(json_text)
        return block_dict
    
    def extract_layoutparser(self, file_name: str, layout: Dict[str, Any], strings_to_remove: List[str]=[]) -> Dict[str, List[Any]]:
        """Create a Json file with the paragraphs from a pdf file

        Parameters
        ----------
        file_name : str
            name of the file created
        layout : Dict[str, Any]
            dictionary containing the layout information
        strings_to_remove : List[str], optional
            list of string to be removed from the final output, by default []

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary containing all text blocks, their types and coordinates
        """
        pdf = PyPDF2.PdfReader(self.input_file)
        document = fitz.open(self.input_file)
        check = ['Title', 'Text']
        reference_list = ['References', 'Reference', 'Bibliography']
        page_index: int = 0
        block_dict: Dict[str, List[Any]] = {}
        text_list: List[str] = []
        type_list: List[str] = []
        coords_list: List[List[float]] = []
        for page in document:
            pdf_page: Any = pdf.pages[page_index]
            page_size: Any = pdf_page.mediabox
            boxes: List[float] = layout['boxes'][page_index]
            types: List[str] = layout['types'][page_index]
            j = 0
            for box in boxes:
                test: bool = False
                text: str = ''
                if types[j] in check:
                    text = get_string_from_box(page, box, page_size)
                    test = verify_string_in_list(text, strings_to_remove)
                if types[j] == 'Title' and text != '' and test is False:
                    text_list.append(text)
                    coords_list.append(box)
                    type_list.append('section_header')
                elif types[j] == 'Text' and text != '' and test is False:
                    text_list.append(text)
                    coords_list.append(box)
                    type_list.append('paragraph')
                if text in reference_list:
                    break
                j = j + 1
            page_index = page_index + 1
        block_dict['Text'] = text_list
        block_dict['Type'] = type_list
        block_dict['Coordinates'] = coords_list
        with open(f"{self.output_folder}/{file_name}.json", "w") as j:
            # Convert the dictionary into a json variable
            json_text = json.dumps(block_dict, indent=4)
            j.write(json_text)
        return block_dict
        
