import os
import json
from typing import Any, Dict, List, Optional
import re
from trieregex import TrieRegEx as TRE
import numpy as np

from pydantic import BaseModel, PrivateAttr

class BlockFinder(BaseModel):
    keywords_file_path: str
    generic_keywords_file_path: Optional[str] = None
    _regex: Optional[re.Pattern] = PrivateAttr(default=None)
    _generic_regex: Optional[re.Pattern] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        extension = os.path.splitext(self.keywords_file_path)[1]
        if extension != ".txt":
            raise ValueError("The keywords file should be in .txt format")
        with open (self.keywords_file_path, "r") as f:
            keywords_list: List[str] = f.readlines()
        clean_keywords_list: List[str] = []
        for keyword in keywords_list:
            clean_keywords_list.append(keyword.strip())
        tre: TRE = TRE(*clean_keywords_list)
        self._regex = re.compile(f"\\b{tre.regex()}\\b(?!-)", re.IGNORECASE | re.MULTILINE)
        print(self._regex)
        if self.generic_keywords_file_path is not None:
            generic_extension = os.path.splitext(self.generic_keywords_file_path)[1]
            if generic_extension != ".txt":
                raise ValueError("The generic keywords file should be in .txt format")
            with open (self.generic_keywords_file_path, "r") as f:
                generic_keywords_list: List[str] = f.readlines()
            clean_generic_keywords_list: List[str] = []
            for keyword in generic_keywords_list:
                clean_generic_keywords_list.append(keyword.strip())
            generic_tre: TRE = TRE(*clean_generic_keywords_list)
            self._generic_regex = re.compile(f"\\b{generic_tre.regex()}\\b(?!-)", re.IGNORECASE | re.MULTILINE)
    
    def find(self, blocks_file_path: str, tables: bool=True, figures: bool=False) -> Dict[str, Any]:
        """find blocks with specific keywords

        Parameters
        ----------
        blocks_file_path : str
            path to the file containing all blocks
        tables : bool, optional
            True to look for tables, False otherwise, by default True
        figures : bool, optional
            True to look for figures, False Otherwise, by default False

        Returns
        -------
        Dict[str, Any]
            A list of the blocks found
        """
        blocks_list: List[str] = []
        if tables is True:
            blocks_list.append("Table")
        if figures is True:
            blocks_list.append("Figure")
        blocks_set: set = set(blocks_list)
        with open(blocks_file_path, "r") as f:
            blocks_dict: Dict[str, Any] = json.load(f)
        doi: str = blocks_dict["doi"]
        blocks: List[Dict[str, Any]] = blocks_dict["blocks"]
        blocks_found: List[Dict[str, Any]] = []
        for block in blocks:
            keywords_found: List[str] = []
            if block["type"] in blocks_set:
                keywords_found = self._regex.findall(block["legend"])
            if len(keywords_found) > 0:
                blocks_found.append(block)
        if len(blocks_found) > 0 or self.generic_keywords_file_path is None:
            return {"blocks" : blocks_found, "doi": doi}
        for block in blocks:
            keywords_found: List[str] = []
            if block["type"] in blocks_set:
                keywords_found = self._generic_regex.findall(block["legend"])
            if len(keywords_found) > 0:
                blocks_found.append(block)
        return {"blocks" : blocks_found, "doi": doi}
    
class TextFinder(BaseModel):
    keywords_file_path: str
    _regex: Optional[re.Pattern] = PrivateAttr(default=None)
    _weights: Optional[Dict[str, int]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        extension = os.path.splitext(self.keywords_file_path)[1]
        if extension != ".json":
            raise ValueError("The keywords file should be in .json format")
        with open(self.keywords_file_path, "r") as f:
            self._weights: Dict[str, Any] = json.load(f)
        keywords_list = list(self._weights.keys())
        tre: TRE = TRE(*keywords_list)
        self._regex = re.compile(f"\\b{tre.regex()}\\b", re.IGNORECASE | re.MULTILINE)

    def find(self, text_file_path: str, word_count_threshold: int, paragraph: bool=True, section_header: bool=False, count_duplicates:bool = False) -> Dict[str, Any]:
        """finds all the text blocks with a word count over a threshold

        Parameters
        ----------
        text_file_path : str
            path to the file containing all text blocks
        word_count_threshold : int
            Minimum amount of word count to consider a text a prositve match
        paragraph : bool, optional
           True to look for paragraph type text, False otherwise, by default True
        section_header : bool, optional
            True to look for section_header type text, False otherwise, by default False

        Returns
        -------
        Dict[str, Any]
            A dictionary with all the text blocks found and the respective word count
        """
        with open(text_file_path, "r") as f:
            text_dict: Dict[str, Any] = json.load(f)
        text_list = text_dict["Text"]
        type_list: List[str] = []
        if paragraph is True:
            type_list.append("paragraph")
        if section_header is True:
            type_list.append("section_header")
        type_set: set = set(type_list)
        j = 0
        text_found: List[str] = []
        word_count_list: List[int] = []
        for text in text_list:
            keywords_found: List[str] = []
            word_count = 0
            if text_dict["Type"][j] in type_set:
                keywords_found = self._regex.findall(text)
            already_found: Dict[str, Any] = {}
            for keyword in keywords_found:
                if count_duplicates is False:
                    try:
                        already_found[keyword.lower()]
                    except KeyError:
                        word_count += self._weights[keyword.lower()]
                        already_found[keyword.lower()] = "exist"
                else:
                    word_count += self._weights[keyword.lower()]
            if word_count > word_count_threshold:
                print(keywords_found)
                text_found.append(text)
                word_count_list.append(word_count)
            j += 1
        if len(text_found) > 1:
            sort_array = np.argsort(word_count_list)
            final_counts: List[str] = []
            final_texts: List[str] = []
            for index in sort_array:
                final_counts.insert(0,word_count_list[index])
                final_texts.insert(0,text_found[index])
        else:
            final_counts: List[str] = word_count_list
            final_texts: List[str] = text_found
        return {"text": final_texts, "word_count": final_counts}



