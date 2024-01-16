import os
from typing import Any, Dict, List

from bs4 import BeautifulSoup as bs


def find_term_in_list(reference: List[str], term: str) -> List[str]:
    """_summary_

    Parameters
    ----------
    reference : List[str]
        List containing info about a reference
    term : str
        Term to be found in the reference list

    Returns
    -------
    List[str]
        List containing the terms found
    """
    all_term: List[str] = reference.find_all(term)
    if len(all_term) > 0:
        term_list: List[str] = []
        for term in all_term:
            volume_text = term.text
            term_list.append(volume_text)
        return term_list
    else:
        return ["Nothing Found"]


def list_into_bs_format(list: List[str], file_format: str = "lxml") -> Any:
    """_summary_

    Parameters
    ----------
    list : List[str]
        List of strings to be transformed
    file_format : str, optional
       format type of the output, by default "lxml"

    Returns
    -------
    Any
        string in the specfied format
    """
    string_object: str = "".join(str(data) for data in list)
    bs_list: Any = bs(string_object, file_format)
    return bs_list


def get_doc_list(folder_path: str, file_format: str) -> List[str]:
    """_summary_

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the desired files
    file_format : str
        file format to find

    Returns
    -------
    List[str]
        A list containing all files with the desired file format inside the folder
    """
    all_docs = []
    # Go through every file inside the desired file_path
    for entry in os.listdir(folder_path):
        # Verifies the final extension of the file
        if entry.endswith(file_format):
            all_docs = all_docs + [entry]
    return all_docs


def find_authors_cerm(reference: List[str]) -> List[Dict[str, str]]:
    """Get the list of authors of a reference

    Parameters
    ----------
    reference : List[str]
        List containing all the information about the reference

    Returns
    -------
    List[Dict[str, str]]
        List of dcitionaries containing the name and surname of the authors inside a reference
    """
    # find all authors tags
    all_authors: List[str] = reference.find_all("string-name")
    all_authors_list: List[Dict[str, str]] = []
    total_authors: int = len(all_authors)
    # counts authors without name and surname tags
    missed_author: int = 0
    # counts authors with only name or surname tags
    incomplete_author: int = 0
    for author in all_authors:
        # find the author surname tag
        surname_string: str = author.find("surname")
        # find the author given-names tag
        name_string: str = author.find("given-names")
        # test if both surname and name was found
        if surname_string is not None and name_string is not None:
            # Get the text from the full string
            name: str = name_string.text
            surname: str = surname_string.text
            author_dic: Dict[str, str] = {"given": name, "family": surname}
            all_authors_list.append(author_dic)
        # test if there is no name
        elif name_string is None:
            surname = surname_string.text
            author_dic = {"given": "", "family": surname}
            all_authors_list.append(author_dic)
            incomplete_author = incomplete_author + 1
        # test if there is no surname
        elif surname_string is None:
            name = name_string.text
            author_dic = {"given": name, "family": ""}
            all_authors_list.append(author_dic)
            incomplete_author = incomplete_author + 1
        else:
            missed_author = missed_author + 1
    print(
        "From {} entries, {} were not retrieved and {} were parcially retrieved".format(
            total_authors, missed_author, incomplete_author
        )
    )
    return all_authors_list
