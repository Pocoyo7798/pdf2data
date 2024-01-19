import os
from typing import Any, Dict, List
import numpy as np

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

def iou(box_1: List[float], box_2: List[float]) -> float:
    """Calculate the ipou valçue between 2 boxes

    Parameters
    ----------
    box_1 : List[float]
        coordinates of box 1
    box_2 : List[float]
        coordinates of box 2

    Returns
    -------
    float
        the iou value
    """
    x_1: float = max(box_1[0], box_2[0])
    y_1: float = max(box_1[1], box_2[1])
    x_2: float = min(box_1[2], box_2[2])
    y_2: float = min(box_1[3], box_2[3])
    # Determine the value of the box interseption
    interseption: float = abs(max((x_2 - x_1), 0)) * max((y_2 - y_1), 0)
    if interseption == 0:
        return 0
    # Determine the boxes are
    box_1_area: float = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area: float = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
    # Return the iou
    return interseption / float(box_1_area + box_2_area - interseption)


def order_horizontal(box_rows: list, output_type='box_list') -> List[Any]:
    """Order a list of boxes horizontally

    Parameters
    ----------
    box_rows : list
        List of boxes to consider
    output_type : str, optional
        'box_list' for the organized boxes values, "argument_list" for the list of organized indexes, by default 'box_list'

    Returns
    -------
    List[Any]
        A list of horizontally organized boxes or indexes
    """
    y1_list: List[List[float]] = []
    for box in box_rows:
        y1_list.append(box[1])
    order_box_index: List[int] = np.argsort(y1_list)
    if output_type == 'argument_list':
        return order_box_index
    new_box_rows = []
    for index in order_box_index:
        new_box_rows.append(box_rows[index])
    return new_box_rows

def block_organizer(box_list: List[List[float]], page_coords: List[float], displacement_factor: float=0.9) -> List[int]:
    """Organize all block in a page taken into account a two collumn format

    Parameters
    ----------
    box_list : List[List[float]]
        list of all block coordinates present in the page
    page_coords : List[float]
        page dimensions
    displacement_factor : float, optional
        factor to move the center of the page to the left, by default 0.9

    Returns
    -------
    List[int]
        List contain the indexes of the organized blocks
    """
    page_midle: float = float(page_coords[0] + page_coords[2]) / 2 * displacement_factor
    index: int = 0
    index_list_1: List[int] = []
    index_list_2: List[int] = []
    for box in box_list:
        if box[0] < page_midle:
            index_list_1.append(index)
        else:
            index_list_2.append(index)
        index = index + 1
    index_list: List[int] = index_list_1 + index_list_2
    return index_list
