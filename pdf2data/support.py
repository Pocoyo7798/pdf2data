import os
from difflib import SequenceMatcher
from io import BytesIO
from typing import Any, Container, Dict, List
from Levenshtein import ratio

import fitz
import numpy as np
from bs4 import BeautifulSoup as bs
from pdfminer.converter import HTMLConverter, TextConverter, XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


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
    """Get a list of all documents with a specific file extension

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


def order_horizontal(box_rows: list, output_type="box_list") -> List[Any]:
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
    if output_type == "argument_list":
        return order_box_index
    new_box_rows = []
    for index in order_box_index:
        new_box_rows.append(box_rows[index])
    return new_box_rows


def block_organizer(
    box_list: List[List[float]],
    page_coords: List[float],
    displacement_factor: float = 0.9,
) -> List[int]:
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


def remove_page_images(document: Any, page: int) -> Any:
    """Remove images from a pdf page

    Parameters
    ----------
    document : Any
        pdf file to be processed
    page : int
        page number to be considered

    Returns
    -------
    Any
        the document without images in the specified page
    """
    image_list: Any = document.get_page_images(page)
    # get the code of all contents in the page
    content_list: Any = document[page].get_contents()
    for content in content_list:
        # generate the source stream of each content
        source_stream: Any = document.xref_stream(content)
        if source_stream is not None:
            for image in image_list:
                # generate the image code
                array: bytes = bytes(image[7], "utf-8")
                # try to find the image code in the content source stream
                test: Any = source_stream.find(array)
                # if test=-1, it means that content is not that specified image
                if test != -1:
                    # create a new blank source stream keeping the position
                    new_stream: Any = source_stream.replace(array, b"")
                    # replace the older source stream by the new one
                    document.update_stream(content, new_stream)
                    source_stream: Any = document.xref_stream(content)
    return document


def remove_pdf_images(document: Any) -> Any:
    """Remove images from a pdf document

    Parameters
    ----------
    document : Any
        document to be processed

    Returns
    -------
    Any
        document without any image
    """
    if len(document) > 0:
        for page in range(len(document) - 1):
            document = remove_page_images(document, page)
        return document
    else:
        print("The document is empty")


def convert_pdfminersix(
    path: str,
    format: str = "text",
    codec: str = "utf-8",
    password: str = "",
    maxpages: int = 0,
    caching: bool = True,
    pagenos: Container[int] = set(),
) -> str:
    # Script from https://gist.github.com/rguliev/3d886d38daa8ac0be8ddb85d645fb0bc
    """Summary
    Parameters
    ----------
    path : str
        Path to the pdf file
    format : str, optional
        Format of output, must be one of: "text", "html", "xml".
        By default, "text" format is used
    codec : str, optional
        Encoding. By default "utf-8" is used
    password : str, optional
        Password
    maxpages : int, optional
        Max number of pages to convert. By default is 0, i.e. reads all pages.
    caching : bool, optional
        Caching. By default is True
    pagenos : Container[int], optional
        Provide a list with numbers of pages to convert
    Returns
    -------
    str
        Converted pdf file
    """
    rsrcmgr = PDFResourceManager()
    retstr = BytesIO()
    laparams = LAParams()
    if format == "text":
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    elif format == "html":
        device = HTMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    elif format == "xml":
        device = XMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    else:
        raise ValueError("provide format, either text, html or xml!")
    fp = open(path, "rb")
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(
        fp,
        pagenos,
        maxpages=maxpages,
        password=password,
        caching=caching,
        check_extractable=True,
    ):
        interpreter.process_page(page)
    text = retstr.getvalue().decode()
    fp.close()
    device.close()
    retstr.close()
    return text


def verify_string_in_list(string: str, list_of_words: List[str]) -> bool:
    """Verify if a string exist inside a list

    Parameters
    ----------
    string : str
        string to be considered
    list_of_words : List[str]
        List of strings

    Returns
    -------
    bool
        True if the string exists, False otherwise
    """
    for entry in list_of_words:
        similarity: float = SequenceMatcher(None, entry, string).ratio()
        if similarity > 0.9:
            return True
    return False


def box_corretor(
    pdf_size: List[float],
    box: List[float],
    x_corrector: float = 0,
    y_corrector: float = 0,
) -> List[float]:
    """increase the box size depending on the page size

    Parameters
    ----------
    pdf_size : List[float]
        size of the pdf page size
    box : List[float]
        inital box coordinates
    x_corrector : float, optional
        corrector of the x axis, by default 0
    y_corrector : float, optional
        corrector of the y axis, by default 0

    Returns
    -------
    List[float]
        a list with the corrected coordinates
    """
    x_1: float = max(
        pdf_size[0],
        int(float(box[0]) - x_corrector * (float(pdf_size[2] - pdf_size[0]))),
    )
    y_1: float = max(
        pdf_size[1],
        int(float(box[1]) - y_corrector * (float(pdf_size[3] - pdf_size[1]))),
    )
    x_2: float = min(
        pdf_size[2],
        int(float(box[2]) + x_corrector * (float(pdf_size[2] - pdf_size[0]))),
    )
    y_2: float = min(
        pdf_size[3],
        int(float(box[3]) + y_corrector * (float(pdf_size[3] - pdf_size[1]))),
    )
    return x_1, y_1, x_2, y_2


def get_string_from_box(
    page: Any,
    box_coords: List[float],
    page_size: List[float],
    x_corrector_value: float = 0.01,
    y_corrector_value: float = 0.002,
) -> str:
    """retrieve the text inside a box

    Parameters
    ----------
    page : Any
        pdf page
    box_coords : List[float]
        box coordinates
    page_size : List[float]
        size of the pdf size
    x_corrector_value : float, optional
        corrector of the x axis, by default 0.01
    y_corrector_value : float, optional
        corrector of the y axis, by default 0.005

    Returns
    -------
    str
        the text string isnde the box
    """
    # Correct the tablle coordinates acording the the page size
    x_1, y_1, x_2, y_2 = box_corretor(
        page_size,
        box_coords,
        x_corrector=x_corrector_value,
        y_corrector=y_corrector_value,
    )
    table_rect: Any = fitz.Rect(x_1, y_1, x_2, y_2)
    # Retrive the text inside the box
    text: str = page.get_text(
        clip=table_rect,
    )
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("- ", " ")
    text = text.replace("\u2010 ", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace(" -", "")
    text = text.replace("  ", " ")
    text = text.replace("   ", " ")
    text = text.replace("    ", " ")
    return text


def sobreposition(box_1: List[float], box_2: List[float]) -> float:
    """Calculate the sobreposition of 2 boxes

    Parameters
    ----------
    box_1 : List[float]
        coordinates of box 1
    box_2 : List[float]
        coordinates of box 2

    Returns
    -------
    float
        the sobreposition amount of the two boxes
    """
    x_1: float = max(box_1[0], box_2[0])
    y_1: float = max(box_1[1], box_2[1])
    x_2: float = min(box_1[2], box_2[2])
    y_2: float = min(box_1[3], box_2[3])
    interseption: float = abs(max((x_2 - x_1), 0)) * max((y_2 - y_1), 0)
    return interseption / ((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))


def order_vertical(box_collumns: List[List[float]]) -> List[List[float]]:
    """order a list of vertical boxes

    Parameters
    ----------
    box_collumns : List[List[float]]
        List of boxes coordinates

    Returns
    -------
    List[List[float]]
        An ordered list of boxes coordinates
    """
    x1_list: List[float] = []
    for box in box_collumns:
        x1_list.append(box[0])
    order_box_index: List[int] = np.argsort(x1_list)
    new_box_collumns: List[List[float]] = []
    for index in order_box_index:
        new_box_collumns.append(box_collumns[index])
    return new_box_collumns


def draw_boxes(
    file_path: str, output_file: str, box_list: List[List[float]], page: int, zoom=2
) -> None:
    """Generate an image from a page with boxes

    Parameters
    ----------
    file_path : str
        path to the pdf file
    output_file : str
       name of the output image file
    box_list : List[float]
        list of boxes to draw
    page : int
        page number
    zoom : int, optional
        zoom for the final image, by default 2
    """
    document: Any = fitz.open(file_path)
    page_index: int = page - 1
    page: Any = document[page_index]
    for box in box_list:
        # Create a rectangle object
        rect: fitz.Rect = fitz.Rect(box[0], box[1], box[2], box[3])
        # Draw the rectangle in the PDF
        page.draw_rect(rect, color=[0, 1, 1, 0], overlay=True, width=2, fill_opacity=1)
    mat: fitz.Matrix = fitz.Matrix(zoom, zoom)
    # Create a page image
    pix: Any = page.get_pixmap(matrix=mat)
    pix.save(output_file)


def words_from_line(
    line: Dict[str, Any], threshold: float, page_area: float
) -> Dict[str, List[Any]]:
    """Get all the words inside a line

    Parameters
    ----------
    line : Dict[str, Any]
        Dict containing all words in a line
    threshold : float
        maximum distance between two words to consider superscript ou subscript
    page_area : float
        area of the page

    Returns
    -------
    Dict[str, List[Any]]
        a dcitionary with all the words and respective boxes inside a line
    """
    words_list: List[str] = []
    box_list: List[List[float]] = []
    if len(line["spans"]) <= 0:
        return {"words": words_list, "boxes": box_list}
    text: List[str] = [line["spans"][0]["text"]]
    box: List[float] = line["spans"][0]["bbox"]
    for span in line["spans"][1:]:
        new_text: str = span["text"]
        new_box: List[float] = span["bbox"]
        if box[0] < new_box[0] < box[2]:
            distance: float = 0
        else:
            distance = abs(new_box[0] - box[2])
        if distance < threshold:
            text.append(new_text)
            box = [
                min(box[0], new_box[0]),
                min(box[1], new_box[1]),
                max(box[2], new_box[2]),
                max(box[3], new_box[3]),
            ]
        else:
            final_text: str = "".join(text)
            final_text = final_text.replace("\u00a0", "")
            words_list.append(final_text)
            box_list.append(box)
            text = [new_text]
            box = new_box
    final_text = "".join(text)
    final_text = final_text.replace("\u00a0", "")
    words_list.append(final_text)
    box_list.append(box)
    j: int = 0
    for box in box_list:
        box_area: float = abs((box[2] - box[0]) * (box[3] - box[1]))
        if box_area > page_area or box_area == 0:
            del box_list[j]
            del words_list[j]
        else:
            j += 1
    return {"words": words_list, "boxes": box_list}


def iou_vert(box_1: List[float], box_2: List[float]) -> float:
    """Calculate the iou of two vertical lines from 2 boxes

    Parameters
    ----------
    box_1 : List[float]
        coordinates of box 1
    box_2 : List[float]
        coordinates of box 2

    Returns
    -------
    float
        the iou value of the vertical lines from both boxes
    """
    # Determine the box sobreposition
    interseption: float = max(min(box_1[3], box_2[3]) - max(box_1[1], box_2[1]), 0)
    # Determine the boxes width
    box_1_size: float = box_1[3] - box_1[1]
    box_2_size: float = box_2[3] - box_2[1]
    return interseption / float(box_1_size + box_2_size - interseption)


def word_horiz_box_corrector(
    words_list: List[str], box_list: List[List[float]], table_coords: List[float]
) -> Dict[str, List[Any]]:
    """merges words close together

    Parameters
    ----------
    words_list : List[str]
        list of words to be considered
    box_list : List[List[float]]
        list of coordinates of each box
    table_coords : List[float]
        table coordinates

    Returns
    -------
    Dict[str, List[Any]]
        a dictionary containing the merged words, their respective coordinates and the table size
    """
    j: int = 0
    while j < len(words_list) - 1:
        i: int = j + 1
        while i < len(words_list):
            if (
                box_list[j][0] < box_list[i][0] < box_list[j][2]
                and iou_vert(box_list[j], box_list[i]) > 0.2
            ):
                new_word: str = "".join([words_list[j], words_list[i]])
                new_coords: List[float] = [
                    min(box_list[j][0], box_list[i][0]),
                    min(box_list[j][1], box_list[i][1]),
                    max(box_list[j][2], box_list[i][2]),
                    max(box_list[j][3], box_list[i][3]),
                ]
                words_list.remove(words_list[i])
                box_list.remove(box_list[i])
                words_list[j] = new_word
                box_list[j] = new_coords
            else:
                i = i + 1
        j = j + 1
    return {"words": words_list, "boxes": box_list, "table_box": table_coords}


def iou_horiz(box_1, box_2) -> float:
    """caluclates the iou of 2 boxes considering only their width

    Parameters
    ----------
    box_1 : _type_
        coordinates of box 1
    box_2 : _type_
        coordinates of box 2

    Returns
    -------
    float
        iou of the 2 boxes considering only the width
    """
    # Determine the box sobreposition
    interseption: float = min(box_1[2], box_2[2]) - max(box_1[0], box_2[0])
    # Determine the boxes width
    box_1_size: float = box_1[2] - box_1[0]
    box_2_size: float = box_2[2] - box_2[0]
    return interseption / float(box_1_size + box_2_size - interseption)


def find_legend(
    page: Any,
    page_size: List[float],
    boxes: List[List[float]],
    types: List[str],
    index: int,
    type: str = "Table",
    block_distance: int = 6,
    iou_value: float = 0.02,
) -> str:
    """retrieve the legend of tables of figures by searching for text close to them

    Parameters
    ----------
    page : Any
        pdf page to be considered
    page_size : List[float]
        size of the oage
    boxes : List[List[float]]
        list of boxes of the page layout
    types : List[str]
        list of the types of the page layout
    index : int
        position of the table or figure in the layout list
    type : str, optional
        type of the block. It can be 'Table' or 'Figure', by default 'Table'
    block_distance : int, optional
        maximum distance between the block and the text legend, by default 6
    iou_value : float, optional
       iou threshold to identify duplicated blocks, by default 0.02

    Returns
    -------
    str
        the legend associated with a block

    Raises
    ------
    AssertionError
        if the type is not 'Table' or 'Figure'
    """
    if type not in ["Table", "Figure"]:
        raise AssertionError("The block type is not valid")
    verification: bool = False
    extra_verification: bool = False
    table_names: List[str] = ["Table", "TABLE", "Table.", "TABLE."]
    figure_names: List[str] = [
        "Image",
        "Figure",
        "Fig.",
        "Fig",
        "Figure.",
        "Scheme",
        "Scheme.",
        "FIG",
        "FIG.",
        "FIGURE",
        "FIGURE.",
        "SCHEME",
        "SCHEME.",
    ]
    legend: str = ""
    entries_list: List[str] = []
    if type == "Table":
        j: int = index - 1
        while j >= max(0, index - block_distance) and verification is False:
            # Verify if the it is a text block and the is enough sobreposition
            if (
                types[j] in ["Text", "Title", "Table Caption"]
                and iou_horiz(boxes[index], boxes[j]) > iou_value
            ):
                entry: str = get_string_from_box(page, boxes[j], page_size)
                # insert in the beginning of the list
                entries_list.insert(0, entry)
                entry_list = entry.split()
                if len(entry_list) > 0:
                    # verify if the first word is 'Table'
                    if entry_list[0] in table_names:
                        verification = True
                        legend = "".join(entries_list)
                        legend = legend.replace("\n", " ")
            elif types[j] in ["Table", "Figure"] and iou(boxes[index], boxes[j]) > 0.3:
                print("Probably there is a duplicated Table")
            elif (
                types[j] in ["Table", "Figure"]
                and iou_horiz(boxes[index], boxes[j]) > iou_value
            ):
                break
            j = j - 1
        j = 0
        entries_list = []
        while j < len(boxes) - 1 and verification is False:
            if (
                types[j] in ["Text", "Title",  "Table Caption"]
                and iou_vert(boxes[index], boxes[j]) > iou_value
            ):
                entry = get_string_from_box(page, boxes[j], page_size)
                # insert in the beginning of the list
                entry_list = entry.split()
                if extra_verification is True:
                    entries_list.append(entry)
                    verification = True
                    legend = "".join(entries_list)
                    legend = legend.replace("\n", " ")
                elif len(entry_list) > 0:
                    # verify if the first word is 'Table'
                    if entry_list[0] in table_names and len(entry_list) < 5:
                        entries_list.append(entry)
                        extra_verification = True
                    elif entry_list[0] in table_names:
                        entries_list.append(entry)
                        verification = True
                        legend = "".join(entries_list)
                        legend = legend.replace("\n", " ")
            j = j + 1
    elif type == "Figure":
        j = index + 1
        while (
            j <= min(len(boxes) - 1, index + block_distance) and verification is False
        ):
            # print(types[j])
            if (
                types[j] in ["Text", "Title", "Figure Caption"]
                and iou_horiz(boxes[index], boxes[j]) > iou_value
            ):
                entry = get_string_from_box(page, boxes[j], page_size)
                # print(f'full entry: {entry}')
                entries_list.insert(0, entry)
                entry_list = entry.split()
                if len(entry_list) > 0:
                    # print(entry_list[0])
                    if entry_list[0] in figure_names:
                        verification = True
                        legend = "".join(entries_list)
                        legend = legend.replace("\n", " ")
            elif types[j] in ["Table", "Figure"] and iou(boxes[index], boxes[j]) > 0.3:
                print("Probably there is a duplicated Figure")
            elif (
                types[j] in ["Text", "Title", "Figure Caption"]
                and iou_horiz(boxes[index], boxes[j]) > iou_value
            ):
                break
            j = j + 1
        j = 0
        entries_list = []
        while j < len(boxes) - 1 and verification is False:
            # print('Cheguei!!!!!!!!')
            # print(types[j])
            # print(boxes[j])
            if (
                types[j] in ["Text", "Title"]
                and iou_vert(boxes[index], boxes[j]) > iou_value
            ):
                entry = get_string_from_box(page, boxes[j], page_size)
                # print(f'full entry is {entry}')
                # insert in the beginning of the list
                entry_list = entry.split()
                if extra_verification is True:
                    entries_list.append(entry)
                    verification = True
                    legend = "".join(entries_list)
                    legend = legend.replace("\n", " ")
                elif len(entry_list) > 0:
                    # verify if the first word is 'Table'
                    # print(entry_list[0])
                    # print(len(entry_list))
                    if entry_list[0] in figure_names and len(entry_list) < 5:
                        entries_list.append(entry)
                        extra_verification = True
                    elif entry_list[0] in figure_names:
                        print("Cheguei2!!!!!!")
                        entries_list.append(entry)
                        verification = True
                        legend = "".join(entries_list)
                        legend = legend.replace("\n", " ")
            j = j + 1
    return legend


def calc_metrics(true_positives: int, false_positives: int, false_negatives: int) -> Dict[str, float]:
    """Calculate de precision, recall and f-score

    Parameters
    ----------
    true_positives : int
       number of true positives
    false_positives : int
        number of false positives
    false_negatives : int
        number of false negatives

    Returns
    -------
    Dict[str, float]
        A dicionary containing the precision, recall and f-score
    """
    if true_positives == 0:
        precision: float = 0
        recall: float = 0
        f_score: float = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f_score = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f_score': f_score}

def verify_boxes(box: List[float], page: int, box_list: List[List[float]], page_list:List[int], iou_value: float=0.85, get_index: bool=False) -> Any:
    """Verify if a box exists inside a list of boxes

    Parameters
    ----------
    box : List[float]
        box coordinates
    page : int
        box page number
    box_list : List[List[float]]
        list of boxes to consider
    page List : List[int]
        list of boxes page list
    iou_value : float, optional
        iou threshold to consider that two boxes are the same, by default 0.85
    get_index : bool, optional
        True to get the box index in the list, False otherwise, by default False

    Returns
    -------
    Any
        True if the box exists inside the list and False otherwise, giving the box index optionally
    """
    index: int = 0
    if len(page_list) > 0:
        for coords in box_list:
            box_iou: float = iou(box, coords)
            ref_page = page_list[index]
            if ref_page > page:
                break
            if box_iou > iou_value and get_index is True and ref_page == page:
                return True, index
            elif box_iou > iou_value and ref_page == page:
                return True
            index = index + 1
    else:
        for coords in box_list:
            box_iou: float = iou(box, coords)
            if box_iou > iou_value and get_index is True:
                return True, index
            elif box_iou > iou_value:
                return True
            index = index + 1
    if get_index is True:
        return False, None
    else:
        return False
    
def verify_string(ref_string: str, string: str, threshold: float=0.8) -> bool:
    """Verify if two strings are similar

    Parameters
    ----------
    ref_string : str
        string one
    string : str
        string two
    threshold : float, optional
        similarity threshold to consider two strings similar, by default 0.8

    Returns
    -------
    bool
        True if the two string are similar, False otherwise
    """
    ref_string = ref_string.replace(' ', '').lower()
    string = string.replace(' ', '').lower()
    similarity: float = SequenceMatcher(None, ref_string, string).ratio()
    if similarity >= threshold:
        return True
    return False

def verify_string_list(ref_word: str, word_list: List[float], get_index: bool=True, threshold_value: float=0.8) -> Any:
    """Verify if a string exists inside a list of strings

    Parameters
    ----------
    ref_word : str
        string to consider
    word_list : List[float]
        List of strings
    get_index : bool, optional
        True to return also de string index, False otherwise, by default True
    threshold_value : float, optional
        Similarity threshold to consider two strings similar, by default 0.8

    Returns
    -------
    Any
        True if the string exist inside the list, False otherwise. Also return the string index optionally
    """
    j: int = 0
    for word in word_list:
        test: bool = verify_string(ref_word, word, threshold=threshold_value)
        if test is True and get_index  is True:
            return True, j
        elif test is True:
            return True
        j = j + 1
    if get_index is True:
        return False, None
    return False

def get_block_info(block_list: List[Dict[str, Any]]) -> Any:
    """get all the info about the block inside lists

    Parameters
    ----------
    block_list : List[Dict[str, Any]]
        list of all the blocks

    Returns
    -------
    Any
        returns lists containing the blocks info
    """
    table_boxes: List[List[float]] = []
    table_legends: List[str] = []
    table_pages: List[int] = []
    table_structure: List[List[List[str]]] = []
    table_row_indexes: List[int] = []
    table_collumn_headers: List[int] = []
    figure_boxes: List[List[float]] = []
    figure_legends: List[str] = []
    figure_pages: List[int] = []
    if block_list == []:
        return table_boxes, table_legends, table_pages, table_structure, table_row_indexes, table_collumn_headers, figure_boxes, figure_legends, figure_pages
    for block in block_list:
        if block['type'] == 'Table':
            table_boxes.append(block['box'])
            table_legends.append(block['legend'])
            table_pages.append(block["page"])
            table_structure.append(block['block'])
            table_row_indexes.append(block['row_indexes'])
            table_collumn_headers.append(block['collumn_headers'])
        else:
            figure_boxes.append(block['box'])
            figure_legends.append(block['legend'])
            figure_pages.append(block["page"])
    return table_boxes, table_legends,table_pages, table_structure, table_row_indexes, table_collumn_headers, figure_boxes, figure_legends, figure_pages

def verify_table_strucuture(ref_table: List[List[str]], table: List[List[str]]) -> Dict[str, Any]:
    """compare the table structure of 2 tables

    Parameters
    ----------
    ref_table : List[List[str]]
        reference table
    table : List[List[str]]
        table extracted

    Returns
    -------
    Dict[str, Any]
        a dictionary containing the false positives, false negatives, true positves and if the tables strcuture are similar
    """
    if len(table) == 0:
        if len(ref_table) == 0:
            true_positives: int = 1
            false_positives: int = 0
            false_negatives: int = 0
            equal_structure: int = True
        else:
            true_positives = 0
            false_positives = 0
            false_negatives = len(ref_table)
            equal_structure = False
    else:
        true_positives = min(len(ref_table), len(table)) + min(len(ref_table[0]), len(table[0]))
        false_positives = max(0, len(table) - len(ref_table)) + max(0, len(table[0]) - len(ref_table[0]))
        false_negatives = max(0, len(ref_table) - len(table)) + max(0, len(ref_table[0]) - len(table[0]))
        if false_positives == 0 and false_negatives == 0:
            equal_structure: bool = True
        else:
            equal_structure = False
    return {'true_positives': true_positives, 'false_positives': false_positives, 'false_negatives': false_negatives, 'correct_structure': equal_structure}

def verify_lists(list1: List[Any], list2: List[Any], dev: bool = False) -> Dict[str, Any]:
    """Compare two lists with eachother

    Parameters
    ----------
    list1 : List[Any]
        first list
    list2 : List[Any]
        second list
    dev : bool, optional
        If True give extra information on how the method is working, by default False

    Returns
    -------
    Dict[str, Any]
        A dictinary with the number of true positives, false positives and false negatives of list1 in list2
    """
    true_positives: int = 0
    false_positives: int = len(list2)
    false_negatives: int = 0
    for ref_entry in list1:
        exists_entry: bool = False
        for entry in list2:
            if ref_entry == entry:
                exists_entry = True
                true_positives = true_positives + 1
                false_positives = false_positives - 1
                if dev is True:
                    print(f'found {ref_entry} in {list2}')
                break
        if exists_entry is False:
            false_negatives = false_negatives + 1
            if dev is True:
                    print(f'did not found {ref_entry} in {list2}')
    return {'true_positives': true_positives, 'false_positives': false_positives, 'false_negatives': false_negatives}

def entries_similarity_horizontal(ref_structure: List[List[str]], structure: List[List[str]]) -> float:
    """Calculate the degree of similarity between two tables, by transforming them into string horizontally

    Parameters
    ----------
    ref_structure : List[List[str]]
        reference table
    structure : List[List[str]]
        table to be comapared

    Returns
    -------
    float
        the levenshtein ratio between the two tables
    """
    ref: List[str] = []
    for line in ref_structure:
        for entry in line:
            ref.append(entry)
    test: List[str] =[]
    for line in structure:
        for entry in line:
            test.append(entry)
    ref_string: str = "".join(ref)
    test_string: str = "".join(test)
    ref_string = ref_string.replace(" ", "").lower()
    test_string = test_string.replace(" ", "").lower()
    return ratio(ref_string, test_string)

def entries_similarity_vertical(ref_structure: List[List[str]], structure: List[List[str]]) -> float:
    """Calculate the degree of similarity between two tables, by transforming them into string vertically

    Parameters
    ----------
    ref_structure : List[List[str]]
        reference table
    structure : List[List[str]]
        table to be comapared
    Returns
    -------
    float
        the levenshtein ratio between the two tables
    """
    ref: List[str] = []
    test: List[str] =[]
    for j in range(len(ref_structure[0])):
        for i in range(len(ref_structure)):
            ref.append(ref_structure[i][j])
    if len(structure) == 0:
        size = 0
    else:
        size = len(structure[0])
    for j in range(size):
        for i in range(len(structure)):
            test.append(structure[i][j])
    ref_string: str = "".join(ref)
    test_string: str = "".join(test)
    ref_string = ref_string.replace(" ", "").lower()
    test_string = test_string.replace(" ", "").lower()
    return ratio(ref_string, test_string)
