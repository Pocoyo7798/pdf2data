from pdf2data.support import get_doc_list, verify_string, verify_string_list, calc_metrics, get_block_info, verify_boxes, verify_table_strucuture, verify_lists, entries_similarity_horizontal, entries_similarity_vertical
import json
import shutil
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup as bs
from PIL import Image
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class Evaluator(BaseModel):
    ref_folder: str
    result_folder : str
    eval_file_path : str
    string_similarity: float = 0.85
    iou_threshold: float = 0.8

    def eval_metadata(self) -> None:
        """evaluates the metadate_finder
        """
        with open(self.ref_folder + '/subset_metadata.json', "r") as f:
            ref_metadata = json.load(f)
        tp_titles: int = 0
        fp_titles: int = 0
        fn_titles: int = 0
        tp_doi: int = 0
        fp_doi: int = 0
        fn_doi: int = 0
        total_tp_authors: int = 0
        total_fp_authors: int = 0
        total_fn_authors: int = 0
        total_docs: int = len(ref_metadata)
        doc_number: int = 1
        for entry in ref_metadata:
            print(f'{doc_number}//{total_docs} processed')
            doc_number += 1
            doc_name: str = entry['id']
            ref_title: str = entry['title']
            ref_title = ref_title.replace('\n', '')
            if entry['doi'] is not None:
                ref_doi: Optional[str] = entry['doi']
                ref_doi = ref_doi.replace('\n', '')
            else:
                ref_doi = None
            author_string: str = entry['authors'].replace(' and', ',')
            author_string = author_string.replace('\n', '')
            ref_authors: List[str] = author_string.split(', ')
            file_path: str = self.result_folder + '/' + doc_name + '_metadata.json'
            with open(file_path, "r") as f:
                metadata: dict = json.load(f)
            title: str = metadata["title"][0]
            doi: str = metadata["doi"][0]
            authors: List[str] = metadata["authors"]
            if verify_string(ref_title, title, threshold=self.string_similarity) is True:
                tp_titles = tp_titles + 1
            elif title == 'Nothing Found':
                fn_titles = fn_titles + 1
            else:
                fp_titles = fp_titles + 1
            if ref_doi is None:
                print(f'The document {doc_name} does not present any doi')
            elif verify_string(ref_doi, doi, threshold=self.string_similarity) is True:
                tp_doi = tp_doi + 1
            elif doi == 'Nothing Found':
                fn_doi = fn_doi + 1
            else:
                fp_doi = fp_doi + 1
            tp_authors: int = 0
            fp_authors: int = len(authors)
            fn_authors: int = 0
            for author in ref_authors:
                if verify_string_list(author, authors, get_index=False, threshold_value=0.4):
                    tp_authors = tp_authors + 1
                    fp_authors = fp_authors - 1
                else:
                    fn_authors = fn_authors + 1
            total_tp_authors = total_tp_authors + tp_authors
            total_fp_authors = total_fp_authors + max(0, fp_authors)
            total_fn_authors = total_fn_authors + fn_authors
            #similarity = SequenceMatcher(None, ref_authors[0], authors[0]).ratio()
            #print(f"ref_authors: {ref_authors}")
            #print(f"authors: {authors}")
            #print(similarity)
        results = {}
        results['Titles'] = calc_metrics(tp_titles, fp_titles, fn_titles)
        results['DOI'] = calc_metrics(tp_doi, fp_doi, fn_doi)
        results['Authors'] = calc_metrics(total_tp_authors, total_fp_authors, total_fn_authors)
        results_json = json.dumps(results, indent=4)
        with open(self.eval_file_path, "w") as f:
            f.write(results_json)


    def eval_text(self) -> None:
        """Evaluate the text_extractor
        """
        doc_list: List[str] = get_doc_list(self.ref_folder, '.json')
        total_tp_lines: int = 0
        total_fp_lines: int = 0
        total_fn_lines: int = 0
        total_correct_type: int = 0
        total_error_type: int = 0
        total_correct_order: int = 0
        total_error_order: int = 0
        all_similarities: int = []
        total_docs: int = len(doc_list)
        doc_number: int = 1
        for file in doc_list:
            print(f'{doc_number}//{total_docs} processed')
            doc_number += 1
            ref_path: str = self.ref_folder + '/' + file
            with open(ref_path, "r") as f:
                ref_zones = json.load(f)
            result_file: str = file.replace("zones", "")
            result_path: str = self.result_folder + '/' + result_file
            with open(result_path, "r") as f:
                zones: dict = json.load(f)
            j: int = 0
            index: int = -1
            tp_lines: int = 0
            fp_lines: int = len(zones['Text'])
            fn_lines: int = 0
            ref_full_text_list: List[str] = []
            full_text_list: List[str] = []
            for line in zones['Text']:
                full_text_list.append(line)
            for line in ref_zones['Text']:
                exist_line, new_index = verify_string_list(line, zones['Text'],threshold_value=self.string_similarity)
                ref_full_text_list.append(line)
                if exist_line is True:
                    tp_lines = tp_lines + 1
                    fp_lines = fp_lines - 1
                    if ref_zones['Type'][j] == zones['Type'][new_index]:
                        total_correct_type = total_correct_type + 1
                    else:
                        total_error_type = total_error_type + 1
                    if new_index > index:
                        total_correct_order = total_correct_order + 1
                        zones['Text'][new_index] = '#####'
                        index = new_index
                    else:
                        total_error_order = total_error_order + 1
                else:
                    fn_lines = fn_lines + 1
                j = j + 1
            total_tp_lines = total_tp_lines + tp_lines
            total_fp_lines = total_fp_lines + fp_lines
            total_fn_lines = total_fn_lines + fn_lines
            ref_full_text: str = ' '.join(ref_full_text_list)
            ref_full_text = ref_full_text.replace('  ', ' ')
            full_text: str = ' '.join(full_text_list)
            similarity_value = SequenceMatcher(None, ref_full_text, full_text).ratio()
            all_similarities.append(similarity_value)
        results: dict = {}
        type_accuracy = total_correct_type / (total_correct_type + total_error_type)
        order_accuracy = total_correct_order / (total_correct_order + total_error_order)
        results['Entries'] = calc_metrics(total_tp_lines, total_fp_lines, total_fn_lines)
        results['Types'] = {'Accuracy': type_accuracy}
        results['Order'] = {'Accuracy': order_accuracy}
        results['Similarity'] = {'Accuracy' : np.average(all_similarities)}
        results_json = json.dumps(results, indent=4)
        with open(self.eval_file_path, "w") as f:
            f.write(results_json)

    def eval_blocks(self) -> None:
        """evaluate the block_extractor method
        """
        doc_list: List[str] = get_doc_list(self.ref_folder, '.json')
        total_tp_table_boxes: int  = 0
        total_fp_table_boxes: int = 0
        total_fn_table_boxes: int = 0
        total_tp_table_structure: int = 0
        total_fp_table_structure: int = 0
        total_fn_table_structure: int = 0
        entries_ratio_list_h: List[str] = []
        entries_ratio_list_v: List[str] = []
        total_tp_table_row_indexes:int = 0
        total_fp_table_row_indexes: int = 0
        total_fn_table_row_indexes: int = 0
        total_tp_table_column_headers: int = 0
        total_fp_table_column_headers: int = 0
        total_fn_table_column_headers: int = 0
        total_tp_figure_boxes: int = 0
        total_fp_figure_boxes: int = 0
        total_fn_figure_boxes: int = 0
        total_tp_block_legends: int = 0
        total_fp_block_legends: int = 0
        total_fn_block_legends: int = 0
        correct_structure: int = 0
        total_tables: int = 0
        for file in doc_list:
            file_path = self.ref_folder + '/' + file
            with open(file_path, "r") as f:
                ref_blocks: dict = json.load(f)
            result_path: str = self.result_folder + '/' + file
            with open(result_path, "r") as f:
                document_blocks: dict = json.load(f)
            blocks: List[Any] = document_blocks["blocks"]
            table_boxes, table_legends, table_pages, table_structure, table_row_indexes, table_column_headers, figure_boxes, figure_legends, figure_pages = get_block_info(blocks)
            tp_table_boxes: int = 0
            fp_table_boxes: int = len(table_boxes)
            fn_table_boxes: int = 0
            tp_figure_boxes: int = 0
            fp_figure_boxes: int = len(figure_boxes)
            fn_figure_boxes: int = 0
            tp_block_legends: int = 0
            fp_block_legends: int = 0
            fn_block_legends: int = 0
            for block in ref_blocks['Blocks']:
                # Boxes, indexes, headers and entries evaluation
                equal_structure: Optional[bool] = None
                exists_table: Optional[bool] = None
                exists_figure: Optional[bool] = None
                index: Optional[int] = None
                box: List[float] = block["box"]
                page: int = block["page"]
                exist_block: bool = False
                legend: str = ''
                structure: List[List[str]] = []
                ref_structure: List[List[str]] = []
                ref_legend: List[List[str]] = block["legend"]
                if block["type"] == "Table":
                    exists_table, index= verify_boxes(box, page, table_boxes, table_pages, iou_value=self.iou_threshold, get_index=True)
                else:
                    exists_figure, index= verify_boxes(box, page, figure_boxes, figure_pages, iou_value=self.iou_threshold, get_index=True)
                if exists_figure is True:
                    exist_block = True
                    tp_figure_boxes = tp_figure_boxes + 1
                    fp_figure_boxes = fp_figure_boxes - 1
                    legend: str = figure_legends[index]
                elif exists_figure is False:
                    fn_figure_boxes = fn_figure_boxes + 1
                elif exists_table is True:
                    exist_block = True
                    tp_table_boxes = tp_table_boxes + 1
                    fp_table_boxes = fp_table_boxes - 1
                    legend = table_legends[index]
                    column_headers = table_column_headers[index]
                    ref_column_headers = block['column_headers']
                    row_indexes = table_row_indexes[index]
                    ref_row_indexes = block['row_indexes']
                    structure = table_structure[index]
                    ref_structure = block['block']
                    total_tables = total_tables + 1
                    structure_evaluation = verify_table_strucuture(ref_structure, structure)
                    total_tp_table_structure = total_tp_table_structure + structure_evaluation['true_positives']
                    total_fp_table_structure = total_fp_table_structure + structure_evaluation['false_positives']
                    total_fn_table_structure = total_fn_table_structure + structure_evaluation['false_negatives']
                    equal_structure = structure_evaluation['correct_structure']
                    if equal_structure is True:
                        correct_structure = correct_structure + 1
                    entries_ratio_h = entries_similarity_horizontal(ref_structure, structure)
                    entries_ratio_list_h.append(entries_ratio_h)
                    entries_ratio_v = entries_similarity_vertical(ref_structure, structure)
                    entries_ratio_list_v.append(entries_ratio_v)
                    collumn_evaluation = verify_lists(ref_column_headers, column_headers)
                    total_tp_table_column_headers = total_tp_table_column_headers + collumn_evaluation['true_positives']
                    total_fp_table_column_headers = total_fp_table_column_headers + collumn_evaluation['false_positives']
                    total_fn_table_column_headers = total_fn_table_column_headers + collumn_evaluation['false_negatives']
                    row_evaluation = verify_lists(ref_row_indexes, row_indexes)
                    total_tp_table_row_indexes = total_tp_table_row_indexes + row_evaluation['true_positives']
                    total_fp_table_row_indexes = total_fp_table_row_indexes + row_evaluation['false_positives']
                    total_fn_table_row_indexes = total_fn_table_row_indexes + row_evaluation['false_negatives']
                elif exists_table is False:
                    fn_table_boxes = fn_table_boxes + 1
                    print(ref_legend)
                # Legends evaluation
                if exist_block is True:
                    if verify_string(ref_legend, legend) is True:
                        tp_block_legends = tp_block_legends + 1
                    elif legend == '':
                        fn_block_legends = fn_block_legends + 1
                    else:
                        fp_block_legends = fp_block_legends + 1
                    #print(ref_structure)
                    #print(structure)
            total_tp_table_boxes = total_tp_table_boxes + tp_table_boxes
            total_fp_table_boxes = total_fp_table_boxes + max(0, fp_table_boxes)
            total_fn_table_boxes = total_fn_table_boxes + fn_table_boxes
            total_tp_figure_boxes = total_tp_figure_boxes + tp_figure_boxes
            total_fp_figure_boxes = total_fp_figure_boxes + max(0, fp_figure_boxes)
            total_fn_figure_boxes = total_fn_figure_boxes + fn_block_legends
            total_tp_block_legends = total_tp_block_legends + tp_block_legends
            total_fp_block_legends = total_fp_block_legends + max(0, fp_block_legends)
            total_fn_block_legends = total_fn_block_legends + fn_block_legends
        results: dict = {}
        structure_accuracy: float = correct_structure / total_tables
        results['table_detection'] = calc_metrics(total_tp_table_boxes, total_fp_table_boxes, total_fn_table_boxes)
        results['figure_detection'] = calc_metrics(total_tp_figure_boxes, total_fp_figure_boxes, total_fn_figure_boxes)
        results['row_indexes'] = calc_metrics(total_tp_table_row_indexes, total_fp_table_row_indexes, total_fn_table_row_indexes)
        results['column_headers'] = calc_metrics(total_tp_table_column_headers, total_fp_table_column_headers, total_fn_table_column_headers)
        results['legends'] = calc_metrics(total_tp_block_legends, total_fp_block_legends, total_fn_block_legends)
        results['table_structure'] = calc_metrics(total_tp_table_structure, total_fp_table_structure, total_fn_table_structure)
        results['table_structure']['accuracy'] = structure_accuracy
        results['entries'] = {}
        results['entries']["horizontal similarity"] = np.average(entries_ratio_list_h)
        results['entries']["vertical similarity"] = np.average(entries_ratio_list_v)
        df = pd.DataFrame.from_dict(results, orient='index') # convert dict to dataframe
        df.to_excel(self.eval_file_path, ".xlsx")
        

    def eval_table_detector(self) -> None:
        doc_list = get_doc_list(self.ref_folder, 'jpg')
        with open(self.ref_folder + '/data.json', "r") as f:
            image_data = json.load(f)
        with open(self.result_folder, "r") as f:
            results_data = json.load(f)
        annotations = image_data['annotations']
        data_index = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_docs = len(doc_list)
        for doc_number in range(1, total_docs + 1):
            print(f'{doc_number}//{total_docs} processed')
            table_boxes = results_data[f"{doc_number}.jpg"]
            true_positives = 0
            false_positives = len(table_boxes)
            false_negatives = 0
            image_id = annotations[data_index]['image_id']
            while image_id == doc_number:
                box_exist = False
                box = annotations[data_index]['bbox']
                new_box = [box[0], box[1], box[0] + box[2], box[1] +  box[3]]
                box_exist = verify_boxes(new_box, 0, table_boxes, [], iou_value=self.iou_threshold)
                if box_exist is True:
                    true_positives = true_positives + 1
                    false_positives = false_positives - 1
                else:
                    false_negatives = false_negatives + 1
                data_index = data_index + 1
                image_id = annotations[data_index]['image_id']
            total_true_positives = total_true_positives + true_positives
            total_false_positives = total_false_positives + max(0, false_positives)
            total_false_negatives = total_false_negatives + false_negatives
        results = calc_metrics(total_true_positives, total_false_positives, total_false_negatives)
        results_json = json.dumps(results, indent=4)
        with open(self.eval_file_path, "w") as f:
            f.write(results_json)