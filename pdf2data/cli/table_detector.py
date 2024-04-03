from typing import Any, Dict, List, Optional
import json

import click
import cv2

from pdf2data.mask import LayoutParser
from pdf2data.support import get_doc_list

@click.command()
@click.argument("input_folder", type=str)
@click.argument("eval_file", type=str)
@click.option(
    "--model",
    default="TableBank_faster_rcnn_R_101_FPN_3x",
    help="model to detect tables",
)
@click.option(
    "--model_threshold",
    default=0.8,
    help="threshold of the model to detect tables",
)
@click.option(
    "--image_type",
    default="jpg",
    help="image file format",
)
@click.option(
    "--device_type",
    default="cpu",
    help="device type to run the code. Ex: 'cuda'. cpu, etc...",
)

def table_detector(input_folder: str, eval_file: str, model: str, model_threshold: str, image_type: str, device_type: str):
    mask: LayoutParser = LayoutParser(
        model=model,
        model_threshold=model_threshold,
        device_type=device_type
    )
    mask.model_post_init(None)
    images: List[str] = get_doc_list(input_folder, "." + image_type)
    results: Dict[str, Any] = {}
    total_images: int = len(images)
    image_number: int = 1
    for image in images:
        print(f"{image_number}/{total_images}")
        image_number += 1
        file_path: str = input_folder + "/" + image
        page: Any = cv2.imread(file_path)
        height, width, channels = page.shape
        if mask.model == "microsoft/table-transformer-detection":
            layout: Dict[str, Any] = LayoutParser.generate_layout_hf(
                mask._model,
                page,
                width,
                width,
                height,
                height,
                mask.model_threshold,
            )
        else:
            layout = LayoutParser.generate_layout_lp(
                mask._model,
                page,
                width,
                width,
                height,
                height,
            )
        results[image] = layout["table_boxes"]
    json_results: Any = json.dumps(results, indent=4)
    with open(eval_file, "w") as j:
        j.write(json_results)


def main():
    table_detector()


if __name__ == "__main__":
    main()