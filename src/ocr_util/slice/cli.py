# -*- coding: utf-8 -*-
"""Generate Sets of Training Data TextLine + Image Pairs"""

import os
import typing

from argparse import (
    ArgumentParser,
    Namespace,
)
from pathlib import Path

from ocr_util.slice.gts_pairs import (
    DEFAULT_OUTDIR_PREFIX,
    DEFAULT_MIN_CHARS,
    DEFAULT_USE_SUMMARY,
    DEFAULT_USE_REORDER,
    DEFAULT_INTRUSION_RATIO,
    DEFAULT_ROTATION_THRESH,
    DEFAULT_SANITIZE,
    DEFAULT_BINARIZE,
    DEFAULT_PADDING,
    SUFFIX_SUMMARY,
    TrainingSets,
)


def _run_single_page(args: typing.Dict):
    path_ocr = args["data"]
    path_img = args["image"]
    output_dir = os.path.abspath(args["output_dir"])
    min_chars = args["minchars"]
    do_summary = args["summary"]
    do_reorder = args["reorder"]
    do_binarize = args["binarize"]
    do_opt = args["sanitize"]
    intrusion_ratio = args["intrusion_ratio"]
    if isinstance(intrusion_ratio, str) and "," in intrusion_ratio:
        intrusion_ratio = [float(n) for n in intrusion_ratio.split(",")]
    else:
        intrusion_ratio = float(intrusion_ratio)
    rotation_thresh = args["rotation_threshold"]
    padding = args["padding"]
    intrusion_ratio = args["intrusion_ratio"]
    if isinstance(intrusion_ratio, str) and "," in intrusion_ratio:
        intrusion_ratio = [float(n) for n in intrusion_ratio.split(",")]
    else:
        intrusion_ratio = float(intrusion_ratio)
    _t_sets = TrainingSets(path_ocr, path_img, output_dir=output_dir)
    prefix_output = args["prefix_output"]
    if prefix_output:
        _t_sets.pair_prefix = prefix_output
    res = _t_sets.create(
        min_chars=min_chars,
        summary=do_summary,
        reorder=do_reorder,
        intrusion_ratio=intrusion_ratio,
        rotation_threshold=rotation_thresh,
        binarize=do_binarize,
        sanitize=do_opt,
        padding=padding,
    )
    print(f"[DEBUG] got '{len(res)}' pairs from '{path_ocr}'" f" and '{path_img}' in '{output_dir}', better review")
    return len(res)


def _run_dir(the_args: typing.Dict):
    path_input_data = the_args["data"]
    path_img_dir = the_args["image"]
    input_data = sorted(
        [
            os.path.join(path_input_data, a_path)
            for a_path in os.listdir(path_input_data)
            if str(a_path).endswith(".xml")
        ]
    )
    print(f"[DEBUG] found total {len(input_data)} OCR files in {path_input_data} ")
    pages_missed = []
    args_stack = []
    for an_input in input_data:
        data_name = Path(an_input).name
        matching_image = __determine_image(path_img_dir, data_name)
        if matching_image:
            args_copy = dict(the_args)
            specific_args = {"data": an_input, "image": matching_image}
            args_copy.update(specific_args)
            args_stack.append(args_copy)
        else:
            print(f"[WARNING] no img for {data_name}")
            pages_missed.append(an_input)
    print(f"[INFO] found {len(args_stack)} pairs, miss {len(pages_missed)} in {path_img_dir}")
    total_n_pairs = 0
    for the_args in args_stack:
        total_n_pairs += _run_single_page(the_args)
    print(f"[INFO] created {total_n_pairs} pairs")


def __determine_image(path_image_dir, data_name):
    all_images = [
        os.path.join(path_image_dir, a_file)
        for a_file in os.listdir(path_image_dir)
        if __is_matching_image(a_file, data_name)
    ]
    if not all_images:
        return None
    if len(all_images) > 1:
        raise RuntimeError(f"Ambigious image match {all_images} for {data_name}")
    return all_images[0]


def __is_matching_image(file_name: str, data_name: str) -> bool:
    """Check first for possible sub-dirs"""
    file_parts = file_name.split(".")
    final_segment: str = file_parts[-1]
    xtn_matches = final_segment.lower() in ["jpg", "tif", "png"]
    lbl_matches = file_parts[0] in data_name
    return xtn_matches and lbl_matches


########
# MAIN #
########
def main():
    PARSER: ArgumentParser = ArgumentParser(
        description="generate pairs of textlines and image frames from existing OCR and image data"
    )
    PARSER.add_argument("data", type=str, help="path to local alto|page file corresponding to image")
    PARSER.add_argument(
        "-i",
        "--image",
        required=False,
        help="path to local image file tif|jpg|png corresponding to ocr. (default: read from OCR-Data)",
    )
    PARSER.add_argument(
        "-o",
        "--output_dir",
        default=DEFAULT_OUTDIR_PREFIX,
        help=f"output directory, re-created if already exists. (default: <script-dir>/<{DEFAULT_OUTDIR_PREFIX}>)",
    )
    PARSER.add_argument(
        "--prefix-output", required=False, help="optional: prefix each pair using this arg. (default: '')"
    )
    PARSER.add_argument(
        "-m",
        "--minchars",
        required=False,
        type=int,
        default=int(DEFAULT_MIN_CHARS),
        help=f"optional: minimum printable chars required for a line to be included into set (default: {DEFAULT_MIN_CHARS})",
    )
    PARSER.add_argument(
        "-s",
        "--summary",
        required=False,
        action="store_true",
        default=DEFAULT_USE_SUMMARY,
        help=f"optional: print all lines in additional file (default: {DEFAULT_USE_SUMMARY}, pattern: <default-output-dir>{SUFFIX_SUMMARY})",
    )
    PARSER.add_argument(
        "-r",
        "--reorder",
        required=False,
        action="store_true",
        default=DEFAULT_USE_REORDER,
        help=f"optional: re-order word tokens from right-to-left (default: {DEFAULT_USE_REORDER})",
    )
    PARSER.add_argument(
        "--binarize",
        required=False,
        action="store_true",
        default=DEFAULT_BINARIZE,
        help=f"optional: binarize textline images (default: {DEFAULT_BINARIZE})",
    )
    PARSER.add_argument(
        "--sanitize",
        required=False,
        type=bool,
        default=DEFAULT_SANITIZE,
        help=f"optional: sanitize textline images (default: {DEFAULT_SANITIZE})",
    )
    PARSER.add_argument("--no-sanitize", dest="sanitize", action="store_false")
    PARSER.add_argument(
        "--intrusion-ratio",
        required=False,
        default=DEFAULT_INTRUSION_RATIO,
        help=f"optional: alter threshold for top and bottom ratios for intrusion detection for sanitizing (default: {DEFAULT_INTRUSION_RATIO})",
    )
    PARSER.add_argument(
        "--rotation-threshold",
        required=False,
        type=float,
        default=DEFAULT_ROTATION_THRESH,
        help=f"optional: alter threshold for rotation of textline image (default: {DEFAULT_ROTATION_THRESH})",
    )
    PARSER.add_argument(
        "-p",
        "--padding",
        required=False,
        type=int,
        default=DEFAULT_PADDING,
        help=f"optional: additional padding for existing textline image (default: {DEFAULT_PADDING})",
    )

    ARGS: Namespace = PARSER.parse_args()
    print(f"[DEBUG] {os.path.basename(__file__)} using args: {ARGS}")
    path_ocr_data = Path(ARGS.data).resolve()
    ARGS.data = path_ocr_data
    path_img_data = Path(ARGS.image).resolve()
    ARGS.image = path_img_data
    if path_ocr_data.is_file() and path_img_data.is_file():
        print(f"[INFO ] generate trainingsets from single file '{path_ocr_data}'")
        _run_single_page(vars(ARGS))
    elif path_ocr_data.is_dir() and path_img_data.is_dir():
        _run_dir(vars(ARGS))
        print(f"[INFO ] inspect OCR-dir '{path_ocr_data}' and image dir '{path_img_data}")
    else:
        print(f"[ERROR  ] invalid OCR '{path_ocr_data}' or Image '{path_img_data}'!")


def start_slice(parse_args: typing.Dict):
    """Start slice processing with parsed arguments from parent CLI"""
    path_ocr_data = Path(parse_args["data"]).resolve()
    parse_args["data"] = path_ocr_data
    path_img_data = Path(parse_args["image"]).resolve()
    parse_args["image"] = path_img_data
    if path_ocr_data.is_file() and path_img_data.is_file():
        print(f"[INFO ] generate trainingsets from single file '{path_ocr_data}'")
        _run_single_page(parse_args)
    elif path_ocr_data.is_dir() and path_img_data.is_dir():
        _run_dir(parse_args)
        print(f"[INFO ] inspect OCR-dir '{path_ocr_data}' and image dir '{path_img_data}")
    else:
        print(f"[ERROR  ] invalid OCR '{path_ocr_data}' or Image '{path_img_data}'!")


if __name__ == "__main__":
    main()
