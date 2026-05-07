# -*- coding: utf-8 -*-
"""Show CLI - OCR Visualization"""

import argparse
import os
import typing

from ocr_util.show.ocr_show_segmentation import _convert, _visualize


def register_arguments(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register show subcommand arguments with parent parser.
    
    Args:
        subparsers: The subparsers action from ArgumentParser.add_subparsers()
        
    Returns:
        The show subcommand parser
    """
    show_parser = subparsers.add_parser(
        "show",
        help="Visualize OCR segmentation on images",
        add_help=True,
    )
    show_parser.add_argument(
        "-i", "--image", required=True, help="path image file (TIF)"
    )
    show_parser.add_argument(
        "-o", "--ocr", required=True, help="path OCR file (ALTO or PAGE XML)"
    )
    show_parser.add_argument(
        "-d", "--output-dir", required=False, default=os.getcwd(),
        help="output directory for visualization results (default: current working directory)"
    )
    show_parser.add_argument("-f", "--format", required=False, default="jpg",
                              choices=["jpg", "jpeg", "png", "tif", "tiff"],
                              help="output image format (default: jpg)")
    show_parser.add_argument(
        "-l", "--show-labels", required=False, action="store_true", default=False,
        help="render polygon ID labels on output image (default: False)"
    )
    show_parser.add_argument(
        "-v", "--verbose", required=False, action="store_true", help="output info"
    )
    return show_parser


def start_show(parse_args: typing.Dict):
    """Start show processing with parsed arguments from parent CLI"""
    path_img = parse_args["image"]
    path_ocr = parse_args["ocr"]
    output_dir = parse_args.get("output_dir", os.getcwd())
    verbose = parse_args.get("verbose", False)
    img_format = parse_args.get("format", "jpg")
    show_labels = parse_args.get("show_labels", False)

    if verbose:
        print(f'[INFO] call with args {parse_args}')
        print(f'[INFO] output directory: {output_dir}')

    if os.path.exists(path_img) and os.path.exists(path_ocr):
        converted_tmp = _convert(path_img, output_dir)
        final_output = _visualize(converted_tmp, path_ocr, img_format=img_format, show_labels=show_labels)
        print(f'[INFO] visualization complete. Final output: {final_output}')
    else:
        print(f'[ERROR] {path_img} or {path_ocr} not existing!')
