# -*- coding: utf-8 -*-
"""OCR Utils"""

import argparse
import logging
import os
import re
from pathlib import Path, PurePath

import ocr_util.eval.model as do
import ocr_util.eval.model.filter as dofi
import ocr_util.eval.cli as eval_cli
import ocr_util.slice.cli as slice_cli
import ocr_util.show.cli as show_cli
import ocr_util.corpus.generate_corpus as gc

from ocr_util.corpus.common import CorpusArgs

# script constants
DEFAULT_VERBOSITY = 0
SUB_CMD_FRAME = "frame"
SUB_CMD_GROUNDTRUTH_CORPUS = "corpus"
CORPUS_CACHE_DIR_NAME = "ocr_util_corpus_mets_cache"
CORPUS_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", CORPUS_CACHE_DIR_NAME
)

SUB_CMD_EVALUATE = "eval"
SUB_CMD_SLICE = "slice"
SUB_CMD_SHOW = "show"

# Remove this constant as it's now managed by show_cli
# SUB_CMD_SHOW is kept for backward compatibility with other references


def points_type(points: str) -> str:
    match: re.Match = re.match(dofi.PolygonFrameFilterUtil.POINT_LIST_PATTERN, points)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid point coordinates: '{points}'")
    return points


def start() -> None:
    # Configure logging once, centrally
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s][%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="ocr",
    )
    sub_arg_parsers = arg_parser.add_subparsers(
        title="Subkommandos",
        dest="subcommand",
        required=True,
    )

    # frames subcommand
    frame_arg_parser = sub_arg_parsers.add_parser(
        SUB_CMD_FRAME,
        help="Filter Contents of provided ALTO-v3-Data by provided Coordinates, where Coordinates span a rectangular"
        " box with",
    )
    frame_arg_parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')",
    )
    frame_arg_parser.add_argument(
        "-i", "--input-ocr-file", help="Path of OCR-Data file to process", required=True
    )
    frame_arg_parser.add_argument(
        "-o",
        "--output-ocr-file",
        help="Path of resulting OCR-Data file",
        required=False,
        default=None,
    )
    frame_arg_parser.add_argument(
        "-p",
        "--points",
        required=True,
        type=points_type,
        help="""
        Frame to slice words/lines/regions from input OCR-Data
        f.e.: --frame "2892,2480 5072,2480 5072,5148 2892,5148"
        """,
    )

    # groundtruth-corpus subcommand
    groundtruth_corpus_arg_parser = sub_arg_parsers.add_parser(
        SUB_CMD_GROUNDTRUTH_CORPUS,
        help="Create METS files from ground truth PAGE-XML files with URN identifiers",
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-i",
        "--input",
        dest="input_dir",
        help="Path to the input directory containing GT PAGE-XML files",
        required=True,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        help="Path to the output directory for generated corpus",
        required=True,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=0,
        help="Number of files to process (default: 0 = unlimited)",
        required=False,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-t",
        "--temp-dir",
        dest="temp_dir",
        default=CORPUS_CACHE_DIR,
        help=f"Path to temporary directory for caching METS files (default: {CORPUS_CACHE_DIR})",
        required=False,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')",
    )
    groundtruth_corpus_arg_parser.add_argument(
        "--corpus-label",
        dest="corpus_label",
        default="Ground Truth Corpus",
        help="Label for the corpus in the METS logical structure (default: 'Ground Truth Corpus')",
        required=False,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "--oai-base-url",
        dest="oai_base_url",
        help="Base URL for OAI-PMH requests",
        required=False,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "--clear-cache",
        dest="clear_cache",
        action="store_true",
        default=False,
        help="Clear the local cache directory before processing (default: False)",
        required=False,
    )

    # evaluate subcommand
    evaluate_arg_parser = sub_arg_parsers.add_parser(
        SUB_CMD_EVALUATE,
        help="Evaluate OCR candidates against ground truth data",
        add_help=True,
    )
    evaluate_arg_parser.add_argument(
        "candidates",
        help="Root directory for evaluation candidates / Path to single candidate file",
    )
    evaluate_arg_parser.add_argument(
        "-ref",
        "--reference",
        required=False,
        help="Root directory for reference/groundtruth data",
    )
    evaluate_arg_parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag (optional; default: '{DEFAULT_VERBOSITY}')",
    )
    evaluate_arg_parser.add_argument(
        "--metrics",
        default=eval_cli.DEFAULT_OCR_METRICS,
        required=False,
        help=f"Comma-separated list of metrics (default: '{eval_cli.DEFAULT_OCR_METRICS}')",
    )
    evaluate_arg_parser.add_argument(
        "--utf8",
        default=eval_cli.DEFAULT_UTF8_NORM,
        required=False,
        help=f"UTF-8 normalization form (default: '{eval_cli.DEFAULT_UTF8_NORM}')",
    )
    evaluate_arg_parser.add_argument(
        "-s",
        "--sequential",
        action="store_true",
        required=False,
        help="Execute calculations sequentially (default: False)",
    )
    evaluate_arg_parser.add_argument(
        "-x",
        "--extra",
        required=False,
        help="Pass additional information to evaluation (e.g. 'ignore_geometry')",
    )
    evaluate_arg_parser.add_argument(
        "-l",
        "--language",
        required=False,
        help="Language code for LanguageTool (ISO 639-2)",
    )
    evaluate_arg_parser.add_argument(
        "-u",
        "--lt-api-url",
        required=False,
        help="LanguageTool API URL",
    )
    evaluate_arg_parser.add_argument(
        "--aggregate-by",
        required=False,
        help=(
            "Aggregation dimensions. Use ',' for independent keys "
            "(e.g. 'mods:language,mods:dateIssued:century') or '+' for "
            "combined/hierarchical keys "
            "(e.g. 'mods:language+mods:dateIssued:century'). "
            "Other supported types: 'directory', 'type', 'pattern:REGEX'."
        ),
    )
    evaluate_arg_parser.add_argument(
        "--mets-file",
        required=False,
        help="Path to METS/MODS file for MODS-based aggregation",
    )
    evaluate_arg_parser.add_argument(
        "--mods-dimensions",
        required=False,
        help="[LEGACY] Comma-separated MODS dimensions; use --aggregate-by instead",
    )

    # slice subcommand
    slice_arg_parser = sub_arg_parsers.add_parser(
        SUB_CMD_SLICE,
        help="Generate pairs of textlines and image frames from OCR and image data",
        add_help=True,
    )
    slice_arg_parser.add_argument(
        "data",
        type=str,
        help="path to local alto|page file corresponding to image",
    )
    slice_arg_parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="path to local image file tif|jpg|png corresponding to ocr",
    )
    slice_arg_parser.add_argument(
        "-o",
        "--output_dir",
        default=slice_cli.DEFAULT_OUTDIR_PREFIX,
        help=f"output directory, re-created if already exists (default: {slice_cli.DEFAULT_OUTDIR_PREFIX})",
    )
    slice_arg_parser.add_argument(
        "--prefix-output",
        required=False,
        help="optional: prefix each pair using this arg (default: '')",
    )
    slice_arg_parser.add_argument(
        "-m",
        "--minchars",
        required=False,
        type=int,
        default=int(slice_cli.DEFAULT_MIN_CHARS),
        help=f"optional: minimum printable chars required for a line to be included into set (default: {slice_cli.DEFAULT_MIN_CHARS})",
    )
    slice_arg_parser.add_argument(
        "-s",
        "--summary",
        required=False,
        action="store_true",
        default=slice_cli.DEFAULT_USE_SUMMARY,
        help=f"optional: print all lines in additional file (default: {slice_cli.DEFAULT_USE_SUMMARY})",
    )
    slice_arg_parser.add_argument(
        "-r",
        "--reorder",
        required=False,
        action="store_true",
        default=slice_cli.DEFAULT_USE_REORDER,
        help=f"optional: re-order word tokens from right-to-left (default: {slice_cli.DEFAULT_USE_REORDER})",
    )
    slice_arg_parser.add_argument(
        "--binarize",
        required=False,
        action="store_true",
        default=slice_cli.DEFAULT_BINARIZE,
        help=f"optional: binarize textline images (default: {slice_cli.DEFAULT_BINARIZE})",
    )
    slice_arg_parser.add_argument(
        "--sanitize",
        required=False,
        type=bool,
        default=slice_cli.DEFAULT_SANITIZE,
        help=f"optional: sanitize textline images (default: {slice_cli.DEFAULT_SANITIZE})",
    )
    slice_arg_parser.add_argument(
        "--no-sanitize", dest="sanitize", action="store_false"
    )
    slice_arg_parser.add_argument(
        "--intrusion-ratio",
        required=False,
        default=slice_cli.DEFAULT_INTRUSION_RATIO,
        help=f"optional: alter threshold for top and bottom ratios for intrusion detection for sanitizing (default: {slice_cli.DEFAULT_INTRUSION_RATIO})",
    )
    slice_arg_parser.add_argument(
        "--rotation-threshold",
        required=False,
        type=float,
        default=slice_cli.DEFAULT_ROTATION_THRESH,
        help=f"optional: alter threshold for rotation of textline image (default: {slice_cli.DEFAULT_ROTATION_THRESH})",
    )
    slice_arg_parser.add_argument(
        "-p",
        "--padding",
        required=False,
        type=int,
        default=slice_cli.DEFAULT_PADDING,
        help=f"optional: additional padding for existing textline image (default: {slice_cli.DEFAULT_PADDING})",
    )

    # show subcommand
    show_cli.register_arguments(sub_arg_parsers)

    args = arg_parser.parse_args()

    verbosity: int = getattr(args, "verbosity", DEFAULT_VERBOSITY)

    if args.subcommand == SUB_CMD_FRAME:
        input_ocr_file: str = args.input_ocr_file
        output_ocr_file: str = args.output_ocr_file
        points: str = args.points
        if verbosity > 1:
            print(
                f"[DEBUG] args: {input_ocr_file}, {output_ocr_file}, {points}, {verbosity}"
            )
        polygon_frame_filter: dofi.PolygonFrameFilter = dofi.PolygonFrameFilter(
            input_ocr_file, points, verbosity
        )
        piece_result: do.DigitalObjectTree = polygon_frame_filter.process()
        file_result: PurePath = do.from_digital_object(piece_result, output_ocr_file)
        if verbosity > 0:
            print("[INFO ] file_result", file_result)

    elif args.subcommand == SUB_CMD_GROUNDTRUTH_CORPUS:
        corpus_args = CorpusArgs(
            input_dir=Path(args.input_dir).absolute(),
            output_dir=Path(args.output_dir).absolute(),
            local_cache_dir=Path(args.temp_dir).absolute(),
            limit=int(args.limit),
            corpus_label=args.corpus_label,
            clear_cache=args.clear_cache
        )
        gc.generate(corpus_args)

    elif args.subcommand == SUB_CMD_EVALUATE:
        eval_args = vars(args)
        eval_args.pop("subcommand", None)
        eval_cli.start_evaluation(eval_args)

    elif args.subcommand == SUB_CMD_SLICE:
        slice_args = vars(args)
        slice_args.pop("subcommand", None)
        slice_cli.start_slice(slice_args)

    elif args.subcommand == SUB_CMD_SHOW:
        show_args = vars(args)
        show_args.pop("subcommand", None)
        show_cli.start_show(show_args)


if __name__ == "__main__":
    start()
