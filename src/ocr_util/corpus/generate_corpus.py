"""Ground truth corpus generation for OCR evaluation.

This module orchestrates the assembly of a METS-based ground truth corpus
from a collection of PAGE-XML files with URN identifiers.  The main entry
point is :func:`generate`, which accepts a :class:`~ocr_util.corpus.common.CorpusArgs`
configuration object and returns a :class:`~ocr_util.corpus.common.CorpusGeneratorResult`.

Typical call chain
------------------
1. :func:`generate` scans the input directory for PAGE-XML ground truth files.
2. METS metadata is fetched (and cached) in parallel via
   :class:`~ocr_util.corpus.load_metadata.RecordMetadataResolver`.
3. :class:`Corpus` iterates over the resolved inputs, extracts the relevant
   METS/MODS sections via
   :class:`~ocr_util.corpus.common.MetsResourceFile`, and attaches them to a
   shared corpus METS document built from a template.
4. The finished METS document is written to the output directory.

Can also be run as a standalone script::

    python generate_corpus.py <input_dir> <output_dir> [-l LIMIT] [-t TEMP_DIR]
"""

import argparse
import logging
import math
import os
import pathlib
import shutil
import typing

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import ocr_util.corpus.common as cc
import ocr_util.corpus.load_metadata as lr

CPUS = os.cpu_count() or 1
NUM_THREADS: typing.Final[int] = math.ceil(CPUS * 0.85)
DEFAULT_TEMP_DIR = pathlib.Path.home().joinpath(".cache", "odem_gt_2_mets")
DEFAULT_LIMIT = 0

logger = logging.getLogger(__name__)


def fetch_resources(cache_path: pathlib.Path, gt_files: list[cc.CorpusPageInput]) -> list[cc.CorpusPageInput]:
    """Fetch required METS metadata parallel for given inputs."""
    if not cache_path.exists():
        cache_path.mkdir()
    resolver: lr.RecordMetadataResolver = lr.RecordMetadataResolver()
    # Use ThreadPoolExecutor directly for parallel execution
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(resolver.fetch, an_input, cache_path=cache_path) for an_input in gt_files]
        return [future.result() for future in futures]


class Corpus:
    """Ground Truth Corpus in METS format."""

    def __init__(self, cargs: cc.CorpusArgs, inputs: list[cc.CorpusPageInput], **kwargs) -> None:
        self.__cargs = cargs
        self.__out_dir: Final[Path] = self.__cargs.output_dir
        self.__inputs: Final[list[cc.CorpusPageInput]] = inputs
        # Template file is located in the same directory as this module
        template_path = Path(__file__).parent / "template.corpus.xml"
        self.corpus_template = template_path
        self.kwargs = kwargs
        self.corpus_label = self.__cargs.corpus_label
        self.file = cc.MetsCorpusFile(template_path, corpus_label=self.corpus_label, **kwargs)

    def build(self) -> cc.CorpusGeneratorResult:
        """Build the corpus by extracting data from original METS files and write new METS file."""

        total: int = len(self.__inputs)
        assert self.file is not None, "Corpus template must be set before start building corpus"
        for idx, page_input in enumerate(self.__inputs, 1):
            logger.info("[%d/%d] Process file with identifier %s", idx, total, page_input.identifier_urn)
            try:
                assert (
                    page_input.cached_media_mets_file is not None
                ), f"METS file for {page_input.identifier_urn} not found"
                mets_res = cc.MetsResourceFile(page_input.cached_media_mets_file, idx, **self.kwargs)
                page_input.metadata = mets_res
                the_section: cc.MetsModsSection = mets_res.extract(
                    page_urn=page_input.identifier_urn,
                    out_dir=self.__out_dir,
                    gt_file_path=page_input.groundtruth_file.file_path,
                )
                self.file.attach(the_section)
            except Exception as exc:
                logger.exception(
                    "Error processing %s: %s - skip file",
                    page_input.cached_media_mets_file,
                    exc,
                )
        # re-sort pages
        self.file.finalize()
        out_path = self.file.write(self.__out_dir)
        return cc.CorpusGeneratorResult(file_path=out_path, n_pages=len(self.__inputs))


def generate(cargs: cc.CorpusArgs) -> cc.CorpusGeneratorResult:
    """Generate GT corpus in METS format from given corpus arguments."""
    if not cargs.input_dir.exists():
        raise cc.CorpusException(f"Input directory '{cargs.input_dir}' does not exist")
    if cargs.output_dir.exists():
        logger.warning(
            "Output directory '%s' already exists. Refusing to overwrite existing data.",
            cargs.output_dir,
        )
    if cargs.local_cache_dir.exists() and cargs.clear_cache:
        logger.info("Wipe existing cache directory %s", cargs.local_cache_dir)
        shutil.rmtree(cargs.local_cache_dir)
    cargs.local_cache_dir.mkdir(parents=True, exist_ok=True)
    cargs.output_dir.mkdir(parents=True, exist_ok=True)
    gt_files: list[cc.GroundtruthFile] = cc.GroundtruthFile.from_dir_copy(
        in_dir=cargs.input_dir, out_dir=cargs.output_dir.joinpath(cc.GT_TARGET_SUBDIR), limit=cargs.limit
    )
    corpus_inputs = [cc.CorpusPageInput(groundtruth_file=gt_file) for gt_file in gt_files]
    local_cache_path = Path(f"{cargs.local_cache_dir}").joinpath("mets")
    corpus_input_with_resources = fetch_resources(local_cache_path, corpus_inputs)
    corpus_file = Corpus(cargs, corpus_input_with_resources)
    return corpus_file.build()


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the input directory")
    parser.add_argument("output", help="Path to the output directory")
    parser.add_argument(
        "-l",
        "--limit",
        help="Number of Files being processed, default = 0 (unlimited)",
        required=False,
        type=int,
        default=DEFAULT_LIMIT,
    )
    parser.add_argument(
        "-t", "--temp-dir", help="Path to the temporary directory", required=False, default=DEFAULT_TEMP_DIR
    )
    args = parser.parse_args()
    corpus_args = cc.CorpusArgs(
        input_dir=Path(args.input).absolute(),
        output_dir=Path(args.output).absolute(),
        local_cache_dir=Path(args.temp_dir).absolute(),
        limit=int(args.limit),
        corpus_label="Ground Truth Corpus",
    )

    outcome = generate(corpus_args)
    logger.info("Corpus generated at %s with %d pages", outcome.file_path, outcome.n_pages)
