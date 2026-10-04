"""filter corpus files based on evaluation-compatible criteria."""

import argparse
import pathlib
import re
import typing

import xml.etree.ElementTree as ET

from ocr_util.eval.cli import _build_filter_spec, _filter_value_matches
import ocr_util.eval as digev

METS_NAMESPACES = digev.METSModsExtractor.DEFAULT_NAMESPACES


def analyse(mets_file: pathlib.Path, filters: typing.Sequence[str]) -> list[str]:
    """return corpus file references matching all evaluation filter specifications."""
    mets_file = pathlib.Path(mets_file)
    if not mets_file.is_file():
        raise FileNotFoundError(f"METS file not found: {mets_file}")

    entries = _corpus_entries(mets_file)
    for filter_by in filters:
        filter_spec = _build_filter_spec(filter_by, mets_file, strict=True)
        if filter_spec is None:
            raise ValueError(f"Invalid filter specification: '{filter_by}'")
        _, extractor, expected_value = filter_spec
        entries = [
            entry
            for entry in entries
            if (value := extractor(entry)) is not None
            and str(value).strip()
            and _filter_value_matches(str(value), expected_value)
        ]

    return [str(entry.path_groundtruth) for entry in entries]


def _corpus_entries(mets_file: pathlib.Path) -> list[digev.EvalEntry]:
    """create entry-like objects for full-text files referenced by a METS corpus."""
    tree = ET.parse(mets_file)
    entries: list[digev.EvalEntry] = []

    for file_group in tree.findall(".//mets:fileGrp", METS_NAMESPACES):
        if "FULLTEXT" not in (file_group.get("USE") or "").upper():
            continue

        for flocat in file_group.findall("./mets:file/mets:FLocat", METS_NAMESPACES):
            href = flocat.get("{http://www.w3.org/1999/xlink}href")
            if href is None:
                continue
            referenced_path = pathlib.Path(str(href))
            entry = digev.EvalEntry(referenced_path)
            entry.path_groundtruth = referenced_path
            entry.domain_directories = list(reversed(referenced_path.parent.parts))
            entry.gt_type = _groundtruth_type(referenced_path) or entry.gt_type
            entries.append(entry)
    return entries


def _groundtruth_type(path: pathlib.Path) -> typing.Optional[str]:
    """infer the evaluation ground-truth type from a corpus file name."""
    match = re.match(r".*(?:gt\.(\w{3,})|\.(\w{3,})\.gt)\.xml$", path.name)
    label = next((group for group in match.groups() if group), None) if match else None
    if label and label.startswith("art"):
        return "article"
    if label and label.startswith("ann"):
        return "announcement"
    return None


def start_analysis(
    args: typing.Mapping[str, typing.Any],
) -> typing.List[str]:
    matches = analyse(pathlib.Path(args["mets_file"]), args["filter_by"])
    for match in matches:
        print(match)
    return matches


def register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--filter-by",
        action="append",
        default=[],
        help=(
            "Evaluation-compatible filter; repeat to combine filters with AND, "
            "for example 'mods:dateIssued:century=16th'"
        ),
    )


def start() -> None:
    """run the standalone corpus analysis CLI."""
    parser = argparse.ArgumentParser()
    register_arguments(parser)
    result = start_analysis(vars(parser.parse_args()))


if __name__ == "__main__":
    start()
