"""Analyse a METS corpus with evaluation-compatible filters."""

from __future__ import annotations

import argparse
import dataclasses
import re
import typing
from pathlib import Path
from urllib.parse import unquote, urlparse

import lxml.etree as ET

import ocr_util.eval as digev
from ocr_util.eval.cli import _build_filter_spec, _filter_value_matches

METS_NAMESPACES = digev.METSModsExtractor.DEFAULT_NAMESPACES


@dataclasses.dataclass(frozen=True)
class CorpusCheckResult:
    """summarize local file references found in a METS corpus."""

    checked: tuple[Path, ...]
    missing: tuple[Path, ...]
    uncheckable: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """return whether every METS file reference is locally present."""
        return not self.missing and not self.uncheckable


def _groundtruth_type(path: Path) -> typing.Optional[str]:
    """infer the evaluation ground-truth type from a corpus file name."""
    match = re.match(r".*(?:gt\.(\w{3,})|\.(\w{3,})\.gt)\.xml$", path.name)
    label = next((group for group in match.groups() if group), None) if match else None
    if label and label.startswith("art"):
        return "article"
    if label and label.startswith("ann"):
        return "announcement"
    return None


def _corpus_entries(mets_file: Path) -> list[digev.EvalEntry]:
    """create entry-like objects for full-text files referenced by a METS corpus."""
    tree = ET.parse(str(mets_file))
    entries: list[digev.EvalEntry] = []

    for file_group in tree.xpath("//mets:fileGrp", namespaces=METS_NAMESPACES):
        if "FULLTEXT" not in (file_group.get("USE") or "").upper():
            continue

        for href in file_group.xpath("./mets:file/mets:FLocat/@xlink:href", namespaces=METS_NAMESPACES):
            referenced_path = Path(str(href))
            entry = digev.EvalEntry(referenced_path)
            entry.path_groundtruth = referenced_path
            entry.domain_directories = list(reversed(referenced_path.parent.parts))
            entry.gt_type = _groundtruth_type(referenced_path) or entry.gt_type
            entries.append(entry)

    return entries


def check_corpus(mets_file: Path) -> CorpusCheckResult:
    """check whether every file referenced by the METS corpus is present locally."""
    mets_file = Path(mets_file)
    if not mets_file.is_file():
        raise FileNotFoundError(f"METS file not found: {mets_file}")

    tree = ET.parse(str(mets_file))
    hrefs = tree.xpath("//mets:FLocat/@xlink:href", namespaces=METS_NAMESPACES)
    checked: list[Path] = []
    missing: list[Path] = []
    uncheckable: list[str] = []

    for href_value in hrefs:
        href = str(href_value)
        parsed = urlparse(href)
        if parsed.scheme not in ("", "file") or (parsed.netloc not in ("", "localhost")):
            uncheckable.append(href)
            continue

        path = Path(unquote(parsed.path))
        if not path.is_absolute():
            path = mets_file.parent / path
        checked.append(path)
        if not path.is_file():
            missing.append(path)

    return CorpusCheckResult(tuple(checked), tuple(missing), tuple(uncheckable))


def analyse(mets_file: Path, filters: typing.Sequence[str]) -> list[str]:
    """return corpus file references matching all evaluation filter specifications."""
    mets_file = Path(mets_file)
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


def register_arguments(parser: argparse.ArgumentParser) -> None:
    """register command-line arguments for corpus analysis."""
    parser.add_argument("mets_file", type=Path, help="Path to the corpus METS file")
    parser.add_argument(
        "--filter-by",
        action="append",
        default=[],
        help=(
            "Evaluation-compatible filter; repeat to combine filters with AND, "
            "for example 'mods:dateIssued:century=16th'"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that every local file referenced by the METS corpus exists",
    )


def start_analysis(
    args: typing.Mapping[str, typing.Any],
) -> typing.Union[list[str], CorpusCheckResult]:
    """analyse a corpus and print each matching file reference to stdout."""
    if args.get("check"):
        result = check_corpus(Path(args["mets_file"]))
        missing = set(result.missing)
        for path in result.checked:
            status = "MISSING" if path in missing else "OK"
            print(f"{status} {path}")
        for href in result.uncheckable:
            print(f"UNCHECKABLE {href}")
        print(
            f"SUMMARY checked={len(result.checked)} missing={len(result.missing)} "
            f"uncheckable={len(result.uncheckable)}"
        )
        return result

    if not args.get("filter_by"):
        raise ValueError("at least one --filter-by or --check is required")

    matches = analyse(Path(args["mets_file"]), args["filter_by"])
    for match in matches:
        print(match)
    return matches


def start() -> None:
    """run the standalone corpus analysis CLI."""
    parser = argparse.ArgumentParser()
    register_arguments(parser)
    result = start_analysis(vars(parser.parse_args()))
    if isinstance(result, CorpusCheckResult) and not result.is_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    start()
