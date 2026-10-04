"""tests for the corpus filter functionality."""

import pytest
from pathlib import Path

import ocr_util.corpus.filter as cfilter

from .conftest import TEST_RES_DIR

METS_FILE = TEST_RES_DIR / "test_mets.xml"


def test_analyse_filters_mods_metadata():
    """return only file references matching a MODS value."""
    matches = cfilter.analyse(METS_FILE, ["mods:genre=article"])

    assert matches == ["1667522809_J_0001_0002.art.gt.xml"]


def test_analyse_combines_repeated_filters_with_and():
    """apply repeated filters with AND semantics."""
    matches = cfilter.analyse(
        METS_FILE,
        ["mods:dateIssued:century=19th", "mods:language=eng"],
    )

    assert matches == ["test_announcement.ann.gt.xml"]


def test_analyse_supports_existing_type_filter():
    """reuse filename-based evaluation type filtering."""
    matches = cfilter.analyse(METS_FILE, ["type=announcement"])

    assert matches == ["test_announcement.ann.gt.xml"]


def test_start_analysis_prints_one_match_per_line(capsys):
    """print only matching file references to stdout."""
    matches = cfilter.start_analysis({"mets_file": METS_FILE, "filter_by": ["mods:genre=article"]})

    assert matches == ["1667522809_J_0001_0002.art.gt.xml"]
    assert capsys.readouterr().out == "1667522809_J_0001_0002.art.gt.xml\n"


def test_analyse_rejects_missing_mets_file(tmp_path: Path):
    """reject a missing corpus METS path."""
    with pytest.raises(FileNotFoundError, match="METS file not found"):
        cfilter.analyse(tmp_path / "missing.xml", ["mods:genre=article"])
