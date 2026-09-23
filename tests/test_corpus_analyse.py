"""Tests for filtering existing METS corpora."""

from pathlib import Path

import pytest

import ocr_util.corpus.analyse as corpus_analyse

from .conftest import TEST_RES_DIR

METS_FILE = TEST_RES_DIR / "test_mets.xml"


def test_analyse_filters_mods_metadata():
    """return only file references matching a MODS value."""
    matches = corpus_analyse.analyse(METS_FILE, ["mods:genre=article"])

    assert matches == ["1667522809_J_0001_0002.art.gt.xml"]


def test_analyse_combines_repeated_filters_with_and():
    """apply repeated filters with AND semantics."""
    matches = corpus_analyse.analyse(
        METS_FILE,
        ["mods:dateIssued:century=19th", "mods:language=eng"],
    )

    assert matches == ["test_announcement.ann.gt.xml"]


def test_analyse_supports_existing_type_filter():
    """reuse filename-based evaluation type filtering."""
    matches = corpus_analyse.analyse(METS_FILE, ["type=announcement"])

    assert matches == ["test_announcement.ann.gt.xml"]


def test_start_analysis_prints_one_match_per_line(capsys):
    """print only matching file references to stdout."""
    matches = corpus_analyse.start_analysis({"mets_file": METS_FILE, "filter_by": ["mods:genre=article"]})

    assert matches == ["1667522809_J_0001_0002.art.gt.xml"]
    assert capsys.readouterr().out == "1667522809_J_0001_0002.art.gt.xml\n"


def test_analyse_rejects_missing_mets_file(tmp_path: Path):
    """reject a missing corpus METS path."""
    with pytest.raises(FileNotFoundError, match="METS file not found"):
        corpus_analyse.analyse(tmp_path / "missing.xml", ["mods:genre=article"])


def _write_mets(tmp_path: Path, hrefs: list[str]) -> Path:
    files = "".join(f'<mets:file><mets:FLocat xlink:href="{href}"/></mets:file>' for href in hrefs)
    mets_file = tmp_path / "mets.xml"
    mets_file.write_text(
        '<mets:mets xmlns:mets="http://www.loc.gov/METS/" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
        f"<mets:fileSec><mets:fileGrp>{files}</mets:fileGrp></mets:fileSec>"
        "</mets:mets>",
        encoding="utf-8",
    )
    return mets_file


def test_check_corpus_reports_missing_files(tmp_path: Path):
    """check every local FLocat reference relative to the METS file."""
    (tmp_path / "present.xml").touch()
    mets_file = _write_mets(tmp_path, ["present.xml", "missing.xml"])

    result = corpus_analyse.check_corpus(mets_file)

    assert result.is_valid is False
    assert result.checked == (
        tmp_path / "present.xml",
        tmp_path / "missing.xml",
    )
    assert result.missing == (tmp_path / "missing.xml",)
    assert result.uncheckable == ()


def test_check_corpus_reports_uncheckable_remote_references(tmp_path: Path):
    """do not claim that remote references are locally present."""
    mets_file = _write_mets(tmp_path, ["https://example.org/page.xml"])

    result = corpus_analyse.check_corpus(mets_file)

    assert result.is_valid is False
    assert result.checked == ()
    assert result.missing == ()
    assert result.uncheckable == ("https://example.org/page.xml",)


def test_start_analysis_check_prints_status_and_summary(tmp_path: Path, capsys):
    """print file statuses and an integrity summary in check mode."""
    (tmp_path / "present.xml").touch()
    mets_file = _write_mets(tmp_path, ["present.xml"])

    result = corpus_analyse.start_analysis({"mets_file": mets_file, "filter_by": [], "check": True})

    assert isinstance(result, corpus_analyse.CorpusCheckResult)
    assert result.is_valid is True
    assert capsys.readouterr().out == (f"OK {tmp_path / 'present.xml'}\n" "SUMMARY checked=1 missing=0 uncheckable=0\n")


def test_start_analysis_requires_filter_or_check():
    """reject analysis requests without an operation."""
    with pytest.raises(ValueError, match="--filter-by or --check"):
        corpus_analyse.start_analysis({"mets_file": METS_FILE, "filter_by": [], "check": False})
