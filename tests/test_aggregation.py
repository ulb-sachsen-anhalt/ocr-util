# -*- coding: utf-8 -*-
"""Test Module for Generic Aggregation System"""

import copy
import os
import shutil
from pathlib import Path

import lxml.etree as ET

import pytest

import ocr_util.eval as digev_main
import ocr_util.eval.evaluation as digev
import ocr_util.eval.metrics as digem

from .conftest import TEST_RES_DIR

_CANDIDATE_FILE = "1667522809_J_0001_0002.xml"
_GT_FILE = "1667522809_J_0001_0002.art.gt.xml"


@pytest.fixture(name="evaluated_single_pair", scope="module")
def _evaluated_single_pair(tmp_path_factory):
    """Run eval_all once for the canonical candidate/gt pair.

    All tests that only differ in aggregation strategy can reuse these
    pre-computed entries instead of repeating the expensive XML parse.
    """
    base = tmp_path_factory.mktemp("agg_eval_data")
    eval_domain = base / "candidate" / "ger_frk"
    gt_domain = base / "groundtruth" / "ger_frk"
    eval_domain.mkdir(parents=True)
    gt_domain.mkdir(parents=True)

    shutil.copy(
        TEST_RES_DIR / "candidate" / "frk_alto" / _CANDIDATE_FILE,
        eval_domain / _CANDIDATE_FILE,
    )
    shutil.copy(
        TEST_RES_DIR / "groundtruth" / "page" / _GT_FILE,
        gt_domain / _GT_FILE,
    )

    evaluator = digev.Evaluator(eval_domain)
    evaluator.metrics = [digem.MetricChars()]
    evaluator.domain_reference = gt_domain

    candidates = digev_main.gather_candidates(eval_domain)
    for entry in candidates:
        gt = digev_main.find_groundtruth(entry, gt_domain)
        if gt:
            entry.path_groundtruth = gt
            entry.align_domains()

    gt_entries = [c for c in candidates if c.path_groundtruth]
    evaluator.eval_all(gt_entries, sequential=True)

    return {
        "eval_domain": eval_domain,
        "gt_domain": gt_domain,
        "entries": evaluator.evaluation_entries,
        "metrics": evaluator.metrics,
    }


def _make_evaluator(data: dict) -> digev.Evaluator:
    """Build a fresh Evaluator seeded from the shared fixture."""
    ev = digev.Evaluator(data["eval_domain"])
    ev.metrics = list(data["metrics"])
    ev.domain_reference = data["gt_domain"]
    ev.evaluation_entries = copy.deepcopy(data["entries"])
    return ev


def test_aggregation_dimension_basic():
    """Test basic AggregationDimension creation"""

    # arrange
    def simple_extractor(entry):
        return "test_value"

    # act
    dimension = digev.AggregationDimension("test_dim", simple_extractor)

    # assert
    assert dimension.name == "test_dim"
    assert dimension.extractor is not None


def test_type_extractor():
    """Test TypeExtractor with EvalEntry"""
    # arrange
    entry = digev.EvalEntry(Path("/test/path.xml"))
    entry.gt_type = "article"
    extractor = digev.TypeExtractor()

    # act
    result = extractor(entry)

    # assert
    assert result == "article"


def test_type_extractor_not_set():
    """Test TypeExtractor returns None when type is not set"""
    # arrange
    entry = digev.EvalEntry(Path("/test/path.xml"))
    entry.gt_type = "n.a."
    extractor = digev.TypeExtractor()

    # act
    result = extractor(entry)

    # assert
    assert result is None


def test_custom_metadata_extractor():
    """Test CustomMetadataExtractor with tags"""
    # arrange
    entry = digev.EvalEntry(Path("/test/path.xml"))
    entry.tags = {"engine": "tesseract", "version": "5.0"}
    extractor = digev.CustomMetadataExtractor("engine")

    # act
    result = extractor(entry)

    # assert
    assert result == "tesseract"


def test_custom_metadata_extractor_missing_key():
    """Test CustomMetadataExtractor with missing key returns default"""
    # arrange
    entry = digev.EvalEntry(Path("/test/path.xml"))
    entry.tags = {"engine": "tesseract"}
    extractor = digev.CustomMetadataExtractor("version", default="unknown")

    # act
    result = extractor(entry)

    # assert
    assert result == "unknown"


def test_filename_pattern_extractor():
    """Test FilenamePatternExtractor extracts date from filename"""
    # arrange
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    extractor = digev.FilenamePatternExtractor(r"(\d{10})")

    # act
    result = extractor(entry)

    # assert
    assert result == "1667522809"


def test_directory_hierarchy_extractor():
    """Test DirectoryHierarchyExtractor with domain directories"""
    # arrange
    entry = digev.EvalEntry(Path("/test/path.xml"))
    entry.domain_directories = ["ger_frk", "article"]
    extractor = digev.DirectoryHierarchyExtractor()

    # act
    result = extractor(entry)

    # assert
    assert result == ["ger_frk", "article"]


def test_directory_hierarchy_extractor_specific_level():
    """Test DirectoryHierarchyExtractor with specific level"""
    # arrange
    entry = digev.EvalEntry(Path("/test/path.xml"))
    entry.domain_directories = ["ger_frk", "article", "1867"]
    extractor = digev.DirectoryHierarchyExtractor(level=-1)

    # act
    result = extractor(entry)

    # assert
    assert result == "ger_frk"


def test_aggregation_strategy_single_dimension():
    """Test AggregationStrategy with single dimension"""
    # arrange
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry.gt_type = "article"

    metric = digem.MetricChars()
    metric._label = "Cs"

    dimension = digev.AggregationDimension("type", digev.TypeExtractor())
    strategy = digev.AggregationStrategy([dimension])

    # act
    keys = strategy.generate_keys(entry, metric)

    # assert
    assert len(keys) == 1
    assert "Cs@type:article" in keys


def test_aggregation_strategy_multiple_dimensions():
    """Test AggregationStrategy with multiple dimensions"""
    # arrange
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry.gt_type = "article"
    entry.tags = {"engine": "tesseract"}

    metric = digem.MetricChars()
    metric._label = "Cs"

    dimensions = [
        digev.AggregationDimension("type", digev.TypeExtractor()),
        digev.AggregationDimension("engine", digev.CustomMetadataExtractor("engine")),
    ]
    strategy = digev.AggregationStrategy(dimensions)

    # act
    keys = strategy.generate_keys(entry, metric)

    # assert
    assert len(keys) == 2
    assert "Cs@type:article" in keys
    assert "Cs@engine:tesseract" in keys


def test_aggregation_strategy_hierarchical():
    """Test AggregationStrategy with hierarchical mode"""
    # arrange
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry.gt_type = "article"
    entry.tags = {"engine": "tesseract"}

    metric = digem.MetricChars()
    metric._label = "Cs"

    dimensions = [
        digev.AggregationDimension("type", digev.TypeExtractor()),
        digev.AggregationDimension("engine", digev.CustomMetadataExtractor("engine")),
    ]
    strategy = digev.AggregationStrategy(dimensions, hierarchical=True)

    # act
    keys = strategy.generate_keys(entry, metric)

    # assert
    assert len(keys) == 3
    assert "Cs@type:article" in keys
    assert "Cs@engine:tesseract" in keys
    assert "Cs@type:article/engine:tesseract" in keys


def test_eval_entry_tags():
    """Test EvalEntry has tags dictionary"""
    # arrange & act
    entry = digev.EvalEntry(Path("/test/path.xml"))

    # assert
    assert hasattr(entry, "tags")
    assert isinstance(entry.tags, dict)
    assert len(entry.tags) == 0


def test_aggregate_generic_with_type_strategy(evaluated_single_pair):
    """Test aggregate_generic with type-based strategy"""
    # arrange - reuse pre-evaluated entries
    evaluator = _make_evaluator(evaluated_single_pair)

    type_strategy = digev.AggregationStrategy([digev.AggregationDimension("type", digev.TypeExtractor())])

    # act
    evaluator.aggregate_generic(type_strategy)

    # assert
    assert len(evaluator.evaluation_map) > 0
    keys = list(evaluator.evaluation_map.keys())
    assert any("type:article" in key for key in keys)


def test_aggregate_generic_with_custom_metadata(evaluated_single_pair):
    """Test aggregate_generic with custom metadata strategy"""
    # arrange - reuse pre-evaluated entries, add metadata tags
    evaluator = _make_evaluator(evaluated_single_pair)
    for entry in evaluator.evaluation_entries:
        entry.tags["ocr_engine"] = "tesseract"
        entry.tags["version"] = "5.0"

    metadata_strategy = digev.AggregationStrategy(
        [digev.AggregationDimension("engine", digev.CustomMetadataExtractor("ocr_engine"))]
    )

    # act
    evaluator.aggregate_generic(metadata_strategy)

    # assert
    assert len(evaluator.evaluation_map) > 0
    keys = list(evaluator.evaluation_map.keys())
    assert any("engine:tesseract" in key for key in keys)


def test_aggregate_generic_default_strategy(evaluated_single_pair):
    """Test aggregate_generic with default (backward compatible) strategy"""
    # arrange - reuse pre-evaluated entries
    evaluator = _make_evaluator(evaluated_single_pair)

    # act - use default strategy (None)
    evaluator.aggregate_generic(strategy=None)

    # assert
    assert len(evaluator.evaluation_map) > 0
    keys = list(evaluator.evaluation_map.keys())
    assert any("ger_frk" in key for key in keys)


def test_backward_compatibility_aggregate_vs_aggregate_generic(evaluated_single_pair):
    """Test that aggregate_generic with default strategy produces similar results to aggregate"""
    # arrange - both evaluators seeded from the same pre-evaluated entries
    evaluator1 = _make_evaluator(evaluated_single_pair)
    evaluator2 = _make_evaluator(evaluated_single_pair)

    # act
    evaluator1.aggregate(by_type=False)
    evaluator2.aggregate_generic(strategy=None)

    # assert - both should have created evaluation maps
    assert len(evaluator1.evaluation_map) > 0
    assert len(evaluator2.evaluation_map) > 0

    keys1 = set(evaluator1.evaluation_map.keys())
    keys2 = set(evaluator2.evaluation_map.keys())
    assert any("ger_frk" in key for key in keys1)
    assert any("ger_frk" in key for key in keys2)


def test_mets_mods_extractor_language():
    """Test METSModsExtractor extracts language from METS/MODS"""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry.path_groundtruth = Path("/test/1667522809_J_0001_0002.art.gt.xml")

    extractor = digev.METSModsExtractor(
        mets_file_path=mets_path,
        xpath_expression=".//mods:language/mods:languageTerm[@type='code']",
    )

    # act
    result = extractor(entry)

    # assert
    assert result == "ger"


def test_mets_mods_extractor_genre():
    """Test METSModsExtractor extracts genre from METS/MODS"""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry.path_groundtruth = Path("/test/1667522809_J_0001_0002.art.gt.xml")

    extractor = digev.METSModsExtractor(mets_file_path=mets_path, xpath_expression=".//mods:genre")

    # act
    result = extractor(entry)

    # assert
    assert result == "article"


def test_mets_mods_extractor_date():
    """Test METSModsExtractor extracts date from METS/MODS"""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry.path_groundtruth = Path("/test/1667522809_J_0001_0002.art.gt.xml")

    extractor = digev.METSModsExtractor(mets_file_path=mets_path, xpath_expression=".//mods:originInfo/mods:dateIssued")

    # act
    result = extractor(entry)

    # assert
    assert result == "1867"


def test_mets_mods_extractor_publisher():
    """Test METSModsExtractor extracts publisher from METS/MODS"""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    entry = digev.EvalEntry(Path("/test/test_announcement.xml"))
    entry.path_groundtruth = Path("/test/test_announcement.ann.gt.xml")

    extractor = digev.METSModsExtractor(mets_file_path=mets_path, xpath_expression=".//mods:originInfo/mods:publisher")

    # act
    result = extractor(entry)

    # assert
    assert result == "Test Publisher 2"


def test_mets_mods_extractor_no_groundtruth():
    """Test METSModsExtractor returns None when no groundtruth path"""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    # No groundtruth path set

    extractor = digev.METSModsExtractor(
        mets_file_path=mets_path,
        xpath_expression=".//mods:language/mods:languageTerm[@type='code']",
    )

    # act
    result = extractor(entry)

    # assert
    assert result is None


def test_mets_mods_extractor_file_not_in_mets():
    """Test METSModsExtractor returns None for file not in METS"""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    entry = digev.EvalEntry(Path("/test/nonexistent_file.xml"))
    entry.path_groundtruth = Path("/test/nonexistent_file.gt.xml")

    extractor = digev.METSModsExtractor(
        mets_file_path=mets_path,
        xpath_expression=".//mods:language/mods:languageTerm[@type='code']",
    )

    # act
    result = extractor(entry)

    # assert
    assert result is None


def test_mets_mods_extractor_caching():
    """Test METSModsExtractor caches parsed METS file"""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    extractor = digev.METSModsExtractor(
        mets_file_path=mets_path,
        xpath_expression=".//mods:language/mods:languageTerm[@type='code']",
        cache_parsed=True,
    )

    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry.path_groundtruth = Path("/test/1667522809_J_0001_0002.art.gt.xml")

    # act - first call parses file
    result1 = extractor(entry)
    assert extractor._parsed_tree is not None
    assert extractor._file_to_mods_map is not None

    # act - second call should use cache
    result2 = extractor(entry)

    # assert - both calls return same result
    assert result1 == result2 == "ger"


def test_aggregate_generic_with_mets_mods_language(evaluated_single_pair, tmp_path):
    """Test aggregate_generic with METS/MODS language strategy"""

    # arrange - reuse pre-evaluated entries; only METS file copy is new
    mets_dst = tmp_path / "test_mets.xml"
    shutil.copy(TEST_RES_DIR / "test_mets.xml", mets_dst)

    evaluator = _make_evaluator(evaluated_single_pair)

    language_strategy = digev.AggregationStrategy(
        [
            digev.AggregationDimension(
                "language",
                digev.METSModsExtractor(
                    mets_file_path=mets_dst,
                    xpath_expression=".//mods:language/mods:languageTerm[@type='code']",
                ),
            )
        ]
    )

    # act
    evaluator.aggregate_generic(language_strategy)

    # assert
    assert len(evaluator.evaluation_map) > 0
    keys = list(evaluator.evaluation_map.keys())
    assert any("language:ger" in key for key in keys)


def test_mets_mods_extractor_multiple_files():
    """Test METSModsExtractor correctly distinguishes between multiple files"""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"

    # First file - article in German
    entry1 = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry1.path_groundtruth = Path("/test/1667522809_J_0001_0002.art.gt.xml")

    # Second file - announcement in English
    entry2 = digev.EvalEntry(Path("/test/test_announcement.xml"))
    entry2.path_groundtruth = Path("/test/test_announcement.ann.gt.xml")

    language_extractor = digev.METSModsExtractor(
        mets_file_path=mets_path,
        xpath_expression=".//mods:language/mods:languageTerm[@type='code']",
    )

    genre_extractor = digev.METSModsExtractor(mets_file_path=mets_path, xpath_expression=".//mods:genre")

    # act
    lang1 = language_extractor(entry1)
    lang2 = language_extractor(entry2)
    genre1 = genre_extractor(entry1)
    genre2 = genre_extractor(entry2)

    # assert
    assert lang1 == "ger"
    assert lang2 == "eng"
    assert genre1 == "article"
    assert genre2 == "announcement"


def test_mets_mods_extractor_joins_multiple_xpath_results(tmp_path):
    """Multiple XPath hits within one mods section are joined with '+'."""

    mets_path = tmp_path / "multi_lang_mets.xml"
    mets_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/"
           xmlns:mods="http://www.loc.gov/mods/v3"
           xmlns:xlink="http://www.w3.org/1999/xlink">
  <mets:dmdSec ID="md_multi">
    <mets:mdWrap MDTYPE="MODS"><mets:xmlData><mods:mods>
      <mods:language><mods:languageTerm type="code">ara</mods:languageTerm></mods:language>
      <mods:language><mods:languageTerm type="code">eng</mods:languageTerm></mods:language>
    </mods:mods></mets:xmlData></mets:mdWrap>
  </mets:dmdSec>
  <mets:fileSec>
    <mets:fileGrp USE="OCR-D-GT-FULLTEXT" DMDID="md_multi">
      <mets:file ID="F1"><mets:FLocat xlink:href="GT-PAGE/page.xml"/></mets:file>
    </mets:fileGrp>
  </mets:fileSec>
</mets:mets>
""",
        encoding="utf-8",
    )

    entry = digev.EvalEntry(Path("/test/page.xml"))
    entry.path_groundtruth = Path("/test/page.xml")

    extractor = digev.METSModsExtractor(
        mets_file_path=mets_path,
        xpath_expression=".//mods:language/mods:languageTerm[@type='code']",
    )

    result = extractor(entry)

    assert result == "ara+eng"


def test_mets_mods_extractor_prefers_structlink_mapping(tmp_path):
    """Use structLink logical mapping when fileGrp DMDID is overly broad."""

    # arrange
    mets_path = tmp_path / "structlink_test_mets.xml"
    mets_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/"
           xmlns:mods="http://www.loc.gov/mods/v3"
           xmlns:xlink="http://www.w3.org/1999/xlink">
  <mets:dmdSec ID="md_generic">
    <mets:mdWrap MDTYPE="MODS"><mets:xmlData><mods:mods><mods:language><mods:languageTerm type="code">lat</mods:languageTerm></mods:language></mods:mods></mets:xmlData></mets:mdWrap>
  </mets:dmdSec>
  <mets:dmdSec ID="md_a">
    <mets:mdWrap MDTYPE="MODS"><mets:xmlData><mods:mods><mods:language><mods:languageTerm type="code">ger</mods:languageTerm></mods:language></mods:mods></mets:xmlData></mets:mdWrap>
  </mets:dmdSec>
  <mets:dmdSec ID="md_b">
    <mets:mdWrap MDTYPE="MODS"><mets:xmlData><mods:mods><mods:language><mods:languageTerm type="code">eng</mods:languageTerm></mods:language></mods:mods></mets:xmlData></mets:mdWrap>
  </mets:dmdSec>
  <mets:fileSec>
    <mets:fileGrp USE="OCR-D-GT-FULLTEXT" DMDID="md_generic">
      <mets:file ID="OCR-D-GT-FULLTEXT-1"><mets:FLocat xlink:href="GT-PAGE/file_a.xml"/></mets:file>
      <mets:file ID="OCR-D-GT-FULLTEXT-2"><mets:FLocat xlink:href="GT-PAGE/file_b.xml"/></mets:file>
    </mets:fileGrp>
  </mets:fileSec>
  <mets:structMap TYPE="PHYSICAL">
    <mets:div ID="physroot">
      <mets:div ID="phys1"><mets:fptr FILEID="OCR-D-GT-FULLTEXT-1"/></mets:div>
      <mets:div ID="phys2"><mets:fptr FILEID="OCR-D-GT-FULLTEXT-2"/></mets:div>
    </mets:div>
  </mets:structMap>
  <mets:structMap TYPE="LOGICAL">
    <mets:div ID="logroot">
      <mets:div ID="log1" DMDID="md_a"/>
      <mets:div ID="log2" DMDID="md_b"/>
    </mets:div>
  </mets:structMap>
  <mets:structLink>
    <mets:smLink xlink:from="log1" xlink:to="phys1"/>
    <mets:smLink xlink:from="log2" xlink:to="phys2"/>
  </mets:structLink>
</mets:mets>
""",
        encoding="utf-8",
    )

    entry_a = digev.EvalEntry(Path("/test/file_a.xml"))
    entry_a.path_groundtruth = Path("/test/file_a.xml")
    entry_b = digev.EvalEntry(Path("/test/file_b.xml"))
    entry_b.path_groundtruth = Path("/test/file_b.xml")

    extractor = digev.METSModsExtractor(
        mets_file_path=mets_path,
        xpath_expression=".//mods:language/mods:languageTerm[@type='code']",
    )

    # act
    lang_a = extractor(entry_a)
    lang_b = extractor(entry_b)

    # assert
    assert lang_a == "ger"
    assert lang_b == "eng"


# ---------------------------------------------------------------------------
# decade_transform
# ---------------------------------------------------------------------------


def test_decade_transform_standard_year():
    """decade_transform maps a year to its decade bucket."""
    assert digev.decade_transform("1867") == "1860s"


def test_decade_transform_decade_boundary():
    """A year exactly on the decade boundary maps to the same decade."""
    assert digev.decade_transform("1900") == "1900s"


def test_decade_transform_century_boundary():
    """1999 maps to 1990s, not 2000s."""
    assert digev.decade_transform("1999") == "1990s"


def test_decade_transform_non_numeric_returns_none():
    """decade_transform returns None for non-numeric input."""
    assert digev.decade_transform("unknown") is None


def test_decade_transform_none_input_returns_none():
    """decade_transform returns None when passed None."""
    assert digev.decade_transform(None) is None


def test_decade_transform_uses_first_four_chars():
    """decade_transform only reads the first 4 characters (e.g. partial ISO dates)."""
    assert digev.decade_transform("1867-01-01") == "1860s"


# ---------------------------------------------------------------------------
# century_transform
# ---------------------------------------------------------------------------


def test_century_transform_19th():
    """century_transform maps a year in the 1800s to '19th'."""
    assert digev.century_transform("1867") == "19th"


def test_century_transform_20th():
    """century_transform maps a year in the 1900s to '20th'."""
    assert digev.century_transform("1999") == "20th"


def test_century_transform_21st():
    """century_transform maps a year in the 2000s to '21st'."""
    assert digev.century_transform("2026") == "21st"


def test_century_transform_boundary_1900():
    """1900 is in the 20th century."""
    assert digev.century_transform("1900") == "20th"


def test_century_transform_11th_century_no_spurious_st():
    """11th century uses 'th', not 'st' (covers the *11 edge case)."""
    assert digev.century_transform("1050") == "11th"


def test_century_transform_non_numeric_returns_none():
    """century_transform returns None for non-numeric input."""
    assert digev.century_transform("unknown") is None


def test_century_transform_none_input_returns_none():
    """century_transform returns None when passed None."""
    assert digev.century_transform(None) is None


# ---------------------------------------------------------------------------
# ValueTransformExtractor
# ---------------------------------------------------------------------------


def test_value_transform_extractor_applies_transform():
    """ValueTransformExtractor wraps an inner extractor and transforms its result."""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    entry = digev.EvalEntry(Path("/test/1667522809_J_0001_0002.xml"))
    entry.path_groundtruth = Path("/test/1667522809_J_0001_0002.art.gt.xml")

    inner = digev.METSModsExtractor(mets_file_path=mets_path, xpath_expression=".//mods:originInfo/mods:dateIssued")
    extractor = digev.ValueTransformExtractor(inner, digev.decade_transform)

    # act
    result = extractor(entry)

    # assert - raw date is "1867", decade is "1860s"
    assert result == "1860s"


def test_value_transform_extractor_none_from_inner_propagates():
    """ValueTransformExtractor returns None when the inner extractor returns None."""

    # arrange
    mets_path = TEST_RES_DIR / "test_mets.xml"
    entry = digev.EvalEntry(Path("/test/nonexistent_file.xml"))
    entry.path_groundtruth = Path("/test/nonexistent_file.gt.xml")

    inner = digev.METSModsExtractor(mets_file_path=mets_path, xpath_expression=".//mods:originInfo/mods:dateIssued")
    extractor = digev.ValueTransformExtractor(inner, digev.decade_transform)

    # act
    result = extractor(entry)

    # assert
    assert result is None


def test_value_transform_extractor_none_from_transform_propagates():
    """ValueTransformExtractor returns None when transform returns None."""

    # arrange - inner always returns a non-parseable string
    def always_invalid(_entry):
        return "not-a-year"

    extractor = digev.ValueTransformExtractor(always_invalid, digev.decade_transform)

    entry = digev.EvalEntry(Path("/test/file.xml"))

    # act
    result = extractor(entry)

    # assert
    assert result is None


# ---------------------------------------------------------------------------
# METSDivAttrExtractor
# ---------------------------------------------------------------------------

_METS_WITH_STRUCTMAP = """\
<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/"
           xmlns:xlink="http://www.w3.org/1999/xlink">
  <mets:fileSec>
    <mets:fileGrp USE="GT">
      <mets:file ID="F1"><mets:FLocat xlink:href="GT-PAGE/page_article.xml"/></mets:file>
      <mets:file ID="F2"><mets:FLocat xlink:href="GT-PAGE/page_chapter.xml"/></mets:file>
      <mets:file ID="F3"><mets:FLocat xlink:href="GT-PAGE/page_preface.xml"/></mets:file>
    </mets:fileGrp>
  </mets:fileSec>
  <mets:structMap TYPE="PHYSICAL">
    <mets:div ID="physroot">
      <mets:div ID="phys1"><mets:fptr FILEID="F1"/></mets:div>
      <mets:div ID="phys2"><mets:fptr FILEID="F2"/></mets:div>
      <mets:div ID="phys3"><mets:fptr FILEID="F3"/></mets:div>
    </mets:div>
  </mets:structMap>
  <mets:structMap TYPE="LOGICAL">
    <mets:div ID="logroot" TYPE="document" LABEL="Test Document">
      <mets:div ID="log1" TYPE="article" LABEL="First Article"/>
      <mets:div ID="log2" TYPE="chapter" LABEL="Second Chapter"/>
      <mets:div ID="log3" TYPE="preface" LABEL="Preface Section"/>
    </mets:div>
  </mets:structMap>
  <mets:structLink>
    <mets:smLink xlink:from="log1" xlink:to="phys1"/>
    <mets:smLink xlink:from="log2" xlink:to="phys2"/>
    <mets:smLink xlink:from="log3" xlink:to="phys3"/>
  </mets:structLink>
</mets:mets>
"""

_METS_WITH_DMDID_ONLY = """\
<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/"
           xmlns:xlink="http://www.w3.org/1999/xlink">
  <mets:fileSec>
    <mets:fileGrp USE="GT" DMDID="dmd1">
      <mets:file ID="F1"><mets:FLocat xlink:href="GT-PAGE/page_fallback.xml"/></mets:file>
    </mets:fileGrp>
  </mets:fileSec>
  <mets:structMap TYPE="LOGICAL">
    <mets:div ID="logroot" TYPE="document">
      <mets:div ID="log1" TYPE="section" LABEL="Fallback Section" DMDID="dmd1"/>
    </mets:div>
  </mets:structMap>
</mets:mets>
"""


def test_mets_div_attr_extractor_type_via_structlink(tmp_path):
    """METSDivAttrExtractor extracts TYPE from logical div via structLink."""

    mets_path = tmp_path / "test.xml"
    mets_path.write_text(_METS_WITH_STRUCTMAP, encoding="utf-8")

    extractor = digev.METSDivAttrExtractor(mets_file_path=mets_path, attribute="TYPE")

    entry = digev.EvalEntry(Path("/test/page_article.xml"))
    entry.path_groundtruth = Path("/test/page_article.xml")

    assert extractor(entry) == "article"


def test_mets_div_attr_extractor_type_different_files(tmp_path):
    """METSDivAttrExtractor returns correct TYPE for each file."""

    mets_path = tmp_path / "test.xml"
    mets_path.write_text(_METS_WITH_STRUCTMAP, encoding="utf-8")

    extractor = digev.METSDivAttrExtractor(mets_file_path=mets_path, attribute="TYPE")

    e1 = digev.EvalEntry(Path("/test/page_article.xml"))
    e1.path_groundtruth = Path("/test/page_article.xml")
    e2 = digev.EvalEntry(Path("/test/page_chapter.xml"))
    e2.path_groundtruth = Path("/test/page_chapter.xml")
    e3 = digev.EvalEntry(Path("/test/page_preface.xml"))
    e3.path_groundtruth = Path("/test/page_preface.xml")

    assert extractor(e1) == "article"
    assert extractor(e2) == "chapter"
    assert extractor(e3) == "preface"


def test_mets_div_attr_extractor_label_via_structlink(tmp_path):
    """METSDivAttrExtractor extracts LABEL from logical div via structLink."""

    mets_path = tmp_path / "test.xml"
    mets_path.write_text(_METS_WITH_STRUCTMAP, encoding="utf-8")

    extractor = digev.METSDivAttrExtractor(mets_file_path=mets_path, attribute="LABEL")

    entry = digev.EvalEntry(Path("/test/page_chapter.xml"))
    entry.path_groundtruth = Path("/test/page_chapter.xml")

    assert extractor(entry) == "Second Chapter"


def test_mets_div_attr_extractor_fallback_dmdid(tmp_path):
    """METSDivAttrExtractor falls back to DMDID when no structLink is present."""

    mets_path = tmp_path / "test.xml"
    mets_path.write_text(_METS_WITH_DMDID_ONLY, encoding="utf-8")

    extractor = digev.METSDivAttrExtractor(mets_file_path=mets_path, attribute="TYPE")

    entry = digev.EvalEntry(Path("/test/page_fallback.xml"))
    entry.path_groundtruth = Path("/test/page_fallback.xml")

    assert extractor(entry) == "section"


def test_mets_div_attr_extractor_no_groundtruth_returns_none(tmp_path):
    """METSDivAttrExtractor returns None when entry has no groundtruth path."""

    mets_path = tmp_path / "test.xml"
    mets_path.write_text(_METS_WITH_STRUCTMAP, encoding="utf-8")

    extractor = digev.METSDivAttrExtractor(mets_file_path=mets_path, attribute="TYPE")

    entry = digev.EvalEntry(Path("/test/page_article.xml"))
    # no path_groundtruth set

    assert extractor(entry) is None


def test_mets_div_attr_extractor_unknown_file_returns_none(tmp_path):
    """METSDivAttrExtractor returns None for a file not referenced in the METS."""

    mets_path = tmp_path / "test.xml"
    mets_path.write_text(_METS_WITH_STRUCTMAP, encoding="utf-8")

    extractor = digev.METSDivAttrExtractor(mets_file_path=mets_path, attribute="TYPE")

    entry = digev.EvalEntry(Path("/test/nonexistent.xml"))
    entry.path_groundtruth = Path("/test/nonexistent.xml")

    assert extractor(entry) is None
