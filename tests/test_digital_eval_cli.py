# -*- coding: utf-8 -*-
"""OCR Evaluation Test Module"""
# pylint: disable=protected-access

import shutil

from pathlib import Path

import pytest

import ocr_util.eval.cli as dig
import ocr_util.eval.preprocessing as dipre

from .conftest import TEST_RES_DIR


_DOMAIN_LABEL = 'ger_frk'


@pytest.fixture(name='cli_paths', scope='module')
def _create_cli_paths(tmp_path_factory):
    """Prepare reusable candidate/reference fixtures for CLI tests."""

    src_candidates = TEST_RES_DIR / 'candidate' / 'frk_alto'
    src_reference = TEST_RES_DIR / 'groundtruth' / 'page'
    src_mets = TEST_RES_DIR / 'test_mets.xml'
    src_candidate_file = src_candidates / '1667522809_J_0001_0002.xml'

    base_dir = tmp_path_factory.mktemp('cli_test_data')

    candidate_dir = base_dir / 'candidate' / _DOMAIN_LABEL
    reference_dir = base_dir / 'reference' / _DOMAIN_LABEL
    reference_gt_page_dir = base_dir / 'reference' / _DOMAIN_LABEL / 'GT-PAGE'
    single_candidate_dir = base_dir / 'single_candidate' / _DOMAIN_LABEL
    single_candidate_file = single_candidate_dir / '1667522809_J_0001_0002.xml'
    single_reference_dir = base_dir / 'single_reference' / _DOMAIN_LABEL
    single_reference_file = single_reference_dir / '1667522809_J_0001_0002.art.gt.xml'
    mets_file = base_dir / 'reference' / 'test_mets.xml'

    shutil.copytree(src_candidates, candidate_dir)
    shutil.copytree(src_reference, reference_dir)
    shutil.copytree(src_reference, reference_gt_page_dir)

    single_candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_candidate_file, single_candidate_file)
    shutil.copytree(src_reference, single_reference_dir)

    shutil.copy(src_mets, mets_file)

    return {
        'candidate_dir': candidate_dir,
        'reference_dir': reference_dir,
        'reference_gt_page_dir': reference_gt_page_dir,
        'single_candidate_file': single_candidate_file,
        'single_reference_dir': single_reference_dir,
        'single_reference_file': single_reference_file,
        'mets_file': mets_file,
    }


@pytest.mark.parametrize(
    'utf8_norm, expected_norm_line, expected_metric_line',
    [
        (
            dig.DEFAULT_UTF8_NORM,
            "[DEBUG] text normalized using 'NFC' code points for 'Cs,Ls'",
            "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.20(5308), Ls:38.54(4383)(- 0.66)]",
        ),
        (
            dipre.UC_NORMALIZATION_NFKD,
            "[DEBUG] text normalized using 'NFKD' code points for 'Cs,Ls'",
            "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.11(5362), Ls:38.52(4437)(- 0.59)]",
        ),
    ],
)
def test_mwe_cli_norm_variants(
    cli_paths,
    capsys,
    utf8_norm,
    expected_norm_line,
    expected_metric_line,
):
    """Minimum working example CLI 
    to fix *real* outcomes when playing with
    metrics implementations

    Match five candidates from subdir 'ger_frk' with
    total 13 references of according gt-subdir to 
    creates 4 default evaluation results (Ls,Cs)
    (no reference for candiate 1667522809_J_0001_0256_corrupt.xml)
    """

    # arrange
    dig.VERBOSITY = 1
    dst_candidates = cli_paths['candidate_dir']
    dst_reference = cli_paths['reference_dir']

    # assert final path segments do match by name frk_alto == frk_alto
    assert _DOMAIN_LABEL == dst_candidates.name
    assert _DOMAIN_LABEL == dst_reference.name

    # act
    cli_args = {"candidates": dst_candidates, "reference": dst_reference,
                "metrics": dig.DEFAULT_OCR_METRICS,
                "verbosity": 1,
                "utf8": utf8_norm,
                "sequential": True}
    eval_results = dig.start_evaluation(cli_args)

    # assert
    assert len(eval_results) == 4
    captured = capsys.readouterr().out
    std_lines = captured.split('\n')
    assert len(std_lines) >= 13  # now includes aggregation strategy debug lines
    assert std_lines[0] == expected_norm_line
    assert str(std_lines[1]).startswith('[DEBUG] from "5" filtered "3" candidates')
    # Find the expected_metric_line (it will be further down due to new aggregation logging)
    assert any(expected_metric_line in line for line in std_lines)


def test_mwe_cli_data_resolving(cli_paths, capsys):
    """Minimum working example CLI 
    to inspect behavior for intermediate missmatches
    => OCR-D GT-PAGE directory
    """

    # arrange
    dig.VERBOSITY = 1
    dst_candidates = cli_paths['candidate_dir']
    dst_reference = cli_paths['reference_gt_page_dir']

    # assert final path segments do match by name frk_alto == frk_alto
    assert _DOMAIN_LABEL == dst_candidates.name

    # act
    cli_args = {"candidates": dst_candidates, "reference": dst_reference,
                "metrics": dig.DEFAULT_OCR_METRICS,
                "verbosity": 1,
                "utf8": dipre.UC_NORMALIZATION_NFKD,
                "sequential": True}
    eval_results = dig.start_evaluation(cli_args)

    # assert
    assert len(eval_results) == 4
    captured = capsys.readouterr().out
    std_lines = captured.split('\n')
    assert len(std_lines) >= 14  # now includes aggregation strategy debug lines
    assert std_lines[0] == "[WARN ] base 'ger_frk' and 'GT-PAGE' mismatch, aggregation might be inaccurate!"
    assert std_lines[1] == "[DEBUG] text normalized using 'NFKD' code points for 'Cs,Ls'"
    # Find the metric line (it will be in the output, exact position may vary due to aggregation logging)
    assert any("[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.11(5362), Ls:38.52(4437)(- 0.59)]" in line for line in std_lines)


def test_single_candidate_file_cli(cli_paths, capsys):
    """Test CLI with a single candidate file as argument
    
    Ensures that a single candidate file can be passed directly
    instead of a directory and is processed correctly.
    """

    # arrange
    dig.VERBOSITY = 1
    dst_candidate_file = cli_paths['single_candidate_file']
    dst_reference = cli_paths['single_reference_dir']

    # assert file exists
    assert dst_candidate_file.is_file()
    assert dst_reference.is_dir()

    # act - pass single file as candidates argument
    cli_args = {"candidates": dst_candidate_file, "reference": dst_reference,
                "metrics": dig.DEFAULT_OCR_METRICS,
                "verbosity": 1,
                "utf8": dig.DEFAULT_UTF8_NORM,
                "sequential": True}
    eval_results = dig.start_evaluation(cli_args)

    # assert - should process single file and match with reference
    # Note: aggregate() creates multiple results: one per metric per domain/type
    # With 2 metrics (Cs, Ls) and by_type=True, we get multiple aggregation results
    assert len(eval_results) > 0
    # Check that the specific file was processed by checking eval_keys
    eval_keys = [result.eval_key for result in eval_results]
    assert any('ger_frk' in key or '1667522809_J_0001_0002' in key for key in eval_keys)
    captured = capsys.readouterr().out
    std_lines = captured.split('\n')
    assert std_lines[0] == "[DEBUG] text normalized using 'NFC' code points for 'Cs,Ls'"
    # Verify the specific file appears in the output
    assert any('1667522809_J_0001_0002' in line for line in std_lines)


def test_single_candidate_and_reference_file_cli(cli_paths, capsys):
    """Evaluate an explicitly selected candidate/reference page pair."""

    candidate_file = cli_paths['single_candidate_file']
    reference_file = cli_paths['single_reference_file']
    assert candidate_file.is_file()
    assert reference_file.is_file()

    results = dig.start_evaluation({
        "candidates": candidate_file,
        "reference": reference_file,
        "metrics": "Cs",
        "verbosity": 2,
        "utf8": dig.DEFAULT_UTF8_NORM,
        "sequential": True,
    })

    assert len(results) > 0
    captured = capsys.readouterr().out
    assert str(candidate_file) in captured
    assert str(reference_file) in captured


def test_reference_file_requires_single_candidate_file(cli_paths):
    """Reject mapping one explicitly supplied reference file to many candidates."""

    with pytest.raises(SystemExit) as excinfo:
        dig.start_evaluation({
            "candidates": cli_paths['candidate_dir'],
            "reference": cli_paths['single_reference_file'],
            "metrics": "Cs",
            "verbosity": 0,
            "utf8": dig.DEFAULT_UTF8_NORM,
            "sequential": True,
        })

    assert excinfo.value.code == 1


def test_cli_with_mets_mods_aggregation(cli_paths, capsys):
    """Test CLI with METS/MODS aggregation parameters"""
    pytest.importorskip("lxml", reason="lxml required for METS/MODS extraction")
    
    # arrange
    dig.VERBOSITY = 1
    dst_candidates = cli_paths['candidate_dir']
    dst_reference = cli_paths['reference_dir']
    dst_mets = cli_paths['mets_file']
    
    # assert files exist
    assert _DOMAIN_LABEL == dst_candidates.name
    assert _DOMAIN_LABEL == dst_reference.name
    assert dst_mets.is_file()
    
    # act - use METS/MODS aggregation with language and genre dimensions
    cli_args = {
        "candidates": dst_candidates,
        "reference": dst_reference,
        "metrics": dig.DEFAULT_OCR_METRICS,
        "verbosity": 1,
        "utf8": dig.DEFAULT_UTF8_NORM,
        "sequential": True,
        "mets_file": str(dst_mets),
        "mods_dimensions": "language,genre"
    }
    eval_results = dig.start_evaluation(cli_args)
    
    # assert
    assert len(eval_results) > 0
    
    # Check that results are aggregated by MODS dimensions
    eval_keys = [result.eval_key for result in eval_results]
    # Should contain keys like "Cs@language:ger" or "Cs@genre:article"
    assert any('language:' in key for key in eval_keys) or any('genre:' in key for key in eval_keys)
    
    # Check debug output
    captured = capsys.readouterr().out
    # New unified aggregation system uses different debug message format
    assert "Added aggregation dimension" in captured or "Converting legacy --mods-dimensions" in captured


def test_cli_with_mets_file_only_warning(cli_paths):
    """Test CLI shows warning when METS file provided without dimensions"""
    
    # arrange
    dig.VERBOSITY = 1
    dst_candidates = cli_paths['candidate_dir']
    dst_reference = cli_paths['reference_dir']
    dst_mets = cli_paths['mets_file']
    
    # act - provide METS file but no dimensions (should use default aggregation)
    cli_args = {
        "candidates": dst_candidates,
        "reference": dst_reference,
        "metrics": dig.DEFAULT_OCR_METRICS,
        "verbosity": 1,
        "utf8": dig.DEFAULT_UTF8_NORM,
        "sequential": True,
        "mets_file": str(dst_mets),
        # No mods_dimensions provided
    }
    eval_results = dig.start_evaluation(cli_args)
    
    # assert - should still work with default aggregation
    assert len(eval_results) > 0


def test_cli_with_mets_invalid_mods_dimension_fails_early(cli_paths):
    """Fail fast when --aggregate-by requests unknown mods dimension with METS file."""

    dig.VERBOSITY = 1
    dst_candidates = cli_paths['candidate_dir']
    dst_reference = cli_paths['reference_dir']
    dst_mets = cli_paths['mets_file']

    cli_args = {
        "candidates": dst_candidates,
        "reference": dst_reference,
        "metrics": dig.DEFAULT_OCR_METRICS,
        "verbosity": 1,
        "utf8": dig.DEFAULT_UTF8_NORM,
        "sequential": True,
        "mets_file": str(dst_mets),
        "aggregate_by": "mods:does_not_exist",
    }

    with pytest.raises(SystemExit) as excinfo:
        dig.start_evaluation(cli_args)

    assert excinfo.value.code == 1


def test_cli_with_legacy_invalid_mods_dimension_fails_early(cli_paths):
    """Fail fast for invalid legacy --mods-dimensions with METS file."""

    dig.VERBOSITY = 1
    dst_candidates = cli_paths['candidate_dir']
    dst_reference = cli_paths['reference_dir']
    dst_mets = cli_paths['mets_file']

    cli_args = {
        "candidates": dst_candidates,
        "reference": dst_reference,
        "metrics": dig.DEFAULT_OCR_METRICS,
        "verbosity": 1,
        "utf8": dig.DEFAULT_UTF8_NORM,
        "sequential": True,
        "mets_file": str(dst_mets),
        "mods_dimensions": "language,does_not_exist",
    }

    with pytest.raises(SystemExit) as excinfo:
        dig.start_evaluation(cli_args)

    assert excinfo.value.code == 1


def test_cli_forwards_weighted_mean_flag(cli_paths, monkeypatch):
    """CLI forwards weighted_mean flag to Evaluator constructor."""

    observed = {}
    original_evaluator = dig.digev.Evaluator

    def _capture_evaluator(*args, **kwargs):
        observed["weighted_mean"] = kwargs.get("weighted_mean", None)
        return original_evaluator(*args, **kwargs)

    monkeypatch.setattr(dig.digev, "Evaluator", _capture_evaluator)

    cli_args = {
        "candidates": cli_paths["candidate_dir"],
        "reference": cli_paths["reference_dir"],
        "metrics": "Cs",
        "verbosity": 0,
        "utf8": dig.DEFAULT_UTF8_NORM,
        "sequential": True,
        "weighted_mean": True,
    }
    results = dig.start_evaluation(cli_args)

    assert len(results) > 0
    assert observed["weighted_mean"] is True


def test_start_parser_exposes_weighted_mean_flag(monkeypatch):
    """Public CLI parser accepts --weighted-mean and forwards it in parsed args."""

    captured = {}

    def _capture_start_evaluation(args):
        captured.update(args)
        return []

    monkeypatch.setattr(dig, "start_evaluation", _capture_start_evaluation)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ocr-util-eval",
            "dummy_candidates",
            "--weighted-mean",
        ],
    )

    dig.start()

    assert captured["candidates"] == "dummy_candidates"
    assert captured["weighted_mean"] is True


def test_top_level_ocr_eval_parser_exposes_weighted_mean(monkeypatch):
    """Top-level `ocr eval` parser accepts and forwards --weighted-mean."""

    import ocr_util.cli as ou_cli

    captured = {}

    def _capture_start_evaluation(args):
        captured.update(args)
        return []

    monkeypatch.setattr(ou_cli.eval_cli, "start_evaluation", _capture_start_evaluation)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ocr",
            "eval",
            "dummy_candidates",
            "--weighted-mean",
        ],
    )

    ou_cli.start()

    assert captured["candidates"] == "dummy_candidates"
    assert captured["weighted_mean"] is True


def test_filter_value_matches_exact_single_value():
    """Single-value filters require exact value match."""

    assert dig._filter_value_matches("ger", "ger") is True
    assert dig._filter_value_matches("ger+eng", "ger") is False
    assert dig._filter_value_matches("eng", "ger") is False


def test_filter_value_matches_set_containment_for_multi_value():
    """Multi-value filter requires expected value-set to be contained in extracted set."""

    assert dig._filter_value_matches("eng+ger", "ger+eng") is True
    assert dig._filter_value_matches("eng+ger+lat", "ger+eng") is True
    assert dig._filter_value_matches("ger", "ger+eng") is False


def test_apply_entry_filter_warns_and_discards_missing_metadata(capsys):
    """Entries missing the filter criterion are warned and discarded."""

    entry_ok = dig.digev.EvalEntry(Path("ok.xml"))
    entry_ok.tags["lang"] = "ger"

    entry_wrong = dig.digev.EvalEntry(Path("wrong.xml"))
    entry_wrong.tags["lang"] = "eng"

    entry_missing = dig.digev.EvalEntry(Path("missing.xml"))

    entries = [entry_ok, entry_wrong, entry_missing]
    filter_spec = (
        "metadata_lang",
        dig.digev.CustomMetadataExtractor("lang"),
        "ger",
    )

    kept = dig._apply_entry_filter(entries, filter_spec, verbosity=1)

    assert kept == [entry_ok]
    captured = capsys.readouterr().out
    assert "Discarded 1 entries with missing filter criterion 'metadata_lang'" in captured


def test_cli_filter_by_and_aggregate_by_with_mets(cli_paths):
    """CLI supports single pre-filter by MODS metadata followed by aggregation."""
    pytest.importorskip("lxml", reason="lxml required for METS/MODS extraction")

    cli_args = {
        "candidates": cli_paths["candidate_dir"],
        "reference": cli_paths["reference_dir"],
        "metrics": "Cs",
        "verbosity": 0,
        "utf8": dig.DEFAULT_UTF8_NORM,
        "sequential": True,
        "mets_file": str(cli_paths["mets_file"]),
        "filter_by": "mods:language=ger",
        "aggregate_by": "mods:dateIssued:century",
    }

    results = dig.start_evaluation(cli_args)

    assert len(results) > 0
    assert all("mods_dateIssued_century" in r.eval_key for r in results)


def test_cli_filter_by_multilanguage_set_containment_end_to_end(tmp_path):
    """End-to-end: multi-language filter keeps only entries containing all expected languages."""
    pytest.importorskip("lxml", reason="lxml required for METS/MODS extraction")

    src_candidates = TEST_RES_DIR / "candidate" / "frk_alto"
    src_reference = TEST_RES_DIR / "groundtruth" / "page"

    candidate_dir = tmp_path / "candidate" / _DOMAIN_LABEL
    reference_dir = tmp_path / "reference" / _DOMAIN_LABEL
    candidate_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    # Two real candidate/GT pairs from test resources.
    cand_a = "1667522809_J_0001_0002.xml"
    gt_a = "1667522809_J_0001_0002.art.gt.xml"
    cand_b = "1667522809_J_0001_0768.xml"
    gt_b = "1667522809_J_0001_0768.art.gt.xml"

    shutil.copy(src_candidates / cand_a, candidate_dir / cand_a)
    shutil.copy(src_reference / gt_a, reference_dir / gt_a)
    shutil.copy(src_candidates / cand_b, candidate_dir / cand_b)
    shutil.copy(src_reference / gt_b, reference_dir / gt_b)

    # Entry A has two languages (ger+eng) and 19th century publication.
    # Entry B has only one language (ger) and 20th century publication.
    mets_file = tmp_path / "reference" / "test_mets_multilang.xml"
    mets_file.write_text(
    f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<mets:mets xmlns:mets=\"http://www.loc.gov/METS/\"
                     xmlns:mods=\"http://www.loc.gov/mods/v3\"
                     xmlns:xlink=\"http://www.w3.org/1999/xlink\">
    <mets:dmdSec ID=\"md_multi\">
        <mets:mdWrap MDTYPE=\"MODS\"><mets:xmlData><mods:mods>
            <mods:language><mods:languageTerm type=\"code\">ger</mods:languageTerm></mods:language>
            <mods:language><mods:languageTerm type=\"code\">eng</mods:languageTerm></mods:language>
            <mods:originInfo><mods:dateIssued>1867</mods:dateIssued></mods:originInfo>
        </mods:mods></mets:xmlData></mets:mdWrap>
    </mets:dmdSec>
    <mets:dmdSec ID=\"md_single\">
        <mets:mdWrap MDTYPE=\"MODS\"><mets:xmlData><mods:mods>
            <mods:language><mods:languageTerm type=\"code\">ger</mods:languageTerm></mods:language>
            <mods:originInfo><mods:dateIssued>1901</mods:dateIssued></mods:originInfo>
        </mods:mods></mets:xmlData></mets:mdWrap>
    </mets:dmdSec>
    <mets:fileSec>
        <mets:fileGrp USE=\"OCR-D-GT-FULLTEXT\">
            <mets:file ID=\"F1\" DMDID=\"md_multi\"><mets:FLocat xlink:href=\"{gt_a}\"/></mets:file>
            <mets:file ID=\"F2\" DMDID=\"md_single\"><mets:FLocat xlink:href=\"{gt_b}\"/></mets:file>
        </mets:fileGrp>
    </mets:fileSec>
</mets:mets>
""",
        encoding="utf-8",
    )

    cli_args = {
        "candidates": candidate_dir,
        "reference": reference_dir,
        "metrics": "Cs",
        "verbosity": 1,
        "utf8": dig.DEFAULT_UTF8_NORM,
        "sequential": True,
        "mets_file": str(mets_file),
        "filter_by": "mods:language=ger+eng",
        "aggregate_by": "mods:dateIssued:century",
    }

    results = dig.start_evaluation(cli_args)

    assert len(results) > 0
    keys = [r.eval_key for r in results]
    assert all("mods_dateIssued_century:19th" in key for key in keys)
    assert all("20th" not in key for key in keys)

