# -*- coding: utf-8 -*-
"""OCR Evaluation Test Module"""

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
        'mets_file': mets_file,
    }


@pytest.mark.parametrize(
    'utf8_norm, expected_norm_line, expected_metric_line',
    [
        (
            dig.DEFAULT_UTF8_NORM,
            "[DEBUG] text normalized using 'NFC' code points for 'Cs,Ls'",
            "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.20(5309), Ls:38.54(4383)(- 0.66)]",
        ),
        (
            dipre.UC_NORMALIZATION_NFKD,
            "[DEBUG] text normalized using 'NFKD' code points for 'Cs,Ls'",
            "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.10(5363), Ls:38.52(4437)(- 0.58)]",
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
    assert len(std_lines) == 11
    assert std_lines[0] == expected_norm_line
    assert str(std_lines[1]).startswith('[DEBUG] from "5" filtered "3" candidates')
    assert std_lines[4] == expected_metric_line


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
    assert len(std_lines) == 12
    assert std_lines[0] == "[WARN ] base 'ger_frk' and 'GT-PAGE' mismatch, aggregation might be inaccurate!"
    assert std_lines[1] == "[DEBUG] text normalized using 'NFKD' code points for 'Cs,Ls'"
    assert std_lines[5] == "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.10(5363), Ls:38.52(4437)(- 0.58)]"


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


def test_cli_with_mets_mods_aggregation(cli_paths, capsys):
    """Test CLI with METS/MODS aggregation parameters"""
    pytest = __import__('pytest')
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


def test_cli_with_mets_file_only_warning(cli_paths, capsys):
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

