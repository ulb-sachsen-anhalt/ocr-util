# -*- coding: utf-8 -*-
"""OCR Evaluation Test Module"""

import math
import os
import shutil

from pathlib import Path
from xml.etree.ElementTree import (
    ParseError
)

import pytest
from pytest import (
    approx
)

import ocr_util.eval.evaluation as digev
import ocr_util.eval.metrics as digem
import ocr_util.eval.preprocessing as dipre

from .conftest import (
    TEST_RES_DIR
)


_CANDIDATE_NAME = '1667522809_J_0001_0002.xml'
_GT_NAME = '1667522809_J_0001_0002.art.gt.xml'
_DOMAIN = '1667522809_J_0001'


@pytest.fixture(name='eval_paths', scope='module')
def _eval_paths(tmp_path_factory):
    """One-time file setup for single-pair evaluation tests."""
    base = tmp_path_factory.mktemp('eval_pairs')

    # alto-candidate + page-groundtruth pair
    alto_eval = base / 'alto_candidate' / _DOMAIN
    alto_gt = base / 'alto_gt' / _DOMAIN
    alto_eval.mkdir(parents=True)
    alto_gt.mkdir(parents=True)
    shutil.copy(
        TEST_RES_DIR / 'candidate' / 'frk_alto' / _CANDIDATE_NAME,
        alto_eval / _CANDIDATE_NAME,
    )
    gt_file = alto_gt / _GT_NAME
    shutil.copy(TEST_RES_DIR / 'groundtruth' / 'page' / _GT_NAME, gt_file)

    # gt-as-candidate pair (evaluate gt with itself)
    self_eval = base / 'self_candidate' / _DOMAIN
    self_gt = base / 'self_gt' / _DOMAIN
    self_eval.mkdir(parents=True)
    self_gt.mkdir(parents=True)
    self_candidate_file = self_eval / _CANDIDATE_NAME
    self_gt_file = self_gt / _GT_NAME
    shutil.copy(TEST_RES_DIR / 'groundtruth' / 'page' / _GT_NAME, self_candidate_file)
    shutil.copy(TEST_RES_DIR / 'groundtruth' / 'page' / _GT_NAME, self_gt_file)

    return {
        'alto_eval_domain': alto_eval,
        'alto_gt_domain': alto_gt,
        'alto_gt_dst': gt_file,
        'self_eval_domain': self_eval,
        'self_gt_domain': self_gt,
        'self_candidate_file': self_candidate_file,
        'self_gt_file': self_gt_file,
    }


def test_piece_to_text_alto_candidate_with_coords():
    """Check lines from ALTO candidate"""

    alto_path = f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0512_01.xml'
    p1 = (300, 375)
    p2 = (6200, 3425)

    # act
    _as_lines, _ = dipre.file_to_text(alto_path, frame=(p1, p2), oneliner=False)
    _gt_type = digev._get_groundtruth_from_filename(alto_path)

    # assert
    assert _gt_type == 'n.a.'
    assert 166 == len(_as_lines)


def test_evaluate_single_alto_candidate_with_page_groundtruth(eval_paths):
    """Illustrate evaluation of single candidate
    with proper organization of groundtruth data"""

    # arrange - reuse pre-copied files
    eval_domain = eval_paths['alto_eval_domain']
    gt_domain = eval_paths['alto_gt_domain']
    gt_dst = eval_paths['alto_gt_dst']
    evaluator = digev.Evaluator(eval_domain)
    evaluator.metrics = [digem.MetricChars()]
    evaluator.domain_reference = gt_domain

    # act
    eval_entry = digev.EvalEntry(eval_domain / _CANDIDATE_NAME, eval_domain)
    eval_entry.path_groundtruth = gt_dst
    evaluator.eval_all([eval_entry], sequential=True)
    evaluator.aggregate(by_type=True)
    evaluator.eval_map()
    result = evaluator.get_results()[0]
    defaults = result.get_defaults()

    # assert
    assert 5 == len(defaults)
    # metric label
    assert 'Cs@1667522809_J_0001@article' == defaults[0]
    assert 1 == defaults[1]  # number of data points
    # metric raw
    assert 39.19 == pytest.approx(defaults[2], rel=1e-3)
    # metric with stripped outliers (no outlier, of course!)
    assert 0 == result.n_outlier
    assert not result.cleared_result
    # reference size chars
    # changed from 5385 to 5309 since
    # children text wins over parent text
    assert 5309 == defaults[4]


def test_evaluate_page_groundtruth_with_itself(eval_paths):
    """Use Groundtruth as candidate and evaluate it
    with itself as reference data shall yield
    accuracy of nearly 100 percent"""

    # arrange - reuse pre-copied files
    eval_domain = eval_paths['self_eval_domain']
    gt_domain = eval_paths['self_gt_domain']
    candidate_dst = eval_paths['self_candidate_file']
    gt_dst = eval_paths['self_gt_file']
    evaluator = digev.Evaluator(eval_domain)
    evaluator.metrics = [digem.MetricChars()]
    evaluator.domain_reference = gt_domain

    # act
    eval_entry = digev.EvalEntry(candidate_dst, eval_domain)
    eval_entry.path_groundtruth = gt_dst
    evaluator.eval_all([eval_entry], sequential=True)
    evaluator.aggregate(by_type=True)
    evaluator.eval_map()
    result = evaluator.get_results()[0]
    defaults = result.get_defaults()

    # assert
    assert 5 == len(defaults)
    # metric label
    assert 'Cs@1667522809_J_0001@article' == defaults[0]
    assert 1 == defaults[1]  # number of data points
    assert defaults[2] == pytest.approx(100.00, rel=1e-3)
    # reference size chars
    # changed from 5385 to 5309 since
    # children text wins over parent text
    assert 5309 == defaults[4]


def test_evaluate_set_with_5_entries(tmp_path):
    """Simulate evaluation with 5 data sets
    * sub_dir 'eng' having 1 pair
    * sub_dir 'ger' having 4 pairs

    Hereby it is required to have *real* paths to make
    this work properly.

    please note:
        in tests it turned out it needs more than 5 data points / aggregation stage
        to measure the impact of the outlier (here: 1/6 with CA 86,447) 

    also note the somehow hacky way of setting the numbers of the reference data
    by injecting wacky test strings - only size matters
    """

    # arrange
    path_dir_gt = tmp_path / 'odem'
    path_dir_gt.mkdir()
    path_dir_c = tmp_path / 'media' / 'jpg' / 'odem'
    path_dir_c.mkdir(parents=True)
    evaluator = digev.Evaluator(path_dir_c)
    evaluator.domain_reference = path_dir_gt
    _metric_ca1 = digem.MetricChars()
    _metric_ca1._value = 95.70
    _metric_ca1.data_reference = 't' * 810
    _metric_ca2 = digem.MetricChars()
    _metric_ca2._value = 96.53
    _metric_ca2.data_reference = 't' * 675
    _metric_ca3 = digem.MetricChars()
    _metric_ca3._value = 94.91
    _metric_ca3.data_reference = 't' * 1395
    _metric_ca4 = digem.MetricChars()
    _metric_ca4._value = 94.40
    _metric_ca4.data_reference = 't' * 1466
    # outlier !
    _metric_ca5 = digem.MetricChars()
    _metric_ca5._value = 86.44
    _metric_ca5.data_reference = 't' * 1520
    _metric_ca6 = digem.MetricChars()
    _metric_ca6._value = 93.44
    _metric_ca6.data_reference = 't' * 1520

    entry1 = digev.EvalEntry(path_dir_c / 'eng' / 'urn+nbn+de+gbv+3+1-135654-p0403-5_eng.xml',
                             path_dir_c / 'eng')
    entry1.path_groundtruth = path_dir_gt / 'eng' / 'urn+nbn+de+gbv+3+1-135654-p0403-5_eng.gt.xml'
    entry1.align_domains()
    entry1.metrics = [_metric_ca1]
    entry2 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-816198-p0493-2_ger.xml',
                             path_dir_c / 'ger')
    entry2.path_groundtruth = path_dir_gt / 'ger' / 'urn+nbn+de+gbv+3+1-816198-p0493-2_ger.gt.xml'
    entry2.align_domains()
    entry2.metrics = [_metric_ca2]
    entry3 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-818383-p0034-5_ger.xml',
                             path_dir_c / 'ger')
    entry3.path_groundtruth = path_dir_gt / 'ger' / 'urn+nbn+de+gbv+3+1-818383-p0034-5_ger.gt.xml'
    entry3.align_domains()
    entry3.metrics = [_metric_ca3]
    entry4 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-822479-p1119-4_ger.xml',
                             path_dir_c / 'ger')
    entry4.path_groundtruth = path_dir_gt / 'ger' / 'urn+nbn+de+gbv+3+1-822479-p1119-4_ger.gt.xml'
    entry4.align_domains()
    entry4.metrics = [_metric_ca4]
    entry5 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-828020-p0173-6_ger.xml',
                             path_dir_c / 'ger')
    entry5.path_groundtruth = path_dir_gt / 'ger' / 'urn+nbn+de+gbv+3+1-828020-p0173-6_ger.gt.xml'
    entry5.align_domains()
    entry5.metrics = [_metric_ca5]
    entry6 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-125584-p0314-6_ger.xml',
                             path_dir_c / 'ger')
    entry6.path_groundtruth = path_dir_gt / 'ger' / 'urn+nbn+de+gbv+3+1-125584-p0314-6_ger.gt.xml'
    entry6.align_domains()
    entry6.metrics = [_metric_ca6]
    evaluator.evaluation_entries = [entry1, entry2, entry3, entry4, entry5, entry6]

    # act
    evaluator.aggregate(by_metrics=[0])
    evaluator.eval_map()
    results = evaluator.get_results()

    # assert
    # ensure standard deviation decreases by dropping outlier
    assert 2 == len(results)
    assert results[1].std == approx(3.498, abs=1e-2)
    assert not results[1].cleared_result


@pytest.mark.parametrize(
    "weighted_mean,expected_mean",
    [
        (False, 70.0),
        (True, (90.0 * 10 + 50.0 * 1000) / 1010),
    ],
)
def test_eval_map_weighted_mean_switch(tmp_path, weighted_mean, expected_mean):
    """Mean aggregation can switch between unweighted and weighted mode."""

    path_dir_gt = tmp_path / "gt"
    path_dir_gt.mkdir()
    path_dir_c = tmp_path / "cand"
    path_dir_c.mkdir()

    evaluator = digev.Evaluator(path_dir_c, weighted_mean=weighted_mean)
    evaluator.domain_reference = path_dir_gt

    m1 = digem.MetricChars()
    m1._value = 90.0
    m1.data_reference = "t" * 10
    m2 = digem.MetricChars()
    m2._value = 50.0
    m2.data_reference = "t" * 1000

    e1 = digev.EvalEntry(path_dir_c / "grp" / "a.xml", path_dir_c / "grp")
    e1.path_groundtruth = path_dir_gt / "grp" / "a.gt.xml"
    e1.align_domains()
    e1.metrics = [m1]

    e2 = digev.EvalEntry(path_dir_c / "grp" / "b.xml", path_dir_c / "grp")
    e2.path_groundtruth = path_dir_gt / "grp" / "b.gt.xml"
    e2.align_domains()
    e2.metrics = [m2]

    evaluator.evaluation_entries = [e1, e2]
    evaluator.aggregate(by_metrics=[0])
    evaluator.eval_map()
    results = evaluator.get_results()

    assert len(results) == 1
    assert results[0].mean == pytest.approx(expected_mean, rel=1e-6)


def test_eval_map_weighted_mean_zero_refs_fallback(tmp_path):
    """Weighted mode falls back to unweighted mean if total refs are zero."""

    path_dir_gt = tmp_path / "gt"
    path_dir_gt.mkdir()
    path_dir_c = tmp_path / "cand"
    path_dir_c.mkdir()

    evaluator = digev.Evaluator(path_dir_c, weighted_mean=True)
    evaluator.domain_reference = path_dir_gt

    m1 = digem.MetricChars()
    m1._value = 90.0
    m1.data_reference = ""
    m2 = digem.MetricChars()
    m2._value = 50.0
    m2.data_reference = ""

    e1 = digev.EvalEntry(path_dir_c / "grp" / "a.xml", path_dir_c / "grp")
    e1.path_groundtruth = path_dir_gt / "grp" / "a.gt.xml"
    e1.align_domains()
    e1.metrics = [m1]

    e2 = digev.EvalEntry(path_dir_c / "grp" / "b.xml", path_dir_c / "grp")
    e2.path_groundtruth = path_dir_gt / "grp" / "b.gt.xml"
    e2.align_domains()
    e2.metrics = [m2]

    evaluator.evaluation_entries = [e1, e2]
    evaluator.aggregate(by_metrics=[0])
    evaluator.eval_map()
    results = evaluator.get_results()

    assert len(results) == 1
    assert results[0].mean == pytest.approx(70.0, rel=1e-6)


def test_no_groundtruth_at_all(tmp_path):
    """
    Behavior if no groundtruth found:
    Raise RuntimeError and exit, since evaluation
    doesn't make any sense so far
    """

    evaluator = digev.Evaluator(tmp_path)
    evaluator.eval_all([])

    with pytest.raises(RuntimeError) as err:
        evaluator.aggregate()

    assert 'missing evaluation data' in str(err.value)


def test_handle_exception_invalid_literal_for_int():
    """Handle evaluation exception: 
        invalid literal for int() with base 10: ''

        Please note, that this exception results from 
        inconsistencies in the groundtruth data, therefore 
        only use a dummy placeholder, no *real* OCR candidate,
        originating from missing Coords
    """

    # arrange
    domain = TEST_RES_DIR / 'groundtruth' / 'page'
    path_gt =  domain / 'urn+nbn+de+gbv+3+1-792101-p0667-5_ger.gt.xml'
    assert path_gt.exists()
    eval_entry = digev.EvalEntry(path_gt, domain)
    eval_entry.path_groundtruth = Path(path_gt).absolute()

    # act
    evaluator = digev.Evaluator('dummy_path')
    evaluator.metrics = [digem.SimilarityMetric()]
    with pytest.raises(RuntimeError) as err:
        evaluator.eval_entry(eval_entry)

    # assert
    assert 'urn+nbn+de+gbv+3+1-792101-p0667-5_ger.gt.xml' in err.value.args[0]
    assert 'too few points' in err.value.args[0] or 'empty Coords' in err.value.args[0]


def test_handle_empty_candidate_information_retrival():
    """Handle evaluation exception: 
        unsupported format string passed to NoneType.__format__
        results from complete failing candidate text
        and nltk behaving inconsistent
    """

    # arrange
    path_gt = TEST_RES_DIR / 'groundtruth' / 'page' / 'urn+nbn+de+gbv+3+1-138193-p0904-0_ger.gt.xml'
    path_cd = TEST_RES_DIR / 'candidate' / 'frk_page' / 'urn+nbn+de+gbv+3+1-138193-p0904-0_ger.xml'
    eval_entry = digev.EvalEntry(path_cd)
    eval_entry.path_groundtruth = Path(path_gt).absolute()
    evaluator = digev.Evaluator('/data')
    evaluator.metrics = [digem.MetricIRPre(), digem.MetricIRRec()]
    evaluator.verbosity = 1

    # act
    evaluator.eval_entry(eval_entry)

    # assert
    assert eval_entry.metrics[0].label == 'Pre'
    assert eval_entry.metrics[0].value == 0.0
    assert eval_entry.metrics[1].label == 'Rec'
    assert eval_entry.metrics[1].value == 0.0


def test_handle_table_text_groundtruth():
    """Handle evaluation properly with very
    poor candidate data from ocr-ing a table
     
    legacy: "missing gt text from urn+nbn+de+gbv+3+1-126343-p0285-7_ger.gt.xml"
    """

    # arrange
    path_gt = TEST_RES_DIR / 'groundtruth' / 'page' / 'urn+nbn+de+gbv+3+1-126343-p0285-7_ger.gt.xml'
    path_cd = TEST_RES_DIR / 'candidate' / 'frk_page' / 'urn+nbn+de+gbv+3+1-126343-p0285-7_ger.xml'
    eval_entry = digev.EvalEntry(path_cd)
    eval_entry.path_groundtruth = path_gt

    # act
    evaluator = digev.Evaluator('/data')
    evaluator.metrics = [digem.MetricChars()]
    evaluator._wrap_eval_entry(eval_entry)

    # assert / legacy 1: 5.825 / legacy2: 4.0
    _result_cca = eval_entry.metrics[0].value
    assert 5.9 < _result_cca < 6.1


def test_get_box_from_empty_page():
    """How to deal with empty PAGE"""

    # arrange
    _path_gt = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-201080-p0034-8_ger.gt.xml'

    # act
    bbox = digev.get_bounding_box(_path_gt)
    assert bbox is not None
    _p1, _p2 = bbox

    # assert 
    assert _p1 == (77, 58)
    assert _p2 == (2012, 2506)


def test_get_box_when_line_points_messy():
    """Ensure to get valid geometric data
    even if Coords@points have trailing
    whitespaces"""

    # arrange
    _path_gt = f'{TEST_RES_DIR}/groundtruth/page/rahbar-1771946695-00000040.xml'

    # act
    bbox = digev.get_bounding_box(_path_gt)
    assert bbox is not None
    _p1, _p2 = bbox

    # assert
    assert _p1 == (368, 619)
    assert _p2 == (1298, 2314)


def test_handle_exception_invalid_alto_xml():
    """Handle invalid XML data

    ALTO data got corrupted by copying it around
    (the latest rows simply missing)
    """

    # arrange
    path_gt = TEST_RES_DIR / "candidate" / "frk_alto" / "1667522809_J_0001_0256_corrupt.xml"
    eval_entry = digev.EvalEntry('dummy_candidate')
    eval_entry.path_groundtruth = path_gt

    # act
    evaluator = digev.Evaluator('dummy_path')
    evaluator.metrics = [digem.SimilarityMetric()]
    with pytest.raises(ParseError) as err:
        evaluator.eval_entry(eval_entry)

    # assert
    assert 'no element found' in err.value.args[0]


def test_handle_textual_input_data():
    """Behavior if textual input data is given instead of XML
    for both candidate and groundtruth
    """

    # arrange
    path_cn_base = TEST_RES_DIR / "candidate" / "txt"
    path_cn = path_cn_base / "OCR-Fraktur_1246734.txt"
    path_gt = TEST_RES_DIR / "groundtruth" / "txt" / "1246734.gt.txt"
    eval_entry = digev.EvalEntry(path_cn)
    eval_entry.path_groundtruth = path_gt

    # act
    evaluator = digev.Evaluator(path_cn_base)
    evaluator.metrics = [digem.MetricChars()]
    evaluated = evaluator.eval_entry(eval_entry)

    # assert
    assert evaluated is not None
    assert len(evaluated.metrics) == 1
    assert evaluated.metrics[0].label == 'Cs'
    curr_value = evaluated.metrics[0].value
    assert evaluated.metrics[0].value == pytest.approx(94.7, rel=1e-3, abs=1e-3)
    assert math.isclose(curr_value, 94.7, rel_tol=1e-3, abs_tol=1e-3)
