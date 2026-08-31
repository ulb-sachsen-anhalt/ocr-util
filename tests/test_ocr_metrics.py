# -*- coding: utf-8 -*-
"""OCR Metric Test Module"""

import random

import rapidfuzz.distance.Levenshtein as rfls

import pytest

import ocr_util.eval.evaluation as digev
import ocr_util.eval.metrics as digem
import ocr_util.eval.preprocessing as dipre
import ocr_util.eval.geometry as digeo

from .conftest import TEST_RES_DIR

# default reference
THE_COMBINED_A_FOX = 'the á lazy brown fox jumps over the hump'
THE_LAZY_FOX = 'the lazy brown fox jumps over the hump'
THE_FOX_LAZY = 'the fox lazy brown jumps over the hump'
THE_FOX_INPUT_IR = 'the hump lazy brown fox fox fox jumps'


def test_metric_unicode_normalization_textual_metric():
    """default OCR-D compliant UTF-8 normalization 
    yield similarity of 95%"
    """

    # arrange
    char_metric = digem.MetricChars()
    char_metric.reference = THE_LAZY_FOX
    char_metric.candidate = THE_COMBINED_A_FOX

    # actsert
    assert 95.0 == char_metric.value


def test_metric_characters_from_empty_gt():
    """Total un-similarity"""

    # arrange
    _metric = digem.MetricChars()
    # _metric.preprocessings = [_filter_whitespaces]
    _metric.reference = ''
    _metric.candidate = THE_LAZY_FOX

    # assert
    assert 0.0 == _metric.value


def test_metric_letter_from_empty_gt_and_empty_candidate():
    """Behavor: Similarity of empty strings"""

    # arrange
    _metric = digem.MetricLetters()
    _metric.reference = ''
    _metric.candidate = ''

    # assert
    assert 100.0 == _metric.value


def test_metric_words_with_only_slight_difference():
    """simple word accurracy test"""

    # arrange
    _metric = digem.MetricWords()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_FOX_LAZY

    # act
    _actual = _metric.value

    # assert
    # string has 38 characters, but tokens are only 8 present
    # assert len(_metric.input_reference) == 38
    assert len(_metric.reference) == 8
    assert 75.0 == _actual


def test_metric_wa_with_identical_data():
    """simple word similarity for similar inputs"""

    # arrange
    _metric = digem.MetricWords()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_LAZY_FOX

    # assert
    assert 100 == _metric.value


def test_metric_bow_from_reasonable_input():
    """simple bag of words test"""

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_FOX_LAZY

    # act
    assert 100 == _metric.value


def test_metric_bow_from_empty_gt_and_empty_candidate():
    """how to handle empty data - means: no errors"""

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = ''
    _metric.candidate = ''

    # act
    _actual = _metric.value

    # assert
    assert 100 == _actual


def test_bow_ocrd_similarity_rate():
    """
    Behavior for bow similarity rate of two strings
    OCR-D spec: https://github.com/OCR-D/spec/blob/master/ocrd_eval.md#bag-of-words-error-rate
    """

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = "der Mann steht an der Ampel"
    _metric.candidate = "cer Mann fteht an der Ampel"

    # act
    _actual = _metric.value

    # assert
    assert 66.66 == pytest.approx(_actual, rel=1e-2)


def test_bow_ocrd_spec_similarity_rate_ref_contains_more_data():
    """
    Behavior for bow error rate of two strings
    OCR-D spec: https://github.com/OCR-D/spec/blob/master/ocrd_eval.md#bag-of-words-error-rate
    """

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = "der Mann steht an der roten Ampel"
    _metric.candidate = "cer Mann fteht an der Ampel"

    # act
    _actual = _metric.value

    # assert
    assert 61.54 == pytest.approx(_actual, rel=1e-2)


def test_bow_ocrd_spec_similarity_rate_ref_contains_less_data():
    """
    Behavior for bow error rate of two strings
    OCR-D spec: https://github.com/OCR-D/spec/blob/master/ocrd_eval.md#bag-of-words-error-rate
    """

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = "der Mann steht an der Ampel"
    _metric.candidate = "cer Mann fteht an der schönen roten Ampel"

    # act
    _actual = _metric.value

    # assert
    assert 57.14 == pytest.approx(_actual, rel=1e-2)


def test_metric_character_accuracy():
    """simple usage of MetricsCA"""

    str1 = 'sthe lazy brown fox jumps overthe hump'
    str2 = 'fthe lazy brown fox jumps ouer the hump'

    # arrange
    char_metric = digem.MetricChars()
    char_metric.reference = str1
    char_metric.candidate = str2

    # assert
    assert 92.31 == pytest.approx(char_metric.value, rel=0.001, abs=0.001)


def test_metric_character_zd1_0002():
    """Test real world bad example
    the infamous first newspaper page
    from 1889's march 2nd General Anzeiger
    """

    # arrange
    src_candidate = TEST_RES_DIR / 'candidate' / 'frk_alto' / '1667522809_J_0001_0002.xml'
    src_reference = TEST_RES_DIR / 'groundtruth' / 'page' / '1667522809_J_0001_0002.art.gt.xml'
    frame_gt = digeo.get_bounding_box(src_reference)
    raw_can, candidate_lines = dipre.file_to_text(src_candidate, frame_gt)
    spatial_report = {}
    dipre.file_to_text(src_candidate, frame_gt, spatial_report=spatial_report)
    raw_ref, _ = dipre.file_to_text(src_reference)

    normed_to_cand = rfls.normalized_similarity(raw_ref, raw_can)
    normed_to_refr = rfls.normalized_similarity(raw_can, raw_ref)

    assert candidate_lines == 103
    assert spatial_report["excluded_words"] > 0
    assert normed_to_cand == normed_to_refr
    assert 0.3920 == pytest.approx(normed_to_cand, abs=1e-4)


def test_metric_bot_ident():
    """BOW with identical tokens"""

    gt1 = "the lazy brown fox jumps over the hump again and again three times the dude"
    list2 = list(gt1.split())
    random.shuffle(list2)
    str2 = ' '.join(list2)

    result = digem.bag_of_tokens(gt1.split(), str2.split())
    assert result == 1.0
    assert len(gt1.split()) == len(str2.split())


def test_metric_bot_candidate_with_only_repetitions():
    """
    Behaviour of BOW with multiple identical entries
    """

    gt1  = "the dizzy brown fox jumps"
    str2 = "the dizzy brown fox fox fox jumps"

    # actsert
    assert 0.83 == pytest.approx(digem.bag_of_tokens(gt1.split(), str2.split()), abs=1e-2)


def test_metric_bot_miss_tokens():
    """BOW with missing tokens in candidate text"""

    gt1 = "the lazy brown fox jumps"
    str2 = "the brown fux jumps"

    # acsert
    assert 0.66 == pytest.approx(digem.bag_of_tokens(gt1.split(), str2.split()), abs=1e-2)


def test_ir_metric_precision_fox():
    """Basic test IR Precision with candidate
    having all tokens included (minus stopwords)"""

    # arrange
    m_prec = digem.MetricIRPre()
    m_prec.reference = THE_LAZY_FOX
    m_prec.candidate = THE_FOX_INPUT_IR

    # act
    actual = m_prec.value

    # assert
    assert actual == 100.0
    assert m_prec.data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}


def test_ir_metric_recall_fox():
    """Basic test IR Recall - everthing has been found
    (minus stoppwords)"""

    # arrange
    m_prec = digem.MetricIRRec()
    m_prec.reference = THE_LAZY_FOX
    m_prec.candidate = THE_FOX_INPUT_IR

    # act
    actual = m_prec.value

    # assert
    assert actual == 100.0
    assert m_prec.data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}


IR_CANDIDATE_TEXT = 'the red fox'


def test_ir_metrics_precision_english_poor_candidate():
    """Example with all IR-Metrics and
    a rather poor candidate"""

    # arrange
    pre = digem.MetricIRPre()
    pre.reference = THE_LAZY_FOX
    pre.candidate = IR_CANDIDATE_TEXT

    # assert
    assert 50.0 == pytest.approx(pre.value, 0.01)
    assert pre.data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}
    assert pre.data_candidate == {'red', 'fox'}


def test_ir_metrics_recall_english_poor_candidate():
    """Example with all IR-Metrics and
    a rather poor candidate"""

    # arrange
    rec = digem.MetricIRRec()
    rec.reference = THE_LAZY_FOX
    rec.candidate = IR_CANDIDATE_TEXT

    # assert
    assert 20.0 == pytest.approx(rec.value, 0.01)


IR_CANDIDATE_TEXT_GERMAN = 'dieser faule Fuchs springt die Hecke'
IR_REFERENCE_TEXT_GERMAN = 'Fuchs springt faule Hecke'
IR_REFERENCE_TEXT_GERMAN_POOR = 'forsche Fuchs hopst'

def test_ir_metrics_precision_german():
    """Candidate with german phrase
    and very nice candidate precision"""

    # arrange
    prec = digem.MetricIRPre(languages=['german'])
    prec.reference = IR_REFERENCE_TEXT_GERMAN
    prec.candidate = IR_CANDIDATE_TEXT_GERMAN

    # act
    assert prec.value == 100.0


def test_ir_metrics_recall_german():
    """Candidate with german phrase
    and very nice candidate recall"""

    # arrange
    rec = digem.MetricIRRec(languages=['german'])
    rec.reference = IR_REFERENCE_TEXT_GERMAN
    rec.candidate = IR_CANDIDATE_TEXT_GERMAN

    # act
    assert rec.value == 100.0


def test_ir_metrics_precision_german_poor_candidate():
    """Candidate with german phrase
    and rather poor candidate"""

    # arrange
    metric_pre = digem.MetricIRPre(languages=['german'])
    metric_pre.reference = IR_CANDIDATE_TEXT_GERMAN
    metric_pre.candidate = IR_REFERENCE_TEXT_GERMAN_POOR

    # assert
    assert metric_pre.value == pytest.approx(33.33, 1e-2)


def test_ir_metrics_recall_german_poor_candidate():
    """Candidate with german phrase
    and rather poor candidate"""

    # arrange
    metric_rec = digem.MetricIRRec(languages=['german'])
    metric_rec.reference = IR_CANDIDATE_TEXT_GERMAN
    metric_rec.candidate = IR_REFERENCE_TEXT_GERMAN_POOR

    # assert
    assert metric_rec.value == 25.0


def test_metrics_token_based_more_gt_than_tc():
    """token edit distance with
    * 2 exchanges (first 2 tokens), followed by
    * 3 insertions ('springt', 'über', 'die')
    => total 5 edit operations required

    aligned means: 5 / (5 + 2) 0.7142 distance
    => 1 - 0.7142 => 0.2857 similarity
    as percents
    => 28.57 %
    """

    # arrange
    gt1 = "der faulte Fuchs springt über die Hecke"
    cand = "faule springt Fuchs Hecke"

    # act
    m_word = digem.MetricWords()
    m_word.reference = gt1
    m_word.candidate = cand

    # assert
    value = m_word.value
    assert 28.57 == pytest.approx(value, rel=1e-2)
    assert len(cand.split()) + 3 == len(gt1.split())



########################################################### OCR-Pipeline-Tests

# def test_step_estimateocr_textline_conversions():
#     """Test functional behavior for valid ALTO-output"""
#
#     test_data = os.path.join('tests', 'resources', '500_gray00003.xml')
#
#     # act
#     # pylint: disable=protected-access
#     xml_data = ET.parse(test_data)
#     lines = get_lines(xml_data)
#     (_, n_lines, _, _, n_lines_out) = textlines2data(lines)
#
#     assert n_lines == 360
#     assert n_lines_out == 346
#
# @mock.patch("requests.post")
# def test_step_estimateocr_lines_and_tokens_err_ratio(mock_requests):
#     """Test behavior of for valid ALTO-output"""
#
#     # arrange
#     test_data = os.path.join(PROJECT_ROOT_DIR,
#                              'tests', 'resources', '500_gray00003.xml')
#     mock_requests.side_effect = _fixture_languagetool
#     params = {'service_url': 'http://localhost:8010/v2/check',
#               'language': 'de-DE',
#               'enabled_rules': 'GERMAN_SPELLER_RULE'
#               }
#     step = StepEstimateOCR(params)
#     step.path_in = test_data
#
#     # act
#     step.execute()
#
#     assert step.statistics
#     assert mock_requests.called == 1
#     assert step.n_errs == 548
#     assert step.n_words == 2636
#     assert step.statistics[0] == pytest.approx(79.211, rel=1e-3)
#
# @mock.patch("requests.post")
# def test_step_estimateocr_lines_and_tokens_hit_ratio(mock_requests):
#     """Test behavior of for valid ALTO-output"""
#
#     # arrange
#     test_data = os.path.join(PROJECT_ROOT_DIR,
#                              'tests', 'resources', '500_gray00003.xml')
#     mock_requests.side_effect = _fixture_languagetool
#     params = {'service_url': 'http://localhost:8010/v2/check',
#               'language': 'de-DE',
#               'enabled_rules': 'GERMAN_SPELLER_RULE'
#               }
#     step = StepEstimateOCR(params)
#     step.path_in = test_data
#
#     # act
#     step.execute()
#
#     assert mock_requests.called == 1
#     err_ratio = (step.n_errs / step.n_words) * 100
#     assert err_ratio == pytest.approx(20.789, rel=1e-3)
#
#     # revert metric to represent hits
#     # to hit into positive compliance
#     hits = (step.n_words - step.n_errs) / step.n_words * 100
#     assert hits == pytest.approx(79.21, rel=1e-3)
#
#     # holds this condition?
#     assert hits == pytest.approx(100 - err_ratio, rel=1e-9)
