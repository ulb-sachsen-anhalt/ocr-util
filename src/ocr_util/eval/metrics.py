# -*- coding: utf-8 -*-
"""Metrics Module"""

from __future__ import annotations

import collections
import typing

from nltk.metrics import precision as nltk_precision
from nltk.metrics import recall, f_measure
import rapidfuzz.distance.Levenshtein as rfls

import ocr_util.eval.preprocessing as dipre
from ocr_util.eval import constants as eval_constants
from ocr_util.eval.dictionary_metrics.common import LANGUAGE_KEY_DEFAULT
from ocr_util.eval.dictionary_metrics.language_tool.LanguageTool import LanguageTool


class DigitalEvalMetricException(Exception):
    """Mark Exception during validation/calculating metrics"""


class OCRMetric:
    """Base class to calculate OCR metric
    for at least a candidate input with
    desired precision and preprocessor
    """

    def __init__(self, precision=2):
        self.precision = precision
        self.preprocessor = None
        self.preprocessing_report = None
        self.candidate_frame = None
        self.geometry_disabled = False
        self.data_candidate = None
        self._label = None
        self._value = None

    @property
    def candidate(self):
        """Candidate data"""
        return self.data_candidate

    @candidate.setter
    def candidate(self, candidate_data):
        self.data_candidate = candidate_data
        self._value = None
        self.preprocessing_report = None

    @property
    def label(self):
        """Metric's label"""
        return self._label


class SimilarityMetric(OCRMetric):
    """Basic definition of OCR Similarity Metric
    comparing given candidate input with
    corresponding reference/groundtruth data

    expressed in percent (0 - 100)
    """

    def __init__(self, precision=2):
        super().__init__(precision)
        self.preprocessor: dipre.Preprocessor = dipre.TextPreprocessor
        self.code_norm = dipre.UC_NORMALIZATION_DEFAULT
        self.data_reference = None

    @property
    def reference(self):
        """Reference/Groundtruth data"""
        return self.data_reference

    @reference.setter
    def reference(self, data_reference):
        self.data_reference = data_reference
        self._value = None
        self.preprocessing_report = None

    @property
    def n_ref(self) -> int:
        """Number of current reference data"""
        if not hasattr(self, "data_reference") or self.data_reference is None:
            raise DigitalEvalMetricException("invalid reference data!")
        return len(self.data_reference)

    @n_ref.setter
    def n_ref(self, value):
        """Only for testing purposes"""

        self.data_reference = "t" * value

    @property
    def value(self):
        """Evaluate each time and round
        with desired precision afterwards"""
        if self._value is None:
            if self.preprocessor is not None:
                pre_can = self.preprocessor(self.data_candidate)
                if isinstance(pre_can, dipre.TextPreprocessor):
                    pre_can.code_norm = self.code_norm
                    pre_can.frame = self.candidate_frame
                pre_can.run()
                self.data_candidate = pre_can.result
                pre_ref = self.preprocessor(self.data_reference)
                if isinstance(pre_ref, dipre.TextPreprocessor):
                    pre_ref.code_norm = self.code_norm
                pre_ref.run()
                self.data_reference = pre_ref.result
                self.preprocessing_report = {
                    eval_constants.REPORT_PREPROCESSOR: type(pre_can).__name__,
                    eval_constants.REPORT_CANDIDATE: pre_can.preprocessing_report,
                    eval_constants.REPORT_REFERENCE: pre_ref.preprocessing_report,
                    eval_constants.REPORT_GEOMETRY_DISABLED: self.geometry_disabled,
                }
            self._forward()
            if self._value is not None:
                self._value *= 100
        return round(self._value, self.precision)

    def _forward(self):
        self._value = levenshtein_norm(self.data_reference, self.data_candidate)


class MetricChars(SimilarityMetric):
    """Calculate plain sequent character based metric"""

    def __init__(self, precision=2):
        super().__init__(precision)
        self._label = "Cs"


class MetricLetters(SimilarityMetric):
    """Calculate metric for only a certain sub-set of
    character sequence"""

    def __init__(self, precision=2):
        super().__init__(precision)
        self._label = "Ls"
        self.preprocessor: dipre.Preprocessor = dipre.LetterPreprocessor


class MetricWords(SimilarityMetric):
    """Calculate metric for a sequence of word tokens"""

    def __init__(self, precision=2):
        super().__init__(precision)
        self._label = "Ws"
        self.preprocessor: dipre.Preprocessor = dipre.SimpleTokenizer


class MetricBoW(SimilarityMetric):
    """Calculate metric for a multiset of word tokens"""

    def __init__(self, precision=2):
        super().__init__(precision)
        self._label = "BoWs"
        self.preprocessor: dipre.Preprocessor = dipre.SimpleTokenizer

    def _forward(self):
        self._value = bag_of_tokens(self.data_reference, self.data_candidate)


class MetricIR(SimilarityMetric):
    """Calculate information retrival metrics"""

    def __init__(self, precision=2, languages=None):
        super().__init__(precision)
        self.languages = languages
        self.preprocessor: dipre.Preprocessor = dipre.LanguageAwareTokenizer


class MetricIRPre(MetricIR):
    """Calculate precision"""

    def __init__(self, precision=2, languages=None):
        super().__init__(precision, languages)
        self._label = "Pre"

    def _forward(self):
        self._value = ir_precision(self.data_reference, self.data_candidate)


class MetricIRRec(MetricIR):
    "Calculate recall"

    def __init__(self, precision=2, languages=None):
        super().__init__(precision, languages)
        self._label = "Rec"

    def _forward(self):
        self._value = ir_recall(self.data_reference, self.data_candidate)


class MetricDictionary(OCRMetric):
    """Calculate metric for a multiset of word tokens"""

    LANGUAGE: str = LANGUAGE_KEY_DEFAULT


class MetricDictionaryLangTool(MetricDictionary):
    """Calculate metric for a multiset of word tokens"""

    def __init__(self, precision=2):
        super().__init__(precision=precision)
        self._label = "DictLT"
        self.diff = 0

    def _forward(self):
        text: str = self.data_candidate
        text_list: typing.List[str] = self.data_candidate.split()
        n_tokens: int = len(text_list)
        lt_response_data: typing.Dict = LanguageTool.check(
            text, MetricDictionary.LANGUAGE
        )
        total_matches = (
            lt_response_data["matches"] if "matches" in lt_response_data else 0
        )
        typo_errors = len(total_matches)
        self.diff = typo_errors if typo_errors <= n_tokens else n_tokens


def levenshtein_norm(reference_data, candidate_data, inverse=False) -> int:
    """Calculate levenshtein metric as ration of sum of edit operations
    normalized to sum of edit and equal operations.

    Works with characters and word-like tokens, where
    tokens correspond also to:
    * abbreviations  (like "Nr." or "Etg.")
    * numbers/years  (like "1899")
    * split-up words (line endings/beginnings)
    """
    if inverse:
        return rfls.normalized_distance(reference_data, candidate_data)
    return rfls.normalized_similarity(reference_data, candidate_data)


def bag_of_tokens(
    reference_tokens: typing.List[str], candidate_tokens: typing.List[str]
) -> int:
    """Calculate difference between reference and candidate token list"""
    false_negatives: typing.List[str] = _diff(reference_tokens, candidate_tokens)
    false_positives: typing.List[str] = _diff(candidate_tokens, reference_tokens)
    delta = len(false_negatives) + len(false_positives)
    total = len(reference_tokens) + len(candidate_tokens)
    subtrahend = (delta / total) if total > 0 else 0
    ratio = 1 - subtrahend
    return ratio


def _diff(gt_tokens, cd_tokens) -> typing.List[str]:
    return list(
        (collections.Counter(gt_tokens) - collections.Counter(cd_tokens)).elements()
    )


def ir_precision(reference_data, candidate_data) -> float:
    """Calculate Precision for given languages"""

    _prec = nltk_precision(reference_data, candidate_data)
    # nltk actually handles this inconsistently ...
    if _prec is None:
        _prec = 0.0
    return _prec


def ir_recall(reference_data, candidate_data) -> float:
    """Calculate Recall for given languages"""

    # here nltk reports 0.0 if nothing recalled
    return recall(reference_data, candidate_data)


def ir_fmeasure(reference_data, candidate_data) -> float:
    """Calculate F-Measure for given languages"""

    _fm = f_measure(reference_data, candidate_data)
    # required since nltk actually handles this inconsistently ...
    if _fm is None:
        _fm = 0.0
    return _fm
