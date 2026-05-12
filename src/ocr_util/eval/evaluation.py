# -*- coding: utf-8 -*-
"""OCR Evaluation Module"""

from __future__ import annotations

import concurrent.futures
import copy
import datetime
import math
import multiprocessing
import os
import re
import sys
import typing

from pathlib import Path

import numpy as np

import ocr_util.eval.metrics as digem

import ocr_util.eval.model.common as dc

from ocr_util.eval.geometry import get_bounding_box

# Import aggregation classes from aggregation module
from ocr_util.eval.aggregation import (
    AggregationDimension,
    AggregationStrategy,
    DirectoryHierarchyExtractor,
    TypeExtractor,
    CustomMetadataExtractor,
    FilenamePatternExtractor,
    METSModsExtractor,
    METSDivAttrExtractor,
    ValueTransformExtractor,
    decade_transform,
    century_transform,
    _NOT_SET as AGGREGATION_NOT_SET,
)

# just use textual information for evaluation
# do *not* respect any geometrics
EVAL_EXTRA_IGNORE_GEOMETRY = "ignore_geometry"

# mark unset values as 'not available'
_NOT_SET = AGGREGATION_NOT_SET

_IGNORE_DIRS = ["GT-PAGE"]

# how long evaluation shall take maximal
# where "None" means "no timeout"
EVAL_TIMEOUT = None


# Note: AggregationDimension and all extractor classes have been moved to aggregation.py
# They are imported at the top of this file for backward compatibility


class EvaluationResult:
    """
    Wrap statistical information
    for groundtruth Evaluation
    regarding a specific set
    = a directory, which's name serves as eval_key

    optional:
        enclose EvaluationResult with outliers removed
    """

    def __init__(self, eval_key: str, n_total: int = 1, n_chars=0, n_lines=0):
        self.eval_key = eval_key
        self.total_mean = 0.0
        self.n_total = n_total
        self.n_outlier = 0
        self.n_chars = n_chars
        self.n_lines = n_lines
        self.mean = 0.0
        self.std = 0.0
        self.median = 0.0
        # set special descendant from same type
        # to hold optional metrics regarding
        # removed outliers
        self.cleared_result = None

    def get_defaults(self):
        """Provide default data (eval_key, number of elements, mean) that must be available"""

        return self.eval_key, self.n_total, self.mean, self.median, self.n_chars


class EvalEntry:
    """Container to transform evaluation inputs
    and results into string representation"""

    def __init__(self, path, candidate_root=None):
        self.path_candidate: Path = path
        self.candidate_root_domain = candidate_root
        self.domain_directories = []
        self.path_groundtruth: typing.Optional[Path] = None
        self.gt_type = _NOT_SET
        self.metrics = []
        self.tags: typing.Dict[str, typing.Any] = {}  # Generic metadata for aggregation

    def align_domains(self):
        """If reference data found, fix domain
        directory order for metriccal
        aggregation"""

        if self.path_groundtruth and self.candidate_root_domain:
            candidate_parents = list(self.path_candidate.parent.parts)
            candidate_rootdom = self.candidate_root_domain.name
            # find start
            ds = []
            curr = candidate_parents.pop()
            ds.append(curr)
            while curr != candidate_rootdom and len(candidate_parents) > 0:
                curr = candidate_parents.pop()
                ds.append(curr)

            # having start?
            dirs = [d for d in ds if d not in _IGNORE_DIRS]
            self.domain_directories = dirs

    def __str__(self) -> str:
        """Dependency between metrics
        * 0=CA => 1=LA
        * 2=WA => 3=BOT
        """
        _pres = [0, 2]
        _accs = [1, 3]
        _raws = []
        _pre_v = None
        for i, m in enumerate(self.metrics):
            _val = m.value
            _ref = m.n_ref
            if _ref > 10000:
                _ref_fmt = f"{(math.floor(float(m.n_ref) / 1000)):>2}K+"
            else:
                _ref_fmt = f"{m.n_ref:>4}"
            _raw = f"{m.label}:{_val:>5.2f}({_ref_fmt})"
            if i in _pres:
                _pre_v = _val
            if i in _accs and _pre_v is not None:
                diff = round(_val, 3) - round(_pre_v, 3)
                _raw += f"(+{diff:>5.2f})" if diff > 0 else f"(-{abs(diff):>5.2f})"
                _pre_v = None
            _raws.append(_raw)
        return ", ".join(_raws)

    def __repr__(self) -> str:
        return f"{self.gt_type} {self.path_candidate}"


class Evaluator:
    """Wrapper for Evaluation given candidates versus reference data

    Raises:
        RuntimeError: if candidates or reference data missing
    """

    def __init__(self, root_candidates, verbosity=0, extras=None):
        """initiate new Evaluator

        Args:
            root_candidates (string|Path): Root domain/path to search for candidates
            verbosity (int, optional): Level of verbosity Defaults to 0.
            extras (_type_, optional): Implementation dependend. Defaults to None.
        """
        self.domain_candidate = root_candidates
        self.domain_reference: typing.Optional[Path] = None
        self.evaluation_entries = []
        self.verbosity = verbosity
        self.evaluation_data = {}
        self.evaluation_results = []
        self.evaluation_map = {}
        self.text_mode = extras == EVAL_EXTRA_IGNORE_GEOMETRY
        self.is_sequential = False
        self.metrics: typing.Sequence[digem.OCRMetric] = []
        self.evaluation_report = {}

    def eval_all(self, entries: typing.List[EvalEntry], sequential=False) -> None:
        """evaluate all pairs groundtruth-candidate"""

        _entries = []
        if sequential or self.is_sequential:
            _entries = [self._wrap_eval_entry(e) for e in entries]
        else:
            cpus = multiprocessing.cpu_count()
            n_executors = cpus // 2 if cpus > 3 else 1
            if self.verbosity == 1:
                print(
                    f"[DEBUG] use {n_executors} executors ({cpus}) to create evaluation data"
                )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_executors
            ) as executor:
                try:
                    _entries = list(
                        executor.map(
                            self._wrap_eval_entry, entries, timeout=EVAL_TIMEOUT
                        )
                    )
                except concurrent.futures.TimeoutError:
                    print(
                        f"[ERROR] takes longer than {EVAL_TIMEOUT}s to evaluate {len(entries)} entries!"
                    )
                    sys.exit(1)
                except Exception as err:
                    print(f"[ERROR] '{err}' creating evaluation data!")
                    sys.exit(1)

        # review evaluation results
        if _entries:
            _not_nones = [e for e in _entries if e is not None]
            if self.verbosity == 1:
                print(
                    f"[DEBUG] processed {len(_entries)}, omitted {len(_entries) - len(_not_nones)} empty results"
                )
            self.evaluation_entries = _not_nones

        self.evaluation_entries = sorted(
            self.evaluation_entries, key=lambda e: e.path_candidate
        )
        # detail report
        self.evaluation_report["candidates"] = [
            self._generate_report_candidate(e) for e in self.evaluation_entries
        ]

    def _wrap_eval_entry(self, entry: EvalEntry):
        """Wrapper for creation of evaluation data
        to be used in common process-pooling"""

        if entry.path_groundtruth:
            try:
                return self.eval_entry(entry)
            except dc.DigitalObjectException as exc:
                print(f"[WARN ] discard {entry.path_groundtruth}: {exc}")
            except Exception as exc:
                print(f"[WARN ][{entry.path_groundtruth}] {exc}")

    def eval_entry(self, entry: EvalEntry) -> EvalEntry:
        """Create evaluation entry for matching pair of
        groundtruth and candidate data
        rather clumsy copy construct due parallelization
        """

        # evaluate metric copies
        _current_metrics = []

        for metric in self.metrics:
            # read coordinate information (if any provided)
            # to create frame for candidate data
            coords = get_bounding_box(entry.path_groundtruth)
            if coords is not None and self.verbosity >= 2:
                print(f"[TRACE] token coordinates {coords[0]}, {coords[1]}")
            # check if text mode turned on
            if self.text_mode and coords is None:
                print(
                    f"[WARN ] no coordinates for {entry.path_groundtruth} - apply extra '{EVAL_EXTRA_IGNORE_GEOMETRY}'!"
                )
                continue

            coords = None if self.text_mode else coords
            current: digem.OCRMetric = copy.copy(metric)
            current.reference = Path(entry.path_groundtruth).absolute()
            current.candidate = Path(entry.path_candidate).absolute()
            current.candidate_frame = coords
            # ATTENZIONE! inital access to this attribute
            # triggers preprocessing and calculation!
            _ = current.value
            _current_metrics.append(current)
            if self.verbosity >= 2:
                _label_ref = os.path.basename(entry.path_groundtruth)
                _label_can = os.path.basename(entry.path_candidate)
                print(
                    f'[TRACE][{_label_ref}][{current.label}] REFERENCE :: "{current.data_reference}"'
                )
                print(
                    f'[TRACE][{_label_can}][{current.label}] CANDIDATE :: "{current.data_candidate}"'
                )

        # enrich entry with metrics and
        # normalize data type (i.e., art or ann or ...)
        _normed_gt_type = _normalize_gt_type(
            _get_groundtruth_from_filename(entry.path_groundtruth)
        )
        entry.gt_type = _normed_gt_type
        entry.metrics = _current_metrics
        return entry

    def _generate_report_candidate(self, the_entry: EvalEntry):
        try:
            image_name = os.path.basename(the_entry.path_candidate)
            _type = the_entry.gt_type
            if "+" in image_name and "_" in image_name:
                _tkns = image_name.split("_")
                image_name = _tkns[0].replace("+", ":") + "_" + _tkns[1]
            if ".xml" in image_name:
                image_name = image_name.replace(".xml", "")
            gt_label = f"({_type[:3]})" if _type and _type != _NOT_SET else ""
            return f"[{image_name}]{gt_label} [{the_entry}]"
        except Exception as exc:
            print(f"[WARN ] {exc}")

    def _add(self, evaluation_result: EvaluationResult):
        self.evaluation_results.append(evaluation_result)

    def eval_map(self):
        for k, data_tuples in self.evaluation_map.items():
            n_total = len(data_tuples)
            data_points = [e[1] for e in data_tuples]
            n_chars = sum([e[2] for e in data_tuples])

            # set initial result level values
            evaluation_result = EvaluationResult(k, n_total, n_chars=n_chars)
            evaluation_result.mean = data_points[0]
            evaluation_result.median = data_points[0]

            # if more than one single evaluation item
            # calculate additional statistics to reflect
            # impact of outlying data sets
            # take CA and number of GT into account
            # also calculate statistics (mean, std)
            if len(data_points) > 1:
                mean, std, median = get_statistics(data_points)
                evaluation_result.mean = mean
                evaluation_result.median = median
                evaluation_result.std = std
                if std >= 1.0:
                    stripped, _, _ = strip_outliers_from(data_tuples)
                    if len(stripped) < len(data_tuples):
                        regulars_data_points = [e[1] for e in stripped]
                        clear_result = EvaluationResult(k, len(stripped))
                        mean2, std2, med2 = get_statistics(regulars_data_points)
                        clear_result.mean = mean2
                        clear_result.std = std2
                        clear_result.median = med2
                        clear_result.n_chars = sum([e[2] for e in stripped])
                        # set as child component
                        evaluation_result.cleared_result = clear_result
            self._add(evaluation_result)
            # re-order
            self.evaluation_results = sorted(
                self.evaluation_results, key=lambda e: e.eval_key
            )

    def aggregate_generic(
        self,
        strategy: typing.Optional[AggregationStrategy] = None,
        by_metrics: typing.Optional[typing.List[int]] = None,
    ) -> None:
        """Generic aggregation using pluggable strategy

        This is the new, flexible aggregation method that supports arbitrary
        aggregation dimensions through the AggregationStrategy pattern.

        Args:
            strategy: AggregationStrategy defining how to aggregate results.
                     If None, uses default strategy (backward compatible)
            by_metrics: List of metric indices to aggregate (e.g., [0, 1, 2, 3]).
                       If None, aggregates all metrics.

        Example:
            # Aggregate by document type only
            type_strategy = AggregationStrategy([
                AggregationDimension("type", TypeExtractor())
            ])
            evaluator.aggregate_generic(type_strategy)

            # Aggregate by custom metadata
            engine_strategy = AggregationStrategy([
                AggregationDimension("engine", CustomMetadataExtractor("ocr_engine"))
            ])
            evaluator.aggregate_generic(engine_strategy)
        """

        # Preconditions
        self._check_aggregate_preconditions()

        # Default strategy: backward compatible (directory hierarchy + type)
        if strategy is None:
            strategy = self._get_default_aggregation_strategy()

        # Default metrics: all metrics
        if by_metrics is None:
            by_metrics = list(range(len(self.metrics)))

        # Clear existing evaluation map
        self.evaluation_map = {}

        # Aggregate using strategy
        for entry in self.evaluation_entries:
            for metrics_index in by_metrics:
                # Skip if this metric index doesn't exist
                if metrics_index >= len(entry.metrics):
                    continue

                metric = entry.metrics[metrics_index]
                metric_value = metric.value
                metric_refs = metric.n_ref

                # Generate all applicable aggregation keys
                agg_keys = strategy.generate_keys(entry, metric)

                # Add to evaluation map
                for key in agg_keys:
                    if key not in self.evaluation_map:
                        self.evaluation_map[key] = []
                    self.evaluation_map[key].append(
                        (entry.path_candidate, metric_value, metric_refs)
                    )

    def _get_default_aggregation_strategy(self) -> AggregationStrategy:
        """Get default aggregation strategy for backward compatibility

        The default strategy aggregates by:
        1. Directory hierarchy (all levels)
        2. Document type (if present)
        3. Combination of root directory + type

        This mimics the behavior of the original aggregate() method.
        """
        dimensions = [AggregationDimension("", DirectoryHierarchyExtractor())]

        # Also aggregate by type at root level for backward compatibility
        # This is handled by generating composite keys
        strategy = AggregationStrategy(dimensions, hierarchical=False)

        return strategy

    def aggregate(self, by_type=False, by_metrics=None):
        """Aggregate item's metrics for domain/directory and/or annotated type

        [LEGACY METHOD] This method uses a fixed aggregation strategy based on
        filesystem directory structure. For more flexible aggregation, consider
        using aggregate_generic() which supports custom aggregation dimensions.

        Args:
            by_type: If True, also aggregate by document type at root level
            by_metrics: List of metric indices to aggregate (default: [0,1,2,3])

        See also:
            aggregate_generic(): More flexible aggregation with custom strategies
        """

        # precheck - having root dir
        self._check_aggregate_preconditions()
        if by_metrics is None:
            by_metrics = [0, 1, 2, 3]

        # aggregate on each directory
        for _metrics_index in by_metrics:
            for entry in self.evaluation_entries:
                # if we do not have all these different metrics set,
                # do of course not aggregate by non-existing index!
                if _metrics_index >= len(self.evaluation_entries[0].metrics):
                    continue
                # ATTENZIONE! works only when forehand
                # the *real* attribute has been accessed
                # *at least one time*
                # kept this way for testing reasons
                metric_value = entry.metrics[_metrics_index].value
                metric_gt_refs = entry.metrics[_metrics_index].n_ref
                domain_dirs = entry.domain_directories
                for domain in domain_dirs:
                    current_domain = f"{entry.metrics[_metrics_index].label}@{domain}"
                    if current_domain not in self.evaluation_map:
                        self.evaluation_map[current_domain] = []
                    self.evaluation_map[current_domain].append(
                        (entry.path_candidate, metric_value, metric_gt_refs)
                    )
                # if by_type, aggregate type at top level
                if by_type and entry.gt_type and entry.gt_type != _NOT_SET:
                    root_base = entry.candidate_root_domain.name
                    path_key = f"{entry.metrics[_metrics_index].label}@{root_base}"
                    type_key = path_key + "@" + entry.gt_type
                    if type_key not in self.evaluation_map:
                        self.evaluation_map[type_key] = []
                    self.evaluation_map[type_key].append(
                        (entry.path_candidate, metric_value, metric_gt_refs)
                    )

    def _check_aggregate_preconditions(self):
        if not self.evaluation_entries:
            raise RuntimeError("missing evaluation data")
        if not Path(self.domain_candidate).is_dir():
            raise RuntimeError("no candidate root dir to aggregate data from")

    def get_results(self):
        return self.evaluation_results


def get_statistics(data_points):
    """Get common statistics like mean, median and std for data_points"""

    the_mean = np.mean(data_points)
    the_deviation = np.std(data_points)
    the_median = np.median(data_points)
    return (the_mean, the_deviation, the_median)


def strip_outliers_from(data_tuples, fence_ratio=1.5):
    """Determine a data set's outliers by the standard Tukey interquartile range (IQR) fence.

    Q1 and Q3 are estimated as the medians of the lower and upper halves of the
    data respectively.  The fences are then:

        lower_fence = Q1 - fence_ratio * IQR
        upper_fence = Q3 + fence_ratio * IQR

    where IQR = Q3 - Q1.  With the default fence_ratio of 1.5 this identifies
    "mild" outliers as defined by Tukey (1977).

    Args:
        data_tuples: Sequence of (path, metric_value, n_refs) triples.
        fence_ratio: Multiplier applied to the IQR.  Use 1.5 for mild outliers
            and 3.0 for extreme outliers (Tukey, 1977, pp. 44–45).

    Returns:
        Tuple (regulars, q1, q3) where *regulars* is the filtered list of tuples
        and *q1* / *q3* are the quartile estimates used for the fences.

    Reference:
        Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley,
        pp. 44–45.
    """

    data_points = [e[1] for e in data_tuples]
    median = np.median(data_points)
    q1 = np.median([v for v in data_points if v < median])
    q3 = np.median([v for v in data_points if v > median])
    iqr = q3 - q1
    regulars = [
        data
        for data in data_tuples
        if data[1] >= (q1 - fence_ratio * iqr)
        and data[1] <= (q3 + fence_ratio * iqr)
    ]
    return (regulars, q1, q3)


def _get_groundtruth_from_filename(file_path) -> str:
    _file_name = os.path.basename(file_path)
    result = re.match(r".*gt.(\w{3,}).xml$", _file_name)
    if result:
        return result[1]
    else:
        alternative = re.match(r".*\.(\w{3,})\.gt\.xml$", _file_name)
        if alternative:
            return alternative[1]
        else:
            return _NOT_SET


def _normalize_gt_type(label) -> str:
    if label.startswith("art"):
        return "article"
    elif label.startswith("ann"):
        return "announcement"
    else:
        return _NOT_SET


def report_stdout(evaluator: Evaluator, verbosity):
    """Generate report data on stdout"""

    if verbosity >= 1:
        if "candidates" in evaluator.evaluation_report:
            for _c in evaluator.evaluation_report["candidates"]:
                print(f"[DEBUG] {_c}")
    results = evaluator.get_results()
    _path_can = evaluator.domain_candidate
    _path_ref = evaluator.domain_reference
    evaluation_date = datetime.date.today().isoformat()
    print(
        f'[INFO ] Evaluation Summary (candidates: "{_path_can}" vs. reference: "{_path_ref}" ({evaluation_date})'
    )
    for result in results:
        gt_type, n_total, mean_total, med, _n_refs = result.get_defaults()
        add_stats = f" M:{med:5.2f} σ:{result.std:5.2f}" if n_total > 1 else ""
        print(
            f"[INFO ] {gt_type}\t{n_total: 3d} items {_n_refs:_} refs\t∅:{mean_total:5.2f}{add_stats}"
        )
        if result.cleared_result:
            _, n_t2, mean2, med2, n_c2 = result.cleared_result.get_defaults()
            ccr_std = result.cleared_result.std
            drops = n_total - n_t2
            if drops > 0:
                print(
                    f"[INFO ] {gt_type}(-{drops})\t{n_t2: 3d} items {n_c2:_} refs\t∅:{mean2:5.2f} M:{med2:5.2f} σ:{ccr_std:5.2f}"
                )
