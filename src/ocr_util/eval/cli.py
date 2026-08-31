# -*- coding: utf-8 -*-
"""OCR QA Evaluation CLI"""

import argparse
import re
import sys
import typing

from pathlib import Path

import ocr_util.eval as digev
import ocr_util.eval.dictionary_metrics.common as digev_cm
import ocr_util.eval.metrics as digem
from ocr_util.eval.dictionary_metrics.language_tool.LanguageTool import LanguageTool

# script constants
DEFAULT_VERBOSITY = 0
VERBOSITY = DEFAULT_VERBOSITY
EVAL_VERBOSITY = DEFAULT_VERBOSITY
DEFAULT_UTF8_NORM = digev.UC_NORMALIZATION_DEFAULT

# metrics
DEFAULT_OCR_METRICS = "Cs,Ls"
DEFAULT_OCR_METRIC_PREPROCESSINGS = ""
DEFAULT_OCR_METRIC_POSTPROCESSINGS = ""
METRIC_DICT = {
    "Cs": digev.MetricChars,
    "Characters": digev.MetricChars,
    "Ls": digev.MetricLetters,
    "Letters": digev.MetricLetters,
    "Ws": digev.MetricWords,
    "Words": digev.MetricWords,
    "BoWs": digev.MetricBoW,
    "BagOfWords": digev.MetricBoW,
    "IRPre": digev.MetricIRPre,
    "Pre": digev.MetricIRPre,
    "Precision": digev.MetricIRPre,
    "IRRec": digev.MetricIRRec,
    "Rec": digev.MetricIRRec,
    "DictLT": digem.MetricDictionaryLangTool,
    "DictionaryLangTool": digem.MetricDictionaryLangTool,
}

# MODS value transforms: name → callable(str) → str | None
MODS_VALUE_TRANSFORMS = {
    "decade": digev.decade_transform,
    "century": digev.century_transform,
}

# MODS dimension XPath mappings
MODS_DIMENSION_XPATHS = {
    "language": ".//mods:language/mods:languageTerm[@type='code']",
    "languageTerm": ".//mods:language/mods:languageTerm",
    "genre": ".//mods:genre",
    "dateIssued": ".//mods:originInfo/mods:dateIssued",
    "dateCreated": ".//mods:originInfo/mods:dateCreated",
    "publisher": ".//mods:originInfo/mods:publisher",
    "place": ".//mods:originInfo/mods:place/mods:placeTerm",
    "title": ".//mods:titleInfo/mods:title",
    "author": ".//mods:name/mods:namePart",
    "subject": ".//mods:subject/mods:topic",
    "classification": ".//mods:classification",
    "accessCondition": ".//mods:accessCondition",
}

_FILTER_MULTI_VALUE_PATTERN = re.compile(r"[+|,]")


def _split_filter_tokens(value: str) -> typing.Set[str]:
    """Split a metadata value into normalized tokens for set-based matching.

    Token separators: '+', '|', ',' (commonly used for multi-value metadata).
    """
    if value is None:
        return set()
    return {token.strip() for token in re.split(r"[+|,]", str(value)) if token.strip()}


def _filter_value_matches(actual_value: str, expected_value: str) -> bool:
    """Match actual value against expected filter value.

    Matching semantics:
    * If expected value is single-valued, require exact string equality.
    * If expected value is multi-valued (contains '+', '|' or ','),
      treat both values as sets and require expected ⊆ actual.
    """
    expected = expected_value.strip()
    actual = str(actual_value).strip()

    # Exact mode for a single provided value.
    if not _FILTER_MULTI_VALUE_PATTERN.search(expected):
        return actual == expected

    # Set mode for multi-valued filters.
    expected_tokens = _split_filter_tokens(expected)
    actual_tokens = _split_filter_tokens(actual)
    return bool(expected_tokens) and expected_tokens.issubset(actual_tokens)


def _build_filter_spec(
    filter_by: typing.Optional[str],
    mets_path: typing.Optional[Path],
    strict: bool = False,
    verbosity: int = 0,
) -> typing.Optional[typing.Tuple[str, typing.Callable, str]]:
    """Build single filter spec from CLI string.

    Supported format:
      <extractor_spec>=<expected_value>

    Example:
      mods:language=ger
      mods:language=ger+eng
    """
    if not filter_by:
        return None

    if "=" not in filter_by:
        msg = (
            "Invalid --filter-by format. Expected '<extractor_spec>=<value>', "
            f"got '{filter_by}'"
        )
        if strict:
            raise ValueError(msg)
        print(f"[WARN ] {msg}")
        return None

    extractor_spec, expected_value = filter_by.split("=", 1)
    extractor_spec = extractor_spec.strip()
    expected_value = expected_value.strip()

    if not extractor_spec or not expected_value:
        msg = (
            "Invalid --filter-by format. Both extractor and value are required, "
            f"got '{filter_by}'"
        )
        if strict:
            raise ValueError(msg)
        print(f"[WARN ] {msg}")
        return None

    parsed = _parse_extractor_spec(extractor_spec, mets_path, verbosity)
    if parsed is None:
        msg = f"Invalid --filter-by extractor specification: '{extractor_spec}'"
        if strict:
            raise ValueError(msg)
        print(f"[WARN ] {msg}")
        return None

    dim_name, extractor = parsed
    return (dim_name, extractor, expected_value)


def _apply_entry_filter(
    entries: typing.List[digev.EvalEntry],
    filter_spec: typing.Tuple[str, typing.Callable, str],
    verbosity: int = 0,
) -> typing.List[digev.EvalEntry]:
    """Filter entries by one metadata criterion.

    Entries with missing filter value are discarded and reported as WARNING.
    """
    dim_name, extractor, expected_value = filter_spec
    kept: typing.List[digev.EvalEntry] = []
    missing_value_entries: typing.List[digev.EvalEntry] = []
    n_filtered_out = 0

    for entry in entries:
        extracted = extractor(entry)

        # Missing criterion: warn + discard.
        if extracted is None or str(extracted).strip() == "":
            missing_value_entries.append(entry)
            continue

        if _filter_value_matches(str(extracted), expected_value):
            kept.append(entry)
        else:
            n_filtered_out += 1

    if missing_value_entries:
        print(
            f"[WARN ] Discarded {len(missing_value_entries)} entries with missing filter criterion '{dim_name}'"
        )
        if verbosity >= 2:
            for miss in missing_value_entries[:5]:
                print(f"[WARN ] missing '{dim_name}' for '{miss.path_candidate}'")

    if verbosity >= 1:
        print(
            f"[DEBUG] Filter '{dim_name}={expected_value}' kept {len(kept)} of {len(entries)} entries "
            f"(non-matching: {n_filtered_out}, missing: {len(missing_value_entries)})"
        )

    return kept


def _parse_extractor_spec(
    spec: str, mets_path: typing.Optional[Path] = None, verbosity: int = 0
) -> typing.Optional[typing.Tuple[str, typing.Callable]]:
    """Parse extractor specification string and return (name, extractor) tuple

    Supported formats:
        - 'directory' or 'directory:all' - all directory levels
        - 'directory:N' - specific directory level (0=root+1, -1=parent)
        - 'type' - groundtruth type
        - 'metadata:KEY' - custom metadata with key
        - 'pattern:REGEX' - filename pattern (first capture group)
        - 'pattern:REGEX:N' - filename pattern (Nth capture group)
        - 'mods:DIMENSION' - METS/MODS dimension (requires mets_path)

    Args:
        spec: Extractor specification string
        mets_path: Path to METS file (required for 'mods:' extractors)
        verbosity: Verbosity level for diagnostic output

    Returns:
        Tuple of (dimension_name, extractor_callable) or None if invalid
    """
    parts = spec.split(":", 1)
    extractor_type = parts[0].strip()

    # Directory extractor
    if extractor_type == "directory":
        if len(parts) == 1 or parts[1].strip() == "all":
            # All directory levels
            return ("directory", digev.DirectoryHierarchyExtractor())
        else:
            try:
                level = int(parts[1].strip())
                return (
                    f"directory_{level}",
                    digev.DirectoryHierarchyExtractor(level=level),
                )
            except ValueError:
                print(f"[ERROR] Invalid directory level specification: '{spec}'")
                return None

    # Type extractor
    elif extractor_type == "type":
        return ("type", digev.TypeExtractor())

    # Metadata extractor
    elif extractor_type == "metadata":
        if len(parts) < 2:
            print("[ERROR] Metadata extractor requires key: 'metadata:KEY'")
            return None
        key = parts[1].strip()
        return (f"metadata_{key}", digev.CustomMetadataExtractor(key=key))

    # Pattern extractor
    elif extractor_type == "pattern":
        if len(parts) < 2:
            print("[ERROR] Pattern extractor requires regex: 'pattern:REGEX'")
            return None

        # Check if group number is specified
        pattern_parts = parts[1].split(":", 1)
        pattern = pattern_parts[0].strip()
        group = 1
        if len(pattern_parts) > 1:
            try:
                group = int(pattern_parts[1].strip())
            except ValueError:
                print(f"[ERROR] Invalid pattern group number: '{pattern_parts[1]}'")
                return None

        try:
            return (
                "pattern",
                digev.FilenamePatternExtractor(pattern=pattern, group=group),
            )
        except Exception as e:
            print(f"[ERROR] Invalid regex pattern '{pattern}': {e}")
            return None

    # MODS extractor
    elif extractor_type == "mods":
        if not mets_path:
            print("[ERROR] MODS extractor requires --mets-file parameter")
            return None

        if len(parts) < 2:
            print("[ERROR] MODS extractor requires dimension: 'mods:DIMENSION'")
            return None

        # Allow optional transform suffix: mods:DIMENSION:TRANSFORM
        # e.g. mods:dateIssued:decade
        dim_and_transform = parts[1].strip().split(":", 1)
        dimension = dim_and_transform[0].strip()
        transform_name = (
            dim_and_transform[1].strip() if len(dim_and_transform) > 1 else None
        )

        # Validate optional transform
        transform_fn = None
        if transform_name is not None:
            transform_fn = MODS_VALUE_TRANSFORMS.get(transform_name)
            if transform_fn is None:
                print(
                    f"[WARN ] Unknown MODS transform '{transform_name}'. Available: {', '.join(MODS_VALUE_TRANSFORMS.keys())}"
                )
                return None

        # Check if it's a predefined dimension or custom XPath
        if dimension in MODS_DIMENSION_XPATHS:
            xpath = MODS_DIMENSION_XPATHS[dimension]
            dim_name = (
                f"mods_{dimension}"
                if transform_name is None
                else f"mods_{dimension}_{transform_name}"
            )
        elif dimension.startswith(".//"):  # Custom XPath
            xpath = dimension
            dim_name = (
                "mods_custom"
                if transform_name is None
                else f"mods_custom_{transform_name}"
            )
        else:
            print(
                f"[WARN ] Unknown MODS dimension '{dimension}'. Available: {', '.join(MODS_DIMENSION_XPATHS.keys())}"
            )
            print("[WARN ] Or provide a custom XPath expression starting with './/'")
            return None

        try:
            base_extractor = digev.METSModsExtractor(
                mets_file_path=mets_path, xpath_expression=xpath
            )
            extractor = (
                digev.ValueTransformExtractor(base_extractor, transform_fn)
                if transform_fn is not None
                else base_extractor
            )
            return (dim_name, extractor)
        except Exception as e:
            print(f"[ERROR] Failed to create MODS extractor for '{dimension}': {e}")
            return None

    # mets:div attribute extractor  (mets:type, mets:label, or any mets:ATTR)
    elif extractor_type == "mets":
        if not mets_path:
            print("[ERROR] mets: extractor requires --mets-file parameter")
            return None

        if len(parts) < 2:
            print(
                "[ERROR] mets: extractor requires attribute name: 'mets:type' or 'mets:label'"
            )
            return None

        attr_spec = parts[1].strip().upper()  # e.g. "type" → "TYPE"
        dim_name = f"mets_div_{attr_spec.lower()}"

        try:
            extractor = digev.METSDivAttrExtractor(
                mets_file_path=mets_path,
                attribute=attr_spec,
            )
            return (dim_name, extractor)
        except Exception as e:
            print(f"[ERROR] Failed to create mets:div extractor for '{attr_spec}': {e}")
            return None

    else:
        print(
            f"[ERROR] Unknown extractor type '{extractor_type}'. Available: directory, type, metadata, pattern, mods, mets"
        )
        return None


def _build_aggregation_strategy(
    aggregate_by: typing.Optional[str],
    mets_path: typing.Optional[Path],
    strict: bool = False,
    verbosity: int = 0,
) -> typing.Optional[digev.AggregationStrategy]:
    """Build aggregation strategy from CLI specification

    Two separator modes are supported:

    * Comma ``,`` — flat/independent dimensions.  Each dimension produces its
      own aggregation keys independently::

          mods:language,mods:dateIssued:century
          → Cs@mods_language:ger
          → Cs@mods_dateIssued_century:19th

    * Plus ``+`` — hierarchical/combined dimensions.  Produces individual keys
      *and* a composite key that joins all dimension values::

          mods:language+mods:dateIssued:century
          → Cs@mods_language:ger
          → Cs@mods_dateIssued_century:19th
          → Cs@mods_language:ger/mods_dateIssued_century:19th

    Args:
        aggregate_by: Extractor specifications separated by ``,`` or ``+``
        mets_path: Path to METS file (for MODS extractors)
        strict: Raise ValueError on any invalid specification
        verbosity: Verbosity level

    Returns:
        AggregationStrategy or None if no valid dimensions

    Raises:
        ValueError: If strict=True and at least one specification is invalid.
    """
    if not aggregate_by:
        return None

    # Determine separator and hierarchical mode
    hierarchical = "+" in aggregate_by
    separator = "+" if hierarchical else ","

    # Parse extractor specifications
    specs = [s.strip() for s in aggregate_by.split(separator)]
    dimensions = []

    invalid_specs = []
    for spec in specs:
        result = _parse_extractor_spec(spec, mets_path, verbosity)
        if result:
            dim_name, extractor = result
            dimensions.append(digev.AggregationDimension(dim_name, extractor))
            if verbosity >= 1:
                print(
                    f"[DEBUG] Added aggregation dimension '{dim_name}' from spec '{spec}'"
                )
        else:
            invalid_specs.append(spec)

    if strict and invalid_specs:
        raise ValueError(
            f"Invalid aggregation dimension specification(s): {', '.join(invalid_specs)}"
        )

    if not dimensions:
        return None

    return digev.AggregationStrategy(dimensions, hierarchical=hierarchical)


def _initialize_metrics(the_metrics, norm) -> typing.Sequence[digem.OCRMetric]:
    _tokens = the_metrics.split(",")
    try:
        metric_objects: typing.List[digem.OCRMetric] = []
        for m in _tokens:
            clazz: typing.Type[digem.OCRMetric] = METRIC_DICT[m]
            metric_inst: digem.OCRMetric = clazz()
            if isinstance(metric_inst, digem.SimilarityMetric):
                metric_inst.code_norm = norm
            metric_objects.append(metric_inst)
        return metric_objects
    except KeyError as _err:
        _keys = ",".join(METRIC_DICT.keys())
        _msg = f"Unknown: '{_err.args[0]}'.\nPlease use one of the following keys: '{_keys}'."
        print(_msg)
        sys.exit(1)


#########
# START #
#########
def start_evaluation(parse_args: typing.Dict):
    """Main workflow"""

    path_candidates = parse_args["candidates"]
    path_reference = parse_args["reference"]
    metrics: str = parse_args["metrics"]
    utf8norm = parse_args["utf8"]
    verbosity = parse_args["verbosity"]
    is_seq = parse_args["sequential"] if "sequential" in parse_args else False
    xtra = parse_args["extra"] if "extra" in parse_args else None
    no_outliers = parse_args.get("no_outliers", False)
    weighted_mean = parse_args.get("weighted_mean", False)

    if "language" in parse_args:
        digem.MetricDictionary.LANGUAGE = parse_args["language"]
    uses_lang_tool: bool = "DictLT" in metrics or "DictionaryLangTool" in metrics
    if uses_lang_tool:
        lt_url: str = (
            parse_args["lt_api_url"]
            if "lp_api_url" in parse_args
            else LanguageTool.DEFAULT_URL
        )
        LanguageTool.initialize(lt_url)

    # go on with basic validation
    path_candidates = Path(path_candidates).absolute()
    if not path_candidates.is_dir() and not path_candidates.is_file():
        print(f'[ERROR] input "{path_candidates}": invalid! exit!')
        sys.exit(1)
    if path_reference is not None:
        path_reference = Path(path_reference).absolute()
        if not path_reference.is_dir() and not path_reference.is_file():
            print(f'[ERROR] reference "{path_reference}": invalid! exit!')
            sys.exit(1)
        if path_reference.is_file() and path_candidates.is_dir():
            print(
                "[ERROR] a reference file requires a single candidate file! exit!"
            )
            sys.exit(1)

    if path_candidates and path_reference:
        base_can = (
            path_candidates.name
            if path_candidates.is_dir()
            else path_candidates.parent.name
        )
        base_ref = (
            path_reference.name
            if path_reference.is_dir()
            else path_reference.parent.name
        )
        if base_can != base_ref:
            print(
                f"[WARN ] base '{base_can}' and '{base_ref}' mismatch, aggregation might be inaccurate!"
            )

    # some diagnostics
    if verbosity >= 2:
        args = f"{path_candidates}, {path_reference}, {verbosity}, {xtra}"
        print(f"[DEBUG] called with {args}")

    # --- Validate aggregation strategy before any expensive work ---
    aggregate_by = parse_args.get("aggregate_by")
    filter_by = parse_args.get("filter_by")
    mets_file = parse_args.get("mets_file")
    mods_dimensions = parse_args.get("mods_dimensions")

    # Validate and resolve METS path upfront
    mets_path = None
    if mets_file:
        mets_path = Path(mets_file)
        if not mets_path.exists():
            print(f"[ERROR] METS file '{mets_file}' not found! exit!")
            sys.exit(1)

    # Build and validate aggregation strategy; always strict so any bad spec
    # aborts before evaluation starts.
    strategy = None
    filter_spec = None
    if aggregate_by:
        try:
            strategy = _build_aggregation_strategy(
                aggregate_by,
                mets_path,
                strict=True,
                verbosity=verbosity,
            )
        except ValueError as err:
            print(f"[ERROR] {err}. exit!")
            sys.exit(1)
        if not strategy:
            print(
                "[WARN ] No valid aggregation dimensions provided. Using default aggregation."
            )
    elif mods_dimensions:
        if not mets_path:
            print("[ERROR] --mods-dimensions requires --mets-file! exit!")
            sys.exit(1)
        dimension_names = [d.strip() for d in mods_dimensions.split(",")]
        unified_specs = [f"mods:{dim}" for dim in dimension_names]
        aggregate_by_converted = ",".join(unified_specs)
        if verbosity >= 1:
            print(
                f"[DEBUG] Converting legacy --mods-dimensions to unified format: {aggregate_by_converted}"
            )
        try:
            strategy = _build_aggregation_strategy(
                aggregate_by_converted,
                mets_path,
                strict=True,
                verbosity=verbosity,
            )
        except ValueError as err:
            print(f"[ERROR] {err}. exit!")
            sys.exit(1)
        if not strategy:
            print(
                "[WARN ] No valid MODS dimensions provided. Using default aggregation."
            )

    # Build and validate single-entry filter; always strict so invalid specs
    # abort before evaluation starts.
    if filter_by:
        try:
            filter_spec = _build_filter_spec(
                filter_by,
                mets_path,
                strict=True,
                verbosity=verbosity,
            )
        except ValueError as err:
            print(f"[ERROR] {err}. exit!")
            sys.exit(1)

    # create basic evaluator instance
    # If a single file is passed, use its parent directory as the root
    evaluator_root = (
        path_candidates if path_candidates.is_dir() else path_candidates.parent
    )
    evaluator = digev.Evaluator(
        evaluator_root,
        verbosity=verbosity,
        extras=xtra,
        strict_mode=no_outliers,
        weighted_mean=weighted_mean,
    )
    evaluator.metrics = _initialize_metrics(metrics, norm=utf8norm)  # , calc=calc)
    if verbosity >= 1:
        print(f"[DEBUG] text normalized using '{utf8norm}' code points for '{metrics}'")

    evaluator.is_sequential = is_seq
    if path_reference is not None:
        evaluator.domain_reference = path_reference

    # gather structure information
    candidates: typing.List[digev.EvalEntry] = digev.gather_candidates(path_candidates)
    if len(candidates) == 0:
        print(
            f"[WARN ] no ocr data (.*xml) in dir starting from '{path_candidates}'! exit."
        )
        sys.exit(0)

    # match groundtruth
    if path_reference:
        if path_reference.is_file():
            candidates[0].path_groundtruth = path_reference
            candidates[0].align_domains()
        else:
            for entry in candidates:
                gt = digev.find_groundtruth(entry, path_reference)
                if gt:
                    entry.path_groundtruth = gt
                    entry.align_domains()

    # remove all paths where no groundtruth exists
    gt_entries = [c for c in candidates if c.path_groundtruth]
    n_entries = len(candidates)
    n_diff = n_entries - len(gt_entries)
    gt_missing = set(gt_entries) ^ set(candidates)
    rnd_str = f" ({gt_missing})" if gt_missing else ""
    if verbosity >= 1:
        print(
            f'[DEBUG] from "{n_entries}" filtered "{n_diff}" candidates missing groundtruth{rnd_str}'
        )

    # Optional metadata/value filter stage before expensive metric evaluation.
    if filter_spec:
        n_before_filter = len(gt_entries)
        filter_dim_name, _, filter_value = filter_spec
        if verbosity >= 1:
            print(
                f"[DEBUG] Applying filter: {filter_dim_name}={filter_value} on {n_before_filter} entries"
            )
        gt_entries = _apply_entry_filter(gt_entries, filter_spec, verbosity=verbosity)
        n_after_filter = len(gt_entries)
        n_filtered_out = n_before_filter - n_after_filter
        pct_filtered = (
            100.0 * n_filtered_out / n_before_filter if n_before_filter > 0 else 0
        )
        print(
            f"[INFO ] Filter '{filter_dim_name}={filter_value}' result: {n_after_filter}/{n_before_filter} entries "
            f"({pct_filtered:.1f}% filtered out)"
        )
        if len(gt_entries) == 0:
            print("[WARN ] No entries left after applying filter. exit.")
            sys.exit(0)

    # trigger actual evaluation
    evaluator.eval_all(gt_entries)

    # Apply aggregation strategy
    if strategy:
        if verbosity >= 1:
            dim_names = [dim.name for dim in strategy.dimensions]
            hierarchical_str = "(hierarchical)" if strategy.hierarchical else "(flat)"
            print(
                f"[DEBUG] Aggregating by {len(strategy.dimensions)} dimension(s) {hierarchical_str}: {', '.join(dim_names)}"
            )
        evaluator.aggregate_generic(strategy)
        if verbosity >= 1:
            print(
                f"[DEBUG] Aggregation complete: {len(evaluator.evaluation_map)} aggregation keys generated"
            )
    else:
        # Use default aggregation (backward compatible)
        if verbosity >= 1:
            print("[DEBUG] Using default aggregation strategy (directory hierarchy + type)")
        evaluator.aggregate(by_type=True)
        if verbosity >= 1:
            print(
                f"[DEBUG] Aggregation complete: {len(evaluator.evaluation_map)} aggregation keys generated"
            )

    # evaluator.evaluate()
    evaluator.eval_map()

    # serialize stdout report
    if verbosity >= 0:
        digev.report_stdout(evaluator, verbosity)

    # for testing purposes
    eval_results = evaluator.get_results()

    # final clean-up
    if uses_lang_tool:
        LanguageTool.deinitialize()

    return eval_results


def register_arguments(parser: argparse.ArgumentParser) -> None:
    """Register eval subcommand arguments on any argparse parser.

    This function is the single source of truth for eval CLI arguments and is
    used by both the standalone eval entrypoint and the top-level `ocr eval`
    subcommand wiring.
    """

    parser.add_argument(
        "candidates",
        help="Root Directory for evaluation candidates / Path to single candidate file",
    )
    parser.add_argument(
        "-ref",
        "--reference",
        required=False,
        help="Root directory for Reference/Groundtruth data or a single reference file when candidates is a file (optional, but necessary for most metrics)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_OCR_METRICS,
        required=False,
        help=f"List of metrics to use (optional, default: '{DEFAULT_OCR_METRICS}'; available: '{','.join(METRIC_DICT.keys())}')",
    )
    parser.add_argument(
        "--utf8",
        default=DEFAULT_UTF8_NORM,
        required=False,
        help=f"UTF-8 Unicode Python Normalization (optional; default: '{DEFAULT_UTF8_NORM}'; available: 'NFC','NFKC','NFD','NFKD')",
    )
    parser.add_argument(
        "-s",
        "--sequential",
        action="store_true",
        required=False,
        help="Execute calculations sequentially (optional; default: 'False')",
    )
    parser.add_argument(
        "--no-outliers",
        dest="no_outliers",
        action="store_true",
        default=False,
        required=False,
        help="Disable outlier detection; report only raw statistics for every group (default: False)",
    )
    parser.add_argument(
        "-x",
        "--extra",
        required=False,
        help="pass additional information to evaluation, like 'ignore_geometry' (compare only text, ignore coords)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=digev_cm.LANGUAGE_KEY_DEFAULT,
        choices=digev_cm.LANGUAGE_KEYS,
        required=False,
        help=f"Language code for LanguagTool according to ISO 639-2 (optional; default: '{digev_cm.LANGUAGE_KEY_DEFAULT}')",
    )
    parser.add_argument(
        "-u",
        "--lt-api-url",
        default=LanguageTool.DEFAULT_URL,
        required=False,
        help=f"Language Tool Api URL (optional; default: '{LanguageTool.DEFAULT_URL}')",
    )
    parser.add_argument(
        "--filter-by",
        required=False,
        help="Single pre-aggregation filter criterion (optional). "
        "Format: '<extractor_spec>=<value>' where extractor_spec follows --aggregate-by formats, "
        "e.g. 'mods:language=ger' (exact) or 'mods:language=ger+eng' (set containment). "
        "Entries missing the criterion are warned and discarded.",
    )
    parser.add_argument(
        "--aggregate-by",
        required=False,
        help="Comma-separated list of aggregation dimensions (optional). "
        "Formats: 'directory' (all levels), 'directory:N' (level N), 'type', "
        "'metadata:KEY', 'pattern:REGEX', 'mods:DIMENSION' (requires --mets-file). "
        f"Available MODS dimensions: {', '.join(MODS_DIMENSION_XPATHS.keys())}",
    )
    parser.add_argument(
        "--mets-file",
        required=False,
        help="Path to METS/MODS file for MODS-based aggregation (optional, required for 'mods:' dimensions)",
    )
    parser.add_argument(
        "--mods-dimensions",
        required=False,
        help=f"[LEGACY] Comma-separated list of MODS dimensions for aggregation (optional; requires --mets-file). "
        f"Use --aggregate-by instead for unified approach. "
        f"Available: {', '.join(MODS_DIMENSION_XPATHS.keys())}. "
        f"Or provide custom XPath expressions starting with './/'",
    )
    parser.add_argument(
        "--weighted-mean",
        action="store_true",
        required=False,
        help="Use weighted mean for summary statistics (weights = number of GT references per data point). Default: False",
    )


def start():
    """Wrap argparsing"""
    parser = argparse.ArgumentParser()
    register_arguments(parser)
    main_args = vars(parser.parse_args())
    start_evaluation(main_args)


if __name__ == "__main__":
    start()
