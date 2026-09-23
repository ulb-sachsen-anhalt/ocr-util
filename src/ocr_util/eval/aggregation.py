# -*- coding: utf-8 -*-
"""Aggregation and Extraction Module

This module provides classes for aggregating evaluation results across various dimensions.
It supports multiple extraction strategies including filesystem hierarchy, document types,
custom metadata, filename patterns, and METS/MODS metadata.
"""

from __future__ import annotations

import re
import typing
from pathlib import Path

import lxml.etree as ET

import ocr_util.eval.metrics as digem

# mark unset values as 'not available'
_NOT_SET = "n.a."


class AggregationDimension:
    """Represents a dimension along which evaluation results can be aggregated

    An aggregation dimension defines a property or attribute that can be extracted
    from evaluation entries to group and aggregate results. For example, directory
    level, document type, date, or custom metadata.

    Args:
        name: Human-readable name for this dimension (e.g., "directory", "type")
        extractor: Callable that extracts the dimension value from an EvalEntry
    """

    def __init__(self, name: str, extractor: typing.Callable[[typing.Any], typing.Any]):
        self.name = name
        self.extractor = extractor


class DirectoryHierarchyExtractor:
    """Extract directory names from filesystem hierarchy

    Extracts directory names at specific levels relative to the candidate root.
    Maintains backward compatibility with the original directory-based aggregation.

    Args:
        level: Directory level to extract (-1 for immediate parent, 0 for root+1, etc.)
               If None, extracts all intermediate directories
    """

    def __init__(self, level: typing.Optional[int] = None):
        self.level = level

    def __call__(self, entry: typing.Any) -> typing.Union[str, typing.List[str], None]:
        """Extract directory name(s) from entry"""
        if not hasattr(entry, "domain_directories"):
            return None

        dirs = entry.domain_directories
        if not dirs:
            return None

        if self.level is None:
            # Return all directories
            return dirs
        elif self.level == -1:
            # Return immediate parent
            return dirs[0] if dirs else None
        elif 0 <= self.level < len(dirs):
            # Return specific level (reversed, since dirs are bottom-up)
            return dirs[-(self.level + 1)]
        return None


class TypeExtractor:
    """Extract groundtruth type from evaluation entry

    Extracts document type annotations (e.g., "article", "announcement")
    from groundtruth filenames.
    """

    def __call__(self, entry: typing.Any) -> typing.Optional[str]:
        """Extract GT type from entry"""
        if hasattr(entry, "gt_type") and entry.gt_type != _NOT_SET:
            return entry.gt_type
        return None


class CustomMetadataExtractor:
    """Extract custom metadata from entry tags

    Allows aggregation by arbitrary metadata attached to evaluation entries.

    Args:
        key: Metadata key to extract from entry.tags dictionary
        default: Default value if key not found
    """

    def __init__(self, key: str, default: typing.Any = None):
        self.key = key
        self.default = default

    def __call__(self, entry: typing.Any) -> typing.Any:
        """Extract metadata value from entry"""
        if hasattr(entry, "tags"):
            return entry.tags.get(self.key, self.default)
        return self.default


class FilenamePatternExtractor:
    """Extract values from filename using regex pattern

    Useful for extracting dates, identifiers, or other structured information
    from standardized filename patterns.

    Args:
        pattern: Regular expression pattern with one capturing group
        group: Which capturing group to extract (default: 1)
    """

    def __init__(self, pattern: str, group: int = 1):
        self.pattern = re.compile(pattern)
        self.group = group

    def __call__(self, entry: typing.Any) -> typing.Optional[str]:
        """Extract pattern match from filename"""
        if hasattr(entry, "path_candidate"):
            filename = entry.path_candidate.name
            match = self.pattern.search(filename)
            if match and len(match.groups()) >= self.group:
                return match.group(self.group)
        return None


def decade_transform(value: str) -> typing.Optional[str]:
    """Transform a 4-digit year string into its decade bucket (e.g. '1867' → '1860s')."""
    try:
        year = int(str(value).strip()[:4])
        return f"{(year // 10) * 10}s"
    except (ValueError, TypeError):
        return None


def century_transform(value: str) -> typing.Optional[str]:
    """Transform a 4-digit year string into its century bucket (e.g. '1867' → '19th')."""
    try:
        year = int(str(value).strip()[:4])
        century = year // 100 + 1
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(century % 10 if century % 100 not in (11, 12, 13) else 0, "th")
        return f"{century}{suffix}"
    except (ValueError, TypeError):
        return None


class ValueTransformExtractor:
    """Wrap any extractor and apply a transform function to its output value.

    Useful for bucketing continuous values such as dates into ranges (e.g.
    decade buckets).

    Args:
        inner: Base extractor callable
        transform: Function applied to the extracted value; should return
                   a string or None (None means the entry is excluded from
                   this dimension)
    """

    def __init__(
        self,
        inner: typing.Callable[[typing.Any], typing.Any],
        transform: typing.Callable[[typing.Any], typing.Optional[str]],
    ):
        self.inner = inner
        self.transform = transform

    def __call__(self, entry: typing.Any) -> typing.Optional[str]:
        value = self.inner(entry)
        if value is None:
            return None
        return self.transform(value)


class METSDivAttrExtractor:
    """Extract mets:div attribute values from the LOGICAL structMap.

    Maps each groundtruth file to its linked logical ``mets:div`` element via
    the structLink section and returns the requested XML attribute value.

    Typical use cases:
    * ``attribute="TYPE"``  → structural type of the logical unit
      (e.g. ``"section"``, ``"chapter"``, ``"article"``, ``"preface"``)
    * ``attribute="LABEL"`` → human-readable label of the logical unit

    Lookup strategy (most-specific first):

    1. **structLink** — ``mets:smLink xlink:to=physId`` gives the linked
       logical div; its attribute is used.
    2. **DMDID fallback** — if no structLink is present, look up the logical
       div that carries the same ``DMDID`` as the file's ``fileGrp``.

    Args:
        mets_file_path: Path to the METS file
        attribute: ``mets:div`` attribute name to extract (e.g. ``"TYPE"``)
        namespaces: Optional namespace mapping override
        cache_parsed: Cache the parsed file-to-attribute mapping (default True)
    """

    DEFAULT_NAMESPACES = {
        "mets": "http://www.loc.gov/METS/",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    def __init__(
        self,
        mets_file_path: Path,
        attribute: str,
        namespaces: typing.Optional[typing.Dict[str, str]] = None,
        cache_parsed: bool = True,
    ):
        self.mets_file_path = mets_file_path
        self.attribute = attribute
        self.namespaces = namespaces or self.DEFAULT_NAMESPACES
        self.cache_parsed = cache_parsed
        self._file_to_attr_map: typing.Optional[typing.Dict[str, str]] = None

    def _build_map(self) -> typing.Dict[str, str]:
        """Parse METS file and build file_href → attribute-value mapping."""
        if self.cache_parsed and self._file_to_attr_map is not None:
            return self._file_to_attr_map

        if not self.mets_file_path.exists():
            raise FileNotFoundError(f"METS file not found: {self.mets_file_path}")

        tree = ET.parse(str(self.mets_file_path))
        ns = self.namespaces
        xlink_ns = ns.get("xlink", "http://www.w3.org/1999/xlink")

        # file ID → href
        file_id_to_href: typing.Dict[str, str] = {}
        for file_elem in tree.xpath("//mets:file", namespaces=ns):
            fid = file_elem.get("ID")
            flocat = file_elem.xpath("./mets:FLocat/@xlink:href", namespaces=ns)
            if fid and flocat:
                file_id_to_href[fid] = flocat[0]

        # physical div ID → [file hrefs]
        phys_to_hrefs: typing.Dict[str, typing.List[str]] = {}
        for pdiv in tree.xpath('//mets:structMap[@TYPE="PHYSICAL"]//mets:div[@ID]', namespaces=ns):
            pid = pdiv.get("ID")
            hrefs = [
                file_id_to_href[fid]
                for fid in pdiv.xpath("./mets:fptr/@FILEID", namespaces=ns)
                if fid in file_id_to_href
            ]
            if pid and hrefs:
                phys_to_hrefs[pid] = hrefs

        # logical div ID → attribute value
        log_to_attr: typing.Dict[str, str] = {}
        # also: DMDID → attribute value (fallback)
        dmdid_to_attr: typing.Dict[str, str] = {}
        for ldiv in tree.xpath('//mets:structMap[@TYPE="LOGICAL"]//mets:div[@ID]', namespaces=ns):
            lid = ldiv.get("ID")
            attr_val = ldiv.get(self.attribute)
            if lid and attr_val:
                log_to_attr[lid] = attr_val
            dmdid = ldiv.get("DMDID")
            if dmdid and attr_val:
                for d in dmdid.split():
                    dmdid_to_attr.setdefault(d, attr_val)

        # physical div ID → [logical div IDs] (reverse of structLink)
        phys_to_logical: typing.Dict[str, typing.List[str]] = {}
        for link in tree.xpath("//mets:structLink/mets:smLink", namespaces=ns):
            logical_id = link.get(f"{{{xlink_ns}}}from")
            physical_id = link.get(f"{{{xlink_ns}}}to")
            if logical_id and physical_id:
                phys_to_logical.setdefault(physical_id, []).append(logical_id)

        file_to_attr: typing.Dict[str, str] = {}

        # Primary: structLink resolution
        for phys_id, hrefs in phys_to_hrefs.items():
            for log_id in phys_to_logical.get(phys_id, []):
                attr_val = log_to_attr.get(log_id)
                if attr_val:
                    for href in hrefs:
                        file_to_attr.setdefault(href, attr_val)
                    break

        # Fallback: DMDID on file / fileGrp → logical div attribute
        for file_elem in tree.xpath("//mets:file", namespaces=ns):
            flocat = file_elem.xpath("./mets:FLocat/@xlink:href", namespaces=ns)
            if not flocat:
                continue
            href = flocat[0]
            if href in file_to_attr:
                continue
            dmdid = file_elem.get("DMDID")
            if not dmdid:
                parent = file_elem.getparent()
                if parent is not None:
                    dmdid = parent.get("DMDID")
            if dmdid:
                for d in dmdid.split():
                    if d in dmdid_to_attr:
                        file_to_attr[href] = dmdid_to_attr[d]
                        break

        if self.cache_parsed:
            self._file_to_attr_map = file_to_attr

        return file_to_attr

    def __call__(self, entry: typing.Any) -> typing.Optional[str]:
        """Extract mets:div attribute for entry's groundtruth file."""
        if not hasattr(entry, "path_groundtruth") or entry.path_groundtruth is None:
            return None
        try:
            file_to_attr = self._build_map()
            gt_filename = entry.path_groundtruth.name
            gt_path_str = str(entry.path_groundtruth)
            for href, attr_val in file_to_attr.items():
                if gt_filename in href or href in gt_path_str:
                    return attr_val
            return None
        except Exception as e:
            raise RuntimeError from e


class METSModsExtractor:
    """Extract MODS metadata from METS/MODS files

    Extracts metadata values from MODS sections embedded in METS files.
    This extractor assumes groundtruth files are referenced as filePointers
    in a METS file with corresponding MODS metadata sections.

    The METS file should structure like:
    - mets:fileSec contains mets:file elements with file pointers
    - mets:dmdSec contains mods:mods elements with metadata
    - Files are linked to metadata via DMDID references

    Args:
        mets_file_path: Path to the METS/MODS file
        xpath_expression: XPath expression to extract MODS element value
                         (e.g., ".//mods:language/mods:languageTerm[@type='code']")
        namespaces: Optional custom namespace mapping (default: standard METS/MODS)
        cache_parsed: If True, caches parsed METS document (default: True)

    Example:
        # Extract language code from MODS
        extractor = METSModsExtractor(
            mets_file_path=Path("path/to/mets.xml"),
            xpath_expression=".//mods:language/mods:languageTerm[@type='code']"
        )

        # Extract genre
        extractor = METSModsExtractor(
            mets_file_path=Path("path/to/mets.xml"),
            xpath_expression=".//mods:genre"
        )
    """

    # Standard METS/MODS namespaces
    DEFAULT_NAMESPACES = {
        "mets": "http://www.loc.gov/METS/",
        "mods": "http://www.loc.gov/mods/v3",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    def __init__(
        self,
        mets_file_path: Path,
        xpath_expression: str,
        namespaces: typing.Optional[typing.Dict[str, str]] = None,
        cache_parsed: bool = True,
    ):
        self.mets_file_path = mets_file_path
        self.xpath_expression = xpath_expression
        self.namespaces = namespaces or self.DEFAULT_NAMESPACES
        self.cache_parsed = cache_parsed
        self._parsed_tree = None
        self._file_to_mods_map = None

    def _parse_mets_file(self):
        """Parse METS file and build file-to-MODS mapping

        Returns parsed lxml tree and mapping dict.

        The mapping prefers page-level logical DMDIDs linked via structLink over
        broad fileGrp-level DMDIDs when both are present.
        """
        if self.cache_parsed and self._parsed_tree is not None:
            return self._parsed_tree, self._file_to_mods_map

        if not self.mets_file_path.exists():
            raise FileNotFoundError(f"METS file not found: {self.mets_file_path}")

        # Parse METS file
        tree = ET.parse(str(self.mets_file_path))

        # Build file-to-MODS mapping (fallback from file/fileGrp DMDID)
        file_map = {}
        file_id_to_href = {}

        files = tree.xpath("//mets:file", namespaces=self.namespaces)
        for file_elem in files:
            file_id = file_elem.get("ID")
            flocat = file_elem.xpath("./mets:FLocat/@xlink:href", namespaces=self.namespaces)
            if not flocat:
                continue

            file_href = flocat[0]
            if file_id:
                file_id_to_href[file_id] = file_href

            # Get DMDID from file element first, then parent fileGrp.
            dmdid = file_elem.get("DMDID")
            if not dmdid:
                parent = file_elem.getparent()
                if parent is not None:
                    dmdid = parent.get("DMDID")

            if dmdid:
                file_map[file_href] = dmdid.split()

        # For complex METS, resolve per-file DMDIDs through
        # LOGICAL -> structLink -> PHYSICAL -> fptr(FILEID) -> FLocat(xlink:href).
        logical_div_to_dmdids = {}
        logical_divs = tree.xpath(
            '//mets:structMap[@TYPE="LOGICAL"]//mets:div[@ID]',
            namespaces=self.namespaces,
        )
        for logical_div in logical_divs:
            logical_id = logical_div.get("ID")
            logical_dmdid = logical_div.get("DMDID")
            if logical_id and logical_dmdid:
                logical_div_to_dmdids[logical_id] = logical_dmdid.split()

        physical_div_to_hrefs = {}
        physical_divs = tree.xpath(
            '//mets:structMap[@TYPE="PHYSICAL"]//mets:div[@ID]',
            namespaces=self.namespaces,
        )
        for physical_div in physical_divs:
            physical_id = physical_div.get("ID")
            if not physical_id:
                continue

            hrefs = []
            for file_id in physical_div.xpath("./mets:fptr/@FILEID", namespaces=self.namespaces):
                file_href = file_id_to_href.get(file_id)
                if file_href:
                    hrefs.append(file_href)

            if hrefs:
                physical_div_to_hrefs[physical_id] = hrefs

        xlink_ns = self.namespaces.get("xlink", "http://www.w3.org/1999/xlink")
        for link in tree.xpath("//mets:structLink/mets:smLink", namespaces=self.namespaces):
            logical_id = link.get(f"{{{xlink_ns}}}from")
            physical_id = link.get(f"{{{xlink_ns}}}to")
            if not logical_id or not physical_id:
                continue

            dmdids = logical_div_to_dmdids.get(logical_id)
            hrefs = physical_div_to_hrefs.get(physical_id)
            if not dmdids or not hrefs:
                continue

            # structLink mapping is more specific than fileGrp-level DMDID.
            for file_href in hrefs:
                file_map[file_href] = dmdids

        if self.cache_parsed:
            self._parsed_tree = tree
            self._file_to_mods_map = file_map

        return tree, file_map

    def _extract_mods_value(self, tree, dmdid: str) -> typing.Optional[str]:
        """Extract MODS metadata value for given DMDID

        Args:
            tree: Parsed lxml tree
            dmdid: DMD section ID

        Returns:
            Extracted metadata value or None
        """
        # Find dmdSec with matching ID
        dmd_xpath = f'//mets:dmdSec[@ID="{dmdid}"]//mods:mods'
        mods_sections = tree.xpath(dmd_xpath, namespaces=self.namespaces)

        if not mods_sections:
            return None

        # Apply user's XPath expression to MODS section
        for mods_section in mods_sections:
            try:
                results = mods_section.xpath(self.xpath_expression, namespaces=self.namespaces)
                if results:
                    values = []
                    for result in results:
                        if hasattr(result, "text") and result.text:
                            values.append(result.text)
                        elif isinstance(result, str):
                            values.append(result)
                    if values:
                        return "+".join(values)
            except Exception as e:
                raise RuntimeError from e

        return None

    def __call__(self, entry: typing.Any) -> typing.Optional[str]:
        """Extract MODS metadata value for entry's groundtruth file

        Args:
            entry: EvalEntry with path_groundtruth attribute

        Returns:
            Extracted MODS metadata value or None if not found
        """
        if not hasattr(entry, "path_groundtruth") or entry.path_groundtruth is None:
            return None

        try:
            # Parse METS file and get mapping
            tree, file_map = self._parse_mets_file()
            assert tree is not None and file_map is not None

            # Get groundtruth filename (may need to match various href formats)
            gt_filename = entry.path_groundtruth.name
            gt_path_str = str(entry.path_groundtruth)

            # Try to find matching file entry in METS
            matched_dmdids = None
            for file_href, dmdids in file_map.items():
                # Match by filename or relative path
                if gt_filename in file_href or file_href in gt_path_str:
                    matched_dmdids = dmdids
                    break

            if not matched_dmdids:
                return None

            # Extract metadata from first matching DMDID
            for dmdid in matched_dmdids:
                value = self._extract_mods_value(tree, dmdid)
                if value:
                    return value

            return None

        except Exception as e:
            raise RuntimeError from e


class AggregationStrategy:
    """Defines how to aggregate evaluation results across dimensions

    An aggregation strategy specifies one or more dimensions along which
    evaluation results should be grouped and aggregated. Each dimension
    can extract different properties from evaluation entries.

    Args:
        dimensions: List of aggregation dimensions to apply
        hierarchical: If True, creates hierarchical keys combining all dimensions
    """

    def __init__(self, dimensions: typing.List[AggregationDimension], hierarchical: bool = False):
        self.dimensions = dimensions
        self.hierarchical = hierarchical

    def generate_keys(self, entry: typing.Any, metric: digem.OCRMetric) -> typing.List[str]:
        """Generate aggregation keys for an evaluation entry

        Args:
            entry: EvalEntry to extract dimensions from
            metric: OCRMetric being aggregated

        Returns:
            List of aggregation key strings (e.g., ["Cs@directory:ger_frk"])
        """
        keys = []

        # Single dimension keys: metric@dimension:value
        for dim in self.dimensions:
            value = dim.extractor(entry)
            if value is not None:
                # Handle both single values and lists
                if isinstance(value, list):
                    # For lists (like directory hierarchy), create keys for each
                    for v in value:
                        keys.append(f"{metric.label}@{dim.name}:{v}")
                else:
                    keys.append(f"{metric.label}@{dim.name}:{value}")

        # Hierarchical keys: metric@dim1:val1/dim2:val2/...
        if self.hierarchical and len(self.dimensions) > 1:
            values = []
            for dim in self.dimensions:
                value = dim.extractor(entry)
                if value is not None:
                    if isinstance(value, list):
                        # For lists, use the first element
                        values.append((dim.name, value[0]))
                    else:
                        values.append((dim.name, value))

            if len(values) == len(self.dimensions):
                dim_path = "/".join(f"{name}:{val}" for name, val in values)
                keys.append(f"{metric.label}@{dim_path}")

        return keys
