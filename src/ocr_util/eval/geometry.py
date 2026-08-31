"""Information about geometry"""

import os
import re
import typing

import xml.dom.minidom
import xml.etree.ElementTree as ET

_XML_NS = {
    "alto": "http://www.loc.gov/standards/alto/ns-v3#",
    "pg2013": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
}
_NOT_SET = "n.a."


def get_page_dimensions(file_path) -> typing.Optional[typing.Tuple[int, int]]:
    """Return the declared image width and height for an ALTO or PAGE file."""
    try:
        root_element = ET.parse(file_path).getroot()
    except (OSError, ET.ParseError):
        return None
    page_element = root_element.find(".//{*}Page")
    if page_element is None:
        return None

    width = page_element.attrib.get("WIDTH", page_element.attrib.get("imageWidth"))
    height = page_element.attrib.get(
        "HEIGHT", page_element.attrib.get("imageHeight")
    )
    if width is None or height is None:
        return None
    return int(width), int(height)


def get_bounding_box(file_path) -> typing.Optional[typing.Tuple]:
    """Get Bounding Box Data from given resource, if any exists"""

    if not isinstance(file_path, str):
        file_path = str(file_path)

    # 1: inspect filename
    file_name = os.path.basename(file_path)
    result = re.match(r".*_(\d{2,})x(\d{2,})_(\d{2,})x(\d{2,})", file_name)
    if result:
        groups = result.groups()
        x0 = int(groups[0])
        x1 = int(groups[2])
        y1 = int(groups[3])
        y0 = int(groups[1])
        return ((x0, y0), (x1, y1))

    with open(file_path, encoding="utf-8") as _handle:
        # rather brute force approach
        # to recognize OCR formats inside
        start_token = _handle.read(128)
        _frame_points = None

        # switch by estimated ocr format
        if "alto" in start_token:
            # legacy: read from custom ALTO meta data
            root_element = ET.parse(file_path).getroot()
            element = root_element.find(
                './/alto:Tags/alto:OtherTag[@ID="ulb_groundtruth_points"]', _XML_NS
            )
            if element is not None:
                points = element.attrib["VALUE"].split(" ")
                _p1 = points[0].split(",")
                p1 = (int(_p1[0]), int(_p1[1]))
                _p2 = points[2].split(",")
                p2 = (int(_p2[0]), int(_p2[1]))
                return (p1, p2)

            # read from given alto coordinates
            raw_elements = root_element.findall(".//alto:String", _XML_NS)
            non_empty = [
                s
                for s in raw_elements
                if s.attrib["CONTENT"].strip()
                and re.match(r"[^\d]", s.attrib["CONTENT"])
            ]
            return _calculate_bounding_box(non_empty, _map_alto)

        if "PcGts" in start_token:
            # read from given page coordinates
            doc_root = xml.dom.minidom.parse(file_path).documentElement
            assert doc_root is not None
            name_space = doc_root.namespaceURI
            root_element = ET.parse(file_path).getroot()
            # step one: read PAGE border coords
            _xpr_page_borders = (
                f"{{{name_space}}}Page/{{{name_space}}}Border/{{{name_space}}}Coords"
            )
            _page_coords = root_element.findall(_xpr_page_borders)
            if len(_page_coords) > 0:
                _frame_points = _calculate_bounding_box(_page_coords, _map_page2013)
            # step two: if possible, go for sub-part geometry
            _xpr_line_coords = f".//{{{name_space}}}TextLine/{{{name_space}}}Coords"
            _line_coords = root_element.findall(_xpr_line_coords)
            if len(_line_coords) > 0:
                _frame_points = _calculate_bounding_box(_line_coords, _map_page2013)
            if _frame_points:
                return _frame_points
            else:
                raise RuntimeError(f"{file_path} missing page/line coords!")
    return None


def _map_alto(e: ET.Element) -> typing.Tuple[str, int, int, int, int]:
    i = e.attrib["ID"]
    x0 = int(e.attrib["HPOS"])
    y0 = int(e.attrib["VPOS"])
    x1 = x0 + int(e.attrib["WIDTH"])
    y1 = y0 + int(e.attrib["HEIGHT"])
    return (i, x0, y0, x1, y1)


def _map_page2013(elem: ET.Element) -> typing.Tuple[str, int, int, int, int]:
    points = elem.attrib["points"].strip().split(" ")
    _xs = [int(p.split(",")[0]) for p in points]
    _ys = [int(p.split(",")[1]) for p in points]
    return (_NOT_SET, min(_xs), min(_ys), max(_xs), max(_ys))


def _calculate_bounding_box(
    elements: typing.List[ET.Element], map_func
) -> typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]:
    """Review element's points to get points for
    minimum (top-left) and maximum (bottom-right)"""

    all_points = [map_func(e) for e in elements]
    all_x1 = [p[1] for p in all_points]
    all_y1 = [p[2] for p in all_points]
    all_x2 = [p[3] for p in all_points]
    all_y2 = [p[4] for p in all_points]
    return ((min(all_x1), min(all_y1)), (max(all_x2), max(all_y2)))
