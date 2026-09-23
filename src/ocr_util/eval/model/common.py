"""This module contains common datatypes and constants"""

from enum import Enum, IntEnum
from typing import Dict, Final, List
from xml.dom.minidom import Element

PAGE_2013: Final[str] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
XML_NS: Final[Dict[str, str]] = {
    "alto": "http://www.loc.gov/standards/alto/ns-v3#",
    "pg2013": PAGE_2013,
}

# mark information as 'not available'
# which *might* be set later on
UNSET: Final[str] = "n.a."

TEXT_ENCODING: Final[str] = "utf-8"

DigitalObjectDimensions = List[List[float]]


class DigitalObjectLevel(IntEnum):
    """hierarchical relations"""

    UNKNOWN = 0
    GLYPH = 1
    WORD = 2
    LINE = 3
    TABLE_CELL = 4
    REGION = 5
    TABLE = 6
    PAGE = 7
    SECTION = 8

    def __lt__(self, other_lvl):
        if not isinstance(other_lvl, DigitalObjectLevel):
            return False
        return self.value < other_lvl.value

    def __gt__(self, other_lvl):
        if not isinstance(other_lvl, DigitalObjectLevel):
            return False
        return self.value > other_lvl.value

    def __eq__(self, other_lvl):
        if not isinstance(other_lvl, DigitalObjectLevel):
            return False
        return self.value == other_lvl.value


class DigitalObjectException(Exception):
    """Mark custom Exception"""


class DigitalObjectGeometryException(DigitalObjectException):
    """Mark custom Exception for geometry issues, for example
    when coordinates are not available or invalid or children
    are not contained in the parent area.
    """


class DigitalObjectContent(Enum):
    """structural content type (Layout and semantics)"""

    UNKNOWN = 0
    PARAGRAPH = 1
    HEADING = 2
    COLUMN = 3
    TABLE = 4
    IMAGE = 5
    ARTICLE = 6
    ANNOUNCEMENT = 7
    ADVERTISEMENT = 8


class DigitalObjectTranscription:
    """textual representation of ISO language and OCR confidence"""

    def __init__(self):
        self.text = ""
        self.language = UNSET
        self.confidence = 0.0


class DigitalObjectData:
    """binary representation and mime type"""

    def __init__(self):
        self.mime_type = UNSET
        self.data = None


class DigitalObjectTreeOCRFileFormat(Enum):
    """known file formats"""

    UNKNOWN = "UNKNOWN"
    ALTO_V3 = "ALTO_V3"
    PAGE = "PAGE"


class DigitalObjectChanges:
    """report container for structual manipulations"""

    removed_elements: List[Element] = []
    resized_elements: List[Element] = []
