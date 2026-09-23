"""digital_object module"""

from __future__ import annotations

import typing

from copy import copy
from pathlib import PurePath
from xml.dom.minidom import Document, Element

import shapely.geometry

from ocr_util.eval.model.common import (
    UNSET,
    DigitalObjectChanges,
    DigitalObjectContent,
    DigitalObjectData,
    DigitalObjectDimensions,
    DigitalObjectLevel,
    DigitalObjectTreeOCRFileFormat,
    DigitalObjectTranscription,
)
from ocr_util.eval.model.minidom_util import MinidomUtil


class DigitalObjectTree:
    """DigitalObject representing a tree-like composition
    cf. OCR hierarchical formats like ALTO or PAGE XML
    """

    def __init__(
        self,
        identifier: str = UNSET,
        xml_element: typing.Optional[Element] = None,
        document: typing.Optional[Document] = None,
        file_format: DigitalObjectTreeOCRFileFormat = DigitalObjectTreeOCRFileFormat.UNKNOWN,
        file_path: typing.Optional[PurePath] = None,
    ):
        self.id: str = identifier
        self.level: DigitalObjectLevel = DigitalObjectLevel.PAGE
        self.subject: DigitalObjectContent = DigitalObjectContent.UNKNOWN
        self.data: typing.Optional[DigitalObjectData] = None
        self.parent: typing.Optional[DigitalObjectTree] = None
        self.custom: typing.Dict = {}
        self.max_level: DigitalObjectLevel = DigitalObjectLevel.WORD
        self._transcriptions: typing.List = []
        self.__file_path: typing.Optional[PurePath] = file_path
        self.__dimensions: DigitalObjectDimensions = []
        self.__children: typing.List[DigitalObjectTree] = []
        if xml_element is not None:
            self.__xml_element: Element = xml_element
        if document is not None:
            self.__document: Document = document
        self.__file_format: DigitalObjectTreeOCRFileFormat = file_format

    @property
    def file_format(self) -> DigitalObjectTreeOCRFileFormat:
        return self.parent.file_format if self.parent is not None else self.__file_format

    @property
    def document(self) -> Document:
        return self.parent.document if self.parent is not None else self.__document

    @property
    def file_path(self) -> PurePath:
        return self.parent.file_path if self.parent is not None else self.__file_path

    @property
    def dimensions(self) -> DigitalObjectDimensions:
        return self.__dimensions

    @dimensions.setter
    def dimensions(self, dims: DigitalObjectDimensions) -> None:
        self.__dimensions = dims
        if self.__file_format != DigitalObjectTreeOCRFileFormat.UNKNOWN and self.__set_dimensions_in_xml(dims):
            DigitalObjectChanges.resized_elements.append(self.xml_element)

    def as_box(self):
        """Simple bounding box for complex shapes"""
        if not self.dimensions:
            return None
        min_x = min(point[0] for point in self.dimensions)
        min_y = min(point[1] for point in self.dimensions)
        max_x = max(point[0] for point in self.dimensions)
        max_y = max(point[1] for point in self.dimensions)
        return [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]

    @property
    def transcriptions(self):
        return self._transcriptions

    @transcriptions.setter
    def transcriptions(self, transcriptions):
        self._transcriptions = transcriptions

    @property
    def xml_element(self) -> Element:
        return self.__xml_element

    @property
    def children(self) -> typing.List[DigitalObjectTree]:
        return copy(self.__children)

    @children.setter
    def children(self, children: typing.List[DigitalObjectTree]) -> None:
        self.__children = children

    def remove_children(self, *children: DigitalObjectTree) -> None:
        """remove children and its native xml element(s)"""
        for piece in children:
            self.__children.remove(piece)
            element: Element = piece.xml_element
            removable_tags: typing.List[str] = []
            if piece.file_format == DigitalObjectTreeOCRFileFormat.ALTO_V3:
                removable_tags.append("SP")
            removed_elements: typing.List[Element] = MinidomUtil.remove_element_and_clear_parent(
                element, removable_tags
            )
            DigitalObjectChanges.removed_elements.extend(removed_elements)

    @property
    def transcription(self) -> str:
        """Get textual content as sequential textual string,
        with the order corresponding to it's _previous and
        next properties.

        Text will come without sanitized linebreaks, but
        includes a whitespace between single lines and words.
        """
        if self._transcriptions:
            return self._transcriptions[0].text

        if not self._transcriptions and self.__is_superstruct():
            return " ".join([_p.transcription for _p in self.children])
        raise RuntimeError(f"ID={self.id}: Can't get text_content for {self.id}!")

    @transcription.setter
    def transcription(self, transscr: str) -> None:
        """Set textual transcription representing this piece"""
        _transcription = DigitalObjectTranscription()
        if transscr is not None and len(transscr.strip()) > 0:
            _transcription.text = transscr
        self._transcriptions.append(_transcription)

    def is_in_polygon(self, poly: shapely.geometry.Polygon) -> bool:
        """check whether current Element is geometrically included in convex hull"""
        digo_poly: shapely.geometry.Polygon = shapely.geometry.Polygon(self.dimensions)
        assert poly is not None
        convex_hull: shapely.geometry.base.BaseGeometry = poly.convex_hull
        return convex_hull.contains(digo_poly.centroid)

    def __repr__(self) -> str:
        return f"{self.id}:{self.transcription}"

    def __contains__(self, other_piece) -> bool:
        """Test for membership of other_piece
        Precondition: other_piece is logical
            child/anchestor of current piece.level
        Calculate hull for self and centroid for
        other_pieces (to catch corner cases, too)
        """

        if not self.dimensions:
            raise RuntimeError(f"ID={self.id}: self has invalid dimensions!")
        if not other_piece.dimensions:
            raise RuntimeError(f"{other_piece.id}: other has invalid dimensions!")
        # check order invariant
        if self.level < other_piece.level or self.level == other_piece.level:
            raise RuntimeError(f"other {other_piece.id} is higher/equal level than {self.id}!")
        # Go for centriod for real life
        # cases where word bounds
        # scratch over region borders
        self_hull = shapely.geometry.Polygon(self.dimensions).convex_hull
        other_shape = shapely.geometry.Polygon(other_piece.dimensions)
        return self_hull.contains(other_shape.centroid)

    def __is_superstruct(self):
        """Logical parent/container elements"""
        return self.level in [
            DigitalObjectLevel.PAGE,
            DigitalObjectLevel.REGION,
            DigitalObjectLevel.LINE,
            DigitalObjectLevel.TABLE,
            DigitalObjectLevel.TABLE_CELL,
        ]

    @staticmethod
    def dimensions_to_str(dimensions: DigitalObjectDimensions) -> str:
        """create string of point pairs"""
        strs: typing.List[str] = [f"{round(p[0])},{round(p[1])}" for p in dimensions]
        return " ".join(strs)

    def __set_dimensions_in_xml(self, dimensions: DigitalObjectDimensions) -> bool:
        """set dimension in xml based on file format"""
        xml_element: Element = self.xml_element
        top_left: typing.List[float] = dimensions[0]
        bottom_right: typing.List[float] = dimensions[2]
        x1: float = top_left[0]
        y1: float = top_left[1]
        x2: float = bottom_right[0]
        y2: float = bottom_right[1]
        width: float = x2 - x1
        height: float = y2 - y1
        has_changed: bool = False
        if self.file_format == DigitalObjectTreeOCRFileFormat.ALTO_V3:
            if MinidomUtil.set_attribute(xml_element, "HPOS", int(x1)):
                has_changed = True
            if MinidomUtil.set_attribute(xml_element, "VPOS", int(y1)):
                has_changed = True
            if MinidomUtil.set_attribute(xml_element, "WIDTH", int(width)):
                has_changed = True
            if MinidomUtil.set_attribute(xml_element, "HEIGHT", int(height)):
                has_changed = True
        elif self.file_format == DigitalObjectTreeOCRFileFormat.PAGE:
            points: str = self.dimensions_to_str(dimensions)
            if MinidomUtil.set_attribute(xml_element, "points", points):
                has_changed = True
        else:
            raise NotImplementedError(f'__set_dimensions_in_xml() not implemented for "{self.file_format}"')
        return has_changed


# class DigitalPAGETree(DigitalObjectTree):
#     """DigitalObject representing a PAGE XML structure"""

#     def __init__(
#             self,
#             identifier: str = UNSET,
#             xml_element: Element = None,
#             document: Document = None,
#             file_path: PurePath = None
#     ):
#         super().__init__(
#             identifier=identifier,
#             xml_element=xml_element,
#             document=document,
#             file_format=DigitalObjectTreeOCRFileFormat.PAGE,
#             file_path=file_path
#         )

#     def __set_dimensions_in_xml(self, dimensions: DigitalObjectDimensions) -> bool:
#         super_changed = super().__set_dimensions_in_xml(dimensions)
#         xml_element: Element = self.xml_element
#         points: str = self.dimensions_to_str(dimensions)
#         if MinidomUtil.set_attribute(xml_element, 'points', points):
#             has_changed = True
#             return has_changed or super_changed
#         return super_changed
