"""digital_object_util module"""

from pathlib import Path
from typing import List, Optional
from xml.dom.minidom import Document, Element, Node, parse

import xml.parsers.expat

import ocr_util.eval.model.common as dmc

from ocr_util.eval.model.common import DigitalObjectDimensions, PAGE_2013, TEXT_ENCODING
from ocr_util.eval.model.digital_object_model import DigitalObjectTree
from ocr_util.eval.model.format_alto_v3_util import FormatAltoV3Util
from ocr_util.eval.model.format_page_util import FormatPageUtil


class DigitalObjectUtil:
    """helper methods for DigitalObject(s)"""

    @staticmethod
    def to_digital_objects(path_in: str) -> DigitalObjectTree:
        """Convert file into DigitalObject structure"""
        return DigitalObjectUtil.read_data(path_in)

    @staticmethod
    def from_digital_objects(root_digo: DigitalObjectTree, path_out: Path = None) -> Path:
        """Convert DigitalObject structure into a xml file"""
        if path_out is None:
            orig_file_path: Path = root_digo.file_path
            parent_dir: Path = orig_file_path.parent
            basename: str = orig_file_path.name
            suffix: str = orig_file_path.suffix
            new_name: str = basename.replace(f"{suffix}", f".gt{suffix}")
            path_out: Path = parent_dir.joinpath(new_name)
        with open(path_out, "w", encoding=TEXT_ENCODING) as file:
            file.write(DigitalObjectUtil.to_xml_str(root_digo.document))
        return path_out

    @staticmethod
    def to_xml_str(node: Node) -> str:
        """Convert xml-object structure into a xml string"""
        return node.toprettyxml(encoding=TEXT_ENCODING).decode(TEXT_ENCODING)

    @staticmethod
    def read_data(path_in: str) -> DigitalObjectTree:
        """read xml and create DigitalObject structure based on file format"""
        try:
            document: Document = parse(path_in)
            assert document.documentElement is not None
            doc_root: Element = document.documentElement
        except xml.parsers.expat.ExpatError as _exc:
            the_exc_info = f"XML parsing error in '{path_in}': {_exc}"
            raise dmc.DigitalObjectException(the_exc_info) from _exc
        if doc_root is None:
            raise dmc.DigitalObjectException("invalid document root")
        name_space = doc_root.getAttribute("xmlns")
        piece: Optional[DigitalObjectTree]
        if doc_root.localName.lower() == "alto":
            piece = FormatAltoV3Util.extract_data(path_in)
        elif name_space == PAGE_2013:
            piece = FormatPageUtil.extract_data(path_in)
        elif doc_root.localName.lower() == "pcgts":
            piece = FormatPageUtil.extract_data(path_in)
        else:
            raise dmc.DigitalObjectException(f'Unknown Data-Format "{doc_root.localName}" in "{path_in}"')
        return piece

    @staticmethod
    def calulate_dimensions_by_children(
        digo: DigitalObjectTree,
    ) -> DigitalObjectDimensions:
        """recursively calculate bounds by involving bounds of all children"""
        if len(digo.children) > 0:
            dims: DigitalObjectDimensions = []
            for child_piece in digo.children:
                dims.extend(child_piece.dimensions)
            dims = DigitalObjectUtil.__calculate_dimensions_rect_bounds(dims)
            return dims
        return digo.dimensions

    @staticmethod
    def flatten(digo: DigitalObjectTree) -> List[DigitalObjectTree]:
        """flattens a DigitalObject structure"""

        def flatten_recursive(pc: DigitalObjectTree, digos: List[DigitalObjectTree] = None) -> List[DigitalObjectTree]:
            if digos is None:
                digos = []
            digos.append(pc)
            for child in pc.children:
                flatten_recursive(child, digos)
            return digos

        return flatten_recursive(digo)

    @staticmethod
    def __calculate_dimensions_rect_bounds(
        dimensions: DigitalObjectDimensions,
    ) -> DigitalObjectDimensions:
        min_x: Optional[float] = None
        min_y: Optional[float] = None
        max_x: Optional[float] = None
        max_y: Optional[float] = None
        for point in dimensions:
            point_x: float = point[0]
            point_y: float = point[1]
            if min_x is None or point_x < min_x:
                min_x = point_x
            if min_y is None or point_y < min_y:
                min_y = point_y
            if max_x is None or point_x > max_x:
                max_x = point_x
            if max_y is None or point_y > max_y:
                max_y = point_y
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
