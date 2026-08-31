"""format_page_util module"""

from pathlib import PurePath
from typing import Dict, List, Tuple
from xml.dom.minidom import Document, Element, parse

from ocr_util.eval.model.digital_object_model import DigitalObjectTree

import ocr_util.eval.model.common as dc


class FormatPageUtil:
    """helper methods for parsing Page XML files"""

    @staticmethod
    def extract_data(path: str) -> DigitalObjectTree:
        document: Document = parse(path)
        assert document.documentElement is not None
        doc_root: Element = document.documentElement
        ns = doc_root.namespaceURI
        assert ns is not None
        pages = doc_root.getElementsByTagNameNS(ns, "Page")
        if len(pages) != 1:
            raise dc.DigitalObjectException(f"No unique page for {path}")
        page_one = pages[0]
        page_width = int(page_one.getAttribute("imageWidth"))
        page_height = int(page_one.getAttribute("imageHeight"))
        top_piece = DigitalObjectTree(
            page_one.getAttribute("imageFilename"),
            page_one,
            document=document,
            file_format=dc.DigitalObjectTreeOCRFileFormat.PAGE,
            file_path=PurePath(path),
        )
        top_piece.level = dc.DigitalObjectLevel.PAGE
        top_piece.dimensions = [
            [0, 0],
            [page_width, 0],
            [page_width, page_height],
            [0, page_height],
        ]
        regions = doc_root.getElementsByTagNameNS(ns, "TextRegion")
        regions.extend(doc_root.getElementsByTagNameNS(ns, "TableCell"))
        # no regions are considered to be reasonable
        # don't raise exception, it's an empty page
        if len(regions) < 1:
            return top_piece

        # extract reading order if available
        reading_order_map = FormatPageUtil.__extract_reading_order(page_one, ns)

        # sort regions by reading order if available, otherwise use DOM order
        if reading_order_map:
            regions = FormatPageUtil.__sort_regions_by_reading_order(
                regions, reading_order_map
            )

        # inspect *all* regions
        region_pieces: List[DigitalObjectTree] = []
        for region in regions:
            this_piece: DigitalObjectTree = FormatPageUtil.__from_text_element(
                region, top_piece, ns
            )
            # go into details
            page_lines = region.getElementsByTagNameNS(ns, "TextLine")
            if len(page_lines) > 0:
                this_piece.children = FormatPageUtil.__read_lines(
                    page_lines, this_piece, ns
                )
            this_piece.parent = top_piece
            region_pieces.append(this_piece)
        top_piece.children = region_pieces
        _all_points = [point for reg in region_pieces for point in reg.dimensions]
        top_piece.dimensions = _all_points
        return top_piece

    @staticmethod
    def __read_lines(
        page_lines: List[Element], parent: DigitalObjectTree, ns
    ) -> List[DigitalObjectTree]:
        line_pieces = []
        for page_line in page_lines:
            line_piece = FormatPageUtil.__from_text_element(page_line, parent, ns)
            if line_piece not in parent:
                msg = f"{line_piece.id}/{line_piece.as_box()} not contained in "
                msg += f"parent box {parent.id}/{parent.as_box()}"
                print(f"[WARNING] geometry inconsistent: {msg}")
            if line_piece.max_level != dc.DigitalObjectLevel.LINE:
                word_tokens = page_line.getElementsByTagNameNS(ns, "Word")
                line_piece.parent = parent
                # inspect PAGE on word level, if set
                if len(word_tokens) > 0:
                    try:
                        word_pieces = [
                            FormatPageUtil.__from_text_element(el, line_piece, ns)
                            for el in word_tokens
                        ]
                        if not word_pieces:
                            raise RuntimeError(f"No words in line {line_piece.id}!")
                        # remove line content in favour of words content
                        line_piece.transcriptions = []
                        line_piece.children = word_pieces
                    except dc.DigitalObjectException as _pex:
                        raise RuntimeError(_pex.args) from _pex
            line_pieces.append(line_piece)
        return line_pieces

    @staticmethod
    def __from_text_element(element: Element, parent, ns) -> DigitalObjectTree:
        """transformation from PAGE XML textual nodes
        to generic pieces with specific transkription.

        If on PAGE level Word creates word pieces, and
        also inspects textual contents.

        catches several *bad* data flavours regarding
        coordinates, points and text content

        * missing Coord node
        * Coord exists, but misses attribute "points"
        * Coord exists, attribute "points" exists, but
          contains less than 3 point-pairs, thous only
          forms a line and not even a triangle
        * missing TextEquiv node
        * TextEquiv exists, but no Unicode child
        * Unicode exists, but lacking any text content

        """
        element_id = element.getAttribute("id")
        the_level, _local = FormatPageUtil.__map_piece_type(element)
        piece = DigitalObjectTree(
            element_id, element, file_format=dc.DigitalObjectTreeOCRFileFormat.PAGE
        )
        piece.level = the_level
        piece.parent = parent

        # inspect geometry
        coords = [n for n in element.childNodes if n.localName == "Coords"]
        if len(coords) < 1 or "points" not in coords[0].attributes:
            raise dc.DigitalObjectGeometryException(
                f"{_local}@ID={element_id} invalid coordinate data"
            )
        first_coord_points = coords[0].getAttribute("points").split()
        # invariant: require at least 3 points, otherwise invalid polygon area
        if len(first_coord_points) < 3:
            raise dc.DigitalObjectGeometryException(
                f"{_local}@ID={element_id} too few points {first_coord_points}"
            )
        try:
            piece.dimensions = [
                [int(_point.split(",")[0]), int(_point.split(",")[1])]
                for _point in first_coord_points
            ]
        except ValueError as _val_err:
            raise dc.DigitalObjectGeometryException(
                f"{_local}@ID={element_id} invalid {first_coord_points}"
            ) from _val_err
        # replace current text with next order children text
        txt_eqs = [
            n
            for n in element.childNodes
            if n.localName == "TextEquiv" and FormatPageUtil.__contains_value(n, ns)
        ]
        if txt_eqs:
            if len(txt_eqs) == 1:
                first_equiv: Element = txt_eqs[0]
            else:
                # pick ZERO indexed element with content
                first_equiv = sorted(txt_eqs, key=lambda x: x.getAttribute("index"))[0]
                if element.localName == "TextLine":
                    piece.max_level = dc.DigitalObjectLevel.LINE
            unicodes = first_equiv.getElementsByTagNameNS(ns, "Unicode")
            if len(unicodes) < 1:
                raise dc.DigitalObjectException(
                    f"{_local}@ID={element_id} text missing unicode"
                )
            the_unicode = unicodes[0]
            if the_unicode.firstChild and the_unicode.firstChild.nodeValue is not None:
                # replace linebreak if text only at region level
                txt_content = the_unicode.firstChild.nodeValue.replace("\n", " ")
                if txt_content:
                    piece.transcription = txt_content.strip() # no final space
                    # overthrow existing parent transcription
                    if piece.parent is not None and piece.parent.transcriptions:
                        piece.parent.transcriptions = []
        return piece

    @staticmethod
    def __contains_value(element: Element, ns) -> bool:
        unicodes = element.getElementsByTagNameNS(ns, "Unicode")
        if len(unicodes) < 1:
            return False
        the_unicode = unicodes[0]
        if (
            the_unicode.firstChild
            and the_unicode.firstChild.nodeValue is not None
            and len(the_unicode.firstChild.nodeValue.strip()) > 0
        ):
            return True
        return False

    @staticmethod
    def __map_piece_type(element) -> Tuple[dc.DigitalObjectLevel, str]:
        local_name = element.localName
        the_level = dc.DigitalObjectLevel.UNKNOWN
        if local_name == "Word":
            the_level = dc.DigitalObjectLevel.WORD
        elif local_name == "TextLine":
            the_level = dc.DigitalObjectLevel.LINE
        elif local_name == "TextRegion":
            the_level = dc.DigitalObjectLevel.REGION
        elif local_name == "TableRegion":
            the_level = dc.DigitalObjectLevel.TABLE
        elif local_name == "TableCell":
            the_level = dc.DigitalObjectLevel.TABLE_CELL
        return the_level, local_name

    @staticmethod
    def __extract_reading_order(page_element: Element, ns: str) -> Dict[str, int]:
        """Extract reading order mapping from PAGE XML

        Returns a dictionary mapping region IDs to their reading order index.
        If no reading order is found, returns an empty dictionary.
        """
        reading_order_map = {}
        reading_orders = page_element.getElementsByTagNameNS(ns, "ReadingOrder")
        if len(reading_orders) > 0:
            region_refs = reading_orders[0].getElementsByTagNameNS(
                ns, "RegionRefIndexed"
            )
            for ref in region_refs:
                region_id = ref.getAttribute("regionRef")
                index = ref.getAttribute("index")
                if region_id and index:
                    try:
                        reading_order_map[region_id] = int(index)
                    except ValueError:
                        # Skip invalid index values
                        pass
        return reading_order_map

    @staticmethod
    def __sort_regions_by_reading_order(
        regions: List[Element], reading_order_map: Dict[str, int]
    ) -> List[Element]:
        """Sort regions according to reading order

        Regions with reading order indices are sorted first by their index.
        Regions without reading order indices maintain their original order after.
        """
        regions_with_order = []
        regions_without_order = []

        for region in regions:
            region_id = region.getAttribute("id")
            if region_id in reading_order_map:
                regions_with_order.append((reading_order_map[region_id], region))
            else:
                regions_without_order.append(region)

        # Sort regions with reading order by their index
        regions_with_order.sort(key=lambda x: x[0])

        # Combine: ordered regions first, then unordered ones
        sorted_regions = [region for _, region in regions_with_order]
        sorted_regions.extend(regions_without_order)
        return sorted_regions
