"""format_alto_v3_util module"""

from pathlib import PurePath
from xml.dom.minidom import Document, Element, parse

from ocr_util.eval.model.digital_object_model import DigitalObjectTree
from ocr_util.eval.model.common import (
    DigitalObjectLevel,
    DigitalObjectTreeOCRFileFormat,
    UNSET,
)


class FormatAltoV3Util:
    """helper methods for parsing Alto V3 XML files"""

    @staticmethod
    def extract_data(path: str) -> DigitalObjectTree:
        document: Document = parse(path)
        doc_root: Element = document.documentElement
        pages = doc_root.getElementsByTagName("Page")
        if len(pages) < 1:
            raise RuntimeError(f"Empty ALTO {doc_root} - no pages!")
        if len(pages) > 1:
            print(
                f"[WARN ] ALTO '{path}' contains {len(pages)} Page elements; using first page only."
            )
        page_one: Element = pages[0]
        _page_width = int(page_one.getAttribute("WIDTH"))
        _page_height = int(page_one.getAttribute("HEIGHT"))
        _dimensions = [
            [0, 0],
            [_page_width, 0],
            [_page_width, _page_height],
            [0, _page_height],
        ]
        top_digo: DigitalObjectTree = DigitalObjectTree(
            page_one.getAttribute("ID"),
            page_one,
            document=document,
            file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3,
            file_path=PurePath(path),
        )
        top_digo.dimensions = _dimensions
        top_digo.level = DigitalObjectLevel.PAGE
        top_digo.subject = FormatAltoV3Util.__get_piece_subject(doc_root)
        # composed level
        _block_digos = []
        comp_blocks = doc_root.getElementsByTagName("ComposedBlock")
        if len(comp_blocks) > 0:
            for _comp_block in comp_blocks:
                comp_digo: DigitalObjectTree = DigitalObjectTree(
                    _comp_block.getAttribute("ID"),
                    _comp_block,
                    file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3,
                )
                comp_digo.level = DigitalObjectLevel.REGION
                comp_digo.parent = top_digo
                comp_digo.dimensions = FormatAltoV3Util.__extract_dimensions(
                    _comp_block
                )
                text_blocks = _comp_block.getElementsByTagName("TextBlock")
                if len(text_blocks) < 1:
                    raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
                comp_digo.children = FormatAltoV3Util.__read_blocks(
                    text_blocks, comp_digo
                )
                _block_digos.append(comp_digo)
        else:
            text_blocks = doc_root.getElementsByTagName("TextBlock")
            if len(text_blocks) < 1:
                raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
            _block_digos = FormatAltoV3Util.__read_blocks(text_blocks, top_digo)
        top_digo.children = _block_digos
        return top_digo

    @staticmethod
    def __read_blocks(block_elements, parent):
        _blocks = []
        for _block_el in block_elements:
            _block = DigitalObjectTree(
                _block_el.getAttribute("ID"),
                _block_el,
                file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3,
            )
            _block.level = DigitalObjectLevel.REGION
            _lines = _block_el.getElementsByTagName("TextLine")
            if len(_lines) == 0:
                raise RuntimeError(f"TextBlock@ID={_block.id} contains no lines!")
            _block.parent = parent
            _block.children = FormatAltoV3Util.__read_lines(_lines, _block)
            _block.dimensions = FormatAltoV3Util.__extract_dimensions(_block_el)
            _blocks.append(_block)
        return _blocks

    @staticmethod
    def __get_piece_subject(doc_root):
        gt_type_el = doc_root.getElementsByTagName("OtherTag")
        _subject = UNSET
        if gt_type_el and len(gt_type_el) > 0:
            # deprecated
            label = gt_type_el[0].getAttribute("LABEL")
            if label:
                _subject = label
            # new alto way
            else:
                gt_els = [
                    e
                    for e in gt_type_el
                    if e.getAttribute("ID") == "ulb_groundtruth_type"
                ]
                if len(gt_els) == 1:
                    value = gt_els[0].getAttribute("VALUE")
                    if value:
                        _subject = value
        return _subject

    @staticmethod
    def __read_lines(the_lines, parent):
        _lines = []
        for _text_line in the_lines:
            _id = _text_line.getAttribute("ID")
            line = DigitalObjectTree(
                _id, _text_line, file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3
            )
            line.level = DigitalObjectLevel.LINE
            text_strings = _text_line.getElementsByTagName("String")
            if len(text_strings) < 1:
                raise RuntimeError(f"No words in line {_id}!")
            line.children = FormatAltoV3Util.__read_words(text_strings, line)
            line.parent = parent
            line.dimensions = FormatAltoV3Util.__extract_dimensions(_text_line)
            _lines.append(line)
        return _lines

    @staticmethod
    def __read_words(text_strings, parent):
        """Read String elements from TextLine, with SP-controlled or fallback spacing.

        In ALTO format:
        - If SP elements are present: String elements are concatenated WITHOUT space
          unless there is an SP (spatium) element between them.
        - If NO SP elements are present: String elements are separated with spaces
          (fallback for ALTO files that don't use SP elements).
        """
        word_tokens = []
        line_element = parent.xml_element

        # Get all child nodes (includes String and SP elements)
        child_nodes = line_element.childNodes

        # Check if any SP elements are present in this line
        has_sp_elements = any(
            node.nodeType == node.ELEMENT_NODE and node.tagName == "SP"
            for node in child_nodes
        )

        if not has_sp_elements:
            # Fallback: No SP elements present, treat each String as a separate word
            for _text_string in text_strings:
                _id = _text_string.getAttribute("ID")
                word = DigitalObjectTree(
                    _id,
                    _text_string,
                    file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3,
                )
                word.level = DigitalObjectLevel.WORD
                _content = _text_string.getAttribute("CONTENT")
                if not _content.strip():
                    continue
                word.transcription = _content
                word.dimensions = FormatAltoV3Util.__extract_dimensions(_text_string)
                word.parent = parent
                word_tokens.append(word)
        else:
            # SP elements present: use them to control word boundaries
            accumulated_content = ""
            accumulated_element = None

            for node in child_nodes:
                if node.nodeType != node.ELEMENT_NODE:
                    continue

                if node.tagName == "String":
                    _content = node.getAttribute("CONTENT")
                    if not _content.strip():
                        continue

                    # Accumulate content
                    if not accumulated_content:
                        accumulated_element = node
                    accumulated_content += _content

                elif node.tagName == "SP":
                    # SP element signals end of current word group
                    if accumulated_content:
                        word = DigitalObjectTree(
                            accumulated_element.getAttribute("ID"),
                            accumulated_element,
                            file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3,
                        )
                        word.level = DigitalObjectLevel.WORD
                        word.transcription = accumulated_content
                        word.dimensions = FormatAltoV3Util.__extract_dimensions(
                            accumulated_element
                        )
                        word.parent = parent
                        word_tokens.append(word)

                        # Reset accumulation
                        accumulated_content = ""
                        accumulated_element = None

            # Add final accumulated word if any
            if accumulated_content:
                word = DigitalObjectTree(
                    accumulated_element.getAttribute("ID"),
                    accumulated_element,
                    file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3,
                )
                word.level = DigitalObjectLevel.WORD
                word.transcription = accumulated_content
                word.dimensions = FormatAltoV3Util.__extract_dimensions(
                    accumulated_element
                )
                word.parent = parent
                word_tokens.append(word)

        return word_tokens

    # does not repspect ALTO shapes
    @staticmethod
    def __extract_dimensions(el):
        _left = int(el.getAttribute("HPOS"))
        _top = int(el.getAttribute("VPOS"))
        _height = int(el.getAttribute("HEIGHT"))
        _width = int(el.getAttribute("WIDTH"))
        return [
            [_left, _top],
            [_left + _width, _top],
            [_left + _width, _top + _height],
            [_left, _top + _height],
        ]
