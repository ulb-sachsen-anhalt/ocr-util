"""Test specification for representation
of digital assets in OCR ALTO format
"""

import os
from pathlib import (
	PurePath,
)

import pytest 

import shapely.geometry


from ocr_util.eval.model.digital_object_model import (
	DigitalObjectTree,
)
from ocr_util.eval.model.common import (
	DigitalObjectLevel,
)
from ocr_util.eval.model.main import (
	to_digital_object,
)
from tests.conftest import (
	TEST_RES_DIR,
)



def test_digital_object_hierarchy_bottom_up():
    """Ensure behavior of digital_objects
    hierarchy from bottom up"""

    assert DigitalObjectLevel.WORD < DigitalObjectLevel.LINE
    assert DigitalObjectLevel.LINE < DigitalObjectLevel.REGION
    assert DigitalObjectLevel.WORD < DigitalObjectLevel.REGION
    assert DigitalObjectLevel.REGION < DigitalObjectLevel.PAGE
    assert DigitalObjectLevel.LINE < DigitalObjectLevel.PAGE
    assert DigitalObjectLevel.WORD < DigitalObjectLevel.PAGE


def test_digital_object_hierarchy_top_down():
    """Fix behaviour of DigitalObject Hierarchy"""

    assert DigitalObjectLevel.REGION > DigitalObjectLevel.LINE
    assert DigitalObjectLevel.LINE > DigitalObjectLevel.WORD
    assert DigitalObjectLevel.REGION > DigitalObjectLevel.WORD
    assert DigitalObjectLevel.WORD > DigitalObjectLevel.GLYPH


def test_contains_digital_object_relation():
    """Ensure DigitalObjectTypes and contains relations"""

    ocr_path = os.path.join(TEST_RES_DIR, 'groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')

    # act
    page_piece = to_digital_object(ocr_path)
    region1 = page_piece.children[0]
    line1 = region1.children[0]
    word1 = line1.children[0]

    # assert
    assert region1 in page_piece
    assert region1.level == DigitalObjectLevel.REGION
    assert line1 in region1
    assert line1.level == DigitalObjectLevel.LINE
    assert word1 in line1 and word1 in region1
    assert word1.level == DigitalObjectLevel.WORD


def test_digital_object_file_path():
    """Ensure type of piece property"""

    ocr_path: str = f'{TEST_RES_DIR}/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml'
    page_piece: DigitalObjectTree = to_digital_object(ocr_path)
    assert isinstance(page_piece.file_path, PurePath)


def test_digital_objects_zd101_page_piece_dimension(zd101):
    """Check ALTO page piece spans
    """

    # explore dimensions
    assert len(zd101.dimensions) == 4

    assert [0, 0] in zd101.dimensions
    assert [6633, 0] in zd101.dimensions
    assert [6633, 9944] in zd101.dimensions
    assert [0, 9944] in zd101.dimensions


def test_digital_objects_zd101_page_bounding_box_dimension(zd101):
    """check if bounding box reflects non-modfied page dimensions

    * top_left: 0,0
    * bottom_right: 6633,9944
    """

    a_polygon = shapely.geometry.Polygon(zd101.dimensions)

    # all regions contained in this box
    assert a_polygon.bounds == (0.0, 0.0, 6633.0, 9944.0)


def test_digital_objects_zd101_region01_dimensions(zd101):
    """
    * a region piece has also dimension and is contained in page
    * a line piece has dimensions and is contained in region
    * a word piece has dimensions and is contained in line + region
    """

    assert len(zd101.children) == 10
    region1 = zd101.children[0]
    line1 = region1.children[0]
    assert region1.dimensions == [
        [802, 2100], [1901, 2100], [1901, 2223], [802, 2223]]

    # this region has only one line
    assert len(region1.children) == 1
    # which is completely same as region
    assert line1.dimensions == [[802, 2100],
                                [1901, 2100], [1901, 2223], [802, 2223]]
    assert region1.dimensions == line1.dimensions
    # this line has two words
    assert len(line1.children) == 2
    word1 = line1.children[0]
    # coords for word01 differ sligthly from line
    assert word1.dimensions == [[802, 2101],
                                [1246, 2101], [1246, 2219], [802, 2219]]
    assert word1 in line1 and line1 in region1

