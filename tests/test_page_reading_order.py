"""Test specification for reading order in PAGE XML format"""

from ocr_util.eval.model.digital_object_model import DigitalObjectTree
from ocr_util.eval.model.main import to_digital_object
from .conftest import TEST_RES_DIR


def test_reading_order_respected():
    """Test that reading order from RegionRefIndexed is respected
    This test uses a file where the reading order differs from
    the natural DOM order of regions. The reading order specifies:
    index 0: region_003, index 1: region_001, index 2: region_002
    while the DOM order is: region_001, region_002, region_003
    """

    # arrange
    xml_path = f"{TEST_RES_DIR}/groundtruth/page/reading_order_respected.xml"

    # act
    page_piece: DigitalObjectTree = to_digital_object(xml_path)

    # assert - regions should be in reading order, not DOM order
    assert len(page_piece.children) == 3
    assert page_piece.children[0].id == "region_003"
    assert page_piece.children[0].transcription == "First region text"
    assert page_piece.children[1].id == "region_001"
    assert page_piece.children[1].transcription == "Second region text"
    assert page_piece.children[2].id == "region_002"
    assert page_piece.children[2].transcription == "Third region text"


def test_reading_order_with_missing_regions():
    """Test that regions without reading order indices are placed after ordered ones"""

    # arrange
    xml_path = f"{TEST_RES_DIR}/groundtruth/page/reading_order_with_missing_regions.xml"

    # act
    page_piece: DigitalObjectTree = to_digital_object(xml_path)

    # assert - region_002 (with reading order) should come first
    assert len(page_piece.children) == 2
    assert page_piece.children[0].id == "region_002"
    assert page_piece.children[0].transcription == "Has reading order"
    assert page_piece.children[1].id == "region_001"
    assert page_piece.children[1].transcription == "No reading order"


def test_no_reading_order_element():
    """Test that regions work correctly when no ReadingOrder element exists"""

    # arrange
    xml_path = f"{TEST_RES_DIR}/groundtruth/page/no_reading_order_element.xml"

    # act
    page_piece: DigitalObjectTree = to_digital_object(xml_path)

    # assert - without reading order, should maintain DOM order
    assert len(page_piece.children) == 2
    assert page_piece.children[0].id == "region_001"
    assert page_piece.children[0].transcription == "First in DOM"
    assert page_piece.children[1].id == "region_002"
    assert page_piece.children[1].transcription == "Second in DOM"
