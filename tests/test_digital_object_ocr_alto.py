"""Test specification for representation
of digital assets in OCR ALTO format
"""

import os
from pathlib import (
	PurePath,
)

import pytest
from shapely import (
	Polygon,
)

from ocr_util.eval.model.digital_object_model import (
	DigitalObjectTree,
)
from ocr_util.eval.model.common import (
	DigitalObjectLevel,
)
from ocr_util.eval.model.main import (
	to_digital_object,
)
from tests.conftest import TEST_RES_DIR


# @pytest.fixture(name="zd101")
# def _fixture_zd101():
#     ocr_path = f'{TEST_RES_DIR}/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml'
#     page_piece = to_digital_object(ocr_path)
#     yield page_piece


def test_to_children_altov3():
    """Ensure old ZD1 groundtruth in ALTO V3 can still be computed"""

    ocr_path = os.path.join(TEST_RES_DIR, 'groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')

    # act
    page_piece = to_digital_object(ocr_path)

    # assert
    # 10 regions
    assert len(page_piece.children) == 10
    # check region IDa
    assert page_piece.children[0].id == 'block_27'
    assert page_piece.children[1].id == 'block_28'

    # check region 2 has 2 lines
    assert len(page_piece.children[1].children) == 2
    # line 1 of region 1 has 2 words
    assert len(page_piece.children[0].children[0].children) == 2
    # line 2 of region 2 has 5 words
    assert len(page_piece.children[1].children[1].children) == 5

    # first line textual content
    line1_text = page_piece.children[0].children[0].transcription
    assert line1_text == 'Neueſte Ereigniſſe.'
    # first region has just one single line,
    # therefore both must match
    region1_text = page_piece.children[0].transcription
    assert region1_text == line1_text
    assert page_piece.transcription.startswith(region1_text)


# def test_digital_objects_zd101_page_piece_dimension(zd101):
#     """Check ALTO page piece spans
#     """

#     # explore dimensions
#     assert len(zd101.dimensions) == 4

#     assert [0, 0] in zd101.dimensions
#     assert [6633, 0] in zd101.dimensions
#     assert [6633, 9944] in zd101.dimensions
#     assert [0, 9944] in zd101.dimensions


# def test_digital_objects_zd101_page_bounding_box_dimension(zd101):
#     """check if bounding box reflects non-modfied page dimensions

#     * top_left: 0,0
#     * bottom_right: 6633,9944
#     """

#     _polygon = Polygon(zd101.dimensions)

#     # all regions contained in this box
#     assert _polygon.bounds == (0.0, 0.0, 6633.0, 9944.0)


# def test_digital_objects_zd101_region01_dimensions(zd101):
#     """
#     * a region piece has also dimension and is contained in page
#     * a line piece has dimensions and is contained in region
#     * a word piece has dimensions and is contained in line + region
#     """

#     assert len(zd101.children) == 10
#     region1 = zd101.children[0]
#     line1 = region1.children[0]
#     assert region1.dimensions == [
#         [802, 2100], [1901, 2100], [1901, 2223], [802, 2223]]

#     # this region has only one line
#     assert len(region1.children) == 1
#     # which is completely same as region
#     assert line1.dimensions == [[802, 2100],
#                                 [1901, 2100], [1901, 2223], [802, 2223]]
#     assert region1.dimensions == line1.dimensions
#     # this line has two words
#     assert len(line1.children) == 2
#     word1 = line1.children[0]
#     # coords for word01 differ sligthly from line
#     assert word1.dimensions == [[802, 2101],
#                                 [1246, 2101], [1246, 2219], [802, 2219]]
#     assert word1 in line1 and line1 in region1


def test_digital_object_file_path():
    """Ensure type of piece property"""

    ocr_path: str = f'{TEST_RES_DIR}/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml'
    page_piece: DigitalObjectTree = to_digital_object(ocr_path)
    assert isinstance(page_piece.file_path, PurePath)


def test_digital_object_with_spatium():
    """Test ALTO format handling with SP (spatium/space) elements
    
    Ensures that:
    - SP elements are correctly parsed between words
    - Word sequences are properly reconstructed with spaces
    - Spatium dimensions are handled correctly
    """
    
    # arrange
    ocr_path = os.path.join(TEST_RES_DIR, 'groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')
    
    # act
    page_piece = to_digital_object(ocr_path)
    
    # assert - check first text block with spatium
    region1 = page_piece.children[0]
    line1 = region1.children[0]
    
    # verify we have 2 words in the first line
    assert len(line1.children) == 2
    
    # verify the words
    word1 = line1.children[0]
    word2 = line1.children[1]
    assert word1.transcription == 'Neueſte'
    assert word2.transcription == 'Ereigniſſe.'
    
    # verify line transcription includes space between words
    assert line1.transcription == 'Neueſte Ereigniſſe.'
    
    # check second text block with more spatium elements
    region2 = page_piece.children[1]
    line2 = region2.children[0]
    
    # second line has 12 words with SP elements between them
    assert len(line2.children) == 12
    
    # verify transcription is properly spaced
    expected_text = 'In Berlin iſt es auch in der letzten Nacht wieder zu blutigen'
    assert line2.transcription == expected_text
    
    # verify each word exists independently
    words = [word.transcription for word in line2.children]
    assert words == ['In', 'Berlin', 'iſt', 'es', 'auch', 'in', 
                     'der', 'letzten', 'Nacht', 'wieder', 'zu', 'blutigen']


def test_digital_object_urn_nbn_de_gbv_spatium():
    """Test ALTO format with urn+nbn+de+gbv+3+5-2064-fp00000404 resource
    
    This tests:
    - SP elements without WIDTH attributes
    - Single TextBlock with single TextLine structure
    - Punctuation handling with spatium elements
    - English text content processing
    - Critical: String elements WITHOUT SP between them are concatenated
    """
    
    # arrange
    ocr_path = os.path.join(TEST_RES_DIR, 'groundtruth/alto/urn+nbn+de+gbv+3+5-2064-fp00000404.xml')
    
    # act
    page_piece = to_digital_object(ocr_path)
    
    # assert - verify page structure
    assert page_piece.id == 'p00000404'
    assert len(page_piece.dimensions) == 4
    
    # page dimensions
    page_polygon = Polygon(page_piece.dimensions)
    assert page_polygon.bounds == (0.0, 0.0, 1910.0, 2841.0)
    
    # single text block
    assert len(page_piece.children) == 1
    text_block = page_piece.children[0]
    assert text_block.id == 'tb203008'
    
    # single text line
    assert len(text_block.children) == 1
    text_line = text_block.children[0]
    assert text_line.id == 'tl2'
    
    # 10 words-like tokens in the line
    # String elements without SP between them are concatenated:
    # - "(" + "Sand" -> "(Sand" (no SP between them in XML)
    # - "Gate" + ")" + "," -> "Gate)," (no SP between them in XML)
    assert len(text_line.children) == 10
    
    # verify complete transcription with proper spacing
    # SP elements control where spaces appear
    expected_text = 'themselves in the Kum Kapu (Sand Gate), a district on'
    assert text_line.transcription == expected_text
    
    # verify individual words
    words = [word.transcription for word in text_line.children]
    assert words == ['themselves', 'in', 'the', 'Kum', 'Kapu',
                     '(Sand', 'Gate),', 'a', 'district', 'on']
    
    # verify text block and page transcriptions match line
    assert text_block.transcription == expected_text
    assert page_piece.transcription == expected_text
    
    # verify line dimensions
    assert text_line.dimensions == [[463, 492], [1583, 492], [1583, 532], [463, 532]]


