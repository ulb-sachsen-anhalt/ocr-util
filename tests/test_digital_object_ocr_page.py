"""Test specification for representation
of digital assets in OCR PAGE format
"""

import pytest

import ocr_util.eval.model.digital_object_model as dom
import ocr_util.eval.model.common as dc
from ocr_util.eval.model.main import (
    to_digital_object,
)
from tests.conftest import (
    TEST_RES_DIR,
)


@pytest.fixture(name='odem01', scope='module')
def _fixture_odem01():
    """Check basic behavior:
    * a piece shall contain all its digital_objects
    * a piece from different region shall not contain
      digital_objects from other super-piece
    """

    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml'
    page_piece = to_digital_object(ocr_path)
    yield page_piece


def test_to_digital_objects_page_odem_transkribus_gt():
    """Ensure PAGE 2013 Transcribus Groundtruth works"""

    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml'

    # act
    page_piece = to_digital_object(ocr_path)

    # assert
    # 1 region
    assert len(page_piece.children) == 1
    # 23 lines
    assert len(page_piece.children[0].children) == 23
    # first line has 7 words
    assert len(page_piece.children[0].children[0].children) == 7
    # last line has 7 words
    assert len(page_piece.children[0].children[22].children) == 4

    # first line textual content
    line1_text = page_piece.children[0].children[0].transcription
    assert line1_text == 'und erklaͤret die Schrift nicht nur al⸗'
    assert len(line1_text) == 39
    # first region textual content ...
    region_text = page_piece.children[0].transcription
    # changed from 829 to 851 since child texts
    # win over parent's existing text
    assert len(region_text) == 829
    # ... which, in this case (only one single region),
    # is equals the complete top_piece textual content
    assert page_piece.transcription == region_text


def test_to_digital_objects_page_odem():
    """Ensure PAGE 2019 straight from OCR-D ODEM is usable"""

    ocr_path = f'{TEST_RES_DIR}/candidate/frk_page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.xml'

    # act
    page_piece = to_digital_object(ocr_path)

    # assert
    # 3 regions
    assert len(page_piece.children) == 3
    # 23 lines
    assert len(page_piece.children[0].children) == 1
    # first line in region 3 has 7 words
    assert len(page_piece.children[2].children[0].children) == 7

    # first line textual content
    line1_text = page_piece.children[2].children[0].transcription
    assert line1_text == 'und erklaͤret die Schrift nicht nur al⸗'
    assert page_piece.children[2].children[0].transcription == line1_text


def test_digital_objects_odem01_region_and_page(odem01):
    """basics: sub-piece one is contained
    in super-piece of total page"""

    # we have only one sub-piece
    # a single text region
    assert len(odem01.children) == 1
    piece_region1 = odem01.children[0]

    # ensure region 1 contained in page_piece
    assert piece_region1 in odem01


def test_digital_objects_odem01_page_not_in_region(odem01):
    """Evidently can't a super-structure
    *not* be contained in it's own child"""

    # top level pageregion 1 is not contained in region 3
    with pytest.raises(RuntimeError) as _rer:
        assert odem01 not in odem01.children[0]

    # assert
    assert 'is higher/equal level than region0003' in _rer.value.args[0]


@pytest.mark.skip(reason="fails due geometry checks, needs fixing in test data")
def test_digital_objects_transcription_from_rahbar_1771946695():
    """Check behavior with persian text punctuation.
    From line transcription (see below) the point
    is considered to be straight left-hand as final char.

    But, instead of ending with '.گفت'
    it *actually* ends just with 'گفت'
    but *accessing* the splitted tokens
    we find the point '.' has moved to
    to the right-most token, resulting
    into '.آن' and leave 'گفت' for left end!
    """

    # arrange
    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/rahbar-1771946695-00000040.xml'

    # act
    page_piece: dom.DigitalObjectTree = to_digital_object(ocr_path)

    # assert
    assert len(page_piece.children) == 2
    expected_first_row_transcription = '.آن نعت بگردان که مرا خواهی گفت'
    final_token = 'گفت'
    # get down to first line
    first_row_transcription = page_piece.children[0].children[0].transcription
    assert first_row_transcription == expected_first_row_transcription
    last_token = first_row_transcription.split()[-1]
    assert last_token == final_token


def test_digital_objects_geometry_from_rahbar_1185565752():
    """Test problematic data behavior"""

    # arrange
    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/rahbar-1185565752-00000086.xml'

    # act
    with pytest.raises(RuntimeError) as _err:
        to_digital_object(ocr_path)

    # assert
    assert 'Word@ID=w_243 invalid ' in str(_err.value.args[0])


def test_digital_object_from_odem_kba_transformed():
    """Ensure converted PAGE data still readable
    They yielded error back inside digital-eval
    after one tries to run evaluations with them
    """

    # urn+nbn+de+gbv+3+1-112032-p0026-5_ger.gt
    # arrange
    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-112032-p0026-5_ger.gt.xml'

    # act
    dot: dom.DigitalObjectTree = to_digital_object(ocr_path)

    # assert
    assert dot is not None
    assert dot.level == dom.DigitalObjectLevel.PAGE
    first_line_text = '„dem Staube, es wird Ernſt!“ fluͤſterte er'
    assert dot.children[0].children[0].transcription == first_line_text


def test_digital_object_from_ocr4all_groundtruth():
    """Ensure PAGE data from OCR4all groundtruth readable
    and respects different text-equiv elements, created during
    the annotation process. Also verifies that reading order is respected.
    """

    # urn+nbn+de+gbv+3+1-112032-p0026-5_ger.gt
    # arrange
    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+5-14325-fp-00000441.xml'

    # act
    tho_tree: dom.DigitalObjectTree = to_digital_object(ocr_path)

    # assert
    assert tho_tree is not None
    assert tho_tree.level == dom.DigitalObjectLevel.PAGE
    # According to reading order, first region is r20 (index 0)
    assert tho_tree.children[0].id == 'r20'
    assert tho_tree.children[0].children[0].transcription == 'MAR'
    # Region r1 with 'Maréchal, sm. dignité, مير' is at reading order index 12
    assert tho_tree.children[12].id == 'r1'
    assert tho_tree.children[12].children[0].transcription == 'Maréchal, sm. dignité, مير'


def test_digital_object_inconsistent_geometry(capsys):
    """Behavior with inconsistent geometry
    
    changed 2026-05-06: print alert to stdout, but go on processing.
    """

    # urn+nbn+de+gbv+3+1-112032-p0026-5_ger.gt
    # arrange
    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/1751466019_18481220.xml'

    # act
    to_digital_object(ocr_path)

    # assert
    captured = capsys.readouterr()
    assert 'not contained in parent box' in captured.out


def test_invalid_page_from_vlm():
    """Fix behavior with previously invalid PAGE data - problem was 
    our way of determing namespaces"""

    # urn+nbn+de+gbv+3+1-112032-p0026-5_ger.gt
    # arrange
    ocr_path = f'{TEST_RES_DIR}/xml/0001.xml'

    # act
    dtr = to_digital_object(ocr_path)

    # assert
    assert dtr is not None
    assert dtr.level == dom.DigitalObjectLevel.PAGE
    assert len(dtr.children) == 1
