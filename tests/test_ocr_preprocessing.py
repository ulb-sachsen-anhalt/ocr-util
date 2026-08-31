# -*- coding: utf-8 -*-
"""OCR Metric Test Module"""

# import random

from pathlib import Path

import pytest

import ocr_util.eval.metrics as digem
import ocr_util.eval.preprocessing as dipre

from .conftest import TEST_RES_DIR


def test_letter_preprocessor_reports_character_class_removals():
    """Report the concrete basis and removals used by a letter metric."""

    preprocessor = dipre.LetterPreprocessor("A 2!\n")

    preprocessor.run()

    report = preprocessor.preprocessing_report
    assert report["input_count"] == 5
    assert report["input_unit"] == "characters"
    assert report["output_count"] == 1
    assert report["output_unit"] == "characters"
    assert report["steps"][-1]["name"] == "remove non-letter characters"
    assert report["steps"][-1]["details"] == {
        "whitespace": 2,
        "punctuation": 1,
        "digits": 1,
    }


# default reference
THE_COMBINED_A_FOX = 'the á lazy brown fox jumps over the hump'
THE_LAZY_FOX = 'the lazy brown fox jumps over the hump'
THE_FOX_LAZY = 'the fox lazy brown jumps over the hump'
THE_FOX_INPUT_IR = 'the hump lazy brown fox fox fox jumps'


def test_metric_unicode_normalization_happens():
    """Normalization required and effects examined
    raw1 has "á" as {U+00E0} => gets canonical decomposed
    raw2 has "á" as {U+0061}+{U+0301}
    """

    # arrange
    raw1 = 'the á lazy brown fox jumps over the hump'
    raw2 = THE_COMBINED_A_FOX
    prepr1 = dipre.TextPreprocessor(raw1)
    prepr1.code_norm = dipre.UC_NORMALIZATION_NFKD
    prepr1.normalize_encoding()
    norm1 = prepr1.result
    prepr2 = dipre.TextPreprocessor(raw2)
    prepr2.code_norm = dipre.UC_NORMALIZATION_NFKD
    prepr2.normalize_encoding()
    norm2 = prepr2.result

    # act
    similarity = digem.levenshtein_norm(norm1, norm2)
    assert 1.0 == pytest.approx(similarity, abs=1e-6)

    # assert
    # although both raw string look similar, they differ in fact
    assert raw1 != raw2
    # after normalization, they *are* similar
    assert norm1 == norm2
    assert len(norm1) == 41
    # the "á" char from raw1 string gets
    # decomposed into {U+0061}+{U+0301}
    # by normalization with de-composition
    # therefore normalised str is
    # one char longer
    assert len(raw1) + 1 == len(norm1)


def test_metric_unicode_normalization_not_happens():
    """Normalization has no effect since
    the letters "a" and "á" are still different
    after normalization, they just stay
    {U+0061} and {U+00e1} for NFC and {U+0061}+{U+0301} for NFKD
    """

    # arrange
    raw1 = THE_LAZY_FOX
    raw2 = THE_COMBINED_A_FOX

    prepr1 = dipre.TextPreprocessor(raw1)
    prepr1.code_norm = dipre.UC_NORMALIZATION_DEFAULT
    prepr1.normalize_encoding()
    norm1_nfc = prepr1.result

    prepr2 = dipre.TextPreprocessor(raw1)
    prepr2.code_norm = dipre.UC_NORMALIZATION_NFKD
    prepr2.normalize_encoding()
    norm1_nfkd = prepr2.result

    prepr3 = dipre.TextPreprocessor(raw2)
    prepr3.code_norm = dipre.UC_NORMALIZATION_DEFAULT
    prepr3.normalize_encoding()
    norm2_nfc = prepr3.result

    prepr4 = dipre.TextPreprocessor(raw2)
    prepr4.code_norm = dipre.UC_NORMALIZATION_NFKD
    prepr4.normalize_encoding()
    norm2_nfkd = prepr4.result

    # act
    sim_nfc = digem.levenshtein_norm(norm1_nfc, norm2_nfc)
    sim_nfkd = digem.levenshtein_norm(norm1_nfkd, norm2_nfkd)

    # assert
    assert 0.95 == sim_nfc
    assert 0.92 == pytest.approx(sim_nfkd, 1e-2)


def test_piece_to_dict_text_alto():
    """Inspect difference between textual preprocessing
    and enhanced texts for dictionary metrics handles
    ALTO format properly
    """

    alto_path = Path(f'{TEST_RES_DIR}/dict_metric/alto.xml').absolute()
    no_sanit, _ = dipre.file_to_text(alto_path, oneliner=True)
    prep1 = dipre.DictionaryTextPreprocessor(alto_path)

    # act
    prep1.code_norm = dipre.UC_NORMALIZATION_NFKD
    prep1.run()
    prep_dict_txt = prep1.result

    # assert
    assert no_sanit == "Dieſe uͤberfruͤhte An⸗ kunft des hailigen Raimarſ. ſachſen- ſtolz, aͤhnlich"
    assert 10 == len(no_sanit.split())
    assert prep_dict_txt == "Diese überfrühte Ankunft des hailigen Raimars sachsenstolz, ähnlich"
    assert 8 == len(prep_dict_txt.split())


def test_piece_to_dict_text_page2019():
    """Can process PAGE 2019 for dict text"""

    page_path = Path(f'{TEST_RES_DIR}/dict_metric/page2019.xml').absolute()
    no_sanit, _ = dipre.file_to_text(page_path, oneliner=True)
    prep1 = dipre.DictionaryTextPreprocessor(page_path)
    prep1.code_norm = dipre.UC_NORMALIZATION_NFKD

    # act
    prep1.run()
    prep_dict_txt = prep1.result

    # assert
    assert no_sanit == "Dieſe uͤberfruͤhte An⸗ kunft des hailigen Raimarſ. ſachſen- ſtolz, aͤhnlich"
    assert 10 == len(no_sanit.split())

    assert prep_dict_txt == "Diese überfrühte Ankunft des hailigen Raimars sachsenstolz, ähnlich"
    assert 8 == len(prep_dict_txt.split())


def test_piece_to_dict_text_page2013():
    """Can also process legacy transkribus PAGE 2013 for dict texts"""

    page_path = Path(f'{TEST_RES_DIR}/dict_metric/page2013.xml').absolute()
    page_text_no_sanit, _ = dipre.file_to_text(page_path, oneliner=True)

    # act
    prep1 = dipre.DictionaryTextPreprocessor(page_path)
    prep1.code_norm = dipre.UC_NORMALIZATION_NFKD

    # act
    prep1.run()
    page_text_norm = prep1.result

    # assert
    assert page_text_no_sanit == "Dieſe uͤberfruͤhte An⸗ kunft des hailigen Raimarſ. ſachſen- ſtolz, aͤhnlich"
    assert 10 == len(page_text_no_sanit.split())
    assert page_text_norm == "Diese überfrühte Ankunft des hailigen Raimars sachsenstolz, ähnlich"
    assert 8 == len(page_text_norm.split())
