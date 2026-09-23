"""Test specification for representation
of digital assets in OCR ALTO format
"""

import os

import pytest

import ocr_util.eval.model.main as model_main
import ocr_util.eval.model.common as mc

from .conftest import TEST_RES_DIR


def test_textual_input():
    """Behavior of text input as digital object"""

    gt_path = os.path.join(TEST_RES_DIR, "groundtruth/txt/1246734.gt.txt")

    # act
    with pytest.raises(mc.DigitalObjectException) as exc_info:
        model_main.to_digital_object(gt_path)

    assert "XML parsing error " in str(exc_info.value)
