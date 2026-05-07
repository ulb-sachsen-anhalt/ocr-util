# -*- coding: utf-8 -*-
"""Shared test functionalities"""

from pathlib import Path

import pytest

import ocr_util.eval.model.main as mm


PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_RES_DIR = Path(PROJECT_ROOT_DIR, 'resources')
TEST_RES_DIR = Path(PROJECT_ROOT_DIR, 'tests', 'resources')


@pytest.fixture(name="zd101")
def _fixture_zd101():
    ocr_path = f'{TEST_RES_DIR}/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml'
    page_piece = mm.to_digital_object(ocr_path)
    yield page_piece
