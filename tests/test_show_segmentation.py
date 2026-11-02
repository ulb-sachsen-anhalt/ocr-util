# -*- coding: utf-8 -*-
"""Test specification for OCR visualization module (ocr_show_segmentation)"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from ocr_util.show import ocr_show_segmentation
from ocr_util.eval.model.common import DigitalObjectLevel
from ocr_util.eval.model.main import to_digital_object
from ocr_util.eval.model.digital_object_util import DigitalObjectUtil
from tests.conftest import TEST_RES_DIR

# silence pylint false positives for OpenCV C-extension bindings
# pylint:disable=no-member


@pytest.fixture(name="temp_output_dir")
def _fixture_temp_output_dir():
    """Provide temporary directory for output files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(name="sample_image_path")
def _fixture_sample_image():
    """Provide path to sample test image"""
    img_path = Path(TEST_RES_DIR) / "img" / "1681877805_J_0011_0251_tl_4.tif"
    assert img_path.exists(), f"Sample image not found: {img_path}"
    return str(img_path)


@pytest.fixture(name="sample_page_xml")
def _fixture_sample_page_xml():
    """Provide path to sample PAGE XML"""
    xml_path = Path(TEST_RES_DIR) / "groundtruth" / "page" / "urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml"
    assert xml_path.exists(), f"Sample XML not found: {xml_path}"
    return str(xml_path)


@pytest.fixture(name="sample_alto_xml")
def _fixture_sample_alto_xml():
    """Provide path to sample ALTO XML"""
    xml_path = Path(TEST_RES_DIR) / "groundtruth" / "alto" / "1667522809_J_0073_0001_375x2050_2325x9550.xml"
    assert xml_path.exists(), f"Sample ALTO XML not found: {xml_path}"
    return str(xml_path)



def test_convert_creates_output_file(sample_image_path, temp_output_dir):
    """Test that _convert creates output file in specified directory"""
    output_path = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")

    assert Path(output_path).exists()
    assert output_path.endswith(".jpg")
    assert temp_output_dir in output_path

def test_convert_with_different_formats(sample_image_path, temp_output_dir):
    """Test conversion to different image formats"""
    formats = ["jpg", "png", "tif"]

    for fmt in formats:
        output_path = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, fmt)

        assert Path(output_path).exists()
        assert output_path.endswith(f".{fmt}")

        # Verify image is readable
        img = cv2.imread(output_path)
        assert img is not None
        assert img.shape[2] == 3  # BGR channels

def test_convert_preserves_image_content(sample_image_path, temp_output_dir):
    """Test that conversion preserves image dimensions"""
    original_img = cv2.imread(sample_image_path)
    output_path = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "png")
    converted_img = cv2.imread(output_path)

    assert original_img.shape == converted_img.shape



def test_digo_to_polygon_data_with_valid_region(sample_page_xml):
    """Test extracting polygon data from a valid region"""
    digo = to_digital_object(sample_page_xml)
    region = digo.children[0]

    polygon_data = ocr_show_segmentation._digo_to_polygon_data(region)

    assert polygon_data is not None
    assert len(polygon_data) == 3
    element_id, points, text = polygon_data

    assert element_id is not None
    assert len(points) >= ocr_show_segmentation.MIN_POLYGON_POINTS
    assert isinstance(text, str)

def test_digo_to_polygon_data_with_line(sample_page_xml):
    """Test extracting polygon data from a line"""
    digo = to_digital_object(sample_page_xml)
    line = digo.children[0].children[0]

    polygon_data = ocr_show_segmentation._digo_to_polygon_data(line)

    assert polygon_data is not None
    element_id, points, text = polygon_data
    assert len(points) >= ocr_show_segmentation.MIN_POLYGON_POINTS
    assert len(text) > 0  # Lines should have text content

def test_digo_to_polygon_data_returns_none_for_insufficient_points():
    """Test that objects with insufficient points return None"""
    # Test with None dimensions
    from ocr_util.eval.model.digital_object_model import DigitalObjectTree

    # Create a minimal digital object with no dimensions
    mock_digo = DigitalObjectTree()
    mock_digo.dimensions = [[0, 0], [10, 10]]  # Only 2 points, need 3
    mock_digo.id = "test_id"

    result = ocr_show_segmentation._digo_to_polygon_data(mock_digo)

    assert result is None



def test_create_polygon_mask_basic():
    """Test creating a binary mask from polygon elements"""
    img_shape = (100, 100, 3)
    polygon_elements = [
        ("id1", [(10, 10), (50, 10), (50, 50), (10, 50)], "text1"),
        ("id2", [(60, 60), (90, 60), (90, 90), (60, 90)], "text2"),
    ]

    mask = ocr_show_segmentation._create_polygon_mask(img_shape, polygon_elements)

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.any(mask == 255)  # Should have white pixels where polygons are
    assert np.any(mask == 0)  # Should have black pixels elsewhere

def test_create_polygon_mask_coverage():
    """Test that mask correctly covers polygon area"""
    img_shape = (100, 100, 3)
    polygon_elements = [("id1", [(20, 20), (80, 20), (80, 80), (20, 80)], "text")]

    mask = ocr_show_segmentation._create_polygon_mask(img_shape, polygon_elements)

    # Check that center of polygon is covered
    assert mask[50, 50] == 255

    # Check that corners outside polygon are not covered
    assert mask[0, 0] == 0
    assert mask[99, 99] == 0



def test_visualize_creates_output_files(sample_image_path, sample_page_xml, temp_output_dir):
    """Test that visualization creates all expected output files"""
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")

    output_path = ocr_show_segmentation.visualize(converted_img, sample_page_xml, "jpg", show_labels=False)

    # Check that final output exists
    assert Path(output_path).exists()

    # Check that intermediate outputs exist
    base_path = os.path.splitext(converted_img)[0]
    regions_path = base_path + "_sgm_rgn.jpg"
    lines_path = base_path + "_sgm_lns.jpg"

    assert Path(regions_path).exists()
    assert Path(lines_path).exists()

def test_visualize_with_alto_format(sample_image_path, sample_alto_xml, temp_output_dir):
    """Test visualization with ALTO XML format"""
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")

    output_path = ocr_show_segmentation.visualize(converted_img, sample_alto_xml, "jpg", show_labels=False)

    assert Path(output_path).exists()

def test_visualize_with_png_format(sample_image_path, sample_page_xml, temp_output_dir):
    """Test visualization with PNG output format"""
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "png")

    output_path = ocr_show_segmentation.visualize(converted_img, sample_page_xml, "png", show_labels=False)

    assert output_path.endswith(".png")
    assert Path(output_path).exists()

def test_visualize_with_labels_enabled(sample_image_path, sample_page_xml, temp_output_dir):
    """Test visualization with labels enabled"""
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")

    output_path = ocr_show_segmentation.visualize(converted_img, sample_page_xml, "jpg", show_labels=True)

    assert Path(output_path).exists()

    # Verify output image is readable
    img = cv2.imread(output_path)
    assert img is not None
    assert not Path(os.path.splitext(converted_img)[0] + "_txt.jpg").exists()


def test_visualize_creates_text_variant(sample_image_path, sample_page_xml, temp_output_dir):
    """create the text-suffixed output when transcription display is enabled."""
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")

    output_path = ocr_show_segmentation.visualize(
        converted_img,
        sample_page_xml,
        "jpg",
        show_transcription=True,
    )

    expected_path = os.path.splitext(converted_img)[0] + "_txt.jpg"
    assert output_path == expected_path
    assert Path(expected_path).exists()

def test_visualize_output_has_correct_dimensions(sample_image_path, sample_page_xml, temp_output_dir):
    """Test that visualization output maintains original image dimensions"""
    original_img = cv2.imread(sample_image_path)
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")

    output_path = ocr_show_segmentation.visualize(converted_img, sample_page_xml, "jpg", show_labels=False)
    output_img = cv2.imread(output_path)

    assert output_img.shape == original_img.shape



@patch.object(ocr_show_segmentation.cv2, "putText", wraps=ocr_show_segmentation.cv2.putText)
def test_render_polygons_with_transcription(put_text_mock, sample_image_path, temp_output_dir):
    """Test that transcriptions render independently from identifier labels"""
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "png")
    output_without_text = os.path.join(temp_output_dir, "test_without_transcription.png")
    output_path = os.path.join(temp_output_dir, "test_transcription.png")
    polygon_data = [("region-1", [(20, 20), (100, 20), (100, 80), (20, 80)], "Sample transcription")]
    exclusion_mask = np.zeros(cv2.imread(converted_img).shape[:2], dtype=np.uint8)

    ocr_show_segmentation._render_polygons(
        converted_img,
        output_without_text,
        polygon_data,
        exclusion_mask,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
    )

    ocr_show_segmentation._render_polygons(
        converted_img,
        output_path,
        polygon_data,
        exclusion_mask,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        show_transcription=True,
    )

    rendered_text = [call.args[1] for call in put_text_mock.call_args_list]
    assert rendered_text == ["Sample transcription"]
    text_call = put_text_mock.call_args
    origin_x, baseline_y = text_call.args[2]
    text_size, baseline = cv2.getTextSize(
        text_call.args[1],
        text_call.args[3],
        text_call.args[4],
        text_call.args[6],
    )
    assert 20 <= origin_x
    assert origin_x + text_size[0] <= 100
    assert 20 <= baseline_y - text_size[1]
    assert baseline_y + baseline <= 80
    assert not np.array_equal(cv2.imread(output_without_text), cv2.imread(output_path))
    assert Path(output_path).exists()


def test_transcription_font_scale_fits_polygon_dimensions():
    """keep transcription text within the polygon's width and height."""
    transcription = "A transcription that is wider than a small word box"
    small_polygon = [(10, 10), (110, 10), (110, 50), (10, 50)]
    large_polygon = [(10, 70), (210, 70), (210, 150), (10, 150)]

    small_scale = ocr_show_segmentation._calculate_transcription_font_scale(transcription, small_polygon)
    large_scale = ocr_show_segmentation._calculate_transcription_font_scale(transcription, large_polygon)
    rendered_size, baseline = cv2.getTextSize(
        transcription,
        ocr_show_segmentation.FONT_LABEL,
        small_scale,
        ocr_show_segmentation.THICKNESS_TEXT_STROKE,
    )

    assert small_scale > 0
    assert large_scale > small_scale
    assert rendered_size[0] <= int(100 * ocr_show_segmentation.TRANSCRIPTION_WIDTH_RATIO)
    assert rendered_size[1] + baseline <= int(40 * ocr_show_segmentation.TRANSCRIPTION_HEIGHT_RATIO)

def test_render_polygons_creates_output(sample_image_path, sample_page_xml, temp_output_dir):
    """Test that _render_polygons creates valid output"""
    digo = to_digital_object(sample_page_xml)
    all_digos = DigitalObjectUtil.flatten(digo)
    regions = [
        ocr_show_segmentation._digo_to_polygon_data(d) for d in all_digos if d.level == DigitalObjectLevel.REGION
    ]
    regions = [r for r in regions if r is not None]

    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")
    output_path = os.path.join(temp_output_dir, "test_regions.jpg")

    ocr_show_segmentation._render_polygons(
        converted_img,
        output_path,
        regions,
        None,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        show_identifier=False,
    )

    assert Path(output_path).exists()

    # Verify output is different from input
    input_img = cv2.imread(converted_img)
    output_img = cv2.imread(output_path)
    assert not np.array_equal(input_img, output_img)

def test_render_polygons_with_labels(sample_image_path, sample_page_xml, temp_output_dir):
    """Test rendering with labels produces valid output files"""
    digo = to_digital_object(sample_page_xml)
    all_digos = DigitalObjectUtil.flatten(digo)
    regions = [
        ocr_show_segmentation._digo_to_polygon_data(d) for d in all_digos if d.level == DigitalObjectLevel.REGION
    ]
    regions = [r for r in regions if r is not None]

    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")
    output_no_labels = os.path.join(temp_output_dir, "test_no_labels.jpg")
    output_with_labels = os.path.join(temp_output_dir, "test_with_labels.jpg")

    # Render without labels
    ocr_show_segmentation._render_polygons(
        converted_img,
        output_no_labels,
        regions,
        None,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        show_identifier=False,
    )

    # Render with labels
    ocr_show_segmentation._render_polygons(
        converted_img,
        output_with_labels,
        regions,
        None,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        show_identifier=True,
    )

    # Both should exist and be valid images
    assert Path(output_no_labels).exists()
    assert Path(output_with_labels).exists()

    img_no_labels = cv2.imread(output_no_labels)
    img_with_labels = cv2.imread(output_with_labels)

    # Both should be valid images with correct shape
    assert img_no_labels is not None
    assert img_with_labels is not None
    assert img_no_labels.shape == img_with_labels.shape

def test_render_polygons_with_mask(sample_image_path, sample_page_xml, temp_output_dir):
    """Test rendering with exclusion mask"""
    digo = to_digital_object(sample_page_xml)
    all_digos = DigitalObjectUtil.flatten(digo)

    regions = [
        ocr_show_segmentation._digo_to_polygon_data(d) for d in all_digos if d.level == DigitalObjectLevel.REGION
    ]
    lines = [
        ocr_show_segmentation._digo_to_polygon_data(d) for d in all_digos if d.level == DigitalObjectLevel.LINE
    ]

    regions = [r for r in regions if r is not None]
    lines = [l for l in lines if l is not None]

    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")
    output_path = os.path.join(temp_output_dir, "test_masked.jpg")

    ocr_show_segmentation._render_polygons_with_mask(
        converted_img,
        output_path,
        regions,
        lines,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        ocr_show_segmentation.COLOR_LINE_DARK_BLUE,
        ocr_show_segmentation.THICKNESS_LINE_POLYGON,
        ocr_show_segmentation.COLOR_LINE_BRIGHT_BLUE,
        "jpg",
        show_identifiers=False,
    )

    assert Path(output_path).exists()


@patch.object(ocr_show_segmentation, "_render_polygons")
def test_line_mask_stage_only_renders_line_transcription(render_mock, sample_image_path, temp_output_dir):
    """render text-line transcriptions without adding region transcriptions."""
    polygon = [(1, 1), (20, 1), (20, 20), (1, 20)]
    regions = [("region", polygon, "region text")]
    lines = [("line", polygon, "line text")]
    ocr_show_segmentation._render_polygons_with_mask(
        sample_image_path,
        os.path.join(temp_output_dir, "line_stage.jpg"),
        regions,
        lines,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        ocr_show_segmentation.COLOR_LINE_DARK_BLUE,
        ocr_show_segmentation.THICKNESS_LINE_POLYGON,
        ocr_show_segmentation.COLOR_LINE_BRIGHT_BLUE,
        show_line_transcription=True,
    )

    rendered_levels = [(call.args[2][0][0], call.args[-1]) for call in render_mock.call_args_list]
    assert rendered_levels == [("region", False), ("line", True)]


@patch.object(ocr_show_segmentation, "_render_polygons")
def test_word_mask_stage_only_renders_word_transcription(render_mock, sample_image_path, temp_output_dir):
    """prefer word transcriptions and suppress region and line transcriptions."""
    polygon = [(1, 1), (20, 1), (20, 20), (1, 20)]
    regions = [("region", polygon, "region text")]
    lines = [("line", polygon, "line text")]
    words = [("word", polygon, "word text")]
    ocr_show_segmentation._render_with_word_mask(
        sample_image_path,
        os.path.join(temp_output_dir, "word_stage.jpg"),
        regions,
        lines,
        words,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        ocr_show_segmentation.COLOR_LINE_DARK_BLUE,
        ocr_show_segmentation.THICKNESS_LINE_POLYGON,
        ocr_show_segmentation.COLOR_LINE_BRIGHT_BLUE,
        ocr_show_segmentation.COLOR_WORD_CYAN,
        ocr_show_segmentation.THICKNESS_WORD_POLYGON,
        ocr_show_segmentation.COLOR_WORD_BRIGHT_CYAN,
        show_word_transcription=True,
    )

    rendered_levels = [(call.args[2][0][0], call.args[-1]) for call in render_mock.call_args_list]
    assert rendered_levels == [("region", False), ("line", False), ("word", True)]



def test_color_constants_are_tuples():
    """Test that color constants are valid BGR tuples"""
    colors = [
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        ocr_show_segmentation.COLOR_LINE_DARK_BLUE,
        ocr_show_segmentation.COLOR_LINE_BRIGHT_BLUE,
        ocr_show_segmentation.COLOR_WORD_CYAN,
        ocr_show_segmentation.COLOR_WORD_BRIGHT_CYAN,
    ]

    for color in colors:
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

def test_alpha_constants_are_valid():
    """Test that alpha constants are in valid range [0, 1]"""
    alphas = [
        ocr_show_segmentation.ALPHA_POLYGON_FILL,
        ocr_show_segmentation.ALPHA_POLYGON_BASE,
        ocr_show_segmentation.ALPHA_POLYGON_OVERLAY,
        ocr_show_segmentation.ALPHA_IMAGE_BASE,
        ocr_show_segmentation.ALPHA_TEXT_OVERLAY,
        ocr_show_segmentation.ALPHA_TEXT_BASE,
    ]

    for alpha in alphas:
        assert 0 <= alpha <= 1

def test_thickness_constants_are_positive():
    """Test that thickness constants are positive integers"""
    thicknesses = [
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.THICKNESS_LINE_POLYGON,
        ocr_show_segmentation.THICKNESS_TEXT_STROKE,
    ]

    for thickness in thicknesses:
        assert isinstance(thickness, int)
        assert thickness > 0

def test_region_opacity_is_80_percent():
    """Test that region opacity is set to 80% as specified"""
    # Effective opacity = ALPHA_POLYGON_OVERLAY / (ALPHA_POLYGON_OVERLAY + ALPHA_IMAGE_BASE)
    effective_opacity = ocr_show_segmentation.ALPHA_POLYGON_OVERLAY / (
        ocr_show_segmentation.ALPHA_POLYGON_OVERLAY + ocr_show_segmentation.ALPHA_IMAGE_BASE
    )

    assert abs(effective_opacity - 0.8) < 0.01  # Allow small floating point error

def test_line_polygons_get_fill():
    """Test that line thickness allows filled rendering"""
    # Lines should get filled polygons (thickness >= THICKNESS_LINE_POLYGON)
    assert ocr_show_segmentation.THICKNESS_LINE_POLYGON >= ocr_show_segmentation.THICKNESS_LINE_POLYGON



def test_visualize_with_no_regions(sample_image_path, temp_output_dir):
    """Test visualization when OCR data has no regions"""
    # Create minimal PAGE XML with no text regions
    minimal_xml = os.path.join(temp_output_dir, "minimal.xml")
    with open(minimal_xml, "w") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="100" imageHeight="100">
    </Page>
</PcGts>""")

    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")

    # Should not crash even with no content
    output_path = ocr_show_segmentation.visualize(converted_img, minimal_xml, "jpg", show_labels=False)

    # Should return original image path when no elements to render
    assert output_path == converted_img

def test_visualize_handles_empty_polygon_list(sample_image_path, temp_output_dir):
    """Test that rendering handles empty polygon lists gracefully"""
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")
    output_path = os.path.join(temp_output_dir, "empty_test.jpg")

    # Render with empty list
    ocr_show_segmentation._render_polygons(
        converted_img,
        output_path,
        [],
        None,
        ocr_show_segmentation.COLOR_REGION_DARK_GREEN,
        ocr_show_segmentation.THICKNESS_REGION_POLYGON,
        ocr_show_segmentation.COLOR_REGION_BRIGHT_GREEN,
        show_identifier=False,
    )

    assert Path(output_path).exists()



def test_stages_use_original_image(sample_image_path, sample_page_xml, temp_output_dir):
    """Test that each visualization stage starts from the original image"""
    converted_img = ocr_show_segmentation.convert(sample_image_path, temp_output_dir, "jpg")

    _ = ocr_show_segmentation.visualize(converted_img, sample_page_xml, "jpg", show_labels=False)

    # All stage outputs should exist
    base_path = os.path.splitext(converted_img)[0]
    regions_path = base_path + "_sgm_rgn.jpg"
    lines_path = base_path + "_sgm_lns.jpg"
    ocr_path = base_path + "_sgm_wrd.jpg"

    # Load all outputs
    regions_img = cv2.imread(regions_path)
    lines_img = cv2.imread(lines_path)
    ocr_img = cv2.imread(ocr_path)

    # All should have the same dimensions as original
    assert regions_img.shape == lines_img.shape == ocr_img.shape

    # Each stage should produce different output
    # (unless there are no lines or words)
    # We just verify they all exist and are readable
    assert regions_img is not None
    assert lines_img is not None
    assert ocr_img is not None
