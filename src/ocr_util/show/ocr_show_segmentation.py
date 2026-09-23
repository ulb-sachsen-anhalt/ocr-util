# -*- coding: utf-8 -*-
"""ULB DD/IT OCR Segmentation Visualization"""

import argparse
import os
from typing import Any, cast

import cv2 as _cv2
import numpy as np

# cv2 bindings are generated dynamically and frequently confuse static analyzers.
cv2 = cast(Any, _cv2)

# silence pylint false positives for OpenCV C-extension bindings
# pylint:disable=no-member
# pylint:disable=c-extension-no-member

from ocr_util.eval.model.common import DigitalObjectLevel
from ocr_util.eval.model.digital_object_util import DigitalObjectUtil
from ocr_util.eval.model.main import to_digital_object

VERBOSE_LEVEL1 = False  # Set to True for detailed debug output

# ===========================
# Visualization Constants
# ===========================

# Colors in BGR format (OpenCV convention)
# Region colors (green tones)
COLOR_REGION_DARK_GREEN = (0, 196, 0)  # Dark green for region polygons
COLOR_REGION_BRIGHT_GREEN = (0, 255, 0)  # Bright green for region labels

# Line colors (blue tones)
COLOR_LINE_DARK_BLUE = (0, 0, 196)  # Dark blue for line polygons
COLOR_LINE_BRIGHT_BLUE = (0, 0, 255)  # Bright blue for line labels

# Word colors (cyan/light blue tones)
COLOR_WORD_CYAN = (255, 255, 0)  # Cyan for word polygons
COLOR_WORD_BRIGHT_CYAN = (255, 255, 128)  # Bright cyan for word labels

# Line thickness for polygon rendering
THICKNESS_REGION_POLYGON = 4  # Thicker lines for regions
THICKNESS_LINE_POLYGON = 2  # Medium lines for text lines
THICKNESS_WORD_POLYGON = 1  # Thin lines for word polygons
THICKNESS_TEXT_STROKE = 2  # Text stroke thickness

# Transparency/Alpha blending weights
ALPHA_POLYGON_FILL = 0.4  # Transparency for filled polygons
ALPHA_POLYGON_BASE = 0.6  # Base image weight for filled polygons
ALPHA_POLYGON_OVERLAY = 0.8  # Polygon overlay transparency (80% opacity)
ALPHA_IMAGE_BASE = 0.2  # Base image weight for polygon overlay
ALPHA_TEXT_OVERLAY = 0.8  # Text overlay transparency
ALPHA_TEXT_BASE = 0.2  # Base image weight for text overlay

# Text positioning margins
MARGIN_LABEL_LEFT = 10  # Left margin for ID labels
MARGIN_LABEL_TOP = 30  # Top margin for ID labels

# Font settings
FONT_SIZE_LABEL = 0.8  # Font scale for ID labels
FONT_LABEL = cv2.FONT_HERSHEY_SIMPLEX  # Simple font for labels

# Polygon requirements
MIN_POLYGON_POINTS = 3  # Minimum points for a valid polygon


def _convert(path_img, output_dir, img_format="jpg"):
    """Convert input image to the specified format in the output directory.

    Args:
        path_img: Path to input image
        output_dir: Output directory
        img_format: Output image format (default: 'jpg')
    """
    img = cv2.imread(path_img, cv2.IMREAD_COLOR)
    filename = os.path.basename(path_img).split(".")[0]
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    full_conversion_path = os.path.join(output_dir, filename + "." + img_format)
    msg_convert = "convert {} to {}".format(path_img, full_conversion_path)
    print("[INFO] " + msg_convert)
    cv2.imwrite(full_conversion_path, img)
    return full_conversion_path


def _digo_to_polygon_data(digo):
    """Convert a DigitalObjectTree node into a (id, polygon_points, text) tuple.
    Returns None when no polygon points are available."""
    if not digo.dimensions or len(digo.dimensions) < MIN_POLYGON_POINTS:
        return None

    # Convert points to integer coordinates for OpenCV
    polygon_points = [(int(pt[0]), int(pt[1])) for pt in digo.dimensions]

    try:
        text = digo.transcription
    except RuntimeError:
        text = ""

    polygon_data = (digo.id, polygon_points, text)
    if VERBOSE_LEVEL1:
        print(
            "[DEBUG] read polygon data: id={}, points={}, text={}".format(
                polygon_data[0], len(polygon_data[1]), polygon_data[2][:20] if polygon_data[2] else ""
            )
        )
    return polygon_data


def _visualize(path_img, path_ocr, img_format="jpg", show_labels=False):
    """Visualize OCR data overlaid on image.

    Args:
        path_img: Path to input image
        path_ocr: Path to OCR data file
        img_format: Output image format (default: 'jpg')
        show_labels: Whether to render polygon ID labels (default: False)
    """
    start_msg = "merge img {} with ocr data {}".format(path_img, path_ocr)
    print("[INFO] " + start_msg)
    root_digo = to_digital_object(path_ocr)
    all_digos = DigitalObjectUtil.flatten(root_digo)

    regions = [_digo_to_polygon_data(d) for d in all_digos if d.level == DigitalObjectLevel.REGION]
    lines = [_digo_to_polygon_data(d) for d in all_digos if d.level == DigitalObjectLevel.LINE]
    words = [_digo_to_polygon_data(d) for d in all_digos if d.level == DigitalObjectLevel.WORD]

    regions = [r for r in regions if r is not None]
    lines = [l for l in lines if l is not None]
    words = [w for w in words if w is not None]

    print("[INFO] read {} regions".format(len(regions)))
    print("[INFO] read {} lines".format(len(lines)))
    print("[INFO] read {} words".format(len(words)))

    # Generate output paths with different suffixes
    base_path = os.path.splitext(path_img)[0]
    path_regions = base_path + "_sgm_rgn." + img_format
    path_lines = base_path + "_sgm_lns." + img_format
    path_words = base_path + "_sgm_wrd." + img_format

    # Stage 1: Render regions only
    if regions:
        _render_polygons(
            path_img,
            path_regions,
            regions,
            None,
            COLOR_REGION_DARK_GREEN,
            THICKNESS_REGION_POLYGON,
            COLOR_REGION_BRIGHT_GREEN,
            show_labels,
        )

    # Stage 2: Render regions + lines, but mask out regions where lines exist
    if lines:
        _render_polygons_with_mask(
            path_img,
            path_lines,
            regions,
            lines,
            COLOR_REGION_DARK_GREEN,
            THICKNESS_REGION_POLYGON,
            COLOR_REGION_BRIGHT_GREEN,
            COLOR_LINE_DARK_BLUE,
            THICKNESS_LINE_POLYGON,
            COLOR_LINE_BRIGHT_BLUE,
            img_format,
            show_labels,
        )

    # Stage 3: Render everything, but mask out regions/lines where words exist
    if words:
        _render_with_word_mask(
            path_img,
            path_words,
            regions,
            lines,
            words,
            COLOR_REGION_DARK_GREEN,
            THICKNESS_REGION_POLYGON,
            COLOR_REGION_BRIGHT_GREEN,
            COLOR_LINE_DARK_BLUE,
            THICKNESS_LINE_POLYGON,
            COLOR_LINE_BRIGHT_BLUE,
            COLOR_WORD_CYAN,
            THICKNESS_WORD_POLYGON,
            COLOR_WORD_BRIGHT_CYAN,
            img_format,
            show_labels,
        )

    # Return the most complete visualization
    current_img = path_img
    if words:
        current_img = path_words
    elif lines:
        current_img = path_lines
    elif regions:
        current_img = path_regions

    return current_img


def _render_polygons(input_path, output_path, els, exclusion_mask, color, thickness, text_color, show_labels=False):
    """Render polygons using actual shape points from OCR data.

    Args:
        input_path: Input image path
        output_path: Output image path
        els: List of polygon elements to render
        exclusion_mask: Binary mask (numpy array) where 255 = exclude rendering, None = render everywhere
        color: Color for polygon outline
        thickness: Line thickness
        text_color: Color for ID labels
        show_labels: Whether to render ID labels (default: False)
    """
    msg_render = "render {} polygon elements from {} to {}".format(len(els), input_path, output_path)
    print("[INFO] " + msg_render)
    img = cv2.imread(input_path)
    overlay_poly = img.copy()
    overlay_text = img.copy()

    for el in els:
        element_id = el[0]
        polygon_points = el[1]

        # Convert to numpy array for OpenCV
        pts = np.array(polygon_points, dtype=np.int32)

        # Draw polygon outline
        cv2.polylines(overlay_poly, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw semi-transparent filled polygon for better visibility
        overlay_fill = overlay_poly.copy()
        cv2.fillPoly(overlay_fill, [pts], color)
        cv2.addWeighted(overlay_fill, ALPHA_POLYGON_FILL, overlay_poly, ALPHA_POLYGON_BASE, 0, overlay_poly)

        # Add ID label at the first point (top-left) if labels are enabled
        if show_labels:
            label_x, label_y = polygon_points[0]
            cv2.putText(
                overlay_text,
                "ID: {}".format(element_id),
                (label_x + MARGIN_LABEL_LEFT, label_y + MARGIN_LABEL_TOP),
                FONT_LABEL,
                FONT_SIZE_LABEL,
                text_color,
                THICKNESS_TEXT_STROKE,
            )

    # Apply exclusion mask if provided
    if exclusion_mask is not None:
        # Where mask is 255 (white), keep original image; where 0 (black), apply overlay
        inv_mask = cv2.bitwise_not(exclusion_mask)
        inv_mask_3ch = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

        # Blend overlays only where mask allows (inv_mask is 255)
        overlay_poly_masked = cv2.bitwise_and(overlay_poly, inv_mask_3ch)
        img_masked = cv2.bitwise_and(img, inv_mask_3ch)
        overlay_text_masked = cv2.bitwise_and(overlay_text, inv_mask_3ch)

        # Keep original image where excluded
        img_excluded = cv2.bitwise_and(img, cv2.bitwise_not(inv_mask_3ch))

        # Combine masked overlay with original image
        temp_img = cv2.addWeighted(overlay_poly_masked, ALPHA_POLYGON_OVERLAY, img_masked, ALPHA_IMAGE_BASE, 0)
        temp_img = cv2.add(temp_img, img_excluded)

        # Apply text overlay if labels are enabled
        if show_labels:
            overlay_text_masked_only = cv2.bitwise_and(overlay_text, inv_mask_3ch)
            img_for_text = cv2.bitwise_and(temp_img, inv_mask_3ch)
            img_excluded_text = cv2.bitwise_and(temp_img, cv2.bitwise_not(inv_mask_3ch))

            temp_img = cv2.addWeighted(overlay_text_masked_only, ALPHA_TEXT_OVERLAY, img_for_text, ALPHA_TEXT_BASE, 0)
            img = cv2.add(temp_img, img_excluded_text)
        else:
            img = temp_img
    else:
        # No mask, render normally
        cv2.addWeighted(overlay_poly, ALPHA_POLYGON_OVERLAY, img, ALPHA_IMAGE_BASE, 0, img)
        if show_labels:
            cv2.addWeighted(overlay_text, ALPHA_TEXT_OVERLAY, img, ALPHA_TEXT_BASE, 0, img)

    cv2.imwrite(output_path, img)
    print("[INFO] written: {}".format(output_path))


def _create_polygon_mask(img_shape, polygon_elements):
    """Create a binary mask from polygon elements.

    Args:
        img_shape: Shape of the image (height, width, channels)
        polygon_elements: List of polygon elements

    Returns:
        Binary mask where 255 = polygon area, 0 = background
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for el in polygon_elements:
        polygon_points = el[1]
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _render_polygons_with_mask(
    input_path,
    output_path,
    regions,
    lines,
    region_color,
    region_thickness,
    region_text_color,
    line_color,
    line_thickness,
    line_text_color,
    img_format="jpg",
    show_labels=False,
):
    """Render regions and lines, masking out regions where lines exist.

    Args:
        input_path: Input image path
        output_path: Output image path
        regions: List of region polygon elements
        lines: List of line polygon elements
        region_color, region_thickness, region_text_color: Region rendering parameters
        line_color, line_thickness, line_text_color: Line rendering parameters
        img_format: Output image format (default: 'jpg')
        show_labels: Whether to render ID labels (default: False)
    """
    print("[INFO] rendering with line mask to prevent region-line color overlap")

    # Load image to get dimensions
    img = cv2.imread(input_path)

    # Create mask for lines
    line_mask = _create_polygon_mask(img.shape, lines)

    # Render regions with line mask (regions excluded where lines exist)
    base_path = os.path.splitext(output_path)[0]
    temp_path = base_path + "_temp." + img_format
    _render_polygons(
        input_path, temp_path, regions, line_mask, region_color, region_thickness, region_text_color, show_labels
    )

    # Render lines on top without mask
    _render_polygons(temp_path, output_path, lines, None, line_color, line_thickness, line_text_color, show_labels)

    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)


def _render_with_word_mask(
    input_path,
    output_path,
    regions,
    lines,
    words,
    region_color,
    region_thickness,
    region_text_color,
    line_color,
    line_thickness,
    line_text_color,
    word_color,
    word_thickness,
    word_text_color,
    img_format="jpg",
    show_labels=False,
):
    """Render regions, lines, and words, masking appropriately to prevent overlaps.

    Args:
        input_path: Input image path
        output_path: Output image path
        regions: List of region polygon elements
        lines: List of line polygon elements
        words: List of word polygon elements
        region_color, region_thickness, region_text_color: Region rendering parameters
        line_color, line_thickness, line_text_color: Line rendering parameters
        word_color, word_thickness, word_text_color: Word rendering parameters
        img_format: Output image format (default: 'jpg')
        show_labels: Whether to render ID labels (default: False)
    """
    print("[INFO] rendering with word mask to prevent color overlap")

    # Load image to get dimensions
    img = cv2.imread(input_path)

    # Create masks
    line_mask = _create_polygon_mask(img.shape, lines)
    word_mask = _create_polygon_mask(img.shape, words)

    # Combine masks: regions excluded where lines OR words exist
    region_exclusion_mask = cv2.bitwise_or(line_mask, word_mask)

    # Lines excluded where words exist
    line_exclusion_mask = word_mask

    # Render regions with combined mask
    base_path = os.path.splitext(output_path)[0]
    temp_path_1 = base_path + "_temp1." + img_format
    _render_polygons(
        input_path,
        temp_path_1,
        regions,
        region_exclusion_mask,
        region_color,
        region_thickness,
        region_text_color,
        show_labels,
    )

    # Render lines with word mask
    temp_path_2 = base_path + "_temp2." + img_format
    _render_polygons(
        temp_path_1, temp_path_2, lines, line_exclusion_mask, line_color, line_thickness, line_text_color, show_labels
    )

    # Render word polygons on top without mask
    _render_polygons(temp_path_2, output_path, words, None, word_color, word_thickness, word_text_color, show_labels)

    # Clean up temporary files
    for temp in [temp_path_1, temp_path_2]:
        if os.path.exists(temp):
            os.remove(temp)


########
# MAIN #
########
if __name__ == "__main__":
    APP_ARGUMENTS = argparse.ArgumentParser()
    APP_ARGUMENTS.add_argument("-i", "--image", required=True, help="path image file (TIF)")
    APP_ARGUMENTS.add_argument("-o", "--ocr", required=True, help="path OCR file (ALTO or PAGE XML)")
    APP_ARGUMENTS.add_argument(
        "-d",
        "--output-dir",
        required=False,
        default=os.getcwd(),
        help="output directory for visualization results (default: current working directory)",
    )
    APP_ARGUMENTS.add_argument(
        "-f",
        "--format",
        required=False,
        default="jpg",
        choices=["jpg", "jpeg", "png", "tif", "tiff"],
        help="output image format (default: jpg)",
    )
    APP_ARGUMENTS.add_argument("-l", "--label", action="store_true", help="render polygon ID labels (default: False)")
    APP_ARGUMENTS.add_argument("-v", "--verbose", required=False, help="output info")

    ARGS = vars(APP_ARGUMENTS.parse_args())
    PATH_IMG = ARGS["image"]
    PATH_OCR = ARGS["ocr"]
    OUTPUT_DIR = ARGS["output_dir"]
    IMG_FORMAT = ARGS["format"]
    SHOW_LABELS = ARGS["label"]
    VERBOSE_LEVEL1 = ARGS["verbose"]

    if VERBOSE_LEVEL1:
        print("[INFO] call with args " + str(ARGS))
        print("[INFO] output directory: {}".format(OUTPUT_DIR))
        print("[INFO] output format: {}".format(IMG_FORMAT))
        print("[INFO] show labels: {}".format(SHOW_LABELS))

    if os.path.exists(PATH_IMG) and os.path.exists(PATH_OCR):
        CONVERTED_TMP = _convert(PATH_IMG, OUTPUT_DIR, IMG_FORMAT)
        final_output = _visualize(CONVERTED_TMP, PATH_OCR, IMG_FORMAT, SHOW_LABELS)
        print("[INFO] visualization complete. Final output: {}".format(final_output))
    else:
        print("[ERROR] {} or {} not existing!".format(PATH_IMG, PATH_OCR))
