"""Test graceful handling of empty TextLine and Word elements in ALTO and PAGE formats

This test suite validates that:
1. Empty TextLine elements are skipped with warnings
2. Empty regions (parents) are DISCARDED if all lines are empty
3. Partial data is extracted when some lines are valid
4. Warnings are logged for debugging

This follows pytest functional style with fixtures, parametrization, and clear test cases.
"""

from pathlib import Path

import pytest

import ocr_util.eval.model.main as model_main

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(name="temp_xml_file")
def _fixture_temp_xml_file(tmp_path: Path):
    """Factory fixture for creating temporary XML files"""

    def _create_file(content: str) -> str:
        xml_file = tmp_path / f"test_{id(content)}.xml"
        xml_file.write_text(content)
        return str(xml_file)

    return _create_file


@pytest.fixture(name="alto_single_line")
def _fixture_alto_single_line(temp_xml_file):
    """ALTO file with single valid line"""
    alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_001" HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="1000">
        <TextLine ID="line_001" HPOS="0" VPOS="0" WIDTH="300" HEIGHT="100">
          <String ID="word_1" CONTENT="Word1" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100"/>
        </TextLine>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""
    return temp_xml_file(alto_xml)


@pytest.fixture(name="alto_multiple_valid_lines")
def _fixture_alto_multiple_valid_lines(temp_xml_file):
    """ALTO file with three valid lines"""
    alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_001" HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="1000">
        <TextLine ID="line_001" HPOS="0" VPOS="0" WIDTH="300" HEIGHT="100">
          <String ID="word_1" CONTENT="Word1" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100"/>
        </TextLine>
        <TextLine ID="line_002" HPOS="0" VPOS="120" WIDTH="300" HEIGHT="100">
          <String ID="word_2" CONTENT="Word2" HPOS="0" VPOS="120" WIDTH="100" HEIGHT="100"/>
        </TextLine>
        <TextLine ID="line_003" HPOS="0" VPOS="240" WIDTH="300" HEIGHT="100">
          <String ID="word_3" CONTENT="Word3" HPOS="0" VPOS="240" WIDTH="100" HEIGHT="100"/>
        </TextLine>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""
    return temp_xml_file(alto_xml)


# ============================================================================
# TESTS: ALTO V3 - VALID TEXTLINES (NO EMPTY ELEMENTS)
# ============================================================================


def test_alto_single_valid_line(alto_single_line):
    """Verify single valid TextLine is extracted correctly"""
    page = model_main.to_digital_object(alto_single_line)

    assert page is not None
    assert len(page.children) == 1
    assert len(page.children[0].children) == 1


def test_alto_multiple_valid_lines(alto_multiple_valid_lines):
    """Verify multiple valid TextLines are extracted correctly"""
    page = model_main.to_digital_object(alto_multiple_valid_lines)

    assert page is not None
    assert len(page.children) == 1
    assert len(page.children[0].children) == 3


@pytest.mark.parametrize(
    "num_blocks,num_lines_per_block",
    [
        (1, 1),
        (1, 5),
        (3, 2),
        (5, 3),
    ],
)
def test_alto_valid_structures_preserved(temp_xml_file, num_blocks, num_lines_per_block, capsys):
    """Parametrized test: various valid ALTO structures are preserved"""
    blocks = []
    for b in range(num_blocks):
        lines = []
        for l in range(num_lines_per_block):
            line_id = f"line_{b}_{l}"
            lines.append(
                f'        <TextLine ID="{line_id}" HPOS="0" VPOS="{l*120}" WIDTH="100" HEIGHT="100">\n'
                f'          <String ID="word_{b}_{l}" CONTENT="W{b}{l}" HPOS="0" VPOS="{l*120}" WIDTH="100" HEIGHT="100"/>\n'
                f"        </TextLine>"
            )

        blocks.append(
            f'      <TextBlock ID="block_{b}" HPOS="{b*350}" VPOS="0" WIDTH="300" HEIGHT="1000">\n'
            + "\n".join(lines)
            + f"\n      </TextBlock>"
        )

    alto_xml = (
        """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="2000" HEIGHT="1000">
    <PrintSpace WIDTH="2000" HEIGHT="1000" HPOS="0" VPOS="0">
"""
        + "\n".join(blocks)
        + """
    </PrintSpace>
  </Page>
</alto>"""
    )

    alto_path = temp_xml_file(alto_xml)
    page = model_main.to_digital_object(alto_path)
    captured = capsys.readouterr()

    assert page is not None
    assert len(page.children) == num_blocks
    for block in page.children:
        assert len(block.children) == num_lines_per_block

    assert "[WARN ]" not in captured.out


# ============================================================================
# TESTS: ALTO V3 - EMPTY TEXTLINES (GRACEFUL SKIPPING)
# ============================================================================


@pytest.mark.parametrize("empty_position", [0, 1, 2, "all"])
def test_alto_empty_lines_skipped(temp_xml_file, empty_position, capsys):
    """Test empty TextLines are skipped gracefully

    Parameters:
        empty_position: 0=first, 1=middle, 2=last, "all"=all empty
    """
    if empty_position == "all":
        alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_001" HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="1000">
        <TextLine ID="line_001" HPOS="0" VPOS="0" WIDTH="0" HEIGHT="100"/>
        <TextLine ID="line_002" HPOS="0" VPOS="120" WIDTH="0" HEIGHT="100"/>
        <TextLine ID="line_003" HPOS="0" VPOS="240" WIDTH="0" HEIGHT="100"/>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""
        expected_lines = 0
        expected_blocks = 0
    else:
        lines = [
            '        <TextLine ID="line_001" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100">',
            '          <String ID="word_001" CONTENT="Valid1" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100"/>',
            "        </TextLine>",
            '        <TextLine ID="line_002" HPOS="0" VPOS="120" WIDTH="0" HEIGHT="100"/>',
            '        <TextLine ID="line_003" HPOS="0" VPOS="240" WIDTH="100" HEIGHT="100">',
            '          <String ID="word_003" CONTENT="Valid2" HPOS="0" VPOS="240" WIDTH="100" HEIGHT="100"/>',
            "        </TextLine>",
        ]

        alto_xml = (
            """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_001" HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="1000">
"""
            + "\n".join(lines)
            + """
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""
        )
        expected_lines = 2
        expected_blocks = 1

    alto_path = temp_xml_file(alto_xml)
    page = model_main.to_digital_object(alto_path)
    captured = capsys.readouterr()

    assert page is not None
    assert len(page.children) == expected_blocks

    if expected_blocks > 0:
        assert len(page.children[0].children) == expected_lines

    assert "[WARN ] Empty TextLine (no words): line_002" in captured.out


# ============================================================================
# TESTS: ALTO V3 - EMPTY REGIONS (CRITICAL VALIDATION)
# ============================================================================


def test_alto_empty_region_discarded(temp_xml_file, capsys):
    """CRITICAL: Verify empty TextBlock (region) is DISCARDED, not kept

    When no words exist and no line objects were created,
    the parent region should be discarded rather than kept as empty element.
    """
    alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_empty" HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="500">
        <TextLine ID="line_1" HPOS="0" VPOS="0" WIDTH="0" HEIGHT="100"/>
        <TextLine ID="line_2" HPOS="0" VPOS="120" WIDTH="0" HEIGHT="100"/>
      </TextBlock>
      <TextBlock ID="block_valid" HPOS="0" VPOS="500" WIDTH="1000" HEIGHT="500">
        <TextLine ID="line_3" HPOS="0" VPOS="500" WIDTH="100" HEIGHT="100">
          <String ID="word_1" CONTENT="Valid" HPOS="0" VPOS="500" WIDTH="100" HEIGHT="100"/>
        </TextLine>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""

    alto_path = temp_xml_file(alto_xml)
    page = model_main.to_digital_object(alto_path)
    captured = capsys.readouterr()

    assert page is not None
    assert len(page.children) == 1, "Empty block should be discarded"
    assert page.children[0].id == "block_valid"
    assert "[WARN ] Empty TextLine (no words): line_1" in captured.out
    assert "[WARN ] Empty TextLine (no words): line_2" in captured.out
    assert "TextBlock@ID=block_empty has no valid lines - discarding region" in captured.out


def test_alto_multi_block_partial_empty(temp_xml_file, capsys):
    """Test file with multiple blocks where some are empty, some are valid"""
    alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_1" HPOS="0" VPOS="0" WIDTH="300" HEIGHT="300">
        <TextLine ID="line_1" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100">
          <String ID="word_1" CONTENT="A" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100"/>
        </TextLine>
      </TextBlock>
      <TextBlock ID="block_2" HPOS="350" VPOS="0" WIDTH="300" HEIGHT="300">
        <TextLine ID="line_2" HPOS="350" VPOS="0" WIDTH="0" HEIGHT="100"/>
        <TextLine ID="line_3" HPOS="350" VPOS="120" WIDTH="0" HEIGHT="100"/>
      </TextBlock>
      <TextBlock ID="block_3" HPOS="700" VPOS="0" WIDTH="300" HEIGHT="300">
        <TextLine ID="line_4" HPOS="700" VPOS="0" WIDTH="100" HEIGHT="100">
          <String ID="word_4" CONTENT="B" HPOS="700" VPOS="0" WIDTH="100" HEIGHT="100"/>
        </TextLine>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""

    alto_path = temp_xml_file(alto_xml)
    page = model_main.to_digital_object(alto_path)
    captured = capsys.readouterr()

    assert len(page.children) == 2
    assert page.children[0].id == "block_1"
    assert page.children[1].id == "block_3"
    assert "TextBlock@ID=block_2 has no valid lines - discarding region" in captured.out


# ============================================================================
# TESTS: BACKWARD COMPATIBILITY
# ============================================================================


def test_existing_alto_test_data():
    """Verify existing test ALTO file still works correctly

    This ensures we didn't break existing functionality.
    """
    from tests.conftest import TEST_RES_DIR
    import os

    ocr_path = os.path.join(TEST_RES_DIR, "groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml")

    if not os.path.exists(ocr_path):
        pytest.skip(f"Test file not found: {ocr_path}")

    page = model_main.to_digital_object(ocr_path)

    assert page is not None
    assert len(page.children) == 10
    assert page.children[0].id == "block_27"
    assert len(page.children[0].children) >= 1


# ============================================================================
# TESTS: WARNING MESSAGE FORMAT AND CLARITY
# ============================================================================


def test_empty_line_warning_format(temp_xml_file, capsys):
    """Warning includes line ID and is consistent with other warnings"""
    alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_001" HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="1000">
        <TextLine ID="line_specific_empty_id" HPOS="0" VPOS="0" WIDTH="0" HEIGHT="100"/>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""

    alto_path = temp_xml_file(alto_xml)
    model_main.to_digital_object(alto_path)
    captured = capsys.readouterr()

    assert "[WARN ] Empty TextLine (no words): line_specific_empty_id" in captured.out


def test_empty_region_warning_includes_block_id(temp_xml_file, capsys):
    """Warning for discarded region includes block ID"""
    alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_to_discard" HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="1000">
        <TextLine ID="line_1" HPOS="0" VPOS="0" WIDTH="0" HEIGHT="100"/>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""

    alto_path = temp_xml_file(alto_xml)
    model_main.to_digital_object(alto_path)
    captured = capsys.readouterr()

    assert "block_to_discard" in captured.out
    assert "has no valid lines - discarding region" in captured.out


# ============================================================================
# TESTS: INTEGRATION WITH EVALUATION PIPELINE
# ============================================================================


def test_evaluation_file_with_empty_region(temp_xml_file):
    """Verify evaluation can proceed with files having empty regions

    This is the key scenario from the original issue.
    """
    alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="1000" HEIGHT="1000">
    <PrintSpace WIDTH="1000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_1" HPOS="0" VPOS="0" WIDTH="500" HEIGHT="1000">
        <TextLine ID="line_1" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100">
          <String ID="word_1" CONTENT="Hello" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100"/>
        </TextLine>
      </TextBlock>
      <TextBlock ID="block_2" HPOS="500" VPOS="0" WIDTH="500" HEIGHT="1000">
        <TextLine ID="line_2" HPOS="500" VPOS="0" WIDTH="0" HEIGHT="100"/>
        <TextLine ID="line_3" HPOS="500" VPOS="120" WIDTH="0" HEIGHT="100"/>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""

    alto_path = temp_xml_file(alto_xml)

    page = model_main.to_digital_object(alto_path)

    assert page is not None
    assert len(page.children) == 1
    assert page.children[0].id == "block_1"
    assert len(page.children[0].children) == 1
    assert page.children[0].children[0].transcription == "Hello"


def test_mixed_valid_and_empty_regions_extraction(temp_xml_file):
    """Verify correct data extraction when mixing valid and empty regions"""
    alto_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/alto.xsd">
  <Page ID="page_1" WIDTH="2000" HEIGHT="1000">
    <PrintSpace WIDTH="2000" HEIGHT="1000" HPOS="0" VPOS="0">
      <TextBlock ID="block_empty_1" HPOS="0" VPOS="0" WIDTH="500" HEIGHT="1000">
        <TextLine ID="line_1" HPOS="0" VPOS="0" WIDTH="0" HEIGHT="100"/>
        <TextLine ID="line_2" HPOS="0" VPOS="120" WIDTH="0" HEIGHT="100"/>
      </TextBlock>
      <TextBlock ID="block_valid" HPOS="500" VPOS="0" WIDTH="500" HEIGHT="1000">
        <TextLine ID="line_3" HPOS="500" VPOS="0" WIDTH="100" HEIGHT="100">
          <String ID="word_1" CONTENT="Content" HPOS="500" VPOS="0" WIDTH="100" HEIGHT="100"/>
        </TextLine>
      </TextBlock>
      <TextBlock ID="block_empty_2" HPOS="1000" VPOS="0" WIDTH="500" HEIGHT="1000">
        <TextLine ID="line_4" HPOS="1000" VPOS="0" WIDTH="0" HEIGHT="100"/>
      </TextBlock>
      <TextBlock ID="block_valid_2" HPOS="1500" VPOS="0" WIDTH="500" HEIGHT="1000">
        <TextLine ID="line_5" HPOS="1500" VPOS="0" WIDTH="100" HEIGHT="100">
          <String ID="word_2" CONTENT="More" HPOS="1500" VPOS="0" WIDTH="100" HEIGHT="100"/>
        </TextLine>
      </TextBlock>
    </PrintSpace>
  </Page>
</alto>"""

    alto_path = temp_xml_file(alto_xml)
    page = model_main.to_digital_object(alto_path)

    assert page is not None
    assert len(page.children) == 2
    assert page.children[0].id == "block_valid"
    assert page.children[1].id == "block_valid_2"

    text_contents = [
        " ".join([w.transcription for w in line.children]) for block in page.children for line in block.children
    ]
    assert text_contents == ["Content", "More"]
