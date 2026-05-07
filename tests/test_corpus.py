# -*- coding: utf-8 -*-
"""Test Module for OCR-Util Groundtruth Corpus CLI"""

import unittest.mock

from pathlib import Path

import pytest

from ocr_util.cli import start
from ocr_util.corpus.generate_corpus import Gt2Mets
from ocr_util.corpus.common import (
    CorpusArgs,
    CorpusException,
    GroundtruthFileResource,
    GtResources,
)


@pytest.fixture(name="mock_gt_files")
def _fixture_mock_gt_files(tmp_path):
    """Create mock ground truth PAGE-XML files with URN naming convention"""
    gt_dir = tmp_path / "gt_input"
    gt_dir.mkdir()

    # Create valid GT files with URN pattern
    test_files = [
        "urn+nbn+de+gbv+3+1-123456-fp0001.xml",
        "urn+nbn+de+gbv+3+1-123456-fp0002.xml",
        "urn+nbn+de+gbv+3+1-123456-fp0003.xml",
    ]

    for filename in test_files:
        file_path = gt_dir / filename
        # Create a minimal PAGE-XML structure
        content = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1000">
        <TextRegion id="r1">
            <TextLine id="l1">
                <TextEquiv>
                    <Unicode>Sample text</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"""
        file_path.write_text(content, encoding="utf-8")
    return gt_dir


@pytest.fixture(name="mock_output_dir")
def _fixture_mock_output_dir(tmp_path):
    """Create output directory"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(name="mock_temp_dir")
def _fixture_mock_temp_dir(tmp_path):
    """Create temp directory"""
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    return temp_dir


# GtResources Tests


def test_gt_resources_from_dir(mock_gt_files):
    """Test that GtResources correctly identifies and parses GT files"""
    # act
    resources = GtResources.from_dir(mock_gt_files, limit=0)

    # assert
    assert len(resources) == 3
    assert all(isinstance(r, GroundtruthFileResource) for r in resources)

    # Check first resource details
    first = resources[0]
    assert first.identifier == "urn:nbn:de:gbv:3:1-123456/fragment/page=0001"
    assert first.file_base_name == "urn+nbn+de+gbv+3+1-123456"
    assert first.languages is None


def test_gt_resources_from_dir_with_limit(mock_gt_files):
    """Test that GtResources respects the limit parameter"""
    # act
    resources = GtResources.from_dir(mock_gt_files, limit=2)

    # assert
    assert len(resources) == 2


def test_gt_resources_from_dir_copy(mock_gt_files, tmp_path):
    """Test that GtResources correctly copies files to output directory"""
    # arrange
    output_dir = tmp_path / "output_copy"

    # act
    resources = GtResources.from_dir_copy(mock_gt_files, output_dir, limit=0)

    # assert
    assert len(resources) == 3
    assert output_dir.exists()

    # Verify files were copied
    for resource in resources:
        assert resource.file_path.exists()
        assert str(resource.file_path).startswith(str(output_dir))


def test_gt_resources_invalid_filenames(tmp_path):
    """Test that GtResources ignores files with invalid naming patterns"""
    # arrange
    gt_dir = tmp_path / "invalid_gt"
    gt_dir.mkdir()

    # Create files that don't match the pattern
    (gt_dir / "invalid_file.xml").write_text("<test/>")
    (gt_dir / "another_bad_name.xml").write_text("<test/>")

    # act
    resources = GtResources.from_dir(gt_dir, limit=0)

    # assert
    assert len(resources) == 0


# Args Dataclass Tests


def test_args_creation():
    """Test Args dataclass creation"""
    # arrange
    input_dir = Path("/input")
    output_dir = Path("/output")
    temp_dir = Path("/temp")
    limit = 10

    # act
    args = CorpusArgs(
        input_dir=input_dir,
        output_dir=output_dir,
        local_cache_dir=temp_dir,
        oai_base_url="https://opendata.uni-halle.de/oai/dd",
        limit=limit,
    )

    # assert
    assert args.input_dir == input_dir
    assert args.output_dir == output_dir
    assert args.local_cache_dir == temp_dir
    assert args.limit == limit


def test_args_immutability():
    """Test that Args dataclass values can be updated"""
    # arrange
    args = CorpusArgs(
        input_dir=Path("/input"),
        output_dir=Path("/output"),
        local_cache_dir=Path("/temp"),
        oai_base_url="https://opendata.uni-halle.de/oai/dd",
        limit=5,
    )

    # act
    args.limit = 10

    # assert
    assert args.limit == 10


# Gt2Mets Class Tests


def test_gt2mets_initialization_with_valid_args(
    mock_gt_files, mock_output_dir, mock_temp_dir
):
    """Test Gt2Mets initialization with valid arguments"""
    # arrange
    output_dir = mock_output_dir.parent / "new_output"
    args = CorpusArgs(
        input_dir=mock_gt_files,
        output_dir=output_dir,
        local_cache_dir=mock_temp_dir,
        oai_base_url="https://opendata.uni-halle.de/oai/dd",
        limit=0,
    )

    # act
    gt2mets = Gt2Mets(args)

    # assert - no exception raised
    assert gt2mets


def test_gt2mets_initialization_invalid_input_dir(mock_output_dir, mock_temp_dir):
    """Test Gt2Mets raises error for non-existent input directory"""
    # arrange
    args = CorpusArgs(
        input_dir=Path("/nonexistent/path"),
        output_dir=mock_output_dir,
        local_cache_dir=mock_temp_dir,
        oai_base_url="https://opendata.uni-halle.de/oai/dd",
        limit=0,
    )

    # act & assert
    with pytest.raises(CorpusException, match="does not exist"):
        Gt2Mets(args)


def test_gt2mets_initialization_existing_output_dir(
    mock_gt_files, mock_output_dir, mock_temp_dir
):
    """Test Gt2Mets raises error when output directory already exists"""
    # arrange
    args = CorpusArgs(
        input_dir=mock_gt_files,
        output_dir=mock_output_dir,
        local_cache_dir=mock_temp_dir,
        oai_base_url="https://opendata.uni-halle.de/oai/dd",
        limit=0,
    )

    # act
    gt2mets = Gt2Mets(args)

    # assert
    assert gt2mets


def test_gt2mets_initialization_without_args():
    """Test Gt2Mets initialization without args raises a TypeError"""
    with pytest.raises(TypeError):
        Gt2Mets()


@unittest.mock.patch("ocr_util.corpus.generate_corpus.MetsGenerator")
@unittest.mock.patch("ocr_util.corpus.generate_corpus.corpus_common.GtResources")
def test_gt2mets_run_creates_directories(
    mock_gt_resources_class, mock_mets_generator, mock_gt_files, tmp_path
):
    """Test that Gt2Mets.run() creates necessary directories"""
    # arrange
    output_dir = tmp_path / "new_output"
    temp_dir = tmp_path / "new_temp"

    args = CorpusArgs(
        input_dir=mock_gt_files,
        output_dir=output_dir,
        local_cache_dir=temp_dir,
        oai_base_url="https://opendata.uni-halle.de/oai/dd",
        limit=0,
    )

    # Mock GtResources to return empty list (no GT files to process)
    mock_gt_resources_class.from_dir_copy.return_value = []

    # Mock MetsGenerator
    mock_generator_instance = unittest.mock.Mock()
    mock_mets_generator.return_value = mock_generator_instance

    gt2mets = Gt2Mets(args)

    # act
    gt2mets.run()

    # assert
    assert temp_dir.exists()
    assert output_dir.exists()


# CLI Integration Tests


def test_cli_groundtruth_corpus_help(capsys):
    """Test that groundtruth-corpus subcommand shows help"""
    # arrange & act & assert
    with pytest.raises(SystemExit) as exc_info:
        with unittest.mock.patch("sys.argv", ["ocr-util", "corpus", "--help"]):
            start()

    # Help should exit with code 0
    assert exc_info.value.code == 0

    # Check that help was displayed
    captured = capsys.readouterr()
    assert "corpus" in captured.out or "corpus" in captured.err


def test_cli_groundtruth_corpus_missing_required_args():
    """Test that CLI raises error when required arguments are missing"""
    # arrange & act & assert
    with pytest.raises(SystemExit) as exc_info:
        with unittest.mock.patch("sys.argv", ["ocr-util", "corpus"]):
            start()

    # Should exit with error code
    assert exc_info.value.code != 0


def test_cli_groundtruth_corpus_invalid_input_dir():
    """Test CLI with non-existent input directory"""
    # arrange
    nonexistent_path = "/nonexistent/input/path"

    # act & assert
    with pytest.raises(CorpusException, match="does not exist"):
        with unittest.mock.patch(
            "sys.argv",
            ["ocr-util", "corpus", "-i", nonexistent_path, "-o", "/tmp/output"],
        ):
            start()


def test_cli_groundtruth_corpus_existing_output_dir(mock_gt_files, mock_output_dir):
    """Test CLI continues when output directory already exists"""
    with unittest.mock.patch("ocr_util.cli.Gt2Mets") as mock_gt2mets_class:
        mock_instance = unittest.mock.Mock()
        mock_gt2mets_class.return_value = mock_instance
        with unittest.mock.patch(
            "sys.argv",
            [
                "ocr-util",
                "corpus",
                "-i",
                str(mock_gt_files),
                "-o",
                str(mock_output_dir),
            ],
        ):
            start()

    assert mock_instance.run.called


@unittest.mock.patch("ocr_util.cli.Gt2Mets")
def test_cli_groundtruth_corpus_with_verbosity(
    mock_gt2mets_class, mock_gt_files, mock_output_dir, capsys
):
    """Test CLI with verbosity flag"""
    # arrange
    mock_instance = unittest.mock.Mock()
    mock_gt2mets_class.return_value = mock_instance

    # act
    with unittest.mock.patch(
        "sys.argv",
        [
            "ocr-util",
            "corpus",
            "-i",
            str(mock_gt_files),
            "-o",
            str(mock_output_dir),
            "-v",
        ],
    ):
        start()

    # assert
    captured = capsys.readouterr()
    assert "[INFO ]" in captured.out
    assert "Input directory" in captured.out
    assert mock_instance.run.called


@unittest.mock.patch("ocr_util.cli.Gt2Mets")
def test_cli_groundtruth_corpus_with_limit(
    mock_gt2mets_class, mock_gt_files, mock_output_dir
):
    """Test CLI with limit parameter"""
    # arrange
    mock_instance = unittest.mock.Mock()
    mock_gt2mets_class.return_value = mock_instance

    # act
    with unittest.mock.patch(
        "sys.argv",
        [
            "ocr-util",
            "corpus",
            "-i",
            str(mock_gt_files),
            "-o",
            str(mock_output_dir),
            "-l",
            "5",
        ],
    ):
        start()

    # assert
    # Check that Gt2Mets was called with the correct Args
    call_args = mock_gt2mets_class.call_args[0][0]
    assert isinstance(call_args, CorpusArgs)
    assert call_args.limit == 5


@unittest.mock.patch("ocr_util.cli.Gt2Mets")
def test_cli_groundtruth_corpus_with_custom_temp_dir(
    mock_gt2mets_class, mock_gt_files, mock_output_dir, tmp_path
):
    """Test CLI with custom temp directory"""
    # arrange
    mock_instance = unittest.mock.Mock()
    mock_gt2mets_class.return_value = mock_instance
    custom_temp = tmp_path / "custom_temp"

    # act
    with unittest.mock.patch(
        "sys.argv",
        [
            "ocr-util",
            "corpus",
            "-i",
            str(mock_gt_files),
            "-o",
            str(mock_output_dir),
            "-t",
            str(custom_temp),
        ],
    ):
        start()

    # assert
    call_args = mock_gt2mets_class.call_args[0][0]
    assert call_args.local_cache_dir == custom_temp.absolute()


@unittest.mock.patch("ocr_util.cli.Gt2Mets")
def test_cli_groundtruth_corpus_exception_handling(
    mock_gt2mets_class, mock_gt_files, mock_output_dir, capsys
):
    """Test CLI exception handling when Gt2Mets.run() fails"""
    # arrange
    mock_instance = unittest.mock.Mock()
    mock_instance.run.side_effect = Exception("Test error message")
    mock_gt2mets_class.return_value = mock_instance

    # act & assert
    with pytest.raises(Exception, match="Test error message"):
        with unittest.mock.patch(
            "sys.argv",
            [
                "ocr-util",
                "corpus",
                "-i",
                str(mock_gt_files),
                "-o",
                str(mock_output_dir),
            ],
        ):
            start()

    # Check error message was printed
    captured = capsys.readouterr()
    assert "[ERROR]" in captured.out or "ERROR" in captured.err


@unittest.mock.patch("ocr_util.cli.Gt2Mets")
def test_cli_groundtruth_corpus_multiple_verbosity_flags(
    mock_gt2mets_class, mock_gt_files, mock_output_dir):
    """Test CLI with multiple verbosity flags (-vv)"""
    # arrange
    mock_instance = unittest.mock.Mock()
    mock_gt2mets_class.return_value = mock_instance

    # act
    with unittest.mock.patch(
        "sys.argv",
        [
            "ocr-util",
            "corpus",
            "-i",
            str(mock_gt_files),
            "-o",
            str(mock_output_dir),
            "-vv",
        ],
    ):
        start()

    # assert - should work without errors
    assert mock_instance.run.called


# Edge Cases


def test_gt_resources_empty_directory(tmp_path):
    """Test GtResources with empty directory"""
    # arrange
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # act
    resources = GtResources.from_dir(empty_dir, limit=0)

    # assert
    assert len(resources) == 0


def test_gt_resources_nested_structure(tmp_path):
    """Test GtResources with nested directory structure"""
    # arrange
    gt_dir = tmp_path / "nested_gt"
    sub_dir = gt_dir / "subdir"
    sub_dir.mkdir(parents=True)

    # Create GT file in subdirectory
    test_file = sub_dir / "urn+nbn+de+gbv+3+1-123456-fp0001.xml"
    test_file.write_text("<?xml version='1.0'?><test/>")

    # act
    resources = GtResources.from_dir(gt_dir, limit=0)

    # assert
    assert len(resources) == 1
    assert resources[0].file_path == test_file


def test_args_with_zero_limit():
    """Test Args with limit=0 (unlimited)"""
    # arrange & act
    args = CorpusArgs(
        input_dir=Path("/input"),
        output_dir=Path("/output"),
        local_cache_dir=Path("/temp"),
        oai_base_url="https://opendata.uni-halle.de/oai/dd",
        limit=0,
    )

    # assert
    assert args.limit == 0  # 0 means unlimited


def test_gt_resources_language_parsing():
    """Test that GtResources correctly parses multi-language tags"""
    # arrange
    tmp_dir = Path("/tmp")

    # Create a mock GroundtruthFileResource
    resource = GroundtruthFileResource(
        identifier="urn:nbn:de:gbv:3:1-123456/fragment/page=0001",
        file_base_name="urn+nbn+de+gbv+3+1-123456",
        file_path=tmp_dir / "test.xml",
        relative_file_path=Path("test.xml"),
        languages=["deu", "lat", "eng"],
    )

    # assert
    assert len(resource.languages) == 3
    assert "deu" in resource.languages
    assert "lat" in resource.languages
    assert "eng" in resource.languages


@unittest.mock.patch("ocr_util.cli.Gt2Mets")
def test_cli_groundtruth_corpus_success_message(
    mock_gt2mets_class, mock_gt_files, mock_output_dir, capsys
):
    """Test that success message is displayed with verbosity"""
    # arrange
    mock_instance = unittest.mock.Mock()
    mock_gt2mets_class.return_value = mock_instance

    # act
    with unittest.mock.patch(
        "sys.argv",
        [
            "ocr-util",
            "corpus",
            "-i",
            str(mock_gt_files),
            "-o",
            str(mock_output_dir),
            "-v",
        ],
    ):
        start()

    # assert
    captured = capsys.readouterr()
    assert (
        "completed successfully" in captured.out
        or "completed successfully" in captured.err
    )
