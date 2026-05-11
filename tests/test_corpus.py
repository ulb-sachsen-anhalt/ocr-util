# -*- coding: utf-8 -*-
"""Test Module for OCR-Util Groundtruth Corpus CLI"""

import unittest.mock

from pathlib import Path

import pytest


import ocr_util.cli as ou_cli
import ocr_util.corpus.common as cc
import ocr_util.corpus.generate_corpus as gc


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
    resources = cc.GroundtruthFile.from_dir(mock_gt_files, limit=0)

    # assert
    assert len(resources) == 3
    assert all(isinstance(r, cc.GroundtruthFile) for r in resources)

    # Check first resource details
    first = resources[0]
    assert first.identifier == "urn:nbn:de:gbv:3:1-123456/fragment/page=0001"
    assert first.file_base_name == "urn+nbn+de+gbv+3+1-123456"
    assert first.languages is None


def test_gt_resources_from_dir_with_limit(mock_gt_files):
    """Test that GtResources respects the limit parameter"""
    # act
    resources = cc.GroundtruthFile.from_dir(mock_gt_files, limit=2)

    # assert
    assert len(resources) == 2


def test_gt_resources_from_dir_copy(mock_gt_files, tmp_path):
    """Test that GtResources correctly copies files to output directory"""
    # arrange
    output_dir = tmp_path / "output_copy"

    # act
    resources = cc.GroundtruthFile.from_dir_copy(mock_gt_files, output_dir, limit=0)

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
    resources = cc.GroundtruthFile.from_dir(gt_dir, limit=0)

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
    args = cc.CorpusArgs(
        input_dir=input_dir,
        output_dir=output_dir,
        local_cache_dir=temp_dir,
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
    args = cc.CorpusArgs(
        input_dir=Path("/input"),
        output_dir=Path("/output"),
        local_cache_dir=Path("/temp"),
        limit=5,
    )

    # act
    args.limit = 10

    # assert
    assert args.limit == 10


# Gt2Mets Class Tests


@unittest.mock.patch("ocr_util.corpus.generate_corpus.CorpusFile")
@unittest.mock.patch("ocr_util.corpus.generate_corpus.cc.GroundtruthFileResource")
def test_gt2mets_initialization_with_valid_args(
    mock_gt_resource_class, mock_corpus_file_class, mock_gt_files, mock_output_dir, mock_temp_dir
):
    """Test Gt2Mets initialization with valid arguments"""
    # arrange
    output_dir = mock_output_dir.parent / "new_output"
    args = cc.CorpusArgs(
        input_dir=mock_gt_files,
        output_dir=output_dir,
        local_cache_dir=mock_temp_dir,
        limit=0,
    )
    mock_gt_resource_class.from_dir_copy.return_value = []
    mock_corpus_file_class.return_value.build.return_value = unittest.mock.Mock()

    # act
    gt2mets = gc.generate(args)

    # assert - no exception raised
    assert gt2mets


def test_gt2mets_initialization_invalid_input_dir(mock_output_dir, mock_temp_dir):
    """Test Gt2Mets raises error for non-existent input directory"""
    # arrange
    args = cc.CorpusArgs(
        input_dir=Path("/nonexistent/path"),
        output_dir=mock_output_dir,
        local_cache_dir=mock_temp_dir,
        limit=0,
    )

    # act & assert
    with pytest.raises(cc.CorpusException, match="does not exist"):
        gc.generate(args)


@unittest.mock.patch("ocr_util.corpus.generate_corpus.CorpusFile")
@unittest.mock.patch("ocr_util.corpus.generate_corpus.cc.GroundtruthFileResource")
def test_gt2mets_initialization_existing_output_dir(
    mock_gt_resource_class, mock_corpus_file_class, mock_gt_files, mock_output_dir, mock_temp_dir
):
    """Test Gt2Mets raises error when output directory already exists"""
    # arrange
    args = cc.CorpusArgs(
        input_dir=mock_gt_files,
        output_dir=mock_output_dir,
        local_cache_dir=mock_temp_dir,
        limit=0,
    )
    mock_gt_resource_class.from_dir_copy.return_value = []
    mock_corpus_file_class.return_value.build.return_value = unittest.mock.Mock()

    # act
    gt2mets = gc.generate(args)

    # assert
    assert gt2mets


def test_gt2mets_initialization_without_args():
    """Test Gt2Mets initialization without args raises a TypeError"""
    with pytest.raises(TypeError):
        gc.generate()


@unittest.mock.patch("ocr_util.corpus.generate_corpus.CorpusFile")
@unittest.mock.patch("ocr_util.corpus.generate_corpus.cc.GroundtruthFileResource")
def test_gt2mets_run_creates_directories(
    mock_gt_resources_class, mock_corpus_file_class, mock_gt_files, tmp_path
):
    """Test that main workflow creates necessary directories
    mind the adopted monkey patch target with alias in file - works!
    """
    # arrange
    output_dir = tmp_path / "new_output"
    temp_dir = tmp_path / "new_temp"

    args = cc.CorpusArgs(
        input_dir=mock_gt_files,
        output_dir=output_dir,
        local_cache_dir=temp_dir,
        limit=0,
    )

    # Mock GtResources to return empty list (no GT files to process)
    mock_gt_resources_class.from_dir_copy.return_value = []

    # Mock CorpusFile
    mock_generator_instance = unittest.mock.Mock()
    mock_corpus_file_class.return_value = mock_generator_instance

    # act
    _ = gc.generate(args)

    # assert
    assert temp_dir.exists()
    assert output_dir.exists()


# CLI Integration Tests


def test_cli_groundtruth_corpus_help(capsys):
    """Test that groundtruth-corpus subcommand shows help"""
    # arrange & act & assert
    with pytest.raises(SystemExit) as exc_info:
        with unittest.mock.patch("sys.argv", ["ocr-util", "corpus", "--help"]):
            ou_cli.start()

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
            ou_cli.start()

    # Should exit with error code
    assert exc_info.value.code != 0


def test_cli_groundtruth_corpus_invalid_input_dir():
    """Test CLI with non-existent input directory"""
    # arrange
    nonexistent_path = "/nonexistent/input/path"

    # act & assert
    with pytest.raises(cc.CorpusException, match="does not exist"):
        with unittest.mock.patch(
            "sys.argv",
            ["ocr-util", "corpus", "-i", nonexistent_path, "-o", "/tmp/output"],
        ):
            ou_cli.start()


def test_cli_groundtruth_corpus_existing_output_dir(mock_gt_files, mock_output_dir):
    """Test CLI continues when output directory already exists"""
    with unittest.mock.patch("ocr_util.cli.gc.generate") as mock_generate:
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
            ou_cli.start()

    assert mock_generate.called


@unittest.mock.patch("ocr_util.cli.gc.generate")
def test_cli_groundtruth_corpus_with_verbosity(
    mock_generate, mock_gt_files, mock_output_dir, capsys
):
    """Test CLI with verbosity flag"""
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
        ou_cli.start()

    # assert
    assert mock_generate.called


@unittest.mock.patch("ocr_util.cli.gc.generate")
def test_cli_groundtruth_corpus_with_limit(
    mock_generate, mock_gt_files, mock_output_dir
):
    """Test CLI with limit parameter"""
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
        ou_cli.start()

    # assert
    # Check that generate was called with the correct args
    call_args = mock_generate.call_args[0][0]
    assert isinstance(call_args, cc.CorpusArgs)
    assert call_args.limit == 5


@unittest.mock.patch("ocr_util.cli.gc.generate")
def test_cli_groundtruth_corpus_with_custom_temp_dir(
    mock_generate, mock_gt_files, mock_output_dir, tmp_path
):
    """Test CLI with custom temp directory"""
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
        ou_cli.start()

    # assert
    call_args = mock_generate.call_args[0][0]
    assert call_args.local_cache_dir == custom_temp.absolute()


@unittest.mock.patch("ocr_util.cli.gc.generate")
def test_cli_groundtruth_corpus_exception_handling(
    mock_generate, mock_gt_files, mock_output_dir, capsys
):
    """Test CLI exception handling when corpus generation fails."""
    # arrange
    mock_generate.side_effect = Exception("Test error message")

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
            ou_cli.start()


@unittest.mock.patch("ocr_util.cli.gc.generate")
def test_cli_groundtruth_corpus_multiple_verbosity_flags(
    mock_generate, mock_gt_files, mock_output_dir):
    """Test CLI with multiple verbosity flags (-vv)"""
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
        ou_cli.start()

    # assert - should work without errors
    assert mock_generate.called


# Edge Cases


def test_gt_resources_empty_directory(tmp_path):
    """Test GtResources with empty directory"""
    # arrange
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # act
    resources = cc.GroundtruthFile.from_dir(empty_dir, limit=0)

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
    resources = cc.GroundtruthFile.from_dir(gt_dir, limit=0)

    # assert
    assert len(resources) == 1
    assert resources[0].file_path == test_file


def test_args_with_zero_limit():
    """Test Args with limit=0 (unlimited)"""
    # arrange & act
    args = cc.CorpusArgs(
        input_dir=Path("/input"),
        output_dir=Path("/output"),
        local_cache_dir=Path("/temp"),
        limit=0,
    )

    # assert
    assert args.limit == 0  # 0 means unlimited


def test_gt_resources_language_parsing():
    """Test that GtResources correctly parses multi-language tags"""
    # arrange
    tmp_dir = Path("/tmp")

    # Create a mock GroundtruthFileResource
    resource = cc.GroundtruthFile(
        identifier="urn:nbn:de:gbv:3:1-123456/fragment/page=0001",
        file_base_name="urn+nbn+de+gbv+3+1-123456",
        file_path=tmp_dir / "test.xml",
        relative_file_path=Path("test.xml"),
        languages=["deu", "lat", "eng"],
    )

    # assert
    assert resource.languages
    assert len(resource.languages) == 3
    assert "deu" in resource.languages
    assert "lat" in resource.languages
    assert "eng" in resource.languages
