

import dataclasses
import os
import pathlib
import re
import shutil
import typing

import lxml.etree as ET


# from ocr_util.corpus.GtResources import GtResource

GT_TARGET_SUBDIR = "GT-PAGE"
GT_METS_FILEGROUP = "OCR-D-GT-FULLTEXT"


class CorpusException(Exception):
    """Base exception for corpus-related errors."""


@dataclasses.dataclass
class CorpusArgs:
    """Arguments for corpus generation."""
    input_dir: pathlib.Path
    output_dir: pathlib.Path
    local_cache_dir: pathlib.Path
    oai_base_url:str
    limit: int = 0
    clear_cache: bool = False
    corpus_label: str = "Ground Truth Corpus"


@dataclasses.dataclass
class MetsResource:
    identifier_urn: str
    file_path: pathlib.Path


@dataclasses.dataclass
class MetsGeneratorResource:
    gt: 'GroundtruthFileResource'
    mets: MetsResource


@dataclasses.dataclass
class MetsGeneratorResult:
    mets: MetsResource


@dataclasses.dataclass
class MetsModsInformation:
    phys_div: ET.Element
    file_image: ET.Element
    file_fulltext: ET.Element
    sm_link: ET.Element
    log_div: ET.Element
    dmd_sec: ET.Element | None

@dataclasses.dataclass
class GroundtruthFileResource:
    """Represents ground truth file resource with associated metadata."""
    identifier: str
    file_base_name: str
    file_path: pathlib.Path
    relative_file_path: pathlib.Path
    languages: typing.Optional[typing.List[str]] = None


EASY_URN_XML_PATTERN = r'^(urn[\+\-\w]+)-fp(\w+).xml$'

# __PATTERN_FILE_NAME: str = r'^((urn\+nbn\+de\+gbv\+\d+\+(?:\d-?)+-p\d+(?:-\d+)?)_((?:\w{3}\+?)+)(?:\.gt)?).xml$'
# __ALTERNATIVE_PATTERN_FILE_NAME = r'.*(?:((urn\+nbn\+de\+gbv\+\d+\+\d+-)+)((?:(?:\d+)-?)+)-fp-?(\d+\d+))(?:_(((?:\w{3}\+?)+))(?:\.gt)?)?.xml'

class GtResources:

    @classmethod
    def from_dir_copy(cls, in_dir: pathlib.Path, out_dir: pathlib.Path, limit: int = 0) -> typing.List[GroundtruthFileResource]:
        gt_resources: typing.List[GroundtruthFileResource] = cls.from_dir(in_dir, limit)
        for gt_resource in gt_resources:
            src_abs_path: pathlib.Path = gt_resource.file_path
            src_rel_path: pathlib.Path = gt_resource.file_path.relative_to(in_dir)
            dest_abs_path: pathlib.Path = out_dir.joinpath(src_rel_path).absolute()
            os.makedirs(dest_abs_path.parent, exist_ok=True)
            shutil.copy2(src_abs_path, dest_abs_path)
            gt_resource.file_path = dest_abs_path
            gt_resource.relative_file_path = src_rel_path
        return gt_resources

    @classmethod
    def from_dir(cls, gt_dir: pathlib.Path, limit: int = 0) -> typing.List[GroundtruthFileResource]:
        resources: typing.List[GroundtruthFileResource] = []
        current_dir: str
        # child_dirs: typing.List[str]
        files: typing.List[str]
        for (current_dir, _, files) in os.walk(gt_dir,topdown=False):
            for file in files:
                file_path: pathlib.Path = pathlib.Path(current_dir).joinpath(file)
                the_match: typing.Optional[typing.Match[str]] = re.match(EASY_URN_XML_PATTERN, file_path.name)
                if the_match is not None:
                    urn_enc = f"{the_match.group(1)}/fragment/page={the_match.group(2)}"
                    urn_dec: str = urn_enc.replace('+', ':').replace('x','X')
                    # urn_enc: str = the_match.group(2)
                    # urn_dec: str = urn_enc.replace('+', ':')
                    # langs_enc: str = the_match.group(3)
                    # langs_dec: list[str] = langs_enc.split('+')
                    resources.append(GroundtruthFileResource(
                        identifier=urn_dec,
                        file_base_name=the_match.group(1),
                        file_path=file_path,
                        relative_file_path=file_path.relative_to(gt_dir),
                    ))
                    if 0 < limit <= len(resources):
                        break
                else:
                    print(f"file {file_path.name} is damaged and will be ignored.")
                    # continue
                    # the_match = re.match(GtResources.__ALTERNATIVE_PATTERN_FILE_NAME, file_path.name)
                    # if(the_match is not None):
                    #     if(the_match.group(1).startswith("urn")):
                    #         urn_enc = f"{the_match.group(1)}{the_match.group(3)}/fragment/page={the_match.group(4)}"
                    #         urn_dec: str = urn_enc.replace('+', ':').replace('x','X')
                    #     else:
                    #         print(f"The xml file {file_path.name} is damaged and will be ignored.")
                    #         continue
                    #     resources.append(GtResource(
                    #         identifier=urn_dec,
                    #         file_base_name=file_path.name,
                    #         file_path=file_path,
                    #         relative_file_path=file_path.relative_to(gt_dir),
                    #         languages="" #is never used anyway and we don't have it in the name in this scheme
                    #     ))
                    #     if 0 < limit <= len(resources):
                    #         break
                    # else:
                    #     continue
            break
        return sorted(resources, key=lambda r: r.file_path.name)
