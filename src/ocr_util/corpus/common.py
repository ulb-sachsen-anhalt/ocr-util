"""Shared data models and METS processing utilities for corpus generation.

This module provides the building blocks used by :mod:`ocr_util.corpus.generate_corpus`
to assemble a METS-based ground truth corpus:

* **Data classes** – :class:`CorpusArgs`, :class:`GroundtruthFile`,
  :class:`CorpusPageInput`, :class:`MetsModsSection`,
  :class:`CorpusGeneratorResult`.
* **METS file abstractions**

  * :class:`MetsFile` – base class for reading and querying a METS XML document.
  * :class:`MetsCorpusFile` – extends :class:`MetsFile` to represent the
    growing output corpus METS document; exposes :meth:`~MetsCorpusFile.attach`
    and :meth:`~MetsCorpusFile.write`.
  * :class:`MetsResourceFile` – extends :class:`MetsFile` to extract page-level
    METS/MODS sections from a published source METS document.

Constants
---------
EASY_URN_XML_PATTERN
    Regular expression that identifies PAGE-XML files carrying a URN-based
    page identifier in their file name
    (e.g. ``urn+nbn+de+gbv+3+5-12345-fp-00000001.xml``).
METS_MEDIA_TYPES
    Set of METS logical structure ``TYPE`` attribute values that are expected
    to carry a ``DMDID`` reference to the descriptive metadata section.
"""

import copy
import dataclasses
import hashlib
import logging
import os
import pathlib
import re
import shutil
import typing

import lxml.etree as ET

logger = logging.getLogger(__name__)


DEFAULT_FULLEXT_FILEGROUP = "FULLTEXT"
DEFAULT_IMAGE_FILEGROUP = "MAX"

LABEL_FILEGROUP_IMAGE = "GT-IMAGE"
KWARG_LABEL_FILEGROUP_FULLTEXT = "label_filegroup_ocr"

GT_TARGET_SUBDIR = "GT-PAGE"
GT_METS_FILEGROUP_FULLTEXT = "GT-FULLTEXT"
GT_METS_FILEGROUP_IMAGE = "GT-IMAGE"
DEFAULT_METS_FILE_NAME = "mets.xml"
INDENT = 4

EASY_URN_XML_PATTERN = r"^(urn[\+\-\w]+)-fp-(\w+).xml$"

METS_MEDIA_TYPES = {
    "monograph",
    "volume",
    "issue",
    "additional",
}


class CorpusException(Exception):
    """Base exception for corpus-related errors."""


@dataclasses.dataclass
class CorpusArgs:
    """Arguments for corpus generation."""

    input_dir: pathlib.Path
    output_dir: pathlib.Path
    local_cache_dir: pathlib.Path
    limit: int = 0
    clear_cache: bool = False
    corpus_label: str = "Ground Truth Corpus"


@dataclasses.dataclass
class GroundtruthFile:
    """Represents local ground truth file resource."""

    identifier: str
    file_base_name: str
    file_path: pathlib.Path
    relative_file_path: pathlib.Path
    languages: typing.Optional[typing.List[str]] = None

    @classmethod
    def from_dir_copy(cls, in_dir: pathlib.Path, out_dir: pathlib.Path, limit: int = 0) -> typing.List:
        """Scan directory for ground truth files, copy to output directory, and return list of resources."""
        gt_resources: typing.List[GroundtruthFile] = cls.from_dir(in_dir, limit)
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
    def from_dir(cls, gt_dir: pathlib.Path, limit: int = 0) -> typing.List:
        """Scan directory for ground truth files and return list of resources."""
        resources: typing.List[GroundtruthFile] = []
        current_dir: str
        # child_dirs: typing.List[str]
        files: typing.List[str]
        for current_dir, _, files in os.walk(gt_dir, topdown=False):
            for file in files:
                file_path: pathlib.Path = pathlib.Path(current_dir).joinpath(file)
                the_match: typing.Optional[typing.Match[str]] = re.match(EASY_URN_XML_PATTERN, file_path.name)
                if the_match is not None:
                    urn_enc = f"{the_match.group(1)}/fragment/page={the_match.group(2)}"
                    urn_dec: str = urn_enc.replace("+", ":").replace("x", "X")
                    gt_file = GroundtruthFile(
                        identifier=urn_dec,
                        file_base_name=the_match.group(1),
                        file_path=file_path,
                        relative_file_path=file_path.relative_to(gt_dir),
                    )
                    resources.append(gt_file)
                    if 0 < limit <= len(resources):
                        break
                else:
                    logger.warning(
                        "File name '%s' does not match pattern '%s'; ignoring file",
                        file_path.name,
                        EASY_URN_XML_PATTERN,
                    )
            break
        return sorted(resources, key=lambda r: r.file_path.name)


@dataclasses.dataclass
class MetsModsSection:
    """Represents extracted METS/MODS sections for a given page."""

    phys_div: ET._Element
    file_image: ET._Element
    file_fulltext: ET._Element
    sm_link: ET._Element
    log_div: ET._Element
    dmd_sec: typing.Optional[ET._Element] = None
    calculated_identifier_hash: typing.Optional[str] = None


class MetsFile:
    """Represents a published METS file resource."""

    def __init__(self, file_path: pathlib.Path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs
        self.label_fgroup_fulltext = self.kwargs.get("label_filegroup_ocr", DEFAULT_FULLEXT_FILEGROUP)
        self.label_fgroup_image = self.kwargs.get("label_filegroup_image", DEFAULT_IMAGE_FILEGROUP)
        self.logical_root: typing.Optional[ET._Element] = None
        self.physical_root: typing.Optional[ET._Element] = None
        self.document = file_path  # trigger setter

    @property
    def document(self) -> ET._ElementTree:
        """Get the METS XML document representing the corpus template."""
        return self.__document

    @document.setter
    def document(self, path_document: pathlib.Path) -> None:
        """Set the METS XML document and extract relevant sections."""
        self.__document = ET.parse(path_document, ET.XMLParser(remove_blank_text=True))
        the_root = self.__document.getroot()
        # Extract namespaces directly using xpath; filter out empty/None prefixes
        raw_ns = typing.cast(typing.List[typing.Tuple[typing.Optional[str], str]], the_root.xpath("//namespace::*"))
        self.__nsmap = {prefix: uri for prefix, uri in raw_ns if prefix}
        logical_root = self.document.find('.//mets:structMap[@TYPE="LOGICAL"]', self.nsmap)
        if logical_root is None:
            raise CorpusException(f"Missing logical structMap in METS file '{self.file_path}'")
        self.logical_root = logical_root
        physical_root = self.document.find('.//mets:structMap[@TYPE="PHYSICAL"]/mets:div', self.nsmap)
        if physical_root is not None:
            self.physical_root = physical_root
        self.file_path = path_document

    @property
    def nsmap(self) -> dict:
        """Get the namespace mapping extracted from the METS document."""
        return self.__nsmap

    def find_dmd_id(self) -> str:
        """Find DMDID in logical structMap of given published METS file."""

        assert self.logical_root is not None, f"No logical structMap in METS file {self.file_path}"
        logical_children = self.logical_root.findall(".//mets:div", self.nsmap)
        for element in logical_children:
            the_type = element.get("TYPE", None)
            dmd_id = element.get("DMDID", None)
            if dmd_id is not None and the_type in METS_MEDIA_TYPES:
                return dmd_id
        raise CorpusException(f"no DMD_ID in METS file {self.file_path} found")


class MetsCorpusFile(MetsFile):
    """Extends and modifies base MetsFile class to represent a METS file resource
    for a given Corpus Page with additional carries required information."""

    def __init__(self, file_path: pathlib.Path, **kwargs):
        super().__init__(file_path, **kwargs)
        self.label_fgroup_fulltext = self.kwargs.get("corpus_filegroup_ocr", GT_METS_FILEGROUP_FULLTEXT)
        self.label_fgroup_image = self.kwargs.get("corpus_filegroup_image", GT_METS_FILEGROUP_IMAGE)
        self.corpus_file_name = self.kwargs.get("corpus_file_name", DEFAULT_METS_FILE_NAME)
        # for the corpus METS file, go a little bit further down the logical line
        assert self.logical_root is not None, f"Missing logical structMap in METS file '{self.file_path}'"
        self.logical_root = self.logical_root.find("mets:div", self.nsmap)
        assert self.logical_root is not None, f"Missing logical structMap in METS file '{self.file_path}'"
        corpus_label = self.kwargs.get("corpus_label")
        if corpus_label:
            self.logical_root.set("LABEL", str(corpus_label))
        self._prepare_filegroups()
        the_root = self.document.getroot()
        prev_phys_root = the_root.findall('.//mets:div[@ID="physroot"]', self.nsmap)
        if len(prev_phys_root) == 0:
            physroot = ET.Element(f'{{{self.nsmap["mets"]}}}div', attrib={"ID": "physroot", "TYPE": "physSequence"})
            assert self.physical_root is not None, f"Missing physical structMap in METS file '{self.file_path}'"
            self.physical_root.append(physroot)
        self.link_root = self.document.find(".//mets:structLink", self.nsmap)

    def _prepare_filegroups(self):
        xpr_fulltext = f'mets:fileGrp[@USE="{self.label_fgroup_fulltext}"]'
        self.file_group_fulltext = self._get_or_create_filegroup(
            "fileGrp", xpr_fulltext, {"USE": self.label_fgroup_fulltext}
        )
        xpr_image = f'mets:fileGrp[@USE="{self.label_fgroup_image}"]'
        self.file_group_image = self._get_or_create_filegroup("fileGrp", xpr_image, {"USE": self.label_fgroup_image})

    def _get_or_create_filegroup(self, tag: str, xpr: str, attrib: dict) -> ET._Element:
        """Get or create an XML element with the specified tag and attributes."""
        filesec_root = self.document.getroot().find("mets:fileSec", self.nsmap)
        if filesec_root is None:
            filesec_root = ET.Element(f'{{{self.nsmap["mets"]}}}fileSec')
            self.document.getroot().append(filesec_root)
        file_group = filesec_root.find(xpr, self.nsmap)
        if file_group is None:
            file_group = ET.Element(f'{{{self.nsmap["mets"]}}}{tag}', attrib=attrib)
            filesec_root.append(file_group)
        return file_group

    def attach(self, extract: MetsModsSection) -> None:
        """Attach extracted METS sections to the corpus METS document
        but attach MODS only if not already present in the corpus METS file."""

        self.file_group_image.append(extract.file_image)
        self.file_group_fulltext.append(extract.file_fulltext)
        assert self.physical_root is not None, f"Missing physical structMap in METS file '{self.file_path}'"
        self.physical_root.append(extract.phys_div)
        assert self.link_root is not None, f"Missing structLink in METS file '{self.file_path}'"
        self.link_root.append(extract.sm_link)
        assert self.logical_root is not None, f"Missing logical structMap in METS file '{self.file_path}'"
        self.logical_root.append(extract.log_div)
        corpus_root = self.document.getroot()
        assert extract.dmd_sec is not None, "DMD section missing in section for attachment"
        xtrct_dmd_id = extract.dmd_sec.get("ID")
        # look for extracted dmd_id in corpus file
        if corpus_root.find(f'.//mets:dmdSec[@ID="{xtrct_dmd_id}"]', namespaces=self.nsmap) is None:
            # if descriptive metadata not yet present in corpus file, append dmd_sec from extract
            logger.info(
                "Attach new DMD section with ID='%s' to corpus METS file",
                xtrct_dmd_id,
            )
            if xtrct_dmd_id is not None and extract.dmd_sec is not None:
                idx: int = len(self.document.findall(".//mets:dmdSec", namespaces=self.nsmap))
                self.document.getroot().insert(idx, extract.dmd_sec)
        else:
            logger.info(
                "DMD section with ID='%s' already present in corpus METS file",
                xtrct_dmd_id,
            )

    def finalize(self):
        """Re-order attached gt-image containers"""
        the_root = self.document.getroot()
        pages_with_order: list[ET._Element] = the_root.findall(
            './/mets:structMap[@TYPE="PHYSICAL"]//mets:div[@ORDER]', self.nsmap
        )
        for i, elm in enumerate(pages_with_order, 1):
            elm.set("ORDER", f"{i}")
        assert self.logical_root is not None, f"Missing logical structMap in METS file '{self.file_path}'"
        log_divs_with_order: list[ET._Element] = self.logical_root.findall(".//mets:div[@ORDER]", self.nsmap)
        for i, elm in enumerate(log_divs_with_order, 1):
            elm.set("ORDER", f"{i}")

    def write(self, out_dir: pathlib.Path) -> pathlib.Path:
        """Write the METS XML document to the specified file path."""
        encoding = self.document.docinfo.encoding if self.document.docinfo.encoding else "UTF-8"
        ET.indent(self.document.getroot(), space=(" " * INDENT))
        out_file: pathlib.Path = out_dir.joinpath(self.corpus_file_name)
        self.document.write(out_file, xml_declaration=True, pretty_print=True, encoding=encoding)
        return out_file


class MetsResourceFile(MetsFile):
    """Represents a METS file resource for extraction."""

    def __init__(self, path_mets_file, idx: int, **kwargs):
        super().__init__(path_mets_file, **kwargs)
        self.mdsec_identifier: typing.Optional[str] = None
        self.idx = idx

    @property
    def identifier_hash(self) -> str:
        """Calculate and return a 8-character hash based on the METS/MODS resource section identifier."""
        if self.mdsec_identifier is None:
            raise CorpusException(f"Identifier for METS resource file {self.file_path} not set")
        return hashlib.sha256(self.mdsec_identifier.encode()).hexdigest()[:8]

    def extract(
        self,
        page_urn: str,
        out_dir: pathlib.Path,
        gt_file_path: pathlib.Path,
    ) -> MetsModsSection:
        """Grab required information and resolve relationships starting from
        the page div with given CONTENTIDS in the METS file."""

        the_root = self.document.getroot()
        page_div = the_root.find(f'.//mets:div[@CONTENTIDS="{page_urn}"]', self.nsmap)
        assert page_div is not None, f"no page with CONTENTIDS='{page_urn}' found"
        file_image, file_fulltext = self._set_page_with_files(the_root, page_div, gt_file_path, out_dir)

        # now resolve relationships to get logical div and dmdSec for the page
        # PHYSICAL
        source_phys_id = page_div.get("ID")
        assert source_phys_id is not None, f"page div with CONTENTIDS='{page_urn}' missing ID attribute"
        # LINK
        sm_link = the_root.find(f'.//mets:smLink[@xlink:to="{source_phys_id}"]', self.nsmap)
        assert sm_link is not None, f"smLink for page div with ID='{source_phys_id}' not found"
        # LOGICAL
        log_id = sm_link.get(f'{{{self.nsmap["xlink"]}}}from')
        assert log_id is not None, f"log ID for smLink with to='{source_phys_id}' not found"
        log_div = the_root.find(f'.//mets:div[@ID="{log_id}"]', self.nsmap)
        assert log_div is not None, f"log div with ID='{log_id}' not found"
        for child in log_div.getchildren():
            child.getparent().remove(child)
        # fix current DMDID for identifier calculation
        source_dmd_id = self.find_dmd_id()
        source_dmd_sec = the_root.find(f'.//mets:dmdSec[@ID="{source_dmd_id}"]', self.nsmap)
        assert source_dmd_sec is not None, f"DMD section with ID='{source_dmd_id}' not found"
        copy_dmd_sec = self._build_dmd_section(source_dmd_sec)
        section = MetsModsSection(
            phys_div=page_div,
            file_image=file_image,
            file_fulltext=file_fulltext,
            sm_link=sm_link,
            log_div=log_div,
            dmd_sec=copy_dmd_sec,
            calculated_identifier_hash=self.identifier_hash,
        )
        self._reindex(section)
        return section

    def _set_page_with_files(
        self, the_root: ET._Element, page_div: ET._Element, gt_file_path: pathlib.Path, out_dir: pathlib.Path
    ) -> typing.Tuple[ET._Element, ET._Element]:
        """Re-Create page container with the same attributes as the source page but with modified children."""

        file_pointers: list[ET._Element] = page_div.findall("mets:fptr", namespaces=self.nsmap)
        # Remove all child elements from actual page
        for el in file_pointers:
            el_pa = el.getparent()
            if el_pa is not None:
                el_pa.remove(el)
        # now re-attach selected file pointers for FULLTEXT and MAX image
        file_image = None
        file_fulltext = None
        for fp in file_pointers:
            the_file = the_root.find(f'.//mets:file[@ID="{fp.get("FILEID")}"]', self.nsmap)
            assert the_file is not None, f"no file with ID='{fp.get('FILEID')}' found in {self.document.docinfo.URL}"
            the_file_group = the_file.xpath("ancestor::mets:fileGrp/@USE", namespaces=self.nsmap)
            assert len(the_file_group) == 1, f"file with ID='{fp.get('FILEID')}' invalid parent fileGrp"
            the_group = the_file_group[0]
            if the_group not in {self.label_fgroup_fulltext, self.label_fgroup_image}:
                continue

            # here we go
            new_id = f"{the_group}-{(self.idx):04d}"
            if the_group == self.label_fgroup_image:
                prev_img_fileid = fp.get("FILEID")
                file_image = the_root.find(f'.//mets:file[@ID="{prev_img_fileid}"]', self.nsmap)
                assert file_image is not None, f"no file with ID='{prev_img_fileid}' found"
                pre_file_id = file_image.get("ID")
                logger.debug(
                    "Re-assigning file ID '%s' to '%s' for page with CONTENTIDS='%s'",
                    pre_file_id,
                    new_id,
                    page_div.get("CONTENTIDS"),
                )
                file_image.set("ID", new_id)
                fp.set("FILEID", new_id)
                file_ptr_image = fp
                page_div.append(file_ptr_image)
            elif the_group == self.label_fgroup_fulltext:
                file_ptr_fulltext = ET.Element(f'{{{self.nsmap["mets"]}}}fptr', attrib={"FILEID": new_id})
                page_div.append(file_ptr_fulltext)
                file_fulltext = ET.Element(
                    f'{{{self.nsmap["mets"]}}}file', attrib={"ID": new_id, "MIMETYPE": "application/vnd.prima.page+xml"}
                )
                file_fulltext.append(
                    ET.Element(
                        f'{{{self.nsmap["mets"]}}}FLocat',
                        attrib={
                            f'{{{self.nsmap["xlink"]}}}href': str(gt_file_path.relative_to(out_dir)),
                            "LOCTYPE": "OTHER",
                            "OTHERLOCTYPE": "FILE",
                        },
                    )
                )
        page_urn = page_div.get("CONTENTIDS")
        assert file_image is not None, f"No image file pointer for page with CONTENTIDS='{page_urn}'"
        assert file_fulltext is not None, f"No fulltext file pointer for page with CONTENTIDS='{page_urn}'"
        return file_image, file_fulltext

    def _build_dmd_section(self, source_dmd_sec: ET._Element) -> ET._Element:

        source_mods_root = source_dmd_sec.find(f".//mods:mods", self.nsmap)
        assert source_mods_root is not None, f"no MODS root in DMD section with ID='{source_dmd_sec.get('ID')}' found"
        identifer_elements: list[ET._Element] = source_mods_root.findall("mods:identifier", self.nsmap)
        identifer_urn: typing.List[str] = [i.text for i in identifer_elements if i.get("type") == "urn"]
        if not identifer_urn:
            raise CorpusException(
                f"No identifier with type='urn' in MODS metadata of METS file {self.document.docinfo.URL} found"
            )
        self.mdsec_identifier = identifer_urn[0]
        copy_dmd_sec = copy.deepcopy(
            source_dmd_sec
        )  # create deep copy of dmd_sec to avoid modifying the original METS file
        copy_root = copy_dmd_sec.find(".//mods:mods", self.nsmap)
        assert copy_root is not None, f"no MODS root in DMD section with ID='{copy_dmd_sec.get('ID')}' found"
        # clear copy: remove all copy child elements
        for child in copy_root.getchildren():
            child.getparent().remove(child)

        copy_root.extend(identifer_elements)
        title_elements: list[ET._Element] = source_mods_root.findall("mods:titleInfo", self.nsmap)
        if len(title_elements) > 0:
            copy_root.extend(title_elements)
        else:
            title_host = source_mods_root.find("mods:relatedItem/mods:titleInfo", self.nsmap)
            if title_host is not None:
                copy_root.append(title_host)
        language_elements: list[ET._Element] = source_mods_root.findall("mods:language", self.nsmap)
        if len(language_elements) > 0:
            copy_root.extend(language_elements)
        genre_elements: list[ET._Element] = source_mods_root.findall("mods:genre", self.nsmap)
        if len(genre_elements) > 0:
            copy_root.extend(genre_elements)
        publication_info = source_mods_root.find('mods:originInfo[@eventType="publication"]', self.nsmap)
        if publication_info is not None:
            copy_root.append(publication_info)
        access_info = source_mods_root.find("mods:accessCondition", self.nsmap)
        if access_info is not None:
            copy_root.append(access_info)
        return copy_dmd_sec

    def _reindex(self, section: MetsModsSection) -> None:
        """Re-assign IDs for phys_div, sm_link and log_div in the given section."""
        new_phys_id = f"PHYS-{(self.idx):04d}"
        prev_phys_id = section.phys_div.get("ID")
        logger.debug("Re-assigning phys div ID '%s' to '%s'", prev_phys_id, new_phys_id)
        new_log_id = f"LOG-{(self.idx):04d}"
        prev_log_id = section.log_div.get("ID")
        logger.debug("Re-assigning log div ID '%s' to '%s'", prev_log_id, new_log_id)
        section.phys_div.set("ID", new_phys_id)
        section.sm_link.set(f'{{{self.nsmap["xlink"]}}}to', new_phys_id)
        section.sm_link.set(f'{{{self.nsmap["xlink"]}}}from', new_log_id)
        section.log_div.set("ID", new_log_id)
        assert section.dmd_sec is not None, "DMD section missing in section for re-indexing"
        prev_dmdid = section.dmd_sec.get("ID")
        new_dmdid = f"{prev_dmdid}#{section.calculated_identifier_hash}"
        section.log_div.set("DMDID", new_dmdid)
        section.dmd_sec.set("ID", new_dmdid)
        logger.debug("Re-assigning DMDID '%s' to '%s'", prev_dmdid, new_dmdid)


@dataclasses.dataclass
class CorpusPageInput:
    """Encapsulate input data required for corpus generation."""

    identifier_urn: str
    groundtruth_file: GroundtruthFile
    cached_media_mets_file: typing.Optional[pathlib.Path] = None
    metadata: typing.Optional[MetsResourceFile] = None

    def __init__(self, groundtruth_file: GroundtruthFile):
        self.groundtruth_file = groundtruth_file
        self.identifier_urn = groundtruth_file.identifier


@dataclasses.dataclass
class CorpusGeneratorResult:
    """Encapsulate final Corpus generation result."""

    file_path: pathlib.Path
    n_pages: int
