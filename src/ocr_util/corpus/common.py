"""OCR corpus generation common code."""

import copy
import dataclasses
import os
import pathlib
import re
import shutil
import typing

import lxml.etree as ET


DEFAULT_FULLEXT_FILEGROUP = "FULLTEXT"
DEFAULT_IMAGE_FILEGROUP = "MAX"

LABEL_FILEGROUP_IMAGE = "GT-IMAGE"
KWARG_LABEL_FILEGROUP_FULLTEXT = "label_filegroup_ocr"

GT_TARGET_SUBDIR = "GT-PAGE"
GT_METS_FILEGROUP = "GT-FULLTEXT"
GT_METS_IMAGE = "GT-IMAGE"
DEFAULT_METS_FILE_NAME = "mets.xml"
INDENT = 4

EASY_URN_XML_PATTERN = r'^(urn[\+\-\w]+)-fp-(\w+).xml$'

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
    # oai_base_urls:str
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
        for (current_dir, _, files) in os.walk(gt_dir,topdown=False):
            for file in files:
                file_path: pathlib.Path = pathlib.Path(current_dir).joinpath(file)
                the_match: typing.Optional[typing.Match[str]] = re.match(EASY_URN_XML_PATTERN, file_path.name)
                if the_match is not None:
                    urn_enc = f"{the_match.group(1)}/fragment/page={the_match.group(2)}"
                    urn_dec: str = urn_enc.replace('+', ':').replace('x','X')
                    resources.append(GroundtruthFile(
                        identifier=urn_dec,
                        file_base_name=the_match.group(1),
                        file_path=file_path,
                        relative_file_path=file_path.relative_to(gt_dir),
                    ))
                    if 0 < limit <= len(resources):
                        break
                else:
                    print(f"file name {file_path.name} not matching pattern '{EASY_URN_XML_PATTERN}' will be ignored")
            break
        return sorted(resources, key=lambda r: r.file_path.name)


# @dataclasses.dataclass
# class MetsResource:
    # local_file_path: pathlib.Path


@dataclasses.dataclass
class CorpusInput:
    """Encapsulate input data required for corpus generation."""
    identifier_urn: str
    groundtruth_file: GroundtruthFile
    cached_media_mets_file: typing.Optional[pathlib.Path] = None

@dataclasses.dataclass
class MetsSection:
    original_dmdid: str
    phys_div: ET._Element
    file_image: ET._Element
    file_fulltext: ET._Element
    sm_link: ET._Element
    log_div: ET._Element
    calculated_dmdid: typing.Optional[str] = None
    dmd_sec: typing.Optional[ET._Element] = None

@dataclasses.dataclass
class CorpusGeneratorResult:
    """Encapsulate final Corpus generation result."""
    file_path: pathlib.Path
    n_pages: int



class MetsFile:
    """Represents a published METS file resource."""

    def __init__(self, file_path: pathlib.Path, **kwargs):
        self.file_path = file_path
        self.document = file_path  # trigger setter
        self.kwargs = kwargs
        self.label_fgroup_fulltext = self.kwargs.get('label_filegroup_ocr', DEFAULT_FULLEXT_FILEGROUP)
        self.label_fgroup_image = self.kwargs.get('label_filegroup_image', DEFAULT_IMAGE_FILEGROUP)

    @property
    def document(self) -> ET._ElementTree:
        """Get the METS XML document representing the corpus template."""
        return self.__document

    @document.setter
    def document(self, path_document: pathlib.Path) -> None:
        """Set the METS XML document and extract relevant sections."""
        self.__document = ET.parse(
            path_document,
            ET.XMLParser(remove_blank_text=True)
        )
        the_root = self.__document.getroot()
        # Extract namespaces directly using xpath; filter out empty/None prefixes
        raw_ns = typing.cast(typing.List[typing.Tuple[typing.Optional[str], str]], the_root.xpath('//namespace::*'))
        self.__nsmap = {prefix: uri for prefix, uri in raw_ns if prefix}
        logical_root = self.document.find('.//mets:structMap[@TYPE="LOGICAL"]/mets:div', self.nsmap)
        if logical_root is None:
            raise CorpusException(f"Missing logical structMap in METS file '{self.file_path}'")
        self.__logical_root = logical_root
        self.file_path = path_document

    @property
    def nsmap(self) -> dict:
        """Get the namespace mapping extracted from the METS document."""
        return self.__nsmap

    @property
    def logical_root(self) -> ET._Element:
        """Get the logical structMap root element."""
        return self.__logical_root


    def find_dmd_id(self) -> str:
        """Find DMDID in logical structMap of given published METS file."""
        # logical_root = self.document.find('.//mets:structMap[@TYPE="LOGICAL"]', self.nsmap)
        # if logical_root is None:
        #     raise CorpusException(f"No logical structMap in METS file {self.document.docinfo.URL} found")
        assert self.logical_root is not None, f"No logical structMap in METS file {self.file_path}"
        logical_children = self.logical_root.findall('.//mets:div', self.nsmap)
        for element in logical_children:
            the_type = element.get("TYPE", None)
            dmd_id = element.get("DMDID", None)
            if dmd_id is not None and the_type in METS_MEDIA_TYPES:
                return dmd_id
        raise CorpusException(f"no DMD_ID in METS file {self.file_path} found")


class MetsCorpusFile(MetsFile):
    """Represents a METS file resource for corpus generation."""
    def __init__(self, file_path: pathlib.Path, **kwargs):
        super().__init__(file_path, **kwargs)
        self._build_internal_structures()
        # the_root = self.document.getroot()
        # prev_phys_root = the_root.findall('.//mets:div[@ID="physroot"]', self.nsmap)
        # if len(prev_phys_root) == 0:
        #     physroot = ET.Element(f'{{{self.nsmap["mets"]}}}div', attrib={'ID': 'physroot', 'TYPE': 'physSequence'})
        #     the_root.append(
        #         ET.Element(f'{{{self.nsmap["mets"]}}}structMap', attrib={'TYPE': 'PHYSICAL'}, children=[physroot])
        #     )
        # self.physical_root = self.document.find('.//mets:div[@ID="physroot"]', self.nsmap)
        # self.link_root = self.document.find('.//mets:structLink', self.nsmap)

    def _build_internal_structures(self):
        # label_fgroup_fulltext = self.kwargs.get('label_filegroup_ocr', DEFAULT_FULLEXT_FILEGROUP)
        # label_fgroup_image = self.kwargs.get('label_filegroup_image', DEFAULT_IMAGE_FILEGROUP)

        xpr_fulltext = f'.//mets:fileGrp[@USE="{self.label_fgroup_fulltext}"]'
        self.file_group_fulltext = self._get_or_create('fileGrp', xpr_fulltext, {'USE': self.label_fgroup_fulltext})

        xpr_image = f'.//mets:fileGrp[@USE="{self.label_fgroup_image}"]'
        self.file_group_image = self._get_or_create('fileGrp', xpr_image, {'USE': self.label_fgroup_image})
        # xpr_image = f'.//mets:fileGrp[@USE="{label_fgroup_image}"]'
        # self.file_group_image = self._get_or_create('fileGrp', xpr_image, {'USE': label_fgroup_image})

        # self.link_root = self._get_or_create('structLink', './/mets:structLink', {})
        # self.physical_root = self._get_or_create('div', './/mets:div[@ID="physroot"]', {'ID': 'physroot', 'TYPE': 'physSequence'})
        # logical_root = self.document.find('.//mets:structMap[@TYPE="LOGICAL"]/mets:div', self.nsmap)
        # if logical_root is None:
        #     raise CorpusException(f"Missing logical structMap in METS file '{self.file_path}'")

    def _get_or_create(self, tag: str, xpr: str, attrib: dict) -> ET._Element:
        """Get or create an XML element with the specified tag and attributes."""
        the_root = self.document.getroot()
        element = the_root.find(xpr, self.nsmap)
        if element is None:
            element = ET.Element(f'{{{self.nsmap["mets"]}}}{tag}', attrib=attrib)
            the_root.append(element)
        return element

    def attach(self, extract: MetsSection) -> None:
        """Attach extracted METS sections to the corpus METS document
        but attach MODS only if not already present in the corpus METS file."""

        # self.physical_root.append(extract.phys_div)
        # self.link_root.append(extract.sm_link)
        self.file_group_image.append(extract.file_image)
        self.file_group_fulltext.append(extract.file_fulltext)
        self.logical_root.append(extract.log_div)
        corpus_root = self.document.getroot()
        xtrct_dmd_id = extract.calculated_dmdid
        # look for extracted dmd_id in corpus file
        if corpus_root.find(f'.//mets:dmdSec[@ID="{xtrct_dmd_id}"]', namespaces=self.nsmap) is None:
            # if descriptive metadata not yet present in corpus file, append dmd_sec from extract
            if xtrct_dmd_id is not None and extract.dmd_sec is not None:
                idx: int = len(self.document.findall('.//mets:dmdSec', namespaces=self.nsmap))
                self.document.getroot().insert(idx, extract.dmd_sec)
            # orig_dmd_sec = extract.dcorpus_root.find(f'.//mets:dmdSec[@ID="{xtrct_dmd_id}"]', self.nsmap)
            # orig_dmd_sec = extract.dmd_sec
            # orig_mods_root = orig_dmd_sec.find(f'.//mods:mods', self.nsmap)
            # if orig_mods_root is None:
            #     raise CorpusException(f"No MODS metadata in METS file {doc.base} found")
            # dmd_sec = copy.deepcopy(orig_dmd_sec) # create deep copy of dmd_sec to avoid modifying the original METS file
            # mods_root: ET._Element = dmd_sec.find('.//mods:mods', nsmap)
            # # Remove all children elements
            # for child in mods_root.getchildren():
            #     child.getparent().remove(child)

    def finalize(self):
        """Re-order attached gt-image containers"""
        # phys_divs_with_order_attrib: list[ET._Element] = self.physical_root.findall(
        #     './/mets:div[@ORDER]',
        #     self.nsmap
        # )
        # for i, elm in enumerate(phys_divs_with_order_attrib):
        #     elm.set('ORDER', str(i + 1))

        log_divs_with_order_attrib: list[ET._Element] = self.logical_root.findall(
            './/mets:div[@ORDER]',
            self.nsmap
        )
        for i, elm in enumerate(log_divs_with_order_attrib):
            elm.set('ORDER', str(i + 1))

    def write(self, out_dir: pathlib.Path) -> pathlib.Path:
        """Write the METS XML document to the specified file path."""
        encoding = self.__document.docinfo.encoding if self.__document.docinfo.encoding else 'UTF-8'
        ET.indent(self.__document.getroot(), space=(" " * INDENT))
        out_file: pathlib.Path = out_dir.joinpath(DEFAULT_METS_FILE_NAME)
        self.__document.write(
            out_file,
            xml_declaration=True,
            pretty_print=True,
            encoding=encoding
        )
        return out_file


class MetsResourceFile(MetsFile):
    """Represents a METS file resource for extraction."""
    def __init__(self, corpus_input: CorpusInput, out_dir: pathlib.Path, **kwargs):
        assert corpus_input.cached_media_mets_file is not None, f"METS file for {corpus_input.identifier_urn} missing"
        super().__init__(corpus_input.cached_media_mets_file, **kwargs)
        self.corpus_input = corpus_input
        self.out_dir = out_dir
        # self.physical_root = the_root.findall('.//mets:div[@ID="physroot"]', self.nsmap)[0]

    def extract(
            self,
            index: int,
            page_urn: str,
    ) -> MetsSection:
        """Grab requried information"""
        the_root = self.document.getroot()
        page_div = the_root.find(f'.//mets:div[@CONTENTIDS="{page_urn}"]', self.nsmap)
        assert page_div is not None, f"no page with CONTENTIDS='{page_urn}' found in {self.document.docinfo.URL}"
        # try:
        #     del page_div.attrib['ORDERLABEL']
        # except KeyError:
        #     pass
        # FILES
        file_pointers: list[ET._Element] = page_div.findall('.//mets:fptr', namespaces=self.nsmap)
        # Remove all child elements from actual page as parent
        for el in file_pointers:
            el_pa = el.getparent()
            if el_pa is not None:
                el_pa.remove(el)
        # now re-attach only selected file pointers
        for fp in file_pointers:
            the_file = the_root.find(f'.//mets:file[@ID="{fp.get("FILEID")}"]', self.nsmap)
            assert the_file is not None, f"no file with ID='{fp.get('FILEID')}' found in {self.document.docinfo.URL}"
            the_file_group = the_file.xpath('ancestor::mets:fileGrp/@USE', namespaces=self.nsmap)
            # the_parent = fp.getparent()
            # assert the_parent is not None
            assert len(the_file_group) == 1, f"file with ID='{fp.get('FILEID')}' invalid parent fileGrp"
            the_group = the_file_group[0]
            if the_group not in {self.label_fgroup_fulltext, self.label_fgroup_image}:
                continue

            # here we go
            new_id = f'{the_group}-{(index + 1):04d}'
            if the_group == self.label_fgroup_image:
                fp.set('FILEID', new_id)
                file_image = fp
                page_div.append(fp)
            elif the_group == self.label_fgroup_fulltext:
                file_ptr_fulltext = ET.Element(f'{{{self.nsmap["mets"]}}}fptr',
                                                    attrib={'FILEID': new_id}
                                                    )
                page_div.append(file_ptr_fulltext)
                file_fulltext = ET.Element(f'{{{self.nsmap["mets"]}}}file',
                                                        attrib={
                        'ID': new_id,
                        'MIMETYPE': 'application/vnd.prima.page+xml',
                    }
                )
                assert self.corpus_input.groundtruth_file is not None, f"Ground truth file for {self.corpus_input.identifier_urn} missing"
                file_fulltext.append(
                    ET.Element(f'{{{self.nsmap["mets"]}}}FLocat',
                        attrib={
                            f'{{{self.nsmap["xlink"]}}}href': str(self.corpus_input.groundtruth_file.file_path.relative_to(self.out_dir)),
                            'LOCTYPE': "OTHER",
                            'OTHERLOCTYPE': "FILE",
                        }
                    )
                )

        # files = [
        #     the_root.find(f'.//mets:file[@ID="{fptr.get("FILEID")}"]', self.nsmap)
        #     for fptr
        #     in file_pointers
        # ]
        # file_image = next(file for file in files if file.getparent().get('USE') == 'MAX')
        # file_ptr_image: ET._Element = next(fptr for fptr in file_pointers if fptr.get('FILEID') == file_image.get('ID'))
        # page_div.append(file_ptr_image)

        # file_fulltext_id: str = f'{GT_METS_FILEGROUP}-{(index + 1)}'
        # file_ptr_fulltext = ET.Element(f'{{{self.nsmap["mets"]}}}fptr',
        #                                             attrib={'FILEID': file_fulltext_id}
        #                                             )
        # page_div.append(file_ptr_fulltext)
        # file_fulltext = ET.Element(f'{{{self.nsmap["mets"]}}}file',
        #                                         attrib={
        #         'ID': file_fulltext_id,
        #         'MIMETYPE': 'application/vnd.prima.page+xml',
        #     }
        # )
        # file_fulltext.append(
        #     ET.Element(f'{{{self.nsmap["mets"]}}}FLocat',
        #         attrib={
        #             f'{{{self.nsmap["xlink"]}}}href': str(gt_file_path.relative_to(self.__out_dir)),
        #             'LOCTYPE': "OTHER",
        #             'OTHERLOCTYPE': "FILE",
        #         }
        #     )
        # )
        # LINK
        phys_id = page_div.get('ID')
        assert phys_id is not None, f"page div with CONTENTIDS='{page_urn}' missing ID attribute in {self.document.docinfo.URL}"
        sm_link = the_root.find(f'.//mets:smLink[@xlink:to="{phys_id}"]', self.nsmap)
        assert sm_link is not None, f"smLink for page div with ID='{phys_id}' not found in {self.document.docinfo.URL}"
        # LOG
        log_id = sm_link.get(f'{{{self.nsmap["xlink"]}}}from')
        assert log_id is not None, f"log ID for smLink with to='{phys_id}' not found in {self.document.docinfo.URL}"
        log_div = the_root.find(f'.//mets:div[@ID="{log_id}"]', self.nsmap)
        assert log_div is not None, f"log div with ID='{log_id}' not found in {self.document.docinfo.URL}"
        dmd_id = self.find_dmd_id()
        log_div.set('DMDID', dmd_id)
        #self.file_group_fulltext.set('DMDID', dmd_id)
        try:
            del log_div.attrib['LABEL']
        except KeyError:
            pass
        # # Remove all children elements
        for child in log_div.getchildren():
            child.getparent().remove(child)
        # log_div.append(file_ptr_image)

        # dmd_sec = None
        # if self.corpus_root.find(f'.//mets:dmdSec[@ID="{dmd_id}"]', namespaces=self.nsmap) is None:

        orig_dmd_sec = the_root.find(f'.//mets:dmdSec[@ID="{dmd_id}"]', self.nsmap)
        orig_mods_root = orig_dmd_sec.find(f'.//mods:mods', self.nsmap)
        # if orig_mods_root is None:
        #     raise CorpusException(f"No MODS metadata in METS file {doc.base} found")
        
        dmd_sec = copy.deepcopy(orig_dmd_sec) # create deep copy of dmd_sec to avoid modifying the original METS file
        mods_root: ET._Element = dmd_sec.find('.//mods:mods', self.nsmap)
        # Remove all children elements
        for child in mods_root.getchildren():
            child.getparent().remove(child)

        mods_root.extend(orig_mods_root.findall('mods:identifier', self.nsmap))
        identifer_elements: list[ET._Element] = orig_mods_root.findall('mods:identifier', self.nsmap)
        mods_root.extend(identifer_elements)
        title_elements: list[ET._Element] = orig_mods_root.findall('mods:titleInfo', self.nsmap)
        if len(title_elements) > 0:
            mods_root.extend(title_elements)
        language_elements: list[ET._Element] = orig_mods_root.findall('mods:language', self.nsmap)
        if len(language_elements) > 0:
            mods_root.extend(language_elements)
        genre_elements: list[ET._Element] = orig_mods_root.findall('mods:genre', self.nsmap)
        if len(genre_elements) > 0:
            mods_root.extend(genre_elements)
        publication_info = orig_mods_root.find('mods:originInfo[@eventType="publication"]', self.nsmap)
        if publication_info is not None:
            mods_root.append(publication_info)

        return MetsSection(
            phys_div=page_div,
            file_image=file_image,
            file_fulltext=file_fulltext,
            sm_link=sm_link,
            log_div=log_div,
            dmd_sec=dmd_sec,
            original_dmdid=dmd_id,
        )










