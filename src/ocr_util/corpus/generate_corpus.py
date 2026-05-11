
import argparse
import copy
import math
import os
import pathlib
import shutil
import typing

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import lxml.etree as ET

import ocr_util.corpus.common as cc
import ocr_util.corpus.load_metadata as lr

CPUS = os.cpu_count() or 1
NUM_THREADS: typing.Final[int] = math.ceil(CPUS * 0.85)
DEFAULT_TEMP_DIR = pathlib.Path.home().joinpath('.cache', 'odem_gt_2_mets')
DEFAULT_LIMIT = 0



def fetch_resources(cache_path:pathlib.Path, gt_files: list[cc.CorpusInput]) -> list[cc.CorpusInput]:
    """Fetch required METS metadata parallel for given inputs."""
    if not cache_path.exists():
        cache_path.mkdir()
    resolver: lr.RecordMetadataResolver = lr.RecordMetadataResolver()
    # Use ThreadPoolExecutor directly for parallel execution
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(resolver.fetch, an_input,
                                   cache_path=cache_path)
                   for an_input in gt_files]
        return [future.result() for future in futures]


class CorpusFile:
    """Ground Truth Corpus in METS format."""

    def __init__(self, corpus_args: cc.CorpusArgs,
                 generator_resources: list[cc.CorpusInput],
                 **kwargs) -> None:
        self.__cargs = corpus_args
        self.__out_dir: Final[Path] = self.__cargs.output_dir
        self.__generator_resources: Final[list[cc.CorpusInput]] = generator_resources
        # Template file is located in the same directory as this module
        template_path = Path(__file__).parent / 'template.corpus.xml'
        self.corpus_template = template_path
        self.kwargs = kwargs
        self.corpus_label = self.__cargs.corpus_label
        self.file = cc.MetsCorpusFile(template_path, corpus_label=self.corpus_label, **kwargs)

    # @corpus_template.setter
    # def corpus_template(self, path_template: Path) -> None:
    #     self.__mets_xml_document = ET.parse(
    #         path_template,
    #         ET.XMLParser(remove_blank_text=True)
    #     )
    #     self.corpus_root = self.__mets_xml_document.getroot()
    #     # Extract namespaces directly using xpath
    #     self.corpus_nsmap = {
    #         ns[0]: ns[1] for ns in self.corpus_root.xpath('//namespace::*') if ns[0]
    #     }
    #     # GET SECTION ROOT NODES FROM DOC
    #     self.__physroot = self.corpus_root.findall('.//mets:div[@ID="physroot"]',
    #                                                             self.corpus_nsmap)[0]
    #     self.__logroot = self.corpus_root.findall('.//mets:div[@ID="logroot"]',
    #                                                            self.corpus_nsmap)[0]
    #     self.__struct_link_root = self.corpus_root.findall(
    #         './/mets:structLink',self.corpus_nsmap)[0]
    #     self.__file_group_image = self.corpus_root.findall(
    #         './/mets:fileGrp[@USE="OCR-D-IMG"]',
    #         self.corpus_nsmap )[0]
    #     self.__file_group_fulltext = self.corpus_root.findall(
    #         './/mets:fileGrp[@USE="FULLTEXT"]',
    #         self.corpus_nsmap )[0]
    #     self.__file_group_fulltext.set('USE', cc.GT_METS_FILEGROUP)

    def build(self) -> cc.CorpusGeneratorResult:
        """Build the corpus by extracting data from original METS files and write new METS file."""

        total: int = len(self.__generator_resources)
        assert self.file is not None, "Corpus template must be set before start building corpus"
        # nsmap = self.file.nsmap
        for i, corpus_input in enumerate(self.__generator_resources):
            print(f'Process {i + 1} of {total} mets files - {corpus_input.identifier_urn}')
            try:
                # extract: cc.MetsSection = self.extract(
                #     index=i,
                #     page_urn=generator_resource.mets.identifier_urn,
                #     gt_file_path=generator_resource.gt.file_path,
                #     metadata_path=generator_resource.mets.local_file_path,
                # )
                assert corpus_input.cached_media_mets_file is not None, f"METS file for {corpus_input.identifier_urn} not found"
                mets_res = cc.MetsResourceFile(corpus_input,
                                               out_dir=self.__out_dir, **self.kwargs)
                the_section = mets_res.extract(index=i, page_urn=corpus_input.identifier_urn)
                self.file.attach(the_section)
                # self.corpus_file.physroot.append(extract.phys_div)
                # self.corpus_file.file_group_image.append(extract.file_image)
                # self.corpus_file.file_group_fulltext.append(extract.file_fulltext)
                # self.corpus_file.link_root.append(extract.sm_link)
                # self.corpus_file.logroot.append(extract.log_div)

                # calculated_dmd_id: str = extract.calculated_dmdid
                # if calculated_dmd_id is not None:
                #     idx: int = len(self.file.document.findall('.//mets:dmdSec', namespaces=nsmap))
                #     self.file.document.getroot().insert(idx, extract.dmd_sec)
            except Exception as exc:
                print(f"Error processing {corpus_input.cached_media_mets_file}: {exc.args[0]} - skip file")

        # phys_divs_with_order_attrib: list[ET._Element] = self.file.__physroot.findall(
        #     './/mets:div[@ORDER]',
        #     self.file.nsmap
        # )
        # for i, elm in enumerate(phys_divs_with_order_attrib):
        #     elm.set('ORDER', str(i + 1))

        # log_divs_with_order_attrib: list[ET._Element] = self.file.__logroot.findall(
        #     './/mets:div[@ORDER]',
        #     self.file.nsmap
        # )
        # for i, elm in enumerate(log_divs_with_order_attrib):
        #     elm.set('ORDER', str(i + 1))
        # ET.indent(self.corpus_root, space=(" " * INDENT))
        # out_file: Path = self.__out_dir.joinpath(DEFAULT_METS_FILE_NAME)
        # self.__mets_xml_document.write(
        #     out_file,
        #     xml_declaration=True,
        #     pretty_print=True,
        #     encoding=self.__mets_xml_document.docinfo.encoding
        # )

        # restore internal order
        self.file.finalize()

        out_path = self.file.write(self.__out_dir)
        return cc.CorpusGeneratorResult(
                file_path=out_path,
                n_pages=len(self.__generator_resources)
            )

    # def extract(
    #         self,
    #         index: int,
    #         page_urn: str,
    #         gt_file_path: Path,
    #         metadata_path: Path,
    # ) -> cc.MetsSection:
    #     try:
    #         doc = ET.parse(metadata_path, ET.XMLParser(remove_blank_text=True))
    #     except Exception as exc:
    #         print(f"XML file {gt_file_path} cant' be used: {exc.args[0]}")
    #         return None
    #     doc_root = doc.getroot()
    #     # Extract namespaces directly using xpath
    #     nsmap: dict[str, str] = {
    #         ns[0]: ns[1] for ns in doc_root.xpath('//namespace::*') if ns[0]
    #     }
    #     struct_page_div: ET._Element = doc_root.find(f'.//mets:div[@CONTENTIDS="{page_urn}"]', nsmap)
    #     try:
    #         del struct_page_div.attrib['ORDERLABEL']
    #     except KeyError:
    #         pass
    #     # FILES
    #     file_pointers: list[ET._Element] = struct_page_div.findall('.//mets:fptr', namespaces=nsmap)
    #     # Remove elements from their parent
    #     for el in file_pointers:
    #         el_pa = el.getparent()
    #         if el_pa is not None:
    #             el_pa.remove(el)
    #     files = [
    #         doc_root.find(f'.//mets:file[@ID="{fptr.get("FILEID")}"]', nsmap)
    #         for fptr
    #         in file_pointers
    #     ]
    #     file_image: ET._Element = next(file for file in files if file.getparent().get('USE') == 'MAX')
    #     file_ptr_image: ET._Element = next(fptr for fptr in file_pointers if fptr.get('FILEID') == file_image.get('ID'))
    #     struct_page_div.append(file_ptr_image)
    #     file_fulltext_id: str = f'{cc.GT_METS_FILEGROUP}-{(index + 1)}'
    #     file_ptr_fulltext = ET.Element(f'{{{nsmap["mets"]}}}fptr',
    #                                                 attrib={'FILEID': file_fulltext_id}
    #                                                 )
    #     struct_page_div.append(file_ptr_fulltext)
    #     file_fulltext = ET.Element(f'{{{nsmap["mets"]}}}file',
    #                                             attrib={
    #             'ID': file_fulltext_id,
    #             'MIMETYPE': 'application/vnd.prima.page+xml',
    #         }
    #     )
    #     file_fulltext.append(
    #         ET.Element(f'{{{nsmap["mets"]}}}FLocat',
    #             attrib={
    #                 f'{{{nsmap["xlink"]}}}href': str(gt_file_path.relative_to(self.__out_dir)),
    #                 'LOCTYPE': "OTHER",
    #                 'OTHERLOCTYPE': "FILE",
    #             }
    #         )
    #     )
    #     # LINK
    #     phys_id: str = struct_page_div.get('ID')
    #     sm_link: ET._Element = doc_root.find(f'.//mets:smLink[@xlink:to="{phys_id}"]', nsmap)
    #     # LOG
    #     log_id: str = sm_link.get(f'{{{nsmap["xlink"]}}}from')
    #     log_div: ET._Element = doc_root.find(f'.//mets:div[@ID="{log_id}"]', nsmap)
    #     dmd_id: str = find_dmd_id(doc_root, nsmap)
    #     log_div.set('DMDID', dmd_id)
    #     self.__file_group_fulltext.set('DMDID', dmd_id)
    #     try:
    #         del log_div.attrib['LABEL']
    #     except KeyError:
    #         pass
    #     # Remove all children elements
    #     for child in log_div.getchildren():
    #         child.getparent().remove(child)
    #     # log_div.append(file_ptr_image)

    #     dmd_sec = None

    #     if self.corpus_root.find(f'.//mets:dmdSec[@ID="{dmd_id}"]', namespaces=self.corpus_nsmap) is None:
    #         orig_dmd_sec = doc_root.find(f'.//mets:dmdSec[@ID="{dmd_id}"]', nsmap)
    #         orig_mods_root = orig_dmd_sec.find(f'.//mods:mods', nsmap)
    #         if orig_mods_root is None:
    #             raise cc.CorpusException(f"No MODS metadata in METS file {doc.base} found")
    #         dmd_sec = copy.deepcopy(orig_dmd_sec) # create deep copy of dmd_sec to avoid modifying the original METS file
    #         mods_root: ET._Element = dmd_sec.find('.//mods:mods', nsmap)
    #         # Remove all children elements
    #         for child in mods_root.getchildren():
    #             child.getparent().remove(child)

    #         mods_root.extend(orig_mods_root.findall('mods:identifier', nsmap))
    #         identifer_elements: list[ET._Element] = orig_mods_root.findall('mods:identifier', nsmap)
    #         mods_root.extend(identifer_elements)
    #         title_elements: list[ET._Element] = orig_mods_root.findall('mods:titleInfo', nsmap)
    #         if len(title_elements) > 0:
    #             mods_root.extend(title_elements)
    #         language_elements: list[ET._Element] = orig_mods_root.findall('mods:language', nsmap)
    #         if len(language_elements) > 0:
    #             mods_root.extend(language_elements)
    #         genre_elements: list[ET._Element] = orig_mods_root.findall('mods:genre', nsmap)
    #         if len(genre_elements) > 0:
    #             mods_root.extend(genre_elements)
    #         publication_info = orig_mods_root.find('mods:originInfo[@eventType="publication"]', nsmap)
    #         if publication_info is not None:
    #             mods_root.append(publication_info)

    #     return cc.MetsSection(
    #         phys_div=struct_page_div,
    #         file_image=file_image,
    #         file_fulltext=file_fulltext,
    #         sm_link=sm_link,
    #         log_div=log_div,
    #         dmd_sec=dmd_sec
    #     )


def generate(cargs: cc.CorpusArgs) -> cc.CorpusGeneratorResult:
    """Generate GT corpus in METS format from given corpus arguments."""
    if not cargs.input_dir.exists():
        raise cc.CorpusException(f"The input directory '{cargs.input_dir}' does not exist")
    if cargs.output_dir.exists():
        print(
            f"The output directory '{cargs.output_dir}' already exists. "
            "Refusing to overwrite existing data."
        )
    if cargs.local_cache_dir.exists() and cargs.clear_cache:
        print(f"Wipe existing cache directory {cargs.local_cache_dir}")
        shutil.rmtree(cargs.local_cache_dir)
    cargs.local_cache_dir.mkdir(parents=True, exist_ok=True)
    cargs.output_dir.mkdir(parents=True, exist_ok=True)
    gt_files: list[cc.GroundtruthFile] = cc.GroundtruthFile.from_dir_copy(
        in_dir=cargs.input_dir,
        out_dir=cargs.output_dir.joinpath(cc.GT_TARGET_SUBDIR),
        limit=cargs.limit
    )
    corpus_inputs = [cc.CorpusInput(groundtruth_file=gt_file)
        for gt_file
        in gt_files
    ]
    local_cache_path = Path(f'{cargs.local_cache_dir}').joinpath('mets')
    corpus_input_with_resources = fetch_resources(local_cache_path, corpus_inputs)
    corpus_file = CorpusFile(cargs, corpus_input_with_resources)
    return corpus_file.build()


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the input directory")
    parser.add_argument("output", help="Path to the output directory")
    parser.add_argument(
        "-l",
        "--limit",
        help="Number of Files being processed, default = 0 (unlimited)",
        required=False,
        type=int,
        default=DEFAULT_LIMIT
    )
    parser.add_argument(
        "-t",
        "--temp-dir",
        help="Path to the temporary directory",
        required=False,
        default=DEFAULT_TEMP_DIR
    )
    args = parser.parse_args()
    corpus_args = cc.CorpusArgs(
        input_dir=Path(args.input).absolute(),
        output_dir=Path(args.output).absolute(),
        local_cache_dir=Path(args.temp_dir).absolute(),
        limit=int(args.limit),
        corpus_label="Ground Truth Corpus"
    )

    outcome = generate(corpus_args)
    print(f"Corpus generated at {outcome.file_path} with {outcome.n_pages} pages")
