
import copy
import os
import shutil
import typing

from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import lxml.etree as ET

# from ocr_util.corpus.GtResources import GtResources
# from ocr_util.corpus.resource_load import RecordMetadataResolver
# from ocr_util.corpus.MetsGenerator import MetsGenerator

import ocr_util.corpus.common as cc
import ocr_util.corpus.load_resources as lr
# from ocr_util.corpus.common import (
#     Args,
#     CorpusException,
#     GtResource,
#     MetsGeneratorResource,
#     MetsResource,
#     GT_TARGET_SUBDIR,
# )


class Gt2Mets:
    # __URL_URN_RESOLVER: typing.Final[str] = "https://nbn-resolving.org/process-urn-form"
    # __URL_OAI_PMH_DATA: typing.Final[str] = "https://opendata.uni-halle.de/oai/dd"
    # __URL_OAI_PMH_ID_PREFIX: typing.Final[str] = "oai:opendata.uni-halle.de"
    __CPUS: typing.Final[int] = os.cpu_count() if os.cpu_count() is not None else 1
    __NUM_TRHEADS: typing.Final[int] = int(__CPUS * 0.85)
    __DEFAULT_TEMP_DIR: typing.Final[str] = os.path.join(os.path.expanduser("~"), '.cache', 'odem_gt_2_mets')
    __DEFAULT_LIMIT: typing.Final[int] = 0

    # @staticmethod
    # def __parse_args() -> corpus_common.Args:
        # parser: ArgumentParser = ArgumentParser()
        # parser.add_argument("input", help="Path to the input directory")
        # parser.add_argument("output", help="Path to the output directory")
        # parser.add_argument(
        #     "-l",
        #     "--limit",
        #     help="Number of Files being processed, default = 0 (unlimited)",
        #     required=False,
        #     type=int,
        #     default=Gt2Mets.__DEFAULT_LIMIT
        # )
        # parser.add_argument(
        #     "-t",
        #     "--temp-dir",
        #     help="Path to the temporary directory",
        #     required=False,
        #     default=Gt2Mets.__DEFAULT_TEMP_DIR
        # )
        # args: Namespace = parser.parse_args()
        # return corpus_common.Args(
        #     input_dir=Path(args.input).absolute(),
        #     output_dir=Path(args.output).absolute(),
        #     temp_dir=Path(args.temp_dir).absolute(),
        #     limit=int(args.limit),
        #     corpus_label="Ground Truth Corpus"
        # )

    def __init__(self, args: typing.Optional[cc.CorpusArgs],
                 preserve_cache: bool = False) -> None:
        if args is None:
            args = Gt2Mets.__parse_args()
        if not args.input_dir.exists():
            raise cc.CorpusException(f"The input directory '{args.input_dir}' does not exist")
        if args.output_dir.exists():
            print(
                f"The output directory '{args.output_dir}' already exists. "
                "Refusing to overwrite existing data."
            )
        self.__args: Final[cc.CorpusArgs] = args
        # self.__preserve_cache: bool = preserve_cache

    def run(self) -> None:
        if self.__args.local_cache_dir.exists() and self.__args.clear_cache:
            print(f"Wipe existing cache directory {self.__args.local_cache_dir}")
            shutil.rmtree(self.__args.local_cache_dir)
        self.__args.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.__args.output_dir.mkdir(parents=True, exist_ok=True)
        gt_resources: list[cc.GroundtruthFileResource] = cc.GtResources.from_dir_copy(
            in_dir=self.__args.input_dir,
            out_dir=self.__args.output_dir.joinpath(cc.GT_TARGET_SUBDIR),
            limit=self.__args.limit
        )
        mets_resources: list[cc.MetsResource] = self.__obtain_mets_files(gt_resources)
        mets_generator_resources: list[cc.MetsGeneratorResource] = [
            cc.MetsGeneratorResource(gt=gt_resource, mets=mets_resources[i])
            for i, gt_resource
            in enumerate(gt_resources)
        ]
        mets_generator: MetsGenerator = MetsGenerator(
            self.__args.output_dir,
            mets_generator_resources,
            corpus_label=self.__args.corpus_label
        )
        mets_generator.run()

    def __obtain_mets_files(self, gt_resources: list[cc.GroundtruthFileResource]) -> list[cc.MetsResource]:
        mets_dir_path: Path = Path(f'{self.__args.local_cache_dir}').joinpath('mets')
        if not mets_dir_path.exists():
            mets_dir_path.mkdir()
        mets_file_obtainers: list[lr.RecordMetadataResolver] = [
            lr.RecordMetadataResolver(
                urn=gt_resource.identifier,
                local_file_path=mets_dir_path.joinpath(f'{gt_resource.file_base_name}.mets.xml'),
            )
            for gt_resource
            in gt_resources
        ]
        # Use ThreadPoolExecutor directly for parallel execution
        with ThreadPoolExecutor(max_workers=self.__NUM_TRHEADS) as executor:
            futures = [executor.submit(obtainer.run) for obtainer in mets_file_obtainers]
            return [future.result() for future in futures]


class MetsGenerator:

    @staticmethod
    def __find_dmd_id(element: ET.Element) -> str:
        parent: ET.Element = element.getparent()
        type_: str = parent.get("TYPE")
        dmd_id: str = parent.get("DMDID")
        print(f"{dmd_id}, {type_}")
        if dmd_id is not None and (type_ == 'monograph' or type_ == 'volume' or type_ == 'issue' or type_ == 'additional'):
            return dmd_id
        return MetsGenerator.__find_dmd_id(parent)

    __METS_FILE_NAME: Final[str] = "mets.xml"
    __INDENT: Final[int] = 4

    def __init__(self, out_dir: Path, generator_resources: list[cc.MetsGeneratorResource], corpus_label: str = "Ground Truth Corpus") -> None:
        self.__out_dir: Final[Path] = out_dir
        self.__generator_resources: Final[list[cc.MetsGeneratorResource]] = generator_resources
        # Template file is located in the same directory as this module
        template_path = Path(__file__).parent / 'template.corpus.xml'
        self.__mets_xml_document: Final[ET.ElementTree] = ET.parse(
            template_path,
            ET.XMLParser(remove_blank_text=True)
        )
        self.__doc_root: Final[ET.Element] = self.__mets_xml_document.getroot()
        # Extract namespaces directly using xpath
        self.__nsmap: Final[dict[str, str]] = {
            ns[0]: ns[1] for ns in self.__doc_root.xpath('//namespace::*') if ns[0]
        }

        # GET SECTION ROOT NODES FROM DOC
        self.__physroot: Final[ET.Element] = self.__doc_root.find(
            './/mets:div[@ID="physroot"]',
            self.__nsmap )
        self.__logroot: Final[ET.Element] = self.__doc_root.find(
            './/mets:div[@ID="logroot"]',
            self.__nsmap )
        # Set the corpus label from parameter
        self.__logroot.set('LABEL', corpus_label)
        self.__struct_link_root: Final[ET.Element] = self.__doc_root.find(
            './/mets:structLink',
            self.__nsmap)
        self.__file_group_image: Final[ET.Element] = self.__doc_root.find(
            './/mets:fileGrp[@USE="OCR-D-IMG"]',
            self.__nsmap )
        self.__file_group_fulltext: Final[ET.Element] = self.__doc_root.find(
            './/mets:fileGrp[@USE="FULLTEXT"]',
            self.__nsmap )
        self.__file_group_fulltext.set('USE', cc.GT_METS_FILEGROUP)

    def run(self) -> cc.MetsResource:

        # EXTRACT DATA FROM ORIG METS AND INSERT
        total: int = len(self.__generator_resources)
        for i, generator_resource in enumerate(self.__generator_resources):
            print(f'Process {i + 1} of {total} mets files - {generator_resource.mets.identifier_urn}')
            extract: cc.MetsModsInformation = self.__get_mets_data(
                index=i,
                page_urn=generator_resource.mets.identifier_urn,
                gt_file_path=generator_resource.gt.file_path,
                orig_mets_file_path=generator_resource.mets.file_path,
            )
            try:
                self.__physroot.append(extract.phys_div)
                self.__file_group_image.append(extract.file_image)
                self.__file_group_fulltext.append(extract.file_fulltext)
                self.__struct_link_root.append(extract.sm_link)
                self.__logroot.append(extract.log_div)
            except Exception:
                continue
            if extract.dmd_sec is not None:
                idx: int = len(self.__doc_root.findall('.//mets:dmdSec', namespaces=self.__nsmap))
                self.__doc_root.insert(idx, extract.dmd_sec)

        phys_divs_with_order_attrib: list[ET.Element] = self.__physroot.findall(
            './/mets:div[@ORDER]',
            self.__nsmap
        )
        for i, elm in enumerate(phys_divs_with_order_attrib):
            elm.set('ORDER', str(i + 1))

        log_divs_with_order_attrib: list[ET.Element] = self.__logroot.findall(
            './/mets:div[@ORDER]',
            self.__nsmap
        )
        for i, elm in enumerate(log_divs_with_order_attrib):
            elm.set('ORDER', str(i + 1))

        # FORMAT
        ET.indent(self.__doc_root, space=(" " * MetsGenerator.__INDENT))

        # SAVE
        out_file: Path = self.__out_dir.joinpath(MetsGenerator.__METS_FILE_NAME)
        self.__mets_xml_document.write(
            out_file,
            xml_declaration=True,
            pretty_print=True,
            encoding=self.__mets_xml_document.docinfo.encoding
        )

        return cc.MetsResource(
            identifier_urn="gt_2_mets",
            file_path=out_file
        )

    # #############################  SUB ##################################

    def __get_mets_data(
            self,
            index: int,
            page_urn: str,
            gt_file_path: Path,
            orig_mets_file_path: Path,
    ) -> cc.MetsModsInformation:
        try:
            doc = ET.parse(orig_mets_file_path, ET.XMLParser(remove_blank_text=True))
        except Exception:
            print(f"The xml file {gt_file_path} is damaged and will be ignored.")
            return None
        doc_root = doc.getroot()
        # Extract namespaces directly using xpath
        nsmap: dict[str, str] = {
            ns[0]: ns[1] for ns in doc_root.xpath('//namespace::*') if ns[0]
        }
        
        # PHYS
        #print(f"EXTRACTED\n{etree.tostring(doc)}\n\n")
        phys_div: ET.Element = doc_root.find(f'.//mets:div[@CONTENTIDS="{page_urn}"]', nsmap)
        try:
            del phys_div.attrib['ORDERLABEL']
        except KeyError:
            pass

        # FILES
        file_pointers: list[ET.Element] = phys_div.findall('.//mets:fptr', namespaces=nsmap)
        # Remove elements from their parent
        for el in file_pointers:
            el.getparent().remove(el)
        files: list[ET.Element] = [
            doc_root.find(f'.//mets:file[@ID="{fptr.get("FILEID")}"]', nsmap)
            for fptr
            in file_pointers
        ]
        file_image: ET.Element = next(file for file in files if file.getparent().get('USE') == 'MAX')
        file_ptr_image: ET.Element = next(fptr for fptr in file_pointers if fptr.get('FILEID') == file_image.get('ID'))
        phys_div.append(file_ptr_image)
        file_fulltext_id: str = f'{cc.GT_METS_FILEGROUP}-{(index + 1)}'
        file_ptr_fulltext: ET.Element = ET.Element(
            '{' + self.__nsmap['mets'] + '}fptr',
            attrib={'FILEID': file_fulltext_id}
        )
        phys_div.append(file_ptr_fulltext)
        file_fulltext: ET.Element = ET.Element(
            '{' + self.__nsmap['mets'] + '}file',
            attrib={
                'ID': file_fulltext_id,
                'MIMETYPE': 'application/vnd.prima.page+xml',
            }
        )
        file_fulltext.append(
            ET.Element(
                '{' + self.__nsmap['mets'] + '}FLocat',
                attrib={
                    "{" + self.__nsmap["xlink"] + "}href": str(gt_file_path.relative_to(self.__out_dir)),
                    'LOCTYPE': "OTHER",
                    'OTHERLOCTYPE': "FILE",
                }
            )
        )

        # LINK
        phys_id: str = phys_div.get('ID')
        sm_link: ET.Element = doc_root.find(f'.//mets:smLink[@xlink:to="{phys_id}"]', nsmap)

        # LOG
        log_id: str = sm_link.get("{" + self.__nsmap["xlink"] + "}from")
        log_div: ET.Element = doc_root.find(f'.//mets:div[@ID="{log_id}"]', nsmap)
        dmd_id: str = MetsGenerator.__find_dmd_id(log_div)
        log_div.set('DMDID', dmd_id)
        self.__file_group_fulltext.set('DMDID', dmd_id)
        try:
            del log_div.attrib['LABEL']
        except KeyError:
            pass
        # Remove all children elements
        for child in log_div.getchildren():
            child.getparent().remove(child)
        # log_div.append(file_ptr_image)

        dmd_sec: ET.Element | None = None

        mods_blocks: list[ET.Element] = doc_root.findall('.//mods:mods', nsmap)
        num_mods_blocks: int = len(mods_blocks)
        if num_mods_blocks > 1:
            raise Exception(
                f'METS file has more than one MODS block: found {num_mods_blocks} MODS blocks in {orig_mets_file_path}'
            )

        if self.__doc_root.find(f'.//mets:dmdSec[@ID="{dmd_id}"]', namespaces=self.__nsmap) is None:
            orig_dmd_sec: ET.Element = doc_root.find(f'.//mets:dmdSec[@ID="{dmd_id}"]', nsmap)
            orig_mods_root: ET.Element = orig_dmd_sec.find(f'.//mods:mods', nsmap)

            dmd_sec = copy.deepcopy(orig_dmd_sec)
            mods_root: ET.Element = dmd_sec.find('.//mods:mods', nsmap)
            # Remove all children elements
            for child in mods_root.getchildren():
                child.getparent().remove(child)

            mods_root.append(orig_mods_root.find('.//mods:titleInfo', nsmap))
            mods_root.extend(orig_mods_root.findall('.//mods:identifier', nsmap))
            mods_root.extend(orig_mods_root.findall('.//mods:language', nsmap))
            mods_root.extend(orig_mods_root.findall('.//mods:genre', nsmap))
            mods_root.append(orig_mods_root.find('.//mods:originInfo[@eventType="publication"]', nsmap))

        return cc.MetsModsInformation(
            phys_div=phys_div,
            file_image=file_image,
            file_fulltext=file_fulltext,
            sm_link=sm_link,
            log_div=log_div,
            dmd_sec=dmd_sec
        )


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("input", help="Path to the input directory")
    parser.add_argument("output", help="Path to the output directory")
    parser.add_argument(
        "-l",
        "--limit",
        help="Number of Files being processed, default = 0 (unlimited)",
        required=False,
        type=int,
        default=Gt2Mets.__DEFAULT_LIMIT
    )
    parser.add_argument(
        "-t",
        "--temp-dir",
        help="Path to the temporary directory",
        required=False,
        default=Gt2Mets.__DEFAULT_TEMP_DIR
    )
    args = parser.parse_args()
    corpus_args = cc.CorpusArgs(
        input_dir=Path(args.input).absolute(),
        output_dir=Path(args.output).absolute(),
        local_cache_dir=Path(args.temp_dir).absolute(),
        limit=int(args.limit),
        corpus_label="Ground Truth Corpus"
    )
    gt_2_mets: Gt2Mets = Gt2Mets()
    gt_2_mets.run()
