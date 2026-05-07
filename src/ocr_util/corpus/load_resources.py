from __future__ import annotations

import pathlib
import re

import typing
import urllib.parse


import requests

import lxml.etree as ET

import ocr_util.corpus.common as cc


DEFAULT_NBN_URL: typing.Final[str] = "https://nbn-resolving.org/"
DEFAULT_REQUEST_HEADER = {"User-Agent": "ulbbot/ocr-util corpus (https://github.com/ulb-sachsen-anhalt/ocr-util)"}

DEFAULT_OAI_BASE_URL_MAPPING = {
    'opendata.uni-halle.de': 'https://opendata.uni-halle.de/oai/dd',
    'opendata2.uni-halle.de': 'https://opendata2.uni-halle.de/oai/dd',
}


class RecordMetadataResolver:
    """Get METS file for given URN, either from local file system
    or by tracking from filename over nbn down to OAI-PMH endpoint."""

    def __init__(
            self,
            urn: str,
            local_file_path: pathlib.Path,
            oai_base_url: typing.Optional[str] = None,
            url_urn_resolver: typing.Optional[str] = None,
    ):
        self.urn = urn
        self.file_path = local_file_path
        if url_urn_resolver is None:
            url_urn_resolver = DEFAULT_NBN_URL
        self.url_urn_resolver = url_urn_resolver
        self.url_oai_pmh_data = oai_base_url

    def run(self) -> cc.MetsResource:
        """Chasing Metadata"""
        if not self.file_path.exists():
            open_data_url: str = self.__obtain_open_data_url_by_urn_resolver()
            oai_handle: str = re.search(r'/handle/(\d+/\d+)', open_data_url).group(1)
            identifier_oai_pmh: str = f'{self.__url_oai_pmh_id_prefix}:{oai_handle}'
            mets_content: bytes = self.__load_mets_file(identifier_oai_pmh)
            mets_file: typing.BinaryIO
            with open(self.file_path, "wb") as mets_file:
                mets_file.write(mets_content)
        return cc.MetsResource(
            identifier_urn=self.urn,
            file_path=self.file_path
        )

    def __obtain_open_data_url_by_urn_resolver(self) -> str:
        req_url = urllib.parse.urljoin(self.url_urn_resolver, self.urn)
        response: requests.Response = requests.get(
            url=req_url,
            timeout=60
        )
        if not response.ok:
            raise cc.CorpusException(f"Request Error status {response.status_code} for {response.url}")
        handlename = 'handle=[0-9]*\/[0-9]*'
        handlefinder = re.compile(handlename)
        url_open_data_ulb: str = "https://opendata.uni-halle.de/handle/"+handlefinder.search(response.url).group(0)[7:]
        url_parse_result: urllib.parse.ParseResult = urllib.parse.urlparse(url_open_data_ulb)
        url_open_data_ulb_without_params: str = urllib.parse.urlunparse(
            (url_parse_result.scheme, url_parse_result.netloc, url_parse_result.path, "", "", "")
        )
        return url_open_data_ulb_without_params

    def __load_mets_file(self, identifier: str) -> bytes:
        response: requests.Response = requests.get(
            url=self.url_oai_pmh_data,
            params={
                "identifier": identifier,
                "verb": "GetRecord",
                "metadataPrefix": "mets",
            },
            headers=DEFAULT_REQUEST_HEADER,
            timeout=30
        )
        if not response.ok:
            raise cc.CorpusException(f"Request Error - {response.status_code} for {response.url}")
        return response.content
