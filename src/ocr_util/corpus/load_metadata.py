"""Resolve and cache METS metadata for corpus records identified by URN.

This module implements a strategy-based handle resolution pipeline used by the
corpus generator:

1. Query SOLR indexes for a record handle.
2. Fall back to NBN resolver when SOLR does not return a handle.
3. Fetch METS via OAI-PMH ``GetRecord`` using the resolved handle.

The main entry point is ``RecordMetadataResolver``, which coordinates lookup,
endpoint resolution, and local cache file creation.
"""

import dataclasses
import pathlib
import re

import typing
import urllib.parse

import lxml.etree as ET
import requests

import ocr_util.corpus.common as cc


DEFAULT_NBN_URL: typing.Final[str] = "https://nbn-resolving.org/"
DEFAULT_REQUEST_HEADER = {"User-Agent": "ulbbot/ocr-util corpus (https://github.com/ulb-sachsen-anhalt/ocr-util)"}

DEFAULT_OAI_BASE_URL_MAPPING = {
    'opendata.uni-halle.de': 'https://opendata.uni-halle.de/oai/dd',
    'opendata2.uni-halle.de': 'https://opendata2.uni-halle.de/oai/dd',
}

FALLBACK_IDX_BASE_URLS = {
    "https://opendata.uni-halle.de/solr/search/select",
    "https://opendata2.uni-halle.de/solr/search/select"
}

DEFAULT_OAI_ID_PREFIX_HOST_MAPPING = {
    "https://opendata.uni-halle.de/oai/dd": "opendata.uni-halle.de",
    "https://opendata2.uni-halle.de/oai/dd": "opendata2.uni-halle.de",
}


@dataclasses.dataclass(frozen=True)
class RecordResolutionResult:
    """Normalized result of a handle lookup attempt.

    Attributes:
        oai_record_urn: Repository handle in the form "<host>/<identifier>".
        source: Name of the strategy that produced the handle.
    """
    source: str
    oai_record_urn: str


class HandleResolutionStrategy(typing.Protocol):
    """Contract for resolving matching repository handle for given URN."""

    def resolve_handle(self, urn: str) -> typing.Optional[RecordResolutionResult]:
        """Return handle lookup result or ``None`` when concrete strategy cannot resolve it."""


def _extract_handle(value: str) -> typing.Optional[str]:
    """Extract a ``<prefix>/<suffix>`` handle from URL-like or query-like text."""
    match = re.search(r"/handle/(\d+/\d+)", value)
    if match is not None:
        return match.group(1)
    match = re.search(r"handle=(\d+/\d+)", value)
    if match is not None:
        return match.group(1)
    return None


def _append_encoded_urn(base_url: str, urn: str) -> str:
    """Build a resolver URL by appending a percent-encoded URN as path segment."""
    encoded_urn = urllib.parse.quote(urn, safe="")
    return f"{base_url.rstrip('/')}/{encoded_urn}"


def _host_from_url(url: str) -> typing.Optional[str]:
    """Return URL host or ``None`` when not present."""
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc or None


class SolrHandleStrategy:
    """Resolve handles from configured SOLR index endpoints.

    The strategy queries each endpoint with the URN, then attempts to extract the
    handle from ``handle`` first and ``local.mets.uri`` as fallback.
    """

    def __init__(
        self,
        idx_base_urls: typing.Iterable[str],
        timeout: int = 60,
    ):
        """Initialize strategy with one or more SOLR base URLs."""
        self._idx_base_urls = tuple(idx_base_urls)
        self._timeout = timeout

    def resolve_handle(self, urn: str) -> typing.Optional[RecordResolutionResult]:
        """Resolve handle via SOLR and return ``None`` when no document matches."""
        urn_main_part = urn.split("/", 1)[0]
        for idx_base_url in self._idx_base_urls:
            response = wrap_request(
                url=idx_base_url,
                params={
                    "q": f'dc.identifier.urn:"{urn_main_part}"',
                    "fl": "handle,local.mets.uri",
                    "wt": "json",
                },
                timeout=self._timeout,
                headers=DEFAULT_REQUEST_HEADER,
            )
            if not response or not response.ok:
                continue
            try:
                data = response.json()
            except ValueError:
                continue

            docs = data.get("response", {}).get("docs", [])
            if not docs:
                continue
            if len(docs) != 1:
                alert = f"from {idx_base_url} for urn'{urn}' invalid docs {docs}"
                raise cc.CorpusException(alert)
            doc = docs[0]
            handle_value = doc.get("handle", None)
            mets_uri_value = doc.get("local.mets.uri", None)
            if isinstance(mets_uri_value, list):
                mets_uri_value = mets_uri_value[0] if mets_uri_value else None

            if handle_value is None or mets_uri_value is None:
                alert = f"from {idx_base_url} for urn'{urn}' missing handle or local.mets.uri in doc {doc}"
                raise cc.CorpusException(alert)
            oai_host = urllib.parse.urlparse(idx_base_url).netloc
            oai_record_urn = f"oai:{oai_host}:{handle_value}"
            return RecordResolutionResult(
                source="solr",
                oai_record_urn=oai_record_urn,
            )
        return None


class NbnResolverHandleStrategy:
    """Resolve handles by following NBN resolver responses."""

    def __init__(self, resolver_base_url: str = DEFAULT_NBN_URL, timeout: int = 60):
        """Initialize strategy with resolver base URL and request timeout."""
        self._resolver_base_url = resolver_base_url
        self._timeout = timeout

    def resolve_handle(self, urn: str) -> typing.Optional[RecordResolutionResult]:
        """Resolve handle from redirect URL or response body returned by the resolver."""
        request_url = _append_encoded_urn(self._resolver_base_url, urn)
        try:
            response = requests.get(url=request_url, timeout=self._timeout)
        except requests.RequestException:
            return None
        if not response.ok:
            return None

        handle = _extract_handle(response.url)
        if handle is None:
            handle = _extract_handle(response.text)
        if handle is None:
            return None
        host = _host_from_url(response.url)
        if host is None:
            return None
        return RecordResolutionResult(
            oai_record_urn=f"oai:{host}:{handle}",
            source="nbn-resolver",
        )


class HandleResolverChain:
    """Try multiple handle resolution strategies in deterministic order."""

    def __init__(self, strategies: typing.Iterable[HandleResolutionStrategy]):
        """Store strategies in call order for fallback execution."""
        self._strategies = tuple(strategies)

    def resolve_handle(self, urn: str) -> RecordResolutionResult:
        """Return first successful result or raise ``CorpusException`` after exhaustion."""
        for strategy in self._strategies:
            result = strategy.resolve_handle(urn)
            if result is not None:
                return result
        raise cc.CorpusException(f"Unable to resolve handle for URN '{urn}' via SOLR or NBN resolver")


class OaiPmhClient:
    """Minimal client for OAI-PMH ``GetRecord`` retrieval of METS payloads."""

    def __init__(self, timeout: int = 30):
        """Initialize client with request timeout in seconds."""
        self._timeout = timeout

    def fetch_mets(self, identifier: str, oai_base_url: str) -> bytes:
        """Fetch METS bytes for one OAI identifier from the configured endpoint.

        Raises CorpusException if:
        - HTTP request fails or returns non-200 status
        - OAI-PMH response contains an error element
        """
        response = wrap_request(
            url=oai_base_url,
            params={
                "identifier": identifier,
                "verb": "GetRecord",
                "metadataPrefix": "mets",
            },
            headers=DEFAULT_REQUEST_HEADER,
            timeout=self._timeout,
        )
        if not response or not response.ok:
            status = response.status_code if response else "unknown"
            raise cc.CorpusException(f"HTTP error {status} for {oai_base_url}")

        # Parse XML and check for OAI-PMH error element
        try:
            root = ET.fromstring(response.content)
            ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}
            error = root.find(".//oai:error", ns)
            if error is not None:
                error_code = error.get("code", "unknown")
                error_text = error.text or ""
                raise cc.CorpusException(
                    f"OAI-PMH error for identifier '{identifier}': "
                    f"{error_code} - {error_text}"
                )
        except ET.XMLSyntaxError as e:
            raise cc.CorpusException(f"Invalid XML response: {e}")

        return response.content


def _resolve_oai_base_url(host: str, explicit_oai_base_url: typing.Optional[str]) -> str:
    """Resolve OAI-PMH endpoint from explicit configuration or host mapping."""
    if host.startswith("oai:"):
        host = host[4:]
    if ":" in host:
        host = host.split(":", 1)[0]
    if explicit_oai_base_url:
        return explicit_oai_base_url
    if host in DEFAULT_OAI_BASE_URL_MAPPING:
        return DEFAULT_OAI_BASE_URL_MAPPING[host]
    raise cc.CorpusException("Unable to determine OAI-PMH base URL")


def _oai_identifier_prefix_host(oai_base_url: str) -> str:
    """Derive host part for OAI identifier prefix from a base URL."""
    if oai_base_url in DEFAULT_OAI_ID_PREFIX_HOST_MAPPING:
        return DEFAULT_OAI_ID_PREFIX_HOST_MAPPING[oai_base_url]
    parsed = urllib.parse.urlparse(oai_base_url)
    if parsed.netloc:
        return parsed.netloc
    raise cc.CorpusException(f"Unable to derive OAI identifier prefix host from '{oai_base_url}'")


class RecordMetadataResolver:
    """Resolve and cache METS metadata for a single URN.

    Resolution flow:
    1. Try strategy chain to obtain repository handle (SOLR first, NBN fallback).
    2. Determine OAI-PMH endpoint from explicit argument or inferred host mapping.
    3. Download METS via OAI-PMH ``GetRecord`` and write to local cache path.
    """

    def __init__(
            self,
            # urn: str,
            # local_file_path: pathlib.Path,
            oai_base_url: typing.Optional[str] = None,
            handle_resolver: typing.Optional[HandleResolverChain] = None,
            oai_client: typing.Optional[OaiPmhClient] = None,
    ):
        """Create a resolver instance for one URN/file target.

        Args:
            urn: URN of the source record.
            local_file_path: Destination path for cached METS file.
            oai_base_url: Optional explicit OAI endpoint override.
            # url_urn_resolver: Base URL of NBN resolver.
            handle_resolver: Optional custom strategy chain (primarily for tests).
            oai_client: Optional custom OAI client implementation.
        """
        # self.urn = urn
        # self.file_path = local_file_path
        self.url_oai_pmh_data = oai_base_url
        self._handle_resolver = handle_resolver or HandleResolverChain(
            strategies=(
                SolrHandleStrategy(idx_base_urls=FALLBACK_IDX_BASE_URLS),
                NbnResolverHandleStrategy(resolver_base_url=DEFAULT_NBN_URL),
            )
        )
        self._oai_client = oai_client or OaiPmhClient()

    def fetch(self, urn:str, file_path:pathlib.Path) -> cc.MetsResource:
        """Ensure METS file exists locally and return resulting ``MetsResource``.

        The method is cache-aware: if the file already exists, no network calls are made.
        """
        if not file_path.exists():
            resolution = self._handle_resolver.resolve_handle(urn)
            calculated_oai_host = resolution.oai_record_urn.split("/")[0]
            oai_base_url = _resolve_oai_base_url(
                host=calculated_oai_host,
                explicit_oai_base_url=self.url_oai_pmh_data,
            )
            mets_content = self._oai_client.fetch_mets(
                identifier=resolution.oai_record_urn,
                oai_base_url=oai_base_url,
            )
            with open(file_path, "wb") as mets_file:
                mets_file.write(mets_content)
        return cc.MetsResource(
            identifier_urn=urn,
            local_file_path=file_path
        )


def wrap_request(url:str, timeout = 30,
                 params: typing.Optional[typing.Dict[str, str]] = None,
                 headers: typing.Optional[typing.Dict[str, str]] = None) -> typing.Optional[requests.Response]:
    """Wrap requests in one single place"""
    response = None
    actual_headers = headers if headers is not None else {}
    actual_params = params if params is not None else {}
    try:
        response = requests.get(
            url=url,
            params=actual_params,
            timeout=timeout,
            headers=actual_headers,
        )
    except requests.RequestException:
        pass
    return response
