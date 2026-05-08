from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import ocr_util.corpus.common as cc
import ocr_util.corpus.load_metadata as lm


class DummyResponse:
    def __init__(
        self,
        *,
        ok: bool,
        url: str,
        text: str = "",
        payload: dict | None = None,
        content: bytes = b"",
        status_code: int = 200,
    ) -> None:
        self.ok = ok
        self.url = url
        self.text = text
        self._payload = payload if payload is not None else {}
        self.content = content
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


@patch("ocr_util.corpus.load_metadata.requests.get")
def test_solr_strategy_resolves_handle_from_handle_field(mock_get: Mock) -> None:
    """Resolve a SOLR hit that provides both handle and METS URI fields."""
    mock_get.return_value = DummyResponse(
        ok=True,
        url="https://opendata.uni-halle.de/solr/search/select?q=test",
        payload={
            "response": {
                "docs": [{
                    "handle": "123/456",
                    "local.mets.uri": "https://opendata.uni-halle.de/handle/123/456"
                }]
            }
        },
    )

    strategy = lm.SolrHandleStrategy(idx_base_urls=("https://opendata.uni-halle.de/solr/search/select",))
    result = strategy.resolve_handle("urn:nbn:de:3-1-23")

    assert result is not None
    assert result.oai_record_urn == "oai:opendata.uni-halle.de:123/456"
    assert result.source == "solr"


@patch("ocr_util.corpus.load_metadata.requests.get")
def test_solr_strategy_resolves_handle_from_local_mets_uri(mock_get: Mock) -> None:
    """Resolve a SOLR hit from the secondary host and preserve host in OAI id."""
    mock_get.return_value = DummyResponse(
        ok=True,
        url="https://opendata2.uni-halle.de/solr/search/select?q=test",
        payload={
            "response": {
                "docs": [{
                    "handle": "345/678",
                    "local.mets.uri": "https://opendata2.uni-halle.de/handle/345/678"
                }]
            }
        },
    )

    strategy = lm.SolrHandleStrategy(idx_base_urls=("https://opendata2.uni-halle.de/solr/search/select",))
    result = strategy.resolve_handle("urn:nbn:de:3-1-23")

    assert result is not None
    assert result.oai_record_urn == "oai:opendata2.uni-halle.de:345/678"
    assert result.source == "solr"


@patch("ocr_util.corpus.load_metadata.requests.get")
def test_handle_resolver_chain_falls_back_from_solr_to_nbn(mock_get: Mock) -> None:
    """Fall back to NBN when SOLR cannot resolve a matching document."""
    mock_get.side_effect = [
        DummyResponse(ok=True, url="https://opendata.uni-halle.de/solr/search/select", payload={"response": {"docs": []}}),
        DummyResponse(
            ok=True,
            url="https://opendata.uni-halle.de/handle/777/999",
            text="",
        ),
    ]

    chain = lm.HandleResolverChain(
        strategies=(
            lm.SolrHandleStrategy(idx_base_urls=("https://opendata.uni-halle.de/solr/search/select",)),
            lm.NbnResolverHandleStrategy(resolver_base_url="https://nbn-resolving.org/"),
        )
    )

    result = chain.resolve_handle("urn:nbn:de:3-1-23")

    assert result.oai_record_urn == "oai:opendata.uni-halle.de:777/999"
    assert result.source == "nbn-resolver"


def test_record_metadata_resolver_writes_file(tmp_path: Path) -> None:
    """Write fetched METS bytes to disk and return a MetsResource."""
    handle_resolver = Mock()
    handle_resolver.resolve_handle.return_value = lm.RecordResolutionResult(
        oai_record_urn="oai:opendata.uni-halle.de:123/456",
        source="solr",
    )
    oai_client = Mock()
    oai_client.fetch_mets.return_value = b"<mets/>"

    out_file = tmp_path / "sample.mets.xml"

    resolver = lm.RecordMetadataResolver(
        handle_resolver=handle_resolver,
        oai_client=oai_client,
    )

    result = resolver.fetch(urn="urn:nbn:de:3-1-23",
                            file_path=out_file)

    handle_resolver.resolve_handle.assert_called_once_with("urn:nbn:de:3-1-23")
    oai_client.fetch_mets.assert_called_once_with(
        identifier="oai:opendata.uni-halle.de:123/456",
        oai_base_url="https://opendata.uni-halle.de/oai/dd",
    )
    assert result.identifier_urn == "urn:nbn:de:3-1-23"
    assert out_file.exists()
    assert out_file.read_bytes() == b"<mets/>"


def test_record_metadata_resolver_raises_without_oai_base_url_mapping(tmp_path: Path) -> None:
    """Raise CorpusException when no OAI base URL can be inferred for host."""
    handle_resolver = Mock()
    handle_resolver.resolve_handle.return_value = lm.RecordResolutionResult(
        oai_record_urn="oai:example.invalid:123/456",
        source="solr",
    )
    oai_client = Mock()

    out_file = tmp_path / "sample.mets.xml"

    resolver = lm.RecordMetadataResolver(
        handle_resolver=handle_resolver,
        oai_client=oai_client,
    )

    with pytest.raises(cc.CorpusException, match="Unable to determine OAI-PMH base URL"):
        resolver.fetch(urn="urn:nbn:de:3-1-23", file_path=out_file)
    oai_client.fetch_mets.assert_not_called()


@patch("ocr_util.corpus.load_metadata.requests.get")
def test_oai_pmh_client_detects_error_response(mock_get: Mock) -> None:
    """Raise CorpusException when OAI-PMH returns an error element."""
    oai_error_response = b'''<?xml version="1.0" encoding="UTF-8"?><?xml-stylesheet type="text/xsl" href="static/style.xsl"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openarchives.org/OAI/2.0/ http://www.openarchives.org/OAI/2.0/OAI-PMH.xsd"><responseDate>2026-05-08T06:58:05Z</responseDate><request verb="GetRecord" identifier="oai:opendata.uni-halle.de:opendata.uni-halle.de/1981185920/113883" metadataPrefix="mets">https://opendata.uni-halle.de/oai/dd</request><error code="idDoesNotExist">The given id does not exist</error></OAI-PMH>'''
    mock_get.return_value = DummyResponse(
        ok=True,
        url="https://opendata.uni-halle.de/oai/dd",
        content=oai_error_response,
    )

    client = lm.OaiPmhClient()

    with pytest.raises(cc.CorpusException, match="OAI-PMH error.*idDoesNotExist.*The given id"):
        client.fetch_mets(
            identifier="oai:opendata.uni-halle.de:bad/id",
            oai_base_url="https://opendata.uni-halle.de/oai/dd",
        )
