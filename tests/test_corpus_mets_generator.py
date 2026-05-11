from pathlib import Path
import lxml.etree as ET

import pytest

import ocr_util.corpus.common as cc
import ocr_util.corpus.generate_corpus as gc

METS_NS = "{http://www.loc.gov/METS/}"


def _write_gt_file(out_dir: Path, name: str) -> Path:
    gt_dir = out_dir / "GT-PAGE"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_file = gt_dir / name
    gt_file.write_text("<PcGts/>", encoding="utf-8")
    return gt_file


def _write_source_mets(
    file_path: Path,
    *,
    page_urn: str,
    phys_id: str,
    log_id: str,
    dmd_id: str,
    with_orderlabel: bool,
    with_log_label: bool,
    mods_blocks: int,
) -> None:
    orderlabel_attr = ' ORDERLABEL="page"' if with_orderlabel else ""
    log_label_attr = ' LABEL="logical"' if with_log_label else ""

    mods_block = """
          <mods:mods>
            <mods:titleInfo><mods:title>Example</mods:title></mods:titleInfo>
            <mods:identifier type="urn">urn:example:test</mods:identifier>
            <mods:language><mods:languageTerm type="code">deu</mods:languageTerm></mods:language>
            <mods:genre>article</mods:genre>
            <mods:originInfo eventType="publication"><mods:dateIssued>1900</mods:dateIssued></mods:originInfo>
          </mods:mods>
    """
    mods_xml = "\n".join(mods_block for _ in range(mods_blocks))

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/"
           xmlns:mods="http://www.loc.gov/mods/v3"
           xmlns:xlink="http://www.w3.org/1999/xlink">
  <mets:dmdSec ID="{dmd_id}">
    <mets:mdWrap MDTYPE="MODS">
      <mets:xmlData>
{mods_xml}
      </mets:xmlData>
    </mets:mdWrap>
  </mets:dmdSec>
  <mets:fileSec>
    <mets:fileGrp USE="MAX">
      <mets:file ID="IMG_{phys_id}" MIMETYPE="image/tiff">
        <mets:FLocat xlink:href="{phys_id}.tif" LOCTYPE="URL" />
      </mets:file>
    </mets:fileGrp>
    <mets:fileGrp USE="FULLTEXT">
      <mets:file ID="TXT_{phys_id}" MIMETYPE="application/xml">
        <mets:FLocat xlink:href="{phys_id}.xml" LOCTYPE="URL" />
      </mets:file>
    </mets:fileGrp>
  </mets:fileSec>
  <mets:structMap TYPE="PHYSICAL">
    <mets:div ID="physroot">
      <mets:div ID="{phys_id}" CONTENTIDS="{page_urn}" ORDER="9"{orderlabel_attr}>
        <mets:fptr FILEID="IMG_{phys_id}" />
        <mets:fptr FILEID="TXT_{phys_id}" />
      </mets:div>
    </mets:div>
  </mets:structMap>
  <mets:structMap TYPE="LOGICAL">
    <mets:div ID="logroot" TYPE="document">
      <mets:div ID="VOL_{phys_id}" TYPE="volume" DMDID="{dmd_id}">
        <mets:div ID="{log_id}" ORDER="7"{log_label_attr}>
          <mets:div ID="child_{log_id}" />
        </mets:div>
      </mets:div>
    </mets:div>
  </mets:structMap>
  <mets:structLink>
    <mets:smLink xlink:from="{log_id}" xlink:to="{phys_id}" />
  </mets:structLink>
</mets:mets>
"""
    file_path.write_text(xml, encoding="utf-8")


def _build_resource(
    out_dir: Path, mets_file: Path, gt_file: Path, page_urn: str
) -> cc.CorpusInput:
    gt = cc.GroundtruthFile(
        identifier=page_urn,
        file_base_name=gt_file.stem,
        file_path=gt_file,
        relative_file_path=gt_file.relative_to(out_dir),
        languages=["deu"],
    )
    # mets = cc.MetsResource(identifier_urn=page_urn, local_file_path=mets_file)
    corpus_input = cc.CorpusInput(groundtruth_file=gt)
    corpus_input.cached_media_mets_file = mets_file
    return corpus_input


def _build_corpus_args(out_dir: Path, corpus_label: str = "Ground Truth Corpus") -> cc.CorpusArgs:
    return cc.CorpusArgs(
        input_dir=out_dir,
        output_dir=out_dir,
        local_cache_dir=out_dir,
        corpus_label=corpus_label,
    )

def _patch_template_parse_with_struct_link():
    original_parse = gc.ET.parse

    def parse_with_struct_link(source, *args, **kwargs):
        tree = original_parse(source, *args, **kwargs)
        if Path(str(source)).name == "template.corpus.xml":
            struct_link = tree.getroot().find(f".//{METS_NS}structLink")
            if struct_link is None:
                gc.ET.SubElement(tree.getroot(), f"{METS_NS}structLink")
        return tree

    return parse_with_struct_link


def test_mwe_corpus_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify successful METS generation, order normalization, and DMD deduplication
    for a corpus with two pages sharing common MODS metadata."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    page_urn_1 = "urn:nbn:de:gbv:3:1-123456/fragment/page=0001"
    page_urn_2 = "urn:nbn:de:gbv:3:1-123456/fragment/page=0002"

    gt_1 = _write_gt_file(out_dir, "urn+nbn+de+gbv+3+1-123456-fp-0001.xml")
    gt_2 = _write_gt_file(out_dir, "urn+nbn+de+gbv+3+1-123456-fp-0002.xml")

    mets_1 = tmp_path / "page_1.mets.xml"
    mets_2 = tmp_path / "page_2.mets.xml"

    _write_source_mets(
        mets_1,
        page_urn=page_urn_1,
        phys_id="PHYS_0001",
        log_id="LOG_0001",
        dmd_id="DMDLOG_0001",
        with_orderlabel=True,
        with_log_label=True,
        mods_blocks=1,
    )
    _write_source_mets(
        mets_2,
        page_urn=page_urn_2,
        phys_id="PHYS_0002",
        log_id="LOG_0002",
        dmd_id="DMDLOG_0001",
        with_orderlabel=True,
        with_log_label=True,
        mods_blocks=1,
    )

    resources = [
        _build_resource(out_dir, mets_1, gt_1, page_urn_1),
        _build_resource(out_dir, mets_2, gt_2, page_urn_2),
    ]

    monkeypatch.setattr(gc, "cc", cc, raising=False)
    monkeypatch.setattr(gc.ET, "parse", _patch_template_parse_with_struct_link())

    cfile = gc.CorpusFile(
      corpus_args=_build_corpus_args(out_dir, corpus_label="Test Corpus"),
      generator_resources=resources,
    )
    result = cfile.build()

    assert result.file_path.exists()
    assert result.file_path.name == "mets.xml"

    document = ET.parse(result.file_path)

    log_root = document.find(f'.//{METS_NS}div[@ID="corpus_root"]')
    assert log_root is not None
    assert log_root.get("LABEL") == "Test Corpus"

    assert len(document.findall(f'.//{METS_NS}dmdSec[@ID="DMDLOG_0001"]')) == 1

    phys_orders = [
        e.get("ORDER")
        for e in document.findall(
            f'.//{METS_NS}structMap[@TYPE="PHYSICAL"]//{METS_NS}div[@ORDER]'
        )
    ]
    log_orders = [
        e.get("ORDER")
        for e in document.findall(
            f'.//{METS_NS}structMap[@TYPE="LOGICAL"]//{METS_NS}div[@ORDER]'
        )
    ]
    assert phys_orders == ["1", "2"]
    assert log_orders == ["1", "2"]

    fulltext_group = document.find(f'.//{METS_NS}fileGrp[@USE="{cc.GT_METS_FILEGROUP}"]')
    assert fulltext_group is not None
    assert len(fulltext_group.findall(f".//{METS_NS}file")) == 2

    sm_links = document.findall(f".//{METS_NS}structLink/{METS_NS}smLink")
    assert len(sm_links) == 2


def _write_source_mets_multivolume(
    file_path: Path, *, page_urn: str, phys_id: str, log_id: str, dmd_id: str
) -> None:
    """Write a source METS with host-level metadata but no publication originInfo."""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/"
           xmlns:mods="http://www.loc.gov/mods/v3"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xmlns:xlink="http://www.w3.org/1999/xlink">
  <mets:dmdSec ID="{dmd_id}">
    <mets:mdWrap MIMETYPE="text/xml" MDTYPE="MODS">
      <mets:xmlData>
        <mods:mods version="3.7"
            xsi:schemaLocation="http://www.loc.gov/mods/v3 http://www.loc.gov/standards/mods/v3/mods-3-7.xsd">
          <mods:identifier type="vd18">9072299X</mods:identifier>
          <mods:identifier type="gbv">190330430</mods:identifier>
          <mods:identifier type="urn">urn:nbn:de:gbv:3:1-831022</mods:identifier>
          <mods:relatedItem type="host">
            <mods:titleInfo>
              <mods:title>Gesammlete Nachrichten und Urkunden</mods:title>
            </mods:titleInfo>
            <mods:identifier type="vd18">90722949</mods:identifier>
            <mods:identifier type="gbv">190330295</mods:identifier>
            <mods:identifier type="urn">urn:nbn:de:gbv:3:1-831036</mods:identifier>
          </mods:relatedItem>
        </mods:mods>
      </mets:xmlData>
    </mets:mdWrap>
  </mets:dmdSec>
  <mets:fileSec>
    <mets:fileGrp USE="MAX">
      <mets:file ID="IMG_{phys_id}" MIMETYPE="image/tiff">
        <mets:FLocat xlink:href="{phys_id}.tif" LOCTYPE="URL" />
      </mets:file>
    </mets:fileGrp>
    <mets:fileGrp USE="FULLTEXT">
      <mets:file ID="TXT_{phys_id}" MIMETYPE="application/xml">
        <mets:FLocat xlink:href="{phys_id}.xml" LOCTYPE="URL" />
      </mets:file>
    </mets:fileGrp>
  </mets:fileSec>
  <mets:structMap TYPE="PHYSICAL">
    <mets:div ID="physroot">
      <mets:div ID="{phys_id}" CONTENTIDS="{page_urn}" ORDER="1">
        <mets:fptr FILEID="IMG_{phys_id}" />
        <mets:fptr FILEID="TXT_{phys_id}" />
      </mets:div>
    </mets:div>
  </mets:structMap>
  <mets:structMap TYPE="LOGICAL">
    <mets:div ID="logroot" TYPE="document">
      <mets:div ID="VOL_{phys_id}" TYPE="volume" DMDID="{dmd_id}">
        <mets:div ID="{log_id}" ORDER="1"></mets:div>
      </mets:div>
    </mets:div>
  </mets:structMap>
  <mets:structLink>
    <mets:smLink xlink:from="{log_id}" xlink:to="{phys_id}" />
  </mets:structLink>
</mets:mets>
"""
    file_path.write_text(xml, encoding="utf-8")


def test_corpus_with_multivolumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify no identifiers from related_host present."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    page_urn = "urn:nbn:de:gbv:3:1-831022"
    gt_file = _write_gt_file(out_dir, "page_mv.xml")
    mets_file = tmp_path / "page_mv.mets.xml"
    _write_source_mets_multivolume(
        mets_file,
        page_urn=page_urn,
        phys_id="PHYS_0010",
        log_id="LOG_0010",
        dmd_id="md15404308",
    )

    resources = [_build_resource(out_dir, mets_file, gt_file, page_urn)]
    monkeypatch.setattr(gc, "cc", cc, raising=False)
    monkeypatch.setattr(gc.ET, "parse", _patch_template_parse_with_struct_link())

    generator = gc.CorpusFile(
      corpus_args=_build_corpus_args(out_dir),
      generator_resources=resources,
    )
    
    # act
    result = generator.build()

    # assert
    assert result.file_path.exists()
    assert result.file_path.name == "mets.xml"
    doc_root = ET.parse(result.file_path).getroot()
    dmd_identifiers = doc_root.xpath('//mods:identifier/text()', namespaces={'mods': 'http://www.loc.gov/mods/v3'})
    assert dmd_identifiers is not None
    assert len(dmd_identifiers) == 3
    assert ['9072299X', '190330430', 'urn:nbn:de:gbv:3:1-831022'] == dmd_identifiers
