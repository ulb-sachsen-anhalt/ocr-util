from pathlib import Path

import lxml.etree as ET
import pytest

import ocr_util.corpus.common as cc
import ocr_util.corpus.generate_corpus as gc

METS_NS = "{http://www.loc.gov/METS/}"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def _write_gt_file(out_dir: Path, name: str) -> Path:
    gt_dir = out_dir / "GT-PAGE"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_file = gt_dir / name
    gt_file.write_text("<PcGts/>", encoding="utf-8")
    return gt_file


def _write_source_mets_without_fulltext(
    file_path: Path,
    *,
    page_urn: str,
    phys_id: str,
    log_id: str,
    dmd_id: str,
) -> None:
    xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<mets:mets xmlns:mets=\"http://www.loc.gov/METS/\"
           xmlns:mods=\"http://www.loc.gov/mods/v3\"
           xmlns:xlink=\"http://www.w3.org/1999/xlink\">
  <mets:dmdSec ID=\"{dmd_id}\">
    <mets:mdWrap MDTYPE=\"MODS\">
      <mets:xmlData>
        <mods:mods>
          <mods:titleInfo><mods:title>Example</mods:title></mods:titleInfo>
          <mods:identifier type=\"urn\">urn:example:no-fulltext</mods:identifier>
          <mods:language><mods:languageTerm type=\"code\">deu</mods:languageTerm></mods:language>
        </mods:mods>
      </mets:xmlData>
    </mets:mdWrap>
  </mets:dmdSec>
  <mets:fileSec>
    <mets:fileGrp USE=\"MAX\">
      <mets:file ID=\"IMG_{phys_id}\" MIMETYPE=\"image/tiff\">
        <mets:FLocat xlink:href=\"{phys_id}.tif\" LOCTYPE=\"URL\" />
      </mets:file>
    </mets:fileGrp>
  </mets:fileSec>
  <mets:structMap TYPE=\"PHYSICAL\">
    <mets:div ID=\"physroot\">
      <mets:div ID=\"{phys_id}\" CONTENTIDS=\"{page_urn}\" ORDER=\"1\">
        <mets:fptr FILEID=\"IMG_{phys_id}\" />
      </mets:div>
    </mets:div>
  </mets:structMap>
  <mets:structMap TYPE=\"LOGICAL\">
    <mets:div ID=\"logroot\" TYPE=\"document\">
      <mets:div ID=\"VOL_{phys_id}\" TYPE=\"volume\" DMDID=\"{dmd_id}\">
        <mets:div ID=\"{log_id}\" ORDER=\"1\"></mets:div>
      </mets:div>
    </mets:div>
  </mets:structMap>
  <mets:structLink>
    <mets:smLink xlink:from=\"{log_id}\" xlink:to=\"{phys_id}\" />
  </mets:structLink>
</mets:mets>
"""
    file_path.write_text(xml, encoding="utf-8")


def _build_resource(out_dir: Path, mets_file: Path, gt_file: Path, page_urn: str) -> cc.CorpusPageInput:
    gt = cc.GroundtruthFile(
        identifier=page_urn,
        file_base_name=gt_file.stem,
        file_path=gt_file,
        relative_file_path=gt_file.relative_to(out_dir),
        languages=["deu"],
    )
    corpus_input = cc.CorpusPageInput(groundtruth_file=gt)
    corpus_input.cached_media_mets_file = mets_file
    return corpus_input


def _build_corpus_args(out_dir: Path, corpus_label: str = "Ground Truth Corpus") -> cc.CorpusArgs:
    return cc.CorpusArgs(
        input_dir=out_dir,
        output_dir=out_dir,
        local_cache_dir=out_dir,
        corpus_label=corpus_label,
    )


def test_build_creates_fulltext_from_scratch_when_missing_in_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    page_urn = "urn:nbn:de:gbv:3:1-999999/fragment/page=0001"
    gt_file = _write_gt_file(out_dir, "urn+nbn+de+gbv+3+1-999999-fp-0001.xml")

    source_mets = tmp_path / "source_no_fulltext.mets.xml"
    _write_source_mets_without_fulltext(
        source_mets,
        page_urn=page_urn,
        phys_id="PHYS_0001",
        log_id="LOG_0001",
        dmd_id="DMDLOG_0001",
    )

    resources = [_build_resource(out_dir, source_mets, gt_file, page_urn)]
    monkeypatch.setattr(gc, "cc", cc, raising=False)

    result = gc.Corpus(
        cargs=_build_corpus_args(out_dir, corpus_label="Test Corpus"),
        inputs=resources,
    ).build()

    document = ET.parse(result.file_path)

    fulltext_group = document.find(f'.//{METS_NS}fileGrp[@USE="{cc.GT_METS_FILEGROUP_FULLTEXT}"]')
    assert fulltext_group is not None

    fulltext_files = fulltext_group.findall(f"{METS_NS}file")
    assert len(fulltext_files) == 1

    created_fulltext = fulltext_files[0]
    created_id = f"{cc.DEFAULT_FULLEXT_FILEGROUP}-0001"
    assert created_fulltext.get("ID") == created_id
    assert created_fulltext.get("MIMETYPE") == "application/vnd.prima.page+xml"

    flocat = created_fulltext.find(f"{METS_NS}FLocat")
    assert flocat is not None
    assert flocat.get(XLINK_HREF) == str(gt_file.relative_to(out_dir))
    assert flocat.get("LOCTYPE") == "OTHER"
    assert flocat.get("OTHERLOCTYPE") == "FILE"

    page_div = document.find(f'.//{METS_NS}structMap[@TYPE="PHYSICAL"]//{METS_NS}div[@CONTENTIDS="{page_urn}"]')
    assert page_div is not None

    fileids = [fptr.get("FILEID") for fptr in page_div.findall(f"{METS_NS}fptr")]
    assert created_id in fileids
