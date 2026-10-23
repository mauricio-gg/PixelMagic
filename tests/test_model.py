"""The M3 fields: they round-trip, they have bounds, and an M2 sidecar still parses."""

from datetime import UTC, datetime

from pixelmagic.model import (
    DEFAULT_CELL_SIZE,
    AssetKind,
    AssetRecord,
    Cutout,
    Decomposition,
    Direction,
    GenerationRequest,
    Manifest,
    ManifestEntry,
    Pitch,
    PitchSource,
    SheetKind,
    TileRole,
)
from pixelmagic.store import Store

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)

#: What `promote_cutouts` will copy from a cutout onto its asset: the six
#: fields, all set, so a round-trip that loses one is caught by name.
SIX = {
    "pitch": 4.25,
    "pitch_source": PitchSource.MEASURED,
    "target": (17, 32),
    "footprint": (2, 2),
    "anchor": (8, 32),
    "role": TileRole.CENTRE,
}


def test_a_cutout_round_trips_the_six_fields_through_the_store(tmp_path):
    store = Store(tmp_path)
    cutout = Cutout(
        name="candle", sheet="20260905-120000", kind=SheetKind.PROPS,
        bbox=(0, 0, 68, 128), **SIX,
    )
    path = store.write_model(tmp_path / "cutout.json", cutout)
    reloaded = store.read_model(path, Cutout)
    assert reloaded == cutout
    assert reloaded.target == (17, 32)
    assert reloaded.pitch_source is PitchSource.MEASURED
    assert reloaded.role is TileRole.CENTRE


def test_an_asset_record_round_trips_the_same_six_fields(tmp_path):
    store = Store(tmp_path)
    record = AssetRecord(
        name="candle", kind=AssetKind.PROP, source_generation="20260905-120000",
        source_prompt="a candle", service="fake", **SIX,
    )
    path = store.write_model(tmp_path / "asset.json", record)
    assert store.read_model(path, AssetRecord) == record


def test_a_decomposition_round_trips_its_pitches(tmp_path):
    store = Store(tmp_path)
    decomposition = Decomposition(
        id="20260905-120000", created=NOW, source_generation="20260905-110000",
        pitches={SheetKind.PROPS: Pitch(value=4.25, source=PitchSource.MEASURED)},
    )
    path = store.write_model(tmp_path / "decomposition.json", decomposition)
    reloaded = store.read_model(path, Decomposition)
    assert reloaded.pitches[SheetKind.PROPS].value == 4.25
    assert reloaded.pitches[SheetKind.PROPS].source is PitchSource.MEASURED
    # The enum's own values are the words the spec uses, and they are what
    # lands in the file.
    assert '"measured"' in path.read_text()


def test_a_manifest_round_trips_cell_size_and_pitches(tmp_path):
    store = Store(tmp_path)
    manifest = Manifest(
        project="DemoGame", direction="cosy-den", exported=NOW, cell_size=16,
        pitches={
            "props": Pitch(value=4.25, source=PitchSource.MEASURED),
            "scene": Pitch(value=6.0, source=PitchSource.MEASURED),
        },
        entries=[
            ManifestEntry(
                name="candle", kind=AssetKind.PROP, width=17, height=32,
                source_prompt="a candle", service="fake", **SIX,
            )
        ],
    )
    path = store.write_model(tmp_path / "manifest.json", manifest)
    reloaded = store.read_model(path, Manifest)
    assert reloaded.cell_size == 16
    assert reloaded.pitches["scene"].value == 6.0
    assert reloaded.entries[0].anchor == (8, 32)


#: Sidecars exactly as M2 wrote them: none of the M3 fields present.
M2_CUTOUT = (
    '{"name": "crate", "ai_name": "Crate", "sheet": "20260905-120000", '
    '"kind": "props", "bbox": [0, 0, 24, 24], "cell": null}'
)
M2_ASSET = (
    '{"name": "crate", "kind": "prop", "source_generation": "20260905-110000", '
    '"source_prompt": "a crate", "service": "fake", "cell": null}'
)
M2_DECOMPOSITION = (
    '{"id": "20260905-120000", "created": "2026-09-05T12:00:00Z", '
    '"source_generation": "20260905-110000", "sheets": {}, "sheet_prompts": {}, '
    '"sheet_services": {}, "cutouts": [], "warnings": []}'
)
M2_DIRECTION = (
    '{"name": "cosy-den", "created": "2026-09-05T12:00:00Z", '
    '"style": "Warm SNES interiors.", "donts": [], "references": [], '
    '"service": "fake", "model": null, "refiner_defaults": {}, "approved_prompts": []}'
)
M2_MANIFEST = (
    '{"project": "DemoGame", "direction": "cosy-den", '
    '"exported": "2026-09-05T12:00:00Z", "entries": [{"name": "crate", "kind": "prop", '
    '"width": 24, "height": 24, "palette": [], "grid": null, "source_prompt": "a crate", '
    '"service": "fake", "refiner": {}, "refiner_tool": "stub", "route": "preserve", '
    '"warnings": [], "files": {"1x": "crate.png"}}]}'
)


def test_an_m2_cutout_and_asset_without_the_fields_still_parse():
    cutout = Cutout.model_validate_json(M2_CUTOUT)
    assert (
        cutout.pitch, cutout.pitch_source, cutout.target,
        cutout.footprint, cutout.anchor, cutout.role,
    ) == (None, None, None, None, None, None)
    asset = AssetRecord.model_validate_json(M2_ASSET)
    assert (
        asset.pitch, asset.pitch_source, asset.target,
        asset.footprint, asset.anchor, asset.role,
    ) == (None, None, None, None, None, None)


def test_an_m2_decomposition_direction_and_manifest_still_parse():
    assert Decomposition.model_validate_json(M2_DECOMPOSITION).pitches == {}
    assert Direction.model_validate_json(M2_DIRECTION).cell_size == DEFAULT_CELL_SIZE
    manifest = Manifest.model_validate_json(M2_MANIFEST)
    assert manifest.cell_size is None
    assert manifest.pitches == {}
    assert manifest.entries[0].target is None
    assert manifest.entries[0].pitch_source is None


#: A generation's request.json as it was written before #106: no `group`.
PRE_106_REQUEST = (
    '{"id": "20260905-110000", "created": "2026-09-05T11:00:00Z", "subject": "a den", '
    '"kind": "scene", "prompt": "Draw a den.", "service": "fake", "model": null, '
    '"size": [1024, 1024], "background": "transparent", "references": [], '
    '"parent_id": null, "imported_from": null}'
)


def test_sidecars_written_before_106_still_load():
    """A project made before #106 must open: none of its new fields may be required."""
    cutout = Cutout.model_validate_json(M2_CUTOUT)
    asset = AssetRecord.model_validate_json(M2_ASSET)
    decomposition = Decomposition.model_validate_json(M2_DECOMPOSITION)
    entry = Manifest.model_validate_json(M2_MANIFEST).entries[0]
    request = GenerationRequest.model_validate_json(PRE_106_REQUEST)
    assert cutout.origin is None
    assert asset.segmenter is None
    assert decomposition.segmenter is None
    assert entry.segmenter is None
    assert request.group is None
