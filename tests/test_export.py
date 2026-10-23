import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    Direction,
    Manifest,
    Pitch,
    PitchSource,
    RefineReport,
    RefinerSettings,
    TileRole,
)
from pixelmagic.refiner.base import LOW_GRID_CONFIDENCE
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.stages.export import (
    MAX_SCALE,
    collapse_notes,
    export,
    scale_nearest,
)
from pixelmagic.stages.refine import TARGET_NOT_USABLE, load_report, refine_asset
from pixelmagic.stages.slice import SEAM_VISIBLE
from pixelmagic.store import Store

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_aseprite(monkeypatch):
    """Task 13's tests are about PNGs and the manifest, not Aseprite."""
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None, raising=False)


def stamp_layout(store, name, **fields):
    """Rewrite one asset's `asset.json` with the M3 grid fields set.

    `write_asset` (in `conftest.py`) stages an M2-shaped asset with none of
    them; tests that care how export copies pitch/target/footprint/anchor/
    role use this to give one asset a chosen set of values.
    """
    path = store.asset_dir("cosy-den", name) / "asset.json"
    record = store.read_model(path, AssetRecord)
    store.write_model(path, record.model_copy(update=fields))


@pytest.fixture
def setup(tmp_path, write_asset):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    write_asset(store, "wooden-chair", AssetKind.PROP)
    write_asset(store, "stone-floor", AssetKind.TILE, cell=(8, 8))
    return store, direction


def test_scale_nearest_keeps_hard_edges():
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0, 255)
    scaled = scale_nearest(Image.fromarray(pixels, "RGBA"), 2)
    assert scaled.size == (4, 4)
    array = np.asarray(scaled)
    assert tuple(array[0, 0]) == (255, 0, 0, 255)
    assert tuple(array[1, 1]) == (255, 0, 0, 255)
    assert tuple(array[0, 2]) == (0, 0, 0, 0)


def test_scale_of_one_returns_the_same_size():
    image = Image.new("RGBA", (5, 3))
    assert scale_nearest(image, 1).size == (5, 3)


def test_export_writes_native_and_upscaled_pngs(setup):
    store, direction = setup
    bundle = export(store, direction, stamp="20260903-191233", now=NOW)
    folder = bundle.directory
    assert (folder / "wooden-chair.png").exists()
    assert (folder / "wooden-chair@2x.png").exists()
    assert (folder / "wooden-chair@4x.png").exists()
    assert Image.open(folder / "wooden-chair@4x.png").size == (32, 32)


def test_the_bundle_is_named_direction_and_stamp(setup):
    store, direction = setup
    bundle = export(store, direction, stamp="20260903-191233", now=NOW)
    assert bundle.directory.name == "cosy-den-20260903-191233"


def test_only_the_requested_scales_are_written(setup):
    store, direction = setup
    bundle = export(store, direction, scales=(1,), stamp="s", now=NOW)
    assert (bundle.directory / "wooden-chair.png").exists()
    assert not (bundle.directory / "wooden-chair@2x.png").exists()


def test_export_refuses_a_scale_of_zero_before_writing_anything(setup):
    """Guardian I1: `scale=0` used to write a bundle with zero entries and a
    page that lied about why — the refusal now happens before any folder
    exists."""
    store, direction = setup
    with pytest.raises(ValueError, match="0"):
        export(store, direction, scales=(0,), stamp="s", now=NOW)
    assert not (store.root / "exports").exists()


def test_export_refuses_a_scale_larger_than_the_maximum(setup):
    """Guardian I1: `scale=20000` used to ask Pillow for a resize large enough
    to exhaust memory. `MAX_SCALE` is the bound every face now shares."""
    store, direction = setup
    with pytest.raises(ValueError, match=str(MAX_SCALE + 1)):
        export(store, direction, scales=(MAX_SCALE + 1,), stamp="s", now=NOW)
    assert not (store.root / "exports").exists()


def test_the_manifest_carries_provenance_for_every_asset(setup):
    store, direction = setup
    bundle = export(store, direction, stamp="s", now=NOW)
    manifest = Manifest.model_validate_json((bundle.directory / "manifest.json").read_text())
    assert manifest.project == "DemoGame"
    assert manifest.direction == "cosy-den"
    assert [entry.name for entry in manifest.entries] == ["stone-floor", "wooden-chair"]
    chair = next(entry for entry in manifest.entries if entry.name == "wooden-chair")
    assert chair.kind is AssetKind.PROP
    assert (chair.width, chair.height) == (8, 8)
    assert chair.palette == ["#0a14c8"]
    assert chair.source_prompt == "draw a wooden chair"
    assert chair.service == "fake"
    assert chair.refiner.preset == "crisp-sprite"
    assert chair.grid is None


def test_a_new_manifest_drops_the_fields_only_the_removed_roads_filled(setup, write_asset):
    """Catches: `role`, `position`, `grid`, the two seam scores or a props pitch coming back
    into a new manifest.json (even as null), or the models losing those fields so a manifest
    or sidecar already on disk stops loading. The old tile and the old prop here still
    export as they are, with the fields the kept roads fill."""
    store, direction = setup
    stamp_layout(store, "stone-floor", role=TileRole.CENTRE, position=(0, 0), cell=(8, 8))
    stamp_layout(
        store, "wooden-chair", pitch=4.0, pitch_source=PitchSource.MEASURED, target=(8, 8),
        footprint=(1, 1), anchor=(4, 8),
    )
    bundle = export(store, direction, stamp="s", now=NOW)
    raw = json.loads((bundle.directory / "manifest.json").read_text())
    assert {entry["name"] for entry in raw["entries"]} == {"stone-floor", "wooden-chair"}
    for entry in raw["entries"]:
        assert not {"role", "position", "grid"} & entry.keys()
        assert "centre_seam_with_cohesion" not in entry["cohesion"]
        assert "centre_seam_without_cohesion" not in entry["cohesion"]
    assert raw["pitches"] == {}  # a prop's pitch is on its entry, never a sheet-wide "props"
    chair = next(entry for entry in raw["entries"] if entry["name"] == "wooden-chair")
    assert (chair["pitch"], chair["target"], chair["footprint"], chair["anchor"]) == (
        4.0, [8, 8], [1, 1], [4, 8]
    )

    # A manifest written before this change still loads, fields and all.
    old = Manifest.model_validate_json(
        '{"project": "P", "direction": "d", "exported": "2026-09-03T19:12:33Z", "cell_size": 16,'
        ' "pitches": {"props": {"value": 4.0, "source": "measured"}},'
        ' "entries": [{"name": "floor", "kind": "tile", "width": 8, "height": 8,'
        ' "grid": [8, 8], "role": "centre", "position": [0, 1],'
        ' "cohesion": {"centre_seam_with_cohesion": 0.5}}]}'
    )
    [floor] = old.entries
    assert (floor.grid, floor.role, floor.position) == ((8, 8), TileRole.CENTRE, (0, 1))
    assert floor.cohesion.centre_seam_with_cohesion == 0.5
    assert old.pitches["props"] == Pitch(value=4.0, source=PitchSource.MEASURED)


def test_the_manifest_lists_every_file_it_wrote(setup):
    store, direction = setup
    bundle = export(store, direction, stamp="s", now=NOW)
    manifest = Manifest.model_validate_json((bundle.directory / "manifest.json").read_text())
    chair = next(entry for entry in manifest.entries if entry.name == "wooden-chair")
    assert chair.files["native"] == "wooden-chair.png"
    assert chair.files["x2"] == "wooden-chair@2x.png"
    assert chair.files["x4"] == "wooden-chair@4x.png"


def test_the_manifest_says_which_tool_refined_each_asset(setup):
    store, direction = setup
    bundle = export(store, direction, stamp="s", now=NOW)
    chair = next(entry for entry in bundle.manifest.entries if entry.name == "wooden-chair")
    assert chair.route == "preserve"
    assert chair.refiner_tool == "stub"


def test_a_second_export_in_the_same_second_does_not_clobber_the_first(setup):
    store, direction = setup
    first = export(store, direction, stamp="s", now=NOW)
    second = export(store, direction, stamp="s", now=NOW)
    assert first.directory != second.directory
    assert second.directory.name == "cosy-den-s-2"
    assert (first.directory / "manifest.json").exists()


def test_the_manifest_json_is_readable_by_a_plain_json_reader(setup):
    store, direction = setup
    bundle = export(store, direction, stamp="s", now=NOW)
    raw = json.loads((bundle.directory / "manifest.json").read_text())
    assert raw["entries"][0]["name"] == "stone-floor"


def test_an_asset_that_was_never_refined_is_skipped_with_a_warning(setup):
    """Includes an old prop (no grid rebuild) that refine now refuses: export still says
    "has not been refined yet" and leaves it out, rather than failing."""
    store, direction = setup
    for name in ("half-done", "old-lamp"):
        folder = store.asset_dir("cosy-den", name)
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (4, 4)).save(folder / "source.png")
    store.write_model(
        store.asset_dir("cosy-den", "old-lamp") / "asset.json",
        AssetRecord(name="old-lamp", kind=AssetKind.PROP, source_generation="20260903-191233"),
    )
    bundle = export(store, direction, stamp="s", now=NOW)
    for name in ("half-done", "old-lamp"):
        assert name not in [entry.name for entry in bundle.manifest.entries]
    assert any(
        "half-done" in warning and "old-lamp" in warning and "has not been refined yet" in warning
        for warning in bundle.warnings
    )


def test_a_refused_asset_is_skipped_with_the_refiner_s_reason(setup):
    store, direction = setup
    folder = store.asset_dir("cosy-den", "refused-asset")
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4)).save(folder / "source.png")
    # The realistic shape: the refine stage unlinks refined.png on a refusal,
    # so no PNG exists here — only the report saying why.
    store.write_model(
        folder / "refine.json",
        RefineReport(
            asset="refused-asset",
            refiner="stub",
            error_code="INPUT_NOT_FOUND",
            error_message="source.png does not exist",
            requested=RefinerSettings(preset="crisp-sprite"),
        ),
    )
    bundle = export(store, direction, stamp="s", now=NOW)
    assert "refused-asset" not in [entry.name for entry in bundle.manifest.entries]
    assert any(
        "refused-asset" in warning and "source.png does not exist" in warning
        for warning in bundle.warnings
    )


def test_a_pixelmagic_refusal_is_skipped_without_blaming_the_refiner(setup):
    """`TARGET_NOT_USABLE` is PixelMagic's own refusal — the refiner never saw
    the request — so the warning must say that, not "refused by the refiner",
    which would send an operator looking at the wrong tool."""
    store, direction = setup
    folder = store.asset_dir("cosy-den", "oversized-prop")
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4)).save(folder / "source.png")
    store.write_model(
        folder / "refine.json",
        RefineReport(
            asset="oversized-prop",
            refiner="stub",
            error_code=TARGET_NOT_USABLE,
            error_message="asset 'oversized-prop': target size (2000, 5) is not usable",
            requested=RefinerSettings(preset="crisp-sprite"),
        ),
    )
    bundle = export(store, direction, stamp="s", now=NOW)
    assert "oversized-prop" not in [entry.name for entry in bundle.manifest.entries]
    warning = next(w for w in bundle.warnings if "oversized-prop" in w)
    assert "PixelMagic did not send it to the refiner" in warning
    assert "target size (2000, 5) is not usable" in warning
    assert "refused by the refiner" not in warning


def test_a_refused_asset_with_a_leftover_png_is_skipped_with_a_warning(setup):
    store, direction = setup
    folder = store.asset_dir("cosy-den", "refused-asset")
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4)).save(folder / "source.png")
    # A stale refined.png from an earlier successful run, left behind despite
    # this run's refusal — export must not trust the file's mere presence.
    Image.new("RGBA", (4, 4)).save(folder / "refined.png")
    store.write_model(
        folder / "refine.json",
        RefineReport(
            asset="refused-asset",
            refiner="stub",
            error_code="INPUT_NOT_FOUND",
            error_message="source.png does not exist",
            requested=RefinerSettings(preset="crisp-sprite"),
        ),
    )
    bundle = export(store, direction, stamp="s", now=NOW)
    assert "refused-asset" not in [entry.name for entry in bundle.manifest.entries]
    assert any(
        "refused-asset" in warning and "source.png does not exist" in warning
        for warning in bundle.warnings
    )


def test_an_asset_sidecar_that_will_not_parse_is_skipped_with_a_warning(setup, write_asset):
    store, direction = setup
    write_asset(store, "broken-lamp", AssetKind.PROP)
    (store.asset_dir("cosy-den", "broken-lamp") / "asset.json").write_text('{"name": "bro')
    bundle = export(store, direction, stamp="s", now=NOW)
    assert [entry.name for entry in bundle.manifest.entries] == ["stone-floor", "wooden-chair"]
    assert any(
        "broken-lamp" in warning and "asset.json" in warning for warning in bundle.warnings
    )
    assert not (bundle.directory / "broken-lamp.png").exists()


def test_a_refine_report_that_will_not_parse_is_skipped_with_a_warning(setup, write_asset):
    store, direction = setup
    write_asset(store, "broken-lamp", AssetKind.PROP)
    (store.asset_dir("cosy-den", "broken-lamp") / "refine.json").write_text("{oh no")
    bundle = export(store, direction, stamp="s", now=NOW)
    assert [entry.name for entry in bundle.manifest.entries] == ["stone-floor", "wooden-chair"]
    assert any(
        "broken-lamp" in warning and "refine.json" in warning for warning in bundle.warnings
    )


def test_a_refined_png_that_is_not_an_image_is_skipped_with_a_warning(setup, write_asset):
    store, direction = setup
    write_asset(store, "broken-lamp", AssetKind.PROP)
    # Zero bytes: the file exists, so the "not refined yet" check passes, and
    # the failure only shows up when Pillow tries to read it.
    (store.asset_dir("cosy-den", "broken-lamp") / "refined.png").write_bytes(b"")
    bundle = export(store, direction, stamp="s", now=NOW)
    assert [entry.name for entry in bundle.manifest.entries] == ["stone-floor", "wooden-chair"]
    assert any(
        "broken-lamp" in warning and "refined.png" in warning for warning in bundle.warnings
    )
    assert not (bundle.directory / "broken-lamp.png").exists()


def test_a_png_that_fails_halfway_leaves_nothing_of_that_asset_behind(setup, monkeypatch):
    """The bundle must never hold a file the manifest does not list."""
    store, direction = setup
    real = scale_nearest

    def fail_on_the_upscale(image, factor):
        if factor > 1:
            raise OSError("disk went away")
        return real(image, factor)

    monkeypatch.setattr("pixelmagic.stages.export.scale_nearest", fail_on_the_upscale)
    bundle = export(store, direction, stamp="s", now=NOW)
    assert bundle.manifest.entries == []
    assert sorted(path.name for path in bundle.directory.iterdir()) == ["manifest.json"]
    # Both assets failed the same way, so it is one line naming both.
    [line] = [w for w in bundle.warnings if "refined.png" in w]
    assert "2 assets: stone-floor, wooden-chair" in line


def test_a_save_that_writes_bytes_then_fails_leaves_nothing_behind(setup, monkeypatch):
    """The half-written file itself must go, not just the ones already finished."""
    store, direction = setup
    real_save = Image.Image.save

    def half_write(self, fp, *args, **kwargs):
        # A disk filling up mid-write: some bytes land, then the save fails.
        Path(fp).write_bytes(b"\x89PNG\r\n")
        raise OSError("no space left on device")

    monkeypatch.setattr(Image.Image, "save", half_write)
    bundle = export(store, direction, stamp="s", now=NOW)
    monkeypatch.undo()
    assert Image.Image.save is real_save
    assert bundle.manifest.entries == []
    listed = {name for entry in bundle.manifest.entries for name in entry.files.values()}
    assert {path.name for path in bundle.directory.iterdir()} == listed | {"manifest.json"}


def test_a_refiner_warning_reaches_the_manifest(tmp_path, write_asset):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    write_asset(store, "chair", AssetKind.PROP)
    report = load_report(store, "cosy-den", "chair")
    report.warnings = [LOW_GRID_CONFIDENCE, "BACKGROUND_UNCERTAIN"]
    store.write_model(store.asset_dir("cosy-den", "chair") / "refine.json", report)

    bundle = export(store, direction)

    entry = bundle.manifest.entries[0]
    # Verbatim codes, not a sentence: a game reads this file too.
    assert entry.warnings == [LOW_GRID_CONFIDENCE, "BACKGROUND_UNCERTAIN"]


def test_a_refiner_warning_is_also_said_out_loud(tmp_path, write_asset):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    write_asset(store, "chair", AssetKind.PROP)
    report = load_report(store, "cosy-den", "chair")
    report.warnings = [LOW_GRID_CONFIDENCE]
    store.write_model(store.asset_dir("cosy-den", "chair") / "refine.json", report)

    bundle = export(store, direction)

    said = " ".join(bundle.warnings)
    assert "chair" in said
    assert LOW_GRID_CONFIDENCE in said
    # The one warning a person can act on gets a sentence, not just a code.
    assert "grid" in said
    # And it is a note, not a refusal: the asset is still in the bundle.
    assert [entry.name for entry in bundle.manifest.entries] == ["chair"]


def test_an_asset_with_no_warnings_says_nothing(tmp_path, write_asset):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    write_asset(store, "chair", AssetKind.PROP)
    bundle = export(store, direction)
    # Not `bundle.warnings == []`: this module's autouse `no_aseprite` fixture
    # makes every export in it report the missing-Aseprite line, so an exact
    # equality here can never pass. The claim is about *this* task's sentence.
    assert not [line for line in bundle.warnings if "refined with warnings" in line]
    assert bundle.manifest.entries[0].warnings == []


def test_a_stand_in_bundles_target_disagrees_with_the_scenes_real_size(setup):
    """Locked decision 9: the stub never resizes, so a scene refined through it
    at a target still comes back at its own source size, and the manifest says
    so twice -- `target` disagrees with the real size and `refiner_tool` is
    "stub"."""
    store, direction = setup
    folder = store.asset_dir("cosy-den", "wooden-chair")
    # `setup` staged wooden-chair by hand, with no real refiner run behind it;
    # both files are removed so `refine_asset` can write its own.
    (folder / "refined.png").unlink()
    (folder / "refine.json").unlink()
    stamp_layout(store, "wooden-chair", kind=AssetKind.SCENE, target=(16, 24))

    refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())
    bundle = export(store, direction, stamp="s", now=NOW)

    chair = next(entry for entry in bundle.manifest.entries if entry.name == "wooden-chair")
    assert chair.target == (16, 24)
    assert chair.target != (chair.width, chair.height)
    assert chair.refiner_tool == "stub"


class CohesionReportingRefiner(StubRefiner):
    """A stand-in that claims a chosen `effectiveOptions` for cohesion, so a
    test can prove the manifest copies `refine.json`'s own cohesion record.

    Re-declared locally rather than imported from `test_refine.py`, matching
    this codebase's existing test style (`RecordingRefiner`/
    `BatchRecordingRefiner` each live only in the file that uses them).
    """

    def __init__(self, effective: dict[str, object]):
        self._effective = effective

    def refine(self, request):
        result = super().refine(request)
        result.effective_options = dict(self._effective)
        return result


def test_the_manifest_copies_the_cohesion_record_from_refine_json(setup):
    store, direction = setup
    stamp_layout(store, "wooden-chair", kind=AssetKind.SCENE)
    refine_asset(
        store, direction, "wooden-chair",
        refiner=CohesionReportingRefiner(
            {"cohesion": True, "cohesionFamilies": 14, "cohesionTones": 2, "cleanup": True}
        ),
    )
    bundle = export(store, direction, scales=(1,), stamp="s", now=NOW)
    manifest = Manifest.model_validate_json((bundle.directory / "manifest.json").read_text())
    entry = next(e for e in manifest.entries if e.name == "wooden-chair")
    assert entry.cohesion.cohesion is True
    assert entry.cohesion.cohesion_families == 14


def test_a_pitch_set_by_hand_is_recorded_as_set(setup, write_asset):
    """`set_scene_pitch` rewrites `pitch_source` on the scene, so the manifest reads the
    source straight off the record."""
    store, direction = setup
    write_asset(store, "a-cosy-den", AssetKind.SCENE, size=(24, 24))
    stamp_layout(store, "a-cosy-den", pitch=4.0, pitch_source=PitchSource.SET, target=(6, 6))
    bundle = export(store, direction, stamp="s", now=NOW)
    assert bundle.manifest.pitches["scene"] == Pitch(value=4.0, source=PitchSource.SET)


def test_scenes_from_two_generations_that_disagree_record_the_newest_and_say_so(
    setup, write_asset
):
    store, direction = setup
    write_asset(store, "the-den", AssetKind.SCENE, size=(24, 24))
    write_asset(store, "the-yard", AssetKind.SCENE, size=(24, 24))
    # The den came from the older generation (ids are timestamp slugs).
    stamp_layout(
        store, "the-den", pitch=4.0, pitch_source=PitchSource.MEASURED, target=(6, 6),
        source_generation="20260901-100000",
    )
    stamp_layout(
        store, "the-yard", pitch=5.0, pitch_source=PitchSource.SET, target=(5, 5),
        source_generation="20260903-191233",
    )
    bundle = export(store, direction, stamp="s", now=NOW)
    assert bundle.manifest.pitches["scene"] == Pitch(value=5.0, source=PitchSource.SET)
    assert (
        "the-den: has pitch 4.0, not the 5.0 the manifest records for the scene"
        in bundle.warnings
    )


def test_a_tie_on_source_generation_is_broken_by_asset_name(setup, write_asset):
    """Two scenes kept from one generation can still disagree on pitch (one set by hand).
    `max` cannot use age to settle that, so the tie goes to whichever asset name sorts first
    (`the-den` before `the-yard`)."""
    store, direction = setup
    write_asset(store, "the-den", AssetKind.SCENE, size=(24, 24))
    write_asset(store, "the-yard", AssetKind.SCENE, size=(24, 24))
    stamp_layout(
        store, "the-den", pitch=5.0, pitch_source=PitchSource.SET, target=(5, 5),
        source_generation="20260903-191233",
    )
    stamp_layout(
        store, "the-yard", pitch=4.0, pitch_source=PitchSource.MEASURED, target=(6, 6),
        source_generation="20260903-191233",
    )
    bundle = export(store, direction, stamp="s", now=NOW)
    assert bundle.manifest.pitches["scene"] == Pitch(value=5.0, source=PitchSource.SET)
    assert (
        "the-yard: has pitch 4.0, not the 5.0 the manifest records for the scene"
        in bundle.warnings
    )


def test_the_scene_pitch_is_the_kept_scene_s(setup, write_asset):
    store, direction = setup
    write_asset(store, "a-cosy-den", AssetKind.SCENE, size=(24, 24))
    stamp_layout(store, "a-cosy-den", pitch=3.0, pitch_source=PitchSource.MEASURED, target=(8, 8))
    bundle = export(store, direction, stamp="s", now=NOW)
    assert bundle.manifest.pitches["scene"] == Pitch(value=3.0, source=PitchSource.MEASURED)


def test_assets_without_a_pitch_leave_pitches_empty(setup):
    """M2 assets: nothing to say, so nothing is said."""
    store, direction = setup
    bundle = export(store, direction, stamp="s", now=NOW)
    assert bundle.manifest.pitches == {}


def test_a_skipped_asset_does_not_set_the_manifest_s_pitch(setup, write_asset):
    store, direction = setup
    write_asset(store, "broken-lamp", AssetKind.SCENE)
    stamp_layout(store, "broken-lamp", pitch=9.0, pitch_source=PitchSource.MEASURED, target=(1, 1))
    (store.asset_dir("cosy-den", "broken-lamp") / "refined.png").write_bytes(b"")
    bundle = export(store, direction, stamp="s", now=NOW)
    assert "broken-lamp" not in [entry.name for entry in bundle.manifest.entries]
    assert bundle.manifest.pitches == {}


def test_a_pitch_with_unknown_source_is_left_out_of_the_scene_pitch(setup, write_asset):
    """PM course correction on #27: a hand-edited sidecar could carry a pitch
    with no `pitch_source`; `Pitch.source` is required, so the manifest omits
    the key entirely rather than guessing a source, and says so in a note.
    The entry itself still carries the pitch, with `pitch_source` null.

    A second, older scene that disagrees with the unsourced winner must not be
    told the manifest recorded a pitch it did not (Task 4 fix, #27).
    """
    store, direction = setup
    write_asset(store, "the-den", AssetKind.SCENE, size=(24, 24))
    write_asset(store, "the-yard", AssetKind.SCENE, size=(24, 24))
    stamp_layout(store, "the-yard", pitch=4.0, pitch_source=None, target=(6, 6))
    stamp_layout(
        store, "the-den", pitch=5.0, pitch_source=PitchSource.MEASURED, target=(5, 5),
        source_generation="20260901-100000",
    )
    bundle = export(store, direction, stamp="s", now=NOW)
    assert "scene" not in bundle.manifest.pitches
    assert (
        "the-yard: has pitch 4.0 but its source is unknown, so the manifest "
        "records no scene pitch"
    ) in bundle.warnings
    assert (
        "the-den: has pitch 5.0; the manifest records no scene pitch, since "
        "the-yard's source is unknown"
    ) in bundle.warnings
    yard = next(entry for entry in bundle.manifest.entries if entry.name == "the-yard")
    assert yard.pitch == 4.0
    assert yard.pitch_source is None


def test_one_note_keeps_the_name_colon_text_shape():
    assert collapse_notes([("wooden-chair", "has not been refined yet — skipped")]) == [
        "wooden-chair: has not been refined yet — skipped"
    ]


def test_forty_three_identical_notes_become_one_line_naming_three():
    notes = [(f"tile-{i:02d}", "was refined with warnings: SEAM_VISIBLE") for i in range(43)]
    assert collapse_notes(notes) == [
        "was refined with warnings: SEAM_VISIBLE — 43 assets: tile-00, tile-01, tile-02, …"
    ]


def test_two_or_three_assets_are_all_named_with_no_ellipsis():
    assert collapse_notes([("a", "x"), ("b", "x")]) == ["x — 2 assets: a, b"]
    assert collapse_notes([("a", "x"), ("b", "x"), ("c", "x")]) == ["x — 3 assets: a, b, c"]


def test_distinct_notes_stay_separate_lines_in_first_appearance_order():
    notes = [
        ("wooden-chair", "has not been refined yet — skipped"),
        ("stone-floor", "was refined with warnings: SEAM_VISIBLE"),
        ("round-rug", "has not been refined yet — skipped"),
    ]
    assert collapse_notes(notes) == [
        "has not been refined yet — skipped — 2 assets: wooden-chair, round-rug",
        "stone-floor: was refined with warnings: SEAM_VISIBLE",
    ]


def test_a_seam_warning_gets_a_sentence_a_person_can_act_on(setup):
    store, direction = setup
    report = load_report(store, "cosy-den", "stone-floor")
    report.warnings = [SEAM_VISIBLE]
    store.write_model(store.asset_dir("cosy-den", "stone-floor") / "refine.json", report)
    bundle = export(store, direction, stamp="s", now=NOW)
    assert (
        "stone-floor: was refined with warnings: SEAM_VISIBLE — "
        "this tile's edges do not meet; laid in a row it will show a seam"
    ) in bundle.warnings
    # The code itself stays verbatim in the manifest: a game reads that file.
    floor = next(entry for entry in bundle.manifest.entries if entry.name == "stone-floor")
    assert floor.warnings == [SEAM_VISIBLE]


def test_forty_three_seamed_tiles_are_one_line_of_the_export(tmp_path, write_asset):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    for index in range(43):
        name = f"tile-{index:02d}"
        write_asset(store, name, AssetKind.TILE)
        report = load_report(store, "cosy-den", name)
        report.warnings = [SEAM_VISIBLE]
        store.write_model(store.asset_dir("cosy-den", name) / "refine.json", report)
    bundle = export(store, direction, scales=(1,), stamp="s", now=NOW)
    seam_lines = [line for line in bundle.warnings if SEAM_VISIBLE in line]
    assert seam_lines == [
        "was refined with warnings: SEAM_VISIBLE — this tile's edges do not meet; "
        "laid in a row it will show a seam — 43 assets: tile-00, tile-01, tile-02, …"
    ]
    assert len(bundle.manifest.entries) == 43


def test_two_different_notes_stay_two_lines_of_the_export(setup):
    store, direction = setup
    # The chair is not refined; the floor has a seam. Two texts, two lines.
    (store.asset_dir("cosy-den", "wooden-chair") / "refine.json").unlink()
    report = load_report(store, "cosy-den", "stone-floor")
    report.warnings = [SEAM_VISIBLE]
    store.write_model(store.asset_dir("cosy-den", "stone-floor") / "refine.json", report)
    bundle = export(store, direction, stamp="s", now=NOW)
    asset_lines = [line for line in bundle.warnings if "Aseprite was not found" not in line]
    assert asset_lines == [
        "stone-floor: was refined with warnings: SEAM_VISIBLE — "
        "this tile's edges do not meet; laid in a row it will show a seam",
        "wooden-chair: has not been refined yet — skipped",
    ]


def test_the_aseprite_line_comes_first_and_is_never_folded(setup):
    store, direction = setup
    # Give the bundle a second, per-asset warning line so "comes first" is an
    # actual ordering claim rather than the only line there is.
    (store.asset_dir("cosy-den", "wooden-chair") / "refine.json").unlink()
    bundle = export(store, direction, stamp="s", now=NOW)
    assert bundle.warnings[0].startswith("Aseprite was not found")
    assert bundle.warnings[1:] == ["wooden-chair: has not been refined yet — skipped"]
