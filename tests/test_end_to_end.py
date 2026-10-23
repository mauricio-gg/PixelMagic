"""The whole pipeline, offline, through the library rather than the CLI."""

from datetime import UTC, datetime

import pytest
from PIL import Image

from pixelmagic.model import (
    AssetKind,
    Direction,
    Manifest,
    RefinerSettings,
    Reviewer,
)
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.stages.decompose import promote_cutouts, promote_generation
from pixelmagic.stages.export import export
from pixelmagic.stages.generate import import_picture
from pixelmagic.stages.pick import pick_suggestions
from pixelmagic.stages.redraw import redraw_picks
from pixelmagic.stages.refine import REBUILD_REFINER, REBUILD_ROUTE, refine_direction
from pixelmagic.stages.suggest import suggest_objects
from pixelmagic.store import Store

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_aseprite(monkeypatch):
    """Do not touch whatever Aseprite the machine happens to have."""
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None)


def build_cosy_den(tmp_path, object_picture, *, picks, scales):
    """Walk the README's whole chain offline, on the object road, and return what it made.

    Import the picture, keep it as the scene, suggest, pick `picks`, redraw,
    promote, refine and export, all with the fake service, the stand-in
    refiner and the stand-in segmenter. Returns the store, the direction, the
    scene's generation, the suggestions (they are the decomposition) and the bundle.
    """
    store = Store(tmp_path / "project")
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(
            name="cosy-den",
            created=NOW,
            style="Warm SNES interiors, soft lamp light, chunky outlines.",
            refiner_defaults=RefinerSettings(preset="crisp-sprite"),
        )
    )
    scene = import_picture(store, direction, object_picture, reviewer=Reviewer.HUMAN, now=NOW)
    # Locked decision 3: scenes are a v1 asset kind, so the picture is kept whole too.
    assert promote_generation(store, direction, scene, refiner=StubRefiner()) == "objects"
    suggested = suggest_objects(store, direction, scene, StubSegmenter(), now=NOW)
    pick_suggestions(store, direction, suggested.id, picks)
    redrawn = redraw_picks(
        store, direction, suggested.id,
        service=FakeImageService(), now=NOW,
    )
    promote_cutouts(store, direction, redrawn)
    refine_direction(store, direction, refiner=StubRefiner())
    bundle = export(store, direction, scales=scales, stamp="s", now=NOW)
    return store, direction, scene, suggested, bundle


def test_picked_run_offline_to_a_bundle(tmp_path, object_picture):
    """The README's walk: a bundle holds the kept scene and the rebuilt props, and says so.

    Catches: a prop that reaches the bundle through the stand-in instead of
    the grid rebuild, a manifest that lists a file the bundle lacks (or the
    reverse), a lost scene, and a missing 2x picture.
    """
    # Seven of the eight objects: two redraw groups, of four and three.
    _, direction, _, _, bundle = build_cosy_den(
        tmp_path, object_picture, picks=[1, 2, 3, 4, 5, 6, 7], scales=(1, 2)
    )
    manifest = Manifest.model_validate_json((bundle.directory / "manifest.json").read_text())

    assert manifest.cell_size == direction.cell_size
    props = [entry for entry in manifest.entries if entry.kind is AssetKind.PROP]
    assert len(props) == 7
    for entry in props:
        assert entry.segmenter == "stub"
        assert entry.refiner_tool == REBUILD_REFINER
        assert entry.route == REBUILD_ROUTE
        assert entry.target is not None and entry.footprint is not None
        assert entry.anchor is not None
    # A rebuilt piece carries its own pitch, so no sheet-wide props pitch is recorded.
    assert "props" not in manifest.pitches
    # Only props are rebuilt: the kept scene went through the refiner (here the stub).
    (scene_entry,) = [entry for entry in manifest.entries if entry.kind is AssetKind.SCENE]
    assert scene_entry.refiner_tool == "stub"
    assert scene_entry.route != REBUILD_ROUTE

    for entry in manifest.entries:
        doubled = bundle.directory / entry.files["x2"]
        assert Image.open(doubled).size == (entry.width * 2, entry.height * 2)

    listed = {"manifest.json"} | {
        path for entry in manifest.entries for path in entry.files.values()
    }
    on_disk = {
        str(path.relative_to(bundle.directory))
        for path in bundle.directory.rglob("*") if path.is_file()
    }
    assert on_disk == listed


def test_every_stage_left_a_sidecar_behind(tmp_path, object_picture):
    """The invariant CLAUDE.md states: every stage writes its JSON sidecar.

    Catches: a stage that stops recording its work, which makes it impossible
    to re-run and leaves the tool with no memory of it.
    """
    store, _, scene, suggested, bundle = build_cosy_den(
        tmp_path, object_picture, picks=[1, 2, 3], scales=(1,)
    )

    generation = store.generation_dir("cosy-den", scene.request.id)
    assert {"request.json", "result.json", "review.json"} <= {
        path.name for path in generation.iterdir()
    }
    decomposition = store.decomposition_dir("cosy-den", suggested.id)
    assert (decomposition / "decomposition.json").exists()
    assert (store.picks_dir("cosy-den", suggested.id) / "picks.json").exists()
    assets = store.list_assets("cosy-den")
    assert len(assets) == 4  # the kept scene and three props
    for name in assets:
        asset = store.asset_dir("cosy-den", name)
        assert (asset / "asset.json").exists()
        assert (asset / "refine.json").exists()
    cut = [
        name for name in assets
        if (store.cutout_dir("cosy-den", suggested.id, name) / "cutout.json").exists()
    ]
    assert len(cut) == 3
    assert (bundle.directory / "manifest.json").exists()
