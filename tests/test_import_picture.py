"""A picture brought in from disk goes through the pipeline like a drawn one."""

import json
from datetime import UTC, datetime

import pytest
from PIL import Image

from pixelmagic.model import Direction, Manifest, Reviewer
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.stages.decompose import promote_cutouts, promote_generation
from pixelmagic.stages.export import export
from pixelmagic.stages.generate import MAX_IMAGE_SIZE, import_picture, load_generation
from pixelmagic.stages.pick import pick_box
from pixelmagic.stages.redraw import redraw_picks
from pixelmagic.stages.refine import refine_direction
from pixelmagic.stages.suggest import suggest_objects
from pixelmagic.store import Store

NOW = datetime(2026, 9, 30, 9, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_aseprite(monkeypatch):
    """Do not touch whatever Aseprite the machine happens to have."""
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None)


def _direction(tmp_path):
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    return store, direction


def test_an_imported_jpeg_is_picked_redrawn_kept_and_exported_without_the_original(tmp_path):
    store, direction = _direction(tmp_path)
    original = tmp_path / "My Den.jpg"
    Image.new("RGB", (96, 64), (200, 120, 60)).save(original)

    record = import_picture(store, direction, original, reviewer=Reviewer.HUMAN, now=NOW)
    original.unlink()  # nothing may depend on the file it came from

    on_disk = load_generation(store, direction.name, record.request.id)
    with Image.open(on_disk.image_path) as stored:
        assert stored.format == "PNG" and stored.size == (96, 64)
    request = json.loads((on_disk.image_path.parent / "request.json").read_text())
    assert request["imported_from"] == "My Den.jpg"
    assert request["subject"] == "My Den"
    assert not request["prompt"] and not request["service"] and not request["background"]
    assert on_disk.review.status.value == "approved"
    assert on_disk.review.reviewer is Reviewer.HUMAN

    # Kept whole as a scene: no prompt or service to record, and no crash.
    promote_generation(store, direction, on_disk, refiner=StubRefiner())
    # Picked and redrawn: the redraw is drawn, so its pieces keep a real prompt/service.
    suggested = suggest_objects(store, direction, on_disk, StubSegmenter(), now=NOW)
    pick_box(
        store, direction, suggested.id, (0, 0, 96, 64), name="lamp", segmenter=StubSegmenter()
    )
    pieces = redraw_picks(store, direction, suggested.id, service=FakeImageService(), now=NOW)
    promote_cutouts(store, direction, pieces)
    refine_direction(store, direction, refiner=StubRefiner())
    bundle = export(store, direction, scales=(1,), stamp="s", now=NOW)

    manifest = Manifest.model_validate_json((bundle.directory / "manifest.json").read_text())
    scene = next(entry for entry in manifest.entries if entry.kind.value == "scene")
    assert scene.source_prompt is None and scene.service is None
    props = [entry for entry in manifest.entries if entry.kind.value == "prop"]
    assert props and all(entry.source_prompt and entry.service == "fake" for entry in props)


@pytest.mark.parametrize("kind", ["text", "too-big"])
def test_a_refused_file_is_named_and_leaves_nothing_behind(tmp_path, kind):
    store, direction = _direction(tmp_path)
    refused = tmp_path / "notes.png"
    if kind == "text":
        refused.write_text("this is not a picture")
    else:
        Image.new("1", (MAX_IMAGE_SIZE + 1, 1)).save(refused)

    with pytest.raises(ValueError, match=r"notes\.png"):
        import_picture(store, direction, refused, reviewer=Reviewer.HUMAN, now=NOW)

    assert not list(store.generations_dir(direction.name).glob("*"))
