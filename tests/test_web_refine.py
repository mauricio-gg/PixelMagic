from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    GridRebuild,
    Manifest,
    PitchSource,
    RefineReport,
    TileRole,
)
from pixelmagic.refiner.base import LOW_GRID_CONFIDENCE
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.stages.refine import load_report
from pixelmagic.stages.slice import SEAM_VISIBLE
from pixelmagic.store import Store
from pixelmagic.web.app import create_app
from pixelmagic.web.refining import asset_rows


@pytest.fixture(autouse=True)
def no_aseprite(monkeypatch):
    """Keep this module's exports off any Aseprite the machine happens to have.

    Every test here that calls `POST .../export` goes through the same
    `stages.export.export` the CLI and the end-to-end tests use, and that
    function reaches for a real Aseprite install when none is stubbed in.
    These tests are about the bundle's PNGs and manifest, not about driving a
    GUI app in batch mode, so `find_aseprite` is patched the same way
    `tests/test_export.py`, `tests/test_cli.py` and `tests/test_end_to_end.py`
    already do.
    """
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None)


def with_assets(client, project_root, cut_props, *, scene_size=120):
    """Draw and keep a scene, then pick, redraw and keep three props: assets to refine.

    The scene (`cosy-den`) is the one asset the refiner works on; the three
    props are rebuilt on their own pixel grid and are only copied through.
    """
    client.post(
        "/directions/cosy-den/generate",
        data={"subject": "a cosy den", "kind": "scene",
              "background": "transparent", "size": str(scene_size)},
    )
    store = Store(project_root)
    generation_id = store.list_generations("cosy-den")[0]
    client.post(
        f"/directions/cosy-den/generations/{generation_id}/approve",
        data={"keep_reference": "1"},
    )
    client.post(f"/directions/cosy-den/generations/{generation_id}/keep-scene")
    decomposition_id = cut_props(store)
    client.post(f"/directions/cosy-den/decompositions/{decomposition_id}/pick/continue")
    return store


def a_prop(store):
    """The name and record of one asset that has an anchor — a prop."""
    for name in store.list_assets("cosy-den"):
        record = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
        if record.anchor is not None:
            return name, record
    raise AssertionError("the fake pipeline made no prop with an anchor")


def with_scene_asset(client, project_root, cut_props, *, scene_size=200):
    """A direction with a kept scene alongside its props, for the scene-pitch tests."""
    return with_assets(client, project_root, cut_props, scene_size=scene_size)


def a_scene(store):
    """The name and record of the one kept scene."""
    for name in store.list_assets("cosy-den"):
        record = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
        if record.kind is AssetKind.SCENE:
            return name, record
    raise AssertionError("with_scene_asset made no scene")


def add_old_refined_tile(store, name):
    """Stage an old bordered centre tile, already refined, whose receipt carries `SEAM_VISIBLE`."""
    folder = store.asset_dir("cosy-den", name)
    folder.mkdir(parents=True, exist_ok=True)
    pixels = np.full((4, 4, 4), 255, dtype=np.uint8)
    pixels[:, 0, :3] = 0
    Image.fromarray(pixels, "RGBA").save(folder / "source.png")
    Image.fromarray(pixels, "RGBA").save(folder / "refined.png")
    store.write_model(
        folder / "asset.json",
        AssetRecord(
            name=name, kind=AssetKind.TILE, source_generation="20260903-191233",
            source_prompt="a tile sheet", service="fake", target=(4, 4), role=TileRole.CENTRE,
        ),
    )
    store.write_model(
        folder / "refine.json",
        RefineReport(asset=name, refiner="stub", route="preserve", warnings=[SEAM_VISIBLE]),
    )


def test_the_settings_screen_is_built_from_the_real_refiners_answer(
    client, project_root, cut_props
):
    with_assets(client, project_root, cut_props)
    body = client.get("/directions/cosy-den/refine").text
    # Palettes and a quick vocabulary the stand-in does not have: if the screen
    # had asked the stand-in, none of these could be on the page.
    assert "gb_legacy" in body
    assert "pico8" in body
    assert 'name="quick.dithering"' in body
    assert 'name="advanced.colorCount"' in body


def test_a_refiner_that_is_not_installed_says_so_instead_of_guessing(
    project_root, broken_options
):
    application = create_app(
        project_root, service="fake", refiner=StubRefiner(), options_source=broken_options
    )
    with TestClient(application) as offline:
        offline.post(
            "/directions/cosy-den/generate",
            data={"subject": "a cosy den", "kind": "scene",
                  "background": "transparent", "size": "120"},
        )
        body = offline.get("/directions/cosy-den/refine").text
    assert "not installed" in body
    # The stand-in's presets must never be offered as if they were the real
    # tool's, so a screen with no vocabulary offers no vocabulary at all.
    assert 'name="quick.dithering"' not in body


def test_the_stand_in_refiner_says_out_loud_that_it_changes_nothing(
    client, project_root, cut_props
):
    with_assets(client, project_root, cut_props)
    assert "not real pixel art" in client.get("/directions/cosy-den/refine").text


def test_the_stand_in_warning_tracks_the_stubs_real_name_not_a_literal(
    project_root, monkeypatch, canned_options
):
    """Guardian minor 4: `refine.html` used to compare against a bare "stub"
    string literal. If `StubRefiner.name` is ever renamed, the yellow warning
    box must keep showing rather than silently vanish, so the route now
    compares against the class's own `name` attribute."""
    monkeypatch.setattr(StubRefiner, "name", "renamed-stub")
    application = create_app(
        project_root, service="fake", refiner=StubRefiner(), options_source=canned_options,
    )
    with TestClient(application) as renamed:
        body = renamed.get("/directions/cosy-den/refine").text
    assert "not real pixel art" in body


def test_refining_the_whole_direction_writes_a_receipt_for_every_asset(
    client, project_root, cut_props
):
    store = with_assets(client, project_root, cut_props)
    response = client.post(
        "/directions/cosy-den/refine",
        data={"preset": "crisp-sprite", "quick.dithering": "off"},
    )
    assert response.status_code == 200
    for name in store.list_assets("cosy-den"):
        report = load_report(store, "cosy-den", name)
        # The scene goes through the refiner (the stand-in here); the props were
        # rebuilt on their own grid and are copied through.
        assert report.refiner == ("stub" if name == "cosy-den" else "pixelmagic-grid-rebuild")
        assert (store.asset_dir("cosy-den", name) / "refined.png").exists()


def test_settings_with_no_asset_named_become_the_directions_defaults(
    client, project_root, cut_props
):
    store = with_assets(client, project_root, cut_props)
    client.post(
        "/directions/cosy-den/refine",
        data={"preset": "crisp-sprite", "quick.dithering": "off"},
    )
    defaults = store.load_direction("cosy-den").refiner_defaults
    assert defaults.preset == "crisp-sprite"
    assert defaults.quick == {"dithering": "off"}


def test_settings_for_one_asset_leave_the_directions_defaults_alone(
    client, project_root, cut_props
):
    store = with_assets(client, project_root, cut_props)
    name = store.list_assets("cosy-den")[0]
    client.post(
        "/directions/cosy-den/refine", data={"asset": name, "preset": "retro-game"}
    )
    assert store.load_direction("cosy-den").refiner_defaults.preset is None
    assert load_report(store, "cosy-den", name).requested.preset == "retro-game"


def test_a_blank_field_leaves_that_setting_alone(client, project_root, cut_props):
    store = with_assets(client, project_root, cut_props)
    client.post(
        "/directions/cosy-den/refine",
        data={"preset": "crisp-sprite", "quick.dithering": ""},
    )
    defaults = store.load_direction("cosy-den").refiner_defaults
    assert defaults.quick == {}


def test_settings_the_refiner_refuses_are_a_page_and_not_a_traceback(
    client, project_root, cut_props
):
    with_assets(client, project_root, cut_props)
    response = client.post(
        "/directions/cosy-den/refine", data={"quick.background": "pick"}
    )
    assert response.status_code == 400
    assert "backgroundColor" in response.text
    assert "Traceback" not in response.text


def test_an_asset_this_direction_does_not_have_is_a_page_and_not_a_receipt(
    client, project_root, cut_props
):
    """A typo deserves a sentence; only a source.png that vanished deserves an
    INPUT_NOT_FOUND receipt."""
    with_assets(client, project_root, cut_props)
    response = client.post("/directions/cosy-den/refine", data={"asset": "sofa"})
    assert response.status_code == 400
    assert "no asset called" in response.text
    assert "Traceback" not in response.text


def test_before_and_after_are_both_fetchable(client, project_root, cut_props):
    store = with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    name = store.list_assets("cosy-den")[0]
    for which in ("source", "refined"):
        picture = client.get(f"/directions/cosy-den/assets/{name}/{which}.png")
        assert picture.status_code == 200
        assert picture.headers["content-type"] == "image/png"


def test_exporting_writes_a_bundle_and_says_where_it_went(client, project_root, cut_props):
    with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    response = client.post("/directions/cosy-den/export", data={"scale": ["2"]})
    assert response.status_code == 200
    bundles = sorted((project_root / "exports").iterdir())
    assert len(bundles) == 1
    assert bundles[0].name in response.text
    manifest = Manifest.model_validate_json(
        (bundles[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.entries
    # Native is always written, whether or not the box was ticked: a bundle
    # with only doubled pixels in it is not a bundle a game can use.
    for entry in manifest.entries:
        assert "native" in entry.files
        assert "x2" in entry.files


def test_refreshing_after_an_export_does_not_write_a_second_bundle(client, project_root, cut_props):
    """Post, redirect, get: the browser's reload button re-runs the GET, not the POST."""
    with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    posted = client.post(
        "/directions/cosy-den/export", data={"scale": ["2"]}, follow_redirects=False
    )
    assert posted.status_code == 303
    landing = posted.headers["location"]
    client.get(landing)
    client.get(landing)
    assert len(sorted((project_root / "exports").iterdir())) == 1


def test_an_export_with_no_scales_at_all_writes_the_native_size(client, project_root, cut_props):
    """Unticking every box asks for native only, which is not the same as not asking."""
    with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    posted = client.post(
        "/directions/cosy-den/export", data={}, follow_redirects=False
    )
    assert posted.status_code == 303
    assert len(sorted((project_root / "exports").iterdir())) == 1


def test_the_export_screen_names_the_bundle_it_was_sent_to(client, project_root, cut_props):
    with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    posted = client.post(
        "/directions/cosy-den/export", data={"scale": ["2"]}, follow_redirects=False
    )
    name = parse_qs(urlparse(posted.headers["location"]).query)["bundle"][0]
    assert (project_root / "exports" / name).is_dir()
    assert name in client.get(posted.headers["location"]).text


def test_a_scale_of_zero_is_a_400_page_and_writes_no_bundle(client, project_root, cut_props):
    """Guardian I1: the form used to accept `scale=0` and write an empty
    bundle whose page then blamed the refiner for it."""
    with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    response = client.post("/directions/cosy-den/export", data={"scale": ["0"]})
    assert response.status_code == 400
    assert not (project_root / "exports").exists()


def test_a_scale_too_large_is_a_400_page_and_writes_no_bundle(client, project_root, cut_props):
    """Guardian I1: `scale=999` used to reach Pillow and could exhaust memory."""
    with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    response = client.post("/directions/cosy-den/export", data={"scale": ["999"]})
    assert response.status_code == 400
    assert not (project_root / "exports").exists()


def test_exporting_a_direction_with_nothing_refined_says_what_is_missing(
    client, project_root, cut_props
):
    with_assets(client, project_root, cut_props)
    response = client.post("/directions/cosy-den/export", data={"scale": ["1"]})
    assert response.status_code == 200
    assert "not been refined" in response.text or "no assets" in response.text


def test_a_refined_prop_offers_its_anchor_as_a_click(client, project_root, cut_props):
    store = with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    name, record = a_prop(store)
    body = client.get("/directions/cosy-den/refine").text
    assert f'id="anchor-{name}"' in body
    assert f'data-anchor-for="anchor-{name}"' in body
    assert f'data-target-w="{record.target[0]}"' in body
    assert 'name="x"' in body and 'name="y"' in body
    assert "/static/refine.js" in body


def test_an_unrefined_prop_offers_no_anchor_yet(client, project_root, cut_props):
    """The click lands on the refined picture, which does not exist yet."""
    store = with_assets(client, project_root, cut_props)
    name, _ = a_prop(store)
    assert f'data-anchor-for="anchor-{name}"' not in client.get("/directions/cosy-den/refine").text


def test_setting_the_anchor_rewrites_the_asset(client, project_root, cut_props):
    store = with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    name, before = a_prop(store)
    assert before.anchor == (before.target[0] // 2, before.target[1])
    response = client.post(
        "/directions/cosy-den/refine/anchor", data={"asset": name, "x": "5", "y": "5"}
    )
    assert response.status_code == 200
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.anchor == (5, 5)
    assert "(5, 5)" in response.text


def test_an_anchor_outside_the_target_is_the_will_not_work_page(client, project_root, cut_props):
    """`check_anchor` raises `ValueError`; `web.app` shows "That will not work"."""
    store = with_assets(client, project_root, cut_props)
    client.post("/directions/cosy-den/refine", data={})
    name, before = a_prop(store)
    response = client.post(
        "/directions/cosy-den/refine/anchor", data={"asset": name, "x": "99", "y": "99"}
    )
    assert response.status_code == 400
    assert "That will not work" in response.text
    assert "Traceback" not in response.text
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.anchor == before.anchor


def test_a_negative_anchor_is_a_page_and_not_a_traceback(client, project_root, cut_props):
    store = with_assets(client, project_root, cut_props)
    name, _ = a_prop(store)
    response = client.post(
        "/directions/cosy-den/refine/anchor", data={"asset": name, "x": "-1", "y": "0"}
    )
    assert response.status_code == 400
    assert "not a value this page takes" in response.text


def test_anchoring_an_asset_nobody_has_is_a_page(client, project_root, cut_props):
    with_assets(client, project_root, cut_props)
    response = client.post(
        "/directions/cosy-den/refine/anchor", data={"asset": "sofa", "x": "1", "y": "1"}
    )
    assert response.status_code == 400
    assert "no asset called" in response.text
    assert "Traceback" not in response.text


def test_a_kept_scene_offers_its_pitch_on_the_refine_screen(client, project_root, cut_props):
    store = with_scene_asset(client, project_root, cut_props, scene_size=200)
    name, record = a_scene(store)
    body = client.get("/directions/cosy-den/refine").text
    assert f"For {name}, one art pixel is" in body
    assert "1.00" in body and "fallback" in body
    assert 'action="/directions/cosy-den/refine/scene-pitch"' in body
    assert 'name="value"' in body
    # Locked decision 6: a scene never gets an anchor or a footprint.
    assert f'data-anchor-for="anchor-{name}"' not in body


def test_setting_the_scene_pitch_rewrites_the_asset(client, project_root, cut_props):
    store = with_scene_asset(client, project_root, cut_props, scene_size=200)
    name, before = a_scene(store)
    assert before.target == (200, 200)
    response = client.post(
        "/directions/cosy-den/refine/scene-pitch", data={"asset": name, "value": "2.0"}
    )
    assert response.status_code == 200
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.pitch == 2.0
    assert after.pitch_source is PitchSource.SET
    assert after.target == (100, 100)
    # "set" alone is not a proof — the page always has "Set the pitch" on
    # its button. The rendered source sentence names where the number came
    # from, in parentheses, and that is what must say "set" once it is.
    assert "2.00" in response.text
    assert "source pixels across (set)" in response.text


def test_a_scene_pitch_outside_the_bound_is_the_will_not_work_page(client, project_root, cut_props):
    """`check_pitch` raises `ValueError`; `web.app` shows "That will not work"."""
    store = with_scene_asset(client, project_root, cut_props, scene_size=200)
    name, before = a_scene(store)
    response = client.post(
        "/directions/cosy-den/refine/scene-pitch", data={"asset": name, "value": "0.1"}
    )
    assert response.status_code == 400
    assert "That will not work" in response.text
    assert "Traceback" not in response.text
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.pitch == before.pitch


def test_a_scene_pitch_that_is_not_a_number_is_a_page_and_not_a_traceback(
    client, project_root, cut_props
):
    store = with_scene_asset(client, project_root, cut_props, scene_size=200)
    name, _ = a_scene(store)
    response = client.post(
        "/directions/cosy-den/refine/scene-pitch", data={"asset": name, "value": "lots"}
    )
    assert response.status_code == 400
    assert "not a value this page takes" in response.text
    assert "Traceback" not in response.text


def test_setting_the_pitch_of_a_prop_asset_is_refused_by_cause(client, project_root, cut_props):
    """A rebuilt prop's pitch is its own, not this route's — `set_scene_pitch` refuses it
    by kind, in its own words, not silently."""
    store = with_scene_asset(client, project_root, cut_props, scene_size=200)
    prop_name, _ = a_prop(store)
    response = client.post(
        "/directions/cosy-den/refine/scene-pitch", data={"asset": prop_name, "value": "2.0"}
    )
    assert response.status_code == 400
    assert "not a scene" in response.text
    assert "Traceback" not in response.text


def test_a_scene_pitch_for_an_asset_nobody_has_is_a_page(client, project_root, cut_props):
    with_scene_asset(client, project_root, cut_props, scene_size=200)
    response = client.post(
        "/directions/cosy-den/refine/scene-pitch", data={"asset": "nope", "value": "2.0"}
    )
    assert response.status_code == 400
    assert "no asset called" in response.text
    assert "Traceback" not in response.text


def test_a_scene_too_big_to_refine_shows_the_refusal_and_a_way_to_fix_it(
    client, project_root, cut_props, monkeypatch
):
    """The 2026-09-23 amendment: with a refiner that honours a forced size (the
    real one; the stand-in is made to claim it here), a scene whose pitch falls
    back to 1.0 on a big picture refuses to refine at all (`TARGET_NOT_USABLE`).
    The refine screen keeps that refusal in the core's own words and adds the
    one sentence that says what to do about it."""
    monkeypatch.setattr(StubRefiner, "honours_forced_size", True)
    store = with_scene_asset(client, project_root, cut_props, scene_size=1200)
    name, _ = a_scene(store)
    client.post("/directions/cosy-den/refine", data={})
    body = client.get("/directions/cosy-den/refine").text
    assert "PixelMagic did not send it to the refiner" in body
    assert "Set the pitch under More options, then refine it again." in body
    assert 'action="/directions/cosy-den/refine/scene-pitch"' in body


def test_the_warnings_line_shows_plain_sentences_not_raw_codes(client, project_root):
    """Guardian: the refine screen used to print `SEAM_VISIBLE` verbatim.
    `ACTIONABLE_WARNINGS` in `stages.export` is the one place that sentence
    is written, and this screen must reuse it, not repeat it. The code is only
    in old `refine.json` files now, which must still read."""
    store = Store(project_root)
    add_old_refined_tile(store, "floor-centre")
    body = client.get("/directions/cosy-den/refine").text
    assert "SEAM_VISIBLE" not in body
    assert "edges do not meet" in body


def test_a_rebuilt_prop_with_an_unsure_grid_receipt_needs_no_attention(client, project_root):
    """Catches: #138, a rebuilt prop whose receipt predates the rebuild still getting the
    "pixel grid is uncertain" strip, which it can do nothing about."""
    store = Store(project_root)
    folder = store.asset_dir("cosy-den", "lamp")
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(folder / "source.png")
    store.write_model(
        folder / "asset.json",
        AssetRecord(
            name="lamp", kind=AssetKind.PROP, source_generation="20260903-191233",
            source_prompt="a lamp", service="fake", target=(4, 4),
            grid_rebuild=GridRebuild(
                pitch_x=8, pitch_y=8, phase_x=0, phase_y=0, confidence=0.9, native_size=(4, 4)
            ),
        ),
    )
    store.write_model(
        folder / "refine.json",
        RefineReport(asset="lamp", refiner="cli", route="auto", warnings=[LOW_GRID_CONFIDENCE]),
    )
    (row,) = [r for r in asset_rows(store, "cosy-den") if r["name"] == "lamp"]
    assert row["rebuilt"] is True
    assert row["attention"] is False
