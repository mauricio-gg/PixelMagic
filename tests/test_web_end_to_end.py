"""The whole web page, clicked through once, entirely offline.

This is the README's walkthrough in library form — the same job
`test_end_to_end.py` does for the command line. If this passes, a person can
open the page and get from a prompt to a bundle without a network, an API key
or a copy of node.
"""

import pytest

from pixelmagic.model import AssetRecord, Manifest
from pixelmagic.stages.decompose import load_decomposition
from pixelmagic.stages.refine import REBUILD_REFINER
from pixelmagic.store import Store


@pytest.fixture(autouse=True)
def no_aseprite(monkeypatch):
    """Keep this walkthrough's export off any Aseprite the machine happens to have.

    The walkthrough calls `POST .../export`, which goes through the same
    `stages.export.export` the CLI and the other end-to-end test use, and that
    function reaches for a real Aseprite install when none is stubbed in. This
    test is about the bundle's PNGs and manifest, not about driving a GUI app
    in batch mode, so `find_aseprite` is patched the same way
    `tests/test_export.py`, `tests/test_cli.py`, `tests/test_end_to_end.py` and
    `tests/test_web_refine.py` already do.
    """
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None)


def test_a_person_can_click_from_a_prompt_to_a_bundle(client, project_root, cut_props):
    store = Store(project_root)

    # A second art direction, made from the form on the front page.
    assert client.post(
        "/directions",
        data={"name": "Cold Cave", "style": "Cold blue caves.", "donts": "no gradients"},
    ).status_code == 200
    assert "cold-cave" in store.list_directions()

    # Renamed, because the first name was wrong.
    client.post("/directions/cold-cave/rename", data={"name": "Icy Cave"})
    assert "icy-cave" in store.list_directions()

    # One picture, turned down once with a reason, then approved.
    client.post(
        "/directions/icy-cave/generate",
        data={"subject": "an icy cave", "kind": "scene",
              "background": "transparent", "size": "120"},
    )
    first = store.list_generations("icy-cave")[0]
    client.post(
        f"/directions/icy-cave/generations/{first}/reject", data={"feedback": "too dark"}
    )
    second = next(one for one in store.list_generations("icy-cave") if one != first)
    client.post(
        f"/directions/icy-cave/generations/{second}/approve",
        data={"keep_reference": "1"},
    )
    client.post(f"/directions/icy-cave/generations/{second}/keep-scene")

    # Three objects picked and redrawn into pieces, which "Continue to Surfaces" keeps as assets.
    decomposition_id = cut_props(store, "icy-cave")
    cut = load_decomposition(store, "icy-cave", decomposition_id)
    pieces = [c.name for c in cut.cutouts]
    assert len(pieces) >= 3
    response = client.post(
        f"/directions/icy-cave/decompositions/{decomposition_id}/pick/continue",
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/directions/icy-cave/generations/{cut.source_generation}/surfaces"
    )
    kept = store.list_assets("icy-cave")
    assert set(pieces) <= set(kept)

    # Refined with a preset (the scene), then exported.
    client.post("/directions/icy-cave/refine", data={"preset": "crisp-sprite"})
    assert store.load_direction("icy-cave").refiner_defaults.preset == "crisp-sprite"

    # One prop told where it stands.
    client.post(
        "/directions/icy-cave/refine/anchor", data={"asset": pieces[0], "x": "3", "y": "9"}
    )
    assert store.read_model(
        store.asset_dir("icy-cave", pieces[0]) / "asset.json", AssetRecord
    ).anchor == (3, 9)

    response = client.post("/directions/icy-cave/export", data={"scale": ["2"]})
    assert response.status_code == 200

    bundles = [f for f in (project_root / "exports").iterdir() if f.name.startswith("icy-cave-")]
    assert len(bundles) == 1
    manifest = Manifest.model_validate_json(
        (bundles[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.project == "DemoGame"
    assert manifest.direction == "icy-cave"
    assert {entry.name for entry in manifest.entries} == set(kept)
    for entry in manifest.entries:
        assert (bundles[0] / entry.files["native"]).exists()
        # The scene went through the stand-in refiner, which copies pixels through, and the
        # bundle has to say so; the props were rebuilt on their own grid and say that instead.
        assert entry.refiner_tool == ("stub" if entry.kind.value == "scene" else REBUILD_REFINER)


def test_no_page_in_the_walkthrough_leaks_a_traceback(client, project_root):
    """Every screen, fetched once, on a direction that has nothing in it yet."""
    for url in (
        "/",
        "/directions/cosy-den",
        "/directions/cosy-den/generate",
        "/directions/cosy-den/refine",
        "/directions/cosy-den/export",
    ):
        response = client.get(url)
        assert response.status_code == 200, (url, response.status_code)
        assert "Traceback" not in response.text
        assert "<html" in response.text
