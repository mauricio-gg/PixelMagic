"""The pixel cut on the web page: starting a run, and the picking screen for one."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pixelmagic.model import AssetRecord, Reviewer
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.base import Mask
from pixelmagic.stages.decompose import keep_scene, load_decomposition
from pixelmagic.stages.generate import import_picture, load_generation
from pixelmagic.stages.pick import load_picks, pick_box
from pixelmagic.store import Store
from pixelmagic.web.app import create_app

WIDTH, HEIGHT = 36, 30


class _KnownOutline:
    """A segmenter that answers every request with one fixed outline of native cells."""

    name = "known"

    def __init__(self, cells: np.ndarray):
        """Keep the outline, a mask of native cells."""
        self.cells = cells

    def segment(self, image_path, requests):
        """Answer at the finder copy's scale (2 pixels per cell)."""
        pixels = np.kron(self.cells, np.ones((2, 2), bool))
        return [Mask((0, 0, pixels.shape[1], pixels.shape[0]), pixels) for _ in requests]


def _blocks_picture(project_root, tmp_path):
    """Import a picture of 3 x 3 blocks of colour (native size 12 x 10) and return its id."""
    palette = np.array([(200, 40, 40, 255), (40, 160, 60, 255), (50, 70, 200, 255)], np.uint8)
    cells = palette[np.random.default_rng(5).integers(0, 3, (HEIGHT // 3, WIDTH // 3))]
    path = tmp_path / "blocks.png"
    Image.fromarray(np.kron(cells, np.ones((3, 3, 1), np.uint8)), "RGBA").save(path)
    store = Store(project_root)
    record = import_picture(
        store, store.load_direction("cosy-den"), path, reviewer=Reviewer.HUMAN
    )
    return store, record.request.id


def test_starting_with_a_typed_size_opens_a_run_and_lands_on_its_picking_page(
    client, project_root, tmp_path
):
    """Catches: the size step dropping the typed size (the run would find its own), or
    redirecting somewhere other than the run's picking page, which must open for a run
    that has no suggestions."""
    store, generation = _blocks_picture(project_root, tmp_path)
    base = f"/directions/cosy-den/generations/{generation}/pixel-cut"

    started = client.post(base, data={"size": "1"}, follow_redirects=False)

    assert started.status_code == 303
    (run,) = store.list_decompositions("cosy-den")
    assert started.headers["location"] == f"/directions/cosy-den/decompositions/{run}/pick"
    grid = load_decomposition(store, "cosy-den", run).pixel_grid
    assert grid.pixel_size == 1 and grid.size_source == "typed"
    assert client.get(started.headers["location"]).status_code == 200


def test_a_size_that_is_not_usable_is_a_sentence_and_starts_nothing(
    client, project_root, tmp_path
):
    """Catches: a typo or an out-of-range size reaching the person as a server error, or
    leaving a half-made run behind in the picture's list."""
    store, generation = _blocks_picture(project_root, tmp_path)
    base = f"/directions/cosy-den/generations/{generation}/pixel-cut"

    for size, words in (
        ("abc", "must be a number"), ("0.5", "between 1 and 30"), ("31", "between 1 and 30")
    ):
        response = client.post(base, data={"size": size})
        assert response.status_code == 400, size
        assert words in response.text, size
        assert "Traceback" not in response.text

    assert store.list_decompositions("cosy-den") == []


def test_a_pick_on_a_pixel_cut_is_drawn_as_whole_cells_at_the_pictures_size(
    client, project_root, tmp_path
):
    """Catches: the pick tint reading suggestions (a pixel cut has none, so every pick's
    layer was a server error) or being drawn at the wrong size or place, so the outline
    would not sit on the cells the cut will keep."""
    store, generation = _blocks_picture(project_root, tmp_path)
    started = client.post(
        f"/directions/cosy-den/generations/{generation}/pixel-cut",
        data={"size": "3"},
        follow_redirects=False,
    )
    base = started.headers["location"].removesuffix("/pick")
    run = base.rsplit("/", 1)[1]
    cells = np.zeros((HEIGHT // 3, WIDTH // 3), bool)
    cells[2:5, 3:7] = True
    pick_box(
        store, store.load_direction("cosy-den"), run, (9, 6, 21, 15),
        segmenter=_KnownOutline(cells),
    )

    page = client.get(base + "/pick")
    assert page.status_code == 200 and "Traceback" not in page.text
    assert f"{base}/picks/1/tint.png" in page.text
    tint = client.get(base + "/picks/1/tint.png")
    assert tint.status_code == 200
    layer = np.asarray(Image.open(io.BytesIO(tint.content)).convert("RGBA"))
    assert layer.shape == (HEIGHT, WIDTH, 4)
    painted = layer[..., 3] > 0
    assert not painted[:6].any() and not painted[15:].any()
    assert not painted[:, :9].any() and not painted[:, 21:].any()
    assert painted[7:14, 10:20].all()  # the outline's own edge may be inside the cells' rim


def _app_finding(project_root, cells):
    """A web app whose object finder answers every request with the outline `cells`."""
    return create_app(
        project_root, service="fake", refiner=StubRefiner(), segmenter=_KnownOutline(cells)
    )


def test_box_cut_continue_over_the_web_keeps_a_pixel_cut_prop_where_the_box_was_dragged(
    project_root, tmp_path
):
    """Catches: the box route dropping or reordering the four numbers (the pick would sit
    elsewhere), the cut route not cutting, or continue not keeping the cut props as assets
    that say they are pixel cuts."""
    store, generation = _blocks_picture(project_root, tmp_path)
    cells = np.zeros((HEIGHT // 3, WIDTH // 3), bool)
    cells[2:5, 3:7] = True
    with TestClient(_app_finding(project_root, cells)) as client:
        started = client.post(
            f"/directions/cosy-den/generations/{generation}/pixel-cut",
            data={"size": "3"},
            follow_redirects=False,
        )
        base = started.headers["location"].removesuffix("/pick")
        run = base.rsplit("/", 1)[1]

        boxed = client.post(
            base + "/pick/box",
            data={"left": 9, "top": 6, "right": 21, "bottom": 15},
            follow_redirects=False,
        )
        assert boxed.status_code == 303
        assert boxed.headers["location"].endswith("/pick?select=1&fresh=1")
        (pick,) = load_picks(store, "cosy-den", run).picks
        assert pick.box == (9, 6, 21, 15)

        cut = client.post(base + "/cut", follow_redirects=False)
        assert cut.status_code == 303 and cut.headers["location"].endswith("/pick?cut=1")
        done = client.get(cut.headers["location"])
        assert done.status_code == 200 and "Traceback" not in done.text
        assert "1 prop cut" in done.text and "prop-1" in done.text and "4 &times; 3 px" in done.text

        kept = client.post(base + "/pick/continue", follow_redirects=False)
        assert kept.status_code == 303 and kept.headers["location"].endswith("/surfaces")

    assert store.list_assets("cosy-den") == ["prop-1"]
    record = AssetRecord.model_validate_json(
        (store.asset_dir("cosy-den", "prop-1") / "asset.json").read_text()
    )
    assert record.pixel_cut is not None and record.pixel_cut.native_size == (4, 3)


def test_a_box_the_core_refuses_is_a_sentence_and_saves_no_pick(project_root, tmp_path):
    """Catches: a box with no area or outside the picture reaching the person as a server
    error, or leaving a pick (and its mask) behind."""
    store, generation = _blocks_picture(project_root, tmp_path)
    cells = np.ones((HEIGHT // 3, WIDTH // 3), bool)
    with TestClient(_app_finding(project_root, cells)) as client:
        started = client.post(
            f"/directions/cosy-den/generations/{generation}/pixel-cut",
            data={"size": "3"},
            follow_redirects=False,
        )
        base = started.headers["location"].removesuffix("/pick")
        run = base.rsplit("/", 1)[1]

        for box in ((9, 6, 9, 15), (30, 20, 99, 99)):
            left, top, right, bottom = box
            response = client.post(
                base + "/pick/box",
                data={"left": left, "top": top, "right": right, "bottom": bottom},
            )
            assert response.status_code == 400, box
            assert "Traceback" not in response.text

    assert load_picks(store, "cosy-den", run).picks == []


def test_a_picture_with_a_pixel_cut_still_offers_the_paid_road_and_the_way_back(
    project_root, tmp_path
):
    """catches: starting a pixel cut hiding the paid road's suggest form, or leaving no link
    back to the pixel cut."""
    store, generation = _blocks_picture(project_root, tmp_path)
    keep_scene(
        store, store.load_direction("cosy-den"), load_generation(store, "cosy-den", generation),
        refiner=StubRefiner(),
    )
    page_url = f"/directions/cosy-den/generations/{generation}"
    with TestClient(_app_finding(project_root, np.zeros((10, 12), bool))) as client:
        started = client.post(page_url + "/pixel-cut", data={"size": "3"}, follow_redirects=False)
        page = client.get(page_url).text

    assert f'action="{page_url}/suggest"' in page
    assert f'href="{started.headers["location"]}"' in page and "Continue the pixel cut" in page
