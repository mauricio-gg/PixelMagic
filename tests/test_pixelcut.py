"""Pixel cut: starting honours a typed size, and a piece is the picture's own pixels."""

from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image

from pixelmagic.journey import scene_steps
from pixelmagic.model import AssetRecord, Direction, Manifest, Point, Reviewer
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.base import Mask
from pixelmagic.stages.decompose import load_decomposition, promote_cutouts
from pixelmagic.stages.export import export
from pixelmagic.stages.generate import import_picture, load_generation
from pixelmagic.stages.pick import add_clicks, load_picks, pick_box
from pixelmagic.stages.pixelcut import cut_pieces, piece_names, start_pixel_cut
from pixelmagic.stages.refine import refine_direction
from pixelmagic.store import Store

NOW = datetime(2026, 10, 22, 9, 0, 0, tzinfo=UTC)
WIDTH, HEIGHT = 36, 30


def _setup(tmp_path):
    """A project holding an imported picture of 3 x 3 blocks, which the finder reads as size 3."""
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    palette = np.array([(200, 40, 40, 255), (40, 160, 60, 255), (50, 70, 200, 255)], np.uint8)
    cells = palette[np.random.default_rng(5).integers(0, 3, (HEIGHT // 3, WIDTH // 3))]
    picture = np.kron(cells, np.ones((3, 3, 1), np.uint8))
    path = tmp_path / "blocks.png"
    Image.fromarray(picture, "RGBA").save(path)
    record = import_picture(store, direction, path, reviewer=Reviewer.HUMAN, now=NOW)
    return store, direction, load_generation(store, direction.name, record.request.id), picture


def test_a_typed_size_of_one_is_used_as_is(tmp_path):
    store, direction, source, picture = _setup(tmp_path)
    cut = start_pixel_cut(store, direction, source, size=1, now=NOW)

    assert cut.pixel_grid.pixel_size == 1 and cut.pixel_grid.size_source == "typed"
    native = store.decomposition_dir(direction.name, cut.id) / "native.png"
    assert np.array_equal(np.asarray(Image.open(native).convert("RGBA")), picture)


@pytest.mark.parametrize("size", [0.5, min(WIDTH, HEIGHT) + 1])
def test_a_size_outside_the_picture_is_refused_and_leaves_nothing(tmp_path, size):
    store, direction, source, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="between 1 and 30"):
        start_pixel_cut(store, direction, source, size=size, now=NOW)

    assert not store.decompositions_dir(direction.name).exists() or not list(
        store.decompositions_dir(direction.name).iterdir()
    )


class _KnownOutline:
    """A segmenter that answers every request with one fixed outline on the finder copy."""

    name = "known"

    def __init__(self, native_cells: np.ndarray):
        """Keep the outline, given as a mask of native cells."""
        self.cells = native_cells

    def segment(self, image_path, requests):
        """Answer with the outline at the finder copy's scale (2 pixels per cell)."""
        pixels = np.kron(self.cells, np.ones((2, 2), bool))
        return [Mask((0, 0, pixels.shape[1], pixels.shape[0]), pixels) for _ in requests]


@pytest.mark.parametrize("crop", [0, 2])
def test_a_piece_is_the_pictures_own_pixels_inside_the_outline(tmp_path, crop):
    """A 3x enlarged picture, whole or cropped so its grid starts off the edge: the piece
    is its native pixels, including the cells on the cropped top-left edge."""
    store, direction, source, picture = _setup(tmp_path)
    picture[12:15, 12:15] = 0  # native cell (4, 4): fully transparent, inside the outline below
    Image.fromarray(picture[crop:, crop:], "RGBA").save(source.image_path)
    cut = start_pixel_cut(store, direction, source, size=3, now=NOW)
    assert (cut.pixel_grid.phase_x != 0) == (crop != 0)

    native = np.asarray(
        Image.open(store.decomposition_dir(direction.name, cut.id) / "native.png").convert("RGBA")
    )
    # Cropping keeps every cell (the first is only partly visible), so native is the cells.
    assert np.array_equal(native, picture[1::3, 1::3])
    outline = np.zeros(native.shape[:2], bool)
    outline[0:6, 0:6] = True  # includes the top-left edge cell
    outline[2, 3:6] = False  # a notch, so the outline is not just its box
    pick_box(store, direction, cut.id, (0, 0, 18, 18), segmenter=_KnownOutline(outline))

    (piece,) = cut_pieces(store, direction, cut.id)

    expected = np.where(outline[..., None], native, 0)[0:6, 0:6]
    saved = np.asarray(
        Image.open(
            store.cutout_dir(direction.name, cut.id, piece.name) / "source.png"
        ).convert("RGBA")
    )
    assert np.array_equal(saved, expected)


def test_a_click_added_to_a_boxed_pick_grows_its_box_on_a_pixel_cut(tmp_path):
    """A pixel cut has no suggestions, so the picture's size must come from its pixel grid."""
    store, direction, source, _ = _setup(tmp_path)
    cut = start_pixel_cut(store, direction, source, size=3, now=NOW)
    outline = np.zeros((HEIGHT // 3, WIDTH // 3), bool)
    outline[0:4, 0:4] = True
    segmenter = _KnownOutline(outline)
    pick_box(store, direction, cut.id, (0, 0, 12, 12), segmenter=segmenter)

    click = Point(x=WIDTH - 1, y=20, label=1)
    add_clicks(store, direction, cut.id, 1, [click], segmenter=segmenter)

    (pick,) = load_picks(store, direction.name, cut.id).picks
    # The box reaches the click's column and stops at the picture's right edge.
    assert pick.box[0] == 0 and pick.box[2] == WIDTH and pick.box[3] > 20


def test_a_kept_pixel_cut_prop_claims_no_image_service(tmp_path):
    """A pixel-cut prop is the picture's own pixels, so no service may be named as its maker."""
    store, direction, source, _ = _setup(tmp_path)
    direction = store.save_direction(direction.model_copy(update={"service": "gemini"}))
    cut = start_pixel_cut(store, direction, source, size=3, now=NOW)
    outline = np.zeros((HEIGHT // 3, WIDTH // 3), bool)
    outline[0:4, 0:4] = True
    pick_box(store, direction, cut.id, (0, 0, 12, 12), segmenter=_KnownOutline(outline))
    (piece,) = cut_pieces(store, direction, cut.id)
    promote_cutouts(store, direction, load_decomposition(store, direction.name, cut.id))

    asset = AssetRecord.model_validate_json(
        (store.asset_dir(direction.name, piece.name) / "asset.json").read_text()
    )
    assert asset.pixel_cut is not None and asset.service is None
    refine_direction(store, direction, refiner=StubRefiner())
    bundle = export(store, direction, scales=[1], stamp="s", now=NOW)
    manifest = Manifest.model_validate_json((bundle.directory / "manifest.json").read_text())
    (entry,) = [e for e in manifest.entries if e.name == piece.name]
    assert entry.service is None


def test_a_pixel_cut_started_but_not_picked_is_what_the_pick_step_links_to(tmp_path):
    """catches: a person who starts a pixel cut and leaves cannot get back to it."""
    store, direction, source, _ = _setup(tmp_path)
    cut = start_pixel_cut(store, direction, source, size=3, now=NOW)

    steps = scene_steps(store, direction.name, source.request.id)
    assert next(s for s in steps if s.key.value == "pick").decomposition == cut.id


def test_the_name_the_page_shows_for_a_pick_is_the_name_the_cut_gives_it(tmp_path):
    """catches: the picking screen saying `prop-1` while the cut files the piece as `prop-1-2`."""
    store, direction, source, _ = _setup(tmp_path)
    outline = np.zeros((HEIGHT // 3, WIDTH // 3), bool)
    outline[0:4, 0:4] = True
    segmenter = _KnownOutline(outline)
    first = start_pixel_cut(store, direction, source, size=3, now=NOW)
    pick_box(store, direction, first.id, (0, 0, 12, 12), segmenter=segmenter)
    cut_pieces(store, direction, first.id)  # prop-1 is now taken in this direction
    second = start_pixel_cut(store, direction, source, size=3, now=NOW.replace(second=5))
    pick_box(store, direction, second.id, (0, 0, 12, 12), segmenter=segmenter)

    picks = load_picks(store, direction.name, second.id).picks
    run = load_decomposition(store, direction.name, second.id)
    shown = piece_names(store, direction, run, picks)
    (piece,) = cut_pieces(store, direction, second.id)

    assert shown == {1: piece.name} and piece.name != "prop-1"
