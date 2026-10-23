"""Tests for the grid finder and the rebuild (`stages/rebuild.py`)."""

import hashlib
import io
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from pixelmagic.providers.fake import FakeImageService
from pixelmagic.stages.rebuild import (
    FOUND_BAR,
    OUTLINE_L_CAP,
    TOP_BAND_START,
    WEAK_TOP_BAR,
    GridRefused,
    PieceTooThin,
    RebuiltPiece,
    _find_axis,
    _oklab,
    _scan,
    padded_profiles,
    rebuild_piece,
)
from pixelmagic.stages.redraw import redraw_picks

PALETTE = [(170, 110, 60), (120, 70, 40), (200, 160, 100), (90, 90, 130)]


def draw_chunky_piece(
    cells=(9, 7),
    pitch=(6.5, 6.5),
    phase=(2.3, 3.1),
    accent=True,
    seed=3,
    palette=PALETTE,
    overrides=None,
) -> tuple[Image.Image, int, int]:
    """Draw a chunky RGBA piece: flat-coloured cells on a grid, transparent outside.

    `cells` is the number of cells (w, h) drawn, `pitch` and `phase` the grid in
    pixels. The picture is `cells * pitch + phase` big, so the grid starts
    `phase` pixels in. Returns the picture and the cells drawn per side. With
    `accent` one cell is small and bright. `palette` is what the cells are drawn
    from; `overrides` maps a cell `(i, j)` to a colour that replaces its own.
    """
    rng = np.random.default_rng(seed)
    nx, ny = cells
    w = int(np.ceil(phase[0] + nx * pitch[0])) + 1
    h = int(np.ceil(phase[1] + ny * pitch[1])) + 1
    img = np.zeros((h, w, 4), np.uint8)
    for j in range(ny):
        for i in range(nx):
            # An uneven outline: the corners are cut so the silhouette is not a plain rectangle.
            if (i, j) in {(0, 0), (nx - 1, ny - 1)}:
                continue
            colour = palette[int(rng.integers(len(palette)))]
            if overrides and (i, j) in overrides:
                colour = overrides[(i, j)]
            if accent and (i, j) == (nx // 2, ny // 2):
                colour = (255, 240, 60)
            x0, x1 = round(phase[0] + i * pitch[0]), round(phase[0] + (i + 1) * pitch[0])
            y0, y1 = round(phase[1] + j * pitch[1]), round(phase[1] + (j + 1) * pitch[1])
            img[y0:y1, x0:x1] = (*colour, 255)
    return Image.fromarray(img, "RGBA"), nx, ny


def test_a_chunky_picture_gives_back_its_grid():
    """Test 1.

    catches: a finder that mis-reads pitch or phase, an off-by-one in the
    padding or the crop, or a sliver of a cell at a tightly cropped edge being
    counted as a whole object cell, which would save pieces at the wrong size.
    """
    padded, nx, ny = draw_chunky_piece(cells=(9, 7), pitch=(6.5, 6.5), phase=(2.3, 3.1))
    # A redrawn piece reaches the rebuild cropped to its own opaque box, so the
    # grid's edge cells are cut by the picture's edge; both shapes must give 9 x 7.
    tight = padded.crop(padded.getbbox())
    assert rebuild_piece(tight).grid.native_size == (nx, ny)
    result = rebuild_piece(padded)
    assert isinstance(result, RebuiltPiece)
    g = result.grid
    assert abs(g.pitch_x - 6.5) <= 0.025 + 1e-9
    assert abs(g.pitch_y - 6.5) <= 0.025 + 1e-9
    for found, true, pitch in ((g.phase_x, 2.3, g.pitch_x), (g.phase_y, 3.1, g.pitch_y)):
        off = abs(found - true) % pitch
        assert min(off, pitch - off) <= 0.5
    assert g.confidence >= FOUND_BAR
    assert g.native_size == (nx, ny)
    assert result.image.size == (nx, ny)


def test_a_piece_keeps_its_proportions_when_the_two_axes_read_different_sizes():
    """Test 1b.

    catches: a rebuild that samples each axis at its own found size, which stretches a
    square-pixel piece (here down by a third) and blurs its colours.
    """
    piece, nx, ny = draw_chunky_piece(cells=(9, 7), pitch=(5.0, 5.0), phase=(2.3, 3.1), seed=1)
    a = np.asarray(piece).astype(float)
    # Blur the colours down the page only: the rows' steps weaken and the finder
    # reads a wrong, much bigger size there while the columns still read 5.
    rgb = ndimage.gaussian_filter1d(a[:, :, :3], 1.5, axis=0)
    blurred = Image.fromarray(np.dstack([rgb, a[:, :, 3]]).astype(np.uint8), "RGBA")
    ex, ey = padded_profiles(np.asarray(blurred))
    found_x, found_y = _find_axis(ex), _find_axis(ey)
    assert abs(found_x.pitch - found_y.pitch) > 2  # the premise: the readings really differ
    assert found_x.score > found_y.score >= FOUND_BAR

    result = rebuild_piece(blurred)
    assert isinstance(result, RebuiltPiece)
    assert result.grid.pitch_x == result.grid.pitch_y
    assert result.grid.native_size == (nx, ny)


def test_a_thin_piece_is_not_made_half_size_when_the_wider_reading_scores_higher():
    """Test 1c.

    catches: a rebuild that takes the size from the higher-scoring axis. On a thin
    prop (a floor lamp, candles) the sideways reading lands on the piece's own width and
    scores higher than the true size, so the piece came out a fraction of its size.
    """
    palette = [*PALETTE, (30, 30, 30), (240, 240, 200)]
    piece, nx, ny = draw_chunky_piece(
        cells=(2, 12), pitch=(3.0, 3.0), phase=(2.3, 3.1), accent=False, palette=palette
    )
    # Soft colour noise, as in a real redraw; a fixed seed keeps it deterministic.
    a = np.asarray(piece).astype(float)
    a[:, :, :3] += np.random.default_rng(1).normal(0, 6, a[:, :, :3].shape)
    noisy = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
    ex, ey = padded_profiles(np.asarray(noisy))
    found_x, found_y = _find_axis(ex), _find_axis(ey)
    # The premise: the higher-scoring reading is far bigger than the true size, and
    # both axes are still found.
    assert found_x.score > found_y.score >= FOUND_BAR
    assert found_x.pitch > 1.6 * found_y.pitch

    result = rebuild_piece(noisy)
    assert isinstance(result, RebuiltPiece)
    assert result.grid.pitch_x == result.grid.pitch_y == pytest.approx(3.0, abs=0.05)
    assert result.grid.native_size == (nx, ny)


def test_a_soft_painting_is_refused():
    """Test 2.

    catches: a finder that "finds" a grid in a painting and returns nonsense
    (#129 part 1's failure: a pitch three and a half times too big accepted).
    """
    # A big piece: on a small one the outline alone scores above the bar at a
    # pitch near the top of the range (reported on the ticket, not tuned here).
    piece, _, _ = draw_chunky_piece(cells=(24, 20), pitch=(6.5, 6.5), phase=(2.3, 3.1))
    a = np.asarray(piece).astype(float)
    rng = np.random.default_rng(11)
    soft = np.stack([ndimage.gaussian_filter(a[:, :, c], 6.5) for c in range(3)], axis=2)
    soft += rng.normal(0, 15, soft.shape)
    # The silhouette goes soft too: a painting has no cell-sized steps in its outline either.
    alpha = ndimage.gaussian_filter(a[:, :, 3], 6.5) + rng.normal(0, 6, a.shape[:2])
    alpha = np.where(alpha > 127, 255, 0)
    out = np.dstack([np.clip(soft, 0, 255), alpha]).astype(np.uint8)
    result = rebuild_piece(Image.fromarray(out, "RGBA"))
    assert isinstance(result, GridRefused)
    assert min(result.confidence_x, result.confidence_y) < FOUND_BAR


def test_a_piece_whose_grid_is_found_but_has_no_solid_cell_is_too_thin_not_gridless():
    # Catches: a thin piece mislabelled as having no pixel grid (the warning would say no grid
    # was found when one was).
    piece, _, _ = draw_chunky_piece(cells=(9, 7), pitch=(6.5, 6.5))
    pixels = np.asarray(piece).copy()
    # Keep only every 7th pixel row: the colour steps still show the grid, but no cell is
    # half opaque.
    pixels[..., 3] = np.where(
        (np.arange(pixels.shape[0])[:, None] % 7 == 0) & (pixels[..., 3] > 0), 255, 0
    )

    result = rebuild_piece(Image.fromarray(pixels, "RGBA"))

    assert isinstance(result, PieceTooThin)
    assert min(result.confidence_x, result.confidence_y) >= FOUND_BAR


def test_a_small_bright_accent_is_kept():
    """Test 3.

    catches: a rebuild that swallows a flame or a knob into the group around
    it (#129 part 2's 0 accents), leaving a brown piece with no orange in it.
    """
    browns = [(150, 100, 60), (130, 85, 50), (165, 115, 70)]
    flame = {(4, 3): (255, 140, 20), (5, 3): (255, 140, 20), (4, 4): (255, 140, 20)}
    piece, nx, ny = draw_chunky_piece(cells=(10, 8), accent=False, palette=browns, overrides=flame)
    result = rebuild_piece(piece)
    assert isinstance(result, RebuiltPiece)
    assert result.grid.native_size == (nx, ny)
    out = np.asarray(result.image).astype(int)
    for i, j in flame:
        # Far closer to the flame orange than to any brown, at its own cell.
        assert np.abs(out[j, i, :3] - (255, 140, 20)).sum() <= 30
    # The browns around it are not pulled toward orange.
    assert np.abs(out[1, 1, :3] - (255, 140, 20)).sum() > 100


def test_accent_cells_cannot_push_a_piece_past_the_colour_cap():
    """Test 3b.

    catches: a piece that ships with more than 64 colours because accent cells
    (kept one colour each) were never reduced, so lowering the other groups'
    shades could not bring the count under the cap.
    """
    cells = np.zeros((40, 40, 3), int)
    cells[:] = np.array([150, 100, 60]) + np.random.default_rng(1).integers(-6, 7, (40, 40, 3))
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = (30, 20, 10)
    k = 0
    for j in range(10, 17):
        for i in range(10, 18):
            cells[j, i] = (40 + 3 * k, 120 + k, 230 - k)
            k += 1
    k = 0
    for j in range(25, 31):
        for i in range(22, 30):
            cells[j, i] = (240 - k, 40 + k, 200 + k // 2)
            k += 1
    big = np.repeat(np.repeat(cells, 6, 0), 6, 1).astype(np.uint8)
    rgba = np.dstack([big, np.full(big.shape[:2], 255, np.uint8)])
    result = rebuild_piece(Image.fromarray(rgba, "RGBA"))
    assert isinstance(result, RebuiltPiece)
    out = np.asarray(result.image)
    assert len(np.unique(out[out[:, :, 3] > 0][:, :3], axis=0)) <= 64


def _object_and_ring(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """The object's cells, and those with a 4-neighbour outside it (the array edge is outside)."""
    solid = np.pad(np.asarray(image)[:, :, 3] > 0, 1)
    touches = ~solid[:-2, 1:-1] | ~solid[2:, 1:-1] | ~solid[1:-1, :-2] | ~solid[1:-1, 2:]
    return solid[1:-1, 1:-1], solid[1:-1, 1:-1] & touches


def test_the_outline_is_closed():
    """Test 4.

    catches: a ring with gaps, where the silhouette is not one clean dark cell
    (the den pieces' open rings in #131), or a ring that invents a colour the
    piece's own outline did not have.
    """
    nx, ny = 12, 10
    dark = (40, 25, 15)
    light_uneven = [(235, 205, 165), (215, 180, 140), (225, 190, 150)]
    overrides = {}
    for i in range(nx):
        for j in range(ny):
            if i in (0, nx - 1) or j in (0, ny - 1):
                # An anti-aliased edge: light, and not the same from cell to cell.
                overrides[(i, j)] = light_uneven[(i + j) % 3]
    for cell in [(3, 3), (8, 6), (6, 4)]:
        overrides[cell] = dark  # the outline group's own colour, inside the piece
    browns = [(160, 105, 60), (140, 92, 52), (175, 118, 70)]
    piece, _, _ = draw_chunky_piece(
        cells=(nx, ny), accent=False, palette=browns, overrides=overrides
    )
    result = rebuild_piece(piece)
    assert isinstance(result, RebuiltPiece)
    obj, ring = _object_and_ring(result.image)
    out = np.asarray(result.image)[:, :, :3]
    assert (_oklab(out[ring])[:, 0] <= OUTLINE_L_CAP).all()
    # One ring: its cells touch each other, corners included.
    _, parts = ndimage.label(ring, structure=np.ones((3, 3)))
    assert parts == 1
    # The ring adds no colour: each of its colours is already inside the piece.
    inside = {tuple(c) for c in out[obj & ~ring]}
    assert {tuple(c) for c in out[ring]} <= inside


def test_a_single_stray_is_removed_but_real_shades_stay():
    """Test 5.

    catches: speckle surviving in the output (one cell of the wrong brown in
    the middle of a surface), and the opposite fault of the clean-up flattening
    a surface's real light and dark shades.
    """
    nx, ny = 12, 10
    dark, shade_a, shade_b = (40, 25, 15), (131, 88, 57), (200, 153, 121)
    between = (171, 126, 94)  # lightness between the two shades, near neither
    overrides = {}
    for i in range(nx):
        for j in range(ny):
            edge = i in (0, nx - 1) or j in (0, ny - 1)
            overrides[(i, j)] = dark if edge else (shade_a if j <= 4 else shade_b)
    # A real step: two neighbouring light cells in the dark half, as a shine would be.
    overrides[(3, 2)] = overrides[(4, 2)] = shade_b
    overrides[(7, 3)] = between
    piece, _, _ = draw_chunky_piece(
        cells=(nx, ny), accent=False, palette=[shade_a], overrides=overrides
    )
    result = rebuild_piece(piece)
    assert isinstance(result, RebuiltPiece)
    out = np.asarray(result.image)[:, :, :3].astype(int)
    # The grid has no cropped row or column: cell (i, j) is out[j, i].
    assert result.grid.native_size == (nx, ny)
    assert (out[3, 7] == out[3, 6]).all() and (out[3, 7] == out[3, 8]).all()
    assert (out[3, 7] == out[2, 7]).all() and (out[3, 7] == out[4, 7]).all()
    # The two light cells keep their own, lighter shade.
    assert (out[2, 3] == out[2, 4]).all()
    assert _oklab(out[2, 3])[0] - _oklab(out[3, 7])[0] > 0.1


def _every_path_piece() -> Image.Image:
    """A piece that makes the rebuild use every step: groups, an accent, a stray and a forced ring.

    Light uneven edge cells (the ring must darken them), two real shades, a
    bright accent in the middle and one off-shade cell inside a surface.
    """
    nx, ny = 12, 10
    dark, shade_a, shade_b = (40, 25, 15), (131, 88, 57), (200, 153, 121)
    light_uneven = [(235, 205, 165), (215, 180, 140), (225, 190, 150)]
    overrides = {}
    for i in range(nx):
        for j in range(ny):
            if i in (0, nx - 1) or j in (0, ny - 1):
                overrides[(i, j)] = light_uneven[(i + j) % 3]
            else:
                overrides[(i, j)] = shade_a if j <= 4 else shade_b
    overrides[(3, 3)] = overrides[(8, 6)] = dark
    overrides[(7, 3)] = (171, 126, 94)  # the stray
    overrides[(5, 5)] = (255, 140, 20)  # the accent
    piece, _, _ = draw_chunky_piece(
        cells=(nx, ny), accent=False, palette=[shade_a], overrides=overrides
    )
    return piece


def _png_bytes(image: Image.Image) -> bytes:
    """The picture as saved PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _rebuild_fingerprint() -> str:
    """The rebuilt piece's PNG hash and its grid record, as one line of text."""
    result = rebuild_piece(_every_path_piece())
    assert isinstance(result, RebuiltPiece)
    return f"{hashlib.sha256(_png_bytes(result.image)).hexdigest()} {result.grid!r}"


def test_two_runs_give_byte_identical_output(tmp_path):
    """Test 9.

    catches: anything non-deterministic (set order, a random seed, a dictionary
    built in an unstable order) in the finder or rebuild, which would make a
    re-run change a piece the CEO had already approved. Two runs in this
    process share one hash seed; the same rebuild in two fresh processes with
    different PYTHONHASHSEED values covers set-order bugs too. The redraw half
    covers the saved bytes of whole pieces, but only within one process.
    """
    result = rebuild_piece(_every_path_piece())
    assert isinstance(result, RebuiltPiece)
    again = rebuild_piece(_every_path_piece())
    assert isinstance(again, RebuiltPiece)
    assert _png_bytes(result.image) == _png_bytes(again.image)
    assert result.grid == again.grid

    # The same rebuild in fresh processes with different hash seeds.
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_rebuild import _rebuild_fingerprint; print(_rebuild_fingerprint())"
    )
    here = str(pathlib.Path(__file__).parent)
    here_print = _rebuild_fingerprint()
    for hash_seed in ("1", "12345"):
        run = subprocess.run(
            [sys.executable, "-c", script, here],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        assert run.stdout.strip() == here_print

    # The redraw's saved pieces, into two separate projects with the same clock.
    from test_redraw import _chunky_cells, _project_with_picks

    boxes = [(10, 10, 90, 50), (120, 10, 160, 130), (200, 10, 264, 74), (10, 160, 170, 256)]
    hashes = []
    for name in ("one", "two"):
        root = tmp_path / name
        store, direction, decomposition_id = _project_with_picks(root, boxes, (400, 300))
        decomposition = redraw_picks(
            store,
            direction,
            decomposition_id,
            service=FakeImageService(painter=_chunky_cells),
        )
        assert len(decomposition.cutouts) == len(boxes)
        hashes.append(
            {
                c.name: hashlib.sha256(
                    (
                        store.cutout_dir(direction.name, decomposition_id, c.name) / "source.png"
                    ).read_bytes()
                ).hexdigest()
                for c in decomposition.cutouts
            }
        )
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("block", [10, 13, 14])
def test_a_big_exact_enlargement_reads_its_own_block_size_not_a_fraction_of_it(block):
    """catches: a block of 10, 13 or 14 px read as 2 or 2.6 (a fraction of it), so the piece
    was saved at several times its real size.

    Every fraction of the true pitch scores a perfect 1.0 on an exact enlargement, so
    the finder must climb back through all the multiples, not only 2x and 3x.
    """
    piece, nx, ny = draw_chunky_piece(cells=(24, 20), pitch=(block, block), phase=(0.0, 0.0))

    result = rebuild_piece(piece)

    assert isinstance(result, RebuiltPiece)
    assert result.grid.native_size == (nx, ny)
    assert result.grid.pitch_x == pytest.approx(block, abs=0.025)
    assert result.grid.pitch_y == pytest.approx(block, abs=0.025)


def _top_noise_profile(seed: int, comb: float) -> np.ndarray:
    """An edge profile with a faint wave of period 13.95 (a big shape's noise) and, if ``comb``,
    a real grid of 8.6 px on top."""
    rng = np.random.default_rng(seed)
    profile = 1.0 + rng.random(1300) * 0.5 + 0.2 * np.cos(2 * np.pi * np.arange(1300) / 13.95)
    if comb:
        for i in range(1, int(1300 / 8.6)):
            profile[round(8.6 * i)] += comb
    return profile


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_weak_reading_at_the_top_of_the_range_does_not_beat_a_real_smaller_grid(seed):
    """catches: noise at the top of the range (13.9 px, score under 0.10) beating a real grid.

    On soft Codex pictures this read a pixel size of about 14 for a true 8.6, so the
    clean-up came out far too coarse. An edge profile is handed to the finder directly.
    """
    profile = _top_noise_profile(seed, comb=0.6)
    # The set-up must really be the failure: the scan's best is a weak top-band pitch.
    pitches, scores = _scan(profile)
    assert pitches[scores.argmax()] >= TOP_BAND_START and scores.max() < WEAK_TOP_BAR

    assert _find_axis(profile).pitch == pytest.approx(8.6, abs=0.05)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_weak_noise_at_the_top_with_nothing_real_below_stays_at_the_top(seed):
    """catches: the re-pick landing on a whole fraction of the noise (6.975, 4.65) and clean-up
    calling that sure; kept at the top, the reading is still flagged unsure."""
    profile = np.random.default_rng(seed).random(1300)
    for i in range(1, int(1300 / 13.95)):
        profile[round(13.95 * i)] += 0.6

    assert _find_axis(profile).pitch == pytest.approx(13.95, abs=0.05)
