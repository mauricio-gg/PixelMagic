"""The pure maths of a group redraw: grouping, the grid, and the cut-back."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixelmagic.model import KEY_COLOURS, BackgroundPolicy
from pixelmagic.stages.groups import (
    BLOCK_WIDTH,
    GROUP_SIZE,
    ObjectMask,
    build_sized_reference,
    cut_back,
    grid_for,
    group_picks,
)
from pixelmagic.stages.slice import SliceBox

MAGENTA = KEY_COLOURS[BackgroundPolicy.MAGENTA]


def _keyed_reply(width: int, height: int, rects: list[tuple[int, int, int, int]]) -> Image.Image:
    """A magenta reply with a solid green rectangle for each `(x0, y0, x1, y1)`."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :] = MAGENTA
    for x0, y0, x1, y1 in rects:
        pixels[y0:y1, x0:x1] = (20, 160, 40)
    return Image.fromarray(pixels, "RGB")


def _transparent_reply(
    width: int, height: int, rects: list[tuple[int, int, int, int]]
) -> Image.Image:
    """A fully transparent reply with an opaque rectangle for each `(x0, y0, x1, y1)`."""
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    for x0, y0, x1, y1 in rects:
        pixels[y0:y1, x0:x1] = (20, 160, 40, 255)
    return Image.fromarray(pixels, "RGBA")


def test_groups_are_size_sorted_at_most_5_near_equal_with_the_right_grid() -> None:
    """Sizes 6, 7 and 11 split near-equally, smallest objects first, and the grid fits the reply."""
    areas = {number: (number * 37) % 101 + 1 for number in range(1, 12)}
    assert group_picks({}) == []
    for total in (1, 5, 6, 7, 10, 11):
        groups = group_picks({n: areas[n] for n in range(1, total + 1)})
        sizes = [len(g) for g in groups]
        assert max(sizes) <= GROUP_SIZE
        assert max(sizes) - min(sizes) <= 1
        assert sum(sizes) == total
    assert [len(g) for g in group_picks({n: areas[n] for n in range(1, 7)})] == [3, 3]
    assert [len(g) for g in group_picks(areas)] == [4, 4, 3]
    flat = [n for g in group_picks(areas) for n in g]
    assert [areas[n] for n in flat] == sorted(areas.values())

    # A group of 5 is a 3 x 2 grid; a reply cut into that grid has each object
    # in the cell the grid says, in reading order.
    grid = grid_for(5)
    assert (grid.cols, grid.rows, grid.count) == (3, 2, 5)
    assert (grid_for(4).cols, grid_for(4).rows) == (2, 2)
    rects = []
    for index in range(5):
        row, col = divmod(index, grid.cols)
        rects.append((col * 100 + 30, row * 100 + 30, col * 100 + 70, row * 100 + 70))
    pieces = cut_back(
        _keyed_reply(300, 200, rects), grid, [f"o{i}" for i in range(5)], BackgroundPolicy.MAGENTA
    )
    assert [p.box for p in pieces] == rects
    assert all(p.warning is None and p.share_in_cell == 1.0 for p in pieces)


def test_cut_back_keeps_a_detached_part_with_its_object_and_ignores_specks() -> None:
    """A flame a few pixels above a lantern stays in the lantern's piece; a speck is no piece."""
    # Two cells of 200 x 200. Cell 0: a body with a flame 4 px above it.
    # Cell 1: a body, and a 3 x 3 speck far from it.
    body = (70, 100, 130, 160)
    flame = (95, 86, 105, 96)
    other = (270, 80, 330, 140)
    speck = (380, 20, 383, 23)
    reply = _keyed_reply(400, 200, [body, flame, other, speck])
    first, second = cut_back(reply, grid_for(2), ["lantern", "chair"], BackgroundPolicy.MAGENTA)

    assert first.box == (70, 86, 130, 160)
    assert first.piece is not None and first.piece.size == (60, 74)
    assert second.box == other
    # Only the object's own opaque pixels survive; the gap is transparent.
    alpha = np.asarray(first.piece)[:, :, 3]
    assert alpha[0, 0] == 0 and alpha[0, 25] == 255 and alpha[12, 25] == 0
    assert first.warning is None and second.warning is None


def test_a_shape_across_two_cells_goes_whole_to_the_cell_holding_most_and_warns_under_90() -> None:
    """A shape 62% in cell 0 is cell 0's whole piece, and a warning names it."""
    # Cells 200 x 200. The wide shape spans x 150..230: 50 px in cell 0, 30 in cell 1.
    wide = (150, 80, 230, 120)
    neighbour = (320, 80, 360, 120)
    reply = _transparent_reply(400, 200, [wide, neighbour])
    first, second = cut_back(reply, grid_for(2), ["rug", "vase"], None)

    assert first.box == wide
    assert first.share_in_cell == 50 / 80
    assert first.warning is not None and "rug" in first.warning
    assert second.box == neighbour
    assert second.share_in_cell == 1.0 and second.warning is None

    # With nothing in a cell, the object is named in the warning and no piece is made.
    empty = cut_back(_transparent_reply(400, 200, [wide]), grid_for(2), ["rug", "vase"], None)[1]
    assert empty.piece is None and empty.warning is not None and "vase" in empty.warning


def _disc_object(side: int) -> tuple[Image.Image, ObjectMask]:
    """A picture of a red disc on blue, and the disc as a picked object (box = whole picture)."""
    yy, xx = np.mgrid[:side, :side]
    mask = (xx - side / 2 + 0.5) ** 2 + (yy - side / 2 + 0.5) ** 2 <= (side / 2) ** 2
    pixels = np.zeros((side, side, 3), dtype=np.uint8)
    pixels[:, :] = (0, 0, 255)
    pixels[mask] = (255, 0, 0)
    return Image.fromarray(pixels, "RGB"), ObjectMask(SliceBox(0, 0, side, side), mask)


def _non_key_box(picture: Image.Image, cell: tuple[int, int, int, int]) -> tuple[int, int]:
    """Width and height of everything in `cell` that is not the key colour."""
    pixels = np.asarray(picture.crop(cell).convert("RGB"))
    drawn = np.any(pixels != MAGENTA, axis=2)
    rows, columns = np.flatnonzero(drawn.any(axis=1)), np.flatnonzero(drawn.any(axis=0))
    return int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1)


def test_the_size_true_reference_uses_only_outline_colours_and_shrinks_only_small_objects():
    # Catches: edge blocks taking blue from outside the outline, a big object shrunk by the
    # small-object compensation (or a small one left at full size), and objects drawn off
    # their wanted size.
    picture, big = _disc_object(100)
    _, small = _disc_object(100)
    reference = build_sized_reference(picture, [big, small], [(60, 60), (10, 10)])

    assert reference.drawn_sizes[0] == (60, 60)
    assert reference.drawn_sizes[1][0] < 10
    side, block = reference.picture.width // 2, reference.block_width
    width, height = _non_key_box(reference.picture, (0, 0, side, reference.picture.height))
    assert (width, height) == (60 * block, 60 * block)
    colours = {tuple(c) for c in np.asarray(reference.picture).reshape(-1, 3)}
    assert colours == {MAGENTA, (255, 0, 0)}, "colour came from outside the outline"


def test_an_object_too_big_for_its_cell_shrinks_the_blocks_instead_of_being_cropped():
    # Catches: a big object silently cropped or overlapping its neighbour.
    picture, obj = _disc_object(100)
    reference = build_sized_reference(picture, [obj, obj], [(200, 200), (10, 10)])

    assert 1 <= reference.block_width < BLOCK_WIDTH
    side = reference.picture.width // 2
    width, _ = _non_key_box(reference.picture, (0, 0, side, reference.picture.height))
    assert width == 200 * reference.block_width
