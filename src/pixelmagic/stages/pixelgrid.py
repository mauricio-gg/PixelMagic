"""The pixel grid of a pixel-art picture, and the maps between the picture and the finder's copy.

A picture that is already pixel art (often enlarged) has a grid: one square size and a
phase on each axis, read by the same finder clean-up uses (`rebuild`). From it come the
native picture (one pixel per cell, the picture's own pixel at each cell's centre, never
averaged) and the finder copy (the native picture enlarged `FINDER_SCALE` times with
nearest-neighbour). The object finder works far better on that copy than on a picture
whose pixels are tiny or huge, so a request goes in on the copy and the answer comes back
as an outline on the picture, whole cells at a time. Pure image maths: no files, no
decomposition.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pixelmagic.model import PixelGrid, Point
from pixelmagic.segmenter.base import Mask, SegmentFailure, SegmentRequest
from pixelmagic.stages import cleanup
from pixelmagic.stages import rebuild as rb

#: The native picture is enlarged this many times for the object finder, whatever size the
#: picture came in at (spike #265: native 2x did best, 4x did worse).
FINDER_SCALE = 2

#: The finder copy's file name, in a pixel cut's decomposition folder.
FINDER_PNG = "finder.png"


def read_grid(pixels: np.ndarray, size: float | None = None) -> PixelGrid:
    """Read the grid of an RGBA picture, with the typed ``size`` replacing the found one.

    The rule is clean-up's: the finder reads both axes, then both are taken at one square
    size (``rb.square_axes``). ``ValueError``, with clean-up's wording, for a picture with
    no solid pixel or a typed size outside 1 to the picture's shorter side.
    """
    height, width = pixels.shape[:2]
    if not (pixels[..., 3] > rb.OPAQUE_ALPHA).any():
        raise ValueError("This picture has no solid pixels, so there is nothing to cut from.")
    if size is not None and not 1 <= size <= min(width, height):
        raise ValueError(
            f"The pixel size must be between 1 and {min(width, height)} "
            "(the picture's shorter side)."
        )
    ex, ey = rb.padded_profiles(pixels)
    found_x, found_y = rb._find_axis(ex), rb._find_axis(ey)
    axis_x, axis_y = rb.square_axes(ex, ey, found_x, found_y, size)
    grid = PixelGrid(
        pixel_size=axis_x.pitch,
        phase_x=axis_x.phase,
        phase_y=axis_y.phase,
        size_source="found" if size is None else "typed",
        found_pitch_x=found_x.pitch,
        found_pitch_y=found_y.pitch,
        found_score_x=found_x.score,
        found_score_y=found_y.score,
        unsure=cleanup.readings_unsure(found_x, found_y),
        picture_size=(width, height),
        native_size=(0, 0),
        finder_scale=FINDER_SCALE,
    )
    xs, ys = cell_spans(grid)
    return grid.model_copy(update={"native_size": (len(xs), len(ys))})


def cell_spans(grid: PixelGrid) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The picture pixels each cell covers, per column and per row.

    Each span is clipped to the picture, and a cell that covers no picture pixel is
    dropped, so every span is non-empty and lies inside the picture.
    """
    width, height = grid.picture_size

    def spans(length: int, phase: float) -> list[tuple[int, int]]:
        clipped = []
        for a, b in rb._cell_bounds(length, grid.pixel_size, phase):
            lo, hi = rb._span(a, b)
            lo, hi = max(lo, 0), min(hi, length)
            if lo < hi:
                clipped.append((lo, hi))
        return clipped

    return spans(width, grid.phase_x), spans(height, grid.phase_y)


def native_picture(pixels: np.ndarray, grid: PixelGrid) -> np.ndarray:
    """One pixel per cell: the picture's own pixel at each cell's centre, nothing averaged."""
    xs, ys = cell_spans(grid)
    rows = [(a + b - 1) // 2 for a, b in ys]
    cols = [(a + b - 1) // 2 for a, b in xs]
    return pixels[np.ix_(rows, cols)]


def finder_copy(native: np.ndarray) -> np.ndarray:
    """The native picture enlarged ``FINDER_SCALE`` times, nearest-neighbour."""
    return np.kron(native, np.ones((FINDER_SCALE, FINDER_SCALE, 1), native.dtype))


def _cell_of(spans: list[tuple[int, int]], value: int) -> int:
    """The index of the cell covering a picture coordinate (the nearest one if none does)."""
    starts = [a for a, _ in spans]
    return int(np.clip(np.searchsorted(starts, value, side="right") - 1, 0, len(spans) - 1))


def to_finder_request(
    grid: PixelGrid, points: Sequence[Point], box: tuple[int, int, int, int] | None
) -> SegmentRequest:
    """A request in picture pixels, as a request on the finder copy.

    A point goes to the middle of its cell's block; a box (right and bottom exclusive) goes
    to the block span of every cell it touches.
    """
    xs, ys = cell_spans(grid)
    scale = FINDER_SCALE
    moved = tuple(
        Point(
            x=_cell_of(xs, p.x) * scale + scale // 2,
            y=_cell_of(ys, p.y) * scale + scale // 2,
            label=p.label,
        )
        for p in points
    )
    moved_box = None
    if box is not None:
        x0, y0, x1, y1 = box
        moved_box = (
            _cell_of(xs, x0) * scale,
            _cell_of(ys, y0) * scale,
            (_cell_of(xs, x1 - 1) + 1) * scale,
            (_cell_of(ys, y1 - 1) + 1) * scale,
        )
    return SegmentRequest("pick", moved, moved_box)


def mask_to_picture(answer: Mask, grid: PixelGrid) -> Mask | SegmentFailure:
    """The finder's outline on its copy, as an outline on the picture made of whole cells.

    A cell is inside when at least ``rb.CELL_IN_MASK`` of its finder pixels are (the rule the
    rebuild uses); the outline is every inside cell's picture span, clipped to the picture.
    A ``SegmentFailure`` when no cell is inside.
    """
    xs, ys = cell_spans(grid)
    scale = FINDER_SCALE
    native_w, native_h = len(xs), len(ys)
    full = np.zeros((native_h * scale, native_w * scale), bool)
    x0, y0, x1, y1 = answer.box
    full[y0:y1, x0:x1] = answer.pixels
    blocks = full.reshape(native_h, scale, native_w, scale).mean(axis=(1, 3))
    inside = blocks >= rb.CELL_IN_MASK
    if not inside.any():
        return SegmentFailure("pick", "no whole pixel of the picture is inside that outline")

    width, height = grid.picture_size
    column_cell = np.full(width, -1)
    row_cell = np.full(height, -1)
    for i, (a, b) in enumerate(xs):
        column_cell[a:b] = i
    for j, (a, b) in enumerate(ys):
        row_cell[a:b] = j
    # Rows and columns no cell covers (index -1) are never painted.
    on_picture = (
        inside[row_cell][:, column_cell]
        & (row_cell >= 0)[:, None]
        & (column_cell >= 0)[None]
    )
    if not on_picture.any():
        return SegmentFailure("pick", "no whole pixel of the picture is inside that outline")
    rows, cols = np.nonzero(on_picture)
    box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    return Mask(box, on_picture[box[1] : box[3], box[0] : box[2]], answer.score)
