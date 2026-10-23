"""The lasso: pick an object made of several parts by drawing a line round it.

The rule is "edge in" (spike #103, plan Stage 5). The segmenter is asked for the
piece under many points inside the line. A piece that mostly lies outside the
line is something running under it (floor, rug, a neighbour) and counts as
background: more than `EDGE_OUT` (15%) of it lies outside, unless it covers the
loop's middle point. A piece mostly covered by background is part of that
run-in and is dropped too. The object is what remains, clipped to the line, so nothing outside
the drawn line is ever in it. A click made afterwards (`fix_lasso`) adds or
removes one piece whole and is not clipped to the line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from pixelmagic.model import Direction, Pick, PickMethod, Point
from pixelmagic.segmenter.base import Mask, Segmenter, SegmentFailure, SegmentRequest
from pixelmagic.stages.decompose import load_decomposition
from pixelmagic.stages.generate import load_generation
from pixelmagic.stages.pick import (
    POINT_GUARD_PIXELS,
    _add_pick,
    _chosen_name,
    _picks_file,
    _valid_numbers,
    _write_mask_png,
    load_picks,
    record_undo,
)
from pixelmagic.stages.suggest import load_suggestions
from pixelmagic.store import Store

#: Starting values from spike #103, to be tuned later. A piece with more than this
#: share of its area outside the line is background (it runs under the line).
EDGE_OUT = 0.15
#: A kept piece is dropped when at least this share of it is covered by background.
COVERED_BY_BACKGROUND = 0.8
#: A line enclosing fewer pixels than this is refused as too small.
MIN_LOOP_AREA = 400
#: The interior is the line's area eroded by a square this wide; points go only there.
INTERIOR_ERODE = 9
#: The ring of extra points is the interior minus the area eroded by a square this wide.
RING_ERODE = 41
#: The grid gets about one point per this many pixels of enclosed area.
AREA_PER_POINT = 60
#: The smallest step between two points, in pixels.
MIN_STEP = 6


def _picture(
    store: Store, direction: Direction, decomposition_id: str
) -> tuple[int, int, Path]:
    """The picture's width, height and file; the decomposition must be one `suggest` made."""
    decomposition = load_decomposition(store, direction.name, decomposition_id)
    width, height = load_suggestions(store, direction.name, decomposition_id).picture_size
    record = load_generation(store, direction.name, decomposition.source_generation)
    return width, height, record.image_path


def _erode(region: np.ndarray, side: int) -> np.ndarray:
    """The region shrunk by a square `side` pixels wide; the picture's edge does not erode it."""
    return ndimage.binary_erosion(region, structure=np.ones((side, side), bool), border_value=1)


def _crop(canvas: np.ndarray) -> Mask | None:
    """A picture-sized boolean array as a mask cropped to its own box; None when empty."""
    rows = np.flatnonzero(canvas.any(axis=1))
    if not len(rows):
        return None
    columns = np.flatnonzero(canvas.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    return Mask(box=(left, top, right, bottom), pixels=canvas[top:bottom, left:right].copy())


def _grid(region: np.ndarray, step: int) -> list[tuple[int, int]]:
    """Points `step` apart from the region's top-left corner, only where it is set."""
    ys, xs = np.nonzero(region)
    return [
        (int(x), int(y))
        for y in range(int(ys.min()), int(ys.max()) + 1, step)
        for x in range(int(xs.min()), int(xs.max()) + 1, step)
        if region[y, x]
    ]


def _edge_in(loop_mask: np.ndarray, image_path: Path, segmenter: Segmenter) -> np.ndarray | None:
    """Run the edge-in rule inside a filled line: the picture-sized result, or None when too small.

    The segmenter is asked once, for every point. Only the background union and
    the result are picture-sized; each piece is handled in its own box.
    """
    interior = _erode(loop_mask, INTERIOR_ERODE)
    if int(loop_mask.sum()) < MIN_LOOP_AREA or not interior.any():
        return None
    step = max(MIN_STEP, int(math.sqrt(loop_mask.sum() / AREA_PER_POINT)))
    grid = _grid(interior, step)
    ring_area = interior & ~_erode(loop_mask, RING_ERODE)
    on_grid = set(grid)
    points = grid
    if ring_area.any():
        ring_step = max(MIN_STEP, step // 2)
        points = grid + [p for p in _grid(ring_area, ring_step) if p not in on_grid]

    # The middle: the mean of the interior, or the interior pixel nearest to it.
    coordinates = np.argwhere(interior)
    mean = coordinates.mean(axis=0)
    cy, cx = (int(v) for v in np.rint(mean))
    if not interior[cy, cx]:
        nearest = np.argmin(((coordinates - mean) ** 2).sum(axis=1))
        cy, cx = (int(v) for v in coordinates[nearest])

    requests = [
        SegmentRequest(str(i), [Point(x=x, y=y, label=1)]) for i, (x, y) in enumerate(points)
    ]
    answers = segmenter.segment(image_path, requests)

    background = np.zeros_like(loop_mask)
    kept: list[Mask] = []
    for answer in answers:
        if isinstance(answer, SegmentFailure) or answer.area == 0:
            continue
        x0, y0, x1, y1 = answer.box
        has_middle = x0 <= cx < x1 and y0 <= cy < y1 and answer.pixels[cy - y0, cx - x0]
        outside = int((answer.pixels & ~loop_mask[y0:y1, x0:x1]).sum())
        if has_middle or outside <= EDGE_OUT * answer.area:
            kept.append(answer)
        else:
            background[y0:y1, x0:x1] |= answer.pixels

    result = np.zeros_like(loop_mask)
    for piece in kept:
        x0, y0, x1, y1 = piece.box
        covered = int((piece.pixels & background[y0:y1, x0:x1]).sum())
        if covered < COVERED_BY_BACKGROUND * piece.area:
            result[y0:y1, x0:x1] |= piece.pixels
    result &= loop_mask
    return result


def _filled(loop: Sequence[tuple[int, int]], width: int, height: int) -> np.ndarray:
    """The polygon's pixels, edge included, on a picture-sized canvas."""
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in loop], fill=1, outline=1)
    return np.asarray(canvas) > 0


def pick_lasso(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    loop: Sequence[tuple[int, int]],
    *,
    name: str | None = None,
    segmenter: Segmenter,
) -> Pick:
    """Pick the object inside a closed line drawn on the original picture.

    `loop` is a list of `(x, y)` corners. A point off the picture is moved to
    its edge (a drag that strays past the edge is normal) and the moved line is
    what is stored. Refused, before anything is written: fewer than 3 points, a
    line too small to hold an object, an unusable or taken name, and a line
    with nothing found inside it. A `SegmenterError` propagates.
    """
    picks = load_picks(store, direction.name, decomposition_id)
    chosen_name = _chosen_name(name, picks)
    if len(loop) < 3:
        raise ValueError("a lasso needs at least 3 points; draw a line round the object")
    width, height, image_path = _picture(store, direction, decomposition_id)
    clamped = [
        (min(max(round(x), 0), width - 1), min(max(round(y), 0), height - 1)) for x, y in loop
    ]
    found = _edge_in(_filled(clamped, width, height), image_path, segmenter)
    if found is None:
        raise ValueError("that line is too small to hold an object; draw it bigger")
    mask = _crop(found)
    if mask is None:
        raise ValueError("nothing was found inside that line; draw it again round the object")
    return _add_pick(
        store, direction, decomposition_id, picks, lambda path: _write_mask_png(mask, path),
        name=chosen_name, method=PickMethod.LASSO, loop=clamped,
        picture_box=mask.box, area=mask.area,
    )


def fix_lasso(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    pick_number: int,
    x: int,
    y: int,
    *,
    add: bool,
    segmenter: Segmenter,
) -> Pick:
    """Add (`add=True`) or remove one piece of a lasso pick: the piece under a click.

    The piece is whatever the segmenter outlines for that one point. An add is
    a union and is not clipped to the line (plan open call 14); a remove is a
    set difference. The click is saved in `Pick.fixes` (label 1 add, 0 remove),
    and the mask is rewritten, which can grow past the old box. A click within
    `POINT_GUARD_PIXELS` of a fix of the same kind made since the last fix of the
    other kind changes nothing and the segmenter is not asked. Refused, leaving
    the pick as it was: an unknown pick, a pick the lasso did not make, a click
    off the picture, a piece the segmenter could not outline, and a remove that would leave nothing.
    """
    picks = load_picks(store, direction.name, decomposition_id)
    found = next((pick for pick in picks.picks if pick.number == pick_number), None)
    if found is None:
        load_decomposition(store, direction.name, decomposition_id)
        raise ValueError(f"no pick {pick_number}; {_valid_numbers(picks)}")
    if found.method is not PickMethod.LASSO:
        raise ValueError(f"pick {pick_number} was not made with the lasso, so it has no fix-ups")
    width, height, image_path = _picture(store, direction, decomposition_id)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"point {x},{y} is not inside the {width} x {height} picture")
    label = 1 if add else 0
    # Fixes apply in order, so only same-label fixes made since the last fix of the
    # other label count: a piece removed and then clicked again must come back.
    recent = []
    for fix in reversed(found.fixes):
        if fix.label != label:
            break
        recent.append(fix)
    if any(math.hypot(fix.x - x, fix.y - y) <= POINT_GUARD_PIXELS for fix in recent):
        return found
    (answer,) = segmenter.segment(image_path, [SegmentRequest("fix", [Point(x=x, y=y, label=1)])])
    if isinstance(answer, SegmentFailure):
        raise ValueError(f"the segmenter could not outline that: {answer.message}")

    mask_path = store.picks_dir(direction.name, decomposition_id) / found.mask
    canvas = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = found.picture_box
    canvas[y0:y1, x0:x1] = np.asarray(Image.open(mask_path)) > 0
    ax0, ay0, ax1, ay1 = answer.box
    if add:
        canvas[ay0:ay1, ax0:ax1] |= answer.pixels
    else:
        canvas[ay0:ay1, ax0:ax1] &= ~answer.pixels
    mask = _crop(canvas)
    if mask is None:
        raise ValueError("removing that piece would leave nothing in the pick")
    record_undo(
        store, direction, decomposition_id, picks, found.number, found.model_copy(deep=True)
    )
    _write_mask_png(mask, mask_path)
    found.fixes.append(Point(x=x, y=y, label=label))
    found.picture_box = mask.box
    found.area = mask.area
    store.write_model(_picks_file(store, direction.name, decomposition_id), picks)
    return found
