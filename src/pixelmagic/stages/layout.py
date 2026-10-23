"""How a piece sits on the grid: target sizes, footprints, anchors, pitches.

Pure arithmetic, deliberately kept apart from `decompose` (which keeps pieces)
and `refine` (which hands the scene to PixelRefiner) so that every caller does
the same sums with the same rounding. Nothing here opens an image or touches
the store.

The vocabulary, in plain words: a *pitch* is how many source pixels make one
art pixel; a piece's *target* is its size in art pixels; its *footprint* is
how many grid cells that covers; its *anchor* is the art pixel it stands on.
"""

from __future__ import annotations

import math

from pixelmagic.model import MAX_PITCH, MIN_PITCH
from pixelmagic.stages.slice import SliceBox


def box_outside_sentence(box: tuple[int, int, int, int], width: int, height: int) -> str | None:
    """The sentence for a box that is not wholly inside a `width` x `height` picture, else None.

    The box is left, top, right, bottom with right and bottom exclusive, so it
    fits when 0 <= left, 0 <= top, right <= width and bottom <= height. The one
    wording of this rule: the pick stage refuses with it as a `ValueError` and
    the recipe colour box refuses with it as a `RecipeRefused`.
    """
    x0, y0, x1, y1 = box
    if 0 <= x0 and 0 <= y0 and x1 <= width and y1 <= height:
        return None
    return f"box {x0},{y0},{x1},{y1} is not wholly inside the {width} \u00d7 {height} picture"


def target_for(box: SliceBox, pitch: float) -> tuple[int, int]:
    """The art-pixel size a box should be refined to at this pitch.

    Box ÷ pitch, rounded to the nearest whole number, never below 1 on either
    side (spec locked decision 2). PixelRefiner takes whole numbers only, and
    a zero-sized piece is not a piece.
    """
    return (
        max(1, round(box.width / pitch)),
        max(1, round(box.height / pitch)),
    )


def scale_picked_box(
    picture_box: SliceBox, piece_size: tuple[int, int]
) -> tuple[SliceBox, float]:
    """Size a redrawn piece from the object's box on the original picture.

    Returns `(box, scale)`. `scale` is original pixels per piece pixel: the
    longer side of `picture_box` over the longer side of the piece. `box`
    keeps the object's top-left and has the piece's own size times `scale`,
    so the piece keeps its own shape (the redraw may not match the outline's
    proportions) and takes its longer side from the object. Feed the box to
    `target_for` for the art-pixel target.
    """
    piece_width, piece_height = piece_size
    scale = max(picture_box.width, picture_box.height) / max(piece_width, piece_height)
    box = SliceBox(
        left=picture_box.left,
        top=picture_box.top,
        right=picture_box.left + max(1, round(piece_width * scale)),
        bottom=picture_box.top + max(1, round(piece_height * scale)),
    )
    return box, scale


def footprint_for(target: tuple[int, int], cell_size: int) -> tuple[int, int]:
    """How many grid cells a target covers, `(cols, rows)`, rounding up.

    A prop 17 art pixels wide on a 16-pixel grid needs two cells: a piece
    that spills over a cell boundary by one pixel still occupies the next
    cell (spec locked decision 6).
    """
    width, height = target
    return (math.ceil(width / cell_size), math.ceil(height / cell_size))


def default_anchor(target: tuple[int, int]) -> tuple[int, int]:
    """Bottom centre of a target, in art pixels from its top-left.

    The default every prop starts with (spec locked decision 6). The tool
    never guesses a better one from what the prop is; a person moves it.
    """
    width, height = target
    return (round(width / 2), height)


def check_pitch(value: float) -> None:
    """Refuse a pitch outside `MIN_PITCH`..`MAX_PITCH`.

    Lives in the core so every face that lets a person set a pitch by hand
    (spec §5) inherits the same bound.
    """
    if not (MIN_PITCH <= value <= MAX_PITCH):
        raise ValueError(
            f"a pitch must be between {MIN_PITCH} and {MAX_PITCH} source pixels "
            f"per art pixel, not {value!r}"
        )


def check_anchor(anchor: tuple[int, int], target: tuple[int, int]) -> None:
    """Refuse an anchor that is not on or inside the target.

    Both edges are inclusive: `(w, h)` is the bottom-right corner, and
    `default_anchor` itself sits on the bottom edge.
    """
    x, y = anchor
    width, height = target
    if not (0 <= x <= width and 0 <= y <= height):
        raise ValueError(
            f"anchor {anchor} is outside the piece, which is {width} x {height} "
            f"art pixels"
        )
