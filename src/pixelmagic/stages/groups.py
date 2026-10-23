"""Redrawing picked objects in groups: the maths around one service call.

Pure image maths, in the spirit of `slice.py`. Nothing here opens a file,
touches the store or calls a service. It does four things: split the picks
into groups (`group_picks`), lay a group out on a grid (`grid_for`), draw the
reference picture that goes to the service (`build_reference`, or
`build_sized_reference` when the direction has a prop scale), and cut the
service's reply back into one piece per object (`cut_back`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from pixelmagic.model import KEY_COLOURS, BackgroundPolicy, GridSpec
from pixelmagic.stages.slice import SliceBox, to_alpha

#: The most objects one service call redraws.
GROUP_SIZE = 5

#: The reference picture's cell is the biggest object's longer side times
#: `1 + GAP`, so neighbours never touch.
GAP = 0.6

#: The longest side of the reference picture, in pixels.
LONG = 1200

#: The opaque background the reference picture is drawn on. The only place the
#: choice of key colour lives for a group redraw.
GROUP_REFERENCE_KEY = BackgroundPolicy.MAGENTA

#: A piece with a smaller share of its own pixels inside its own cell than
#: this earns a warning.
WARN_SHARE = 0.9

#: A blob smaller than this share of a cell's area is a speck, not an object.
MIN_BLOB_SHARE = 0.005

#: The dilation that joins a detached part to its object is this share of the
#: smaller cell side, and never less than 3 pixels.
JOIN_SHARE = 0.03

#: The width of one art pixel in the size-true reference, in reference pixels.
#: 8 is the block width Codex itself draws at. Tried 7 on the mixed group (3
#: runs, #147): big pieces came out 1.16 to 1.39x, the same as at 8, so it stays.
BLOCK_WIDTH = 8

#: The side of the size-true reference canvas, in pixels; its cells are this
#: divided by the grid's columns and rows. The canvas every #147 run used.
SIZED_CANVAS = 1024

#: Codex draws a small object bigger than the reference shows it, so a small
#: object is drawn in the reference at its wanted size times this. Settled by
#: 8 real Codex runs at this value (#147): small props came out 0.88 to 1.29x
#: of target, apart from 1 mixed run in 3 where Codex ignored the reference
#: and drew them 2.6 to 2.8x. At 0.5 (8 runs) they came out worse: 0.75 to
#: 1.53x, and a lone small piece 1.38 to 1.52x.
SMALL_COMPENSATION = 1 / 1.7

#: An object whose wanted longer side is under this many art pixels counts as
#: small and gets `SMALL_COMPENSATION`; bigger ones are drawn at their full
#: wanted size, since shrinking them made big furniture too small. At 32 the
#: den group's big pieces came out 0.89 to 1.12x (3 runs). Raising the limit to
#: 50 (6 runs) changed nothing the noise did not hide: the mixed group's tall
#: pieces were 1.2 to 1.6x either way.
SMALL_LIMIT = 32

_CONNECTIVITY = np.ones((3, 3), dtype=bool)


def group_picks(areas: Mapping[int, int]) -> list[list[int]]:
    """Split picks into near-equal groups of at most `GROUP_SIZE`.

    `areas` maps a pick's number to its cut-out's area. Picks are sorted
    smallest first (ties by number) so objects of like size share a call, then
    cut into `ceil(n / 5)` groups whose sizes differ by at most one. The
    larger groups come first, so 7 gives 4 + 3 and 11 gives 4 + 4 + 3, and
    the first group holds the smallest objects. No input gives no groups.
    """
    ordered = sorted(areas, key=lambda pick: (areas[pick], pick))
    if not ordered:
        return []
    count = math.ceil(len(ordered) / GROUP_SIZE)
    base, extra = divmod(len(ordered), count)
    groups: list[list[int]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        groups.append(ordered[start : start + size])
        start += size
    return groups


def grid_for(count: int) -> GridSpec:
    """The grid for a group of `count` objects: 2 columns up to 4, else 3."""
    cols = 2 if count <= 4 else 3
    return GridSpec(cols=cols, rows=math.ceil(count / cols), count=count)


@dataclass(frozen=True)
class ObjectMask:
    """One picked object: where it sits on the original and its outline.

    `mask` is a boolean array of shape `(box.height, box.width)`, True on the
    pixels inside the object's outline.
    """

    box: SliceBox
    mask: np.ndarray


def build_reference(
    picture: Image.Image, objects: Sequence[ObjectMask]
) -> tuple[Image.Image, GridSpec]:
    """Draw the reference picture for one group, and the grid it uses.

    Each object is cut from the original `picture` through its own mask (so
    nothing outside the outline comes along) and pasted, centred and at its
    true relative size, into its cell on an opaque `GROUP_REFERENCE_KEY`
    background. Cells are square, `1 + GAP` times the biggest object's longer
    side, laid out in reading order. The whole canvas is then scaled so its
    longest side is `LONG` pixels.
    """
    grid = grid_for(len(objects))
    cell = math.ceil(max(max(o.box.width, o.box.height) for o in objects) * (1 + GAP))
    canvas = Image.new(
        "RGB", (grid.cols * cell, grid.rows * cell), KEY_COLOURS[GROUP_REFERENCE_KEY]
    )
    source = picture.convert("RGB")
    for index, obj in enumerate(objects):
        row, col = divmod(index, grid.cols)
        cut = source.crop(obj.box.as_tuple)
        outline = Image.fromarray(obj.mask.astype(np.uint8) * 255, "L")
        left = col * cell + (cell - obj.box.width) // 2
        top = row * cell + (cell - obj.box.height) // 2
        canvas.paste(cut, (left, top), outline)
    factor = LONG / max(canvas.size)
    size = (max(1, round(canvas.width * factor)), max(1, round(canvas.height * factor)))
    return canvas.resize(size, Image.Resampling.LANCZOS), grid


@dataclass(frozen=True)
class SizedReference:
    """A size-true reference picture and what the prompt must say about it.

    `block_width` is the width of one art pixel as actually drawn (smaller than
    `BLOCK_WIDTH` when the biggest object would not fit its cell). `drawn_sizes`
    is each object's drawn size in art pixels `(width, height)`, in grid order.
    """

    picture: Image.Image
    grid: GridSpec
    block_width: int
    drawn_sizes: list[tuple[int, int]]


def drawn_size(wanted: tuple[int, int]) -> tuple[int, int]:
    """The art-pixel size an object is drawn at in the size-true reference.

    Its wanted size, times `SMALL_COMPENSATION` when its longer side is under
    `SMALL_LIMIT`; never below 1 on either side.
    """
    factor = SMALL_COMPENSATION if max(wanted) < SMALL_LIMIT else 1.0
    return max(1, round(wanted[0] * factor)), max(1, round(wanted[1] * factor))


def _pixelate(
    picture: Image.Image, obj: ObjectMask, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """The object as `size` art pixels: `(colours, inside)`, both of shape `(h, w)`.

    A colour is the average of the object's pixels inside the outline that fall
    in that art pixel, so edge blocks take nothing from outside the mask. An art
    pixel is inside when more than half of it is; the best covered one is forced
    inside when none is, so a tiny object never vanishes.
    """
    width, height = size
    inside = obj.mask.astype(np.float32)
    crop = np.asarray(picture.convert("RGB").crop(obj.box.as_tuple), dtype=np.float32)

    def shrink(plane: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(plane, "F").resize((width, height), Image.Resampling.BOX)
        )

    cover = shrink(inside)
    colours = np.stack([shrink(crop[:, :, c] * inside) for c in range(3)], axis=-1)
    colours = colours / np.maximum(cover, 1e-6)[:, :, None]
    solid = cover > 0.5
    if not solid.any():
        solid = cover >= cover.max()
    return np.clip(np.rint(colours), 0, 255).astype(np.uint8), solid


def fitting_block_width(wanted: Sequence[tuple[int, int]]) -> int:
    """The block width a group of objects of these wanted sizes is drawn at.

    `BLOCK_WIDTH`, or less when the biggest drawn object would not fit its cell
    at that width. Refused with a `ValueError` a person can act on when it would
    not fit at 1. Pure, so `redraw_picks` can refuse before spending anything.
    """
    grid = grid_for(len(wanted))
    cell_w, cell_h = SIZED_CANVAS // grid.cols, SIZED_CANVAS // grid.rows
    drawn = [drawn_size(size) for size in wanted]
    block = min(
        BLOCK_WIDTH,
        min(cell_w // max(w for w, _ in drawn), cell_h // max(h for _, h in drawn)),
    )
    if block < 1:
        raise ValueError(
            f"the biggest object is {max(max(d) for d in drawn)} art pixels on its longer "
            f"side, too big to draw in a {cell_w} x {cell_h} cell; set a larger prop scale"
        )
    return block


def build_sized_reference(
    picture: Image.Image,
    objects: Sequence[ObjectMask],
    wanted: Sequence[tuple[int, int]],
) -> SizedReference:
    """Draw the size-true reference for one group: each object at its wanted size.

    `wanted` is each object's size in art pixels, in the same order. Each object
    is cut through its own outline, shrunk to `drawn_size(wanted)` art pixels
    (every art pixel the average colour of the object's pixels inside the
    outline) and drawn as solid square blocks `block_width` wide, centred in its
    cell on the opaque `GROUP_REFERENCE_KEY` background. The canvas is about
    `SIZED_CANVAS` square, cells in reading order on `grid_for`'s grid. When the
    biggest drawn object would not fit its cell at `BLOCK_WIDTH`, the block
    width of this group shrinks until it does (`fitting_block_width`, which
    refuses with a `ValueError` when it would not fit at 1), and the width used
    is returned. Only this
    reference is resampled: the service's reply never is.
    """
    grid = grid_for(len(objects))
    cell_w, cell_h = SIZED_CANVAS // grid.cols, SIZED_CANVAS // grid.rows
    drawn = [drawn_size(size) for size in wanted]
    block = fitting_block_width(wanted)
    canvas = Image.new(
        "RGB", (cell_w * grid.cols, cell_h * grid.rows), KEY_COLOURS[GROUP_REFERENCE_KEY]
    )
    for index, (obj, size) in enumerate(zip(objects, drawn, strict=True)):
        colours, solid = _pixelate(picture, obj, size)
        blocks = Image.fromarray(colours, "RGB").resize(
            (size[0] * block, size[1] * block), Image.Resampling.NEAREST
        )
        outline = Image.fromarray(solid.astype(np.uint8) * 255, "L").resize(
            blocks.size, Image.Resampling.NEAREST
        )
        row, col = divmod(index, grid.cols)
        left = col * cell_w + (cell_w - blocks.width) // 2
        top = row * cell_h + (cell_h - blocks.height) // 2
        canvas.paste(blocks, (left, top), outline)
    return SizedReference(canvas, grid, block, drawn)


def sizing_rules(reference: SizedReference, labels: Sequence[str]) -> str:
    """The prompt lines that ask the service to keep the reference's sizes.

    Says the objects are already drawn at the wanted size in blocks of the
    reference's block width, lists each object's drawn size in art pixels in
    grid order (named by `labels`), and asks for the same sizes back. Fixed
    wording and numbers computed here; nothing is taken from a model.
    """
    lines = [
        "- The objects in the first image are already drawn at the size wanted, each art "
        f"pixel a solid square block {reference.block_width} px wide on a "
        f"{reference.picture.width} px canvas:"
    ]
    for number, (label, (width, height)) in enumerate(
        zip(labels, reference.drawn_sizes, strict=True), start=1
    ):
        lines.append(f"  object {number} ({label}): {width} x {height} art pixels")
    lines.append(
        "- Redraw each as clean pixel art with the SAME art-pixel size and the SAME overall "
        "size and place: do not enlarge any object, do not use finer pixels, do not fill "
        "the cell; leave plain empty background around each. Only tidy and complete the shapes."
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class CellPiece:
    """What the cut-back found for one cell.

    `piece` is RGBA cropped to its box and `box` is where it sits on the reply
    (`right`/`bottom` exclusive); both are `None` for an empty cell.
    `share_in_cell` is the share of the piece's pixels inside its own cell.
    `warning` is a sentence naming the object, or `None` when all is well.
    """

    piece: Image.Image | None
    box: tuple[int, int, int, int] | None
    share_in_cell: float
    warning: str | None


def _edges(length: int, parts: int) -> np.ndarray:
    """The pixel positions where a side of `length` is cut into `parts` even cells."""
    return np.array([round(i * length / parts) for i in range(parts + 1)])


def cut_back(
    reply: Image.Image,
    grid: GridSpec,
    labels: Sequence[str],
    key: BackgroundPolicy | None,
) -> list[CellPiece]:
    """Cut a service's reply back into one piece per object.

    `key` is the key colour the reply was asked on, or `None` for a
    transparent reply. `labels` names the object in each cell, in reading
    order, so a warning can say which one. The reply is divided evenly into
    `grid.cols` x `grid.rows` cells. Its opaque mask is dilated (only to join
    a detached part to its object) and labelled; each blob goes whole to the
    cell holding most of its pixels, and blobs under `MIN_BLOB_SHARE` of a
    cell are ignored. A cell's piece is its blobs' own pre-dilation pixels,
    cropped to their box, so a piece may reach past its cell. An empty cell
    yields no piece and a warning.
    """
    keyed = np.asarray(to_alpha(reply, key or BackgroundPolicy.TRANSPARENT))
    opaque = keyed[:, :, 3] > 0
    height, width = opaque.shape
    x_edges = _edges(width, grid.cols)
    y_edges = _edges(height, grid.rows)
    cell_w, cell_h = width / grid.cols, height / grid.rows

    side = max(3, math.ceil(JOIN_SHARE * min(cell_w, cell_h)))
    side += 1 - side % 2
    joined = ndimage.binary_dilation(opaque, structure=np.ones((side, side), dtype=bool))
    blobs, blob_count = ndimage.label(joined, structure=_CONNECTIVITY)

    # Which grid cell every pixel is in, and how many of each blob's own
    # pixels fall in each cell.
    cell_of = (
        np.searchsorted(y_edges, np.arange(height), side="right")[:, None] - 1
    ) * grid.cols + (np.searchsorted(x_edges, np.arange(width), side="right")[None, :] - 1)
    cell_total = grid.cols * grid.rows
    counts = np.bincount(
        blobs[opaque] * cell_total + cell_of[opaque],
        minlength=(blob_count + 1) * cell_total,
    ).reshape(blob_count + 1, cell_total)

    min_pixels = MIN_BLOB_SHARE * cell_w * cell_h
    assigned: list[list[int]] = [[] for _ in range(cell_total)]
    for blob in range(1, blob_count + 1):
        if counts[blob].sum() < min_pixels:
            continue
        # A blob that lands mostly in an unused cell is dropped here without a
        # word, and nothing is silent per object: the object whose cell it
        # should have filled is then empty and warns "left its cell empty".
        assigned[int(counts[blob].argmax())].append(blob)

    results: list[CellPiece] = []
    for index in range(grid.count):
        name = labels[index]
        mine = np.isin(blobs, assigned[index]) & opaque
        if not mine.any():
            results.append(CellPiece(None, None, 0.0, f"{name}: the redraw left its cell empty"))
            continue
        rows = np.flatnonzero(mine.any(axis=1))
        columns = np.flatnonzero(mine.any(axis=0))
        box = (int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)
        share = float(mine[cell_of == index].sum() / mine.sum())
        pixels = keyed.copy()
        pixels[~mine] = 0
        piece = Image.fromarray(pixels, "RGBA").crop(box)
        warning = None
        if share < WARN_SHARE:
            warning = (
                f"{name}: only {share:.0%} of its redraw is inside its own cell, "
                "so it may have taken part of a neighbour"
            )
        results.append(CellPiece(piece, box, share, warning))
    return results
