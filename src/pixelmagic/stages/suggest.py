"""Suggest objects: ask the segmenter for outlines, tidy them, page them.

The segmenter returns every mask it can find, duplicates included (the real
one finds about 1500 on a room and 425 of them are distinct). This stage keeps
one mask per object (dropping any that covers over a fifth of the picture,
which is background), orders them largest first, numbers them from 1 and saves
the numbers with the sidecar, so "7" means the same object on every page, in
every face, on every re-read. A person then picks by number off the contact
sheets.

The result is a decomposition folder with no cutouts yet; `stages.pick` and the
redraw stage build on it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pixelmagic.model import (
    Decomposition,
    Direction,
    GenerationRecord,
    ReviewStatus,
    Suggestion,
    SuggestionSet,
)
from pixelmagic.segmenter.base import Mask, Segmenter
from pixelmagic.stages.decompose import NotApproved, save_decomposition
from pixelmagic.store import Store, claim_id

#: A mask whose overlap with a bigger, already-kept mask is above this share
#: of their union is the same object drawn again, and is dropped.
DUPLICATE_IOU = 0.8

#: A mask covering more than this share of the picture is the room, a wall or
#: the floor: background, not an object, and is dropped before numbering.
BACKGROUND_SHARE = 0.2

#: How many suggestions one contact sheet shows.
PAGE_SIZE = 40
#: The contact sheet's grid: `COLUMNS * ROWS == PAGE_SIZE`.
COLUMNS = 8
ROWS = 5
#: One cell of the contact sheet, in pixels.
CELL = 160
#: What a cell is filled with before the cut-out goes on.
CELL_GREY = (128, 128, 128, 255)


class NoSuggestions(ValueError):
    """This decomposition has no suggestions to pick from or page through."""


def _overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]):
    """The rectangle two boxes share as `(x0, y0, x1, y1)`, or None when disjoint."""
    x0, y0 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x1, y1 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def _iou(kept: Mask, other: Mask, kept_area: int, other_area: int) -> float:
    """Intersection over union of two masks, computed only where their boxes meet.

    The intersection cannot lie outside the overlap of the two boxes, so only
    that rectangle is read from each mask; the union follows from the areas.
    That is what takes 1500 full-size masks from minutes to a second.
    """
    shared = _overlap(kept.box, other.box)
    if shared is None:
        return 0.0
    x0, y0, x1, y1 = shared
    a = kept.pixels[y0 - kept.box[1]:y1 - kept.box[1], x0 - kept.box[0]:x1 - kept.box[0]]
    b = other.pixels[y0 - other.box[1]:y1 - other.box[1], x0 - other.box[0]:x1 - other.box[0]]
    intersection = int(np.logical_and(a, b).sum())
    return intersection / (kept_area + other_area - intersection)


def remove_duplicates(masks: list[Mask]) -> list[Mask]:
    """Keep one mask per object, largest first.

    Masks are taken biggest first; one is dropped when its IoU with a mask
    already kept is over `DUPLICATE_IOU`. Empty masks are dropped. Ties keep
    the segmenter's own order, so the result is deterministic.
    """
    sized = sorted(((mask.area, mask) for mask in masks), key=lambda pair: -pair[0])
    kept: list[tuple[int, Mask]] = []
    for area, mask in sized:
        if area == 0:
            continue
        if any(_iou(other, mask, other_area, area) > DUPLICATE_IOU for other_area, other in kept):
            continue
        kept.append((area, mask))
    return [mask for _area, mask in kept]


def drop_background(masks: list[Mask], picture_area: int) -> list[Mask]:
    """Drop the masks that cover more than `BACKGROUND_SHARE` of the picture."""
    return [mask for mask in masks if mask.area <= BACKGROUND_SHARE * picture_area]


def page_count(suggestions: SuggestionSet) -> int:
    """How many contact sheets a set of suggestions fills."""
    return math.ceil(len(suggestions.suggestions) / suggestions.page_size)


def suggestions_on(suggestions: SuggestionSet, page: int) -> list[Suggestion]:
    """The suggestions one contact sheet shows, in number order.

    A run that found nothing has no pages and shows nothing on any of them, so
    a face never has to special-case it to avoid a refusal. Otherwise refuses
    a page the set does not have, naming the valid pages.
    """
    if not suggestions.suggestions:
        return []
    pages = page_count(suggestions)
    if not 1 <= page <= pages:
        raise ValueError(f"no page {page}; valid pages are 1 to {pages}")
    start = (page - 1) * suggestions.page_size
    return suggestions.suggestions[start:start + suggestions.page_size]


def _suggestions_file(store: Store, direction: str, decomposition_id: str) -> Path:
    """Where a decomposition's suggestions sidecar lives."""
    return store.suggestions_dir(direction, decomposition_id) / "suggestions.json"


def has_suggestions(store: Store, direction: str, decomposition_id: str) -> bool:
    """Whether this decomposition came from `suggest_objects` and so can be picked from."""
    return _suggestions_file(store, direction, decomposition_id).is_file()


def contact_sheet_path(
    store: Store, direction: str, decomposition_id: str, page: int
) -> Path:
    """The contact sheet picture for one page (1-based); the writer and every face use this."""
    return store.suggestions_dir(direction, decomposition_id) / f"contact-{page:02d}.png"


def load_suggestions(store: Store, direction: str, decomposition_id: str) -> SuggestionSet:
    """Read a decomposition's suggestions, or say there are none to read."""
    if not has_suggestions(store, direction, decomposition_id):
        raise NoSuggestions(
            f"{decomposition_id} has no suggestions; the objects must be suggested "
            "from an approved picture first"
        )
    return store.read_model(_suggestions_file(store, direction, decomposition_id), SuggestionSet)


def draw_number(image: Image.Image, number: int, corner: tuple[int, int] = (0, 0)) -> None:
    """Write `number` in white on a black tag at `corner` of `image`, in place."""
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox(corner, str(number))
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=(0, 0, 0, 255))
    draw.text(corner, str(number), fill=(255, 255, 255, 255))


def cut_out(original: Image.Image, box: tuple[int, int, int, int], mask: np.ndarray) -> Image.Image:
    """The part of `original` inside `box` and inside `mask`, the rest transparent."""
    piece = original.crop(box).convert("RGBA")
    alpha = np.asarray(piece)[:, :, 3] * mask
    piece.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    return piece


def _contact_sheet(
    original: Image.Image, page: list[Suggestion], masks: list[np.ndarray]
) -> Image.Image:
    """One numbered contact sheet: each suggestion's cut-out on mid-grey in its cell."""
    sheet = Image.new("RGBA", (COLUMNS * CELL, ROWS * CELL), CELL_GREY)
    for index, (suggestion, mask) in enumerate(zip(page, masks, strict=True)):
        piece = cut_out(original, suggestion.box, mask)
        scale = min(CELL / piece.width, CELL / piece.height)
        size = (max(1, round(piece.width * scale)), max(1, round(piece.height * scale)))
        piece = piece.resize(size, Image.Resampling.NEAREST)
        left, top = (index % COLUMNS) * CELL, (index // COLUMNS) * CELL
        sheet.alpha_composite(piece, (left + (CELL - size[0]) // 2, top + (CELL - size[1]) // 2))
        draw_number(sheet, suggestion.number, (left + 2, top + 2))
    return sheet


def suggest_objects(
    store: Store,
    direction: Direction,
    source: GenerationRecord,
    segmenter: Segmenter,
    *,
    now: datetime | None = None,
) -> Decomposition:
    """Find the objects on an approved picture and write them out to pick from.

    Refuses a picture nobody approved, in decompose's own words. The segmenter
    is asked first, so a `SegmenterError` propagates before anything is written
    and a refusal leaves no folder behind. Then one decomposition folder is
    made holding no cutouts, `suggestions/suggestions.json`, each kept mask as
    `suggestions/NNNN.png` and one `suggestions/contact-NN.png` per page.
    `decomposition.json` goes last, so a run cut short is never listed.
    """
    if source.review.status is not ReviewStatus.APPROVED:
        raise NotApproved(
            f"only an approved picture can be cut up; this one is {source.review.status}"
        )
    raw = segmenter.suggest(source.image_path)
    moment = now or datetime.now(UTC)
    decomposition_id = claim_id(store.decompositions_dir(direction.name), now=moment)
    folder = store.suggestions_dir(direction.name, decomposition_id)
    folder.mkdir(parents=True, exist_ok=True)

    original = Image.open(source.image_path).convert("RGBA")
    kept = remove_duplicates(drop_background(raw, original.width * original.height))
    suggestions: list[Suggestion] = []
    for number, mask in enumerate(kept, start=1):
        name = f"{number:04d}.png"
        Image.fromarray(mask.pixels.astype(np.uint8) * 255, "L").save(folder / name)
        suggestions.append(
            Suggestion(number=number, box=mask.box, area=mask.area, mask=name, score=mask.score)
        )
    result = SuggestionSet(
        segmenter=segmenter.name,
        source_generation=source.request.id,
        picture_size=original.size,
        raw_count=len(raw),
        page_size=PAGE_SIZE,
        created=moment,
        suggestions=suggestions,
    )
    store.write_model(_suggestions_file(store, direction.name, decomposition_id), result)
    for page in range(1, page_count(result) + 1):
        shown = suggestions_on(result, page)
        sheet = _contact_sheet(original, shown, [kept[s.number - 1].pixels for s in shown])
        sheet.save(
            contact_sheet_path(store, direction.name, decomposition_id, page), format="PNG"
        )

    decomposition = Decomposition(
        id=decomposition_id,
        created=moment,
        source_generation=source.request.id,
        segmenter=segmenter.name,
    )
    return save_decomposition(store, direction.name, decomposition)
