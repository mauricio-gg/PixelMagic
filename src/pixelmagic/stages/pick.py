"""Pick objects: choose which suggestions to redraw, and keep the choice.

A pick is one object chosen for redrawing, with its outline saved beside it as
`picks/<number>.png`. A pick is written to disk (mask and `picks.json`) the
moment it is made, so closing a face loses nothing. Pick numbers start at 1 and
are never reused: `PickSet.last_number` remembers the highest one ever handed
out, so after an unpick the next pick still gets a fresh number and a later
redraw can never overwrite the wrong piece.
"""

from __future__ import annotations

import math
import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from pixelmagic.model import Decomposition, Direction, Pick, PickMethod, PickSet, PickUndo, Point
from pixelmagic.naming import typed_slug
from pixelmagic.segmenter.base import Mask, Segmenter, SegmentFailure, SegmentRequest
from pixelmagic.stages import pixelgrid
from pixelmagic.stages.decompose import load_decomposition, save_decomposition
from pixelmagic.stages.generate import load_generation
from pixelmagic.stages.layout import box_outside_sentence
from pixelmagic.stages.suggest import draw_number, load_suggestions
from pixelmagic.store import Store

#: Outline colours, cycled by pick number so neighbouring picks differ.
OUTLINE_COLOURS = (
    (255, 64, 64, 255), (64, 200, 255, 255), (255, 220, 0, 255),
    (80, 255, 120, 255), (255, 100, 255, 255), (255, 150, 30, 255),
)
#: How thick an outline is on the overlay, in pixels.
OUTLINE_WIDTH = 2
#: A click this close to an earlier click of the same kind adds nothing and is
#: dropped: repeated near-identical points make Segment Anything jump to a
#: different object (the shift-click bug the M5 spike met).
POINT_GUARD_PIXELS = 12
#: How far past a box a "this" click may reach: `add_clicks` widens the box that
#: far around the click so the segmenter's prompt can include what was clicked.
CLICK_BOX_MARGIN = 16
#: How many changes Undo can take back; the oldest is forgotten, with its mask copy.
UNDO_LIMIT = 20


def _picks_file(store: Store, direction: str, decomposition_id: str) -> Path:
    """Where a decomposition's `picks.json` lives."""
    return store.picks_dir(direction, decomposition_id) / "picks.json"


def load_picks(store: Store, direction: str, decomposition_id: str) -> PickSet:
    """Read a decomposition's picks; an empty set when nothing is picked yet."""
    path = _picks_file(store, direction, decomposition_id)
    return store.read_model(path, PickSet) if path.is_file() else PickSet()


def save_picks(store: Store, direction: str, decomposition_id: str, picks: PickSet) -> None:
    """Write a decomposition's `picks.json` back, for a stage that named a pick."""
    store.write_model(_picks_file(store, direction, decomposition_id), picks)


def _undo_dir(store: Store, direction: str, decomposition_id: str) -> Path:
    """Where the copies of overwritten masks wait, inside the picks folder."""
    return store.picks_dir(direction, decomposition_id) / "undo"


def record_undo(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    picks: PickSet,
    number: int,
    before: Pick | None,
) -> None:
    """Note, in memory, what a change to pick `number` is about to overwrite.

    `before` is a copy of the pick as it was (None when the change makes the
    pick). Its mask is copied aside now, because the change is about to
    rewrite or delete it. The entry rides in `picks`, so the caller's own
    `picks.json` write saves it atomically with the change. The oldest entry
    past `UNDO_LIMIT` is dropped with its copy. Call it only once the change is
    certain to go ahead, so a refused change records nothing.
    """
    mask_copy = None
    if before is not None:
        folder = _undo_dir(store, direction.name, decomposition_id)
        folder.mkdir(parents=True, exist_ok=True)
        mask_copy = f"{uuid.uuid4().hex}.png"
        shutil.copyfile(
            store.picks_dir(direction.name, decomposition_id) / before.mask, folder / mask_copy
        )
    picks.undo.append(PickUndo(number=number, before=before, mask_copy=mask_copy))
    while len(picks.undo) > UNDO_LIMIT:
        dropped = picks.undo.pop(0)
        if dropped.mask_copy:
            (_undo_dir(store, direction.name, decomposition_id) / dropped.mask_copy).unlink(
                missing_ok=True
            )


def clear_undo(
    store: Store, direction: Direction, decomposition_id: str, picks: PickSet
) -> None:
    """Forget every undo entry and delete their mask copies; save `picks.json`.

    A redraw calls this: once pieces are drawn from the picks, an old outline
    can no longer be put back without the pieces disagreeing with it.
    """
    picks.undo.clear()
    save_picks(store, direction.name, decomposition_id, picks)
    shutil.rmtree(_undo_dir(store, direction.name, decomposition_id), ignore_errors=True)


def undo_last(store: Store, direction: Direction, decomposition_id: str) -> int:
    """Take back the newest recorded change to a pick's outline.

    A pick that was new is removed with its mask; a fixed pick gets its old
    outline (mask, method, clicks, box, loop, fixes, suggestion, picture box,
    area) back and keeps its current name; a removed pick comes back under its
    own number, in number order, with its mask. `last_number` is untouched, so
    a number is still never handed out twice. Returns the number of the pick
    that was changed. Raises `ValueError` ("There is
    nothing to undo.") when no change is recorded.
    """
    load_decomposition(store, direction.name, decomposition_id)
    picks = load_picks(store, direction.name, decomposition_id)
    if not picks.undo:
        raise ValueError("There is nothing to undo.")
    entry = picks.undo.pop()
    picks_dir = store.picks_dir(direction.name, decomposition_id)
    current = next((pick for pick in picks.picks if pick.number == entry.number), None)
    copy_path = (
        _undo_dir(store, direction.name, decomposition_id) / entry.mask_copy
        if entry.mask_copy
        else None
    )
    if entry.before is None:
        if current is not None:
            picks.picks.remove(current)
    else:
        restored = entry.before
        if current is not None:
            restored.name = current.name
            picks.picks[picks.picks.index(current)] = restored
        else:
            if any(pick.name == restored.name for pick in picks.picks):
                restored.name = None
            picks.picks.append(restored)
            picks.picks.sort(key=lambda pick: pick.number)
        shutil.copyfile(copy_path, picks_dir / restored.mask)
    store.write_model(_picks_file(store, direction.name, decomposition_id), picks)
    # The file goes after the record, as `unpick` does, so a failure leaves a stray
    # file and not a pick without its mask.
    if entry.before is None and current is not None:
        (picks_dir / current.mask).unlink(missing_ok=True)
    if copy_path is not None:
        copy_path.unlink(missing_ok=True)
    return entry.number


def _valid_numbers(picks: PickSet) -> str:
    """The pick numbers that exist, as a sentence fragment."""
    if not picks.picks:
        return "there are no picks yet"
    return "picks are " + ", ".join(str(pick.number) for pick in picks.picks)


def dedupe_points(points: Sequence[Point]) -> list[Point]:
    """Drop a click within `POINT_GUARD_PIXELS` of an earlier click with the same label.

    A click with the other label at the same spot is kept: "this" and "not
    this" close together is a real instruction. Order is preserved.
    """
    kept: list[Point] = []
    for point in points:
        near = any(
            earlier.label == point.label
            and math.hypot(earlier.x - point.x, earlier.y - point.y) <= POINT_GUARD_PIXELS
            for earlier in kept
        )
        if not near:
            kept.append(point)
    return kept


def _chosen_name(name: str | None, picks: PickSet) -> str | None:
    """A pick name after `typed_slug`, refused when unusable, too long or already taken."""
    if name is None:
        return None
    chosen = typed_slug(name, "pick name")
    if chosen is None:
        raise ValueError(f"{name!r} has nothing in it that can be a pick name")
    if any(pick.name == chosen for pick in picks.picks):
        raise ValueError(f"a pick is already called {chosen!r}")
    return chosen


def _add_pick(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    picks: PickSet,
    write_mask: Callable[[Path], None],
    **record,
) -> Pick:
    """Give a new pick a fresh number, write its mask, and save `picks.json` at once.

    `write_mask` puts the outline at the path it is given; `record` is the rest
    of the `Pick` (name, method, box, area, ...).
    """
    picks.last_number += 1
    mask_name = f"{picks.last_number}.png"
    picks_dir = store.picks_dir(direction.name, decomposition_id)
    picks_dir.mkdir(parents=True, exist_ok=True)
    write_mask(picks_dir / mask_name)
    pick = Pick(
        number=picks.last_number, mask=mask_name, created=datetime.now(UTC), **record
    )
    picks.picks.append(pick)
    record_undo(store, direction, decomposition_id, picks, pick.number, None)
    store.write_model(_picks_file(store, direction.name, decomposition_id), picks)
    return pick


def _write_mask_png(mask: Mask, path: Path) -> None:
    """Save a mask as an 8-bit PNG, white inside, the way suggestions are saved.

    Written to a scratch file beside the target and renamed into place, as
    `Store.write_model` does, so an interrupted write never leaves half a mask.
    """
    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    Image.fromarray(mask.pixels.astype(np.uint8) * 255, "L").save(scratch, format="PNG")
    os.replace(scratch, path)


def _picture_size(
    store: Store, direction: Direction, decomposition_id: str, decomposition: Decomposition
) -> tuple[int, int]:
    """The picture's size for a run: its pixel grid's when it is a pixel cut, else the suggestions'.

    The one place this is decided, so every pick operation agrees on the road it is on.
    """
    if decomposition.pixel_grid is not None:
        return decomposition.pixel_grid.picture_size
    return load_suggestions(store, direction.name, decomposition_id).picture_size


def _ask_segmenter(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    points: Sequence[Point],
    box: tuple[int, int, int, int] | None,
    segmenter: Segmenter,
) -> tuple[Mask, list[Point]]:
    """Check the clicks and box, ask the segmenter once, and return its mask.

    Every refusal happens before the segmenter is called. The decomposition
    must be one `suggest` made (it carries the picture's size) or a pixel cut
    (`pixel_grid`: the segmenter is asked on `finder.png` and its answer is
    brought back to an outline on the picture, and the segmenter's name is recorded on
    the decomposition, which so holds the last finder used on the run). Returns the mask, always in
    picture pixels, and the deduped points that were actually sent. A
    `SegmentFailure` is refused with its message; a `SegmenterError` propagates.
    """
    decomposition = load_decomposition(store, direction.name, decomposition_id)
    grid = decomposition.pixel_grid
    width, height = _picture_size(store, direction, decomposition_id, decomposition)
    if not points and box is None:
        raise ValueError("give at least one point or a box")
    if box is not None:
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"box {x0},{y0},{x1},{y1} has no area: right and bottom must exceed left and top"
            )
        outside = box_outside_sentence(box, width, height)
        if outside is not None:
            raise ValueError(outside)
    for point in points:
        if not (0 <= point.x < width and 0 <= point.y < height):
            raise ValueError(
                f"point {point.x},{point.y} is not inside the {width} x {height} picture"
            )
    if box is None and not any(point.label == 1 for point in points):
        raise ValueError("a 'not this' point needs a 'this' point or a box to go with it")
    sent = dedupe_points(points)
    if grid is not None:
        finder = store.decomposition_dir(direction.name, decomposition_id) / pixelgrid.FINDER_PNG
        (answer,) = segmenter.segment(finder, [pixelgrid.to_finder_request(grid, sent, box)])
        if not isinstance(answer, SegmentFailure):
            answer = pixelgrid.mask_to_picture(answer, grid)
    else:
        record = load_generation(store, direction.name, decomposition.source_generation)
        (answer,) = segmenter.segment(record.image_path, [SegmentRequest("pick", sent, box)])
    if isinstance(answer, SegmentFailure):
        raise ValueError(f"the segmenter could not outline that: {answer.message}")
    if grid is not None and decomposition.segmenter != segmenter.name:
        # A pixel cut has no suggest step to record who finds the objects; the pieces'
        # `origin.segmenter` is read from here, so it holds the last finder used on the run.
        decomposition.segmenter = segmenter.name
        save_decomposition(store, direction.name, decomposition)
    return answer, sent


def _pick_by_clicks(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    points: Sequence[Point],
    box: tuple[int, int, int, int] | None,
    name: str | None,
    segmenter: Segmenter,
) -> Pick:
    """Shared body of `pick_points` and `pick_box`: check, ask, save as a new pick."""
    picks = load_picks(store, direction.name, decomposition_id)
    chosen_name = _chosen_name(name, picks)
    mask, sent = _ask_segmenter(store, direction, decomposition_id, points, box, segmenter)
    return _add_pick(
        store, direction, decomposition_id, picks, lambda path: _write_mask_png(mask, path),
        name=chosen_name,
        method=PickMethod.BOX if box is not None else PickMethod.POINTS,
        points=sent, box=box, picture_box=mask.box, area=mask.area,
    )


def pick_points(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    points: Sequence[Point],
    *,
    name: str | None = None,
    segmenter: Segmenter,
) -> Pick:
    """Pick the object under some clicks ("this" = 1, "not this" = 0), on the original picture.

    Refused, before the segmenter is asked: a point outside the picture, no
    points, only "not this" points, an unusable or taken name. A segmenter
    that finds nothing is refused with its message and saves nothing.
    """
    return _pick_by_clicks(store, direction, decomposition_id, points, None, name, segmenter)


def pick_box(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    box: tuple[int, int, int, int],
    *,
    points: Sequence[Point] = (),
    name: str | None = None,
    segmenter: Segmenter,
) -> Pick:
    """Pick the object inside a box `(x0, y0, x1, y1)`, with optional "not this" clicks.

    Same refusals as `pick_points`, plus a box outside the picture or with
    no area.
    """
    return _pick_by_clicks(store, direction, decomposition_id, points, box, name, segmenter)


def fix_pick(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    pick_number: int,
    *,
    points: Sequence[Point] = (),
    box: tuple[int, int, int, int] | None = None,
    segmenter: Segmenter,
) -> Pick:
    """Replace one pick's outline with the segmenter's answer for new clicks or a box.

    The pick keeps its number and name; its mask, box, area and clicks are
    replaced, its method becomes `box` or `points`, and what described the old
    outline (the suggestion it took, a lasso and its fix-ups) is cleared. A
    refusal or a failed answer leaves the old pick and mask untouched. A
    piece already redrawn from this pick is NOT changed; redraw it by name.
    """
    picks = load_picks(store, direction.name, decomposition_id)
    found = next((pick for pick in picks.picks if pick.number == pick_number), None)
    if found is None:
        # A wrong decomposition id must be a file error, not "no pick N".
        load_decomposition(store, direction.name, decomposition_id)
        raise ValueError(f"no pick {pick_number}; {_valid_numbers(picks)}")
    mask, sent = _ask_segmenter(store, direction, decomposition_id, points, box, segmenter)
    record_undo(
        store, direction, decomposition_id, picks, found.number, found.model_copy(deep=True)
    )
    _write_mask_png(mask, store.picks_dir(direction.name, decomposition_id) / found.mask)
    found.method = PickMethod.BOX if box is not None else PickMethod.POINTS
    found.suggestion = None
    found.loop = None
    found.fixes = []
    found.points = sent
    found.box = box
    found.picture_box = mask.box
    found.area = mask.area
    store.write_model(_picks_file(store, direction.name, decomposition_id), picks)
    return found


def add_clicks(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    pick_number: int,
    points: Sequence[Point],
    *,
    segmenter: Segmenter,
) -> Pick:
    """Add new clicks to a pick and outline it again; the pick keeps its number and name.

    The pick's saved clicks stay and the new ones follow them, all through
    `fix_pick`'s own `dedupe_points` (they are passed on together, once), so a
    click within `POINT_GUARD_PIXELS` of an earlier one with the same label is
    ignored. A pick that has a box (one drawn by hand, or the outline of the
    suggestion it took) keeps prompting with a box, so one Add-a-spot click refines
    the object instead of replacing it with a one-point guess, but the box
    is no cage (the picture's size, which clamps it, is the suggestions' or, on a pixel cut,
    the pixel grid's): it becomes the union of the old box, every "this" click and
    `CLICK_BOX_MARGIN` pixels around them, so a click outside it grows the pick
    that way. (Tried on the den against dropping the box and seeding a point
    inside the old outline: that one ran on into the rug and walls; this one
    stayed on the bed.) A "not this" click never widens the box. The work is
    `fix_pick`'s, so a refusal leaves the pick untouched.
    """
    picks = load_picks(store, direction.name, decomposition_id)
    found = next((pick for pick in picks.picks if pick.number == pick_number), None)
    if found is None:
        load_decomposition(store, direction.name, decomposition_id)
        raise ValueError(f"no pick {pick_number}; {_valid_numbers(picks)}")
    box = found.box
    if box is None and found.method is PickMethod.SUGGESTION:
        box = found.picture_box
    all_points = [*found.points, *points]
    if box is not None:
        width, height = _picture_size(
            store, direction, decomposition_id,
            load_decomposition(store, direction.name, decomposition_id),
        )
        this = [point for point in all_points if point.label == 1]
        box = (
            max(0, min([box[0], *(p.x - CLICK_BOX_MARGIN for p in this)])),
            max(0, min([box[1], *(p.y - CLICK_BOX_MARGIN for p in this)])),
            min(width, max([box[2], *(p.x + 1 + CLICK_BOX_MARGIN for p in this)])),
            min(height, max([box[3], *(p.y + 1 + CLICK_BOX_MARGIN for p in this)])),
        )
    return fix_pick(
        store, direction, decomposition_id, pick_number,
        points=all_points, box=box, segmenter=segmenter,
    )


def rename_pick(
    store: Store, direction: Direction, decomposition_id: str, pick_number: int, name: str
) -> Pick:
    """Give a pick a name, by the same rules as naming it when it is made.

    `slug` cleans it, a name with nothing usable in it is refused, and so is
    one that is another pick's. Keeping a pick's own name is allowed. Nothing
    but the name changes.

    Refused once a piece has been redrawn from the pick: the piece carries the
    name, and renaming only the pick would leave the two disagreeing.
    """
    # Imported here because `stages.redraw` itself imports this module.
    from pixelmagic.stages.redraw import live_pieces

    picks = load_picks(store, direction.name, decomposition_id)
    decomposition = load_decomposition(store, direction.name, decomposition_id)
    found = next((pick for pick in picks.picks if pick.number == pick_number), None)
    if found is None:
        raise ValueError(f"no pick {pick_number}; {_valid_numbers(picks)}")
    if live_pieces(decomposition, found):
        raise ValueError(
            f"{found.name or f'pick {found.number}'} already has a piece; "
            "rename the piece instead"
        )
    others = PickSet(picks=[pick for pick in picks.picks if pick is not found])
    found.name = _chosen_name(name, others)
    store.write_model(_picks_file(store, direction.name, decomposition_id), picks)
    return found


def how_picked(pick: Pick) -> str:
    """Say how a pick was made in a few words: "suggestion 7", "2 points", "box 1,2,3,4"."""
    if pick.method is PickMethod.SUGGESTION:
        return f"suggestion {pick.suggestion}"
    if pick.method is PickMethod.BOX and pick.box is not None:
        return "box " + ",".join(str(edge) for edge in pick.box)
    if pick.method is PickMethod.POINTS:
        return f"{len(pick.points)} point{'' if len(pick.points) == 1 else 's'}"
    if pick.method is PickMethod.LASSO:
        fixes = len(pick.fixes)
        return "lasso" if not fixes else f"lasso, {fixes} fix-up{'' if fixes == 1 else 's'}"
    return pick.method.value


def pick_suggestions(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    numbers: Sequence[int],
    *,
    name: str | None = None,
) -> list[Pick]:
    """Pick suggestions by number: one pick for each, in the order given.

    Everything is checked before anything is saved: the decomposition must
    have suggestions (else `NoSuggestions` says to run `suggest`), every
    number must exist (refused with the valid range), a suggestion already
    picked cannot be picked twice, and a name only makes sense with one
    number, must have something usable in it and must not be another pick's.
    An unnamed pick keeps `name = None`; it is named when it is redrawn.
    """
    load_decomposition(store, direction.name, decomposition_id)
    suggestions = load_suggestions(store, direction.name, decomposition_id)
    picks = load_picks(store, direction.name, decomposition_id)
    count = len(suggestions.suggestions)
    if not numbers:
        raise ValueError("give at least one suggestion number to pick")
    for number in numbers:
        if not 1 <= number <= count:
            valid = f"1 to {count}" if count else "none (nothing was found)"
            raise ValueError(f"no suggestion {number}; valid numbers are {valid}")
    taken = {pick.suggestion: pick.number for pick in picks.picks if pick.suggestion is not None}
    for number in numbers:
        if number in taken:
            raise ValueError(f"suggestion {number} is already pick {taken[number]}")
    if len(set(numbers)) != len(numbers):
        raise ValueError("a suggestion can only be picked once")
    if name is not None and len(numbers) != 1:
        raise ValueError("a name goes with one pick; give one suggestion number with it")
    chosen_name = _chosen_name(name, picks)

    made: list[Pick] = []
    for number in numbers:
        suggestion = suggestions.suggestions[number - 1]
        source = store.suggestions_dir(direction.name, decomposition_id) / suggestion.mask
        made.append(
            _add_pick(
                store, direction, decomposition_id, picks,
                lambda path, source=source: shutil.copyfile(source, path),
                name=chosen_name, method=PickMethod.SUGGESTION, suggestion=number,
                picture_box=suggestion.box, area=suggestion.area,
            )
        )
    return made


def unpick(
    store: Store, direction: Direction, decomposition_id: str, pick_number: int
) -> Pick:
    """Remove one pick and its mask file; its number is never handed out again.

    An unknown number is refused with the numbers that do exist.
    """
    load_decomposition(store, direction.name, decomposition_id)
    picks = load_picks(store, direction.name, decomposition_id)
    found = next((pick for pick in picks.picks if pick.number == pick_number), None)
    if found is None:
        raise ValueError(f"no pick {pick_number}; {_valid_numbers(picks)}")
    record_undo(
        store, direction, decomposition_id, picks, found.number, found.model_copy(deep=True)
    )
    picks.picks.remove(found)
    store.write_model(_picks_file(store, direction.name, decomposition_id), picks)
    (store.picks_dir(direction.name, decomposition_id) / found.mask).unlink(missing_ok=True)
    return found


def render_overlay(store: Store, direction: Direction, decomposition_id: str) -> Path:
    """Draw the original picture with every pick's outline and number; return the file.

    Written to `picks/overlay.png`. The picture sits on mid-grey so a
    transparent scene still shows its outlines.
    """
    decomposition = load_decomposition(store, direction.name, decomposition_id)
    record = load_generation(store, direction.name, decomposition.source_generation)
    picture = Image.open(record.image_path).convert("RGBA")
    canvas = Image.new("RGBA", picture.size, (128, 128, 128, 255))
    canvas.alpha_composite(picture)
    picks = load_picks(store, direction.name, decomposition_id)
    picks_dir = store.picks_dir(direction.name, decomposition_id)
    for pick in picks.picks:
        mask = np.asarray(Image.open(picks_dir / pick.mask)) > 0
        edge = mask & ~ndimage.binary_erosion(mask, iterations=OUTLINE_WIDTH, border_value=0)
        colour = OUTLINE_COLOURS[(pick.number - 1) % len(OUTLINE_COLOURS)]
        x0, y0 = pick.picture_box[:2]
        pixels = np.asarray(canvas.crop(pick.picture_box)).copy()
        pixels[edge] = colour
        canvas.paste(Image.fromarray(pixels, "RGBA"), (x0, y0))
        draw_number(canvas, pick.number, (x0 + 2, y0 + 2))
    path = picks_dir / "overlay.png"
    picks_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path


#: A pick on the page: `--info` blue, a fill at about 20 percent and a solid edge.
PICK_BLUE = (108, 180, 255)
TINT_ALPHA = 51


def render_tint(
    store: Store, direction: Direction, decomposition_id: str, pick_number: int
) -> Image.Image:
    """One pick's area as a picture-sized PNG: transparent except the pick, in a tint.

    The same size as the original so a page can lay it over the picture with no
    arithmetic. The area is `PICK_BLUE` at `TINT_ALPHA`, with an opaque edge of
    the same blue, `OUTLINE_THICKNESS` pixels thick.
    """
    width, height = _picture_size(
        store, direction, decomposition_id,
        load_decomposition(store, direction.name, decomposition_id),
    )
    picks = load_picks(store, direction.name, decomposition_id)
    found = next((pick for pick in picks.picks if pick.number == pick_number), None)
    if found is None:
        raise ValueError(f"no pick {pick_number}; {_valid_numbers(picks)}")
    mask_path = store.picks_dir(direction.name, decomposition_id) / found.mask
    mask = np.asarray(Image.open(mask_path)) > 0
    edge = mask & ~ndimage.binary_erosion(mask, iterations=OUTLINE_THICKNESS)
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    x0, y0, x1, y1 = found.picture_box
    area = layer[y0:y1, x0:x1]
    area[mask] = (*PICK_BLUE, TINT_ALPHA)
    area[edge] = (*PICK_BLUE, 255)
    return Image.fromarray(layer, "RGBA")


def _suggestion_mask(store: Store, direction: Direction, decomposition_id: str, mask_name: str):
    """A suggestion's saved box-cropped mask as a boolean array."""
    path = store.suggestions_dir(direction.name, decomposition_id) / mask_name
    return np.asarray(Image.open(path)) > 0


@lru_cache(maxsize=512)
def _mask_rows(path: Path, modified_ns: int):
    """One saved mask decoded and packed eight pixels to a byte, for `suggestions_under`.

    Decoding a big mask costs about 10 ms and a scroll at one spot reads 40 to 50 of
    them, so they are kept, packed so that all of a den's suggestions fit in about 40 MB
    (as plain booleans they would not). `modified_ns` keeps a rewritten file from being
    served stale.
    """
    return np.packbits(np.asarray(Image.open(path)) > 0, axis=1)


def render_suggestion_map(
    store: Store, direction: Direction, decomposition_id: str
) -> Image.Image:
    """Which suggestion is under each pixel, as a picture-sized opaque PNG.

    Each pixel holds the number of the smallest suggestion covering it that is
    not picked yet, or 0 where there is none. The number is 16 bits in two
    channels: red is the high byte, green the low byte (`red * 256 + green`),
    blue is 0 and alpha is 255 so no browser premultiplies the values away.
    Suggestions are painted largest first, so a smaller one ends up on top.
    A page decodes it by reading the pixel under the cursor; nothing else in
    the page needs to know about suggestions. A picked suggestion is left out
    so the cursor falls through to whatever lies under it.

    Changes whenever a pick is made or removed; built fresh on each call
    (measured on a 1536x1024 room with about 400 suggestions, see the PR).
    """
    found = load_suggestions(store, direction.name, decomposition_id)
    width, height = found.picture_size
    taken = {
        pick.suggestion
        for pick in load_picks(store, direction.name, decomposition_id).picks
        if pick.suggestion is not None
    }
    numbers = np.zeros((height, width), dtype=np.uint16)
    for suggestion in sorted(found.suggestions, key=lambda s: -s.area):
        if suggestion.number in taken:
            continue
        x0, y0, x1, y1 = suggestion.box
        mask = _suggestion_mask(store, direction, decomposition_id, suggestion.mask)
        numbers[y0:y1, x0:x1][mask] = suggestion.number
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    layer[:, :, 0] = numbers >> 8
    layer[:, :, 1] = numbers & 0xFF
    layer[:, :, 3] = 255
    return Image.fromarray(layer, "RGBA")


def suggestions_under(
    store: Store, direction: Direction, decomposition_id: str, x: int, y: int
) -> list[int]:
    """Numbers of the not-yet-picked suggestions whose mask covers `(x, y)`, smallest first.

    The page's scroll wheel steps through this list: the smallest is what the
    hover shows, and each step out is the next bigger object under the cursor
    (pillow, quilt, whole bed, bed and rug). A suggestion counts only where its
    mask is opaque, not anywhere in its box, and a picked one is left out, as
    in `render_suggestion_map`. Boxes are checked first, so only the few
    suggestions near the point have their mask read. A point outside the
    picture has none.
    """
    found = load_suggestions(store, direction.name, decomposition_id)
    taken = {
        pick.suggestion
        for pick in load_picks(store, direction.name, decomposition_id).picks
        if pick.suggestion is not None
    }
    covering = []
    for suggestion in found.suggestions:
        x0, y0, x1, y1 = suggestion.box
        if suggestion.number in taken or not (x0 <= x < x1 and y0 <= y < y1):
            continue
        path = store.suggestions_dir(direction.name, decomposition_id) / suggestion.mask
        rows = _mask_rows(path, path.stat().st_mtime_ns)
        column = x - x0
        if (rows[y - y0, column // 8] >> (7 - column % 8)) & 1:
            covering.append(suggestion)
    covering.sort(key=lambda suggestion: suggestion.area)
    return [suggestion.number for suggestion in covering]


#: How many pixels thick a suggestion's outline is drawn.
OUTLINE_THICKNESS = 2
#: The hover outline: white edge over a faint white fill (DESIGN.md).
HOVER_EDGE = (255, 255, 255, 255)
HOVER_FILL = (255, 255, 255, 26)


def render_suggestion_outline(
    store: Store, direction: Direction, decomposition_id: str, number: int
) -> Image.Image:
    """One suggestion's edge as a picture-sized PNG, transparent everywhere else.

    Picture-sized, like `render_tint`, so a page lays it over the picture with
    no arithmetic. The edge is the mask minus its erosion, `OUTLINE_THICKNESS`
    pixels thick, following the object's real outline. A suggestion never
    changes, so the picture never does either.
    """
    found = load_suggestions(store, direction.name, decomposition_id)
    suggestion = next((s for s in found.suggestions if s.number == number), None)
    if suggestion is None:
        count = len(found.suggestions)
        valid = f"1 to {count}" if count else "none (nothing was found)"
        raise ValueError(f"no suggestion {number}; valid numbers are {valid}")
    width, height = found.picture_size
    mask = _suggestion_mask(store, direction, decomposition_id, suggestion.mask)
    edge = mask & ~ndimage.binary_erosion(mask, iterations=OUTLINE_THICKNESS)
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    x0, y0, x1, y1 = suggestion.box
    area = layer[y0:y1, x0:x1]
    area[mask] = HOVER_FILL
    area[edge] = HOVER_EDGE
    return Image.fromarray(layer, "RGBA")
