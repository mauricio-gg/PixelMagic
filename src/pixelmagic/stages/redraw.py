"""Redraw picked objects in groups and write each one back as a piece.

The orchestration of cutting props out object by object: for every pick with no
piece yet, cut the object out of the original picture, lay a few of them on a
grid, ask the image service to redraw the grid in one call, cut the reply back
into one piece per object, rebuild each piece on its own pixel grid and size it
from the object's own box on the original. The pure maths is in `groups.py`,
`rebuild.py` and `layout.py`; this module only sequences it, talks to the
service, and writes the sidecars.

One bad group never costs the batch: a group that fails both attempts leaves
its objects picked, with no piece and a warning each, so a later `redraw`
with no `--only` picks exactly those up again.
"""

from __future__ import annotations

import logging
import shutil
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from pixelmagic.approval import AutoApprove
from pixelmagic.model import (
    BackgroundPolicy,
    CellReceipt,
    Cutout,
    CutoutOrigin,
    Decomposition,
    Direction,
    GenerationRecord,
    GridSpec,
    Group,
    GroupAttempt,
    GroupCell,
    Pick,
    PickSet,
    PitchSource,
    PromptKind,
    SheetKind,
)
from pixelmagic.naming import describe_cutout, kebab_case, unique_name
from pixelmagic.providers.base import ImageService, ServiceError
from pixelmagic.stages.decompose import (
    NoPicks,
    _all_cutout_names,
    layout_cutout,
    load_decomposition,
    save_decomposition,
)
from pixelmagic.stages.generate import generate, load_generation, promote, reference_name
from pixelmagic.stages.groups import (
    GROUP_REFERENCE_KEY,
    ObjectMask,
    build_reference,
    build_sized_reference,
    cut_back,
    fitting_block_width,
    group_picks,
    sizing_rules,
)
from pixelmagic.stages.layout import check_pitch, scale_picked_box, target_for
from pixelmagic.stages.pick import clear_undo, load_picks, save_picks
from pixelmagic.stages.rebuild import FOUND_BAR, GridRefused, PieceTooThin, rebuild_piece
from pixelmagic.stages.slice import SliceBox, opaque_fraction
from pixelmagic.stages.suggest import cut_out
from pixelmagic.store import Store, claim_id

_LOGGER = logging.getLogger(__name__)

#: How many group calls run at the same time.
PARALLEL_CALLS = 4

#: A "transparent" reply this opaque has no transparency at all — the service
#: ignored the instruction, so the request is retried on a key colour.
DIRTY_TRANSPARENCY_FLOOR = 0.98

#: The size asked of the image service for a group redraw whose grid is not
#: wider than it is tall, in pixels.
GROUP_REPLY_SIZE = (1024, 1024)

#: The size asked when the grid has more columns than rows. A landscape grid
#: reference asked back as a square would be squeezed, and the cut-back's even
#: cells would stop matching what was drawn. 1536 × 1024 is one of the sizes
#: OpenAI images take as they are; codex and gemini take no size at all, and
#: the fake paints whatever it is asked for.
GROUP_REPLY_SIZE_WIDE = (1536, 1024)


#: How many calls one group may make: the first, and one retry.
ATTEMPTS = 2

#: How far, relative to the median, a rebuilt piece's native size over its object's
#: box may stray before the scale check warns about it.
SCALE_TOLERANCE = 0.15

#: The fixed phrase every scale warning carries, so each run can clear the earlier
#: ones (and only those) before writing them afresh.
SCALE_PHRASE = "compared with the other pieces"

#: The phrase of a scale warning written when the direction has a prop scale:
#: the piece is compared with its own target size, not with the others. Cleared
#: with `SCALE_PHRASE` on every run.
TARGET_PHRASE = "compared with its target size"


def _reply_size(grid: GridSpec) -> tuple[int, int]:
    """The reply size whose shape matches the grid: landscape when wider than tall."""
    return GROUP_REPLY_SIZE_WIDE if grid.cols > grid.rows else GROUP_REPLY_SIZE


#: Called on the main thread each time a group has been dealt with.
GroupCallback = Callable[[Group], None]


def _is_dirty(record: GenerationRecord) -> bool:
    """A "transparent" reply with no transparency at all: the instruction was ignored."""
    if record.request.background is not BackgroundPolicy.TRANSPARENT:
        return False
    with Image.open(record.image_path) as sheet:
        return opaque_fraction(sheet) >= DIRTY_TRANSPARENCY_FLOOR


@dataclass
class _Outcome:
    """What the calls for one group came to, gathered on a worker thread."""

    attempts: list[GroupAttempt] = field(default_factory=list)
    #: The reply that was used, or None when every attempt failed.
    record: GenerationRecord | None = None
    #: The reason the group failed, when it did.
    error: str | None = None


@dataclass
class _Planned:
    """One group before its call: who is in it and where its files go."""

    id: str
    picks: list[Pick]
    grid: GridSpec
    folder: Path
    #: Generation ids minted for its attempts, folders already made.
    generation_ids: list[str]
    #: The prompt lines asking for the size-true reference's sizes, or None.
    sizing: str | None = None


def _listing(picks: Sequence[Pick]) -> str:
    """The valid picks as a sentence fragment, numbers with their names."""
    if not picks:
        return "there are no picks"
    return "picks are " + ", ".join(
        f"{pick.number} ({pick.name})" if pick.name else str(pick.number) for pick in picks
    )


def _resolve_only(picks: Sequence[Pick], only: Sequence[str]) -> list[Pick]:
    """The picks `only` names, each by number or by name, in the order given.

    An item made of digits is a pick number, anything else a pick name. An
    unknown one is refused with the picks that do exist, before anything is
    spent.
    """
    chosen: list[Pick] = []
    for item in only:
        found = next(
            (
                pick
                for pick in picks
                if (pick.number == int(item) if item.isdigit() else pick.name == item)
            ),
            None,
        )
        if found is None:
            raise ValueError(f"no pick {item!r}; {_listing(picks)}")
        if found not in chosen:
            chosen.append(found)
    return chosen


def set_prop_scale(
    store: Store, direction: Direction, decomposition_id: str, pick: str, height: float
) -> Direction:
    """Set the direction's prop scale from one pick's wanted height, and save it.

    `pick` is a pick's number or name, as `only` takes. The scale is the pick's
    box height on the original picture over `height` art pixels (original
    pixels per art pixel). Refused with a `ValueError`, before anything is
    saved, for a height under 1, an unknown pick, or a scale outside the pitch
    bounds. Returns the saved direction.
    """
    if height < 1:
        raise ValueError(f"a piece must be at least 1 art pixel tall, not {height!r}")
    picks = load_picks(store, direction.name, decomposition_id).picks
    chosen = _resolve_only(picks, [str(pick)])[0]
    top, bottom = chosen.picture_box[1], chosen.picture_box[3]
    scale = (bottom - top) / height
    check_pitch(scale)
    direction.prop_scale = scale
    return store.save_direction(direction)


def live_pieces(decomposition: Decomposition, pick: Pick) -> list[Cutout]:
    """Every cutout in this decomposition that holds this pick's object.

    A piece holds it when the pick is its origin's own `pick` or one of the
    picks an earlier merge folded into it (`merged_picks`, read from an old
    sidecar). An earlier split left two pieces with the same origin, so there
    can be more than one.
    """
    return [
        c
        for c in decomposition.cutouts
        if c.origin is not None and pick.number in (c.origin.pick, *c.origin.merged_picks)
    ]


def _refuse_ambiguous(decomposition: Decomposition, chosen: Sequence[Pick]) -> None:
    """Refuse `--only` for a pick whose object no longer sits in exactly one plain piece.

    In a project made before merging and splitting were removed, a pick that
    was split has several pieces and a redraw would overwrite one half and
    leave the other, duplicating the object; a pick folded into a merged piece
    has no piece of its own to replace. Both are refused, naming the pieces,
    and the person deletes first. Simplest rule: only a pick with no piece, or
    with one plain piece, is redrawn by name.
    """
    for pick in chosen:
        pieces = live_pieces(decomposition, pick)
        merged = [c for c in pieces if c.origin.merged_picks]
        if merged:
            raise ValueError(
                f"pick {pick.number} is part of the merged piece "
                f"{', '.join(repr(c.name) for c in merged)}; "
                "delete that piece first to redraw it"
            )
        if len(pieces) > 1:
            listed = ", ".join(c.name for c in pieces)
            raise ValueError(
                f"pick {pick.number} has more than one piece ({listed}); "
                "delete all but one first"
            )


def _load_mask(store: Store, direction: str, decomposition_id: str, pick: Pick) -> np.ndarray:
    """A pick's outline as a boolean array the size of its box."""
    with Image.open(store.picks_dir(direction, decomposition_id) / pick.mask) as mask:
        return np.asarray(mask) > 0


def _name_picks(
    store: Store,
    direction: Direction,
    decomposition: Decomposition,
    service: ImageService,
    original: Image.Image,
    everything: PickSet,
    chosen: Sequence[Pick],
) -> tuple[dict[int, str], dict[int, str]]:
    """Settle a cutout name for every chosen pick; return names and labels by pick number.

    A pick that already made a piece keeps that piece's name. A named pick
    keeps its name, made unique against every cutout in the direction. An
    unnamed pick is named as `decompose` names a piece (`describe_cutout`
    on its cut-out, then `kebab_case` and `unique_name`), one at a time in
    pick order, and the name is saved onto the pick so it sticks. The label
    is what the service said the object is (or the name, when it was not
    asked) and becomes the piece's `ai_name`.
    """
    taken = set(store.list_assets(direction.name)) | _all_cutout_names(store, direction.name)
    reserved = {pick.name for pick in everything.picks if pick.name}
    names: dict[int, str] = {}
    labels: dict[int, str] = {}
    renamed = False
    for pick in chosen:
        pieces = live_pieces(decomposition, pick)
        piece = pieces[0] if pieces else None
        if piece is not None:
            names[pick.number] = piece.name
            labels[pick.number] = piece.ai_name or piece.name
            if pick.name is None:
                pick.name = piece.name
                renamed = True
            continue
        if pick.name:
            # Other picks' names count as taken too, so the saved name stays this pick's alone.
            name = unique_name(pick.name, taken | (reserved - {pick.name}))
            label = pick.name
            if name != pick.name:
                # The pick carries the name its piece and notes will use.
                pick.name = name
                renamed = True
        else:
            temporary = store.decomposition_dir(direction.name, decomposition.id) / "pending.png"
            mask = _load_mask(store, direction.name, decomposition.id, pick)
            cut_out(original, pick.picture_box, mask).save(temporary, format="PNG")
            try:
                label = describe_cutout(service, temporary)
            finally:
                temporary.unlink()
            name = unique_name(kebab_case(label), taken | reserved)
            pick.name = name
            renamed = True
        taken.add(name)
        reserved.add(name)
        names[pick.number] = name
        labels[pick.number] = label
    if renamed:
        save_picks(store, direction.name, decomposition.id, everything)
    return names, labels


def _next_group_number(store: Store, direction: str, decomposition_id: str) -> int:
    """One more than the highest `group-NN` already in the decomposition."""
    root = store.groups_dir(direction, decomposition_id)
    numbers = [
        int(folder.name.removeprefix("group-"))
        for folder in (root.iterdir() if root.is_dir() else [])
        if folder.name.removeprefix("group-").isdigit()
    ]
    return max(numbers, default=0) + 1


def _call_group(
    store: Store,
    direction: Direction,
    planned: _Planned,
    subject: str,
    *,
    service: ImageService,
    references: Sequence[str],
    now: datetime | None,
) -> _Outcome:
    """Make the calls for one group, on a worker thread. Writes nothing shared.

    The first call asks for a transparent result when the service can give
    one, else for the key colour. A reply with no transparency at all is
    retried once on the key colour with that reply as the parent, as sheets
    are; a `ServiceError` is retried once on the same background. A reply that
    is still opaque on its last attempt counts as a failure. Only a
    `ServiceError` is caught: anything else is a bug and propagates.
    """
    outcome = _Outcome()
    background = (
        BackgroundPolicy.TRANSPARENT
        if service.capabilities().transparent_background
        else GROUP_REFERENCE_KEY
    )
    parent: GenerationRecord | None = None
    for generation_id in planned.generation_ids[:ATTEMPTS]:
        started = time.monotonic()
        try:
            record = generate(
                store, direction, subject,
                kind=PromptKind.GROUP_REDRAW, service=service, approval=AutoApprove(),
                background=background, size=_reply_size(planned.grid), references=references,
                parent=parent, group=planned.grid, sizing=planned.sizing,
                extra_reference_paths=[planned.folder / "reference.png"],
                generation_id=generation_id, now=now,
            )
        except ServiceError as error:
            outcome.error = f"{error.service}: {error}"
            outcome.attempts.append(
                GroupAttempt(
                    background=background, ok=False, error=outcome.error,
                    seconds=time.monotonic() - started,
                )
            )
            continue
        seconds = time.monotonic() - started
        if _is_dirty(record):
            outcome.error = "the reply had no transparency at all"
            outcome.attempts.append(
                GroupAttempt(
                    generation=generation_id, background=background, ok=False,
                    error=outcome.error, seconds=seconds,
                )
            )
            # The dirty reply is the retry's parent, and the retry asks on the key colour.
            parent, background = record, GROUP_REFERENCE_KEY
            continue
        outcome.attempts.append(
            GroupAttempt(generation=generation_id, background=background, ok=True, seconds=seconds)
        )
        outcome.record = record
        outcome.error = None
        break
    return outcome


#: What an object's failure notes begin with after its name and a colon: no grid,
#: a grid with no solid cell, a failed group, and an empty cell.
_FAILURE_NOTES = (
    "no pixel grid found",
    "a pixel grid was found",
    "not redrawn",
    "the redraw left its cell empty",
)


def notes_for(decomposition: Decomposition, name: str) -> list[tuple[bool, str]]:
    """An object's notes as `(is_problem, text)`, the text without its "name: " start.

    A note belongs to the object when it starts with `name` and ": " (the rule
    `_clear_failure_notes` and `_land_group` use), so "bed-2: ..." is never
    listed under "bed". A problem is one of the `_FAILURE_NOTES` kinds; any
    other note is a plain warning.
    """
    start = f"{name}: "
    return [
        (any(text.startswith(kind) for kind in _FAILURE_NOTES), text)
        for text in (w[len(start):] for w in decomposition.warnings if w.startswith(start))
    ]


def unowned_notes(decomposition: Decomposition, names: Iterable[str]) -> list[tuple[bool, str]]:
    """The notes no current pick owns, as `(is_problem, full_text)`.

    Uses `notes_for`'s ownership rule (a note starts with "name: "), so a note about
    an object since removed, or one with no name at all, is still shown somewhere.
    The text keeps its name start, since no pick label sits above it. A problem is a
    note whose words after the first ": " begin one of the `_FAILURE_NOTES`.
    """
    starts = tuple(f"{name}: " for name in names)
    return [
        (any(w.partition(": ")[2].startswith(kind) for kind in _FAILURE_NOTES), w)
        for w in decomposition.warnings
        if not w.startswith(starts)
    ]


def _clear_failure_notes(decomposition: Decomposition, name: str) -> None:
    """Drop the object's earlier failure notes, so a new failure replaces them.

    Only the four failure kinds in `_FAILURE_NOTES` go. Any other note that
    starts with the name (a cut-back share note about a piece it kept) stays.
    """
    decomposition.warnings[:] = [
        w for w in decomposition.warnings
        if not any(w.startswith(f"{name}: {kind}") for kind in _FAILURE_NOTES)
    ]


def check_scale(store: Store, direction: Direction, decomposition: Decomposition) -> None:
    """Warn about rebuilt pieces drawn at the wrong scale, then save.

    A rebuilt piece's ratio is the longer side of its native size over the longer
    side of its object's box on the original picture (the inverse of its pitch).
    Every rebuilt piece of the decomposition counts, not only this call's. Every
    earlier scale warning (the strings carrying `SCALE_PHRASE` or `TARGET_PHRASE`)
    is cleared first, even when nothing is flagged, since a later redraw moves
    the median.

    With no `direction.prop_scale`, pieces are compared with the median: with
    three or more, a piece more than `SCALE_TOLERANCE` from the median gets one
    warning naming it and the factor. With a prop scale, each piece is compared
    with its own target (its box over the scale, `layout.target_for`), whatever
    the number of pieces, and one more than `SCALE_TOLERANCE` off it gets a
    warning. Nothing about a piece or its files is changed.
    """
    decomposition.warnings[:] = [
        w for w in decomposition.warnings if SCALE_PHRASE not in w and TARGET_PHRASE not in w
    ]
    ratios = {}
    for cutout in decomposition.cutouts:
        if cutout.grid_rebuild is None or cutout.origin is None:
            continue
        box = SliceBox(*cutout.origin.picture_box)
        if direction.prop_scale is None:
            ratios[cutout.name] = max(cutout.grid_rebuild.native_size) / max(box.width, box.height)
        else:
            wanted = max(target_for(box, direction.prop_scale))
            ratios[cutout.name] = max(cutout.grid_rebuild.native_size) / wanted
    if direction.prop_scale is None:
        phrase = SCALE_PHRASE
        middle = statistics.median(ratios.values()) if ratios else 1.0
        checking = len(ratios) >= 3
    else:
        phrase = TARGET_PHRASE
        middle = 1.0
        checking = True
    if checking:
        for name, ratio in ratios.items():
            if abs(ratio - middle) <= SCALE_TOLERANCE * middle:
                continue
            bigger = ratio > middle
            factor = ratio / middle if bigger else middle / ratio
            decomposition.warnings.append(
                f"{name}: drawn about {factor:.1f}x too {'big' if bigger else 'small'} "
                f"{phrase} (redraw it by name if that is wrong)"
            )
    save_decomposition(store, direction.name, decomposition)


def _land_group(
    store: Store,
    direction: Direction,
    decomposition: Decomposition,
    planned: _Planned,
    outcome: _Outcome,
    *,
    names: dict[int, str],
    labels: dict[int, str],
) -> Group:
    """Write one finished group: its folder, its pieces, its warnings. Main thread.

    A failed group adds one "not redrawn" warning per object and writes
    nothing else beyond its own folder. A successful one is cut back, and each
    piece is rebuilt on its own pixel grid (`rebuild.rebuild_piece`). A rebuilt
    piece is saved at native size as `cutouts/<name>/source.png` with
    `cutout.json`; its pitch is the object's box on the original over the
    native size (`layout.scale_picked_box`), the same number as
    `origin.scale`, and its target is the native size. A piece with no grid, or
    one whose grid was found but whose cells are all too thin to keep, is
    treated like a cell with no piece:
    the object stays picked, nothing is written for it (an older piece of it is
    left as it was), and one current warning names it (with the weaker axis's
    score for no grid). A new failure for an object (no grid, grid but too thin, group
    not redrawn, cell empty) replaces its earlier failure notes of all four kinds;
    a note about a piece it still has that is not a failure note (a cut-back share
    note) is left alone.
    An object that lands a piece loses its older
    warnings (the ones that begin with its name and a colon), so the notes never
    say "not redrawn" about a piece that now exists; an object that does not
    keeps them. The decomposition is saved at the end so a piece is never
    listed without its folder.
    """
    group = Group(
        id=planned.id, cols=planned.grid.cols, rows=planned.grid.rows,
        cells=[
            GroupCell(pick=pick.number, cell=divmod(index, planned.grid.cols))
            for index, pick in enumerate(planned.picks)
        ],
        reference_key=GROUP_REFERENCE_KEY, attempts=outcome.attempts,
    )
    record = outcome.record
    if record is None:
        for pick in planned.picks:
            _clear_failure_notes(decomposition, names[pick.number])
            decomposition.warnings.append(
                f"{names[pick.number]}: not redrawn — {outcome.error}"
            )
        store.write_model(planned.folder / "group.json", group)
        save_decomposition(store, direction.name, decomposition)
        return group

    group.generation = record.request.id
    key = None if record.request.background is BackgroundPolicy.TRANSPARENT else (
        record.request.background
    )
    with Image.open(record.image_path) as reply:
        found = cut_back(reply, planned.grid, [names[p.number] for p in planned.picks], key)
    for index, (pick, cell) in enumerate(zip(planned.picks, found, strict=True)):
        row, col = divmod(index, planned.grid.cols)
        rebuilt = None
        if cell.piece is not None and cell.box is not None:
            rebuilt = rebuild_piece(cell.piece)
        if rebuilt is None or isinstance(rebuilt, GridRefused | PieceTooThin):
            _clear_failure_notes(decomposition, names[pick.number])
        if isinstance(rebuilt, GridRefused | PieceTooThin):
            name = names[pick.number]
            if isinstance(rebuilt, GridRefused):
                cell_warning = (
                    f"{name}: no pixel grid found in the redraw (confidence "
                    f"{min(rebuilt.confidence_x, rebuilt.confidence_y):.3f} on the weaker axis; "
                    f"needs {FOUND_BAR:.3f})"
                )
            else:
                cell_warning = (
                    f"{name}: a pixel grid was found in the redraw, but no cell of the "
                    "object was solid enough to keep, so the piece is too thin to rebuild"
                )
            decomposition.warnings.append(cell_warning)
            if cell.warning:
                decomposition.warnings.append(cell.warning)
            group.receipts.append(CellReceipt(pick=pick.number, warning=cell_warning))
            continue
        if rebuilt is not None:
            stale = f"{names[pick.number]}: "
            decomposition.warnings[:] = [
                w for w in decomposition.warnings if not w.startswith(stale)
            ]
        if cell.warning:
            decomposition.warnings.append(cell.warning)
        if rebuilt is None:
            group.receipts.append(CellReceipt(pick=pick.number, warning=cell.warning))
            continue
        group.receipts.append(
            CellReceipt(
                pick=pick.number, piece_size=cell.piece.size,
                share_in_cell=cell.share_in_cell, warning=cell.warning,
            )
        )
        picture_box = SliceBox(*pick.picture_box)
        native_size = rebuilt.image.size
        _, scale = scale_picked_box(picture_box, native_size)
        name = names[pick.number]
        folder = store.cutout_dir(direction.name, decomposition.id, name)
        folder.mkdir(parents=True, exist_ok=True)
        rebuilt.image.save(folder / "source.png", format="PNG")
        cutout = layout_cutout(
            Cutout(
                name=name, ai_name=labels[pick.number], sheet=group.generation,
                kind=SheetKind.PROPS, bbox=cell.box, pitch=scale,
                pitch_source=PitchSource.MEASURED, target=native_size,
                grid_rebuild=rebuilt.grid,
                origin=CutoutOrigin(
                    pick=pick.number, group=group.id, cell=(row, col),
                    picture_box=pick.picture_box, scale=scale,
                    segmenter=decomposition.segmenter,
                ),
            ),
            direction.cell_size,
        )
        store.write_model(folder / "cutout.json", cutout)
        existing = next(
            (i for i, c in enumerate(decomposition.cutouts) if c.name == name), None
        )
        if existing is None:
            decomposition.cutouts.append(cutout)
        else:
            decomposition.cutouts[existing] = cutout
    decomposition.sheet_prompts[group.generation] = record.request.prompt
    decomposition.sheet_services[group.generation] = record.request.service
    shutil.copyfile(record.image_path, planned.folder / "image.png")
    store.write_model(planned.folder / "request.json", record.request)
    store.write_model(planned.folder / "result.json", record.result)
    store.write_model(planned.folder / "review.json", record.review)
    store.write_model(planned.folder / "group.json", group)
    save_decomposition(store, direction.name, decomposition)
    return group


def refuse_redraw(
    decomposition: Decomposition, picks: Sequence[Pick], service: ImageService
) -> None:
    """Refuse a redraw that can never start, before anything is said or paid.

    The one place these three rules live, checked in this order: a pixel-cut run
    (`ValueError`), nothing picked (`NoPicks`), and a service that cannot take
    reference images (`ValueError`). `redraw_picks` calls it first, and a face calls
    it before naming the service it is about to pay, so nothing is said about
    drawing when the redraw is going to be refused.
    """
    if decomposition.pixel_grid is not None:
        raise ValueError(
            "this picture is cut straight from its own pixels, not redrawn; "
            "a redraw here would give each pick two pieces"
        )
    if not picks:
        raise NoPicks(
            "nothing is picked yet; suggest the objects on a picture and pick some first"
        )
    if not service.capabilities().reference_images:
        raise ValueError(
            f"the {service.name} service cannot take reference images, "
            "which a redraw needs; use a service that can"
        )


def redraw_picks(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    *,
    service: ImageService,
    only: Sequence[str] | None = None,
    on_group: GroupCallback | None = None,
    now: datetime | None = None,
) -> Decomposition:
    """Redraw the picked objects in groups and write each as a sized piece.

    Refuses before spending anything: `NoPicks` when nothing is picked, a
    `ValueError` for a service that cannot take reference images, for an
    `only` item that is not a pick (number or name), and for a call that would
    redraw nothing. `only` is refused for a pick held by several pieces or by a
    merged piece (old projects).

    Without `only`, the picks with no piece yet are redrawn, so a second call
    picks up exactly what failed the first time. With `only`, exactly those
    picks are redrawn, replacing an existing piece under its own name. Groups
    go to the service `PARALLEL_CALLS` at a time; each is written on this
    thread as it lands and `decomposition.json` is saved after every group.
    `on_group` is told about each group as it lands. A group that fails both
    attempts costs only its own objects, with a warning naming each.
    """
    decomposition = load_decomposition(store, direction.name, decomposition_id)
    everything = load_picks(store, direction.name, decomposition_id)
    refuse_redraw(decomposition, everything.picks, service)
    if only:
        chosen = _resolve_only(everything.picks, only)
        _refuse_ambiguous(decomposition, chosen)
    else:
        chosen = [pick for pick in everything.picks if not live_pieces(decomposition, pick)]
        if not chosen:
            raise ValueError(
                "every pick already has a piece; name the ones to redraw again"
            )

    if direction.prop_scale is not None:
        # Refuse a group whose biggest object cannot be drawn in its cell now, before
        # naming, undo-clearing or any folder spends or writes anything.
        for members in group_picks({pick.number: pick.area for pick in chosen}):
            fitting_block_width(
                [
                    target_for(SliceBox(*pick.picture_box), direction.prop_scale)
                    for pick in chosen
                    if pick.number in members
                ]
            )

    source = load_generation(store, direction.name, decomposition.source_generation)
    original = Image.open(source.image_path).convert("RGBA")
    names, labels = _name_picks(
        store, direction, decomposition, service, original, everything, chosen
    )
    # A redraw is a commit point: pieces now come from these outlines. After naming,
    # so a failed naming call does not cost the history.
    clear_undo(store, direction, decomposition_id, everything)
    save_decomposition(store, direction.name, decomposition)

    # The scene is a reference so every group call sees it.
    promote(store, direction, source, as_prompt=False)
    references = [reference_name(source)]

    by_number = {pick.number: pick for pick in chosen}
    number = _next_group_number(store, direction.name, decomposition.id)
    planned: list[_Planned] = []
    pool = ThreadPoolExecutor(max_workers=PARALLEL_CALLS)
    try:
        for members in group_picks({pick.number: pick.area for pick in chosen}):
            picks = [by_number[n] for n in members]
            objects = [
                ObjectMask(
                    SliceBox(*pick.picture_box),
                    _load_mask(store, direction.name, decomposition.id, pick),
                )
                for pick in picks
            ]
            sizing = None
            if direction.prop_scale is None:
                reference, grid = build_reference(original, objects)
            else:
                sized = build_sized_reference(
                    original, objects,
                    [target_for(obj.box, direction.prop_scale) for obj in objects],
                )
                reference, grid = sized.picture, sized.grid
                sizing = sizing_rules(sized, [labels[p.number] for p in picks])
            folder = store.group_dir(direction.name, decomposition.id, f"group-{number:02d}")
            folder.mkdir(parents=True, exist_ok=True)
            reference.save(folder / "reference.png", format="PNG")
            # claim_id makes the folder as it mints each id, which reserves the
            # id even against another run.
            ids = [
                claim_id(store.generations_dir(direction.name), now=now)
                for _ in range(ATTEMPTS)
            ]
            planned.append(_Planned(f"group-{number:02d}", picks, grid, folder, ids, sizing))
            number += 1

        pending = {
            pool.submit(
                _call_group, store, direction, group,
                ", ".join(labels[p.number] for p in group.picks),
                service=service, references=references, now=now,
            ): group
            for group in planned
        }
        for future in as_completed(pending):
            group = pending[future]
            landed = _land_group(
                store, direction, decomposition, group, future.result(),
                names=names, labels=labels,
            )
            if on_group is not None:
                on_group(landed)
        check_scale(store, direction, decomposition)
    finally:
        # On an interruption or a bug, calls not yet started are dropped rather
        # than spent; waiting lets the calls already running finish, so the cleanup
        # below never races a folder `generate` is still removing.
        pool.shutdown(wait=True, cancel_futures=True)
        # An id minted for a retry that was never needed leaves an empty folder.
        for group in planned:
            for generation_id in group.generation_ids:
                folder = store.generation_dir(direction.name, generation_id)
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
    return decomposition
