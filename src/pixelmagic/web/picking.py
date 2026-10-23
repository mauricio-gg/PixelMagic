"""Choosing which objects to redraw, on the picture itself.

The page shows the original with a tool tray over it. The suggestions are not
drawn, only quiet number tags on the objects not yet picked (and only when there
are few enough of them, `show_numbers`). On a busier picture the panel lists the
objects not yet picked instead (`found_list`), so a keyboard can reach them. In
the default mode a mouse hovers and outlines the smallest object under the pointer
from a lookup picture and an outline picture the core renders (`stages/pick.py`);
a touch screen has no hover, so a tap previews the object and a second tap on it
picks it. Bigger and Smaller step through the objects under the point, from a list the core gives
(`pick/under`). A click is posted in picture pixels and the core decides what it
means; every pick is saved the moment it is made, and every POST ends in a
redirect back to the page, so nothing lives only in the browser. The script
(`picking.html`) does lookup, drawing and posting and decides nothing: the "add a
point to a pick" rule, the 12 px guard and every refusal are the core's
(`stages/pick.py`), and each refusal reaches the person as a sentence through
`web/app.py`'s handlers.

Add a spot and Remove a spot are modes of the tray: the next click on the
selected pick posts to `points` with "this" or "not this". Lasso is a third mode:
the script records a dragged line as picture-pixel points and posts it once
(`pick/lasso`); the face only checks that it is a list of number pairs and the
core (`stages/lasso.py`) finds the object. On a pick the lasso made, the two spot
modes post to `lasso-fix` (add or remove one piece) instead of `points`; the page
learns which from `selected_by_lasso`. Undo (`pick/undo`) takes back the newest
change to a pick's outline; the list lives in the core, and a redraw empties it.

Redraw redirects back here with `redrawn=1`; the page lists each pick's own
warnings and problems (`notes_for`), so the person sees what failed without
leaving the screen.
"""

from __future__ import annotations

import io
import json
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import Response

from pixelmagic.journey import StepKey
from pixelmagic.model import Decomposition, Direction, PickMethod, Point
from pixelmagic.stages.decompose import load_decomposition, promote_cutouts
from pixelmagic.stages.generate import load_generation
from pixelmagic.stages.lasso import fix_lasso, pick_lasso
from pixelmagic.stages.pick import (
    add_clicks,
    how_picked,
    load_picks,
    pick_points,
    pick_suggestions,
    rename_pick,
    render_suggestion_map,
    render_suggestion_outline,
    render_tint,
    suggestions_under,
    undo_last,
    unpick,
)
from pixelmagic.stages.redraw import live_pieces, notes_for, redraw_picks, unowned_notes
from pixelmagic.stages.suggest import PAGE_SIZE, load_suggestions, suggest_objects
from pixelmagic.store import check_run_id
from pixelmagic.web.deps import Workbench, page, require_direction, see_other, workbench
from pixelmagic.web.pixelcutting import pick_screen
from pixelmagic.web.timeline import frame_context

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]


def load_cut(
    bench: Workbench, direction: str, decomposition: str
) -> tuple[Direction, Decomposition]:
    """The direction and the decomposition a route was asked about."""
    found = require_direction(bench, direction)
    return found, load_decomposition(
        bench.store, found.name, check_run_id(decomposition, "cut-up run id")
    )


def _back(
    direction: str,
    decomposition: str,
    *,
    select: int | None = None,
    fixed: int | None = None,
    fresh: int | None = None,
    redrawn: bool = False,
) -> Response:
    """Send the browser back to the picking page, keeping its selection."""
    url = f"/directions/{direction}/decompositions/{decomposition}/pick"
    given = (
        ("select", select), ("fixed", fixed), ("fresh", fresh),
        ("redrawn", 1 if redrawn else None),
    )
    query = [f"{name}={value}" for name, value in given if value is not None]
    return see_other(url + ("?" + "&".join(query) if query else ""))


@router.post("/directions/{direction}/generations/{generation}/suggest")
def suggest(direction: str, generation: str, bench: Bench) -> Response:
    """Ask the segmenter for the objects on an approved picture and open the picking page."""
    found = require_direction(bench, direction)
    source = load_generation(
        bench.store, found.name, check_run_id(generation, "generation id")
    )
    result = suggest_objects(bench.store, found, source, bench.segmenter)
    return _back(found.name, result.id)


def _centre(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float]:
    """A box's centre as a percentage of the picture's width and height."""
    x0, y0, x1, y1 = box
    return round((x0 + x1) / 2 / width * 100, 2), round((y0 + y1) / 2 / height * 100, 2)


@router.get("/directions/{direction}/decompositions/{decomposition}/pick")
def picking(
    request: Request,
    direction: str,
    decomposition: str,
    bench: Bench,
    select: int | None = None,
    fixed: int | None = None,
    fresh: int | None = None,
    redrawn: int | None = None,
    done: Annotated[int | None, Query(alias="cut")] = None,
) -> Response:
    """The picture, the picks made so far and what to do next.

    `cut=1` is the redirect after a pixel cut; only a pixel-cut run reads it.

    `select` is the pick a spot click adds to; it defaults to the last one
    made. `fixed` is set by the redirect after a fix, so the page can say when
    that pick already has a redrawn piece that the fix left alone. `fresh` is
    the pick just made, so the page can flash it; `redrawn=1` says the person
    has just come back from a redraw.
    """
    found, cut = load_cut(bench, direction, decomposition)
    if cut.pixel_grid is not None:
        return pick_screen(request, bench, found, cut, select, fresh, done == 1)
    suggestions = load_suggestions(bench.store, found.name, cut.id)
    pick_set = load_picks(bench.store, found.name, cut.id)
    picks = pick_set.picks
    numbers = [pick.number for pick in picks]
    selected = select if select in numbers else (numbers[-1] if numbers else None)
    width, height = suggestions.picture_size
    chosen = next((pick for pick in picks if pick.number == selected), None)
    show_numbers = len(suggestions.suggestions) <= PAGE_SIZE
    held = {pick.suggestion for pick in picks}
    tags = []
    found_list = []
    for suggestion in suggestions.suggestions:
        if suggestion.number in held:
            continue
        if show_numbers:
            left, top = _centre(suggestion.box, width, height)
            tags.append({"number": suggestion.number, "left": left, "top": top})
        else:
            x0, y0, x1, y1 = suggestion.box
            found_list.append({"number": suggestion.number, "wide": x1 - x0, "high": y1 - y0})
    rows = []
    for pick in picks:
        pieces = [piece.name for piece in live_pieces(cut, pick)]
        left, top = _centre(pick.picture_box, width, height)
        rows.append(
            {
                "number": pick.number,
                "name": pick.name,
                "label": pick.name or f"Pick {pick.number}",
                "how": how_picked(pick),
                "lasso": pick.method is PickMethod.LASSO,
                "left": left,
                "top": top,
                "box": pick.picture_box,
                "piece": pieces[0] if pieces else None,
                "notes": [
                    {"problem": problem, "text": text}
                    for problem, text in (notes_for(cut, pick.name) if pick.name else [])
                ],
                "stale": pieces if pick.number == fixed else [],
            }
        )
    to_redraw = sum(1 for row in rows if row["piece"] is None)
    return page(
        request,
        "picking.html",
        direction=found,
        **frame_context(bench, found.name, cut.source_generation, StepKey.PICK, "Pick props"),
        decomposition=cut,
        service=bench.service_name,
        base=f"/directions/{found.name}/decompositions/{cut.id}",
        picture_src=(
            f"/directions/{found.name}/generations/{cut.source_generation}/image.png"
        ),
        width=width,
        height=height,
        found_any=bool(suggestions.suggestions),
        show_numbers=show_numbers,
        tags=tags,
        found_list=found_list,
        picks=rows,
        other_notes=[
            {"problem": problem, "text": text}
            for problem, text in unowned_notes(cut, [p.name for p in picks if p.name])
        ],
        selected=selected,
        selected_by_lasso=chosen is not None and chosen.method is PickMethod.LASSO,
        fresh=fresh,
        just_redrawn=redrawn == 1,
        can_undo=bool(pick_set.undo),
        to_redraw=to_redraw,
        all_redrawn=bool(rows) and to_redraw == 0,
        continue_url=f"/directions/{found.name}/decompositions/{cut.id}/pick/continue",
    )


@router.get(
    "/directions/{direction}/decompositions/{decomposition}/picks/{number}/tint.png"
)
def tint(direction: str, decomposition: str, number: int, bench: Bench) -> Response:
    """One pick's area as a picture-sized tinted layer, drawn fresh each time."""
    found, cut = load_cut(bench, direction, decomposition)
    buffer = io.BytesIO()
    render_tint(bench.store, found, cut.id, number).save(buffer, format="PNG")
    # A fix changes the layer under the same address; never show a stale one.
    return Response(
        buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"}
    )


@router.get("/directions/{direction}/decompositions/{decomposition}/pick/map.png")
def suggestion_map(direction: str, decomposition: str, bench: Bench) -> Response:
    """The "which suggestion is here" lookup picture the hover reads."""
    found, cut = load_cut(bench, direction, decomposition)
    buffer = io.BytesIO()
    render_suggestion_map(bench.store, found, cut.id).save(buffer, format="PNG")
    # It changes with every pick; never show a stale one.
    return Response(
        buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"}
    )


@router.get("/directions/{direction}/decompositions/{decomposition}/pick/under")
def suggestions_under_point(
    direction: str, decomposition: str, bench: Bench, x: int, y: int
) -> list[int]:
    """The suggestions under a point, smallest first, for the scroll wheel to step through."""
    found, cut = load_cut(bench, direction, decomposition)
    return suggestions_under(bench.store, found, cut.id, x, y)


@router.get(
    "/directions/{direction}/decompositions/{decomposition}/suggestions/{number}/outline.png"
)
def suggestion_outline(
    direction: str, decomposition: str, number: int, bench: Bench
) -> Response:
    """One suggestion's edge as a picture-sized transparent layer."""
    found, cut = load_cut(bench, direction, decomposition)
    buffer = io.BytesIO()
    render_suggestion_outline(bench.store, found, cut.id, number).save(buffer, format="PNG")
    # A suggestion's outline never changes, so the browser may keep it.
    return Response(
        buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.post(
    "/directions/{direction}/decompositions/{decomposition}/pick/suggestion/{number}"
)
def pick_suggestion(
    direction: str,
    decomposition: str,
    number: int,
    bench: Bench,
) -> Response:
    """Pick one of the suggestions as it is."""
    found, cut = load_cut(bench, direction, decomposition)
    (made,) = pick_suggestions(bench.store, found, cut.id, [number])
    return _back(found.name, cut.id, select=made.number, fresh=made.number)


@router.post("/directions/{direction}/decompositions/{decomposition}/pick/click")
def pick_click(
    direction: str,
    decomposition: str,
    bench: Bench,
    x: Annotated[int, Form()],
    y: Annotated[int, Form()],
) -> Response:
    """Pick whatever object is under one click, in picture pixels."""
    found, cut = load_cut(bench, direction, decomposition)
    made = pick_points(
        bench.store, found, cut.id, [Point(x=x, y=y, label=1)], segmenter=bench.segmenter
    )
    return _back(found.name, cut.id, select=made.number, fresh=made.number)


@router.post(
    "/directions/{direction}/decompositions/{decomposition}/picks/{number}/points"
)
def fix_points(
    direction: str,
    decomposition: str,
    number: int,
    bench: Bench,
    x: Annotated[int, Form()],
    y: Annotated[int, Form()],
    label: Annotated[int, Form(ge=0, le=1)],
) -> Response:
    """Add a "this" (1) or "not this" (0) click to a pick and outline it again."""
    found, cut = load_cut(bench, direction, decomposition)
    add_clicks(
        bench.store, found, cut.id, number, [Point(x=x, y=y, label=label)],
        segmenter=bench.segmenter,
    )
    return _back(found.name, cut.id, select=number, fixed=number)


def _read_loop(text: str) -> list[tuple[int, int]]:
    """The dragged line from its form field: a JSON list of `[x, y]` number pairs."""
    refusal = ValueError("The line was not a list of [x, y] points, so nothing was picked.")
    try:
        raw = json.loads(text)
        pairs = [tuple(pair) for pair in raw]
    except (ValueError, TypeError) as error:
        raise refusal from error
    if not all(
        len(pair) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pair)
        for pair in pairs
    ):
        raise refusal
    try:
        return [(round(x), round(y)) for x, y in pairs]
    except (OverflowError, ValueError) as error:  # JSON allows NaN and 1e999 (infinity)
        raise refusal from error


@router.post("/directions/{direction}/decompositions/{decomposition}/pick/lasso")
def pick_by_lasso(
    direction: str,
    decomposition: str,
    bench: Bench,
    loop: Annotated[str, Form()],
) -> Response:
    """Pick the object inside a dragged line, given in picture pixels."""
    found, cut = load_cut(bench, direction, decomposition)
    made = pick_lasso(
        bench.store, found, cut.id, _read_loop(loop), segmenter=bench.segmenter
    )
    return _back(found.name, cut.id, select=made.number, fresh=made.number)


@router.post(
    "/directions/{direction}/decompositions/{decomposition}/picks/{number}/lasso-fix"
)
def fix_by_lasso(
    direction: str,
    decomposition: str,
    number: int,
    bench: Bench,
    x: Annotated[int, Form()],
    y: Annotated[int, Form()],
    label: Annotated[int, Form(ge=0, le=1)],
) -> Response:
    """Add (1) or remove (0) the piece under a click on a pick the lasso made."""
    found, cut = load_cut(bench, direction, decomposition)
    fix_lasso(
        bench.store, found, cut.id, number, x, y, add=label == 1, segmenter=bench.segmenter
    )
    return _back(found.name, cut.id, select=number, fixed=number)


@router.post(
    "/directions/{direction}/decompositions/{decomposition}/picks/{number}/rename"
)
def rename(
    direction: str,
    decomposition: str,
    number: int,
    bench: Bench,
    name: Annotated[str, Form()],
) -> Response:
    """Name a pick before it is redrawn; the core refuses it once a piece exists."""
    found, cut = load_cut(bench, direction, decomposition)
    rename_pick(bench.store, found, cut.id, number, name)
    return _back(found.name, cut.id, select=number)


@router.post("/directions/{direction}/decompositions/{decomposition}/pick/undo")
def undo(direction: str, decomposition: str, bench: Bench) -> Response:
    """Take back the newest change to a pick's outline; the core says if there is none."""
    found, cut = load_cut(bench, direction, decomposition)
    number = undo_last(bench.store, found, cut.id)
    return _back(found.name, cut.id, select=number)


@router.post(
    "/directions/{direction}/decompositions/{decomposition}/picks/{number}/remove"
)
def remove(
    direction: str,
    decomposition: str,
    number: int,
    bench: Bench,
) -> Response:
    """Take a pick away. Its number is never handed out again."""
    found, cut = load_cut(bench, direction, decomposition)
    unpick(bench.store, found, cut.id, number)
    return _back(found.name, cut.id)


@router.post("/directions/{direction}/decompositions/{decomposition}/redraw")
def redraw(
    direction: str,
    decomposition: str,
    bench: Bench,
    only: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """Redraw the picks that have no piece yet, or exactly the ones named in `only`.

    Takes a minute or more with a real service; the page waits. The person
    lands back on the picking page, which shows each pick's own warnings.
    """
    found, cut = load_cut(bench, direction, decomposition)
    redraw_picks(
        bench.store,
        found,
        cut.id,
        service=bench.service(),
        only=only or None,
    )
    return _back(found.name, cut.id, redrawn=True)


@router.post("/directions/{direction}/decompositions/{decomposition}/pick/continue")
def continue_to_surfaces(direction: str, decomposition: str, bench: Bench) -> Response:
    """Keep the redrawn props as assets and move on to the picture's surfaces.

    The core refuses (`OldPieceRefused`) a run that still holds a piece from a
    road that is gone, before anything is written; the handlers say so.
    """
    found, cut = load_cut(bench, direction, decomposition)
    promote_cutouts(bench.store, found, cut)
    return see_other(f"/directions/{found.name}/generations/{cut.source_generation}/surfaces")
