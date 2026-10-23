"""The pixel cut on the web page: the size step, and the pixel-cut picking screen.

A picture that is already pixel art is cut into props from its own pixels. The size
step (`/pixel-cut`) asks how big one art pixel is, or lets the core find it, and starts a
run with `stages/pixelcut.start_pixel_cut`; the person then lands on the run's ordinary
picking address, where `picking.picking` hands a run that carries a `pixel_grid` to
`pick_screen` here. Everything that matters is the core's: what a usable size is, whether
the finder was sure, the sentence for an unsure one (stored on the run), and what a pick
is. The page only reads a form and shows what the core kept.

On the screen a person drags a box round a prop (`pick/box`, the only route a box needs
here: the core's `pick_box`), fixes a spot with the picking page's own `picks/{n}/points`
route, and cuts every pick into a prop (`cut`, the core's `cut_pieces`). After the cut the
same address, with `?cut=1`, shows what was cut; "Continue to Surfaces" is the picking
page's own `pick/continue`. The routes that change picks otherwise (`picks/{n}/rename`,
`remove`, `pick/undo`) are the picking page's and are not repeated here. The page script
(`static/pixelcut_pick.js`) records drags and clicks in picture pixels and posts them; the
meaning of a box, the 12 px guard, the cut and every refusal are the core's, and each
refusal reaches the person as a sentence through `web/app.py`'s handlers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import Response

from pixelmagic.journey import StepKey
from pixelmagic.model import Decomposition, Direction
from pixelmagic.stages.decompose import load_decomposition
from pixelmagic.stages.generate import load_generation
from pixelmagic.stages.pick import how_picked, load_picks, pick_box
from pixelmagic.stages.pixelcut import cut_pieces, piece_names, start_pixel_cut, unsure_note
from pixelmagic.stages.redraw import live_pieces, notes_for, unowned_notes
from pixelmagic.store import check_run_id
from pixelmagic.web.cleaning import number
from pixelmagic.web.deps import Workbench, page, require_direction, see_other, workbench
from pixelmagic.web.timeline import frame_context

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]


def _read_size(size: str) -> float | None:
    """The typed size, or None for an empty box (the core finds it)."""
    text = size.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError("The pixel size must be a number, such as 4 or 8.5.") from None


@router.get("/directions/{direction}/generations/{generation}/pixel-cut")
def size_step(request: Request, direction: str, generation: str, bench: Bench) -> Response:
    """The size step: find the pixel size or type it. Starts nothing."""
    found = require_direction(bench, direction)
    generation = check_run_id(generation, "generation id")
    picture = load_generation(bench.store, found.name, generation)
    return page(
        request,
        "pixelcut.html",
        direction=found,
        **frame_context(bench, found.name, generation, StepKey.PICK, "Cut props"),
        picture=picture,
        generation_url=f"/directions/{found.name}/generations/{generation}",
        action=f"/directions/{found.name}/generations/{generation}/pixel-cut",
    )


@router.post("/directions/{direction}/generations/{generation}/pixel-cut")
def start(
    direction: str,
    generation: str,
    bench: Bench,
    size: Annotated[str, Form()] = "",
) -> Response:
    """Start a pixel cut and go to its picking page.

    An empty size means the core finds it. A size that is not a number is refused here;
    one outside the picture, or a picture nobody approved, is refused by the core before
    anything is written.
    """
    found = require_direction(bench, direction)
    generation = check_run_id(generation, "generation id")
    source = load_generation(bench.store, found.name, generation)
    run = start_pixel_cut(bench.store, found, source, size=_read_size(size))
    return see_other(f"/directions/{found.name}/decompositions/{run.id}/pick")


def _run(bench: Workbench, direction: str, run: str) -> tuple[Direction, Decomposition]:
    """The direction and the pixel-cut run a route was asked about."""
    found = require_direction(bench, direction)
    return found, load_decomposition(bench.store, found.name, check_run_id(run, "cut-up run id"))


def _picking_url(direction: str, run: str, query: str = "") -> str:
    """The run's picking address, with an optional query string."""
    return f"/directions/{direction}/decompositions/{run}/pick{query}"


@router.post("/directions/{direction}/decompositions/{run}/pick/box")
def pick_by_box(
    direction: str,
    run: str,
    bench: Bench,
    left: Annotated[int, Form()],
    top: Annotated[int, Form()],
    right: Annotated[int, Form()],
    bottom: Annotated[int, Form()],
) -> Response:
    """Pick the object inside a dragged box, given in picture pixels.

    A box with no area, one outside the picture and an object the finder cannot find are
    the core's refusals; nothing is saved then.
    """
    found, cut = _run(bench, direction, run)
    made = pick_box(
        bench.store, found, cut.id, (left, top, right, bottom), segmenter=bench.segmenter
    )
    flash = f"?select={made.number}&fresh={made.number}"
    return see_other(_picking_url(found.name, cut.id, flash))


@router.post("/directions/{direction}/decompositions/{run}/cut")
def cut_props(direction: str, run: str, bench: Bench) -> Response:
    """Cut every pick into a prop from the picture's own pixels (free) and show the result.

    A pick that already has a piece keeps its name and the piece is replaced; the core
    refuses a pick with no whole cell inside it before anything is written.
    """
    found, decomposition = _run(bench, direction, run)
    cut_pieces(bench.store, found, decomposition.id)
    return see_other(_picking_url(found.name, decomposition.id, "?cut=1"))


def pick_screen(
    request: Request,
    bench: Workbench,
    found: Direction,
    cut: Decomposition,
    select: int | None,
    fresh: int | None = None,
    done: bool = False,
) -> Response:
    """The pixel-cut picking screen for a run that carries a `pixel_grid`.

    `select` is the pick shown as selected; it defaults to the newest and `fresh` is the
    pick just made (it is flashed). A pick is labelled with the name the core will give it
    when it is cut (`pixelcut.piece_names`). With `done` (the redirect after the cut) the
    page shows the props cut instead, when at least one pick has a piece.
    """
    grid = cut.pixel_grid
    width, height = grid.picture_size
    pick_set = load_picks(bench.store, found.name, cut.id)
    numbers = [pick.number for pick in pick_set.picks]
    selected = select if select in numbers else (numbers[-1] if numbers else None)
    names = piece_names(bench.store, found, cut, pick_set.picks, pick_set)
    rows = []
    for pick in pick_set.picks:
        pieces = live_pieces(cut, pick)
        grid_cut = pieces[0].pixel_cut if pieces else None
        rows.append(
            {
                "number": pick.number,
                "label": names[pick.number],
                "piece_size": grid_cut.native_size if grid_cut else None,
                "cut": bool(pieces),
                "how": how_picked(pick),
                "box": pick.picture_box,
                "notes": [
                    {"problem": problem, "text": text}
                    for problem, text in (notes_for(cut, pick.name) if pick.name else [])
                ],
            }
        )
    base = f"/directions/{found.name}/decompositions/{cut.id}"
    unsure = unsure_note(cut)
    cut_rows = [row for row in rows if row["cut"]]
    showing_done = done and bool(cut_rows)
    return page(
        request,
        "pixelcut_done.html" if showing_done else "pixelcut_pick.html",
        direction=found,
        **frame_context(bench, found.name, cut.source_generation, StepKey.PICK, "Pick props"),
        decomposition=cut,
        base=base,
        size_url=f"/directions/{found.name}/generations/{cut.source_generation}/pixel-cut",
        picture_src=f"/directions/{found.name}/generations/{cut.source_generation}/image.png",
        width=width,
        height=height,
        size_text=number(grid.pixel_size),
        typed=grid.size_source == "typed",
        native=grid.pixel_size == 1,
        unsure=unsure,
        picks=rows,
        other_notes=[
            {"problem": problem, "text": text}
            for problem, text in unowned_notes(cut, [p.name for p in pick_set.picks if p.name])
            if text != unsure
        ],
        selected=selected,
        fresh=fresh,
        can_undo=bool(pick_set.undo),
        cut_rows=cut_rows,
        any_cut=bool(cut_rows),
        continue_url=f"{base}/pick/continue",
        pick_url=f"{base}/pick",
        left_top=[
            {
                "label": row["label"],
                "left": round(row["box"][0] / width * 100, 2),
                "top": round(row["box"][1] / height * 100, 2),
            }
            for row in cut_rows
        ],
    )
