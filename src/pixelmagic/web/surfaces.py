"""The surfaces screen: floors and walls drawn on the picture, one at a time.

A person draws shapes over a surface and a small box on a plain patch of it;
the page only draws and posts. Every rule is the core's (`stages/decompose.py`
and `naming.parse_surface`): which kinds there are, how big a box must be, what
the colours are, what a name may be. The page posts one surface in the typed
form the command line uses, and each refusal comes back as the core's own
sentence in a small JSON body (`{"error": ...}`) for the script to show, never
as a traceback.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import ValidationError

from pixelmagic.journey import StepKey
from pixelmagic.model import ReviewStatus
from pixelmagic.naming import check_name, parse_box, parse_surface
from pixelmagic.stages.decompose import (
    DrawnSurface,
    draw_surface,
    drawn_surfaces,
    read_box_shades,
    remove_surface,
)
from pixelmagic.stages.generate import load_generation
from pixelmagic.stages.recipes import MIN_BOX_PIXELS, RECIPE_KINDS, RecipeSeamBroken
from pixelmagic.store import AlreadyExists, check_run_id
from pixelmagic.web.deps import Workbench, page, require_direction, see_other, workbench
from pixelmagic.web.timeline import frame_context

LOGGER = logging.getLogger(__name__)

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]


def _base(direction: str, generation: str) -> str:
    """The URL every route of one picture's surfaces screen starts with."""
    return f"/directions/{direction}/generations/{generation}/surfaces"


def _json_error(error: Exception) -> JSONResponse:
    """Turn a failure into the `{"error": sentence}` body the page's script shows.

    The core's refusals keep their own sentence: `FileNotFoundError` is a 404,
    `AlreadyExists` a 409, any other `ValueError` (which holds `UnusableName`,
    `RecipeRefused`, `NoPicks` and the not-approved refusal) a 400, and any
    other `OSError` a 400 "A file could not be read". This table must agree with
    the handlers in `web/app.py`, which answer the same errors as pages. A broken
    recipe is PixelMagic's bug and says so. Anything else is not explained, as
    in `web/app.py`'s catch-all: its message is an internal detail.
    """
    if isinstance(error, RecipeSeamBroken):
        LOGGER.error("recipe seam check failed: %s", error)
        sentence = (
            f"The {error.kind} recipe for the surface {error.surface!r} made a set of "
            "tiles that do not fit together. This is a bug in PixelMagic, not "
            "something you drew."
        )
        return JSONResponse({"error": sentence}, status_code=500)
    if isinstance(error, FileNotFoundError):
        return JSONResponse({"error": str(error)}, status_code=404)
    if isinstance(error, AlreadyExists):
        return JSONResponse({"error": str(error)}, status_code=409)
    if isinstance(error, ValueError) and not isinstance(error, ValidationError):
        return JSONResponse({"error": str(error)}, status_code=400)
    if isinstance(error, OSError):
        return JSONResponse({"error": f"A file could not be read: {error}"}, status_code=400)
    LOGGER.exception("unhandled error on the surfaces screen")
    return JSONResponse({"error": "Something went wrong."}, status_code=500)


def _drawn_row(found_name: str, generation: str, surface: DrawnSurface) -> dict[str, object]:
    """One drawn surface as the page and its script read it, with its two links."""
    base = _base(found_name, generation)
    return {
        "name": surface.name,
        "text": surface.text,
        "kind": surface.kind,
        "shapes": surface.shapes,
        "box": list(surface.box),
        "shades": surface.shades,
        "remove_url": f"{base}/{quote(surface.name)}/remove",
        "edit_url": f"{base}?edit={quote(surface.name)}",
    }


@router.get("/directions/{direction}/generations/{generation}/surfaces")
def show(
    request: Request,
    direction: str,
    generation: str,
    bench: Bench,
    edit: str | None = None,
    drawn: str | None = None,
) -> Response:
    """The picture with the surfaces already drawn on it and the editor for one more.

    `edit` names a drawn surface whose geometry goes into the editor; an
    unknown name is a 404 sentence. `drawn` names the surface just drawn, so
    the page can offer "Continue to Refine" and highlight its row. A picture
    that is not approved gets a sentence and no drawing: whether it may be
    drawn on is the core's call (`decompose`), the page only reads the status
    to choose what to show.
    """
    found = require_direction(bench, direction)
    record = load_generation(bench.store, found.name, check_run_id(generation, "generation id"))
    base = _base(found.name, record.request.id)
    with Image.open(record.image_path) as picture:
        width, height = picture.size
    rows = [
        _drawn_row(found.name, record.request.id, surface)
        for surface in drawn_surfaces(bench.store, found.name, record.request.id)
    ]
    editing = None
    if edit is not None:
        editing = next((row for row in rows if row["name"] == edit), None)
        if editing is None:
            raise FileNotFoundError(f"no surface called {edit!r} is drawn on this picture")
        editing = {key: editing[key] for key in ("name", "text", "kind", "box", "shapes")}
    # `<` is escaped so a surface's words can never close the script tag they sit in.
    geometry = json.dumps({"drawn_count": len(rows), "editing": editing}).replace("<", "\\u003c")
    return page(
        request,
        "surfaces.html",
        direction=found,
        **frame_context(bench, found.name, record.request.id, StepKey.SURFACES, "Surfaces"),
        generation=record.request.id,
        approved=record.review.status is ReviewStatus.APPROVED,
        generation_url=f"/directions/{found.name}/generations/{record.request.id}",
        picture_url=f"/directions/{found.name}/generations/{record.request.id}/image.png",
        picture_width=width,
        picture_height=height,
        kinds=list(RECIPE_KINDS),
        min_box_pixels=MIN_BOX_PIXELS,
        drawn=rows,
        editing=editing,
        just_drawn=drawn,
        geometry_json=geometry,
        colours_url=f"{base}/colours",
        draw_url=f"{base}/draw",
        refine_url=f"/directions/{found.name}/refine",
    )


@router.post("/directions/{direction}/generations/{generation}/surfaces/colours")
def colours(
    direction: str,
    generation: str,
    bench: Bench,
    kind: Annotated[str, Form()],
    box: Annotated[str, Form()],
    name: Annotated[str, Form()] = "",
) -> Response:
    """The colours a recipe would read from a box, as `{"shades": [...], "shares": [...]}`.

    Nothing is drawn or written. Refusals come back as `{"error": sentence}`.
    `name` is the typed name of the surface, used only in the sentences so they
    read like the draw route's; empty, they say "this surface".
    """
    try:
        found = require_direction(bench, direction)
        source = load_generation(bench.store, found.name, check_run_id(generation, "generation id"))
        shades = read_box_shades(
            found, source, kind, parse_box(box), name.strip() or "this surface"
        )
    except Exception as error:
        return _json_error(error)
    return JSONResponse(
        {
            "shades": [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in shades],
            "shares": [share for _, share in shades],
        }
    )


@router.post("/directions/{direction}/generations/{generation}/surfaces/draw")
def draw(
    direction: str,
    generation: str,
    bench: Bench,
    surface: Annotated[str, Form()],
    replaces: Annotated[str, Form()] = "",
) -> Response:
    """Draw one surface, given in the typed form, and say where to go next.

    Answers `{"next": url}` on success, the picture's surfaces page with the
    new surface marked as just drawn; any refusal is `{"error": sentence}`.
    `replaces` is the slug of the drawn surface being edited, empty for a new
    one; what it means is the core's (`draw_surface`).
    """
    try:
        found = require_direction(bench, direction)
        source = load_generation(bench.store, found.name, check_run_id(generation, "generation id"))
        parsed = parse_surface(surface)
        draw_surface(bench.store, found, source, parsed, replaces=replaces.strip() or None)
    except Exception as error:
        return _json_error(error)
    return JSONResponse(
        {"next": f"{_base(found.name, source.request.id)}?drawn={quote(parsed.name)}"}
    )


@router.post("/directions/{direction}/generations/{generation}/surfaces/{name}/remove")
def remove(direction: str, generation: str, name: str, bench: Bench) -> Response:
    """Take a drawn surface off the picture and show the screen again.

    A surface that is already gone (a stale page, a second tab) is the state the
    person asked for, so it also goes back to the screen: an error page here would
    sit at this POST-only address, and reloading it would send the remove again.
    """
    found = require_direction(bench, direction)
    source = load_generation(bench.store, found.name, check_run_id(generation, "generation id"))
    surface = check_name(name, "surface")
    drawn = drawn_surfaces(bench.store, found.name, source.request.id)
    if all(row.name != surface for row in drawn):
        LOGGER.warning(
            "remove of %s on %s/%s: surface already gone", name, found.name, source.request.id
        )
    else:
        remove_surface(bench.store, found, source.request.id, surface)
    return see_other(_base(found.name, source.request.id))
