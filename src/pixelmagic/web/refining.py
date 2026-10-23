"""The Refine step (step 6): the scene refined with the house recipe, its receipt, and the options.

The page shows the kept scene, refines it with one button, and says in plain
sentences what the refiner did. Everything the old settings screen could do is
still here, folded under "More options": the refiner's settings, the scene's
pitch, the scene's pixel grid readings and a prop's anchor. This module only reads and posts;
every decision stays in the core's stage functions.

The knobs on this screen come from `PixelRefinerCli.options()` and never from
the stand-in's (ticket M2.1): the stand-in answers out of the contract's own
constants, so a screen built on it would offer keys the real tool does not have
and hide every palette it does. When the real tool cannot be reached, or is not
turned on (the stand-in is doing the work), the screen says so and offers
nothing rather than guessing.

The form's own field names carry the refiner's vocabulary — one field per key
it said it accepts, named `quick.dithering` or `advanced.colorCount` — so this
module never holds a second copy of the list.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData

from pixelmagic import journey
from pixelmagic.journey import StepKey
from pixelmagic.model import MAX_PITCH, MIN_PITCH, AssetRecord, RefinerSettings
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.stages.decompose import scene_holding, set_anchor, set_scene_pitch
from pixelmagic.stages.export import ACTIONABLE_WARNINGS, ASSET_FILE
from pixelmagic.stages.refine import (
    PIXELMAGIC_REFUSAL_CODES,
    REFINED_FILE,
    WRONG_SIZE,
    grid_needs_attention,
    load_report,
    refine_asset,
    refine_direction,
    require_asset,
    set_refiner_defaults,
)
from pixelmagic.store import Store
from pixelmagic.web.deps import Workbench, page, require_direction, see_other, workbench
from pixelmagic.web.timeline import kept_scene_context

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]


def _asset_record(store: Store, direction_name: str, name: str) -> AssetRecord | None:
    """The asset's own sidecar, or None when it will not read back.

    The same rule `asset_rows` applies to the receipt: one damaged
    `asset.json` costs its own row the anchor and scene-pitch forms, not the
    page.
    """
    try:
        return store.read_model(store.asset_dir(direction_name, name) / ASSET_FILE, AssetRecord)
    except (OSError, ValueError):
        return None


def warning_sentences(codes: list[str]) -> list[str]:
    """Every warning code, in plain words where the core has one for it.

    `ACTIONABLE_WARNINGS` — `stages.export`'s own public table of code to
    sentence — is reused rather than repeated here (CLAUDE.md: a rule every
    face needs lives in one place); a code with no entry there is shown as
    it is, exactly as it always has been.
    """
    return [ACTIONABLE_WARNINGS.get(code, code) for code in codes]


def asset_rows(store: Store, direction_name: str) -> list[dict[str, object]]:
    """Every asset, with its receipt when it has one.

    A receipt that will not read back is skipped rather than raised: one
    damaged `refine.json` must not cost the other nineteen assets their row,
    which is the same rule `export` follows.
    """
    rows: list[dict[str, object]] = []
    for name in store.list_assets(direction_name):
        try:
            report = load_report(store, direction_name, name)
        except (OSError, ValueError):
            report = None
        record = _asset_record(store, direction_name, name)
        refined_file = store.asset_dir(direction_name, name) / REFINED_FILE
        rows.append(
            {
                "name": name,
                "report": report,
                "refined": refined_file.exists(),
                # The refined picture keeps one path, so the file's modification
                # time rides along as a version: refining again shows the new one.
                "refined_url": (
                    f"/directions/{direction_name}/assets/{name}/refined.png"
                    f"?v={refined_file.stat().st_mtime_ns}"
                    if refined_file.exists()
                    else None
                ),
                # An old tile or prop left as it is has an old receipt that can
                # no longer be acted on (there is no pitch to set); the
                # same goes for a rebuilt prop that still has a receipt from
                # before it was rebuilt (`RebuiltPieceRefused`).
                "attention": bool(report and grid_needs_attention(report, record)),
                # Whether this asset's error, if any, is PixelMagic refusing
                # to send it to the refiner rather than the refiner refusing
                # it — read off the one list in `stages.refine` so the page
                # never repeats it (CLAUDE.md: logic never goes in a face).
                "pixelmagic_refusal": bool(
                    report and report.error_code in PIXELMAGIC_REFUSAL_CODES
                ),
                # A `WRONG_SIZE` receipt is neither: the refiner succeeded and
                # PixelMagic did not keep the picture (`wrong_size_sentence`).
                "wrong_size": bool(report and report.error_code == WRONG_SIZE),
                "record": record,
                # A rebuilt prop is the one whose asset record carries its grid:
                # how refine and export recognise it, and what the count uses.
                "rebuilt": bool(record and record.grid_rebuild is not None),
                # A recipe tile is drawn by PixelMagic and never refined; its
                # asset record carries the recipe, which is how it is told apart.
                "recipe": bool(record and record.recipe is not None),
            }
        )
    return rows


def _layer(form: FormData, prefix: str) -> dict[str, str]:
    """Read the settings fields whose names start with `prefix`.

    A field left blank means "leave this one alone" and is dropped, so a form
    that offers every key the refiner has does not force a value for each one.
    `multi_items()` and an explicit overwrite, rather than `form.items()`,
    because a field name repeated in the same POST must keep its last value on
    purpose — not by accident of whichever value a plain mapping view happens
    to keep.
    """
    layer: dict[str, str] = {}
    for name, value in form.multi_items():
        if name.startswith(prefix) and isinstance(value, str) and value.strip():
            layer[name[len(prefix) :]] = value.strip()
    return layer


@router.get("/directions/{direction}/refine")
def screen(request: Request, direction: str, bench: Bench) -> Response:
    """The kept scene, its receipt, and the options folded under it."""
    found = require_direction(bench, direction)
    rows = asset_rows(bench.store, found.name)
    generation = journey.followed_scene(bench.store, found.name)
    scene_name = scene_holding(bench.store, found.name, generation) if generation else None
    return page(
        request,
        "refine.html",
        direction=found,
        **kept_scene_context(bench, found.name, StepKey.REFINE, "Refine"),
        view=bench.refiner_options(),
        rows=rows,
        # The kept scene is the only asset the refiner works on; its row has the
        # picture, the receipt and the pitch the screen is built around. It is
        # the one the timeline follows, so an older kept scene does not take over.
        scene=next((row for row in rows if row["name"] == scene_name), None),
        rebuilt_count=sum(1 for row in rows if row["rebuilt"]),
        recipe_count=sum(1 for row in rows if row["recipe"]),
        # Guardian minor 4: the template used to compare `refiner == "stub"`,
        # a bare string literal drifting apart from `StubRefiner.name` the
        # moment either one changed. Comparing against the class's own `name`
        # here means a rename can never make the "not real pixel art"
        # warning silently disappear.
        stand_in=bench.refiner.name == StubRefiner.name,
        pitch_bounds=(MIN_PITCH, MAX_PITCH),
        warning_sentences=warning_sentences,
    )


@router.post("/directions/{direction}/scene")
def choose_scene(
    direction: str,
    bench: Bench,
    generation: Annotated[str, Form()],
    page_name: Annotated[str, Form(alias="return")],
) -> Response:
    """Store which kept scene Refine and Export follow, then go back to the page it was chosen on.

    The page to return to is one of the two words `refine` or `export`, never a
    URL from the form. The core refuses a generation that is not a kept scene.
    """
    found = require_direction(bench, direction)
    if page_name not in (StepKey.REFINE.value, StepKey.EXPORT.value):
        raise ValueError("The scene can only be chosen from the Refine or Export page.")
    journey.choose_scene(bench.store, found.name, generation)
    return see_other(f"/directions/{found.name}/{page_name}")


@router.post("/directions/{direction}/refine")
async def run(request: Request, direction: str, bench: Bench) -> Response:
    """Refine the whole direction, or one asset with settings of its own.

    Settings with no asset named become the direction's new defaults, exactly
    as they do on the command line — `set_refiner_defaults` layers them onto
    what is already there and refuses a block the refiner would not accept, so
    a bad combination never becomes a default that breaks every later run.

    This is the only `async def` route on the page, because it is the only one
    that has to read the form itself: the quick and advanced field names are
    invented from the refiner's own vocabulary, so no fixed signature can
    declare them. That makes it the one route FastAPI runs *on the event loop*
    rather than in a threadpool — every other route here is a plain `def` and
    gets a worker thread for free. Refining is also by far the heaviest thing
    this page does: with `--refiner cli` it spawns the real binary over up to
    sixty-four assets in a batch, and for the whole of that a blocked loop
    would serve no image, no second tab and no way to see what is happening.
    So the stage calls go through `run_in_threadpool`, which puts this route
    back on the same footing as every other one.
    """
    found = require_direction(bench, direction)
    form = await request.form()
    given = RefinerSettings(
        preset=str(form.get("preset") or "").strip() or None,
        quick=_layer(form, "quick."),
        advanced=_layer(form, "advanced."),
    )
    asset = str(form.get("asset") or "").strip()
    if asset:
        await run_in_threadpool(
            refine_asset,
            bench.store,
            found,
            require_asset(bench.store, found.name, asset),
            refiner=bench.refiner,
            override=given,
        )
    else:
        if given.preset or given.quick or given.advanced:
            found = set_refiner_defaults(bench.store, found, given)
        await run_in_threadpool(
            refine_direction,
            bench.store,
            found,
            refiner=bench.refiner,
        )
    return see_other(f"/directions/{found.name}/refine")


@router.post("/directions/{direction}/refine/anchor")
def anchor(
    direction: str,
    bench: Bench,
    asset: Annotated[str, Form()],
    x: Annotated[int, Form(ge=0)],
    y: Annotated[int, Form(ge=0)],
) -> Response:
    """Say where one prop stands, in art pixels of its target size.

    Spec locked decision 6: the anchor is editable and never guessed from
    what the prop is. The click on the refined picture fills `x` and `y`;
    a person can type them too. `set_anchor` does its own lookup by name —
    the same `require_asset` the scene-pitch route and the MCP tools already
    let the core call once, rather than this route looking the asset up a
    second time before handing it the name unchanged. Its `ValueError` is
    `web.app`'s "That will not work" page; a negative number never reaches
    it, because `Form(ge=0)` refuses it first.
    """
    found = require_direction(bench, direction)
    set_anchor(bench.store, found.name, asset, (x, y))
    return see_other(f"/directions/{found.name}/refine")


@router.post("/directions/{direction}/refine/scene-pitch")
def scene_pitch(
    direction: str,
    bench: Bench,
    asset: Annotated[str, Form()],
    value: Annotated[float, Form()],
) -> Response:
    """Overrule a kept scene's own pitch reading with one number.

    The web page's counterpart to the CLI's `set-scene-pitch` and the MCP's
    `set_scene_pitch` tool: a scene has no cut-up run to carry a pitch on,
    so it gets this route, right beside
    `/refine/anchor` since both edit one asset's own sidecar from this
    screen. `set_scene_pitch` does everything else — the lookup by name
    (`require_asset`'s own "no asset called" sentence), the bound check
    (`check_pitch`'s `ValueError`, `web.app`'s "That will not work" page),
    and refusing an asset that is not a scene, all in the core's own words.
    """
    found = require_direction(bench, direction)
    set_scene_pitch(bench.store, found.name, asset, value)
    return see_other(f"/directions/{found.name}/refine")
