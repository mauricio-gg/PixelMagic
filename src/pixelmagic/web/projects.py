"""The home, the new-direction form and the direction page: create, rename, look at."""

from __future__ import annotations

import shlex
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import Response

from pixelmagic import registry
from pixelmagic.directions import start_direction
from pixelmagic.journey import direction_summary, order_for_home
from pixelmagic.model import DEFAULT_CELL_SIZE, MAX_CELL_SIZE, MIN_CELL_SIZE, Direction
from pixelmagic.naming import check_name, typed_slug
from pixelmagic.stages.generate import load_generation
from pixelmagic.store import Store
from pixelmagic.web.deps import Workbench, page, require_direction, see_other, workbench
from pixelmagic.web.timeline import latest_scene_context

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]

#: Where a new direction can go straight on to, as the tail of its URL.
THEN_PAGES = {"generate": "/generate", "import": "/generate#import"}


def direction_rows(store: Store) -> tuple[list[Direction], list[str]]:
    """Every direction whose sidecar reads back, and the names of those that do not.

    One hand-edited `direction.json` must not hide every direction sorted
    after it — the same argument `cli.project_list` makes — so a broken one is
    named on the page instead of ending the listing.

    `ValueError` and not `ValidationError`, which is the same correction
    046d920 made to `cli.project_list` and for the same reason: a sidecar that
    is not UTF-8 fails on the *decode*, as a `UnicodeDecodeError`, before
    pydantic ever sees it. Catching only `ValidationError` lets that one
    escape to the `ValueError` error page and turns the whole front page into
    a 400 that does not even name the file that is broken. `ValidationError`
    is a subclass of `ValueError`, so this catches both.
    """
    good: list[Direction] = []
    broken: list[str] = []
    for name in store.list_directions():
        try:
            good.append(store.load_direction(name))
        except (OSError, ValueError):
            broken.append(name)
    return good, broken


def generation_rows(
    store: Store, direction_name: str, *, limit: int = 20
) -> list[dict[str, str]]:
    """The most recent generations, with enough to tell them apart.

    Three small files per row, which is why the list is capped: this is a
    local page showing one person their own folder, not something that has to
    stay fast at ten thousand rows.
    """
    rows: list[dict[str, str]] = []
    for generation_id in store.list_generations(direction_name)[:limit]:
        try:
            record = load_generation(store, direction_name, generation_id)
        except (OSError, ValueError):
            # Same rule as `direction_rows`: one unreadable sidecar skips its
            # own row rather than emptying the list.
            continue
        rows.append(
            {
                "id": generation_id,
                "subject": record.request.subject,
                "kind": record.request.kind.value,
                "status": record.review.status.value,
            }
        )
    return rows


def edited_phrase(moment: datetime, now: datetime) -> str:
    """How long ago a card was last touched, in whole days: today, yesterday, N days ago.

    Days are calendar days in UTC, the zone every timestamp here is kept in, so
    the phrase rolls over at UTC midnight and not 24 hours after the edit.
    """
    days = (now.date() - moment.astimezone(UTC).date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


@router.get("/")
def home(request: Request, bench: Bench) -> Response:
    """The game: its name, and a card for every art direction in it.

    A folder with no `pixelmagic.json` is the first thing a new user meets, so
    it gets a page of its own (still a 404) that says how to make one. A
    project file that will not parse is not that case and keeps the error page.
    """
    try:
        project = bench.store.load_project()
    except FileNotFoundError:
        return page(
            request, "no_project.html", status_code=404, root=shlex.quote(str(bench.store.root))
        )
    directions, broken = direction_rows(bench.store)
    summaries = order_for_home(
        [direction_summary(bench.store, direction.name) for direction in directions]
    )
    now = datetime.now(UTC)
    cards = [
        {
            "summary": summary,
            "picture": (
                f"/directions/{summary.name}/generations/{summary.latest_scene}/image.png"
                if summary.latest_scene
                else None
            ),
            "edited": edited_phrase(summary.last_edited, now),
        }
        for summary in summaries
    ]
    if summaries:
        lead = f"/directions/{summaries[0].name}"
        new_scene_href, import_href = f"{lead}/generate", f"{lead}/generate#import"
    else:
        new_scene_href = "/new-direction?then=generate"
        import_href = "/new-direction?then=import"
    return page(
        request,
        "home.html",
        project=project,
        cards=cards,
        broken=broken,
        new_scene_href=new_scene_href,
        import_href=import_href,
        service=bench.service_name,
        planned=sorted(registry.PLANNED_SERVICES.items()),
    )


@router.get("/new-direction")
def new_direction_form(
    request: Request, bench: Bench, then: str | None = None
) -> Response:
    """The form that makes an art direction.

    `then` (generate or import) is only carried through to the form's post, so
    the first direction can go straight on to the step the person came for.
    """
    return page(
        request,
        "direction_new.html",
        project=bench.store.load_project(),
        crumb="New art direction",
        then=then if then in THEN_PAGES else None,
        service=bench.service_name,
        cell_size_bounds=(MIN_CELL_SIZE, MAX_CELL_SIZE),
        default_cell_size=DEFAULT_CELL_SIZE,
    )


@router.post("/directions")
def new_direction(
    bench: Bench,
    style: Annotated[str, Form()],
    name: Annotated[str, Form()] = "",
    donts: Annotated[str, Form()] = "",
    # The bound is `Direction`'s own; the same two constants here mean an
    # out-of-range value is refused by FastAPI as a `RequestValidationError`
    # and shown as "That is not a value this page takes", exactly as the
    # decompose form's `size` mirrors `MAX_IMAGE_SIZE`.
    cell_size: Annotated[int, Form(ge=MIN_CELL_SIZE, le=MAX_CELL_SIZE)] = DEFAULT_CELL_SIZE,
    then: Annotated[str, Form()] = "",
) -> Response:
    """Make an art direction. Without a name, the service invents one.

    The rule itself — read the project first, squeeze a typed name or ask the
    service for one, make it the default when there is none — is
    `directions.start_direction`, shared with the command line and the MCP
    server. A name with nothing usable in it comes back as
    `UnusableName`, which is a `ValueError`, so `web.app`'s handler turns it
    into the same 400 page this route always gave; a missing project is a
    `FileNotFoundError` and the 404 page, as before.

    `then` is a fixed choice (`THEN_PAGES`), never a URL: anything else goes to
    the direction page, so the redirect cannot be pointed elsewhere.
    """
    # A typed name needs no service; a blank one asks the service to invent it.
    direction = start_direction(
        bench.store,
        style=style,
        service=bench.service() if not name.strip() else None,
        service_name=bench.service_name,
        name=name,
        donts=donts.splitlines(),
        cell_size=cell_size,
    )
    return see_other(f"/directions/{direction.name}{THEN_PAGES.get(then, '')}")


@router.get("/directions/{direction}")
def show_direction(request: Request, direction: str, bench: Bench) -> Response:
    """One art direction: its style, what it has drawn, and what to do next."""
    found = require_direction(bench, direction)
    return page(
        request,
        "direction.html",
        direction=found,
        **latest_scene_context(bench, found.name, None),
        generations=generation_rows(bench.store, found.name),
        assets=bench.store.list_assets(found.name),
        service=bench.service_name,
    )


@router.post("/directions/{direction}/rename")
def rename(direction: str, bench: Bench, name: Annotated[str, Form()]) -> Response:
    """Give an art direction a different name and move its folder.

    The typed name is squeezed through `naming.typed_slug` here, at the boundary
    where free text arrives, so the store is handed a name it can find again.
    An empty box — or one with nothing `typed_slug` can use, such as text in another
    script — is refused rather than silently becoming the `kebab_case`
    fallback, which would name the direction "item".
    """
    chosen = typed_slug(name, "direction name")
    if chosen is None:
        raise ValueError(f"{name!r} has nothing in it that can be a direction name")
    renamed = bench.store.rename_direction(check_name(direction, "direction"), chosen)
    return see_other(f"/directions/{renamed.name}")
