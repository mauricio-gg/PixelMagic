"""Starting an art direction: the one rule all three faces share.

A person at the terminal, a person on the web page and an agent over MCP all
start a direction the same way — check there is a project here, choose a
name (the one they typed, squeezed into a folder name, or one the image
service invents from the style paragraph), create the folder tree, and make
it the project's default when the project has none. Until this module
existed the rule was written twice, in `cli.direction_new` and in
`web.projects.new_direction`, and the MCP tool `new_direction` would have
been a third copy. CLAUDE.md's rule is that something every face needs lives
in the core, so it lives here, and each face only translates what this
raises: a usage error, a page with one sentence, a `ToolError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pixelmagic.model import DEFAULT_CELL_SIZE, Direction
from pixelmagic.naming import UnusableName, suggest_direction_name, typed_slug
from pixelmagic.providers.base import ImageService
from pixelmagic.services import NoServiceChosen
from pixelmagic.stages.refine import HOUSE_BASELINE
from pixelmagic.store import Store


def start_direction(
    store: Store,
    *,
    style: str,
    service: ImageService | None,
    service_name: str | None,
    name: str | None = None,
    donts: Sequence[str] = (),
    cell_size: int = DEFAULT_CELL_SIZE,
    now: datetime | None = None,
) -> Direction:
    """Create an art direction, and make it the default when there is none.

    The project is read before anything is written: a direction tree inside
    a folder that is not a project is litter, and `store.load_project`
    raising `FileNotFoundError` here is what stops that folder tree from
    ever being created. The CLI reads the project itself first, for its own
    sentence ("no project here — run project new first"); the web page and
    an MCP agent instead let this call's `FileNotFoundError` pass through and
    handle it at their own boundary: the web page turns it into its 404
    page, and the MCP server passes the error's own text through as a
    `ToolError`.

    A `name` a *person* typed goes through `naming.slug`, never
    `naming.kebab_case`: a name that squeezes down to nothing (an emoji, a
    script with no Latin letters in it) is refused with `UnusableName` rather
    than silently becoming "item", because the folder name is the
    direction's identity. A missing or blank `name` means the service invents
    one from the style paragraph; that is a paid call, so with no `service`
    it raises `NoServiceChosen`. A typed name needs no service.

    `cell_size` is bounded by `Direction` itself (`MIN_CELL_SIZE` to
    `MAX_CELL_SIZE`, spec locked decision 3), so a value outside it raises
    pydantic's `ValidationError` out of the constructor, before the folder
    tree exists. All three faces — the command line, the web form and the MCP
    server — refuse such a value at their own boundary with the same two
    constants, so that error is a backstop and not the sentence a person
    reads.

    `AlreadyExists` comes from `Store.create_direction` unchanged: rewriting
    a direction would drop the style, the don'ts and the references it
    holds.

    `refiner_defaults` is seeded with a deep copy of `HOUSE_BASELINE` (M4
    spec, "Defaults and the record") so a fresh `direction.json` is
    self-documenting about the M4 defaults from the day it is created,
    rather than relying on `resolve_settings` layering the baseline in at
    refine time the way an existing direction does. `model_copy(deep=True)`
    matters: without it every direction created in one process would share
    the module-level constant's own `quick`/`advanced` dict instances.
    """
    project = store.load_project()
    if name is not None and name.strip():
        chosen = typed_slug(name, "direction name")
        if chosen is None:
            raise UnusableName(
                f"{name!r} has nothing usable in it — try letters or numbers "
                f"('Café Noir' becomes 'cafe-noir'), or leave the name out "
                f"and let the service invent one"
            )
    else:
        if service is None:
            raise NoServiceChosen
        chosen = suggest_direction_name(service, style)
    direction = store.create_direction(
        Direction(
            name=chosen,
            created=now or datetime.now(UTC),
            style=style,
            donts=[line.strip() for line in donts if line.strip()],
            service=service_name,
            cell_size=cell_size,
            refiner_defaults=HOUSE_BASELINE.model_copy(deep=True),
        )
    )
    if project.default_direction is None:
        project.default_direction = direction.name
        store.save_project(project)
    return direction
