"""The MCP server: the same pipeline, driven by an agent instead of a person.

One of the three thin faces (locked decision 7). The tools mirror the command
line one for one, and none of them holds logic a stage does not already own.

Two things are this face's own job. The first is that the calling agent has to
be able to *see* what it asked for, so every tool that produces a picture hands
it back as image content rather than a file path — an agent with a path has no
way to look at it. The second is that a failure has to arrive as words: the SDK
withholds the message of an unexpected exception (the client is told only
"Error executing tool X" and the server logs a crash), so every failure the
core raises is re-raised as a `ToolError`, whose message does reach the caller.

The server is built by a factory rather than declared at module scope, because
the root folder, the image service and the refiner are chosen when it starts —
and because a test needs its own server over its own temporary folder.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field, ValidationError

from pixelmagic import registry
from pixelmagic.approval import ApprovalDecision, PendingApproval
from pixelmagic.directions import start_direction
from pixelmagic.model import (
    DEFAULT_CELL_SIZE,
    MAX_CELL_SIZE,
    MIN_CELL_SIZE,
    BackgroundPolicy,
    Decomposition,
    Direction,
    GenerateKind,
    GenerationRecord,
    Point,
    PromptKind,
    RefinerSettings,
    Reviewer,
    ReviewStatus,
)
from pixelmagic.naming import RECIPE_PLACEHOLDER, check_name, parse_surfaces
from pixelmagic.refiner.base import MAX_REFINE_SCALE, Refiner
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.base import Segmenter
from pixelmagic.services import chosen_service, require_service
from pixelmagic.stages.decompose import (
    NoPicks,
    decompose,
    delete_cutout,
    file_promotion,
    load_decomposition,
    missing_run_sentence,
    pitch_note,
    prepare_promotion,
    promote_cutouts,
    rename_cutout,
)
from pixelmagic.stages.decompose import keep_scene as keep_scene_stage
from pixelmagic.stages.decompose import rename_scene as rename_scene_stage
from pixelmagic.stages.decompose import set_anchor as set_anchor_stage
from pixelmagic.stages.decompose import set_scene_pitch as set_scene_pitch_stage
from pixelmagic.stages.export import DEFAULT_SCALES, export
from pixelmagic.stages.generate import (
    MAX_IMAGE_SIZE,
    generate,
    import_picture,
    load_generation,
    promote,
    record_decision,
    reference_name,
    retry_generation,
)
from pixelmagic.stages.generate import (
    keep_reference as keep_reference_stage,
)
from pixelmagic.stages.pick import (
    fix_pick as fix_pick_stage,
)
from pixelmagic.stages.pick import (
    how_picked,
    load_picks,
    pick_box,
    pick_points,
    pick_suggestions,
    render_overlay,
)
from pixelmagic.stages.pick import unpick as unpick_stage
from pixelmagic.stages.pixelcut import cut_pieces
from pixelmagic.stages.pixelcut import start_pixel_cut as start_pixel_cut_stage
from pixelmagic.stages.rebuild import RebuiltPieceRefused
from pixelmagic.stages.redraw import live_pieces, redraw_picks
from pixelmagic.stages.refine import (
    OLD_ITEM_ROUTE,
    PIXELMAGIC_REFUSAL_CODES,
    REFINED_FILE,
    SOURCE_FILE,
    WRONG_SIZE,
    load_report,
    refine_asset,
    refine_direction,
    refused_suffix,
    require_asset,
    set_refiner_defaults,
    stored_grid_needs_attention,
    tally_reports,
    wrong_size_sentence,
)
from pixelmagic.stages.suggest import (
    contact_sheet_path,
    load_suggestions,
    page_count,
    suggestions_on,
)
from pixelmagic.stages.suggest import (
    suggest_objects as suggest_objects_stage,
)
from pixelmagic.store import Store, check_run_id

#: How many pictures one tool answers with at most. A decomposition can
#: produce twenty pieces, and twenty images in one reply is a lot of an
#: agent's context for a look it can take one at a time with `show_asset`.
PREVIEW_LIMIT = 8

#: What `refine` and `export_bundle` say when the stand-in refiner did the
#: work. The plan's own deviation 10 lists where this has to be said — the web
#: page's yellow box, the manifest's `refiner_tool`, the README — and named
#: the agent as the face most likely to report the pixel art finished without
#: it. Compared against `StubRefiner.name`, never the literal `"stub"`, so a
#: rename of the class can never make this warning silently stop firing.
STAND_IN_WARNING = (
    "the scene is concept art copied through unchanged, not pixel art (rebuilt "
    "props and recipe tiles are real pieces the refiner never touches); start the "
    "server with --refiner cli for the real thing"
)

#: What the agent is told about the order things happen in.
INSTRUCTIONS = """
PixelMagic turns a text prompt into game-ready pixel art. The order is:
draw a picture, approve it, pick the objects on it, redraw them as pieces,
fix the pieces, keep them, refine them into pixel art, export a bundle.

Call project_overview first: it names the art directions and the default one.
Every tool takes a `direction` and falls back to the project's default.

draw makes one attempt and leaves it waiting for your verdict. Look at the
picture it returns, then call approve or reject. Rejecting draws the next
attempt with your words folded into the request. bring_in_picture takes a
picture file you already have instead, already approved and ready for
suggest_objects.

Objects are cut out one by one. suggest_objects finds them on an approved
picture and gives a cut id; show_suggestions shows numbered contact sheets
page by page; pick_objects takes the numbers you want. An object the sheets
missed is picked with pick_objects by points or a box, and a wrong outline
is redone with fix_pick (unpick takes a pick back, show_picks shows the current picks outlined
on the picture). redraw_objects then redraws the picks as separate pieces.
A picture that is already pixel art can skip the redraw: start_pixel_cut opens
a cut on it, pick_objects (by box or points) and fix_pick choose the objects,
cut_pixel_pieces cuts each one from the picture's own pixels, and keep_pieces
keeps them. No image service and no PixelRefiner are involved.
Surfaces that repeat (a floor, a rug) still go through cut_into_pieces with
`surfaces`, which makes tiles only: "plaster wall:plaster; box 180,240
212,285; shape 140,400 200,230 330,110 520,70" is a recipe surface (see
cut_into_pieces for the form).

Between redraw_objects and keep_pieces, rename_piece and drop_piece fix the
pieces. A redrawn prop is rebuilt on its own pixel grid, so it is changed by
redrawing it by name with redraw_objects. set_scene_pitch says how many source
pixels make one art pixel for a kept scene, at any time. Once a prop is kept,
set_anchor says where it stands. new_direction takes the cell size tiles and
footprints are counted in.
""".strip()


@contextmanager
def readable_errors() -> Iterator[None]:
    """Turn anything the core raises into a message the caller actually reads.

    The SDK only passes a `ToolError`'s text through. Every other exception is
    treated as a crash: the client is told "Error executing tool X" and nothing
    else. But each failure the core raises is a fact the agent needs in order
    to do something different — there is no such asset, the refiner is not
    installed, that service is not built yet, nobody approved a picture — so
    each one is re-raised as words.

    `ValidationError` is caught ahead of the wider `ValueError` family it
    belongs to, and gets its own opening words: a sidecar that will not read
    back is a file PixelMagic itself wrote being broken, not a bad argument
    the agent gave — the same distinction `web/app.py`'s
    handler order draws for a browser. Without that ordering an agent that hit
    a corrupt `decomposition.json` would be told "that was a bad request" and
    retry the same call with different arguments forever, which is exactly the
    failure mode this whole function exists to prevent.

    `ValueError` otherwise covers `registry.NotBuiltYet` and the rest of that
    family; `OSError` covers a missing file and a name already taken; and
    `RuntimeError` covers the family a service fails in —
    `providers.base.ServiceError` is a bare `RuntimeError` subclass, and
    "nobody is logged in" or "the content filter refused that prompt" is
    exactly the kind of fact an agent can act on. `RefinerError` and
    `ApprovalExhausted` are `RuntimeError` subclasses too, so naming
    `RuntimeError` covers all three and this module still imports nothing from
    `providers/`.

    The cost of that width is that a `RuntimeError` raised by a library rather
    than by us also comes back as a sentence instead of a crash. That is in the
    deviations register; on balance a wrong-looking sentence is better than the
    bare "Error executing tool draw" an agent gets today for the most common
    real failure there is.
    """
    try:
        yield
    except ToolError:
        raise
    except ValidationError as error:
        raise ToolError(
            f"a file PixelMagic wrote will not read back: {error}"
        ) from error
    except (ValueError, OSError, RuntimeError) as error:
        raise ToolError(str(error) or type(error).__name__) from error


class Click(BaseModel):
    """One click on the original picture, as an agent writes it."""

    x: int
    y: int
    #: True: the spot is on the object. False: the spot is NOT on the object.
    this: bool


def create_server(
    root: Path,
    *,
    service: str | None = None,
    refiner: Refiner | None = None,
    segmenter: Segmenter | None = None,
) -> MCPServer:
    """Build a server that drives one project folder.

    `service`, `refiner` and `segmenter` are the same three choices the
    command line's `--service`, `--refiner` and `--segmenter` make. The refiner
    and segmenter are fixed when the server starts, because an MCP client
    launches it once and talks to it for the whole session; `service` is the
    flag only, or None to follow the project's saved choice, which is read on
    every tool call so a choice made on the web page holds without a restart.
    The defaults are `registry.DEFAULT_REFINER` and
    `registry.DEFAULT_SEGMENTER`, the same constants the web page and the
    command line read. Loading the real segmenter starts nothing, so a machine
    without the helper only finds out when a tool that uses it is called.
    """
    store = Store(root)

    def existing_run(direction_name: str, cut_id: str, *, pixel_cut: bool = False) -> str:
        """A checked cut-up run id, or a `ToolError` with the core's sentence if the run is missing.

        Looks for the run's own `decomposition.json` before the tool does
        anything else, so only that file being absent gets the "picture id"
        wording; any other missing file keeps its own message. `pixel_cut` names
        `start_pixel_cut` as the step that makes a run, for `cut_pixel_pieces`.
        """
        cut_id = check_run_id(cut_id, "cut-up run id")
        if not (store.decomposition_dir(direction_name, cut_id) / "decomposition.json").exists():
            if pixel_cut:
                raise ToolError(missing_run_sentence(
                    store, direction_name, cut_id, "start_pixel_cut(generation_id={id!r})",
                    making="Starting a pixel cut on the picture",
                ))
            raise ToolError(
                missing_run_sentence(
                    store, direction_name, cut_id, "suggest_objects(generation_id={id!r})"
                )
            )
        return cut_id

    def drawing_service() -> str:
        """The service name that draws now: the flag, else the project's choice.

        Raises `NoServiceChosen` (a `ValueError`, so a `ToolError` with the
        core's own sentence) when there is none.
        """
        return require_service(store, service)

    tool_refiner = refiner or registry.load_refiner(registry.DEFAULT_REFINER)
    tool_segmenter = segmenter or registry.load_segmenter(registry.DEFAULT_SEGMENTER)
    mcp = MCPServer(
        "PixelMagic",
        # Read rather than repeated: `pyproject.toml` already holds the
        # version, and a second copy here is one that goes stale on the first
        # release nobody thinks to update it for.
        version=metadata.version("pixelmagic"),
        description="Turn a text prompt into game-ready pixel art.",
        instructions=INSTRUCTIONS,
    )

    def pick_direction(name: str) -> Direction:
        """Load the named direction, or the project's default one."""
        chosen = name.strip() or (store.load_project().default_direction or "")
        if not chosen:
            raise ToolError(
                "this project has no default art direction — name one, or make "
                "one first"
            )
        return store.load_direction(check_name(chosen, "direction"))

    def preview(record: GenerationRecord) -> list[str | Image]:
        """One attempt, described and shown.

        The agent is the reviewer, so it has to be able to look at the picture
        rather than be told where the file is: a path is not something a model
        can see.
        """
        if record.request.imported_from:
            words = (
                f"{record.request.id}: {record.request.subject} "
                f"(imported from {record.request.imported_from}). "
                f"It is already approved: pass this id to suggest_objects, or to "
                f"start_pixel_cut if the picture is already pixel art."
            )
        else:
            words = (
                f"{record.request.id}: {record.request.subject} "
                f"({record.request.kind.value}, {record.request.background.value} "
                f"background, drawn by {record.request.service}). "
                f"Call approve or reject with this id."
            )
        return [
            words,
            Image(path=record.image_path),
        ]

    @mcp.tool()
    def project_overview() -> str:
        """Name the project, its art directions, and which image services work.

        Call this first. Every other tool takes an art direction and falls back
        to the default one named here.
        """
        with readable_errors():
            project = store.load_project()
            lines = [
                f"project: {project.name}",
                f"default art direction: {project.default_direction or '(none)'}",
                f"drawing with: {chosen_service(store, service) or '(no service chosen)'}",
                f"image services that work: {', '.join(registry.known_services())}",
                f"refiner: {tool_refiner.name}",
            ]
            if registry.PLANNED_SERVICES:
                # Only when there is something to say. Once M2.3 to M2.5 have
                # merged this mapping is empty, and a heading with nothing
                # after it reads as a bug rather than as good news.
                lines.insert(
                    -1,
                    "image services not built yet: "
                    + ", ".join(
                        f"{name} (ticket {ticket})"
                        for name, ticket in sorted(registry.PLANNED_SERVICES.items())
                    ),
                )
            for name in store.list_directions():
                try:
                    found = store.load_direction(name)
                except (OSError, ValueError):
                    lines.append(f"- {name}: its direction.json will not read back")
                    continue
                lines.append(
                    f"- {name}: {found.style} "
                    f"({len(store.list_assets(name))} asset(s), "
                    f"{len(store.list_generations(name))} picture(s), "
                    f"{len(store.list_decompositions(name))} cut-up run(s))"
                )
            return "\n".join(lines)

    @mcp.tool()
    def new_direction(
        style: str,
        name: str = "",
        donts: list[str] | None = None,
        cell_size: Annotated[int, Field(ge=MIN_CELL_SIZE, le=MAX_CELL_SIZE)] = DEFAULT_CELL_SIZE,
    ) -> str:
        """Start an art direction: a look, described once, that every picture follows.

        `style` is a paragraph saying what things should look like; `donts`
        are things never to draw. Leave `name` empty and the image service
        invents one from the style. `cell_size` is how many art pixels one
        tile is on a side (16 unless you say otherwise); tiles are refined to
        exactly that and a prop's footprint is counted in it. The first
        direction in a project becomes its default.
        """
        # The bound is `Direction`'s own; the same two constants on the
        # argument mean the SDK refuses an out-of-range value before this
        # runs, as it refuses `draw`'s size, and no folder is made.
        with readable_errors():
            # A typed name needs no service; asking for one is a paid call.
            drawing = drawing_service() if not name.strip() else chosen_service(store, service)
            made = start_direction(
                store,
                style=style,
                service=registry.load_service(drawing) if drawing else None,
                service_name=drawing,
                name=name,
                donts=list(donts or []),
                cell_size=cell_size,
            )
            default = store.load_project().default_direction
            note = f"made the art direction {made.name} (cell size {made.cell_size})"
            if not name.strip():
                note += f"; {drawing} was asked to invent the name"
            if default == made.name:
                note += "; it is the project's default"
            return note + ". Call draw with a subject to start drawing in it."

    @mcp.tool()
    def list_assets(direction: str = "") -> str:
        """List an art direction's assets and what the refiner did to each."""
        with readable_errors():
            found = pick_direction(direction)
            names = store.list_assets(found.name)
            if not names:
                return f"{found.name} has no assets yet"
            lines = [f"{found.name}: {len(names)} asset(s)"]
            for name in names:
                try:
                    report = load_report(store, found.name, name)
                except (OSError, ValueError):
                    lines.append(f"- {name}: not refined yet")
                    continue
                note = f"- {name}: {report.route or 'not run'} by {report.refiner}"
                if report.error_code in PIXELMAGIC_REFUSAL_CODES:
                    note += (
                        f" — PixelMagic did not send it to the refiner "
                        f"({report.error_code}): {report.error_message}"
                    )
                elif report.error_code == WRONG_SIZE:
                    note += f" — {wrong_size_sentence(report)}"
                elif report.error_code:
                    note += (
                        f" — the refiner refused it ({report.error_code}): "
                        f"{report.error_message}"
                    )
                elif stored_grid_needs_attention(store, found.name, report):
                    note += (
                        " — the pixel grid is uncertain; check the scene's pitch "
                        "and set it with set_scene_pitch"
                    )
                lines.append(note)
            return "\n".join(lines)

    @mcp.tool()
    def show_asset(asset: str, direction: str = "") -> list[str | Image]:
        """Show one asset before and after the refiner touched it."""
        with readable_errors():
            found = pick_direction(direction)
            # `require_asset` refuses a typo and an asset whose picture was
            # deleted by hand, as ValueError, which `readable_errors` words.
            folder = store.asset_dir(found.name, require_asset(store, found.name, asset))
            source = folder / SOURCE_FILE
            reply: list[str | Image] = [f"{asset}: what went into the refiner"]
            reply.append(Image(path=source))
            refined = folder / REFINED_FILE
            if refined.is_file():
                reply.append(f"{asset}: what came out")
                reply.append(Image(path=refined))
            else:
                reply.append(f"{asset} has not been refined yet")
            return reply

    @mcp.tool()
    def draw(
        subject: str,
        direction: str = "",
        kind: GenerateKind = GenerateKind.SCENE,
        background: BackgroundPolicy = BackgroundPolicy.TRANSPARENT,
        size: Annotated[int, Field(ge=1, le=MAX_IMAGE_SIZE)] = 1024,
    ) -> list[str | Image]:
        """Draw one picture and show it to you, without deciding anything.

        One attempt, not a loop: the verdict is yours, so the picture comes
        back waiting for it. Look at what is returned and then call `approve`
        or `reject`. Rejecting draws the next attempt with your words in it.
        """
        with readable_errors():
            found = pick_direction(direction)
            record = generate(
                store,
                found,
                subject,
                kind=PromptKind(kind.value),
                service=registry.load_service(drawing_service()),
                approval=PendingApproval(reviewer=Reviewer.AGENT),
                background=background,
                size=(size, size),
                references=list(found.references),
            )
            return preview(record)

    @mcp.tool()
    def approve(
        generation_id: str,
        direction: str = "",
        keep_as_reference: bool = True,
        keep_as_asset: bool = False,
    ) -> str:
        """Say yes to a picture you drew, and choose what to keep it as.

        A reference steers every later picture in this art direction towards
        this one, which is how a direction settles. An asset means the whole
        picture goes into the export bundle as it stands, as well as any
        pieces cut out of it.
        """
        with readable_errors():
            found = pick_direction(direction)
            record = load_generation(
                store, found.name, check_run_id(generation_id, "generation id")
            )
            # The refiner call happens before the decision is written: it is
            # the one thing here that can fail, and running it first means a
            # failure leaves the generation pending, not approved with no
            # asset to show for it, so the agent can just call approve again
            # once the refiner works.
            prepared = None
            if keep_as_asset:
                prepared = prepare_promotion(
                    tool_refiner, found.name, record.request.id, record.image_path
                )
            record = record_decision(
                store,
                found.name,
                record,
                ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.AGENT),
            )
            kept = []
            asset_note = None
            if keep_as_reference:
                promote(store, found, record)
                kept.append("a style reference")
            if keep_as_asset:
                filed = file_promotion(store, found, record, prepared)
                kept.append(f"the asset {filed}")
                asset_note = pitch_note(store, found.name, filed)
            note = f"approved {record.request.id}"
            if kept:
                note += " and kept it as " + " and ".join(kept)
            note += ". Call suggest_objects with this id to find its objects."
            return f"{note} {asset_note}" if asset_note else note

    @mcp.tool()
    def bring_in_picture(path: str, direction: str = "") -> list[str | Image]:
        """Bring in a picture you already have, instead of drawing one.

        Takes the path of a PNG, JPEG or WebP file on this machine. It goes in
        as an approved scene, so pass the id it returns straight to
        cut_into_pieces.
        """
        with readable_errors():
            found = pick_direction(direction)
            return preview(
                import_picture(store, found, Path(path), reviewer=Reviewer.AGENT)
            )

    @mcp.tool()
    def keep_scene(generation_id: str, direction: str = "", name: str = "") -> str:
        """Keep an already-approved picture, imported or drawn, as the scene.

        The whole picture then refines and exports as the scene. `name` is the
        scene's name; leave it empty and one is made from the picture's subject.
        Refused when the picture is not approved, is already kept as the scene,
        or the name is not a safe name or is already taken.
        """
        with readable_errors():
            found = pick_direction(direction)
            record = load_generation(
                store, found.name, check_run_id(generation_id, "generation id")
            )
            kept = keep_scene_stage(
                store, found, record, refiner=tool_refiner, name=name or None
            )
            note = f"kept {record.request.id} as the scene, asset {kept}"
            hint = pitch_note(store, found.name, kept)
            return f"{note}. {hint}" if hint else note

    @mcp.tool()
    def rename_scene(scene: str, new_name: str, direction: str = "") -> str:
        """Rename a kept scene.

        An export bundle already made keeps the old name; the next export uses
        the new one. Refused when `scene` is not a kept scene or the new name
        is not a safe name or is already taken.
        """
        with readable_errors():
            found = pick_direction(direction)
            renamed = rename_scene_stage(store, found, scene, new_name)
            return f"renamed {scene} to {renamed}"

    @mcp.tool()
    def keep_reference(generation_id: str, direction: str = "") -> str:
        """Keep an already-approved picture, imported or drawn, as a style reference.

        Later pictures drawn in this direction are then conditioned on it.
        Refused when the picture is not approved; keeping one that already is a
        reference changes nothing.
        """
        with readable_errors():
            found = pick_direction(direction)
            record = load_generation(
                store, found.name, check_run_id(generation_id, "generation id")
            )
            added = keep_reference_stage(store, found, record)
            name = reference_name(record)
            if added:
                return f"kept {record.request.id} as a style reference, references/{name}"
            return f"{record.request.id} is already a style reference, references/{name}"

    @mcp.tool()
    def reject(
        generation_id: str, feedback: str, direction: str = ""
    ) -> list[str | Image]:
        """Say no to a picture, say what is wrong, and get the next attempt.

        The words go into the next request, and the attempt you turned down
        stays on disk as the parent of the new one, so the folder shows how a
        picture was arrived at.
        """
        with readable_errors():
            found = pick_direction(direction)
            record = load_generation(
                store, found.name, check_run_id(generation_id, "generation id")
            )
            # Built before the rejection is saved: with no service chosen this
            # refuses, and a saved rejection with no follow-up would strand the picture.
            drawing = registry.load_service(drawing_service())
            record = record_decision(
                store,
                found.name,
                record,
                ApprovalDecision(
                    status=ReviewStatus.REJECTED,
                    reviewer=Reviewer.AGENT,
                    feedback=feedback.strip() or None,
                ),
            )
            return preview(
                retry_generation(
                    store,
                    found,
                    record,
                    service=drawing,
                    approval=PendingApproval(reviewer=Reviewer.AGENT),
                )
            )

    def pieces_reply(
        found: Direction, result: Decomposition, lead: str
    ) -> list[str | Image]:
        """Describe a decomposition's pieces and show the first few.

        `PREVIEW_LIMIT` caps the pictures to keep a large decomposition from
        spending an agent's whole context on one reply.
        """
        names = [cutout.name for cutout in result.cutouts]
        reply: list[str | Image] = [
            f"{lead} {len(names)} piece(s) as {result.id}: "
            + ", ".join(names)
            + ". Fix them with rename_piece and drop_piece, then call keep_pieces."
        ]
        for cutout in result.cutouts[:PREVIEW_LIMIT]:
            reply.append(f"piece: {cutout.name} ({cutout.kind.value})")
            reply.append(
                Image(path=store.cutout_dir(found.name, result.id, cutout.name) / SOURCE_FILE)
            )
        if len(names) > PREVIEW_LIMIT:
            reply.append(f"({len(names) - PREVIEW_LIMIT} more not shown)")
        reply += [f"note: {warning}" for warning in result.warnings]
        return reply

    @mcp.tool()
    def cut_into_pieces(
        generation_id: str,
        direction: str = "",
        surfaces: list[str] | None = None,
    ) -> list[str | Image]:
        """Draw the tiles of an approved picture's repeating surfaces.

        `surfaces` names each repeating surface to tile. It is required:
        objects are not cut here but picked with suggest_objects,
        pick_objects and redraw_objects. A surface is typed as a recipe:

        A recipe surface is built from the picture itself and is ONE string of
        segments separated by semicolons: "name:kind; box x,y x,y; shape ...;
        shape ...". `kind` is plaster, stone or wood. `box` is exactly one: two
        opposite corners (x,y in the picture's pixels, either order) of a small
        patch of plain surface; its colours are read. `shape` is one or more:
        the corners (at least 3) of a rough outline drawn over where the
        surface is; the joined shapes are where it goes in the room, so use
        several shapes when furniture splits it. Example:
        "plaster wall:plaster; box 180,240 212,285; shape 140,400 200,230
        330,110 520,70; shape 1100,440 1100,330 1170,285". The core says
        plainly what is wrong with one it refuses.

        Plain text with no kind is refused. A surface is built from the
        picture's own colours, with no image service. The
        review that matters is the one over the pieces, and you do it with
        rename_piece and drop_piece before calling keep_pieces.
        Nothing is an asset until you do.

        At most a few pieces are shown; call show_asset, or look at them one at
        a time after keeping them, if you need to see the rest.
        """
        with readable_errors():
            found = pick_direction(direction)
            source = load_generation(
                store, found.name, check_run_id(generation_id, "generation id")
            )
            try:
                result = decompose(
                    store,
                    found,
                    source,
                    surfaces=parse_surfaces(surfaces or []),
                )
            except NoPicks as error:
                raise ToolError(
                    f"{error} — call suggest_objects with {generation_id}, then "
                    f"pick_objects to choose objects, then redraw_objects; or pass "
                    f"`surfaces` here to make tiles, typed like "
                    f"{RECIPE_PLACEHOLDER}"
                ) from error
            return pieces_reply(found, result, f"cut {generation_id} into")

    @mcp.tool()
    def start_pixel_cut(
        generation_id: str, size: float | None = None, direction: str = ""
    ) -> str:
        """Open a pixel cut on an approved picture that is already pixel art.

        Props are then cut from the picture's own pixels: no image service and
        no PixelRefiner are involved. `size` is how many picture pixels make
        one art pixel; leave it out to have it found, and give 1 when the
        picture is already one pixel per art pixel. Returns the cut id; choose
        objects with pick_objects (by `box` or points, not by number) and
        fix_pick, then call cut_pixel_pieces.
        """
        with readable_errors():
            found = pick_direction(direction)
            source = load_generation(
                store, found.name, check_run_id(generation_id, "generation id")
            )
            result = start_pixel_cut_stage(store, found, source, size=size)
            grid = result.pixel_grid
            lines = [
                f"pixel cut {result.id} opened; each art pixel is {grid.pixel_size:g} "
                f"picture pixel(s) wide ({grid.size_source}). Pick objects with "
                f"pick_objects (box or points), fix them with fix_pick, then call "
                f"cut_pixel_pieces."
            ]
            return "\n".join(lines + [f"note: {warning}" for warning in result.warnings])

    @mcp.tool()
    def cut_pixel_pieces(
        cut_id: str, only: list[int] | None = None, direction: str = ""
    ) -> list[str | Image]:
        """Cut each pick of a pixel cut into a prop from the picture's own pixels.

        With no `only`, every pick is cut; `only` lists pick numbers, and
        cutting a pick again replaces its piece. The pieces are not kept yet:
        call keep_pieces.
        """
        with readable_errors():
            found = pick_direction(direction)
            cut_id = existing_run(found.name, cut_id, pixel_cut=True)
            cut_pieces(store, found, cut_id, only=only)
            return pieces_reply(
                found, load_decomposition(store, found.name, cut_id), "cut"
            )

    @mcp.tool()
    def suggest_objects(generation_id: str, direction: str = "") -> list[str | Image]:
        """Find the objects on an approved picture, numbered on contact sheets.

        Takes up to a minute with the real segmenter. Returns the cut id every
        later picking tool needs and the first contact sheet; look at the
        rest with show_suggestions, then choose with pick_objects.
        """
        with readable_errors():
            found = pick_direction(direction)
            source = load_generation(
                store, found.name, check_run_id(generation_id, "generation id")
            )
            result = suggest_objects_stage(store, found, source, tool_segmenter)
            suggestions = load_suggestions(store, found.name, result.id)
            pages = page_count(suggestions)
            reply: list[str | Image] = [
                f"suggestions in {result.id}: {len(suggestions.suggestions)} shown "
                f"(of {suggestions.raw_count} found), {pages} pages. Pick next with "
                f"pick_objects, or look at another page with show_suggestions."
            ]
            if pages:
                reply.append(
                    Image(path=contact_sheet_path(store, found.name, result.id, 1))
                )
            return reply

    @mcp.tool()
    def show_suggestions(
        cut_id: str, page: int = 1, direction: str = ""
    ) -> str | list[str | Image]:
        """Show one numbered contact sheet, with each suggestion's box and area.

        The boxes are in the original picture's pixels (x0,y0 to x1,y1, right
        and bottom not included): use them to place points or a box for
        fix_pick.
        """
        with readable_errors():
            found = pick_direction(direction)
            cut_id = existing_run(found.name, cut_id)
            suggestions = load_suggestions(store, found.name, cut_id)
            if not suggestions.suggestions:
                return (
                    "nothing was found on this picture, so there are no suggestions to show. "
                    "Pick with pick_objects by `points` or a `box` instead."
                )
            shown = suggestions_on(suggestions, page)
            lines = [
                f"{one.number}: box {one.box[0]},{one.box[1]}\u2013{one.box[2]},{one.box[3]}, "
                f"area {one.area}"
                for one in shown
            ]
            sheet = contact_sheet_path(store, found.name, cut_id, page)
            return [
                f"page {page} of {page_count(suggestions)}:\n" + "\n".join(lines),
                Image(path=sheet),
            ]

    @mcp.tool()
    def pick_objects(
        cut_id: str,
        numbers: list[int] | None = None,
        name: str | None = None,
        points: list[Click] | None = None,
        box: Annotated[list[int], Field(min_length=4, max_length=4)] | None = None,
        direction: str = "",
    ) -> str:
        """Pick objects to redraw (or to cut, on a pixel cut): by number, points or a box.

        Give `numbers` (suggestions off the contact sheets), or `points` and/or
        `box` for an object no suggestion covers. Each point is a spot with
        `this` true (on the object) or false (not on it); `box` is x0,y0,x1,y1
        with the object inside (right and bottom not included) and may come
        with "not this" points. `name` names the pick and only works with one
        pick; an unnamed pick is named when it is redrawn. Everything is
        checked before anything is saved.
        """
        with readable_errors():
            found = pick_direction(direction)
            cut_id = existing_run(found.name, cut_id)
            if bool(numbers) == bool(points or box):
                raise ToolError("give either numbers, or points and/or a box, not both or neither")
            if numbers:
                made = pick_suggestions(store, found, cut_id, numbers, name=name)
                lines = [
                    f"suggestion {one.suggestion} is now pick {one.number}"
                    + (f" ({one.name})" if one.name else "")
                    for one in made
                ]
            else:
                clicks = [Point(x=c.x, y=c.y, label=1 if c.this else 0) for c in points or []]
                if box is not None:
                    one = pick_box(
                        store, found, cut_id, tuple(box),
                        points=clicks, name=name, segmenter=tool_segmenter,
                    )
                else:
                    one = pick_points(
                        store, found, cut_id, clicks, name=name, segmenter=tool_segmenter
                    )
                lines = [
                    f"picked {how_picked(one)} as pick {one.number}, "
                    f"outline box {one.picture_box}"
                ]
            return "\n".join(lines) + (
                "\nPick more, check with show_picks, or call cut_pixel_pieces."
                if load_decomposition(store, found.name, cut_id).pixel_grid is not None
                else "\nPick more, check with show_picks, or call redraw_objects."
            )

    @mcp.tool()
    def fix_pick(
        cut_id: str,
        pick: int,
        points: list[Click] | None = None,
        box: Annotated[list[int], Field(min_length=4, max_length=4)] | None = None,
        direction: str = "",
    ) -> str:
        """Redo one pick's outline from clicks or a box on the original picture.

        Each point is a spot with `this` true (on the object) or false (not on
        it); `box` is x0,y0,x1,y1 with the object inside (right and bottom not
        included). The pick keeps its number and name.
        """
        with readable_errors():
            found = pick_direction(direction)
            cut_id = existing_run(found.name, cut_id)
            fixed = fix_pick_stage(
                store,
                found,
                cut_id,
                pick,
                points=[Point(x=c.x, y=c.y, label=1 if c.this else 0) for c in points or []],
                box=tuple(box) if box is not None else None,
                segmenter=tool_segmenter,
            )
            lines = [
                f"fixed pick {fixed.number}: now {how_picked(fixed)}, "
                f"outline box {fixed.picture_box}"
            ]
            decomposition = load_decomposition(store, found.name, cut_id)
            if live_pieces(decomposition, fixed):
                if decomposition.pixel_grid is not None:
                    # A pixel cut cannot be redrawn; cutting again replaces the piece.
                    lines.append(
                        "a piece already cut from it is unchanged; replace it with "
                        f"cut_pixel_pieces(cut_id={cut_id!r}, only=[{fixed.number}])"
                    )
                else:
                    lines.append(
                        "a piece already redrawn from it is unchanged; replace it with "
                        f"redraw_objects(cut_id={cut_id!r}, "
                        f"only=[{str(fixed.name or fixed.number)!r}])"
                        " (a piece from an old merge or split cannot be redrawn by name)"
                    )
            return "\n".join(lines)

    @mcp.tool()
    def unpick(cut_id: str, pick: int, direction: str = "") -> str:
        """Take one pick back."""
        with readable_errors():
            found = pick_direction(direction)
            removed = unpick_stage(store, found, existing_run(found.name, cut_id), pick)
            return f"removed pick {removed.number}; its number is not used again"

    @mcp.tool()
    def show_picks(cut_id: str, direction: str = "") -> list[str | Image]:
        """List the picks and show them outlined and numbered on the picture."""
        with readable_errors():
            found = pick_direction(direction)
            cut_id = existing_run(found.name, cut_id)
            chosen = load_picks(store, found.name, cut_id)
            overlay = render_overlay(store, found, cut_id)
            named_at = (
                "named when cut"
                if load_decomposition(store, found.name, cut_id).pixel_grid is not None
                else "named at redraw"
            )
            lines = [
                f"{one.number}: {one.name or f'({named_at})'}, {how_picked(one)}, "
                f"outline box {one.picture_box}, area {one.area}"
                for one in chosen.picks
            ]
            return ["\n".join(lines) or "nothing picked yet", Image(path=overlay)]

    @mcp.tool()
    def redraw_objects(
        cut_id: str, only: list[str] | None = None, direction: str = ""
    ) -> list[str | Image]:
        """Redraw the picked objects, a few at a time, into separate pieces.

        Takes minutes with a real image service. With no `only`, the picks
        that have no piece yet are redrawn, so calling this again redraws just
        what failed. `only` names picks (numbers or names) to redraw again,
        replacing their pieces.
        """
        with readable_errors():
            found = pick_direction(direction)
            drawing = drawing_service()
            cut_id = existing_run(found.name, cut_id)
            result = redraw_picks(
                store,
                found,
                cut_id,
                service=registry.load_service(drawing),
                only=only,
            )
            reply = pieces_reply(
                found, result, f"redrew the picks with {drawing} into"
            )
            if not result.cutouts:
                # No pieces to list or to fix; the notes below say why.
                reply[0] = f"no piece was made in {result.id}"
            # A pick with no piece (a group that failed, or an object with no
            # pixel grid) stays picked; the core's warning says why, so this
            # line only says how to try it again.
            cut_id = existing_run(found.name, cut_id)
            missing = [
                str(one.name or one.number)
                for one in load_picks(store, found.name, cut_id).picks
                if not live_pieces(result, one)
            ]
            if missing:
                reply.append(
                    f"{len(missing)} pick(s) still have no piece; try again with "
                    f"redraw_objects(cut_id={cut_id!r}, only={missing!r})"
                )
            return reply

    @mcp.tool()
    def rename_piece(cut_id: str, piece: str, name: str, direction: str = "") -> str:
        """Give one piece a better name than the one the service guessed."""
        with readable_errors():
            found = pick_direction(direction)
            cut = load_decomposition(store, found.name, existing_run(found.name, cut_id))
            after = rename_cutout(
                store, found, cut, check_name(piece, "piece"), name
            )
            return "pieces now: " + ", ".join(one.name for one in after.cutouts)

    @mcp.tool()
    def drop_piece(cut_id: str, piece: str, direction: str = "") -> str:
        """Throw one piece away before it can become an asset."""
        with readable_errors():
            found = pick_direction(direction)
            cut = load_decomposition(store, found.name, existing_run(found.name, cut_id))
            after = delete_cutout(store, found, cut, check_name(piece, "piece"))
            return "pieces now: " + ", ".join(one.name for one in after.cutouts)

    @mcp.tool()
    def keep_pieces(cut_id: str, direction: str = "") -> str:
        """Turn the pieces into assets, ready to be refined."""
        with readable_errors():
            found = pick_direction(direction)
            cut = load_decomposition(store, found.name, existing_run(found.name, cut_id))
            kept = promote_cutouts(store, found, cut)
            lines = [f"kept {len(kept)} asset(s): " + ", ".join(kept)]
            lines += [f"note: {warning}" for warning in cut.warnings]
            return "\n".join(lines)

    @mcp.tool()
    def refine(
        direction: str = "",
        asset: str = "",
        preset: str = "",
        quick: dict[str, str] | None = None,
        advanced: dict[str, str] | None = None,
        scale: Annotated[int, Field(ge=1, le=MAX_REFINE_SCALE)] | None = None,
        candidate_id: str = "",
    ) -> str:
        """Refine the scene with PixelRefiner; copy rebuilt props and recipe tiles through.

        Only the kept scene is sent to PixelRefiner. A rebuilt prop and a recipe tile
        are already pixel art and are copied as they are. An old tile or prop whose
        road is gone is left as it is when already refined and refused when not.

        Settings with no `asset` named become this art direction's defaults, so
        everything drawn later matches. With an `asset` they are a one-off for
        that asset and the defaults are left alone. `candidate_id` is a grid
        candidate id from an earlier receipt; it only applies to a single
        asset, and it is refused for the kept scene, a rebuilt prop and a
        recipe tile (use set_scene_pitch for the scene).
        """
        with readable_errors():
            found = pick_direction(direction)
            given = RefinerSettings(
                preset=preset.strip() or None,
                quick=dict(quick or {}),
                advanced=dict(advanced or {}),
                scale=scale,
            )
            if asset:
                try:
                    report = refine_asset(
                        store,
                        found,
                        require_asset(store, found.name, asset),
                        refiner=tool_refiner,
                        override=given,
                        candidate_id=candidate_id.strip() or None,
                    )
                except RebuiltPieceRefused as error:
                    # An asset does not remember which cut-up run made it, so
                    # the tool names the call without inventing an id, and the
                    # asset's name may differ from its pick's.
                    raise ToolError(
                        f"{error}: call redraw_objects with the cut_id of the "
                        f"run it came from and that object's pick number "
                        f"(show_picks lists them)"
                    ) from error
                reports = [report]
            else:
                if candidate_id.strip():
                    raise ToolError(
                        "a grid candidate applies to one asset — name it with `asset`"
                    )
                if given.preset or given.quick or given.advanced or given.scale:
                    found = set_refiner_defaults(store, found, given)
                reports = refine_direction(
                    store, found, refiner=tool_refiner
                )
            # A rebuilt prop, recipe tile or old tile or prop (skipped) and a refused one were
            # not refined, so none counts in "refined N". A rebuilt prop whose picture will
            # not open has no rebuild route, only its refusal receipt.
            tally = tally_reports(reports)
            rebuilt, recipe, old_items = tally.rebuilt, tally.recipe, tally.old_items
            lines = [
                f"refined {tally.refined} asset(s) with {tool_refiner.name}"
                f"{refused_suffix(tally)}"
            ]
            if rebuilt:
                lines.append(
                    f"{rebuilt} rebuilt prop(s) were not sent to the refiner: they "
                    f"were rebuilt on their own pixel grid, so refine settings do "
                    f"not apply to them"
                )
            if recipe:
                lines.append(
                    f"{recipe} recipe tile(s) were not sent to the refiner: they "
                    f"are built by their recipe, so refine settings do not apply "
                    f"to them"
                )
            if tally.pixel_cut:
                lines.append(
                    f"{tally.pixel_cut} pixel-cut prop(s) were not sent to the refiner: "
                    f"they are the picture's own pixels, so refine settings do not "
                    f"apply to them"
                )
            if old_items:
                lines.append(
                    f"{old_items} old tile(s) or prop(s) were left as they are: they "
                    f"cannot be refined again"
                )
            if tool_refiner.name == StubRefiner.name:
                lines.append(STAND_IN_WARNING)
            for report in reports:
                if report.route == OLD_ITEM_ROUTE:
                    # Left as it is; its old receipt is nothing to act on.
                    continue
                if report.error_code in PIXELMAGIC_REFUSAL_CODES:
                    # PixelMagic stopped this one before the refiner saw it.
                    lines.append(
                        f"- {report.asset}: not refined, PixelMagic did not send "
                        f"it to the refiner ({report.error_code}) "
                        f"{report.error_message}"
                    )
                elif report.error_code == WRONG_SIZE:
                    lines.append(f"- {report.asset}: {wrong_size_sentence(report)}")
                elif report.error_code:
                    lines.append(
                        f"- {report.asset}: refused ({report.error_code}) "
                        f"{report.error_message}"
                    )
                elif stored_grid_needs_attention(store, found.name, report):
                    lines.append(
                        f"- {report.asset}: the pixel grid is uncertain; check "
                        "the scene's pitch and set it with set_scene_pitch"
                    )
            return "\n".join(lines)

    @mcp.tool()
    def set_scene_pitch(asset: str, value: float, direction: str = "") -> str:
        """Set the scene's pitch — how many source pixels make one art pixel.

        Only a kept scene has a pitch to set; a rebuilt prop's is its own and a
        tile has none.
        """
        with readable_errors():
            found = pick_direction(direction)
            # `set_scene_pitch_stage` already refuses an unknown name through
            # `require_asset` itself; looking it up again here would only be
            # a second read of the same directory for the same answer.
            record = set_scene_pitch_stage(store, found.name, asset, value)
            return (
                f"pitch of {record.name} is now {record.pitch} "
                f"({record.pitch_source.value}); target is "
                f"{record.target[0]} x {record.target[1]} art pixels"
            )

    @mcp.tool()
    def set_anchor(asset: str, x: int, y: int, direction: str = "") -> str:
        """Set the anchor — where a prop stands, in art pixels of its refined picture.

        (0, 0) is the top-left corner and the default is bottom centre. A
        game places the prop on a tile by this point. Usable once a prop is
        kept: the target size comes from redrawing it, not from refine. The
        point has to be inside that target; PixelMagic never guesses the anchor
        from what the prop is, so if the lamp hangs from the ceiling, say so here.
        """
        with readable_errors():
            found = pick_direction(direction)
            # `set_anchor_stage` already refuses an unknown name through
            # `require_asset` itself; see `set_scene_pitch`'s own comment.
            record = set_anchor_stage(store, found.name, asset, (x, y))
            return (
                f"{record.name} now stands on ({record.anchor[0]}, {record.anchor[1]}) "
                f"of its {record.target[0]} x {record.target[1]} art pixels"
            )

    @mcp.tool()
    def export_bundle(direction: str = "", scales: list[int] | None = None) -> str:
        """Write the folder a game reads: PNGs, manifest.json and Aseprite files.

        Native size is always written whether or not it is in `scales`; the
        core adds it. Leaving `scales` out gets the usual set; ask for `[1]`
        if you want the native size and nothing else, because an empty list
        reads here as not having asked.
        """
        with readable_errors():
            found = pick_direction(direction)
            bundle = export(store, found, scales=tuple(scales or DEFAULT_SCALES))
            lines = [
                f"exported {len(bundle.manifest.entries)} asset(s) to "
                f"{bundle.directory}",
                f"manifest: {bundle.directory / 'manifest.json'}",
            ]
            if tool_refiner.name == StubRefiner.name:
                lines.append(STAND_IN_WARNING)
            lines += [f"note: {warning}" for warning in bundle.warnings]
            if not bundle.manifest.entries:
                lines.append(
                    "nothing went in — every asset has either not been refined "
                    "yet or was refused; call refine first"
                )
            return "\n".join(lines)

    return mcp
