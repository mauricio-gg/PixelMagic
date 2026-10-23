"""The command line — one of the three thin faces over the core.

Every command here does the same three things: build a service, an approval
hook and a refiner from the flags, then call a stage function. There is no
logic in this file that a stage does not already own; that is what keeps the
web page and the MCP server (plan 2) from having to reimplement anything.

The one job that *is* this file's own is turning a stage's exception into a
sentence. A person at a terminal should never see a traceback for something
the tool already understands — a picture nobody approved, a refiner that is
not installed, a settings block the refiner refuses — so each of those becomes
one line and a non-zero exit code.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from typer.core import TyperGroup

from pixelmagic import registry
from pixelmagic.approval import ApprovalHook, AutoApprove, TerminalApproval
from pixelmagic.directions import start_direction
from pixelmagic.model import (
    DEFAULT_CELL_SIZE,
    MAX_CELL_SIZE,
    MIN_CELL_SIZE,
    BackgroundPolicy,
    Decomposition,
    Direction,
    GenerateKind,
    Group,
    Point,
    Project,
    PromptKind,
    RefineReport,
    RefinerSettings,
    Reviewer,
)
from pixelmagic.naming import (
    RECIPE_EXAMPLE,
    RECIPE_PLACEHOLDER,
    Surface,
    UnusableName,
    parse_surfaces,
)
from pixelmagic.providers.base import ImageService, ServiceError
from pixelmagic.refiner.base import Refiner, RefinerError
from pixelmagic.segmenter.base import Segmenter, SegmenterError
from pixelmagic.services import NoServiceChosen, chosen_service, require_service
from pixelmagic.stages.decompose import (
    NoPicks,
    NotApproved,
    OldPieceRefused,
    load_decomposition,
    missing_run_sentence,
    pitch_note,
    promote_cutouts,
    promote_generation,
)
from pixelmagic.stages.decompose import decompose as decompose_stage
from pixelmagic.stages.decompose import keep_scene as keep_scene_stage
from pixelmagic.stages.decompose import rename_scene as rename_scene_stage
from pixelmagic.stages.decompose import set_anchor as set_anchor_stage
from pixelmagic.stages.decompose import set_scene_pitch as set_scene_pitch_stage
from pixelmagic.stages.export import DEFAULT_SCALES, ExportBundle
from pixelmagic.stages.export import export as export_stage
from pixelmagic.stages.generate import (
    MAX_IMAGE_SIZE,
    ApprovalExhausted,
    generate_until_approved,
    import_picture,
    load_generation,
    promote,
    reference_name,
)
from pixelmagic.stages.generate import (
    keep_reference as keep_reference_stage,
)
from pixelmagic.stages.pick import (
    fix_pick,
    how_picked,
    load_picks,
    pick_box,
    pick_points,
    pick_suggestions,
    render_overlay,
    unpick,
)
from pixelmagic.stages.pixelcut import cut_pieces, start_pixel_cut
from pixelmagic.stages.rebuild import RebuiltPieceRefused
from pixelmagic.stages.recipes import RecipeRefused
from pixelmagic.stages.redraw import live_pieces, redraw_picks, refuse_redraw
from pixelmagic.stages.refine import (
    OLD_ITEM_ROUTE,
    PIXELMAGIC_REFUSAL_CODES,
    WRONG_SIZE,
    needs_attention,
    refine_asset,
    refine_direction,
    refused_suffix,
    require_asset,
    set_refiner_defaults,
    tally_reports,
    wrong_size_sentence,
)
from pixelmagic.stages.suggest import load_suggestions, page_count, suggest_objects
from pixelmagic.store import PROJECT_FILE, AlreadyExists, Store


class _UsageErrorGroup(TyperGroup):
    """The top-level group, which turns an unsafe run id into a usage error.

    The store refuses a generation or cut-up run id that is not one of our
    names (`Store.generation_dir`, `Store.decomposition_dir`), so that a typed
    `../..` never becomes part of a path. Every command that takes such an id
    reaches that refusal, and this is the one place it becomes a usage error
    naming the id, rather than a handler in each command. A command that
    catches `ValueError` around a run id (`pick`, `unpick`) lets
    `UnusableName` through to here, and `redraw` checks its id before it says
    which service will be paid. Commands that translate `UnusableName` for a
    typed name of their own (a scene or direction name) are unaffected.
    """

    def invoke(self, context: typer.Context):
        """Run the command, turning an `UnusableName` into a usage error."""
        try:
            return super().invoke(context)
        except UnusableName as error:
            raise typer.BadParameter(str(error), ctx=context) from error


app = typer.Typer(
    help="Turn a text prompt into game-ready pixel art.",
    no_args_is_help=True,
    cls=_UsageErrorGroup,
)
project_app = typer.Typer(help="Create and list projects (a project is one game).")
direction_app = typer.Typer(help="Create and list art directions inside a project.")
app.add_typer(project_app, name="project")
app.add_typer(direction_app, name="direction")


class Context:
    """The flags every command shares."""

    def __init__(self, root: Path, service: str | None, direction: str | None) -> None:
        """Hold the root folder, the `--service` flag (None when not given) and the direction."""
        self.store = Store(root)
        self.service_override = service
        self.direction_name = direction

    @property
    def service_name(self) -> str | None:
        """The service that draws: the flag, else the project's saved choice, else None."""
        return chosen_service(self.store, self.service_override)

    def direction(self) -> Direction:
        """Load the direction named by `--direction`, or the project default.

        The project is read whether or not `--direction` was given. Every
        command that resolves a direction needs the project anyway — `export`
        stamps its name into the manifest — so reading it here is what turns
        a missing `pixelmagic.json` into one sentence at the boundary rather
        than a `FileNotFoundError` out of the last stage, after the earlier
        stages have already drawn and cut.
        """
        project = load_project(self.store)
        name = self.direction_name or project.default_direction
        if name is None:
            raise typer.BadParameter("no direction given and the project has no default")
        try:
            return self.store.load_direction(name)
        except FileNotFoundError as error:
            # A mistyped --direction is a usage error, not a crash: say which
            # name failed and which ones exist.
            known = ", ".join(self.store.list_directions()) or "none yet"
            raise typer.BadParameter(
                f"no direction called {name!r}; known: {known}"
            ) from error


def load_project(store: Store) -> Project:
    """Read the project sidecar, or say plainly that there is no project here.

    Every command needs the project, and a missing `pixelmagic.json` is the
    first thing a new user hits — so it is one sentence naming the fix, not a
    `FileNotFoundError` from three frames down.
    """
    try:
        return store.load_project()
    except FileNotFoundError as error:
        raise typer.BadParameter(
            f"no project at {store.root} — run `pixelmagic project new` first"
        ) from error


def load_service(name: str) -> ImageService:
    """Build the named image service, as a usage error when the name is bad.

    The lookup itself lives in `registry` because the web page and the MCP
    server need it too and neither can raise a Click exception. This wrapper
    is the CLI's translation of it, and it stays a module-level function
    because tests reach for it by name.
    """
    try:
        return registry.load_service(name)
    except registry.RegistryError as error:
        # `RegistryError` and not bare `ValueError`: a pydantic
        # `ValidationError` is a `ValueError`, and once a real service is
        # registered its constructor could raise one — which belongs in a
        # traceback, not flattened into "usage error, exit 2".
        raise typer.BadParameter(str(error)) from error


def check_service_flag(name: str | None) -> None:
    """Refuse a mistyped `--service` when a long-running face starts, not on its first draw."""
    if name:
        load_service(name)


def build_service(context: Context) -> tuple[str, ImageService]:
    """Build the chosen image service without saying anything about it.

    With nothing chosen it is a usage error carrying the core's one sentence.
    Returns the name too, for `say_drawing_with`.
    """
    try:
        name = require_service(context.store, context.service_override)
    except NoServiceChosen as error:
        raise typer.BadParameter(str(error)) from error
    return name, load_service(name)


def say_drawing_with(name: str) -> None:
    """Say who is about to be paid ("Drawing with gemini.") on standard error.

    Standard error, so a command whose output is piped stays clean.
    """
    typer.echo(f"Drawing with {name}.", err=True)


def draw_service(context: Context) -> ImageService:
    """Build the chosen image service for a command that is about to draw.

    It says so on standard error first ("Drawing with gemini."), so a person
    running a command in a project they were given sees who is about to be
    paid before the call is made. A command with refusals of its own to make
    first (`redraw`) uses `build_service` and `say_drawing_with` instead.
    """
    name, service = build_service(context)
    say_drawing_with(name)
    return service


def load_refiner(name: str) -> Refiner:
    """Build the named refiner, as a usage error when the name is bad."""
    try:
        return registry.load_refiner(name)
    except registry.RegistryError as error:
        raise typer.BadParameter(str(error)) from error


def load_segmenter(name: str) -> Segmenter:
    """Build the named segmenter, as a usage error when the name is bad or not built."""
    try:
        return registry.load_segmenter(name)
    except registry.RegistryError as error:
        raise typer.BadParameter(str(error)) from error


def load_approval(
    wait: bool, auto_approve: bool, *, interactive: bool | None = None
) -> ApprovalHook:
    """Pick the approval hook the flags asked for.

    With neither flag the default follows where the command is running: at a
    terminal it stops and asks, in a script or a scheduled job it approves
    automatically. Silently approving every picture at an interactive prompt
    would defeat the point of the tool, and blocking on a prompt in a script
    would hang it.
    """
    if wait and auto_approve:
        raise typer.BadParameter("choose either --wait or --auto-approve, not both")
    if wait:
        return TerminalApproval()
    if auto_approve:
        return AutoApprove()
    if interactive is None:
        interactive = sys.stdin.isatty()
    return TerminalApproval() if interactive else AutoApprove()


def _refiner_failed(error: Exception) -> typer.Exit:
    """Turn a refiner failure into a message and a non-zero exit.

    The adapter spends real effort on these sentences — which binary is
    missing, which file it would not overwrite — and a traceback throws all of
    it away while looking like a crash in PixelMagic. A `ValueError` gets the
    same treatment: that is how the refine stage reports settings it will not
    send (`background: pick` with no colour) and an override naming an asset
    the direction does not have.
    """
    typer.echo(str(error), err=True)
    return typer.Exit(1)


def _asset_not_kept(error: RefinerError, generation_id: str) -> typer.Exit:
    """Turn a refiner failure into an honest message when the picture is already approved.

    `generate --as-asset` has this shape. It approves inline — as part of
    generating — and only calls the refiner afterwards, in
    `promote_generation`, unlike the web page and the
    MCP server, which run the refiner first precisely so a failure there
    leaves nothing approved yet (see `stages.decompose.promote_generation`'s
    own docstring). Here the picture is already approved by the time the
    refiner can refuse, so `_refiner_failed`'s plain "here is the refiner's
    sentence" is not enough: a person reading it would not know the picture
    still exists, approved, with no asset — this message says both, and names
    the generation id needed to try keeping it again.
    """
    typer.echo(
        f"approved {generation_id}, but it was not kept as an asset: {error}", err=True
    )
    return typer.Exit(1)


def _echo_pitch_note(store: Store, direction_name: str, asset_name: str) -> None:
    """Say, after a scene is kept, when its pitch is a fallback or an unsure reading."""
    note = pitch_note(store, direction_name, asset_name)
    if note:
        typer.echo(note)


def _refused(error: Exception) -> typer.Exit:
    """Turn a plain refusal into a message and a non-zero exit.

    Unlike `_refiner_failed`, nothing here was ever handed to the refiner —
    this is for a stage saying no on its own terms, such as
    `stages.generate.NotApproved`. Guardian minor 7: the `decompose` command
    used to route this through `_refiner_failed`, whose printed message was
    right but whose name told a reader the refiner had failed when it was
    never involved.
    """
    typer.echo(str(error), err=True)
    return typer.Exit(1)


def _rebuilt_refused(error: RebuiltPieceRefused) -> typer.Exit:
    """Word the way forward after a rebuilt piece refused an edit.

    The core's sentence says "redraw it by name"; on the command line that is
    `redraw --only`, which matches a pick's number or name, never a piece's
    (a piece can be renamed). `refine` is the only command that meets this
    refusal and it names no decomposition, so the hint says where to look
    instead of inventing an id.
    """
    hint = (
        "redraw it with `pixelmagic redraw <decomposition-id> --only <pick number>`; "
        "`pixelmagic picks <decomposition-id>` lists the numbers"
    )
    return _refused(ValueError(f"{error} ({hint})"))


def _service_failed(error: ServiceError) -> typer.Exit:
    """Turn an image service's failure into a message and a non-zero exit.

    A service that could not draw is an ordinary outcome — nobody logged in,
    a filter refused the prompt, the network was down — and the service has
    already written the sentence that says which. A traceback out of a
    subprocess or an SDK throws that sentence away and looks like a crash in
    PixelMagic.
    """
    typer.echo(f"{error.service}: {error}", err=True)
    return typer.Exit(1)


def _approval_failed(error: ApprovalExhausted | EOFError) -> typer.Exit:
    """Turn a picture nobody approved into a message and a non-zero exit.

    Two endings look the same from the outside: every allowed round was turned
    down, or whoever was answering walked away (Ctrl-D at the prompt). Neither
    is a fault in PixelMagic, so neither should print like one. The rejected
    attempts stay on disk, which is why the last one is named here.
    """
    if isinstance(error, ApprovalExhausted):
        typer.echo(
            f"{error} — the last attempt is {error.record.request.id}; look at it, "
            f"then try again with different words.",
            err=True,
        )
    else:
        typer.echo("no answer (end of input), so nothing was approved", err=True)
    return typer.Exit(1)


def _echo_reports(reports: Sequence[RefineReport]) -> None:
    """Name the assets a person still has to look at.

    `refine` says this, so a shaky grid or a refused asset is never hidden
    from the person who ran it.
    """
    for report in reports:
        if report.route == OLD_ITEM_ROUTE:
            # Left as it is: this receipt is an old one, from a refine that
            # cannot be run again, so there is nothing for a person to act on.
            continue
        if needs_attention(report):
            typer.echo(f"  {report.asset}: the pixel grid is uncertain — check it")
        if report.error_code in PIXELMAGIC_REFUSAL_CODES:
            # PixelMagic refused this one itself, before the refiner ever saw
            # it: `error_message` already says why (the target it could not
            # use, or the sidecar it could not read), so this line does not
            # repeat the code — it says who said no.
            reason = report.error_message
            typer.echo(f"  {report.asset}: PixelMagic did not send it to the refiner — {reason}")
        elif report.error_code == WRONG_SIZE:
            typer.echo(f"  {report.asset}: {wrong_size_sentence(report)}")
        elif report.error_code:
            # The message carries the detail.
            detail = f": {report.error_message}" if report.error_message else ""
            typer.echo(
                f"  {report.asset}: the refiner refused it ({report.error_code}){detail}"
            )


def _echo_bundle(bundle: ExportBundle) -> None:
    """Say where an export landed, and every asset it had to leave out."""
    typer.echo(f"exported {len(bundle.manifest.entries)} asset(s) to {bundle.directory}")
    typer.echo(f"{bundle.directory / 'manifest.json'}")
    for warning in bundle.warnings:
        typer.echo(f"  note: {warning}")


def _settings_object(flag: str, value: str | None) -> dict:
    """Read one settings flag's JSON object, refusing anything else by name."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except ValueError as error:
        raise typer.BadParameter(f"{flag} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f'{flag} must be a JSON object like \'{{"key": "value"}}\'')
    return parsed


def _problems(error: ValidationError) -> str:
    """Pydantic's complaints as one line, each naming its field.

    Used by `_settings`, which builds a model out of flags a person typed
    before anything on disk is read, so a `ValidationError` there is
    unambiguously a usage error and can be flattened.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}"
        for problem in error.errors()
    )


def _settings(
    preset: str | None, quick: str | None, advanced: str | None, scale: int | None
) -> RefinerSettings:
    """Build the settings block the refine flags describe.

    Pydantic already knows the shape of every key, so a value of the wrong
    type is reported here as a usage error naming the key rather than as a
    validation traceback three frames into the stage.
    """
    try:
        return RefinerSettings(
            preset=preset,
            quick=_settings_object("--quick", quick),
            advanced=_settings_object("--advanced", advanced),
            scale=scale,
        )
    except ValidationError as error:
        raise typer.BadParameter(
            f"those refiner settings are not usable — {_problems(error)}"
        ) from error


def _missing_run(
    store: Store, direction_name: str, decomposition_id: str, *, pixel_cut: bool = False
) -> typer.BadParameter:
    """The usage error for a cut-up run this direction does not have.

    `keep` meets a bare `FileNotFoundError` from `load_decomposition` for an
    unknown id, and that is the only file it ever reads, so every
    `FileNotFoundError` it sees means this. The commands that read
    more than the run's own sidecar are not that simple — they call
    `_run_file_error` instead. `pixel_cut` names the pixel-cut step as the way to
    make a run, for `cut`.
    """
    if pixel_cut:
        return typer.BadParameter(missing_run_sentence(
            store, direction_name, decomposition_id, "pixelmagic pixel-cut {id}",
            making="Starting a pixel cut on the picture",
        ))
    return typer.BadParameter(
        missing_run_sentence(
            store, direction_name, decomposition_id, "pixelmagic suggest {id}"
        )
    )


def _run_file_error(
    store: Store,
    direction_name: str,
    decomposition_id: str,
    error: FileNotFoundError,
    *,
    pixel_cut: bool = False,
) -> typer.BadParameter:
    """Turn a `FileNotFoundError` from a command that reads a cut-up run into a usage error.

    Serves every command whose reads go beyond the run's own
    `decomposition.json` (`pick`, `picks`, `unpick` and the
    like): they also read picks, pictures and a promoted piece's own
    `asset.json`. Any of those missing is a real, separate problem, and the run named on the
    command line is sitting right there — `_missing_run`'s sentence would be false. This is
    only true when the missing file really is `decomposition.json`; anything else is named
    by its own path instead, the way the CLI reports other file problems
    (`load_project`, `Context.direction`).
    """
    sidecar = store.decomposition_dir(direction_name, decomposition_id) / "decomposition.json"
    if error.filename == str(sidecar):
        return _missing_run(store, direction_name, decomposition_id, pixel_cut=pixel_cut)
    return typer.BadParameter(f"could not find {error.filename}")


def find_projects(root: Path) -> list[Path]:
    """Every project folder at or one level below `root`."""
    if (root / "pixelmagic.json").exists():
        return [root]
    if not root.is_dir():
        return []
    return sorted(child for child in root.iterdir() if (child / "pixelmagic.json").exists())


@app.callback()
def main(
    context: typer.Context,
    root: Annotated[Path, typer.Option(help="The project folder.")] = Path("."),
    service: Annotated[
        str | None,
        typer.Option(
            help="Image service to draw with; without it, the project's saved choice. "
            "There is no default: drawing needs one."
        ),
    ] = None,
    direction: Annotated[
        str | None, typer.Option(help="Which art direction to work in.")
    ] = None,
) -> None:
    """Set up the flags every command shares."""
    context.obj = Context(root, service, direction)


@project_app.command("new")
def project_new(context: typer.Context, name: str) -> None:
    """Start a new project (one game)."""
    store: Store = context.obj.store
    try:
        project = store.create_project(name)
    except AlreadyExists as error:
        # The guard itself lives in the store now, so the web page and an MCP
        # agent get it too; this is only the translation into a usage error.
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"created project {project.name} in {store.root}")


@project_app.command("list")
def project_list(context: typer.Context) -> None:
    """List the projects found at or below the root folder."""
    for folder in find_projects(context.obj.store.root):
        # `find_projects` only checks that the sidecar exists. A hand-edited
        # or half-copied one would otherwise end a read-only listing on a
        # traceback and hide every project sorted after it — so the broken
        # folder names itself and the listing carries on. `ValueError` rather
        # than `ValidationError`: a file that is not UTF-8 fails on the decode,
        # before pydantic ever sees it, and that is the same kind of broken.
        try:
            typer.echo(f"{Store(folder).load_project().name}  ({folder})")
        except (OSError, ValueError):
            typer.echo(f"(unreadable {PROJECT_FILE})  ({folder})")


@direction_app.command("new")
def direction_new(
    context: typer.Context,
    style: Annotated[str, typer.Option(help="A paragraph describing the look.")],
    name: Annotated[str | None, typer.Option(help="What to call it.")] = None,
    dont: Annotated[list[str] | None, typer.Option(help="Something to avoid.")] = None,
    cell_size: Annotated[
        int,
        typer.Option(
            # The bound is `Direction`'s own (spec locked decision 3); these
            # are the same two constants, so Typer refuses out of range with
            # a usage error before the core is called, the way `--size`
            # mirrors `MAX_IMAGE_SIZE`. Not a `ValidationError` wrapper:
            # `start_direction` reads `pixelmagic.json` in the same call, and
            # a sidecar that will not parse must stay a traceback.
            min=MIN_CELL_SIZE,
            max=MAX_CELL_SIZE,
            help="How many art pixels one tile is on a side. Tiles and footprints use it.",
        ),
    ] = DEFAULT_CELL_SIZE,
) -> None:
    """Create an art direction. Without --name, the service invents one."""
    store: Store = context.obj.store
    # Read first, for the sentence: `start_direction` reads the project too,
    # but its `FileNotFoundError` would come out bare, and "no project here —
    # run project new first" is the first thing a new user has to be told.
    load_project(store)
    # The service is only asked for a name when none was typed, so a typed
    # name needs no service at all.
    if not (name and name.strip()):
        service: ImageService | None = draw_service(context.obj)
    elif context.obj.service_name:
        service = load_service(context.obj.service_name)
    else:
        service = None
    try:
        direction = start_direction(
            store,
            style=style,
            service=service,
            service_name=context.obj.service_name,
            name=name,
            donts=list(dont or []),
            cell_size=cell_size,
        )
    except (AlreadyExists, UnusableName) as error:
        # Both guards live in the core now, so the web page and an MCP agent
        # get them too; this is only the translation into a usage error.
        # `UnusableName` and not bare `ValueError`, for the reason
        # `direction_rename` gives below: a pydantic `ValidationError` is a
        # `ValueError` as well, and a corrupt `pixelmagic.json` must stay a
        # traceback rather than be flattened into one line.
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"created direction {direction.name} (cell size {direction.cell_size})")


@direction_app.command("list")
def direction_list(context: typer.Context) -> None:
    """List the art directions in this project."""
    for name in context.obj.store.list_directions():
        typer.echo(name)


@direction_app.command("rename")
def direction_rename(context: typer.Context, old: str, new: str) -> None:
    """Rename an art direction, keeping everything inside it."""
    store: Store = context.obj.store
    try:
        direction = store.rename_direction(old, new)
    except (FileNotFoundError, AlreadyExists, UnusableName) as error:
        # `UnusableName` and not bare `ValueError`, the same reason
        # `load_service` catches `RegistryError` and not `ValueError`: the
        # sidecar `rename_direction` reads is validated inside this same call,
        # and a pydantic `ValidationError` is a `ValueError` too — catching
        # the wider family would flatten a corrupt `direction.json` into a
        # usage error instead of the traceback it must stay.
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"renamed {old} to {direction.name}")


@app.command()
def generate(
    context: typer.Context,
    subject: str,
    kind: Annotated[GenerateKind, typer.Option(help="scene or prop.")] = GenerateKind.SCENE,
    background: Annotated[
        BackgroundPolicy, typer.Option(help="Background to ask for.")
    ] = BackgroundPolicy.TRANSPARENT,
    size: Annotated[
        int, typer.Option(min=1, max=MAX_IMAGE_SIZE, help="Square image size in pixels.")
    ] = 1024,
    rounds: Annotated[int, typer.Option(min=1, help="How many tries before giving up.")] = 3,
    wait: Annotated[bool, typer.Option("--wait", help="Ask before continuing.")] = False,
    auto_approve: Annotated[
        bool, typer.Option("--auto-approve", help="Accept whatever comes back.")
    ] = False,
    keep: Annotated[bool, typer.Option(help="Keep the result as a reference.")] = True,
    as_asset: Annotated[
        bool, typer.Option("--as-asset", help="Also keep the whole picture as an asset.")
    ] = False,
    refiner: Annotated[
        str,
        typer.Option(
            help="stub (the default) or cli, the real PixelRefiner; reads the picture's pixel size."
        ),
    ] = registry.DEFAULT_REFINER,
) -> None:
    """Draw one picture and approve it."""
    store: Store = context.obj.store
    direction = context.obj.direction()
    tool = load_refiner(refiner)
    try:
        record = generate_until_approved(
            store,
            direction,
            subject,
            kind=PromptKind(kind.value),
            service=draw_service(context.obj),
            approval=load_approval(wait, auto_approve),
            max_rounds=rounds,
            background=background,
            size=(size, size),
            references=list(direction.references),
        )
    except ServiceError as error:
        raise _service_failed(error) from error
    except (ApprovalExhausted, EOFError) as error:
        raise _approval_failed(error) from error
    if keep:
        promote(store, direction, record)
    if as_asset:
        try:
            staged = promote_generation(store, direction, record, refiner=tool)
        except RefinerError as error:
            raise _asset_not_kept(error, record.request.id) from error
        typer.echo(f"kept the whole picture as asset {staged}")
        _echo_pitch_note(store, direction.name, staged)
    typer.echo(f"approved {record.image_path}")
    typer.echo(record.request.id)


@app.command("keep-scene")
def keep_scene_command(
    context: typer.Context,
    generation_id: Annotated[str, typer.Argument(help="An approved picture, imported or drawn.")],
    refiner: Annotated[
        str,
        typer.Option(
            help="stub (the default) or cli, the real PixelRefiner; reads the picture's pixel size."
        ),
    ] = registry.DEFAULT_REFINER,
    name: Annotated[
        str | None, typer.Option(help="The scene's name; with none, one is made from the subject.")
    ] = None,
) -> None:
    """Keep an already-approved picture as the scene, so it can be refined and exported."""
    store: Store = context.obj.store
    direction = context.obj.direction()
    tool = load_refiner(refiner)
    try:
        record = load_generation(store, direction.name, generation_id)
    except (FileNotFoundError, ValidationError) as error:
        raise typer.BadParameter(
            f"{direction.name} has no readable generation {generation_id!r}"
        ) from error
    try:
        staged = keep_scene_stage(store, direction, record, refiner=tool, name=name)
    except (NotApproved, AlreadyExists, UnusableName) as error:
        raise _refused(error) from error
    except RefinerError as error:
        # Nothing was written, so the plain refiner sentence is the whole truth.
        raise _refiner_failed(error) from error
    typer.echo(f"kept the whole picture as asset {staged}")
    _echo_pitch_note(store, direction.name, staged)


@app.command("rename-scene")
def rename_scene_command(context: typer.Context, scene: str, new_name: str) -> None:
    """Rename a kept scene. An export bundle already made keeps the old name."""
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        renamed = rename_scene_stage(store, direction, scene, new_name)
    except ValidationError:
        raise  # a sidecar that will not parse is a broken file, not a bad name
    except (AlreadyExists, UnusableName) as error:
        raise _refused(error) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"renamed {scene} to {renamed}")


@app.command("keep-reference")
def keep_reference_command(
    context: typer.Context,
    generation_id: Annotated[str, typer.Argument(help="An approved picture, imported or drawn.")],
) -> None:
    """Keep an already-approved picture as a style reference of the direction."""
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        record = load_generation(store, direction.name, generation_id)
    except (FileNotFoundError, ValidationError) as error:
        raise typer.BadParameter(
            f"{direction.name} has no readable generation {generation_id!r}"
        ) from error
    try:
        added = keep_reference_stage(store, direction, record)
    except NotApproved as error:
        raise _refused(error) from error
    name = reference_name(record)
    if added:
        typer.echo(f"kept {record.request.id} as a style reference, references/{name}")
    else:
        typer.echo(f"{record.request.id} is already a style reference, references/{name}")


@app.command("import")
def import_command(
    context: typer.Context,
    picture: Annotated[
        Path, typer.Argument(help="A PNG, JPEG or WebP file to bring into the direction.")
    ],
) -> None:
    """Bring in a picture of your own as an approved scene, ready to cut up."""
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        record = import_picture(store, direction, picture, reviewer=Reviewer.HUMAN)
    except (ValueError, FileNotFoundError) as error:
        raise _refused(error) from error
    typer.echo(f"imported {record.request.imported_from} as {record.request.id}")
    typer.echo(f"style next: pixelmagic keep-reference {record.request.id}")
    typer.echo(f"props next: pixelmagic suggest {record.request.id}, then pick and redraw")
    typer.echo(
        f"tiles next: pixelmagic decompose {record.request.id} --surface {RECIPE_PLACEHOLDER}"
    )
    typer.echo(
        f"already pixel art? pixelmagic pixel-cut {record.request.id} (finds the pixel "
        f"size and says when unsure; add --size 1 only if it is already one pixel per "
        f"art pixel)"
    )


@app.command()
def suggest(
    context: typer.Context,
    generation_id: str,
    segmenter: Annotated[
        str, typer.Option(help="stub or cli; finds the objects on the picture.")
    ] = registry.DEFAULT_SEGMENTER,
) -> None:
    """Find the objects on an approved picture and make numbered contact sheets.

    The first step of cutting props out object by object: look at the contact
    sheets, then `pick` the numbers you want redrawn.
    """
    store: Store = context.obj.store
    direction = context.obj.direction()
    tool = load_segmenter(segmenter)
    try:
        source = load_generation(store, direction.name, generation_id)
    except (FileNotFoundError, ValidationError) as error:
        raise typer.BadParameter(
            f"{direction.name} has no readable generation {generation_id!r}"
        ) from error
    started = time.perf_counter()
    try:
        result = suggest_objects(store, direction, source, tool)
    except SegmenterError as error:
        raise _refused(error) from error
    except NotApproved as error:
        raise _refused(error) from error
    found = load_suggestions(store, direction.name, result.id)
    pages = page_count(found)
    typer.echo(f"suggestions in {result.id}")
    typer.echo(f"{len(found.suggestions)} shown (of {found.raw_count} found), {pages} pages")
    folder = store.suggestions_dir(direction.name, result.id)
    for page in range(1, pages + 1):
        typer.echo(f"  {folder / f'contact-{page:02d}.png'}")
    took = time.perf_counter() - started
    typer.echo(f"this took {'under a second' if took < 1 else f'{took:.0f} seconds'}")
    if found.suggestions:
        numbers = " ".join(str(one.number) for one in found.suggestions[:3])
        typer.echo(f"pick with: pixelmagic pick {result.id} {numbers}")
    lead = "or by point" if found.suggestions else "pick by point"
    typer.echo(
        f"{lead}: pixelmagic pick {result.id} --point X,Y (also --not X,Y, --box X0,Y0,X1,Y1)"
    )


def _parsed_numbers(text: str, count: int, what: str) -> tuple[int, ...]:
    """Read `count` whole numbers separated by commas, or refuse with a usage error."""
    try:
        numbers = tuple(int(part) for part in text.split(","))
    except ValueError:
        numbers = ()
    if len(numbers) != count:
        raise typer.BadParameter(f"{what} must be {count} whole numbers with commas, got {text!r}")
    return numbers


def _parsed_clicks(points: list[str] | None, nots: list[str] | None) -> list[Point]:
    """`--point` values as "this" clicks, then `--not` values as "not this" clicks."""
    clicks = [Point(x=x, y=y, label=1) for x, y in (
        _parsed_numbers(text, 2, "--point X,Y") for text in points or []
    )]
    clicks += [Point(x=x, y=y, label=0) for x, y in (
        _parsed_numbers(text, 2, "--not X,Y") for text in nots or []
    )]
    return clicks


@app.command()
def pick(
    context: typer.Context,
    decomposition_id: str,
    numbers: Annotated[
        list[int] | None, typer.Argument(help="Suggestion numbers off the contact sheet.")
    ] = None,
    name: Annotated[
        str | None, typer.Option(help="Name the pick (one pick only).")
    ] = None,
    point: Annotated[
        list[str] | None,
        typer.Option("--point", help="X,Y on the picture: a spot on the object (repeat it)."),
    ] = None,
    not_point: Annotated[
        list[str] | None,
        typer.Option("--not", help="X,Y on the picture: a spot NOT on the object (repeat it)."),
    ] = None,
    box: Annotated[
        str | None,
        typer.Option(
            help="X0,Y0,X1,Y1 on the picture: the object lies inside this box "
            "(right and bottom exclusive)."
        ),
    ] = None,
    fix: Annotated[
        int | None,
        typer.Option(help="Redo this pick's outline from --point/--not/--box, not add one."),
    ] = None,
    segmenter: Annotated[
        str,
        typer.Option(
            help="stub or cli; finds the outline. Only used with --point, --box or --fix."
        ),
    ] = registry.DEFAULT_SEGMENTER,
) -> None:
    """Pick objects to be redrawn: by suggestion number, or by point or box.

    `pick ID 3 7` takes suggestions off the contact sheet. `pick ID --point
    120,80 --not 150,90` or `pick ID --box 10,10,90,70` asks the segmenter for
    the outline under your clicks. `--fix N` redoes pick N's outline.
    """
    store: Store = context.obj.store
    direction = context.obj.direction()
    clicks = _parsed_clicks(point, not_point)
    drawn = _parsed_numbers(box, 4, "--box X0,Y0,X1,Y1") if box is not None else None
    by_clicks = bool(clicks) or drawn is not None
    if fix is not None and name is not None:
        raise typer.BadParameter("--fix keeps the pick's name; --name cannot be combined with it")
    if numbers and (by_clicks or fix is not None):
        raise typer.BadParameter(
            "suggestion numbers cannot be combined with --point, --not, --box or --fix"
        )
    if fix is not None and not by_clicks:
        raise typer.BadParameter("--fix needs --point or --box to say where the new outline is")
    if not numbers and not by_clicks:
        raise typer.BadParameter("give suggestion numbers, or --point / --box")
    try:
        if not by_clicks:
            made = pick_suggestions(store, direction, decomposition_id, numbers, name=name)
        else:
            tool = load_segmenter(segmenter)
            if fix is not None:
                made = [fix_pick(
                    store, direction, decomposition_id, fix,
                    points=clicks, box=drawn, segmenter=tool,
                )]
            elif drawn is not None:
                made = [pick_box(
                    store, direction, decomposition_id, drawn,
                    points=clicks, name=name, segmenter=tool,
                )]
            else:
                made = [pick_points(
                    store, direction, decomposition_id, clicks, name=name, segmenter=tool
                )]
    except SegmenterError as error:
        raise _refused(error) from error
    except FileNotFoundError as error:
        raise _run_file_error(store, direction.name, decomposition_id, error) from error
    except ValidationError:
        raise  # a sidecar that will not parse is a broken file, not a bad number
    except UnusableName:
        raise  # an unsafe run id is the group's usage error, not this command's
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    for one in made:
        if fix is not None:
            typer.echo(
                f"fixed pick {one.number}: now {how_picked(one)}, outline box {one.picture_box}"
            )
            run = load_decomposition(store, direction.name, decomposition_id)
            if live_pieces(run, one):
                if run.pixel_grid is not None:
                    typer.echo(
                        f"a piece already cut from it is unchanged; replace it with: "
                        f"pixelmagic cut {decomposition_id} --only {one.number}"
                    )
                else:
                    typer.echo(
                        f"a piece already redrawn from it is unchanged; replace it with: "
                        f"pixelmagic redraw {decomposition_id} --only {one.name or one.number}"
                    )
        elif one.suggestion is not None:
            typer.echo(f"picked suggestion {one.suggestion} as pick {one.number}")
        else:
            typer.echo(
                f"picked {how_picked(one)} as pick {one.number}, outline box {one.picture_box}"
            )


@app.command()
def picks(context: typer.Context, decomposition_id: str) -> None:
    """List the picks and draw them, outlined and numbered, on the picture."""
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        chosen = load_picks(store, direction.name, decomposition_id)
        overlay = render_overlay(store, direction, decomposition_id)
    except FileNotFoundError as error:
        raise _run_file_error(store, direction.name, decomposition_id, error) from error
    named_at = (
        "named when cut"
        if load_decomposition(store, direction.name, decomposition_id).pixel_grid is not None
        else "named at redraw"
    )
    for one in chosen.picks:
        where = f"{how_picked(one)}  outline box {one.picture_box}  area {one.area}"
        if one.name:
            typer.echo(f"{one.number}: {one.name}  {where}")
        else:
            typer.echo(f"{one.number}: {where}  ({named_at})")
    if not chosen.picks:
        typer.echo("nothing picked yet")
    typer.echo(f"overlay: {overlay}")


@app.command("unpick")
def unpick_command(context: typer.Context, decomposition_id: str, pick_number: int) -> None:
    """Take one pick back; its number is never used again."""
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        removed = unpick(store, direction, decomposition_id, pick_number)
    except FileNotFoundError as error:
        raise _run_file_error(store, direction.name, decomposition_id, error) from error
    except ValidationError:
        raise
    except UnusableName:
        raise  # an unsafe run id is the group's usage error, not this command's
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"removed pick {removed.number}")


def _keep_pieces(store: Store, direction: Direction, run: Decomposition) -> list[str]:
    """Keep a run's pieces as assets, turning the old-pieces refusal into one line."""
    try:
        return promote_cutouts(store, direction, run)
    except OldPieceRefused as error:
        raise _refused(error) from error


def _parsed_surfaces(surface: list[str] | None) -> list[Surface]:
    """Turn the `--surface` values into surfaces, or refuse with a usage error.

    `decompose` uses it so a bad name is refused before anything is drawn.
    """
    try:
        return parse_surfaces(surface or [])
    except UnusableName as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def decompose(
    context: typer.Context,
    generation_id: str,
    surface: Annotated[
        list[str] | None,
        typer.Option(
            help='A surface to tile; repeat it for more. A surface is one string: "name:kind; box '
            'x,y x,y; shape x,y x,y x,y; ..." where kind is plaster, stone or wood, '
            "box is two opposite corners of a small plain patch whose colours are "
            "read, and each shape is the corners (at least 3) of a rough outline "
            "over the surface, with one or more shapes (picture pixels), e.g. "
            f"{RECIPE_EXAMPLE}. With none, decompose is refused and says what to do instead."
        ),
    ] = None,
    keep: Annotated[
        bool,
        typer.Option(
            help="Make the pieces assets straight away. --no-keep leaves them as "
            "pieces so you can look at them first, then run `keep`."
        ),
    ] = True,
) -> None:
    """Draw the tiles of the surfaces you typed on an approved picture.

    Each surface is drawn from the picture's own colours, with no image
    service. The pieces made stay in the direction's decompositions folder with
    a sidecar naming them.
    """
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        source = load_generation(store, direction.name, generation_id)
    except (FileNotFoundError, ValidationError) as error:
        raise typer.BadParameter(
            f"{direction.name} has no readable generation {generation_id!r}"
        ) from error
    surfaces = _parsed_surfaces(surface)
    try:
        result = decompose_stage(
            store,
            direction,
            source,
            surfaces=surfaces,
        )
    except RecipeRefused as error:
        # `RecipeSeamBroken` is deliberately not caught: a bug in a recipe
        # should stop the run loudly, with its message.
        raise _refused(error) from error
    except NoPicks as error:
        typer.echo(
            f"{error} — run `pixelmagic suggest {generation_id}`, then "
            f"`pixelmagic pick <decomposition> ...` to choose objects, then "
            f"`pixelmagic redraw <decomposition>`; or pass --surface "
            f"{RECIPE_PLACEHOLDER} (repeat for more) to draw tiles",
            err=True,
        )
        raise typer.Exit(1) from error
    except NotApproved as error:
        # `decompose` refuses a `source` nobody approved yet (moved out of the
        # web page alone, into the core, so every face gets it for free) — one
        # line rather than a traceback, through `_refused` because the refiner
        # was never asked to do anything here.
        raise _refused(error) from error
    if not result.cutouts:
        typer.echo("no piece was made")
    elif keep:
        names = _keep_pieces(store, direction, result)
        typer.echo(f"{len(names)} piece(s): {', '.join(names)}")
    else:
        names = [cutout.name for cutout in result.cutouts]
        typer.echo(f"{len(names)} piece(s) in {result.id}, not kept yet: {', '.join(names)}")
        typer.echo(f"look at them, then run `pixelmagic keep {result.id}`")
    for warning in result.warnings:
        typer.echo(f"  note: {warning}")


@app.command()
def redraw(
    context: typer.Context,
    decomposition_id: str,
    only: Annotated[
        list[str] | None,
        typer.Option(
            help="A pick number or name to redraw again (repeat it). With none, "
            "every pick that has no piece yet is redrawn."
        ),
    ] = None,
    keep: Annotated[
        bool,
        typer.Option(
            help="Make the pieces assets straight away. --no-keep leaves them as "
            "pieces so you can look at them first, then run `keep`."
        ),
    ] = True,
) -> None:
    """Redraw the picked objects, a few at a time, into separate pieces.

    Each group of picks is drawn in one call and cut back into one piece per
    object. A group that fails leaves its objects picked, so running this
    again redraws just those.
    """
    store: Store = context.obj.store
    direction = context.obj.direction()

    def report(group: Group) -> None:
        """Say what became of one group, the moment it lands."""
        made = sum(1 for receipt in group.receipts if receipt.piece_size is not None)
        seconds = sum(attempt.seconds for attempt in group.attempts)
        state = f"{made} of {len(group.cells)} piece(s)" if group.generation else "failed"
        typer.echo(f"{group.id}: {state}, {seconds:.0f}s")

    # An unsafe run id is refused here, before "Drawing with ..." is said.
    store.decomposition_dir(direction.name, decomposition_id)
    try:
        # Every refusal the core can make up front comes before "Drawing with ..." is said.
        name, service = build_service(context.obj)
        refuse_redraw(
            load_decomposition(store, direction.name, decomposition_id),
            load_picks(store, direction.name, decomposition_id).picks,
            service,
        )
        say_drawing_with(name)
        result = redraw_picks(
            store, direction, decomposition_id,
            service=service, only=only, on_group=report,
        )
    except FileNotFoundError as error:
        raise _run_file_error(store, direction.name, decomposition_id, error) from error
    except ServiceError as error:
        raise _service_failed(error) from error
    except ValidationError:
        raise  # a sidecar that will not parse is a broken file, not a refusal
    except NoPicks as error:
        raise _refused(
            ValueError(f"{error} (`pixelmagic suggest <picture>`, then `pixelmagic pick <id> ...`)")
        ) from error
    except ValueError as error:
        raise _refused(error) from error
    if not result.cutouts:
        typer.echo("no piece was made")
    elif keep:
        names = _keep_pieces(store, direction, result)
        typer.echo(f"{len(names)} piece(s): {', '.join(names)}")
    else:
        names = [cutout.name for cutout in result.cutouts]
        typer.echo(f"{len(names)} piece(s) in {result.id}, not kept yet: {', '.join(names)}")
        typer.echo(f"look at them, then run `pixelmagic keep {result.id}`")
    for warning in result.warnings:
        typer.echo(f"  note: {warning}")
    waiting = [
        pick.number for pick in load_picks(store, direction.name, result.id).picks
        if not live_pieces(result, pick)
    ]
    if waiting:
        numbers = ", ".join(str(number) for number in waiting)
        typer.echo(
            f"pick(s) {numbers} have no piece: `pixelmagic redraw {result.id}` tries them "
            f"again (`--only <n>` for one)"
        )


@app.command("pixel-cut")
def pixel_cut_command(
    context: typer.Context,
    generation_id: str,
    size: Annotated[
        float | None,
        typer.Option(
            help="Picture pixels per art pixel; 1 means the picture is already one pixel "
            "per art pixel. Left out, the size is found."
        ),
    ] = None,
) -> None:
    """Start cutting props straight from a picture that is already pixel art.

    No image service and no PixelRefiner is involved: a prop is the picture's
    own pixels inside an outline you pick. Next `pick` objects by --box or
    --point, then `cut` them.
    """
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        source = load_generation(store, direction.name, generation_id)
    except (FileNotFoundError, ValidationError) as error:
        raise typer.BadParameter(
            f"{direction.name} has no readable generation {generation_id!r}"
        ) from error
    try:
        result = start_pixel_cut(store, direction, source, size=size)
    except ValidationError:
        raise  # an unreadable picture record is a broken file, not a refusal
    except UnusableName:
        raise  # an unsafe id is the group's usage error, not this command's
    except (NotApproved, ValueError) as error:
        raise _refused(error) from error
    grid = result.pixel_grid
    typer.echo(f"pixel cut {result.id}: pixel size {grid.pixel_size:g}")
    for warning in result.warnings:
        typer.echo(f"  note: {warning}")
    typer.echo(
        f"next: pixelmagic pick {result.id} --box X0,Y0,X1,Y1, "
        f"then pixelmagic cut {result.id}"
    )


@app.command("cut")
def cut_command(
    context: typer.Context,
    decomposition_id: str,
    only: Annotated[
        list[int] | None,
        typer.Option(help="A pick number to cut (repeat it). With none, every pick is cut."),
    ] = None,
    keep: Annotated[
        bool,
        typer.Option(
            help="Make the pieces assets straight away. --no-keep leaves them as "
            "pieces so you can look at them first, then run `keep`."
        ),
    ] = True,
) -> None:
    """Cut the picks of a `pixel-cut` run into props from the picture's own pixels."""
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        made = cut_pieces(store, direction, decomposition_id, only=only)
    except FileNotFoundError as error:
        raise _run_file_error(
            store, direction.name, decomposition_id, error, pixel_cut=True
        ) from error
    except ValidationError:
        raise  # a sidecar that will not parse is a broken file, not a refusal
    except UnusableName:
        raise  # an unsafe run id is the group's usage error, not this command's
    except ValueError as error:
        raise _refused(error) from error
    result = load_decomposition(store, direction.name, decomposition_id)
    if keep:
        names = _keep_pieces(store, direction, result)
        typer.echo(f"{len(names)} piece(s): {', '.join(names)}")
    else:
        names = [cutout.name for cutout in made]
        typer.echo(f"{len(names)} piece(s) in {result.id}, not kept yet: {', '.join(names)}")
        typer.echo(f"look at them, then run `pixelmagic keep {result.id}`")
    for warning in result.warnings:
        typer.echo(f"  note: {warning}")


@app.command()
def keep(context: typer.Context, decomposition_id: str) -> None:
    """Turn a cut-up run's pieces into assets, once they are right.

    The other half of `decompose --no-keep`: the same promotion the web
    page's "Keep these pieces as assets" button and the agent's
    `keep_pieces` do. A run that still holds old pieces (from the old props
    sheet, a bordered surface, or before props were rebuilt on their own pixel
    grid) is refused with the way forward.
    """
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        cut = load_decomposition(store, direction.name, decomposition_id)
    except FileNotFoundError as error:
        raise _missing_run(store, direction.name, decomposition_id) from error
    names = _keep_pieces(store, direction, cut)
    typer.echo(f"kept {len(names)} asset(s): {', '.join(names)}")
    for warning in cut.warnings:
        typer.echo(f"  note: {warning}")


@app.command("set-scene-pitch")
def set_scene_pitch(context: typer.Context, scene: str, value: float) -> None:
    """Set the scene's pitch — source pixels per art pixel.

    Spec locked decision 1: pitch can be set from any face. Only a kept scene
    has a pitch to set; a rebuilt prop's is its own and a tile has none.
    """
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        result = set_scene_pitch_stage(store, direction.name, scene, value)
    except ValidationError:
        raise  # a sidecar that will not parse is a broken file, not a bad pitch
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"pitch of {result.name} is now {result.pitch} ({result.pitch_source.value}); "
        f"target is {result.target[0]} x {result.target[1]} art pixels"
    )


@app.command()
def anchor(context: typer.Context, asset: str, x: int, y: int) -> None:
    """Say where a prop stands, in art pixels of its refined picture.

    (0, 0) is the top-left corner and the default is bottom centre. The
    point has to be inside the prop's target size; the core refuses one that
    is not (spec locked decision 6: editable, never guessed).
    """
    store: Store = context.obj.store
    direction = context.obj.direction()
    try:
        record = set_anchor_stage(store, direction.name, asset, (x, y))
    except ValidationError:
        raise  # a broken asset.json is not a bad anchor
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    # `set_anchor` always returns a record with both fields set: it refuses
    # an asset with no target before it ever writes one, and it always
    # writes the anchor it was just given — so there is no fallback to fall
    # back to.
    typer.echo(
        f"{record.name} now stands on ({record.anchor[0]}, {record.anchor[1]}) of its "
        f"{record.target[0]} x {record.target[1]} art pixels"
    )


@app.command()
def refine(
    context: typer.Context,
    refiner: Annotated[
        str, typer.Option(help="stub (the default) or cli, the real PixelRefiner.")
    ] = registry.DEFAULT_REFINER,
    asset: Annotated[
        str | None, typer.Option(help="Refine only this asset, with these settings.")
    ] = None,
    preset: Annotated[str | None, typer.Option(help="A PixelRefiner preset.")] = None,
    quick: Annotated[str | None, typer.Option(help="Quick settings as JSON.")] = None,
    advanced: Annotated[str | None, typer.Option(help="Advanced settings as JSON.")] = None,
    scale: Annotated[
        int | None, typer.Option(min=1, help="Nearest-neighbour upscale.")
    ] = None,
) -> None:
    """Turn the scene into true pixel art, and say what was left as it is.

    Settings flags on their own become the direction's new defaults. Combined
    with `--asset` they are a one-off override for that asset and the
    direction's defaults are left alone. `--asset` is the per-asset layer the
    spec describes, reachable without writing Python.
    """
    store: Store = context.obj.store
    direction = context.obj.direction()
    tool = load_refiner(refiner)
    given = _settings(preset, quick, advanced, scale)
    if asset is not None:
        try:
            asset = require_asset(store, direction.name, asset)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    try:
        if asset is not None:
            reports = [refine_asset(store, direction, asset, refiner=tool, override=given)]
        else:
            if preset or quick or advanced or scale:
                direction = set_refiner_defaults(store, direction, given)
            reports = refine_direction(store, direction, refiner=tool)
    except ValidationError:
        # A `ValidationError` is a `ValueError`, and a sidecar that will not
        # parse is a broken file, not a refiner refusing an asset. Flattening
        # its several lines into one would name the wrong problem, so it is
        # left to travel as itself.
        raise
    except RebuiltPieceRefused as error:
        raise _rebuilt_refused(error) from error
    except (RefinerError, ValueError) as error:
        raise _refiner_failed(error) from error
    tally = tally_reports(reports)
    rebuilt, recipe, old_items = tally.rebuilt, tally.recipe, tally.old_items
    typer.echo(f"refined {tally.refined} asset(s) with {tool.name}{refused_suffix(tally)}")
    if rebuilt:
        typer.echo(
            f"{rebuilt} rebuilt prop(s) were not sent to the refiner: they were rebuilt "
            f"on their own pixel grid, so refine settings do not apply to them"
        )
    if recipe:
        typer.echo(
            f"{recipe} recipe tile(s) were not sent to the refiner: they are drawn "
            f"by their recipe, so refine settings do not apply to them"
        )
    if tally.pixel_cut:
        typer.echo(
            f"{tally.pixel_cut} pixel-cut prop(s) were not sent to the refiner: they are "
            f"the picture's own pixels, so refine settings do not apply to them"
        )
    if old_items:
        typer.echo(
            f"{old_items} old tile(s) or prop(s) were left as they are: they cannot be "
            f"refined again"
        )
    _echo_reports(reports)
    if tally.refused:
        raise typer.Exit(code=1)


@app.command()
def export(
    context: typer.Context,
    scale: Annotated[
        list[int] | None,
        typer.Option(min=1, help="Sizes to write; native size is always written too."),
    ] = None,
) -> None:
    """Write the finished bundle: PNGs, manifest.json and Aseprite files."""
    try:
        bundle = export_stage(
            context.obj.store,
            context.obj.direction(),
            scales=tuple(scale) if scale else DEFAULT_SCALES,
        )
    except ValueError as error:
        # `--scale`'s own `min=1` catches a floor of zero before this ever
        # runs, but it has no ceiling: `stages.export.export` is where the
        # upper bound lives, and its refusal is a `ValueError` like the
        # refiner's, so it gets the same one-line treatment rather than a
        # traceback.
        raise _refiner_failed(error) from error
    _echo_bundle(bundle)


@app.command()
def web(
    context: typer.Context,
    host: Annotated[str, typer.Option(help="Which address to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Which port.")] = 8000,
    refiner: Annotated[
        str, typer.Option(help="stub (the default) or cli, the real PixelRefiner.")
    ] = registry.DEFAULT_REFINER,
    segmenter: Annotated[
        str, typer.Option(help="stub or cli; finds the objects on the picture.")
    ] = registry.DEFAULT_SEGMENTER,
) -> None:
    """Open the web page for this project folder.

    FastAPI is imported here rather than at the top of the file so that
    `pixelmagic --help` and every other command stay as fast as they were
    before the web page existed.
    """
    from pixelmagic.web.app import LOOPBACK_HOSTS, create_app, serve

    store: Store = context.obj.store
    # Say plainly that there is no project here, rather than serving a page
    # whose every route would 404.
    load_project(store)
    check_service_flag(context.obj.service_override)
    if host not in LOOPBACK_HOSTS:
        raise typer.BadParameter(
            f"PixelMagic's web page is local-only; use one of "
            f"{', '.join(sorted(LOOPBACK_HOSTS))}"
        )
    application = create_app(
        store.root,
        service=context.obj.service_override,
        refiner=load_refiner(refiner),
        segmenter=load_segmenter(segmenter),
    )
    typer.echo(f"PixelMagic is at http://{host}:{port}/ — press Ctrl-C to stop")
    serve(application, host=host, port=port)


@app.command()
def mcp(
    context: typer.Context,
    refiner: Annotated[
        str, typer.Option(help="stub (the default) or cli, the real PixelRefiner.")
    ] = registry.DEFAULT_REFINER,
    segmenter: Annotated[
        str, typer.Option(help="stub or cli; finds the objects on a picture.")
    ] = registry.DEFAULT_SEGMENTER,
) -> None:
    """Serve PixelMagic to an AI agent over MCP (stdio).

    Nothing may be printed to standard output here: on stdio, standard output
    *is* the protocol, and one friendly line of greeting makes every message
    after it unreadable to the client. Anything worth saying goes to standard
    error, which the client shows as a log.

    The server is imported inside the command so that `pixelmagic --help` and
    every other command stay as fast as they were before the SDK arrived.
    """
    from pixelmagic.mcp_server import create_server

    store: Store = context.obj.store
    load_project(store)
    check_service_flag(context.obj.service_override)
    server = create_server(
        store.root,
        service=context.obj.service_override,
        refiner=load_refiner(refiner),
        segmenter=load_segmenter(segmenter),
    )
    typer.echo(f"PixelMagic MCP server on stdio, project {store.root}", err=True)
    server.run()
