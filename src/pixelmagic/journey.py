"""Where a scene stands on its seven-step road, read from the folder.

The faces draw the timeline and the home cards; what each step means and
whether it is done is decided here, once, so the command line, the web page
and the MCP server cannot disagree. Nothing about a scene's progress is
stored: every answer is read off the sidecars each time, so it can never go
stale. The one thing stored is the scene the person chose for Refine and
Export (`scene-choice.json`, by generation id). No URLs and no wording
beyond the step labels live here; a face builds its own links.

A sidecar that will not parse counts as "not there" for that one file and
never raises, the same rule `web.projects.direction_rows` follows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import NamedTuple

from pydantic import AwareDatetime, BaseModel

from pixelmagic.model import (
    AssetKind,
    Decomposition,
    Manifest,
    PromptKind,
    RefineReport,
    ReviewStatus,
    SceneChoice,
)
from pixelmagic.stages.decompose import decompositions_of, is_props_run, scene_holding
from pixelmagic.stages.export import MANIFEST_FILE
from pixelmagic.stages.generate import load_generation
from pixelmagic.store import ID_FORMAT, Store


class StepKey(StrEnum):
    """The seven steps, in order. The values are stable and safe to put in a URL."""

    START = "start"
    APPROVE = "approve"
    KEEP = "keep"
    PICK = "pick"
    SURFACES = "surfaces"
    REFINE = "refine"
    EXPORT = "export"


#: The label each step carries on screen.
STEP_LABELS: dict[StepKey, str] = {
    StepKey.START: "Start",
    StepKey.APPROVE: "Approve",
    StepKey.KEEP: "Keep scene",
    StepKey.PICK: "Pick props",
    StepKey.SURFACES: "Surfaces",
    StepKey.REFINE: "Refine",
    StepKey.EXPORT: "Export",
}


class StepState(StrEnum):
    """How a step looks: finished, the one to do now, the one after, or not yet."""

    DONE = "done"
    CURRENT = "current"
    NEXT = "next"
    WAITING = "waiting"


class Step(BaseModel):
    """One step of a scene's road."""

    model_config = {"frozen": True}

    key: StepKey
    label: str
    state: StepState
    #: The decomposition a face should link to. Set on the pick step
    #: only, and None when there is none yet.
    decomposition: str | None = None


class DirectionSummary(BaseModel):
    """What a home card shows about one direction."""

    model_config = {"frozen": True}

    name: str
    cell_size: int
    scene_count: int
    asset_count: int
    #: The newest scene picture's generation id and its subject; None with no scene.
    latest_scene: str | None
    latest_subject: str | None
    last_edited: AwareDatetime


def _generation_scenes(store: Store, direction: str) -> list[tuple[str, str]]:
    """`(generation id, subject)` of every scene picture, newest first.

    A scene picture is a generation whose request kind is scene (an import is
    one too) and whose review is not rejected. An unreadable generation is
    left out.
    """
    scenes: list[tuple[str, str]] = []
    for generation_id in store.list_generations(direction):
        try:
            record = load_generation(store, direction, generation_id)
        except (OSError, ValueError):
            continue
        rejected = record.review.status is ReviewStatus.REJECTED
        if record.request.kind is PromptKind.SCENE and not rejected:
            scenes.append((generation_id, record.request.subject))
    return scenes


def latest_scene(store: Store, direction: str) -> str | None:
    """The id of the direction's newest scene picture, or None when there is none."""
    scenes = _generation_scenes(store, direction)
    return scenes[0][0] if scenes else None


class KeptScene(BaseModel):
    """One kept scene, as the scene chooser lists it."""

    model_config = {"frozen": True}

    #: The scene picture's generation id.
    generation: str
    #: The scene asset's name.
    asset: str


def kept_scenes(store: Store, direction: str) -> list[KeptScene]:
    """Every kept scene of the direction, newest first."""
    found: list[KeptScene] = []
    for generation_id, _subject in _generation_scenes(store, direction):
        asset = scene_holding(store, direction, generation_id)
        if asset is not None:
            found.append(KeptScene(generation=generation_id, asset=asset))
    return found


class _PiecesDone(NamedTuple):
    """Whether a picture's decompositions hold finished props and finished surfaces."""

    props: bool
    surfaces: bool


def _pieces_done(decompositions: list[Decomposition]) -> _PiecesDone:
    """The one test for the Pick props and Surfaces steps being done.

    Props are done when some cutout was rebuilt on its own grid or cut from a
    pixel-art picture's own pixels, surfaces
    when some cutout is a recipe tile. `scene_steps` and the followed-scene
    default both use it, so they cannot disagree.
    """
    cutouts = [cutout for decomposition in decompositions for cutout in decomposition.cutouts]
    return _PiecesDone(
        props=any(
            cutout.grid_rebuild is not None or cutout.pixel_cut is not None for cutout in cutouts
        ),
        surfaces=any(cutout.recipe is not None for cutout in cutouts),
    )


class PropsRuns(NamedTuple):
    """The newest props run of each road for one picture; None where it has none."""

    suggestions: str | None
    pixel_cut: str | None


def props_runs(store: Store, direction: str, generation: str) -> PropsRuns:
    """The newest object-by-object run of each road cut from `generation`.

    The suggestions road is the paid one (find objects, pick, redraw); the pixel cut is
    the free one for a picture that is already pixel art. Each road has its own newest
    run, so starting one never hides the other. The Pick props step still links the
    newest run of either (`scene_steps`).
    """
    runs = [
        decomposition
        for decomposition in decompositions_of(store, direction, generation)
        if is_props_run(store, direction, decomposition)
    ]
    return PropsRuns(
        suggestions=next((run.id for run in runs if run.pixel_grid is None), None),
        pixel_cut=next((run.id for run in runs if run.pixel_grid is not None), None),
    )


def choose_scene(store: Store, direction: str, generation: str) -> None:
    """Record the scene Refine and Export follow, by generation id.

    Raises `FileNotFoundError` with a sentence when the generation is not a
    kept scene of this direction; nothing is written then.
    """
    if all(scene.generation != generation for scene in kept_scenes(store, direction)):
        raise FileNotFoundError(
            f"{generation} is not a kept scene of {direction}, so it cannot be chosen."
        )
    store.write_model(store.scene_choice_file(direction), SceneChoice(generation=generation))


def followed_scene(store: Store, direction: str) -> str | None:
    """The scene picture the Refine and Export pages follow, as a generation id.

    In order: the person's choice while it still names a kept scene of this
    direction; the newest kept scene that has pieces (props or surfaces); the
    newest kept scene; with none kept, `latest_scene` (the newest scene
    picture, or None). A choice that is stale (its scene gone) or whose
    sidecar will not parse is ignored, never an error.
    """
    kept = kept_scenes(store, direction)
    try:
        chosen = store.read_model(store.scene_choice_file(direction), SceneChoice).generation
    except (OSError, ValueError):
        chosen = None
    if any(scene.generation == chosen for scene in kept):
        return chosen
    for scene in kept:
        pieces = _pieces_done(decompositions_of(store, direction, scene.generation))
        if pieces.props or pieces.surfaces:
            return scene.generation
    return kept[0].generation if kept else latest_scene(store, direction)


def _is_refined(store: Store, direction: str, asset: str) -> bool:
    """Whether the asset has a refined picture and a refine record with no refusal."""
    folder = store.asset_dir(direction, asset)
    if not (folder / "refined.png").exists():
        return False
    try:
        report = store.read_model(folder / "refine.json", RefineReport)
    except (OSError, ValueError):
        return False
    return report.error_code is None


def _is_exported(store: Store, direction: str, asset: str) -> bool:
    """Whether some bundle of this direction lists the scene asset.

    Bundles are found by reading each `manifest.json`, never by parsing folder
    names; an unreadable manifest is not a bundle.
    """
    root = store.root / "exports"
    if not root.is_dir():
        return False
    for folder in root.iterdir():
        try:
            manifest = store.read_model(folder / MANIFEST_FILE, Manifest)
        except (OSError, ValueError):
            continue
        if manifest.direction == direction and any(
            entry.kind is AssetKind.SCENE and entry.name == asset for entry in manifest.entries
        ):
            return True
    return False


def scene_steps(store: Store, direction: str, generation: str | None) -> list[Step]:
    """The seven steps of one scene, in order, each with its state.

    `generation` is a generation id, or None before any picture exists. A
    generation that does not exist raises `FileNotFoundError`. Each step's
    done-ness is independent of the others; "current" is the first step not
    done, "next" the step right after it when that is not done either, and the
    rest of the undone ones wait. With all seven done there is no current.
    """
    done = dict.fromkeys(StepKey, False)
    pick_link: str | None = None
    if generation is not None:
        record = load_generation(store, direction, generation)
        done[StepKey.START] = True
        done[StepKey.APPROVE] = record.review.status is ReviewStatus.APPROVED
        kept = scene_holding(store, direction, generation)
        done[StepKey.KEEP] = kept is not None
        decompositions = decompositions_of(store, direction, generation)
        pick_link = next(
            (
                decomposition.id
                for decomposition in decompositions
                if is_props_run(store, direction, decomposition)
            ),
            None,
        )
        pieces = _pieces_done(decompositions)
        done[StepKey.PICK] = pieces.props
        done[StepKey.SURFACES] = pieces.surfaces
        if kept is not None:
            done[StepKey.REFINE] = _is_refined(store, direction, kept)
            done[StepKey.EXPORT] = _is_exported(store, direction, kept)

    links = {StepKey.PICK: pick_link}
    first_undone = next((key for key in StepKey if not done[key]), None)
    steps: list[Step] = []
    previous: StepKey | None = None
    for key in StepKey:
        if done[key]:
            state = StepState.DONE
        elif key is first_undone:
            state = StepState.CURRENT
        elif previous is first_undone:
            state = StepState.NEXT
        else:
            state = StepState.WAITING
        steps.append(
            Step(key=key, label=STEP_LABELS[key], state=state, decomposition=links.get(key))
        )
        previous = key
    return steps


def _id_time(identifier: str) -> datetime | None:
    """The UTC time in a timestamp-slug id, ignoring a `-3` suffix; None if it has none."""
    try:
        return datetime.strptime(identifier[:15], ID_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def direction_summary(store: Store, direction: str) -> DirectionSummary:
    """The facts a home card shows about one direction.

    `last_edited` is the newest time among its generation and decomposition
    ids, falling back to the direction's `created`.
    """
    loaded = store.load_direction(direction)
    scenes = _generation_scenes(store, direction)
    times = [
        moment
        for identifier in (
            *store.list_generations(direction),
            *store.list_decompositions(direction),
        )
        if (moment := _id_time(identifier)) is not None
    ]
    return DirectionSummary(
        name=loaded.name,
        cell_size=loaded.cell_size,
        scene_count=len(scenes),
        asset_count=len(store.list_assets(direction)),
        latest_scene=scenes[0][0] if scenes else None,
        latest_subject=scenes[0][1] if scenes else None,
        last_edited=max(times, default=loaded.created),
    )


def order_for_home(summaries: list[DirectionSummary]) -> list[DirectionSummary]:
    """Put the direction with the newest scene first; those with no scene go after.

    The scene-less ones are ordered by most recent edit, newest first.
    """
    with_scene = sorted(
        (s for s in summaries if s.latest_scene is not None),
        key=lambda s: s.latest_scene or "",
        reverse=True,
    )
    without = sorted((s for s in summaries if s.latest_scene is None),
                     key=lambda s: s.last_edited, reverse=True)
    return [*with_scene, *without]
