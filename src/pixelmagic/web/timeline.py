"""What the page frame needs to draw a scene's seven-step timeline.

`journey.scene_steps` says which steps are done; this module, a face's own
job, turns that into links and the names the frame's template reads. No rule
about the road lives here and no URL lives in the core.
"""

from __future__ import annotations

from pixelmagic import journey
from pixelmagic.journey import Step, StepKey, StepState
from pixelmagic.web.deps import Workbench


def _href(step: Step, direction: str, generation: str | None) -> str | None:
    """Where a step's label leads, or None when it cannot be visited yet.

    A waiting step is never a link. Approve and Keep scene live on the
    generation page, so without a generation they have nowhere to go. Pick
    props goes to its own page once a cut-up run exists and falls back to the
    generation page, which holds the form that starts it. Surfaces is one page
    per picture, so it needs only the generation.
    """
    base = f"/directions/{direction}"
    if step.state is StepState.WAITING:
        return None
    on_generation = f"{base}/generations/{generation}" if generation is not None else None
    match step.key:
        case StepKey.START:
            return f"{base}/generate"
        case StepKey.APPROVE | StepKey.KEEP:
            return on_generation
        case StepKey.PICK if step.decomposition is not None:
            return f"{base}/decompositions/{step.decomposition}/pick"
        case StepKey.SURFACES if on_generation is not None:
            return f"{on_generation}/surfaces"
        case StepKey.PICK:
            return on_generation
        case StepKey.REFINE:
            return f"{base}/refine"
        case StepKey.EXPORT:
            return f"{base}/export"
    return None


def frame_context(
    bench: Workbench,
    direction: str,
    generation: str | None,
    here: StepKey | None,
    crumb: str | None = None,
    *,
    timeline: bool = True,
    steps: list[Step] | None = None,
) -> dict[str, object]:
    """The names every framed page passes: `project`, `crumb`, and the timeline.

    `generation` is the scene the page belongs to (None before any picture);
    `here` the step the page is, or None when it is none of them. `reopened`
    is the 1-based position of that step when it is already done, else 0. With
    `timeline` off only `project` and `crumb` are given, for a page whose
    direction has no scene yet. A caller that already read the steps passes them
    as `steps`, so the scene's journey is not worked out twice.
    """
    context: dict[str, object] = {"project": bench.store.load_project(), "crumb": crumb}
    if not timeline:
        return context
    if steps is None:
        steps = journey.scene_steps(bench.store, direction, generation)
    # A page that is a finished step reopens it; the frame dims the later steps.
    reopened = next(
        (n for n, step in enumerate(steps, 1) if step.key is here and step.state is StepState.DONE),
        0,
    )
    context["reopened"] = reopened
    context["timeline"] = [
        {
            "label": step.label,
            "state": step.state.value,
            "href": _href(step, direction, generation),
            "here": step.key is here,
        }
        for step in steps
    ]
    context["timeline_key"] = f"{direction}/{generation or 'new'}"
    return context


def kept_scene_context(
    bench: Workbench, direction: str, here: StepKey | None, crumb: str | None = None
) -> dict[str, object]:
    """`frame_context` for Refine and Export, which follow the scene the person chose.

    `journey.followed_scene` decides which scene that is. With two or more kept
    scenes the context also carries `scene_choices` (one row each: generation,
    asset name, thumbnail, whether it is the one followed) and `scene_return`,
    the page the chooser posts back to; with fewer there is nothing to choose.
    """
    generation = journey.followed_scene(bench.store, direction)
    context = frame_context(
        bench, direction, generation, here, crumb, timeline=generation is not None
    )
    kept = journey.kept_scenes(bench.store, direction)
    if len(kept) > 1 and here in (StepKey.REFINE, StepKey.EXPORT):
        base = f"/directions/{direction}"
        context["scene_return"] = here.value
        context["scene_choices"] = [
            {
                "generation": scene.generation,
                "asset": scene.asset,
                "thumb": f"{base}/generations/{scene.generation}/image.png",
                "current": scene.generation == generation,
            }
            for scene in kept
        ]
    return context


def latest_scene_context(
    bench: Workbench, direction: str, here: StepKey | None, crumb: str | None = None
) -> dict[str, object]:
    """`frame_context` for a page about the whole direction, not one picture.

    The timeline shown is the direction's newest scene; with no scene there is
    no timeline at all.
    """
    generation = journey.latest_scene(bench.store, direction)
    return frame_context(
        bench, direction, generation, here, crumb, timeline=generation is not None
    )
