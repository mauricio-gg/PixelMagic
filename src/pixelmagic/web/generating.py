"""The generate loop, one attempt per click.

The command line can afford `generate_until_approved` because the person is at
the terminal answering inside the loop. A browser cannot: that loop would have
to hold an HTTP request open waiting for a click that can only arrive on a
different request. So this screen runs exactly one attempt, records it as
pending, and shows it; approving or rejecting is the next request, and a
rejection draws the next attempt. The chain of parents and feedback on disk is
the same one the command line leaves behind — spread over one request each,
with the sidecars as the only state. No background threads, no queue, no
session.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response

from pixelmagic.approval import ApprovalDecision, PendingApproval
from pixelmagic.journey import StepKey, props_runs, scene_steps
from pixelmagic.model import (
    BackgroundPolicy,
    Direction,
    GenerateKind,
    GenerationRecord,
    PromptKind,
    Reviewer,
    ReviewStatus,
)
from pixelmagic.services import service_status
from pixelmagic.stages.decompose import keep_scene, rename_scene, scene_asset_name, scene_holding
from pixelmagic.stages.generate import (
    MAX_IMAGE_SIZE,
    generate,
    import_picture,
    load_generation,
    next_attempt,
    promote,
    record_decision,
    reference_name,
    retry_generation,
)
from pixelmagic.stages.generate import (
    keep_reference as keep_reference_stage,
)
from pixelmagic.store import check_run_id
from pixelmagic.web.deps import Workbench, page, require_direction, see_other, workbench
from pixelmagic.web.timeline import frame_context

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]


@router.get("/directions/{direction}/generate")
def form(request: Request, direction: str, bench: Bench) -> Response:
    """Ask what to draw."""
    found = require_direction(bench, direction)
    return page(
        request,
        "start.html",
        direction=found,
        **frame_context(bench, found.name, None, StepKey.START, "Start"),
        kinds=list(GenerateKind),
        backgrounds=list(BackgroundPolicy),
        service=bench.service_name,
        # Asked per request: a key saved a moment ago on the Services page counts.
        readiness=service_status(bench.service_name) if bench.service_name else None,
    )


@router.post("/directions/{direction}/generate")
def start(
    direction: str,
    bench: Bench,
    subject: Annotated[str, Form()],
    kind: Annotated[GenerateKind, Form()] = GenerateKind.SCENE,
    background: Annotated[BackgroundPolicy, Form()] = BackgroundPolicy.TRANSPARENT,
    size: Annotated[int, Form(ge=1, le=MAX_IMAGE_SIZE)] = 1024,
) -> Response:
    """Draw one attempt and go and look at it."""
    found = require_direction(bench, direction)
    record = generate(
        bench.store,
        found,
        subject,
        kind=PromptKind(kind.value),
        service=bench.service(),
        approval=PendingApproval(),
        background=background,
        size=(size, size),
        references=list(found.references),
    )
    return see_other(f"/directions/{found.name}/generations/{record.request.id}")


@router.post("/directions/{direction}/import")
def bring_in(
    direction: str, bench: Bench, picture: Annotated[UploadFile, File()]
) -> Response:
    """Bring an uploaded picture in as an approved scene and go and look at it.

    The upload is copied under its own file name (the name only: a client can
    send a path there) into a folder that is deleted afterwards, worked or
    not, because `import_picture` stores its own copy as `image.png`.
    """
    found = require_direction(bench, direction)
    name = Path(picture.filename or "").name
    if not name:
        raise ValueError("no picture was chosen")
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / name
        copy.write_bytes(picture.file.read())
        record = import_picture(bench.store, found, copy, reviewer=Reviewer.HUMAN)
    return see_other(f"/directions/{found.name}/generations/{record.request.id}")


def generation_view(
    bench: Workbench, found: Direction, record: GenerationRecord
) -> dict[str, object]:
    """What `picture.html` needs to draw one attempt.

    A function of its own, so a route that re-shows this page builds exactly
    the context `show` does.
    """
    approved = record.review.status is ReviewStatus.APPROVED
    rejected = record.review.status is ReviewStatus.REJECTED
    kept = scene_holding(bench.store, found.name, record.request.id)
    steps = scene_steps(bench.store, found.name, record.request.id)
    runs = props_runs(bench.store, found.name, record.request.id)
    runs_base = f"/directions/{found.name}/decompositions"
    return {
        "direction": found,
        "record": record,
        "pending": record.review.status is ReviewStatus.PENDING,
        "approved": approved,
        "scene_asset": kept,
        "proposed_name": (
            scene_asset_name(bench.store, found, record) if approved and kept is None else None
        ),
        "is_reference": reference_name(record) in found.references,
        # The paid road (suggestions) and the free one (pixel cut) each keep their own newest run.
        "suggest_href": f"{runs_base}/{runs.suggestions}/pick" if runs.suggestions else None,
        "pixel_cut_href": f"{runs_base}/{runs.pixel_cut}/pick" if runs.pixel_cut else None,
        "service": bench.service_name,
        "next_attempt": (
            next_attempt(bench.store, found.name, record.request.id) if rejected else None
        ),
        **frame_context(
            bench,
            found.name,
            record.request.id,
            # Once the scene is kept nothing on this page can be edited, so no
            # step is "here" and the timeline does not show a reopened step.
            None if kept is not None else StepKey.KEEP if approved else StepKey.APPROVE,
            "Picture",
            steps=steps,
        ),
    }


@router.get("/directions/{direction}/generations/{generation}")
def show(request: Request, direction: str, generation: str, bench: Bench) -> Response:
    """One attempt: the picture, what was asked for, and the verdict form."""
    found = require_direction(bench, direction)
    record = load_generation(
        bench.store, found.name, check_run_id(generation, "generation id")
    )
    return page(request, "picture.html", **generation_view(bench, found, record))


@router.post("/directions/{direction}/generations/{generation}/approve")
def approve(
    direction: str,
    generation: str,
    bench: Bench,
    keep_reference: Annotated[bool, Form()] = False,
) -> Response:
    """Say yes, and keep the picture as a style reference if the form asked.

    The reference box is ticked on the page, and the MCP tool's
    `keep_as_reference` defaults to `True` for the same reason: keeping the
    picture somebody just approved as the style reference is how an art
    direction settles. Keeping the whole picture as the scene is its own
    step afterwards (the keep-scene route). The parameter is `False` here so
    that a caller who posts nothing keeps nothing, which is the path
    `test_approving_without_ticking_anything_keeps_nothing` pins; the page's
    own default is the ticked box, not this signature.
    """
    found = require_direction(bench, direction)
    record = load_generation(
        bench.store, found.name, check_run_id(generation, "generation id")
    )
    record = record_decision(
        bench.store,
        found.name,
        record,
        ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.HUMAN),
    )
    if keep_reference:
        promote(bench.store, found, record)
    return see_other(f"/directions/{found.name}/generations/{record.request.id}")


@router.post("/directions/{direction}/generations/{generation}/keep-reference")
def keep_as_reference(direction: str, generation: str, bench: Bench) -> Response:
    """Keep an approved picture as a style reference, then show it again.

    The refusal (not approved) is the core's; `app.py` turns it into a page.
    """
    found = require_direction(bench, direction)
    record = load_generation(
        bench.store, found.name, check_run_id(generation, "generation id")
    )
    keep_reference_stage(bench.store, found, record)
    return see_other(f"/directions/{found.name}/generations/{record.request.id}")


@router.post("/directions/{direction}/generations/{generation}/keep-scene")
def keep_as_scene(
    direction: str,
    generation: str,
    bench: Bench,
    name: Annotated[str, Form()] = "",
) -> Response:
    """Keep an approved picture as the scene, then show the picture again.

    The typed `name` goes to the core as it is; an empty box means the
    suggested name. The refusals (not approved, already kept, a name that is
    unsafe or taken) and a refiner failure are the core's; `app.py` turns each
    into a page.
    """
    found = require_direction(bench, direction)
    record = load_generation(
        bench.store, found.name, check_run_id(generation, "generation id")
    )
    keep_scene(bench.store, found, record, refiner=bench.refiner, name=name or None)
    return see_other(f"/directions/{found.name}/generations/{record.request.id}")


@router.post("/directions/{direction}/generations/{generation}/rename-scene")
def rename_kept_scene(
    direction: str, generation: str, bench: Bench, name: Annotated[str, Form()]
) -> Response:
    """Rename the scene kept from this picture, then show the picture again.

    The refusals (nothing kept yet, a name that is unsafe or taken) are the
    core's; `app.py` turns each into a page.
    """
    found = require_direction(bench, direction)
    record = load_generation(
        bench.store, found.name, check_run_id(generation, "generation id")
    )
    kept = scene_holding(bench.store, found.name, record.request.id)
    if kept is None:
        raise ValueError(f"generation {record.request.id} is not kept as the scene yet")
    rename_scene(bench.store, found, kept, name)
    return see_other(f"/directions/{found.name}/generations/{record.request.id}")


@router.post("/directions/{direction}/generations/{generation}/reject")
def reject(
    direction: str,
    generation: str,
    bench: Bench,
    feedback: Annotated[str, Form()] = "",
) -> Response:
    """Say no, say why, and draw the next attempt.

    The rejection is written onto the picture that was turned down and the
    words go into the next prompt, so the folder ends up holding the same
    chain the command line's loop leaves behind. Rejecting is what starts the
    next attempt, rather than a separate "try again" button: a rejection with
    no follow-up would leave the person on a dead page.
    """
    found = require_direction(bench, direction)
    record = load_generation(
        bench.store, found.name, check_run_id(generation, "generation id")
    )
    # Built before the rejection is saved: with no service chosen this refuses,
    # and a saved rejection with no follow-up would strand the picture.
    service = bench.service()
    record = record_decision(
        bench.store,
        found.name,
        record,
        ApprovalDecision(
            status=ReviewStatus.REJECTED,
            reviewer=Reviewer.HUMAN,
            feedback=feedback.strip() or None,
        ),
    )
    nxt = retry_generation(
        bench.store, found, record, service=service, approval=PendingApproval()
    )
    return see_other(f"/directions/{found.name}/generations/{nxt.request.id}")
