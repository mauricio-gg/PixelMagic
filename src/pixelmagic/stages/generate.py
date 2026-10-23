"""Build a prompt, draw it, and ask whether it is any good.

This is the loop the whole tool hangs off. A rejection is not a failure: it
becomes a child generation whose prompt carries the feedback, so the record on
disk shows how an approved picture was arrived at.
"""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from pixelmagic.approval import ApprovalDecision, ApprovalHook
from pixelmagic.model import (
    BackgroundPolicy,
    Direction,
    GenerationRecord,
    GenerationRequest,
    GenerationResult,
    GridSpec,
    PromptKind,
    Review,
    Reviewer,
    ReviewStatus,
)
from pixelmagic.naming import check_name
from pixelmagic.providers.base import ImageService, ServiceError, load_template
from pixelmagic.store import Store, claim_id

#: The background sentence pasted into every prompt.
_BACKGROUND_SENTENCES: dict[BackgroundPolicy, str] = {
    BackgroundPolicy.TRANSPARENT: (
        "The background must be fully transparent — save a PNG with a real "
        "alpha channel, not a white or checkerboard background."
    ),
    BackgroundPolicy.MAGENTA: (
        "Put everything on a flat, solid magenta background, hex #ff00ff. "
        "Use that exact colour nowhere else in the picture."
    ),
    BackgroundPolicy.LIME: (
        "Put everything on a flat, solid lime green background, hex #00ff00. "
        "Use that exact colour nowhere else in the picture."
    ),
}

#: The key colour a service without alpha support falls back to.
DEFAULT_KEY = BackgroundPolicy.MAGENTA

#: The largest side, in pixels, PixelMagic will ever ask a service to draw.
#: Pillow refuses to *open* anything over roughly 89 megapixels (its own
#: `Image.MAX_IMAGE_PIXELS` guard), so a picture drawn any bigger than this
#: cannot later be cut up, shown to a reviewer or exported — the failure would
#: surface three steps downstream as Pillow's own `DecompressionBombError`,
#: which is a bare `Exception` and slips past every face's error handling. No
#: real image service draws anywhere near 4096 pixels on a side, so this costs
#: nothing legitimate. The bound lives here, in the core, so the CLI's
#: `--size`, the web form and the MCP tools all share one ceiling instead of
#: three separate opinions — the same argument that put `MAX_SCALE` in
#: `stages/export.py`.
MAX_IMAGE_SIZE = 4096


def check_image_size(size: tuple[int, int]) -> None:
    """Refuse a picture size before anything is drawn, sliced or written.

    Both sides of `size` must be whole numbers from 1 to `MAX_IMAGE_SIZE`.
    Called at the very top of `generate` and `stages.decompose.decompose`, so
    a bad size never gets as far as minting an id, drawing a picture, or
    writing a sidecar that names it.
    """
    for value in size:
        if isinstance(value, bool) or not isinstance(value, int) or not (
            1 <= value <= MAX_IMAGE_SIZE
        ):
            raise ValueError(
                f"size {size} is not usable — each side must be a whole "
                f"number from 1 to {MAX_IMAGE_SIZE}, not {value!r}"
            )


#: The picture formats a person may bring in, by Pillow's own format name.
IMPORTABLE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


def _read_picture(path: Path) -> Image.Image:
    """Open, fully decode and convert a file to RGBA, or refuse it in a sentence.

    Decoding it fully here (not just reading the header) is what catches a
    truncated file before anything is written. Pillow's own errors and its
    decompression-bomb guard are turned into `ValueError`s naming the file, so
    no face has to know Pillow's exception types.
    """
    try:
        with warnings.catch_warnings():
            # A "possible bomb" warning is Pillow's early notice of what would
            # become a hard error later; either way the picture is too big.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                if opened.format not in IMPORTABLE_FORMATS:
                    raise ValueError(
                        f"{path.name} is not a PNG, JPEG or WebP picture "
                        f"(it looks like {opened.format})"
                    )
                # The header's size is checked before decoding, so a huge but
                # under-the-bomb-threshold picture is refused without being
                # decoded into memory first.
                try:
                    check_image_size(opened.size)
                except ValueError as error:
                    raise ValueError(f"{path.name} cannot be brought in: {error}") from error
                opened.load()
                return opened.convert("RGBA")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError(
            f"{path.name} is too large to bring in — each side must be "
            f"at most {MAX_IMAGE_SIZE} pixels"
        ) from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        # A FileNotFoundError is also an OSError; it must stay one.
        if isinstance(error, FileNotFoundError):
            raise
        raise ValueError(f"{path.name} is not a readable PNG, JPEG or WebP picture") from error


def import_picture(
    store: Store,
    direction: Direction,
    path: Path,
    *,
    reviewer: Reviewer,
    now: datetime | None = None,
) -> GenerationRecord:
    """Bring a picture from disk into a direction as an already-approved generation.

    From here the pipeline treats it like any drawn scene: it can be cut up,
    kept, refined and exported. The record says honestly that nothing drew it:
    `request.json` has no prompt, service or background and names the original
    file in `imported_from` (the name only — a path is the person's disk
    layout and goes stale). The picture is stored as `image.png`, converted to
    RGBA, so nothing depends on the original file afterwards.

    Every refusal (a missing file, something that is not a PNG, JPEG or WebP,
    a truncated file, a side over `MAX_IMAGE_SIZE`) happens before an id is
    minted or a folder made, so a refused file leaves nothing in
    `generations/`. `reviewer` is whoever is answering for the face that
    called: HUMAN from the terminal and the web page, AGENT from MCP.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no picture at {path}")
    picture = _read_picture(path)

    moment = now or datetime.now(UTC)
    # claim_id makes the folder as it mints the id, so a second run in the same
    # second cannot take the same one. Every refusal above comes before it.
    generation_id = claim_id(store.generations_dir(direction.name), now=moment)
    folder = store.generation_dir(direction.name, generation_id)
    image_path = folder / "image.png"
    picture.save(image_path, format="PNG")

    request = GenerationRequest(
        id=generation_id,
        created=moment,
        subject=path.stem,
        kind=PromptKind.SCENE,
        size=picture.size,
        imported_from=path.name,
    )
    result = GenerationResult(id=generation_id, created=moment)
    review = Review(
        status=ReviewStatus.APPROVED, reviewer=reviewer, decided=moment
    )
    store.write_model(folder / "request.json", request)
    store.write_model(folder / "result.json", result)
    store.write_model(folder / "review.json", review)
    return GenerationRecord(
        request=request, result=result, review=review, image_path=image_path
    )


class ApprovalExhausted(RuntimeError):
    """Every allowed round was rejected."""

    def __init__(self, record: GenerationRecord, rounds: int) -> None:
        """Carry the last rejected attempt so a face can show it."""
        super().__init__(f"still rejected after {rounds} round(s)")
        self.record = record


def choose_background(service: ImageService, requested: BackgroundPolicy) -> BackgroundPolicy:
    """Honour "transparent first, key colour second" (spec locked decision 6).

    A key colour that was asked for explicitly is always kept; only a
    transparent request is downgraded, and only when the service says it
    cannot produce an alpha channel.
    """
    if requested is not BackgroundPolicy.TRANSPARENT:
        return requested
    if service.capabilities().transparent_background:
        return BackgroundPolicy.TRANSPARENT
    return DEFAULT_KEY


#: The group template's sizing line when the reference is not size-true.
GROUP_SIZING = "- Keep each object at the same size relative to the others as in the first image."


def compose_prompt(
    direction: Direction,
    subject: str,
    kind: PromptKind,
    background: BackgroundPolicy,
    *,
    feedback: str | None = None,
    group: GridSpec | None = None,
    sizing: str | None = None,
) -> str:
    """Expand the kind's template with this direction's style, don'ts and background.

    `group` fills `{count}`, `{cols}` and `{rows}` for `PromptKind.GROUP_REDRAW`;
    the other templates do not use those fields, so it is ignored for them. A
    group redraw without one raises `KeyError` rather than sending a prompt
    with a hole in it. `sizing` replaces the group template's line about keeping
    the objects' relative size (`GROUP_SIZING`) with the lines that ask for the
    sizes of a size-true reference; with none the prompt is unchanged.

    The fully expanded text is what gets stored in `request.json`, so a prompt
    can always be read back exactly as it was sent.
    """
    donts = ""
    if direction.donts:
        donts = "Do not: " + "; ".join(direction.donts) + "."
    prompt = load_template(kind).format(
        style=direction.style.strip(),
        subject=subject.strip(),
        background=_BACKGROUND_SENTENCES[background],
        donts=donts,
        sizing=sizing or GROUP_SIZING,
        **({"count": group.count, "cols": group.cols, "rows": group.rows} if group else {}),
    )
    prompt = prompt.strip() + "\n"
    if feedback:
        prompt += f"\nThe previous attempt was rejected. Fix this: {feedback.strip()}\n"
    return prompt


def generate(
    store: Store,
    direction: Direction,
    subject: str,
    *,
    kind: PromptKind,
    service: ImageService,
    approval: ApprovalHook,
    background: BackgroundPolicy = BackgroundPolicy.TRANSPARENT,
    size: tuple[int, int] = (1024, 1024),
    references: Sequence[str] = (),
    parent: GenerationRecord | None = None,
    feedback: str | None = None,
    group: GridSpec | None = None,
    sizing: str | None = None,
    extra_reference_paths: Sequence[Path] = (),
    generation_id: str | None = None,
    now: datetime | None = None,
) -> GenerationRecord:
    """Draw one image and record the request, the result and the verdict.

    `extra_reference_paths` are files handed to the service before the named
    references, so a group's grid arrives first and the scene second, the
    order the group-redraw prompt describes. They are not recorded in `request.references` and never
    promoted to `references/`, so a one-off picture (a group's reference)
    does not become a default reference for the whole direction.

    `generation_id` is an id the caller minted, whose folder it already made:
    group calls run several at a time, and minting here would race. On a
    `ServiceError` that folder is removed exactly as a folder minted here
    would be.
    """
    check_image_size(size)
    moment = now or datetime.now(UTC)
    effective_background = choose_background(service, background)
    if generation_id is None:
        # claim_id makes the folder as it mints the id, so two runs in the same
        # second cannot share one.
        generation_id = claim_id(store.generations_dir(direction.name), now=moment)
    folder = store.generation_dir(direction.name, generation_id)

    # References only travel when the service can actually take them; sending
    # them to a service that ignores them would make request.json a lie.
    usable_references = list(references) if service.capabilities().reference_images else []
    # The request records names; the service is handed paths. Resolving here
    # rather than in each service keeps `references/` the one place a
    # reference can live, and keeps a service from having to know the store.
    reference_paths = [
        store.references_dir(direction.name) / name for name in usable_references
    ]

    request = GenerationRequest(
        id=generation_id,
        created=moment,
        subject=subject,
        kind=kind,
        prompt=compose_prompt(
            direction, subject, kind, effective_background, feedback=feedback, group=group,
            sizing=sizing,
        ),
        service=service.name,
        model=direction.model,
        size=size,
        background=effective_background,
        references=usable_references,
        parent_id=parent.request.id if parent else None,
        group=group,
    )
    store.write_model(folder / "request.json", request)

    image_path = folder / "image.png"
    try:
        result = service.generate(
            request, image_path, reference_paths=[*extra_reference_paths, *reference_paths]
        )
    except ServiceError:
        # A real service fails often enough for this to matter. Without it the
        # folder stays behind holding a request, no picture and no verdict —
        # an attempt that never happened, listed for ever beside the ones that
        # did, and counted by anything that walks `generations/`. The id was
        # claimed here or by our caller, and nothing outside this function has seen it,
        # so removing the folder cannot orphan a reference; it also takes any
        # half-written image.png with it. The prompt is not kept, which is the
        # price (see the deviations register).
        shutil.rmtree(folder, ignore_errors=True)
        raise
    store.write_model(folder / "result.json", result)

    # `now` is passed on: a test or a replay that injects a clock would
    # otherwise get the real wall clock in review.json alone, and the three
    # sidecars of one generation would disagree about when it happened.
    review = approval.review(image_path, request).to_review(now=moment)
    store.write_model(folder / "review.json", review)

    return GenerationRecord(
        request=request, result=result, review=review, image_path=image_path
    )


def record_decision(
    store: Store,
    direction_name: str,
    record: GenerationRecord,
    decision: ApprovalDecision,
    *,
    force: bool = False,
    now: datetime | None = None,
) -> GenerationRecord:
    """Write a verdict onto a generation that was left waiting.

    The other half of `PendingApproval`: the face that deferred the question
    comes back with the answer, and the answer is written the same way
    `generate` writes one — through the store, into `review.json` — so a
    picture approved in a browser is indistinguishable on disk from one
    approved at the terminal.

    A picture that has already been judged is refused unless `force` is set.
    A stale tab, a back button and a second agent call all arrive here, and
    rejecting a picture that was approved would leave `review.json` saying
    `rejected` while the reference that approval promoted stayed in
    `direction.references` — the sidecar and the folder disagreeing with
    nothing to say which is right. `force` exists because changing your mind
    on purpose is a real thing to want; doing it by accident is not.

    `direction_name` is checked and the generation is required to already be
    on disk because both remaining faces call this with a value read straight
    out of a URL or a tool argument: `store.write_model` creates parent
    folders on demand, so an unchecked, wrong, or stale name would silently
    mint an empty generation folder and return a record claiming a verdict
    that nothing on disk backs up, while the real generation stayed pending.
    """
    check_name(direction_name, "direction")
    if not (store.generation_dir(direction_name, record.request.id) / "request.json").exists():
        raise FileNotFoundError(
            f"no generation {record.request.id} in direction {direction_name!r}"
        )
    if not force and record.review.status is not ReviewStatus.PENDING:
        raise ValueError(
            f"generation {record.request.id} was already "
            f"{record.review.status.value}; pass force to change it"
        )
    review = decision.to_review(now=now)
    store.write_model(
        store.generation_dir(direction_name, record.request.id) / "review.json", review
    )
    return record.model_copy(update={"review": review})


def retry_generation(
    store: Store,
    direction: Direction,
    record: GenerationRecord,
    *,
    service: ImageService,
    approval: ApprovalHook,
    feedback: str | None = None,
    now: datetime | None = None,
) -> GenerationRecord:
    """Draw the next attempt at the same subject, carrying the feedback forward.

    Everything about the request is taken from the attempt being redone, so a
    retry is the same picture asked for again rather than a new request that
    happens to look similar. `feedback` defaults to whatever the reviewer said
    when they turned the last one down.

    Reading the request back rather than keeping the original arguments is
    what makes this usable from a face that has nothing but a generation id:
    `request.background` is the *effective* background, already downgraded if
    the service cannot do transparency, and `choose_background` leaves a key
    colour alone — so feeding it back in is a fixed point. `request.references`
    is likewise the list that actually travelled, already emptied for a service
    that cannot take references. That is why `generate_until_approved` can be
    built on this without changing what it does.
    """
    request = record.request
    if request.imported_from is not None:
        raise ValueError(
            f"generation {request.id} was imported from {request.imported_from}, "
            "not drawn, so there is no prompt to try again from"
        )
    return generate(
        store,
        direction,
        request.subject,
        kind=request.kind,
        service=service,
        approval=approval,
        background=request.background,
        size=request.size,
        references=request.references,
        parent=record,
        feedback=feedback if feedback is not None else record.review.feedback,
        now=now,
    )


def generate_until_approved(
    store: Store,
    direction: Direction,
    subject: str,
    *,
    kind: PromptKind,
    service: ImageService,
    approval: ApprovalHook,
    max_rounds: int = 3,
    background: BackgroundPolicy = BackgroundPolicy.TRANSPARENT,
    size: tuple[int, int] = (1024, 1024),
    references: Sequence[str] = (),
    parent: GenerationRecord | None = None,
) -> GenerationRecord:
    """Generate, and on a rejection try again with the feedback folded in.

    Raises `ApprovalExhausted` — carrying the last attempt — once `max_rounds`
    tries have all been turned down. This is the loop the terminal can afford
    because the person is answering inside it; a face that answers on a later
    request calls `generate` and `retry_generation` itself, one attempt at a
    time, which is why the retry step is its own function.

    `parent` seeds the chain: the first attempt is recorded as that record's
    child and carries its reviewer's words forward. That is how a caller which
    has already spent an attempt of its own leaves a trail — `stages.decompose`
    asks again on magenta when a "transparent" sheet comes back fully opaque,
    and without a seed the replacement and the picture it replaced would sit in
    `generations/` sharing nothing but a subject, so nobody reading the folder
    later could tell why there are two.

    `max_rounds` below one raises `ValueError` before anything is drawn. The
    old loop refused it by accident — `range(0)` ran nothing and an `assert`
    fired — and hoisting the first attempt out of the loop removes that
    accident, so the rule is stated. It lives here and not in a face because
    every face calls this (locked decision 7); the CLI's `--rounds` minimum
    stays where it is.
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be at least 1, not {max_rounds}")
    record = generate(
        store,
        direction,
        subject,
        kind=kind,
        service=service,
        approval=approval,
        background=background,
        size=size,
        references=references,
        parent=parent,
        feedback=parent.review.feedback if parent else None,
    )
    for _ in range(max_rounds - 1):
        if record.review.status is ReviewStatus.APPROVED:
            return record
        record = retry_generation(
            store, direction, record, service=service, approval=approval
        )
    if record.review.status is ReviewStatus.APPROVED:
        return record
    raise ApprovalExhausted(record, max_rounds)


def next_attempt(store: Store, direction_name: str, generation_id: str) -> str | None:
    """The id of the attempt drawn from this one after it was turned down, or None.

    That is the generation whose `parent_id` is this one. A generation whose
    request will not parse is skipped rather than stopping the look.
    """
    for other in store.list_generations(direction_name):
        try:
            request = store.read_model(
                store.generation_dir(direction_name, other) / "request.json", GenerationRequest
            )
        except (OSError, ValueError):
            continue
        if request.parent_id == generation_id:
            return other
    return None


def load_generation(
    store: Store, direction_name: str, generation_id: str
) -> GenerationRecord:
    """Read one generation's three sidecars back off disk."""
    folder = store.generation_dir(direction_name, generation_id)
    return GenerationRecord(
        request=store.read_model(folder / "request.json", GenerationRequest),
        result=store.read_model(folder / "result.json", GenerationResult),
        review=store.read_model(folder / "review.json", Review),
        image_path=folder / "image.png",
    )


class NotApproved(ValueError):
    """A picture nobody has said yes to yet was asked to be kept or cut up.

    A `ValueError`, so a face that catches that family at its boundary still
    catches this — the same pattern `naming.UnusableName` follows. A class of
    its own rather than a bare `ValueError` is what let `cli.py`'s `decompose`
    command stop routing this refusal through `_refiner_failed`, a helper
    named for a failure that was never involved here.

    Lives here, beside `keep_reference`, because `decompose` imports this module
    and not the other way round; `stages.decompose` still re-exports the name.
    """


def reference_name(record: GenerationRecord) -> str:
    """The file name an approved generation gets inside `references/`.

    `GenerationRequest.references` holds names, not paths, so anything that
    wants to condition a request on an earlier picture promotes it first and
    then passes this name.
    """
    return f"{record.request.id}.png"


def promote(
    store: Store,
    direction: Direction,
    record: GenerationRecord,
    *,
    as_reference: bool = True,
    as_prompt: bool = True,
) -> Direction:
    """Keep an approved image as a style reference and its prompt as a known-good one.

    This is how a direction gets better over time: the pictures that worked
    become the references the next request is conditioned on. Safe to call
    twice on the same record — the reference list holds each name once.
    """
    if as_reference:
        name = reference_name(record)
        target = store.references_dir(direction.name) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(record.image_path).read_bytes())
        if name not in direction.references:
            direction.references.append(name)
    if (
        as_prompt
        and record.request.prompt is not None
        and record.request.prompt not in direction.approved_prompts
    ):
        direction.approved_prompts.append(record.request.prompt)
    return store.save_direction(direction)


def keep_reference(store: Store, direction: Direction, record: GenerationRecord) -> bool:
    """Keep an already-approved picture, imported or drawn, as a style reference.

    This is the "use it as the style" step for a picture that was approved
    earlier, such as an imported one, so it did not pass through `--keep`. It
    reuses `promote` for the picture only: an import has no prompt, and this
    step is not about prompts. `NotApproved` comes first, before anything is
    written. Returns True when the picture was newly added to the direction's
    references, False when it already was (nothing is rewritten then).
    """
    if record.review.status is not ReviewStatus.APPROVED:
        raise NotApproved(
            f"generation {record.request.id} is {record.review.status.value}; "
            "approve it before keeping it as a style reference"
        )
    if reference_name(record) in direction.references:
        return False
    promote(store, direction, record, as_reference=True, as_prompt=False)
    return True
