from datetime import UTC, datetime

import pytest
from PIL import Image

from pixelmagic.approval import ApprovalDecision, AutoApprove, CallableApproval, PendingApproval
from pixelmagic.model import (
    BackgroundPolicy,
    Direction,
    PromptKind,
    Reviewer,
    ReviewStatus,
)
from pixelmagic.providers.base import ImageCapabilities, ServiceError
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.stages.generate import (
    MAX_IMAGE_SIZE,
    ApprovalExhausted,
    NotApproved,
    check_image_size,
    choose_background,
    compose_prompt,
    generate,
    generate_until_approved,
    import_picture,
    keep_reference,
    load_generation,
    next_attempt,
    promote,
    record_decision,
    reference_name,
    retry_generation,
)
from pixelmagic.store import Store

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(
            name="cosy-den",
            created=NOW,
            style="Warm SNES interiors, soft lamp light.",
            donts=["no gradients", "no text"],
        )
    )
    return store, direction


def test_compose_prompt_folds_in_style_subject_and_donts(setup):
    _, direction = setup
    prompt = compose_prompt(
        direction, "a wooden chair", PromptKind.PROP, BackgroundPolicy.TRANSPARENT
    )
    assert "Warm SNES interiors" in prompt
    assert "a wooden chair" in prompt
    assert "no gradients" in prompt
    assert "transparent" in prompt.lower()


def test_compose_prompt_names_the_key_colour(setup):
    _, direction = setup
    prompt = compose_prompt(direction, "a chair", PromptKind.PROP, BackgroundPolicy.MAGENTA)
    assert "#ff00ff" in prompt.lower()


def test_compose_prompt_appends_feedback(setup):
    _, direction = setup
    prompt = compose_prompt(
        direction, "a chair", PromptKind.PROP, BackgroundPolicy.TRANSPARENT,
        feedback="the chair is facing away",
    )
    assert "the chair is facing away" in prompt


def test_generate_writes_the_image_and_all_three_sidecars(setup):
    store, direction = setup
    record = generate(
        store, direction, "a wooden chair",
        kind=PromptKind.PROP, service=FakeImageService(), approval=AutoApprove(),
        size=(64, 64), now=NOW,
    )
    folder = store.generation_dir("cosy-den", record.request.id)
    assert (folder / "image.png").exists()
    assert (folder / "request.json").exists()
    assert (folder / "result.json").exists()
    assert (folder / "review.json").exists()
    assert record.review.status is ReviewStatus.APPROVED


@pytest.mark.parametrize(
    "size", [(0, 1024), (1024, 0), (MAX_IMAGE_SIZE + 1, 1024), (1024, MAX_IMAGE_SIZE + 1)]
)
def test_generate_refuses_a_size_out_of_range_and_writes_nothing(setup, size):
    """Guardian M2.2 #1: Pillow cannot even open a picture much bigger than
    this, so the ceiling has to stop the request before anything is drawn —
    not fail three steps downstream as a bare `DecompressionBombError`."""
    store, direction = setup
    with pytest.raises(ValueError, match=str(MAX_IMAGE_SIZE)):
        generate(
            store, direction, "a wooden chair",
            kind=PromptKind.PROP, service=FakeImageService(), approval=AutoApprove(),
            size=size, now=NOW,
        )
    assert store.list_generations("cosy-den") == []


def test_check_image_size_accepts_the_floor_and_the_ceiling():
    check_image_size((1, 1))
    check_image_size((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))


def test_an_injected_clock_reaches_the_review_too(setup):
    """All three sidecars of one generation must agree about when it happened."""
    store, direction = setup
    record = generate(
        store, direction, "a wooden chair",
        kind=PromptKind.PROP, service=FakeImageService(), approval=AutoApprove(),
        size=(64, 64), now=NOW,
    )
    assert record.request.created == NOW
    assert record.review.decided == NOW
    assert load_generation(store, "cosy-den", record.request.id).review.decided == NOW


def test_a_generation_round_trips_off_disk(setup):
    store, direction = setup
    record = generate(
        store, direction, "a wooden chair",
        kind=PromptKind.PROP, service=FakeImageService(), approval=AutoApprove(),
        size=(64, 64), now=NOW,
    )
    reloaded = load_generation(store, "cosy-den", record.request.id)
    assert reloaded.request.prompt == record.request.prompt
    assert reloaded.review.status is ReviewStatus.APPROVED


def test_a_rejection_records_the_feedback(setup):
    store, direction = setup
    hook = CallableApproval(lambda path, request: (False, "too dark"))
    record = generate(
        store, direction, "a wooden chair",
        kind=PromptKind.PROP, service=FakeImageService(), approval=hook, size=(64, 64), now=NOW,
    )
    assert record.review.status is ReviewStatus.REJECTED
    assert record.review.feedback == "too dark"


def test_a_retry_becomes_a_child_generation_carrying_the_feedback(setup):
    store, direction = setup
    verdicts = iter([(False, "too dark"), (True, None)])
    hook = CallableApproval(lambda path, request: next(verdicts))
    record = generate_until_approved(
        store, direction, "a wooden chair",
        kind=PromptKind.PROP, service=FakeImageService(), approval=hook,
        size=(64, 64), max_rounds=3,
    )
    assert record.review.status is ReviewStatus.APPROVED
    assert record.request.parent_id is not None
    assert "too dark" in record.request.prompt


def test_running_out_of_rounds_raises_with_the_last_attempt(setup):
    store, direction = setup
    hook = CallableApproval(lambda path, request: (False, "still wrong"))
    with pytest.raises(ApprovalExhausted) as caught:
        generate_until_approved(
            store, direction, "a wooden chair",
            kind=PromptKind.PROP, service=FakeImageService(), approval=hook,
            size=(64, 64), max_rounds=2,
        )
    assert caught.value.record.review.feedback == "still wrong"


def test_promote_adds_a_reference_and_an_approved_prompt(setup):
    store, direction = setup
    record = generate(
        store, direction, "a wooden chair",
        kind=PromptKind.PROP, service=FakeImageService(), approval=AutoApprove(),
        size=(64, 64), now=NOW,
    )
    updated = promote(store, direction, record)
    assert updated.references == [f"{record.request.id}.png"]
    assert updated.approved_prompts == [record.request.prompt]
    assert (store.references_dir("cosy-den") / f"{record.request.id}.png").exists()
    assert store.load_direction("cosy-den").references == updated.references


def test_promote_twice_keeps_one_reference(setup):
    store, direction = setup
    record = generate(
        store, direction, "a wooden chair",
        kind=PromptKind.PROP, service=FakeImageService(), approval=AutoApprove(),
        size=(64, 64), now=NOW,
    )
    promote(store, direction, record)
    updated = promote(store, direction, record)
    assert updated.references == [reference_name(record)]
    assert len(updated.approved_prompts) == 1


def test_an_imported_picture_kept_as_a_reference_conditions_the_next_generate(setup, tmp_path):
    """catches: an import that never becomes a reference, so later requests ignore its style."""
    store, direction = setup
    Image.new("RGB", (64, 48), (90, 60, 40)).save(tmp_path / "den.png")
    imported = import_picture(store, direction, tmp_path / "den.png", reviewer=Reviewer.AGENT)
    assert keep_reference(store, direction, imported) is True
    name = f"{imported.request.id}.png"
    assert store.load_direction("cosy-den").references == [name]
    kept = store.references_dir("cosy-den") / name
    assert kept.read_bytes() == imported.image_path.read_bytes()
    assert keep_reference(store, direction, imported) is False
    # Like a face: read the direction back from disk and pass its references.
    reloaded = store.load_direction("cosy-den")
    later = generate(
        store, reloaded, "a rainy yard", kind=PromptKind.SCENE,
        service=FakeImageService(), approval=AutoApprove(),
        references=reloaded.references, size=(96, 96), now=NOW,
    )
    assert later.request.references == [name]


def test_keeping_an_unapproved_picture_as_a_reference_is_refused_and_writes_nothing(setup):
    """catches: a picture nobody approved becoming a style reference."""
    store, direction = setup
    pending = generate(
        store, direction, "a rainy yard", kind=PromptKind.SCENE,
        service=FakeImageService(), approval=PendingApproval(), size=(96, 96), now=NOW,
    )
    with pytest.raises(NotApproved):
        keep_reference(store, direction, pending)
    assert store.load_direction("cosy-den").references == []
    assert not (store.references_dir("cosy-den") / f"{pending.request.id}.png").exists()


def test_a_service_without_alpha_falls_back_to_magenta():
    service = FakeImageService(capabilities=ImageCapabilities(transparent_background=False))
    assert choose_background(service, BackgroundPolicy.TRANSPARENT) is BackgroundPolicy.MAGENTA
    assert choose_background(service, BackgroundPolicy.LIME) is BackgroundPolicy.LIME


def test_a_service_with_alpha_keeps_transparent():
    service = FakeImageService()
    assert choose_background(service, BackgroundPolicy.TRANSPARENT) is BackgroundPolicy.TRANSPARENT


def plain_service():
    """A service that can do neither transparency nor references."""
    return FakeImageService(
        capabilities=ImageCapabilities(
            transparent_background=False,
            reference_images=False,
            edit=False,
            vision=True,
            sizes=[(64, 64)],
        )
    )


def waiting(tmp_path, *, service=None, references=()):
    """A direction with one generation nobody has judged yet."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    record = generate(
        store, direction, "a cosy den",
        kind=PromptKind.SCENE, service=service or FakeImageService(),
        approval=PendingApproval(), size=(64, 64), references=references,
    )
    return store, direction, record


def test_a_deferred_verdict_is_written_to_disk_as_pending(tmp_path):
    store, direction, record = waiting(tmp_path)
    assert record.review.status is ReviewStatus.PENDING
    reloaded = load_generation(store, "cosy-den", record.request.id)
    assert reloaded.review.status is ReviewStatus.PENDING
    assert reloaded.review.decided is None
    assert record.image_path.exists()


def test_recording_an_approval_later_updates_the_sidecar(tmp_path):
    store, direction, record = waiting(tmp_path)
    decided = record_decision(
        store, "cosy-den", record,
        ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.HUMAN),
        now=NOW,
    )
    assert decided.review.status is ReviewStatus.APPROVED
    assert load_generation(store, "cosy-den", record.request.id).review.status is (
        ReviewStatus.APPROVED
    )
    assert load_generation(store, "cosy-den", record.request.id).review.decided == NOW


def test_recording_a_rejection_keeps_the_words_that_were_said(tmp_path):
    store, direction, record = waiting(tmp_path)
    record_decision(
        store, "cosy-den", record,
        ApprovalDecision(
            status=ReviewStatus.REJECTED, reviewer=Reviewer.HUMAN, feedback="too dark"
        ),
    )
    assert load_generation(store, "cosy-den", record.request.id).review.feedback == "too dark"


def test_a_verdict_that_was_already_given_is_not_quietly_replaced(tmp_path):
    """A stale tab, a back button, or a second agent call all land here."""
    store, direction, record = waiting(tmp_path)
    approved = record_decision(
        store, "cosy-den", record,
        ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.HUMAN),
    )
    with pytest.raises(ValueError, match="already"):
        record_decision(
            store, "cosy-den", approved,
            ApprovalDecision(status=ReviewStatus.REJECTED, reviewer=Reviewer.HUMAN),
        )
    assert load_generation(store, "cosy-den", record.request.id).review.status is (
        ReviewStatus.APPROVED
    )


def test_a_verdict_can_be_changed_on_purpose(tmp_path):
    store, direction, record = waiting(tmp_path)
    approved = record_decision(
        store, "cosy-den", record,
        ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.HUMAN),
    )
    changed = record_decision(
        store, "cosy-den", approved,
        ApprovalDecision(status=ReviewStatus.REJECTED, reviewer=Reviewer.HUMAN),
        force=True,
    )
    assert changed.review.status is ReviewStatus.REJECTED


def test_recording_a_decision_on_a_direction_that_does_not_exist_creates_nothing(tmp_path):
    """Both faces still to come call this with a direction name read out of a
    URL or a tool argument. A wrong one must not mint an orphan folder that
    says "approved" while the real generation stays pending.
    """
    store, direction, record = waiting(tmp_path)
    with pytest.raises(FileNotFoundError):
        record_decision(
            store, "other-den", record,
            ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.HUMAN),
        )
    assert not store.direction_dir("other-den").exists()
    assert load_generation(store, "cosy-den", record.request.id).review.status is (
        ReviewStatus.PENDING
    )


def test_recording_a_decision_refuses_a_name_that_could_leave_the_project(tmp_path):
    """The same `../escape` a face could pass to `rename_direction` must be
    refused here too, before it ever reaches a path.
    """
    store, direction, record = waiting(tmp_path)
    with pytest.raises(ValueError):
        record_decision(
            store, "../escape", record,
            ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.HUMAN),
        )
    assert not (tmp_path.parent / "escape").exists()


def test_a_retry_is_a_child_of_the_attempt_it_redoes(tmp_path):
    store, direction, record = waiting(tmp_path)
    rejected = record_decision(
        store, "cosy-den", record,
        ApprovalDecision(
            status=ReviewStatus.REJECTED, reviewer=Reviewer.HUMAN, feedback="too dark"
        ),
    )
    child = retry_generation(
        store, direction, rejected,
        service=FakeImageService(), approval=PendingApproval(),
    )
    assert child.request.parent_id == record.request.id
    assert child.request.subject == "a cosy den"
    assert child.request.kind is PromptKind.SCENE
    assert child.request.size == (64, 64)
    assert "too dark" in child.request.prompt
    assert child.review.status is ReviewStatus.PENDING


def test_next_attempt_finds_the_redraw_of_a_turned_down_picture(tmp_path):
    """"See the redraw" must open the child, and a picture with none opens nothing."""
    store, direction, record = waiting(tmp_path)
    assert next_attempt(store, direction.name, record.request.id) is None
    child = retry_generation(
        store, direction, record,
        service=FakeImageService(), approval=PendingApproval(),
    )
    assert next_attempt(store, direction.name, record.request.id) == child.request.id
    assert next_attempt(store, direction.name, child.request.id) is None


def test_a_retry_can_be_given_different_words_than_the_ones_on_the_record(tmp_path):
    store, direction, record = waiting(tmp_path)
    child = retry_generation(
        store, direction, record,
        service=FakeImageService(), approval=PendingApproval(),
        feedback="make the lamp bigger",
    )
    assert "make the lamp bigger" in child.request.prompt


def test_a_retry_with_nothing_to_say_leaves_the_prompt_alone(tmp_path):
    store, direction, record = waiting(tmp_path)
    child = retry_generation(
        store, direction, record,
        service=FakeImageService(), approval=PendingApproval(),
    )
    assert "was rejected" not in child.request.prompt


def test_a_retry_asks_for_the_background_the_first_attempt_actually_got(tmp_path):
    """The fixed point the rebuilt loop rests on, pinned rather than argued.

    A service that cannot do transparency has its request downgraded to a key
    colour on the way in, and `retry_generation` reads the *effective*
    background back off the record — so the child asks for magenta again,
    rather than asking for transparency and being downgraded a second time.
    """
    store, direction, record = waiting(tmp_path, service=plain_service())
    assert record.request.background is BackgroundPolicy.MAGENTA
    child = retry_generation(
        store, direction, record,
        service=plain_service(), approval=PendingApproval(),
    )
    assert child.request.background is BackgroundPolicy.MAGENTA


def test_a_retry_carries_the_references_that_actually_travelled(tmp_path):
    """The other half of the fixed point: a service that takes no references had
    them emptied on the way in, and the retry must not put them back."""
    store, direction, record = waiting(
        tmp_path, service=plain_service(), references=["a-lamp.png"]
    )
    assert record.request.references == []
    child = retry_generation(
        store, direction, record,
        service=plain_service(), approval=PendingApproval(),
    )
    assert child.request.references == []


def test_a_seeded_parent_becomes_the_first_attempts_parent(setup):
    store, direction = setup
    first = generate(
        store, direction, "a chair", kind=PromptKind.PROP,
        service=FakeImageService(), approval=AutoApprove(), now=NOW,
    )
    second = generate_until_approved(
        store, direction, "a chair", kind=PromptKind.PROP,
        service=FakeImageService(), approval=AutoApprove(), parent=first,
    )
    assert second.request.parent_id == first.request.id


def test_a_seeded_parent_hands_its_feedback_to_the_first_attempt(setup):
    """The words that turned an attempt down travel to the one replacing it.

    This is the whole point of seeding a parent rather than just recording an
    id, and it is also the one place the carry can read oddly — see the
    deviations register.
    """
    store, direction = setup
    first = generate(
        store, direction, "a chair", kind=PromptKind.PROP,
        service=FakeImageService(),
        approval=CallableApproval(lambda path, request: (False, "too dark")),
        now=NOW,
    )
    second = generate_until_approved(
        store, direction, "a chair", kind=PromptKind.PROP,
        service=FakeImageService(), approval=AutoApprove(), parent=first,
    )
    assert "too dark" in second.request.prompt


def test_max_rounds_below_one_is_refused(tmp_path):
    """Not an assert, and not a picture nobody asked for.

    `main` reached the right outcome by accident: `range(0)` ran nothing and an
    `assert record is not None` fired. Hoisting the first attempt out of the
    loop removes that accident, so the rule is written down instead — and it is
    written down here rather than in a face because every face calls this
    (locked decision 7), and once a real service is registered a zero would
    spend a subscription turn before raising.
    """
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    with pytest.raises(ValueError, match="at least 1"):
        generate_until_approved(
            store, direction, "a cosy den", kind=PromptKind.SCENE,
            service=FakeImageService(), approval=AutoApprove(), max_rounds=0,
        )
    assert store.list_generations("cosy-den") == []


class RefusingService(FakeImageService):
    """A fake that refuses the way a real service refuses: one typed error."""

    name = "refusing"

    def generate(self, request, out_path, *, reference_paths=()):
        """Fail before writing anything, as a filtered prompt does."""
        raise ServiceError(
            "the content filter would not draw that",
            service=self.name,
            code="moderation_blocked",
        )


class HalfWritingService(FakeImageService):
    """A fake that writes a broken file and only then gives up."""

    name = "half-writing"

    def generate(self, request, out_path, *, reference_paths=()):
        """Leave a truncated PNG behind, then fail."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x89PNG\r\n\x1a\n truncated")
        raise ServiceError("the connection dropped", service=self.name)


def test_a_service_that_refuses_leaves_no_generation_behind(setup):
    store, direction = setup
    with pytest.raises(ServiceError) as caught:
        generate(
            store, direction, "a chair", kind=PromptKind.PROP,
            service=RefusingService(), approval=AutoApprove(), now=NOW,
        )
    assert caught.value.service == "refusing"
    assert caught.value.code == "moderation_blocked"
    assert list(store.generations_dir("cosy-den").iterdir()) == []


def test_a_half_written_image_is_cleaned_up_too(setup):
    store, direction = setup
    with pytest.raises(ServiceError):
        generate(
            store, direction, "a chair", kind=PromptKind.PROP,
            service=HalfWritingService(), approval=AutoApprove(), now=NOW,
        )
    assert list(store.generations_dir("cosy-den").iterdir()) == []


def test_references_reach_the_service_as_real_paths(setup):
    store, direction = setup
    service = FakeImageService()
    record = generate(
        store, direction, "a chair", kind=PromptKind.PROP, service=service,
        approval=AutoApprove(), references=["20260903-191233.png"], now=NOW,
    )
    # The request still holds names, so it can be replayed from the folder …
    assert record.request.references == ["20260903-191233.png"]
    # … and the service still got somewhere it can actually open.
    assert service.reference_paths_seen == [
        [store.references_dir("cosy-den") / "20260903-191233.png"]
    ]


def test_a_service_that_cannot_take_references_is_sent_none(setup):
    store, direction = setup
    service = FakeImageService(
        capabilities=ImageCapabilities(
            transparent_background=True, reference_images=False, vision=True
        )
    )
    record = generate(
        store, direction, "a chair", kind=PromptKind.PROP, service=service,
        approval=AutoApprove(), references=["20260903-191233.png"], now=NOW,
    )
    # Both halves matter: request.json must not claim references that were
    # never sent, and the service must not be handed paths it would ignore.
    assert record.request.references == []
    assert service.reference_paths_seen == [[]]
