from datetime import UTC, datetime
from pathlib import Path

from pixelmagic.approval import (
    ApprovalDecision,
    AutoApprove,
    CallableApproval,
    PendingApproval,
    TerminalApproval,
)
from pixelmagic.model import (
    BackgroundPolicy,
    GenerationRequest,
    PromptKind,
    Reviewer,
    ReviewStatus,
)

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)
REQUEST = GenerationRequest(
    id="20260903-191233",
    created=NOW,
    subject="a wooden chair",
    kind=PromptKind.PROP,
    prompt="draw a wooden chair",
    service="fake",
    size=(64, 64),
    background=BackgroundPolicy.TRANSPARENT,
)
IMAGE = Path("/tmp/image.png")


def test_auto_approve_approves_as_the_auto_reviewer():
    decision = AutoApprove().review(IMAGE, REQUEST)
    assert decision.approved is True
    assert decision.reviewer is Reviewer.AUTO
    assert decision.feedback is None


def test_a_decision_converts_to_a_review():
    review = AutoApprove().review(IMAGE, REQUEST).to_review(now=NOW)
    assert review.status is ReviewStatus.APPROVED
    assert review.reviewer is Reviewer.AUTO
    assert review.decided == NOW


def test_callable_approval_sees_the_image_and_the_request():
    seen: list[tuple[Path, str]] = []

    def decide(path, request):
        seen.append((path, request.subject))
        return True

    assert CallableApproval(decide).review(IMAGE, REQUEST).approved is True
    assert seen == [(IMAGE, "a wooden chair")]


def test_callable_approval_carries_feedback_from_a_pair():
    decision = CallableApproval(lambda path, request: (False, "too dark")).review(IMAGE, REQUEST)
    assert decision.approved is False
    assert decision.feedback == "too dark"
    assert decision.reviewer is Reviewer.AGENT
    assert decision.to_review(now=NOW).status is ReviewStatus.REJECTED


def test_terminal_approval_accepts_a_yes():
    hook = TerminalApproval(prompt=lambda text: "y", out=lambda text: None)
    decision = hook.review(IMAGE, REQUEST)
    assert decision.approved is True
    assert decision.reviewer is Reviewer.HUMAN


def test_terminal_approval_asks_for_feedback_after_a_no():
    answers = iter(["n", "the chair is facing away"])
    hook = TerminalApproval(prompt=lambda text: next(answers), out=lambda text: None)
    decision = hook.review(IMAGE, REQUEST)
    assert decision.approved is False
    assert decision.feedback == "the chair is facing away"


def test_terminal_approval_shows_the_image_path():
    shown: list[str] = []
    hook = TerminalApproval(prompt=lambda text: "y", out=shown.append)
    hook.review(IMAGE, REQUEST)
    assert any(str(IMAGE) in line for line in shown)


def test_a_pending_verdict_is_neither_a_yes_nor_a_no():
    decision = PendingApproval().review(IMAGE, REQUEST)
    assert decision.status is ReviewStatus.PENDING
    assert decision.approved is False
    assert decision.reviewer is Reviewer.HUMAN


def test_a_pending_verdict_records_no_moment_of_deciding():
    """Stamping a time would say somebody decided when nobody had."""
    review = PendingApproval().review(IMAGE, REQUEST).to_review(now=NOW)
    assert review.status is ReviewStatus.PENDING
    assert review.decided is None


def test_a_pending_hook_can_name_the_agent_as_the_one_who_will_answer():
    assert PendingApproval(reviewer=Reviewer.AGENT).review(IMAGE, REQUEST).reviewer is (
        Reviewer.AGENT
    )


def test_a_decision_built_by_hand_still_answers_approved():
    """The faces build these directly when a click or a tool call arrives."""
    yes = ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.HUMAN)
    no = ApprovalDecision(
        status=ReviewStatus.REJECTED, reviewer=Reviewer.HUMAN, feedback="too dark"
    )
    assert yes.approved is True
    assert no.approved is False
    assert no.to_review(now=NOW).feedback == "too dark"
    assert no.to_review(now=NOW).decided == NOW
