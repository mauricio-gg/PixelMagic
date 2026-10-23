"""The single place a picture is said yes or no to.

Spec locked decision 7: the core asks one question — "is this image good?" —
and each face answers it its own way. The web page answers with a click, the
command line with a flag or a terminal prompt, and an MCP agent with its own
judgement. Nothing else in the core knows which face is running.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from pixelmagic.model import GenerationRequest, Review, Reviewer, ReviewStatus

#: What a `CallableApproval` callback may return: a bare yes/no, or a yes/no
#: paired with the feedback that should steer the next attempt.
Decider = Callable[[Path, GenerationRequest], "bool | tuple[bool, str | None]"]


class ApprovalDecision(BaseModel):
    """One verdict on one image — including "nobody has decided yet".

    A verdict used to be a boolean, which had no way to say that the question
    is still open. `ReviewStatus.PENDING` has existed since M1 and has never
    been reachable; it is what the web page and the MCP server need, because
    both of them answer across two round trips rather than inside the call.
    """

    status: ReviewStatus
    reviewer: Reviewer
    feedback: str | None = None

    @property
    def approved(self) -> bool:
        """True only for a decided yes. Pending is not a yes."""
        return self.status is ReviewStatus.APPROVED

    def to_review(self, *, now: datetime | None = None) -> Review:
        """Convert the verdict into the `review.json` sidecar model.

        A pending verdict carries no `decided` moment: stamping one would say
        somebody decided at a time when nobody had.
        """
        return Review(
            status=self.status,
            reviewer=self.reviewer,
            feedback=self.feedback,
            decided=None if self.status is ReviewStatus.PENDING else (now or datetime.now(UTC)),
        )


class ApprovalHook(Protocol):
    """Anything that can look at an image and say yes or no."""

    def review(self, image_path: Path, request: GenerationRequest) -> ApprovalDecision:
        """Decide whether the image at `image_path` answers `request`."""
        ...


class AutoApprove:
    """Says yes to everything. Used by `--auto-approve` and by most tests."""

    def review(self, image_path: Path, request: GenerationRequest) -> ApprovalDecision:
        """Approve without looking."""
        return ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.AUTO)


class CallableApproval:
    """Delegates the decision to a plain function.

    This is the hook the MCP server and the web page will use — they hand in a
    function that consults the agent or the browser — and it is how tests script
    a rejection followed by an approval.
    """

    def __init__(self, decide: Decider, *, reviewer: Reviewer = Reviewer.AGENT) -> None:
        """Wrap `decide`, recording verdicts as coming from `reviewer`."""
        self._decide = decide
        self._reviewer = reviewer

    def review(self, image_path: Path, request: GenerationRequest) -> ApprovalDecision:
        """Ask the wrapped function, accepting a bool or a (bool, feedback) pair."""
        answer = self._decide(image_path, request)
        approved, feedback = answer if isinstance(answer, tuple) else (answer, None)
        return ApprovalDecision(
            status=ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED,
            feedback=feedback,
            reviewer=self._reviewer,
        )


class TerminalApproval:
    """Asks the person at the keyboard. Used by the CLI's `--wait` flag.

    `prompt` and `out` are injected so the behaviour can be tested without a
    terminal; in real use they are `input` and `print`.
    """

    def __init__(
        self,
        *,
        prompt: Callable[[str], str] = input,
        out: Callable[[str], None] = print,
    ) -> None:
        """Bind the input and output functions to use."""
        self._prompt = prompt
        self._out = out

    def review(self, image_path: Path, request: GenerationRequest) -> ApprovalDecision:
        """Show the image path, ask yes/no, and collect feedback on a no."""
        self._out(f"\n{request.kind.value}: {request.subject}")
        self._out(f"open this and look at it: {image_path}")
        answer = self._prompt("keep it? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return ApprovalDecision(status=ReviewStatus.APPROVED, reviewer=Reviewer.HUMAN)
        feedback = self._prompt("what should change? ").strip() or None
        return ApprovalDecision(
            status=ReviewStatus.REJECTED, feedback=feedback, reviewer=Reviewer.HUMAN
        )


class PendingApproval:
    """Answers "not yet" — the hook the web page and the MCP server use.

    Both of those faces answer across two round trips: one to draw the picture
    and show it, another to carry the click or the tool call back. The hook is
    still the only place a verdict is decided (locked decision 7); what changes
    is that this one records that the question is open, and
    `stages.generate.record_decision` closes it when the answer arrives. The
    alternative — holding an HTTP request open while a loop waits for a click
    that can only arrive on a different request — needs a background thread and
    a place to keep it, and PixelMagic has no state outside the sidecars.

    It has no caller inside the core; both of its callers are the faces landing
    next (see the deviations register).
    """

    def __init__(self, *, reviewer: Reviewer = Reviewer.HUMAN) -> None:
        """Record who is expected to answer, so `review.json` names them."""
        self._reviewer = reviewer

    def review(self, image_path: Path, request: GenerationRequest) -> ApprovalDecision:
        """Leave the picture waiting for a verdict."""
        return ApprovalDecision(status=ReviewStatus.PENDING, reviewer=self._reviewer)
