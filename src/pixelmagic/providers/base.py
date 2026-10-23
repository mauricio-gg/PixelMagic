"""The one interface every image service implements, plus prompt composition.

Services differ wildly (a local Codex process, a Google SDK, an HTTP API), so
the core only ever sees this protocol. Adding a service in plan 2 means writing
a class with these three methods — nothing in `stages/` changes.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from pixelmagic.model import GenerationRequest, GenerationResult, PromptKind

#: Template file for each prompt kind, shipped as package data.
TEMPLATE_FILES: dict[PromptKind, str] = {
    PromptKind.SCENE: "scene.md",
    PromptKind.PROP: "prop.md",
    PromptKind.GROUP_REDRAW: "group-redraw.md",
}

DIRECTION_NAME_FILE = "direction-name.md"


def load_template(kind: PromptKind) -> str:
    """Read one prompt template out of the package."""
    return (
        importlib.resources.files("pixelmagic.prompts")
        .joinpath(TEMPLATE_FILES[kind])
        .read_text(encoding="utf-8")
    )


def load_direction_name_question() -> str:
    """Read the template that asks a service to name an art direction."""
    return (
        importlib.resources.files("pixelmagic.prompts")
        .joinpath(DIRECTION_NAME_FILE)
        .read_text(encoding="utf-8")
    )


class ImageCapabilities(BaseModel):
    """What one service can actually do.

    The stages read this rather than branching on a service's name — that is
    how "transparent first, key colour second" stays a rule and not a special
    case per provider.
    """

    transparent_background: bool = False
    reference_images: bool = False
    edit: bool = False
    vision: bool = False
    sizes: list[tuple[int, int]] = Field(default_factory=list)


class ServiceError(RuntimeError):
    """An image service could not produce a picture.

    Every real service catches whatever its own transport raises — a
    subprocess that would not start, an SDK's HTTP error, a filter refusing a
    prompt — and re-raises it as this. That gives the stages one exception to
    catch and lets `stages.generate` promise that a failed generation leaves
    nothing behind: a stage cannot clean up after a failure it did not expect.

    `service` is the failing service's `name`, so a message can say who failed
    without the caller having to remember. `code` is the service's own word for
    the failure when it has one (`"moderation_blocked"`, `"NO_IMAGE"`), so a
    face can tell "try again" apart from "change the prompt" without reading
    the sentence.
    """

    def __init__(self, message: str, *, service: str, code: str | None = None) -> None:
        """Record which service failed and, when it gave one, its own error code."""
        super().__init__(message)
        self.service = service
        self.code = code


class ImageService(Protocol):
    """An image service PixelMagic can draw with."""

    name: str

    def capabilities(self) -> ImageCapabilities:
        """Describe what this service supports."""
        ...

    def generate(
        self,
        request: GenerationRequest,
        out_path: Path,
        *,
        reference_paths: Sequence[Path] = (),
    ) -> GenerationResult:
        """Draw `request` and save the PNG at `out_path`, returning provenance.

        `reference_paths` are the real files behind `request.references`, in
        the same order and already resolved by the caller. The request itself
        holds only names inside the direction's `references/` folder — that is
        deliberate, so a request can be replayed from the direction folder
        alone — and this keyword is how a service that can actually use them
        gets somewhere it can open.

        Raises `ServiceError`, and nothing else, when it cannot draw.
        """
        ...

    def describe(self, image_path: Path | None, question: str) -> str:
        """Answer a short question about an image, or about nothing at all.

        `image_path` is None when the question stands on its own.
        Used to name cutouts and to invent an art-direction name. Services
        without vision may still answer text-only questions.
        """
        ...
