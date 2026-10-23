"""Turning a name a face was given into the thing it names.

All three faces take a service name (`--service fake`) and a refiner name
(`--refiner cli`) and have to turn that string into an object. M1 kept the
lookup inside `cli.py`, where it raised `typer.BadParameter` — a Click
exception. A browser would render that as an unhandled 500 and an MCP agent
would receive it as a traceback, and importing `cli.py` to reach it would drag
Typer and the whole command tree into both faces for the sake of a dictionary
lookup.

So the lookup lives here and raises plain `ValueError` subclasses. Each face
translates them once, into its own vocabulary: a usage error at the terminal,
a page in the browser, a `ToolError` for an agent.
"""

from __future__ import annotations

from collections.abc import Callable

from pixelmagic.providers.base import ImageService
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.refiner.base import Refiner
from pixelmagic.refiner.cli_adapter import PixelRefinerCli
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.base import Segmenter
from pixelmagic.segmenter.cli_adapter import SegmenterCli
from pixelmagic.segmenter.stub import StubSegmenter


class RegistryError(ValueError):
    """A name this registry cannot turn into an object.

    A `ValueError`, so a face that catches that family at its boundary catches
    both kinds below without importing either. A class of its own, so
    `cli.load_service` can catch *this* rather than a bare `ValueError`: a
    pydantic `ValidationError` is a `ValueError` too, and once M2.3–M2.5
    register real factories any `ValueError` a provider's `__init__` raises
    would otherwise be flattened into a one-line usage error — which is exactly
    what `test_a_sidecar_that_will_not_parse_stays_a_traceback` exists to stop.
    """


class UnknownName(RegistryError):
    """A name nothing has ever offered."""


class NotBuiltYet(RegistryError):
    """A name the spec plans but no ticket has landed yet.

    Kept apart from `UnknownName` because the two ask the reader to do
    different things: one is a typo to fix, the other is a ticket to wait for.
    """

    def __init__(self, name: str, ticket: str) -> None:
        """Name the service and the ticket that will build it."""
        super().__init__(
            f"the {name} service is not built yet (ticket {ticket}) — use fake for now"
        )
        self.name = name
        self.ticket = ticket


def _codex_service() -> ImageService:
    """Build the Codex service.

    Imported inside the factory rather than at the top of the module: naming a
    service has to stay a dictionary lookup that costs nothing and starts
    nothing, and a provider that grows an optional dependency later (Gemini and
    OpenAI both do) must not be able to break every face by not being
    installed.
    """
    from pixelmagic.providers.codex import CodexImageService

    return CodexImageService()


def _gemini_service() -> ImageService:
    """Build the Gemini service.

    The module imports with `google-genai` absent — it only reaches for the
    SDK when a client has to be made — so the missing-package message comes
    out of the service as a `ServiceError`, not from here.
    """
    from pixelmagic.providers.gemini import GeminiImageService

    return GeminiImageService()


def _openai_service() -> ImageService:
    """Build the OpenAI service; the SDK is imported inside the module."""
    from pixelmagic.providers.openai_images import OpenAIImageService

    return OpenAIImageService()


#: Image services that exist today.
SERVICES: dict[str, Callable[[], ImageService]] = {
    "fake": FakeImageService,
    "codex": _codex_service,
    "gemini": _gemini_service,
    "openai": _openai_service,
}

#: Every service the spec plans now exists. Kept as an empty mapping rather
#: than deleted: `load_service` still reads it, the faces still render it, and
#: a fourth service would otherwise have nowhere to be announced.
PLANNED_SERVICES: dict[str, str] = {}

#: Refiners that exist today: the stand-in and the real command-line tool.
REFINERS: dict[str, Callable[[], Refiner]] = {
    "stub": StubRefiner,
    "cli": PixelRefinerCli,
}

#: What `--refiner` means when nobody says. Every face reads this one name —
#: four `--refiner` options and two factory fallbacks — so it is the only
#: place the default lives. The stand-in is the default because the real
#: PixelRefiner is opt-in (`--refiner cli`): a machine without it still works.
DEFAULT_REFINER = "stub"


def known_services() -> list[str]:
    """Every service name that works today, alphabetically."""
    return sorted(SERVICES)


def load_service(name: str) -> ImageService:
    """Build the named image service.

    Raises `NotBuiltYet` for a service the spec plans and `UnknownName` for a
    name nothing offers — two sentences, because they are two problems.
    """
    if name in SERVICES:
        return SERVICES[name]()
    if name in PLANNED_SERVICES:
        raise NotBuiltYet(name, PLANNED_SERVICES[name])
    raise UnknownName(f"unknown service {name!r}; known: {', '.join(known_services())}")


def known_refiners() -> list[str]:
    """Every refiner name that works today, alphabetically."""
    return sorted(REFINERS)


def load_refiner(name: str) -> Refiner:
    """Build the named refiner: the stand-in, or the real command-line tool."""
    if name in REFINERS:
        return REFINERS[name]()
    raise UnknownName(f"unknown refiner {name!r}; known: {', '.join(known_refiners())}")


#: Segmenters that exist today: the stand-in and the real Segment Anything
#: helper. Naming `cli` costs nothing and starts nothing; a machine without the
#: helper finds out on first use, as a `SegmenterError`.
SEGMENTERS: dict[str, Callable[[], Segmenter]] = {
    "stub": StubSegmenter,
    "cli": SegmenterCli,
}

#: What `--segmenter` means when nobody says: the real one.
DEFAULT_SEGMENTER = "cli"


def known_segmenters() -> list[str]:
    """Every segmenter name that works today, alphabetically."""
    return sorted(SEGMENTERS)


def load_segmenter(name: str) -> Segmenter:
    """Build the named segmenter.

    Raises `UnknownName` for a name nothing offers.
    """
    if name in SEGMENTERS:
        return SEGMENTERS[name]()
    raise UnknownName(f"unknown segmenter {name!r}; known: {', '.join(known_segmenters())}")
