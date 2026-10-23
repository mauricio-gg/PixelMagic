import pytest

from pixelmagic.providers.fake import FakeImageService
from pixelmagic.refiner.cli_adapter import PixelRefinerCli
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.registry import (
    DEFAULT_REFINER,
    PLANNED_SERVICES,
    NotBuiltYet,
    RegistryError,
    UnknownName,
    known_refiners,
    known_services,
    load_refiner,
    load_service,
)


def test_the_services_that_work_today():
    """The ledger of what has actually been built, one line per service ticket."""
    from pixelmagic.providers.codex import CodexImageService
    from pixelmagic.providers.gemini import GeminiImageService
    from pixelmagic.providers.openai_images import OpenAIImageService

    assert isinstance(load_service("fake"), FakeImageService)
    assert isinstance(load_service("codex"), CodexImageService)
    assert isinstance(load_service("gemini"), GeminiImageService)
    assert isinstance(load_service("openai"), OpenAIImageService)
    assert known_services() == ["codex", "fake", "gemini", "openai"]


def test_every_planned_service_is_now_built():
    assert PLANNED_SERVICES == {}


def test_an_unknown_service_lists_the_ones_that_work():
    with pytest.raises(UnknownName, match="fake"):
        load_service("dall-e-2")


def test_both_refiners_load_by_name():
    assert isinstance(load_refiner("stub"), StubRefiner)
    assert isinstance(load_refiner("cli"), PixelRefinerCli)
    assert known_refiners() == ["cli", "stub"]


def test_an_unknown_refiner_lists_the_ones_that_work():
    with pytest.raises(UnknownName, match="stub"):
        load_refiner("photoshop")


def test_the_default_refiner_is_the_stand_in():
    """One constant is every face's default, so one assertion covers all six sites.
    The real PixelRefiner is opt-in: a machine without it must still work."""
    assert isinstance(load_refiner(DEFAULT_REFINER), StubRefiner)


def test_both_refusals_are_registry_errors_a_face_can_catch():
    """Every face catches this family at its boundary, and `cli.load_service`
    catches `RegistryError` and not bare `ValueError` — so the relationship is
    load-bearing and gets a test rather than a comment."""
    assert issubclass(RegistryError, ValueError)
    assert issubclass(UnknownName, RegistryError)
    assert issubclass(NotBuiltYet, RegistryError)
