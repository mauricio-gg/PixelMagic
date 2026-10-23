"""The one rule for starting an art direction, shared by all three faces.

Each test drives `start_direction` itself and checks what actually landed on
disk (or did not) — never the wording of the module under test.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pixelmagic.directions import start_direction
from pixelmagic.naming import UnusableName
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.services import NoServiceChosen
from pixelmagic.store import AlreadyExists, Store

NOW = datetime(2026, 9, 6, 12, 0, 0, tzinfo=UTC)


def project(tmp_path) -> Store:
    """A fresh project with nothing in it yet."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    return store


def test_a_typed_name_is_squeezed_and_becomes_the_default(tmp_path):
    store = project(tmp_path)
    made = start_direction(
        store, style="Warm SNES interiors.", service=FakeImageService(),
        service_name="fake", name="Cosy Den", donts=["no gradients", "  "], now=NOW,
    )
    assert made.name == "cosy-den"
    assert made.donts == ["no gradients"]
    assert made.service == "fake"
    assert store.load_direction("cosy-den").style == "Warm SNES interiors."
    assert store.load_project().default_direction == "cosy-den"


def test_a_blank_name_is_treated_as_no_name(tmp_path):
    """The web form posts "" for an empty box; the service invents one."""
    store = project(tmp_path)
    made = start_direction(
        store, style="Warm SNES interiors.", service=FakeImageService(),
        service_name="fake", name="",
    )
    assert made.name == "wooden-chair"


def test_a_cell_size_inside_the_bound_is_stored(tmp_path):
    store = project(tmp_path)
    made = start_direction(
        store, style="Warm SNES interiors.", service=FakeImageService(),
        service_name="fake", name="cosy-den", cell_size=32,
    )
    assert made.cell_size == 32
    assert store.load_direction("cosy-den").cell_size == 32


def test_a_cell_size_outside_the_models_bound_writes_nothing(tmp_path):
    """The bound is `Direction`'s own; this proves the core does not widen it."""
    store = project(tmp_path)
    with pytest.raises(ValidationError):
        start_direction(
            store, style="Warm SNES interiors.", service=FakeImageService(),
            service_name="fake", name="cosy-den", cell_size=3,
        )
    assert store.list_directions() == []


def test_an_unusable_name_writes_nothing(tmp_path):
    store = project(tmp_path)
    with pytest.raises(UnusableName, match="日本語"):
        start_direction(
            store, style="Warm SNES interiors.", service=FakeImageService(),
            service_name="fake", name="日本語",
        )
    assert store.list_directions() == []


def test_a_blank_name_with_no_service_asks_for_none_and_writes_nothing(tmp_path):
    """Catches: an unnamed direction making a paid naming call, or leaving a folder,
    when nothing is chosen."""
    store = project(tmp_path)
    with pytest.raises(NoServiceChosen):
        start_direction(
            store, style="Warm SNES interiors.", service=None, service_name=None, name="  "
        )
    assert store.list_directions() == []


def test_a_folder_with_no_project_gets_no_direction(tmp_path):
    with pytest.raises(FileNotFoundError):
        start_direction(
            Store(tmp_path), style="Warm SNES interiors.", service=FakeImageService(),
            service_name="fake", name="cosy-den",
        )
    assert not (tmp_path / "directions").exists()


def test_a_second_direction_leaves_the_default_alone(tmp_path):
    store = project(tmp_path)
    kwargs = dict(style="Warm SNES interiors.", service=FakeImageService(), service_name="fake")
    start_direction(store, name="cosy-den", **kwargs)
    start_direction(store, name="cold-cave", **kwargs)
    assert store.load_project().default_direction == "cosy-den"


def test_an_existing_name_is_refused_and_the_old_direction_is_untouched(tmp_path):
    store = project(tmp_path)
    kwargs = dict(style="Warm SNES interiors.", service=FakeImageService(), service_name="fake")
    start_direction(store, name="cosy-den", **kwargs)
    with pytest.raises(AlreadyExists, match="already exists"):
        start_direction(store, name="cosy-den", style="Something else.",
                         service=FakeImageService(), service_name="fake")
    assert store.load_direction("cosy-den").style == "Warm SNES interiors."
