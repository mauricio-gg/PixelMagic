from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from pixelmagic.journey import StepKey
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.stages.generate import load_generation
from pixelmagic.store import Store
from pixelmagic.web.app import create_app
from pixelmagic.web.generating import generation_view
from pixelmagic.web.projects import edited_phrase
from pixelmagic.web.timeline import frame_context


def draw_one(client, project_root):
    """Draw one pending scene picture and return its generation id."""
    client.post(
        "/directions/cosy-den/generate",
        data={"subject": "a cosy den", "kind": "scene", "background": "transparent", "size": "64"},
    )
    return Store(project_root).list_generations("cosy-den")[0]


def test_a_pending_pictures_timeline_links_only_the_steps_that_can_be_visited(
    client, project_root
):
    """A waiting step linking to Refine would send a person to a screen with nothing to refine."""
    generation = draw_one(client, project_root)
    bench = client.app.state.workbench
    context = frame_context(bench, "cosy-den", generation, StepKey.APPROVE)
    by_label = {step["label"]: step for step in context["timeline"]}
    page = f"/directions/cosy-den/generations/{generation}"
    assert by_label["Start"]["href"] == "/directions/cosy-den/generate"
    assert by_label["Approve"]["href"] == page
    assert by_label["Approve"]["here"] is True
    assert by_label["Keep scene"]["href"] == page
    assert [by_label[name]["href"] for name in ("Pick props", "Surfaces", "Refine", "Export")] == [
        None
    ] * 4
    assert context["timeline_key"] == f"cosy-den/{generation}"


def test_the_picture_page_marks_approve_then_keep_then_nothing_once_kept(
    client, project_root
):
    """The frame's "you are here" must follow the verdict, not stay on the first step."""
    generation = draw_one(client, project_root)
    bench = client.app.state.workbench

    def here() -> list[str]:
        record = load_generation(bench.store, "cosy-den", generation)
        view = generation_view(bench, bench.store.load_direction("cosy-den"), record)
        return [step["label"] for step in view["timeline"] if step["here"]]

    assert here() == ["Approve"]
    client.post(f"/directions/cosy-den/generations/{generation}/approve", data={})
    assert here() == ["Keep scene"]
    client.post(f"/directions/cosy-den/generations/{generation}/keep-scene")
    # A kept scene has nothing left to edit: no step is "here", so no step reads as reopened.
    assert here() == []


@pytest.mark.parametrize(
    ("then", "where"),
    [
        ("generate", "/directions/cold-cave/generate"),
        ("import", "/directions/cold-cave/generate#import"),
        ("https://evil.example/", "/directions/cold-cave"),
        ("", "/directions/cold-cave"),
    ],
)
def test_a_new_direction_goes_on_only_to_the_places_the_form_offers(client, then, where):
    """`then` is a choice, not a URL: a forged value must not turn the redirect into an open one."""
    response = client.post(
        "/directions",
        data={"name": "Cold Cave", "style": "Blue caves.", "then": then},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == where


def test_a_direction_called_new_opens_while_the_form_is_served_elsewhere(client):
    client.post("/directions", data={"name": "new", "style": "Plain."})
    assert client.get("/directions/new").status_code == 200
    form = client.get("/new-direction")
    assert form.status_code == 200
    assert 'name="cell_size"' in form.text
    # catches: a person's direction named "new" being unreachable.


def test_a_folder_with_no_project_names_the_command_that_makes_one(tmp_path):
    empty = tmp_path / "nothing here"
    empty.mkdir()
    application = create_app(empty, service="fake", refiner=StubRefiner())
    with TestClient(application) as bare:
        response = bare.get("/")
    assert response.status_code == 404
    assert "project new" in response.text
    assert str(empty) in response.text.replace("&#39;", "'")


def test_the_homes_buttons_lead_to_the_lead_direction(client):
    """New scene must lead to a direction that exists, or the home's first button is a 404."""
    body = client.get("/").text
    assert 'href="/directions/cosy-den/generate"' in body
    assert 'href="/directions/cosy-den/generate#import"' in body


def test_edited_rolls_over_at_utc_midnight_not_after_24_hours():
    now = datetime(2026, 10, 8, 0, 5, tzinfo=UTC)
    assert edited_phrase(datetime(2026, 10, 7, 23, 59, tzinfo=UTC), now) == "yesterday"
    assert edited_phrase(datetime(2026, 10, 8, 0, 1, tzinfo=UTC), now) == "today"
    assert edited_phrase(datetime(2026, 10, 5, 12, 0, tzinfo=UTC), now) == "3 days ago"


def test_the_surfaces_step_links_to_the_pictures_surfaces_page(client, project_root):
    """Catches: the timeline sending Surfaces to a page that no longer exists."""
    generation = draw_one(client, project_root)
    client.post(f"/directions/cosy-den/generations/{generation}/approve", data={})
    client.post(f"/directions/cosy-den/generations/{generation}/keep-scene")
    context = frame_context(client.app.state.workbench, "cosy-den", generation, StepKey.SURFACES)
    step = next(step for step in context["timeline"] if step["label"] == "Surfaces")
    assert step["href"] == f"/directions/cosy-den/generations/{generation}/surfaces"
    assert step["here"] is True


def test_the_scene_choice_only_returns_to_refine_or_export(client, project_root):
    """The form carries a page word, not a URL: anything else is a sentence, not a redirect."""
    response = client.post(
        "/directions/cosy-den/scene",
        data={"generation": "20260903-191233", "return": "https://example.com/"},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert "location" not in response.headers
    assert not Store(project_root).scene_choice_file("cosy-den").exists()
    # catches: the chooser becoming an open redirect to a URL the form supplied.
