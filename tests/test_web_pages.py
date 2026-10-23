import pytest
from fastapi.testclient import TestClient

from pixelmagic.refiner.cli_adapter import PixelRefinerCli
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.store import Store
from pixelmagic.web.app import LOOPBACK_HOSTS, create_app, serve
from pixelmagic.web.images import send_image


def test_the_front_page_names_the_project_and_its_directions(client):
    body = client.get("/").text
    assert "DemoGame" in body
    assert "cosy-den" in body


def test_the_front_page_says_nothing_about_planned_services_when_there_are_none(client):
    """Which is the state this plan actually runs in, M2.3 to M2.5 having merged.

    The same case `project_overview` guards on the other face: a heading with
    nothing after it reads as a bug.
    """
    assert "Not built yet" not in client.get("/").text


def test_the_real_refiner_doing_the_work_is_also_the_settings_source(project_root):
    """Catches: a `--refiner cli` user losing the settings screen because no
    options source was built for the real refiner (constructing it spawns nothing)."""
    real = PixelRefinerCli()
    application = create_app(project_root, service="fake", refiner=real)
    assert application.state.workbench.options_source is real


def test_the_stand_in_as_work_refiner_never_spawns_the_real_one_for_settings(project_root):
    """A stranger has no PixelRefiner: with the stand-in doing the work the
    settings screen reads nothing from the real tool, offers no stand-in
    settings either, and says how to turn the real one on."""
    application = create_app(project_root, service="fake", refiner=StubRefiner())
    assert application.state.workbench.options_source is None
    with TestClient(application) as page:
        body = page.get("/directions/cosy-den/refine").text
    assert "The real PixelRefiner is not turned on" in body
    assert "--refiner cli" in body
    assert 'name="quick.dithering"' not in body


def test_a_new_direction_can_be_made_from_the_form(client, project_root):
    response = client.post(
        "/directions",
        data={"name": "Cold Cave", "style": "Blue caves.", "donts": "no gradients\nno blur"},
    )
    assert response.status_code == 200
    assert Store(project_root).load_direction("cold-cave").donts == ["no gradients", "no blur"]


def test_a_direction_without_a_name_gets_one_from_the_service(client, project_root):
    client.post("/directions", data={"name": "", "style": "Blue caves.", "donts": ""})
    assert "wooden-chair" in Store(project_root).list_directions()


def test_making_a_direction_that_exists_is_a_page_and_not_a_traceback(client, project_root):
    response = client.post(
        "/directions", data={"name": "cosy-den", "style": "Something else.", "donts": ""}
    )
    assert response.status_code == 409
    assert "already exists" in response.text
    assert "Traceback" not in response.text
    assert Store(project_root).load_direction("cosy-den").style.startswith("Warm SNES")


def test_the_direction_page_shows_its_style_and_its_donts(client):
    body = client.get("/directions/cosy-den").text
    assert "Warm SNES interiors" in body
    assert "no gradients" in body


def test_a_direction_can_be_renamed_from_its_page(client, project_root):
    response = client.post("/directions/cosy-den/rename", data={"name": "Warm Den"})
    assert response.status_code == 200
    assert Store(project_root).list_directions() == ["warm-den"]
    assert Store(project_root).load_project().default_direction == "warm-den"
    assert client.get("/directions/warm-den").status_code == 200
    assert client.get("/directions/cosy-den").status_code == 404


def test_renaming_to_nothing_says_so_instead_of_calling_it_item(client, project_root):
    response = client.post("/directions/cosy-den/rename", data={"name": "   "})
    assert response.status_code == 400
    assert Store(project_root).list_directions() == ["cosy-den"]


def test_a_direction_name_in_another_script_is_refused(client, project_root):
    """`slug` returns None for text with nothing Latin-alphabet in it.

    The same rule `direction new --name` enforces at the command line: a
    direction called "item" that nobody asked for is worse than a 400.
    """
    response = client.post(
        "/directions", data={"name": "日本語", "style": "Blue caves.", "donts": ""}
    )
    assert response.status_code == 400
    assert Store(project_root).list_directions() == ["cosy-den"]


def test_a_direction_nobody_has_is_a_404_page(client):
    response = client.get("/directions/nope")
    assert response.status_code == 404
    assert "nope" in response.text
    assert "Traceback" not in response.text


@pytest.mark.parametrize("bad", ["%2e%2e", "%2e%2e%5cescape", "a%00b"])
def test_a_url_that_is_not_one_of_our_names_is_refused(client, bad):
    """These three reach the handler intact, so `check_name` is what refuses them.

    Not `%2e%2e%2fescape`, which is what an earlier draft of this test used:
    `%2f` is decoded *before* routing, so that URL becomes two path segments,
    no route matches, and Starlette's own 404 answers — the test passes
    without `check_name` ever being called. `%2e%2e` alone is not decoded into
    a separator, so it does reach the handler, and a 400 here is the guard
    actually working. Measured on starlette 1.6.0 on 2026-09-04.
    """
    response = client.get(f"/directions/{bad}")
    assert response.status_code == 400
    assert "Traceback" not in response.text


def test_a_folder_with_no_project_in_it_is_a_404_page(tmp_path):
    """The first thing a new user hits: a root that holds no pixelmagic.json."""
    empty = tmp_path / "nothing-here"
    empty.mkdir()
    application = create_app(empty, service="fake", refiner=StubRefiner())
    with TestClient(application) as bare:
        response = bare.get("/")
    assert response.status_code == 404
    assert "Traceback" not in response.text
    assert "<html" in response.text


def test_a_broken_project_sidecar_is_a_page_and_not_a_traceback(project_root):
    (project_root / "pixelmagic.json").write_text("{not json", encoding="utf-8")
    application = create_app(project_root, service="fake", refiner=StubRefiner())
    with TestClient(application) as broken:
        response = broken.get("/")
    assert response.status_code == 500
    assert "Traceback" not in response.text


class Boom(Exception):
    """An exception no handler above knows about, for the catch-all test.

    Deliberately not a `RuntimeError`: that family has its own page now,
    because `ServiceError`, `RefinerError` and `ApprovalExhausted` are all
    `RuntimeError` subclasses and all three are ordinary outcomes a person
    should be told about. "Nobody planned for it" has to be something nobody
    planned for.
    """


def test_anything_nobody_planned_for_is_still_a_page(project_root, monkeypatch):
    """The catch-all. TestClient re-raises unhandled errors, so turn that off."""
    application = create_app(project_root, service="fake", refiner=StubRefiner())
    monkeypatch.setattr(
        "pixelmagic.web.projects.direction_rows",
        lambda store: (_ for _ in ()).throw(Boom("secret internal detail")),
    )
    with TestClient(application, raise_server_exceptions=False) as unlucky:
        response = unlucky.get("/")
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text


def test_a_service_that_could_not_draw_gets_a_page_that_says_why(project_root, monkeypatch):
    """A `RuntimeError` is shown, because that is the family a service fails in.

    `providers.base.ServiceError` — nobody logged in, a filter refused the
    prompt, the connection dropped — is a bare `RuntimeError` subclass. On the
    catch-all it would come out as a blank "Something went wrong", which is the
    one failure the person can actually act on.
    """
    application = create_app(project_root, service="fake", refiner=StubRefiner())
    monkeypatch.setattr(
        "pixelmagic.web.projects.direction_rows",
        lambda store: (_ for _ in ()).throw(RuntimeError("nobody is logged in")),
    )
    with TestClient(application, raise_server_exceptions=False) as unlucky:
        response = unlucky.get("/")
    assert response.status_code == 400
    assert "nobody is logged in" in response.text
    assert "Traceback" not in response.text


def test_send_image_refuses_a_path_outside_the_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("TOPSECRET")
    with pytest.raises(ValueError, match="not inside"):
        send_image(root / ".." / "secret.txt", root)


def test_send_image_refuses_a_symlink_that_points_out_of_the_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"\x89PNG")
    (root / "sneaky.png").symlink_to(outside)
    with pytest.raises(ValueError, match="not inside"):
        send_image(root / "sneaky.png", root)


def test_send_image_says_plainly_when_there_is_no_picture(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        send_image(root / "missing.png", root)


def test_a_picture_that_was_never_drawn_is_a_404_page(client):
    response = client.get("/directions/cosy-den/generations/20260101-000000/image.png")
    assert response.status_code == 404
    assert "Traceback" not in response.text


def test_the_page_refuses_to_listen_to_the_whole_network(project_root):
    """Remote use is out of scope for v1 and there is no login to put in front."""
    application = create_app(project_root, service="fake", refiner=StubRefiner())
    with pytest.raises(ValueError, match="local"):
        serve(application, host="0.0.0.0")
    assert "127.0.0.1" in LOOPBACK_HOSTS


def test_a_direction_form_takes_a_cell_size(client, project_root):
    response = client.post(
        "/directions",
        data={"name": "Cold Cave", "style": "Blue caves.", "donts": "", "cell_size": "32"},
    )
    assert response.status_code == 200
    assert Store(project_root).load_direction("cold-cave").cell_size == 32
    # A bare "32" can match a generated id or a timestamp; the rendered
    # sentence is the actual proof the direction page shows this direction's
    # own cell size.
    assert "Cell size: 32 art pixels" in client.get("/directions/cold-cave").text


def test_a_direction_form_defaults_the_cell_size(client, project_root):
    client.post("/directions", data={"name": "Cold Cave", "style": "Blue caves.", "donts": ""})
    assert Store(project_root).load_direction("cold-cave").cell_size == 16


def test_a_cell_size_outside_the_bound_is_a_page_and_writes_nothing(client, project_root):
    """`Form(ge=, le=)` with the model's own constants: FastAPI refuses it as a
    `RequestValidationError`, and `web.app` turns that into the 400 page
    headlined "That is not a value this page takes" — not the 500 page a
    `ValidationError` out of `Direction(...)` would land on, which blames a
    file PixelMagic wrote."""
    response = client.post(
        "/directions",
        data={"name": "Cold Cave", "style": "Blue caves.", "donts": "", "cell_size": "3"},
    )
    assert response.status_code == 400
    assert "not a value this page takes" in response.text
    assert "Traceback" not in response.text
    assert Store(project_root).list_directions() == ["cosy-den"]


def test_the_direction_page_says_its_cell_size(client):
    assert "Cell size: 16 art pixels" in client.get("/directions/cosy-den").text


def test_a_segmenter_that_cannot_run_is_its_own_page_and_not_an_image_service_failure(
    project_root,
):
    """Catches: a missing helper model reading as "The image service could not draw that"
    (SegmenterError is a RuntimeError, which the image-service handler would otherwise take)
    or as a 500."""
    from pixelmagic.segmenter.base import EXIT_NOT_RUN, SegmenterError

    class Broken:
        name = "broken"

        def suggest(self, image_path):
            raise SegmenterError(
                "the Segment Anything model was never fetched", exit_code=EXIT_NOT_RUN
            )

    application = create_app(
        project_root, service="fake", refiner=StubRefiner(), segmenter=Broken()
    )
    with TestClient(application) as client:
        client.post(
            "/directions/cosy-den/generate",
            data={"subject": "a cosy den", "kind": "scene",
                  "background": "transparent", "size": "120"},
        )
        generation_id = Store(project_root).list_generations("cosy-den")[0]
        client.post(
            f"/directions/cosy-den/generations/{generation_id}/approve",
            data={"keep_reference": "1"},
        )
        response = client.post(f"/directions/cosy-den/generations/{generation_id}/suggest")

    assert response.status_code == 400
    assert "The object finder could not do that" in response.text
    assert "the Segment Anything model was never fetched" in response.text
    assert "image service" not in response.text
    assert "Traceback" not in response.text
