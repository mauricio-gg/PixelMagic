from pixelmagic.model import ReviewStatus
from pixelmagic.stages.generate import load_generation
from pixelmagic.store import Store


def draw(client, subject="a cosy den", **extra):
    """Ask for one picture and return the response the browser lands on."""
    data = {"subject": subject, "kind": "scene", "background": "transparent", "size": "64"}
    data.update(extra)
    return client.post("/directions/cosy-den/generate", data=data)


def only_generation(project_root):
    """The single generation this direction has, read off disk."""
    store = Store(project_root)
    ids = store.list_generations("cosy-den")
    assert len(ids) == 1, ids
    return store, load_generation(store, "cosy-den", ids[0])


def test_drawing_once_leaves_the_picture_waiting_for_a_verdict(client, project_root):
    response = draw(client)
    assert response.status_code == 200
    store, record = only_generation(project_root)
    assert record.review.status is ReviewStatus.PENDING
    assert record.image_path.exists()


def test_the_picture_is_shown_and_can_be_fetched(client, project_root):
    draw(client)
    _, record = only_generation(project_root)
    url = f"/directions/cosy-den/generations/{record.request.id}/image.png"
    assert url in client.get(f"/directions/cosy-den/generations/{record.request.id}").text
    picture = client.get(url)
    assert picture.status_code == 200
    assert picture.headers["content-type"] == "image/png"
    assert "content-disposition" not in picture.headers


def test_approving_records_the_yes_and_keeps_it_as_a_reference(client, project_root):
    draw(client)
    store, record = only_generation(project_root)
    response = client.post(
        f"/directions/cosy-den/generations/{record.request.id}/approve",
        data={"keep_reference": "1"},
    )
    assert response.status_code == 200
    after = load_generation(store, "cosy-den", record.request.id)
    assert after.review.status is ReviewStatus.APPROVED
    assert after.review.reviewer.value == "human"
    assert store.load_direction("cosy-den").references == [f"{record.request.id}.png"]


def test_approving_without_ticking_anything_keeps_nothing(client, project_root):
    draw(client)
    store, record = only_generation(project_root)
    client.post(f"/directions/cosy-den/generations/{record.request.id}/approve", data={})
    assert store.load_direction("cosy-den").references == []
    assert store.list_assets("cosy-den") == []


def test_keep_reference_adds_an_approved_picture_and_refuses_a_pending_one(
    client, project_root
):
    """The keep-reference route files an approved picture as a style reference;
    on a picture nobody approved it is a 400 page and adds nothing."""
    draw(client)
    store, record = only_generation(project_root)
    url = f"/directions/cosy-den/generations/{record.request.id}/keep-reference"

    refused = client.post(url)
    assert refused.status_code == 400
    assert store.load_direction("cosy-den").references == []

    client.post(f"/directions/cosy-den/generations/{record.request.id}/approve", data={})
    kept = client.post(url)
    assert kept.status_code == 200
    assert store.load_direction("cosy-den").references == [f"{record.request.id}.png"]


def test_rejecting_records_the_no_and_draws_the_next_attempt(client, project_root):
    draw(client)
    store, record = only_generation(project_root)
    response = client.post(
        f"/directions/cosy-den/generations/{record.request.id}/reject",
        data={"feedback": "too dark"},
    )
    assert response.status_code == 200
    first = load_generation(store, "cosy-den", record.request.id)
    assert first.review.status is ReviewStatus.REJECTED
    assert first.review.feedback == "too dark"

    ids = store.list_generations("cosy-den")
    assert len(ids) == 2
    child_id = next(one for one in ids if one != record.request.id)
    child = load_generation(store, "cosy-den", child_id)
    assert child.request.parent_id == record.request.id
    assert "too dark" in child.request.prompt
    assert child.review.status is ReviewStatus.PENDING
    assert child_id in response.url.path


def test_rejecting_with_nothing_to_say_still_works(client, project_root):
    draw(client)
    store, record = only_generation(project_root)
    client.post(
        f"/directions/cosy-den/generations/{record.request.id}/reject", data={"feedback": "  "}
    )
    assert load_generation(store, "cosy-den", record.request.id).review.feedback is None


def test_a_stale_tab_cannot_change_a_verdict_that_was_already_given(client, project_root):
    """The tab that still shows the buttons is the whole reason this guard exists.

    A freshly rendered approved page has no verdict buttons; the stale tab is
    the one that still has them. `record_decision` raises
    `ValueError`, which the 400 page handles, so the second click is refused
    rather than silently overwriting the first verdict.
    """
    draw(client)
    _, record = only_generation(project_root)
    url = f"/directions/cosy-den/generations/{record.request.id}"
    client.post(f"{url}/approve", data={})
    again = client.post(f"{url}/reject", data={"feedback": "changed my mind"})
    assert again.status_code == 400
    assert "already" in again.text


def test_a_picture_nobody_drew_is_a_404_page(client):
    response = client.get("/directions/cosy-den/generations/20260101-000000")
    assert response.status_code == 404
    assert "Traceback" not in response.text


def test_a_service_that_is_not_built_yet_is_a_page_that_names_the_ticket(
    project_root, monkeypatch
):
    """The ticket string reaches the browser, proved without naming a real service.

    Never `service="codex"`. By the time this plan runs M2.3 has merged, so
    `codex` builds a real service and this test would spend a subscription turn
    drawing a picture — measured at 91 seconds for the suite. `PLANNED_SERVICES`
    is empty by then too, so there is no unbuilt name left to borrow. Announcing
    one that will never exist exercises the same branch and constructs nothing.
    """
    from fastapi.testclient import TestClient

    from pixelmagic import registry
    from pixelmagic.refiner.stub import StubRefiner
    from pixelmagic.web.app import create_app

    monkeypatch.setitem(registry.PLANNED_SERVICES, "someday", "M9.9")
    application = create_app(project_root, service="someday", refiner=StubRefiner())
    with TestClient(application) as unbuilt:
        response = unbuilt.post(
            "/directions/cosy-den/generate",
            data={"subject": "a cosy den", "kind": "scene",
                  "background": "transparent", "size": "64"},
        )
    assert response.status_code == 400
    assert "not built yet" in response.text
    assert "M9.9" in response.text
    # The front page says it as well, and with the ticket rather than the bare
    # name: one dict reaching all three faces is what M2.1 asks for.
    with TestClient(application) as unbuilt:
        front = unbuilt.get("/").text
    assert "someday" in front
    assert "M9.9" in front


def test_a_refiner_failure_while_keeping_the_scene_leaves_it_retryable(project_root):
    """Keeping the scene calls `refiner.analyze`. If that fails, no asset may be
    written, and a second keep with a working refiner must succeed."""
    from fastapi.testclient import TestClient

    from pixelmagic.refiner.base import EXIT_ENGINE_FAILURE, RefinerError
    from pixelmagic.refiner.stub import StubRefiner
    from pixelmagic.web.app import create_app

    class FlakyAnalyzeRefiner(StubRefiner):
        """A stub whose `analyze` refuses once, then works like the real thing."""

        def __init__(self):
            self.analyze_calls = 0

        def analyze(self, input_path, settings=None):
            self.analyze_calls += 1
            if self.analyze_calls == 1:
                raise RefinerError(
                    "pixel-refiner crashed", exit_code=EXIT_ENGINE_FAILURE, code="ENGINE_FAILURE"
                )
            return super().analyze(input_path, settings)

    application = create_app(project_root, service="fake", refiner=FlakyAnalyzeRefiner())
    with TestClient(application) as flaky:
        draw(flaky)
        store, record = only_generation(project_root)
        base = f"/directions/cosy-den/generations/{record.request.id}"
        flaky.post(f"{base}/approve", data={"keep_reference": "1"})

        failed = flaky.post(f"{base}/keep-scene")
        assert failed.status_code == 400
        assert store.list_assets("cosy-den") == []

        retried = flaky.post(f"{base}/keep-scene")
        assert retried.status_code == 200
        assert store.list_assets("cosy-den") == ["cosy-den"]


def test_the_typed_scene_name_reaches_the_core_on_keep_and_on_rename(client, project_root):
    """Catches: the page's name box being dropped on the way to the core, so the scene is
    filed under the suggestion, and a refused name showing a traceback instead of a sentence."""
    draw(client)
    store, record = only_generation(project_root)
    base = f"/directions/cosy-den/generations/{record.request.id}"
    client.post(f"{base}/approve", data={"keep_reference": "1"})

    client.post(f"{base}/keep-scene", data={"name": "main-room"})
    assert store.list_assets("cosy-den") == ["main-room"]

    refused = client.post(f"{base}/rename-scene", data={"name": "Main Room"})
    assert refused.status_code == 400
    assert "Traceback" not in refused.text
    client.post(f"{base}/rename-scene", data={"name": "big-room"})
    assert store.list_assets("cosy-den") == ["big-room"]
