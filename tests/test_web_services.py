"""The Services page and the Start page's use of it, through the app, offline.

No real keychain (the autouse `isolated_keys` fixture) and no network: the one
key test points the models-list address at a loopback server.
"""

import http.server
import logging
import threading

from fastapi.testclient import TestClient

from conftest import CannedOptions
from pixelmagic import keys, services
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.stages.generate import load_generation
from pixelmagic.store import Store
from pixelmagic.web.app import create_app

SECRET = "AIza-distinctive-SECRET-key-4711"


def app_client(project_root, **flags):
    """A client over the project; `service=` is the --service flag, absent by default."""
    application = create_app(
        project_root,
        refiner=StubRefiner(),
        options_source=CannedOptions(),
        segmenter=StubSegmenter(),
        **flags,
    )
    return TestClient(application)


def test_the_games_saved_choice_decides_who_draws(project_root):
    """Catches: a Start page that ignores the game's choice, so a person who chose the
    stand-in is still sent to a key service (or the reverse, a paid draw they never chose)."""
    with app_client(project_root) as client:
        client.post("/services/choose", data={"service": "gemini"}, follow_redirects=False)
        assert Store(project_root).load_project().service == "gemini"
        start = client.get("/directions/cosy-den/generate")
        assert services.service_status("gemini").nudge in start.text
        assert "disabled" in start.text

    # The stand-in is reached only by the flag, which wins over the saved choice.
    with app_client(project_root, service="fake") as client:
        start = client.get("/directions/cosy-den/generate")
        assert services.service_status("gemini").nudge not in start.text
        assert f"Draw with {services.draw_phrase('fake')}" in start.text

        client.post(
            "/directions/cosy-den/generate",
            data={"subject": "a den", "kind": "scene", "background": "transparent", "size": "64"},
        )
        store = Store(project_root)
        (generation,) = store.list_generations("cosy-den")
        assert load_generation(store, "cosy-den", generation).request.service == "fake"


def test_a_form_posted_by_hand_cannot_draw_when_no_service_is_chosen(project_root):
    """Catches: the Start form drawing with the stand-in or a paid service when the game
    has no chosen service, because the page only hid the button instead of refusing."""
    with app_client(project_root) as client:
        refused = client.post(
            "/directions/cosy-den/generate",
            data={"subject": "a den", "kind": "scene", "background": "transparent", "size": "64"},
        )
    assert refused.status_code == 400
    assert services.NO_SERVICE_SENTENCE in refused.text
    assert Store(project_root).list_generations("cosy-den") == []


def test_the_service_flag_wins_over_the_saved_choice(project_root):
    """Catches: a server started with --service fake that still tries the saved paid service."""
    Store(project_root).save_project(
        Store(project_root).load_project().model_copy(update={"service": "gemini"})
    )
    with app_client(project_root, service="fake") as client:
        start = client.get("/directions/cosy-den/generate")
        assert services.service_status("gemini").nudge not in start.text
        assert "--service fake" in client.get("/services").text


def test_choosing_an_unknown_service_or_with_no_project_is_a_sentence(project_root, tmp_path):
    """Catches: a made-up service name written into pixelmagic.json, or a traceback
    when a person chooses in a folder that holds no game."""
    with app_client(project_root) as client:
        refused = client.post("/services/choose", data={"service": "nonsense"})
        assert refused.status_code == 400
        assert Store(project_root).load_project().service is None
        assert client.get("/services/nonsense/x").status_code == 404
        assert client.post("/services/nonsense/test").status_code == 404
    with app_client(tmp_path / "empty") as client:
        assert client.get("/services").status_code == 200
        refused = client.post("/services/choose", data={"service": "fake"})
        assert refused.status_code == 400
        assert "Traceback" not in refused.text


class RefusingModels(http.server.BaseHTTPRequestHandler):
    """Answers the models-list call with 401, like a service refusing a key."""

    def do_GET(self):
        self.send_response(401)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_a_key_is_never_sent_back_or_logged(project_root, monkeypatch, caplog):
    """Catches: the pasted key echoed into a page, a redirect address or a log line
    (the Services page, the Start page, the test result, Replace or Remove)."""
    caplog.set_level(logging.DEBUG)
    server = http.server.HTTPServer(("127.0.0.1", 0), RefusingModels)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setitem(
        services.MODELS_URLS, "gemini", f"http://127.0.0.1:{server.server_port}/v1beta/models"
    )
    seen = []
    try:
        with app_client(project_root) as client:
            def look(response):
                seen.append(response.text + str(response.headers))
                return response

            chose = client.post(
                "/services/choose", data={"service": "gemini"}, follow_redirects=False
            )
            look(chose)
            saved = look(
                client.post("/services/gemini/key", data={"key": SECRET}, follow_redirects=False)
            )
            assert saved.status_code == 303
            page = look(client.get("/services"))
            assert "Key saved in your keychain" in page.text
            assert 'type="password"' in look(client.get("/services")).text  # Replace key's field
            look(client.get("/directions/cosy-den/generate"))
            tested = look(client.post("/services/gemini/test"))
            assert "did not accept" in tested.text
            look(client.post("/services/gemini/key/remove", follow_redirects=False))
            assert "No key yet" in look(client.get("/services")).text
    finally:
        server.shutdown()
        server.server_close()
    assert SECRET not in "".join(seen)
    assert SECRET not in caplog.text


def test_a_page_on_another_site_cannot_change_anything(project_root):
    """Catches: any website the person has open posting a form to this server to switch
    the game to a paid service or swap in its own key, or a rebinding hostname reaching it."""
    keys.save_key("gemini", "the-persons-own-key")
    with app_client(project_root) as client:
        same_origin = {"Origin": "http://127.0.0.1", "Sec-Fetch-Site": "same-origin"}
        for headers in (
            {"Origin": "http://evil.example"},
            {"Referer": "http://evil.example/page"},
            {"Sec-Fetch-Site": "cross-site"},
        ):
            chose = client.post(
                "/services/choose", data={"service": "gemini"}, headers=headers,
                follow_redirects=False,
            )
            assert chose.status_code == 403
            swapped = client.post(
                "/services/gemini/key", data={"key": "attackers-key"}, headers=headers,
                follow_redirects=False,
            )
            assert swapped.status_code == 403
        assert Store(project_root).load_project().service is None
        assert keys.find_key("gemini").key == "the-persons-own-key"

        allowed = client.post(
            "/services/choose", data={"service": "codex"}, headers=same_origin,
            follow_redirects=False,
        )
        assert allowed.status_code == 303
        assert Store(project_root).load_project().service == "codex"

    rebound = create_app(project_root, refiner=StubRefiner())
    with TestClient(rebound, base_url="http://evil.example") as far:
        assert far.get("/services").status_code == 400


def test_a_key_with_a_line_break_inside_is_not_used_and_never_reaches_a_page(
    project_root, monkeypatch
):
    """Catches: an environment key with an internal line break being used, so the HTTP
    library quotes it in an error that a page or the terminal then prints. It is skipped,
    and the row says which variable was refused."""
    monkeypatch.setenv("GEMINI_API_KEY", "FAKE-LEAKME-123\nPART2")
    with app_client(project_root) as client:
        pages = client.post("/services/gemini/test").text + client.get("/services").text
    assert "FAKE-LEAKME" not in pages
    assert "PART2" not in pages
    assert "GEMINI_API_KEY has a space" in pages
    assert keys.find_key("gemini") is None


def fake_codex(tmp_path, signed_in):
    """A stand-in `codex` script that appends a line to a log on every call."""
    log = tmp_path / "calls.log"
    script = tmp_path / "codex"
    script.write_text(f"#!/bin/sh\necho call >> {log}\nexit {0 if signed_in else 1}\n")
    script.chmod(0o755)
    return script, log


def test_page_loads_reuse_the_codex_answer_and_check_login_replaces_it(
    project_root, monkeypatch, tmp_path
):
    """Catches: every page load running (and, if Codex hangs, waiting on) `codex login
    status`; Check login showing a stale "signed in" after the person signed out; the
    old answer being used after the Codex command was changed."""
    script, log = fake_codex(tmp_path, signed_in=True)
    monkeypatch.setenv("PIXELMAGIC_CODEX_BIN", str(script))
    services.set_service_choice(Store(project_root), "codex")
    with app_client(project_root) as client:
        client.get("/services")
        client.get("/services")
        client.get("/directions/cosy-den/generate")
        assert len(log.read_text().splitlines()) == 1

        script.write_text(f"#!/bin/sh\necho call >> {log}\nexit 1\n")
        stale = client.get("/services").text
        assert "Signed in with your ChatGPT login" in stale  # still within the window

        checked = client.post("/services/codex/test")
        assert "Codex is not signed in" in checked.text
        assert "Signed in with your ChatGPT login" not in client.get("/services").text
        assert len(log.read_text().splitlines()) == 2

        (tmp_path / "other").mkdir()
        other, other_log = fake_codex(tmp_path / "other", signed_in=True)
        monkeypatch.setenv("PIXELMAGIC_CODEX_BIN", str(other))
        assert "Signed in with your ChatGPT login" in client.get("/services").text
        assert len(other_log.read_text().splitlines()) == 1


def test_a_key_test_redirects_with_a_code_and_the_next_page_shows_the_result(
    project_root, monkeypatch
):
    """Catches: reloading the page after Test key sending the test again (the answer was
    rendered straight from the POST), and the result being lost across the redirect or
    carried as text in the address."""
    server = http.server.HTTPServer(("127.0.0.1", 0), RefusingModels)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setitem(
        services.MODELS_URLS, "gemini", f"http://127.0.0.1:{server.server_port}/v1beta/models"
    )
    keys.save_key("gemini", SECRET)
    try:
        with app_client(project_root) as client:
            posted = client.post("/services/gemini/test", follow_redirects=False)
            assert posted.status_code == 303
            assert posted.headers["location"] == "/services?done=rejected&service=gemini#svc-gemini"
            page = client.get("/services?done=rejected&service=gemini")
            assert "did not accept this key" in page.text
            assert 'type="password"' in page.text  # a rejected keychain key offers Replace
            assert "did not accept" not in client.get("/services").text
            for unfit in ("codex", "fake", "nonsense"):
                wrong = client.get(f"/services?done=rejected&service={unfit}")
                assert "did not accept" not in wrong.text
    finally:
        server.shutdown()
        server.server_close()
