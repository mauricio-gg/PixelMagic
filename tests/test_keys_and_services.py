"""The key lookup, the "test key" call and the per-game service choice."""

import logging
from datetime import UTC, datetime

import keyring
import pytest
from keyring.errors import KeyringError

from pixelmagic import keys, services
from pixelmagic.model import Project
from pixelmagic.store import Store

SECRET = "sk-test-SECRET-0123456789"


def test_the_environment_wins_over_a_saved_key(monkeypatch):
    keys.save_key("openai", "saved-in-keychain")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    found = keys.find_key("openai")
    assert found.key == SECRET
    assert found.source == keys.FROM_ENVIRONMENT
    assert found.variable == "OPENAI_API_KEY"


def test_a_blank_variable_is_not_a_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GOOGLE_API_KEY", SECRET)
    assert keys.find_key("gemini").variable == "GOOGLE_API_KEY"


def test_a_saved_key_is_found_in_the_keychain_and_gone_once_removed():
    keys.save_key("gemini", f"  {SECRET}\n")
    found = keys.find_key("gemini")
    assert (found.key, found.source) == (SECRET, keys.FROM_KEYCHAIN)
    keys.remove_key("gemini")
    assert keys.find_key("gemini") is None


def test_a_service_without_a_key_refuses_to_save_one():
    with pytest.raises(keys.KeyStoreError):
        keys.save_key("codex", SECRET)
    with pytest.raises(keys.KeyStoreError):
        keys.save_key("openai", "   ")


class BrokenKeychain(keyring.backend.KeyringBackend):
    """A keychain that always fails, like a headless Linux box with no backend."""

    priority = 1

    def get_password(self, service, username):
        raise KeyringError("no backend")

    def set_password(self, service, username, password):
        raise KeyringError("no backend")

    def delete_password(self, service, username):
        raise KeyringError("no backend")


def test_a_broken_keychain_is_a_sentence_and_the_key_is_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    keyring.set_keyring(BrokenKeychain())
    assert keys.find_key("openai") is None
    with pytest.raises(keys.KeyStoreError) as saved:
        keys.save_key("openai", SECRET)
    with pytest.raises(keys.KeyStoreError) as removed:
        keys.remove_key("openai")
    assert "openai" in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in str(saved.value) + str(removed.value)
    assert SECRET not in repr(keys.FoundKey(SECRET, keys.FROM_KEYCHAIN))


def test_a_missing_codex_command_is_not_ready_rather_than_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("PIXELMAGIC_CODEX_BIN", str(tmp_path / "no-such-codex"))
    status = services.service_status("codex")
    assert status.ready is False
    assert "not found" in status.sentence


def _project(root):
    """A project folder on disk, as `project new` leaves it."""
    store = Store(root)
    store.create_project("den", now=datetime(2026, 10, 9, tzinfo=UTC))
    return store


def test_override_beats_the_saved_choice_and_with_neither_nothing_is_chosen(tmp_path):
    store = _project(tmp_path)
    assert services.chosen_service(store) is None
    services.set_service_choice(store, "gemini")
    assert services.chosen_service(store) == "gemini"
    assert services.chosen_service(store, "codex") == "codex"
    assert "gemini" in store.project_file().read_text()


def test_an_unknown_service_cannot_be_chosen(tmp_path):
    store = _project(tmp_path)
    with pytest.raises(ValueError):
        services.set_service_choice(store, "nonsense")
    assert services.chosen_service(store) is None


def test_the_stand_in_cannot_be_chosen_for_a_game(tmp_path):
    """Catches: the stand-in being saved as a game's choice, so a person who never meant
    to use it gets flat shapes and a "Saved" row for a service that has no key."""
    store = _project(tmp_path)
    with pytest.raises(ValueError, match="--service fake"):
        services.set_service_choice(store, "fake")
    assert store.load_project().service is None
    assert "fake" not in [status.name for status in services.all_statuses()]


def test_a_project_file_from_before_the_choice_existed_still_loads(tmp_path):
    (tmp_path / "pixelmagic.json").write_text(
        '{"name": "old", "created": "2026-09-01T00:00:00Z", "default_direction": null}'
    )
    assert Store(tmp_path).load_project() == Project(
        name="old", created=datetime(2026, 9, 1, tzinfo=UTC)
    )


@pytest.mark.parametrize("saved", ["../evil --paid", "fake"])
def test_a_saved_value_that_cannot_be_chosen_draws_with_nothing(tmp_path, caplog, saved):
    """Catches: a shared project whose pixelmagic.json names something else (a typo, a
    value meant to reach a command, or the stand-in from an older project) being trusted
    or silently drawing with the stand-in instead of asking for a real choice."""
    store = _project(tmp_path)
    store.save_project(store.load_project().model_copy(update={"service": saved}))
    with caplog.at_level(logging.WARNING):
        assert services.chosen_service(store) is None
        with pytest.raises(services.NoServiceChosen):
            services.require_service(store)
    assert saved in caplog.text
