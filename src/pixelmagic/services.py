"""What the Services page and every face need to know about the image services.

Four things live here, all in the core so that no face decides any of them:
the display names, a status per service (is it ready, and why), a free "test
key" check that draws nothing, and the per-game choice of which service draws.
The choice is stored in the project's own `pixelmagic.json` and holds only a
service name, never a secret. There is no default service: when nothing is
chosen, nothing draws (`require_service`). The offline stand-in is registered
but is not choosable; it is reached only by `--service fake`.
"""

from __future__ import annotations

import logging
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from pixelmagic import keys, registry
from pixelmagic.providers import codex
from pixelmagic.store import Store

LOGGER = logging.getLogger(__name__)

#: What a face shows for each service name, in the order the page lists them.
DISPLAY_NAMES: dict[str, str] = {
    "codex": "Codex",
    "gemini": "Gemini",
    "openai": "OpenAI Images",
    "fake": "Offline stand-in",
}

#: How a sentence names who draws, where the display name alone reads badly.
DRAWER_PHRASES: dict[str, str] = {"fake": "the offline stand-in"}

#: The services a person may choose for a game, in the order the page lists them.
#: The offline stand-in is not here: only `--service fake` reaches it.
CHOOSABLE: tuple[str, ...] = ("codex", "gemini", "openai")

#: The one sentence every face shows when something needs to draw and nothing is chosen.
NO_SERVICE_SENTENCE = (
    "No image service is chosen. Choose one on the Services page, or with --service."
)

#: Seconds to wait for `codex login status`, and for a models-list call.
CODEX_CHECK_TIMEOUT = 10
KEY_CHECK_TIMEOUT = 15

#: Seconds a Codex login answer is reused by page loads, so a hanging Codex
#: costs one stall per window rather than one per load.
CODEX_CACHE_SECONDS = 30

#: The free models-list call per service, and how its key is sent. Never in the URL.
MODELS_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
    "openai": "https://api.openai.com/v1/models",
}

#: Outcomes of `check_key`.
ACCEPTED = "accepted"
REJECTED = "rejected"
NO_KEY = "no_key"
UNREACHABLE = "unreachable"


class NoServiceChosen(ValueError):
    """Something needs to draw and neither `--service` nor the project names a service.

    A `ValueError`, so a face that already turns `ValueError` into a sentence
    shows `NO_SERVICE_SENTENCE` unchanged.
    """

    def __init__(self) -> None:
        """Carry `NO_SERVICE_SENTENCE` and nothing else."""
        super().__init__(NO_SERVICE_SENTENCE)


@dataclass(frozen=True)
class ServiceStatus:
    """One service as the Services page shows it."""

    name: str
    display_name: str
    ready: bool
    #: One honest sentence about the state, never containing a key.
    sentence: str
    #: For a key service with a key: `keys.FROM_ENVIRONMENT` or `keys.FROM_KEYCHAIN`.
    key_source: str | None = None
    #: The environment variable the key was read from, when it came from one.
    key_variable: str | None = None
    #: For a service that is not ready: the sentence the Start page shows, which
    #: sends the person to Services. None when it is ready.
    nudge: str | None = None


@dataclass(frozen=True)
class KeyCheck:
    """The outcome of a test-key call; `detail` is set only for `UNREACHABLE`."""

    outcome: str
    #: For `UNREACHABLE`: the HTTP code or network reason, for a log. Never the key.
    detail: str | None = None


#: The last Codex check: (command it ran, `time.monotonic()` when, its status).
#: The one short-lived piece of memory in the core; see `codex_status`.
_codex_last: tuple[list[str], float, ServiceStatus] | None = None


def forget_codex_status() -> None:
    """Drop the remembered Codex answer, so the next `codex_status` runs the command."""
    global _codex_last
    _codex_last = None


def codex_status(*, fresh: bool = False) -> ServiceStatus:
    """Whether the Codex command-line tool is signed in.

    The last answer, whatever it was (signed in, not signed in, not found, did
    not answer), is reused for `CODEX_CACHE_SECONDS` so page loads do not each
    wait on Codex. It applies only to the same command: changing
    `PIXELMAGIC_CODEX_BIN` makes the old answer not count. `fresh=True` (the
    "Check login" button) always runs the command and replaces the remembered
    answer, so the page only ever shows the most recent check.
    """
    global _codex_last
    command = codex.default_command()
    if not fresh and _codex_last is not None:
        last_command, when, last_status = _codex_last
        if last_command == command and time.monotonic() - when < CODEX_CACHE_SECONDS:
            return last_status
    status = _run_codex_check(command)
    _codex_last = (command, time.monotonic(), status)
    return status


def _run_codex_check(command: list[str]) -> ServiceStatus:
    """Run `codex login status` (exit 0 means signed in); offline and free.

    It inherits `CODEX_HOME` as the provider does.
    """
    name = DISPLAY_NAMES["codex"]
    try:
        done = subprocess.run(
            [*command, "login", "status"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=CODEX_CHECK_TIMEOUT,
            check=False,
        )
    except OSError:
        return ServiceStatus(
            "codex",
            name,
            False,
            f"The Codex command was not found. Install Codex, or set {codex.COMMAND_ENV}.",
            nudge="Codex was not found. Fix it in Services.",
        )
    except subprocess.TimeoutExpired:
        return ServiceStatus(
            "codex",
            name,
            False,
            f"Codex did not answer within {CODEX_CHECK_TIMEOUT} seconds. Try Check login again.",
            nudge="Codex did not answer. Check it in Services.",
        )
    if done.returncode == 0:
        return ServiceStatus("codex", name, True, "Signed in with your ChatGPT login.")
    return ServiceStatus(
        "codex",
        name,
        False,
        "Codex is not signed in. Run `codex login` in a terminal.",
        nudge="Codex is not signed in. Fix it in Services.",
    )


def _key_status(service: str) -> ServiceStatus:
    """Ready when a key is found; says where it came from."""
    name = DISPLAY_NAMES[service]
    found = keys.find_key(service)
    if found is None:
        refused = keys.refused_variable(service)
        sentence = (
            f"The key in {refused} has a space, line break or unusual character "
            "inside it, so it is not used."
            if refused
            else "No key yet."
        )
        return ServiceStatus(
            service, name, False, sentence, nudge=f"{name} needs a key. Add it in Services."
        )
    if found.source == keys.FROM_ENVIRONMENT:
        sentence = f"Key found in the environment variable {found.variable}."
    else:
        sentence = "Key saved in your keychain."
    return ServiceStatus(service, name, True, sentence, found.source, found.variable)


def service_status(service: str) -> ServiceStatus:
    """The status of one service by name; `ValueError` for a name nothing offers."""
    if service == "codex":
        return codex_status()
    if service in keys.ENV_VARIABLES:
        return _key_status(service)
    if service == "fake":
        return ServiceStatus("fake", DISPLAY_NAMES["fake"], True, "Always ready.")
    raise ValueError(f"unknown service {service!r}")


def display_name(service: str) -> str:
    """What a face calls a service; a name nothing offers is shown as it was typed."""
    return DISPLAY_NAMES.get(service, service)


def draw_phrase(service: str) -> str:
    """Who to name on a button or in a sentence: "Gemini", "the offline stand-in"."""
    return DRAWER_PHRASES.get(service, display_name(service))


def all_statuses() -> list[ServiceStatus]:
    """The status of every service a person may choose, in the order the page lists them."""
    return [service_status(name) for name in CHOOSABLE]


def check_key(service: str) -> KeyCheck:
    """Test the service's key with its free models-list call, which draws nothing.

    Returns only an outcome; the faces word it. Never raises for a bad key or a
    bad network, and nothing returned ever contains the key. A key with a line
    break or a character a header cannot carry makes the request itself refuse
    to build or send (a `ValueError`, whose message would hold the key), so it
    is reported as rejected without its message.
    """
    if service not in MODELS_URLS:
        raise ValueError(f"the {service} service has no key to test")
    found = keys.find_key(service)
    if found is None:
        return KeyCheck(NO_KEY)
    if service == "gemini":
        headers = {"x-goog-api-key": found.key}
    else:
        headers = {"Authorization": f"Bearer {found.key}"}
    try:
        request = urllib.request.Request(MODELS_URLS[service], headers=headers)
        with urllib.request.urlopen(request, timeout=KEY_CHECK_TIMEOUT):
            pass
    except ValueError:
        # Deliberately not `as error`: the message would carry the key.
        return KeyCheck(REJECTED)
    except urllib.error.HTTPError as error:
        # Gemini answers an invalid key with 400; both answer a refused one with 401/403.
        if error.code in (401, 403) or (service == "gemini" and error.code == 400):
            return KeyCheck(REJECTED)
        return KeyCheck(UNREACHABLE, f"answered with an error (HTTP {error.code})")
    except (urllib.error.URLError, OSError) as error:
        return KeyCheck(UNREACHABLE, f"could not be reached ({getattr(error, 'reason', error)})")
    return KeyCheck(ACCEPTED)


def set_service_choice(store: Store, service: str) -> None:
    """Save which service draws for this game, in the project's `pixelmagic.json`.

    Refuses a name that is not in `CHOOSABLE` (the offline stand-in included:
    it is reached only by `--service fake`), and a folder with no project yet
    (there is no game to choose for). Only the name is stored.
    """
    if service not in CHOOSABLE:
        raise registry.UnknownName(
            f"{service!r} is not a service you can choose; choose one of: "
            f"{', '.join(CHOOSABLE)}. The offline stand-in is only reached with --service fake."
        )
    try:
        project = store.load_project()
    except FileNotFoundError:
        raise ValueError(
            "there is no game in this folder yet, so there is nothing to choose for; "
            "run `pixelmagic project new` first"
        ) from None
    store.save_project(project.model_copy(update={"service": service}))


def chosen_service(store: Store, override: str | None = None) -> str | None:
    """Which service draws: the override, else the project's saved choice, else None.

    The override is the person's own `--service` flag and is returned as given
    (`registry.load_service` refuses a bad one). The saved choice comes from a
    project folder that may have been shared or downloaded, so it is trusted
    only when it is exactly a name in `CHOOSABLE`; anything else (unknown text,
    or `fake` from an older project) gives None and logs a warning naming the
    bad value, so a file in a project can never point the app at a service by
    accident. A folder with no project yet, or no saved choice, gives None.
    """
    if override:
        return override
    try:
        saved = store.load_project().service
    except FileNotFoundError:
        return None
    if saved is None:
        return None
    if saved not in CHOOSABLE:
        LOGGER.warning(
            "the project's saved service %r is not a service that can be chosen; "
            "nothing will draw until one is chosen",
            saved,
        )
        return None
    return saved


def require_service(store: Store, override: str | None = None) -> str:
    """The service that draws now, or `NoServiceChosen` when there is none.

    What every face calls just before it draws, so none has its own check.
    """
    name = chosen_service(store, override)
    if name is None:
        raise NoServiceChosen
    return name
