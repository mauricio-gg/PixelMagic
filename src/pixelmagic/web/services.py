"""The Services page: who draws for this game, and the keys they need.

A thin face over `pixelmagic.services` and `pixelmagic.keys`: every rule (what
counts as ready, the strict choice, where a key is kept, the free key test)
lives there. This module only renders those answers and posts to them.

A key travels one way. It is posted to a route, handed to `keys.save_key` and
forgotten: no page, redirect, log line or error ever carries it, and the field
is never filled in. A redirect after a post carries a short fixed code, never
text: the sentence is looked up here from `RESULTS`, so nothing in a URL is
ever shown.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import Response

from pixelmagic import keys, services
from pixelmagic.web.deps import Workbench, page, see_other, workbench

LOGGER = logging.getLogger(__name__)

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]

#: What the person is told under each service. Fixed text, written by us.
TAGS: dict[str, str] = {
    "codex": "Uses your ChatGPT plan. No key needed.",
    "gemini": "Google's image service. Draws from your Google credits.",
    "openai": "Billed per picture to your OpenAI account.",
}

#: Where to make a key, per key-taking service: (link text, address). Fixed here;
#: nothing from a model or a project folder ever supplies a link.
KEY_LINKS: dict[str, tuple[str, str]] = {
    "gemini": ("Make a key in Google AI Studio", "https://aistudio.google.com/apikey"),
    "openai": ("Make a key on the OpenAI platform site", "https://platform.openai.com/api-keys"),
}

#: The sentence for each result code a redirect may carry, and whether it is good news.
#: `{name}` is the service's display name, `{game}` the project's name.
RESULTS: dict[str, tuple[str, str]] = {
    "saved": ("ok", "Saved to your keychain. Press Test key to check it."),
    "removed": ("ok", "Key removed from your keychain."),
    "chosen": ("ok", "Now drawing with {name} for {game}."),
    # The outcomes of Test key, by the codes `services.check_key` returns. The
    # core's detail for "unreachable" is a reason a code cannot carry, so the
    # route logs it and this fixed sentence is shown.
    services.ACCEPTED: (
        "ok",
        "{name} accepted the key. Nothing was drawn and nothing was charged.",
    ),
    services.REJECTED: (
        "err",
        "{name} did not accept this key. Check that you copied all of it, "
        "with no spaces, or paste a new one.",
    ),
    services.NO_KEY: ("err", "There is no {name} key yet. Paste one first."),
    services.UNREACHABLE: (
        "err",
        "{name} could not be reached or answered with an error. "
        "Nothing was saved or changed.",
    ),
}

#: The codes Test key redirects with: the outcomes of `services.check_key`.
TEST_OUTCOMES = (services.ACCEPTED, services.REJECTED, services.NO_KEY, services.UNREACHABLE)

#: The code after Check login. Its sentence is the row's own fresh status, not a table entry.
LOGIN_CHECKED = "checked"


def _known(name: str) -> str:
    """The service name from a URL, or a 404 page when nothing offers it."""
    if name not in services.DISPLAY_NAMES:
        raise HTTPException(status_code=404, detail=f"there is no service called {name!r}")
    return name


def _row(
    status: services.ServiceStatus,
    *,
    chosen: str | None,
    result: tuple[str, str] | None,
    rejected: bool,
    offer_choose: bool,
) -> dict[str, object]:
    """Everything the template needs to draw one service's row."""
    name = status.name
    takes_key = name in keys.ENV_VARIABLES
    from_keychain = status.key_source == keys.FROM_KEYCHAIN
    has_key = status.key_source is not None
    return {
        "name": name,
        "display_name": status.display_name,
        "tag": TAGS[name],
        "link": KEY_LINKS.get(name),
        "chosen": name == chosen,
        "status": status,
        "takes_key": takes_key,
        "login": name == "codex",
        # Never colour alone: the template pairs this with an icon and the sentence.
        "state": "err" if rejected else "ok" if status.ready else "warn",
        "rejected": rejected,
        "show_field": takes_key and (not has_key or (rejected and from_keychain)),
        "can_replace": takes_key and from_keychain and not rejected,
        "can_remove": takes_key and from_keychain,
        "can_test": takes_key and has_key,
        "result": result,
        "offer_choose": offer_choose,
    }


def _render(
    request: Request,
    bench: Workbench,
    *,
    done: str | None = None,
    done_for: str | None = None,
    status_code: int = 200,
) -> Response:
    """Draw the page.

    `done` is a code from `RESULTS` (or `LOGIN_CHECKED`) and `done_for` the
    service it concerns; a code for a different kind of service shows nothing.
    """
    try:
        project = bench.store.load_project()
    except FileNotFoundError:
        project = None
    chosen = bench.service_name
    rows = []
    for status in services.all_statuses():
        result = None
        rejected = False
        offer_choose = False
        takes_key = status.name in keys.ENV_VARIABLES
        is_test = done in TEST_OUTCOMES
        if done == LOGIN_CHECKED and done_for == status.name == "codex":
            result = (
                ("ok", "Codex is signed in and ready to draw.")
                if status.ready
                else ("err", status.sentence)
            )
        elif done in RESULTS and done_for == status.name and (takes_key or not is_test):
            kind, text = RESULTS[done]
            game = project.name if project else ""
            result = (kind, text.format(name=status.display_name, game=game))
            offer_choose = done == "saved" and project is not None and status.name != chosen
            rejected = done == services.REJECTED
        rows.append(
            _row(
                status,
                chosen=chosen,
                result=result,
                rejected=rejected,
                offer_choose=offer_choose,
            )
        )
    return page(
        request,
        "services.html",
        status_code=status_code,
        project=project,
        crumb="Services",
        rows=rows,
        override=bench.service_override,
    )


@router.get("/services")
def show(
    request: Request, bench: Bench, done: str | None = None, service: str | None = None
) -> Response:
    """The page. `done` and `service` come from a redirect and are only looked up, never shown."""
    return _render(request, bench, done=done, done_for=service)


@router.post("/services/choose")
def choose(bench: Bench, service: Annotated[str, Form()]) -> Response:
    """Save which service draws for this game. The refusals are the core's."""
    services.set_service_choice(bench.store, service)
    return see_other(f"/services?done=chosen&service={service}#svc-{service}")


@router.post("/services/{name}/key")
def save_key(name: str, key: Annotated[str, Form()]) -> Response:
    """Keep a pasted key in the keychain and go back to the page. The key is never echoed."""
    keys.save_key(_known(name), key)
    return see_other(f"/services?done=saved&service={name}#svc-{name}")


@router.post("/services/{name}/key/remove")
def remove_key(name: str) -> Response:
    """Remove the key from the keychain and go back to the page."""
    keys.remove_key(_known(name))
    return see_other(f"/services?done=removed&service={name}#svc-{name}")


@router.post("/services/{name}/test")
def test(name: str) -> Response:
    """Test the key (free) or check the login, then redirect so a reload does not test again.

    The redirect carries only a short fixed code. The core's detail for an
    unreachable service (a reason or HTTP code, never the key) goes to the log
    and the page shows a fixed sentence.
    """
    _known(name)
    if name == "codex":
        services.codex_status(fresh=True)
        code = LOGIN_CHECKED
    else:
        check = services.check_key(name)
        code = check.outcome
        if code == services.UNREACHABLE:
            LOGGER.warning("testing the %s key: %s", name, check.detail)
    return see_other(f"/services?done={code}&service={name}#svc-{name}")
