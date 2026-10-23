"""The web page: FastAPI, server-rendered HTML, two clicks' worth of script.

One of the three thin faces (locked decision 7). Every route builds a service,
an approval hook and a refiner and then calls the same stage functions the
command line calls; there is no pipeline logic in this package.

The scripts are small and live in `static/`: on the refine page a click on
a refined prop becomes its anchor, and the picking page's
picture has five listeners (click, mousedown, mousemove, mouseleave and wheel),
two on the document while a Lasso line is dragged (mousemove, mouseup) and one
on the Lasso button, plus one GET to `/pick/under`. Clicks and a dragged line are
posted in picture pixels; the mouse and wheel only choose which outline to show,
from what the core gave.
None decides anything — the anchor click fills in a form that is posted to a
route like every other one (and works without the script, typed by hand), and
the picking page only reads and posts what was clicked; the core decides what
it picks.

Three jobs are this file's own. The first is that nothing the core raises may
reach a browser as a traceback: a picture nobody approved, a refiner that is
not installed, a name that is already taken, a sidecar that will not parse —
each becomes a page with one sentence and a way back, the same job `cli.py`
does with one line and an exit code. The second is that the server listens on
this machine only. Remote and hosted use are out of scope for v1, the page
shows one person's project folder, and there is no login to put in front of it.
The third is that only this page may change anything: with no login, any site the
person has open could post a form to the loopback address and spend their keys,
so `LocalOnlyGuard` refuses a POST, PUT, PATCH or DELETE whose `Sec-Fetch-Site`,
`Origin` or `Referer` says another site sent it (403), and any request whose Host
name is not a loopback name, which is what DNS rebinding looks like (400). A
request that carries none of those headers (curl, the test client) is allowed.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from pixelmagic import registry
from pixelmagic.refiner.base import Refiner, RefinerError
from pixelmagic.refiner.cli_adapter import PixelRefinerCli
from pixelmagic.segmenter.base import Segmenter, SegmenterError
from pixelmagic.services import NoServiceChosen
from pixelmagic.stages.generate import ApprovalExhausted
from pixelmagic.stages.rebuild import RebuiltPieceRefused
from pixelmagic.stages.recipes import RecipeSeamBroken
from pixelmagic.store import AlreadyExists, Store
from pixelmagic.web import (
    cleaning,
    exporting,
    generating,
    images,
    picking,
    pixelcutting,
    projects,
    refining,
    services,
    surfaces,
)
from pixelmagic.web.deps import TEMPLATES, Workbench
from pixelmagic.web.settings import OptionsSource

LOGGER = logging.getLogger(__name__)

#: The only addresses the page will listen on.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

#: The page's own stylesheet, script, fonts and textures, shipped inside the
#: package. Nothing is loaded from another host, so the page works offline.
STATIC_DIR = Path(__file__).parent / "static"

#: Requests that change something. The cross-site guard looks at these only.
CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _host_name(host: str) -> str:
    """The name in a Host header, without its port (`[::1]:8000` is `::1`)."""
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _is_cross_site(headers: dict[str, str]) -> bool:
    """Whether a changing request came from another site than this page.

    `Sec-Fetch-Site`, when the browser sends it, must say same-origin (or
    none, a typed address). Otherwise `Origin`, else `Referer`, must name the
    request's own Host. A request with none of these (curl, a script, the test
    client) is not a browser's cross-site post and is allowed.
    """
    fetch_site = headers.get("sec-fetch-site")
    if fetch_site is not None:
        return fetch_site not in ("same-origin", "none")
    sender = headers.get("origin") or headers.get("referer")
    if sender is None:
        return False
    return urlsplit(sender).netloc.lower() != headers.get("host", "").lower()


class LocalOnlyGuard:
    """Refuse a request that did not come from this page on this machine.

    Two checks, both before any route runs. A Host header whose name is not a
    loopback name is a DNS-rebinding attempt and gets a 400. A changing request
    (`CHANGING_METHODS`) that is cross-site, so any website the person has open
    could otherwise post a form to this server, gets a 403. Both answer with the
    app's own error page.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Let the request through, or answer it with a sentence."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        refusal = None
        if _host_name(headers.get("host", "")) not in LOOPBACK_HOSTS:
            refusal = (400, "PixelMagic only answers on this machine's own address.")
        elif scope["method"] in CHANGING_METHODS and _is_cross_site(headers):
            refusal = (403, "PixelMagic only accepts changes from its own page.")
        if refusal is None:
            await self.app(scope, receive, send)
            return
        page = TEMPLATES.TemplateResponse(
            Request(scope),
            "error.html",
            {"headline": "PixelMagic cannot do that here", "detail": refusal[1]},
            status_code=refusal[0],
        )
        await page(scope, receive, send)


def create_app(
    root: Path,
    *,
    service: str | None = None,
    refiner: Refiner | None = None,
    options_source: OptionsSource | None = None,
    segmenter: Segmenter | None = None,
) -> FastAPI:
    """Build the web application for one project folder.

    `service` is the `--service` flag, or None to use the project's saved
    choice, read afresh on each request so a choice made on the Services page
    holds without a restart.

    `refiner` does the work and defaults to `registry.DEFAULT_REFINER`, which
    is the same one constant the command line's `--refiner` reads, so one
    change moves all six sites that hold a default at once.
    `options_source` is what the settings screen reads its knobs from. Left
    out, it is the work refiner itself when that is the real `PixelRefinerCli`,
    and nothing when the stand-in is: the screen then says the real
    PixelRefiner is not turned on rather than spawning a tool the person may
    not have, or offering the stand-in's answer, which would list keys the
    real tool does not have and hide the palettes it does (ticket M2.1).

    `segmenter` finds the objects on a picture and defaults to
    `registry.DEFAULT_SEGMENTER`. Building the real one starts nothing, so a
    missing helper model fails only the use of it, as a `SegmenterError`.

    `docs_url` and `redoc_url` are off: this is a page for one person, and an
    interactive API console is a second, confusing front door onto it.
    """
    app = FastAPI(title="PixelMagic", docs_url=None, redoc_url=None, debug=False)
    work_refiner = refiner or registry.load_refiner(registry.DEFAULT_REFINER)
    if options_source is None and isinstance(work_refiner, PixelRefinerCli):
        options_source = work_refiner
    app.state.workbench = Workbench(
        store=Store(root),
        service_override=service,
        refiner=work_refiner,
        options_source=options_source,
        segmenter=segmenter or registry.load_segmenter(registry.DEFAULT_SEGMENTER),
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(projects.router)
    app.include_router(images.router)
    app.include_router(generating.router)
    app.include_router(picking.router)
    app.include_router(pixelcutting.router)
    app.include_router(surfaces.router)
    app.include_router(cleaning.router)
    app.include_router(refining.router)
    app.include_router(exporting.router)
    app.include_router(services.router)
    _install_error_pages(app)
    app.add_middleware(LocalOnlyGuard)
    return app


def _error_page(
    status: int, headline: str, *, say_why: bool = True
) -> Callable[[Request, Exception], Awaitable[Response]]:
    """Build a handler that renders one failure as a page.

    `say_why` is off for the catch-all: an exception nobody planned for can
    carry anything in its message, and this is the one place where showing it
    would be showing an internal detail rather than an explanation.
    """

    async def render(request: Request, error: Exception) -> Response:
        if not say_why:
            LOGGER.exception("unhandled error on %s", request.url.path)
        return TEMPLATES.TemplateResponse(
            request,
            "error.html",
            {
                "headline": headline,
                "detail": (str(error) or type(error).__name__) if say_why else "",
            },
            status_code=status,
        )

    return render


async def _rebuilt_piece_page(request: Request, error: Exception) -> Response:
    """The "That will not work" page for an edit a rebuilt piece cannot take.

    The core's sentence (`RebuiltPieceRefused`) is kept as it is; this adds the
    page's own way forward. The only control that redraws one named pick is
    "Redraw just this one", which the picking page shows after a pick is
    adjusted with Add a spot or Remove a spot.
    """
    return await _error_page(400, "That will not work")(
        request,
        ValueError(
            f"{error}. On the picking page, adjust that pick with "
            "Add a spot or Remove a spot and press \"Redraw just this one\"."
        ),
    )


async def _recipe_seam_broken_page(request: Request, error: Exception) -> Response:
    """The page for a recipe whose own tiles failed the seam check.

    This is a bug in a recipe, not something a person typed, so it is a 500 and
    says so, naming the surface and the kind the exception carries. It is never
    the image-service sentence its `RuntimeError` parent would get.
    """
    LOGGER.error("recipe seam check failed on %s: %s", request.url.path, error)
    return await _error_page(500, "A recipe made a broken set of tiles")(
        request,
        ValueError(
            f"The {error.kind} recipe for the surface {error.surface!r} made a "
            "set of tiles that do not fit together. This is a bug in PixelMagic, "
            "not something you typed."
        ),
    )


async def _http_error_page(request: Request, error: Exception) -> Response:
    """The page for an unmatched URL, or any other plain HTTP status.

    Starlette answers an unmatched route itself, with JSON, before any of the
    handlers above can see it. This is the same page in the same shape, so
    somebody who mistypes a URL gets a sentence and a way back rather than
    `{"detail":"Not Found"}`. The status the exception carries is kept: a 404
    stays a 404. The headline branches on it, because the same handler answers
    a 405 on a mistyped method, and "PixelMagic has no page there" over a page
    that exists is a wrong sentence rather than a terse one.
    """
    status = getattr(error, "status_code", 500)
    headline = (
        "PixelMagic has no page there"
        if status == 404
        else "PixelMagic cannot do that here"
    )
    return TEMPLATES.TemplateResponse(
        request,
        "error.html",
        {"headline": headline, "detail": str(getattr(error, "detail", ""))},
        status_code=status,
    )


def _install_error_pages(app: FastAPI) -> None:
    """Turn every failure the core can raise into a page a person can read.

    Starlette looks a handler up by walking the exception's own class chain,
    so the specific entries win over the general ones: `AlreadyExists` before
    `FileExistsError`, `ValidationError` before `ValueError`, `RefinerError`,
    `SegmenterError`, `RecipeSeamBroken` and `ApprovalExhausted` before `RuntimeError`,
    `NoServiceChosen` before `ValueError`,
    everything before `Exception`. `ValidationError` gets its own page and a 500 because it
    means a file PixelMagic itself wrote will not read back — that is our bug,
    not the reader's mistake, and it should not be filed next to a typo.

    Three of these are here because of what a *real* image service does. A
    service that cannot draw raises `providers.base.ServiceError`, which is a
    bare `RuntimeError` subclass — nobody logged in, a filter refused the
    prompt, the connection dropped. Without the `RuntimeError` entry it would
    land on the `Exception` catch-all, which is registered with
    `say_why=False` *specifically to withhold the message*, and the person
    would get a blank "Something went wrong" for the one failure they could
    actually do something about. The entry is a family, not a special case for
    one class, and it needs no import from `providers/`.

    The other two are what FastAPI and Starlette answer with when nothing is
    registered: a form value that will not coerce is a `RequestValidationError`
    and comes back as a raw JSON blob, and an unmatched URL is a
    `StarletteHTTPException` and comes back as `{"detail":"Not Found"}`. Both
    are pages a person hits by hand — mistyping a size, editing a URL — so
    both get a page like everything else.
    """
    app.add_exception_handler(AlreadyExists, _error_page(409, "That name is taken"))
    app.add_exception_handler(
        FileNotFoundError, _error_page(404, "PixelMagic could not find that")
    )
    app.add_exception_handler(
        ApprovalExhausted, _error_page(400, "Nobody approved a picture")
    )
    app.add_exception_handler(
        RefinerError, _error_page(400, "PixelRefiner would not do that")
    )
    app.add_exception_handler(
        SegmenterError, _error_page(400, "The object finder could not do that")
    )
    app.add_exception_handler(
        ValidationError, _error_page(500, "A file PixelMagic wrote will not read back")
    )
    app.add_exception_handler(RebuiltPieceRefused, _rebuilt_piece_page)
    app.add_exception_handler(RecipeSeamBroken, _recipe_seam_broken_page)
    app.add_exception_handler(
        NoServiceChosen, _error_page(400, "No image service is chosen")
    )
    app.add_exception_handler(ValueError, _error_page(400, "That will not work"))
    app.add_exception_handler(OSError, _error_page(400, "A file could not be read"))
    app.add_exception_handler(
        RequestValidationError, _error_page(400, "That is not a value this page takes")
    )
    app.add_exception_handler(StarletteHTTPException, _http_error_page)
    app.add_exception_handler(
        RuntimeError, _error_page(400, "The image service could not draw that")
    )
    app.add_exception_handler(
        Exception, _error_page(500, "Something went wrong", say_why=False)
    )


def serve(app: FastAPI, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the page on this machine only.

    Remote and hosted use are out of scope for v1 and the page has no login,
    so a non-loopback host is refused here rather than quietly putting one
    person's project folder on the network.
    """
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"PixelMagic's web page is local-only; {host!r} is not one of "
            f"{', '.join(sorted(LOOPBACK_HOSTS))}"
        )
    uvicorn.run(app, host=host, port=port)
