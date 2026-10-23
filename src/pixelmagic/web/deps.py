"""What every route needs: the project, the tools, and the safe way to render.

There is no session, no cookie and no state in memory between requests. A
`Workbench` is built once when the server starts and holds only what the
command line's flags would have held; everything else is read off disk on the
request that needs it. The one exception is `Workbench._options`, the settings
vocabulary, which is read once and kept for the life of the process because
reading it spawns a subprocess: install PixelRefiner while the page is running
and the settings screen goes on saying it is not installed until the server is
restarted. The core also keeps the last Codex login answer for 30 seconds
(`services.codex_status`), so page loads do not each wait on Codex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from pixelmagic import registry
from pixelmagic.model import Direction
from pixelmagic.naming import check_name
from pixelmagic.providers.base import ImageService
from pixelmagic.refiner.base import Refiner
from pixelmagic.segmenter.base import Segmenter
from pixelmagic.services import chosen_service, display_name, draw_phrase, require_service
from pixelmagic.store import Store
from pixelmagic.web.settings import OptionsSource, OptionsView, read_options

#: Jinja2 autoescapes `.html` by default, which is the whole reason it is here:
#: every page interpolates free text a person or a model wrote — a style
#: paragraph, a prompt, a rejection, a name a service invented.
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
#: How a template names a service: `display_name("fake")` is "Offline stand-in",
#: `draw_phrase("fake")` is "the offline stand-in". Both come from the core, so a
#: button that draws always names who is paid by the same words everywhere.
TEMPLATES.env.globals.update(display_name=display_name, draw_phrase=draw_phrase)


@dataclass
class Workbench:
    """The one project this server shows, and the tools it drives it with."""

    store: Store
    service_override: str | None
    refiner: Refiner
    #: Where the settings screen reads its knobs, or None when the real
    #: PixelRefiner is not turned on (the stand-in is doing the work).
    options_source: OptionsSource | None
    segmenter: Segmenter
    _options: OptionsView | None = field(default=None, repr=False)

    @property
    def service_name(self) -> str | None:
        """The service that draws: the flag, else the project's saved choice, else None."""
        return chosen_service(self.store, self.service_override)

    def service(self) -> ImageService:
        """Build the image service that draws now.

        Built per use rather than held: a service is cheap, and one built at
        startup would outlive a `registry` that has learned a new one. With
        nothing chosen it raises `NoServiceChosen`, so a form posted by hand
        cannot draw either.
        """
        return registry.load_service(require_service(self.store, self.service_override))

    def refiner_options(self) -> OptionsView:
        """The real refiner's settings vocabulary, read at most once.

        Reading it means spawning the refiner, so the answer is kept: a
        settings screen that shelled out on every page load would be slow for
        no reason, and the vocabulary cannot change while the server runs.
        With no source (the real refiner is not turned on) nothing is spawned
        and the view says so.
        """
        if self.options_source is None:
            return OptionsView(options=None, turned_off=True)
        if self._options is None:
            self._options = read_options(self.options_source)
        return self._options


def workbench(request: Request) -> Workbench:
    """The `Workbench` this app was built with, for FastAPI's `Depends`."""
    return request.app.state.workbench


def require_direction(bench: Workbench, name: str) -> Direction:
    """Load a direction named in a URL, refusing anything that is not a name.

    `check_name` first: a path parameter can still hold a percent-encoded `..`
    once Starlette has decoded it, and `Store` deliberately does not sanitise.
    A safe but unknown name raises `FileNotFoundError`, which the app turns
    into a 404 page.
    """
    return bench.store.load_direction(check_name(name, "direction"))


def page(
    request: Request, template: str, *, status_code: int = 200, **context: object
) -> Response:
    """Render one template. Jinja2 needs the request for `url_for`.

    `status_code` is for a page that re-shows a form with a refusal on it.
    """
    return TEMPLATES.TemplateResponse(
        request, template, context, status_code=status_code
    )


def see_other(url: str) -> RedirectResponse:
    """Send the browser to `url` with a GET, so a refresh cannot repeat a post."""
    return RedirectResponse(url, status_code=303)
