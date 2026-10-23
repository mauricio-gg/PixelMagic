"""Fixtures and test doubles shared between the test modules."""

import os
from datetime import UTC, datetime
from pathlib import Path

import keyring
import numpy as np
import pytest
from keyring.backend import KeyringBackend
from PIL import Image

from pixelmagic import services
from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    Direction,
    RefineCandidate,
    RefineReport,
    RefinerSettings,
    Reviewer,
)
from pixelmagic.refiner.base import (
    EXIT_NOT_RUN,
    LOW_GRID_CONFIDENCE,
    RefineRequest,
    RefinerError,
    RefineResult,
    RefinerOptions,
)
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.store import Store


class MemoryKeyring(KeyringBackend):
    """A keychain that lives in a dict, so no test touches the real one."""

    priority = 1

    def __init__(self):
        """Start empty."""
        self.passwords = {}

    def get_password(self, service, username):
        """The saved password, or None."""
        return self.passwords.get((service, username))

    def set_password(self, service, username, password):
        """Save a password."""
        self.passwords[(service, username)] = password

    def delete_password(self, service, username):
        """Forget a password; the real backends raise when there is none."""
        if (service, username) not in self.passwords:
            raise keyring.errors.PasswordDeleteError("not found")
        del self.passwords[(service, username)]


@pytest.fixture(autouse=True)
def isolated_keys(monkeypatch):
    """Give every test an empty in-memory keychain and no API keys in the environment.

    No test may reach the real OS keychain or a key a developer exported. A
    live run (any `PIXELMAGIC_LIVE_*` variable set) keeps its real environment
    keys, since those tests exist to use them.
    """
    previous = keyring.get_keyring()
    memory = MemoryKeyring()
    keyring.set_keyring(memory)
    if not any(name.startswith("PIXELMAGIC_LIVE") for name in os.environ):
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    yield memory
    keyring.set_keyring(previous)


@pytest.fixture(autouse=True)
def forgotten_codex_check():
    """Start every test with no remembered Codex login answer, so tests stay independent."""
    services.forget_codex_status()
    yield
    services.forget_codex_status()


@pytest.fixture(autouse=True)
def loopback_test_clients(monkeypatch):
    """Make every `TestClient` talk to 127.0.0.1, as a browser would.

    The app refuses a Host that is not a loopback name (`LocalOnlyGuard`), and
    the test client's own default is "testserver". A test that wants another
    host passes `base_url` itself.
    """
    from fastapi.testclient import TestClient

    original = TestClient.__init__

    def init(self, app, *args, **kwargs):
        kwargs.setdefault("base_url", "http://127.0.0.1")
        original(self, app, *args, **kwargs)

    monkeypatch.setattr(TestClient, "__init__", init)


@pytest.fixture
def write_asset():
    """Return a helper that stages one finished asset under a direction."""

    def _write(store, name, kind: AssetKind, *, size=(8, 8), cell=None):
        folder = store.asset_dir("cosy-den", name)
        folder.mkdir(parents=True, exist_ok=True)
        pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        pixels[1:-1, 1:-1] = (10, 20, 200, 255)
        Image.fromarray(pixels, "RGBA").save(folder / "source.png")
        Image.fromarray(pixels, "RGBA").save(folder / "refined.png")
        store.write_model(
            folder / "asset.json",
            AssetRecord(
                name=name, kind=kind, source_generation="20260903-191233",
                source_prompt="draw a wooden chair", service="fake", cell=cell,
            ),
        )
        store.write_model(
            folder / "refine.json",
            RefineReport(
                asset=name, route="preserve", confidence=1.0, refiner="stub",
                palette=["#0a14c8"], width=size[0], height=size[1], color_count=1,
                requested=RefinerSettings(preset="crisp-sprite"),
            ),
        )

    return _write


WEB_NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


class CannedOptions:
    """Stands in for the real refiner's `options()` on the settings screen.

    Shaped like what `PixelRefinerCli.options()` returns for the reply recorded
    on 2026-09-04 — palettes the stand-in refiner does not have at all — which
    is how a test can tell which refiner the settings screen actually read.
    """

    name = "canned"

    def options(self) -> RefinerOptions:
        """Answer with a small but real-shaped options block."""
        return RefinerOptions(
            presets=["auto", "crisp-sprite"],
            palettes=["gb_legacy", "pico8"],
            quick={
                "dithering": {
                    "key": "dithering",
                    "values": ["off", "subtle", "strong"],
                    "default": "off",
                },
                "backgroundColor": {"key": "backgroundColor", "default": "#ff00ff"},
            },
            advanced={
                "colorCount": {
                    "key": "colorCount", "kind": "int", "min": 2, "max": 256, "default": 32,
                }
            },
        )


class BrokenOptions:
    """An options source that cannot answer, the way an uninstalled CLI cannot."""

    name = "broken"

    def options(self) -> RefinerOptions:
        """Fail the way `PixelRefinerCli` fails when node is not installed."""
        raise RefinerError(
            "pixel-refiner is not installed", exit_code=EXIT_NOT_RUN, code=None
        )


@pytest.fixture
def canned_options():
    """The real-shaped options block, for a test that builds its own app."""
    return CannedOptions()


@pytest.fixture
def broken_options():
    """An options source that raises, for the "not installed" screen."""
    return BrokenOptions()


class CandidateRefiner(StubRefiner):
    """A stub that is unsure of the grid and offers two ways out.

    The stand-in cannot detect a grid at all, so a test that needs the
    candidate flow has to say what a real reply looks like. It also remembers
    which candidate it was asked for, which is the thing under test.

    It lives here rather than in a test module because two modules use it:
    `tests/test_refine.py` for the stage and `tests/test_web_refine.py` for the
    re-pick button on the refine screen.
    """

    def __init__(self):
        """Start with nothing asked for."""
        self.asked: list[str | None] = []

    def refine(self, request: RefineRequest) -> RefineResult:
        """Refine as the stand-in would, then offer two grids and remember."""
        self.asked.append(request.candidate_id)
        result = super().refine(request)
        result.warnings = [LOW_GRID_CONFIDENCE]
        result.candidates = [
            RefineCandidate(
                id="auto-result:40x22", kind="auto-result", recommended=True,
                processing_mode="refine", out_w=40, out_h=22,
            ),
            RefineCandidate(
                id="grid:20x11", kind="grid", processing_mode="refine",
                out_w=20, out_h=11,
            ),
        ]
        return result


@pytest.fixture
def candidate_refiner():
    """One `CandidateRefiner`, for a test that wants to read `.asked`."""
    return CandidateRefiner()


@pytest.fixture
def project_root(tmp_path):
    """A project folder with one direction, ready for a face to drive."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=WEB_NOW)
    store.create_direction(
        Direction(
            name="cosy-den",
            created=WEB_NOW,
            style="Warm SNES interiors, soft lamp light, chunky outlines.",
            donts=["no gradients"],
        )
    )
    project = store.load_project()
    project.default_direction = "cosy-den"
    store.save_project(project)
    return Path(tmp_path)


@pytest.fixture
def client(project_root):
    """A TestClient over that project, offline: fake service, stand-in refiner."""
    from fastapi.testclient import TestClient

    from pixelmagic.web.app import create_app

    application = create_app(
        project_root,
        service="fake",
        refiner=StubRefiner(),
        options_source=CannedOptions(),
        segmenter=StubSegmenter(),
    )
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def cut_props(client, object_picture):
    """Pick three objects on a picture of separate ones and redraw them into rebuilt props.

    The same road the page's picking screen takes, through the core with the
    fake service and the stub segmenter: these tests are about what happens to
    pieces once they exist. Returns the cut-up run's id.
    """
    from pixelmagic.providers.fake import FakeImageService
    from pixelmagic.stages.generate import import_picture
    from pixelmagic.stages.pick import pick_suggestions
    from pixelmagic.stages.redraw import redraw_picks
    from pixelmagic.stages.suggest import suggest_objects

    def cut(store, direction_name="cosy-den"):
        direction = store.load_direction(direction_name)
        record = import_picture(store, direction, object_picture, reviewer=Reviewer.HUMAN)
        made = suggest_objects(store, direction, record, StubSegmenter())
        pick_suggestions(store, direction, made.id, [1, 2, 3])
        redraw_picks(store, direction, made.id, service=FakeImageService())
        return made.id

    return cut


@pytest.fixture
def anyio_backend():
    """anyio's pytest plugin drives the async MCP tests; asyncio is enough."""
    return "asyncio"


@pytest.fixture
async def agent(project_root):
    """An in-process MCP client talking to a server over that project folder.

    No subprocess: `Client` takes the `MCPServer` object and wires an
    in-memory transport to it, so a test is as fast and as debuggable as any
    other call.
    """
    from mcp import Client

    from pixelmagic.mcp_server import create_server

    server = create_server(
        project_root, service="fake", refiner=StubRefiner(), segmenter=StubSegmenter()
    )
    async with Client(server) as connected:
        yield connected


@pytest.fixture
def object_picture(tmp_path):
    """A drawn picture of eight separate opaque objects on transparency, as a PNG path.

    The objects differ in size, and two of them (the first two) sit six pixels
    apart with nothing joining them, so a stand-in segmenter that cuts by
    connected blob still finds eight objects.
    """
    picture = Image.new("RGBA", (480, 320), (0, 0, 0, 0))
    boxes = [
        (10, 10, 50, 40), (56, 10, 90, 44),
        (140, 10, 200, 70), (260, 10, 300, 100), (380, 10, 470, 60),
        (10, 150, 44, 200), (120, 160, 220, 300), (300, 180, 340, 210),
    ]
    for index, (left, top, right, bottom) in enumerate(boxes):
        picture.paste((40 + index * 25, 90, 200 - index * 15, 255), (left, top, right, bottom))
    path = tmp_path / "objects.png"
    picture.save(path)
    return path
