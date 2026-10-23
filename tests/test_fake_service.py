from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from pixelmagic.model import BackgroundPolicy, GenerationRequest, PromptKind
from pixelmagic.providers.base import load_template
from pixelmagic.providers.fake import FakeImageService

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


def request(kind: PromptKind, background: BackgroundPolicy, size=(64, 64)) -> GenerationRequest:
    return GenerationRequest(
        id="20260903-191233",
        created=NOW,
        subject="a wooden chair",
        kind=kind,
        prompt="draw a wooden chair",
        service="fake",
        size=size,
        background=background,
    )


def test_generate_writes_a_png_of_the_requested_size(tmp_path):
    out = tmp_path / "image.png"
    result = FakeImageService().generate(
        request(PromptKind.PROP, BackgroundPolicy.TRANSPARENT), out
    )
    assert out.exists()
    assert Image.open(out).size == (64, 64)
    assert result.service == "fake"
    assert result.id == "20260903-191233"


def test_a_transparent_request_comes_back_with_alpha(tmp_path):
    out = tmp_path / "image.png"
    FakeImageService().generate(request(PromptKind.PROP, BackgroundPolicy.TRANSPARENT), out)
    alpha = np.asarray(Image.open(out).convert("RGBA"))[:, :, 3]
    assert alpha[0, 0] == 0
    assert alpha.max() == 255


def test_a_magenta_request_comes_back_on_magenta(tmp_path):
    out = tmp_path / "image.png"
    FakeImageService().generate(request(PromptKind.PROP, BackgroundPolicy.MAGENTA), out)
    assert Image.open(out).convert("RGB").getpixel((0, 0)) == (255, 0, 255)


def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    service = FakeImageService()
    service.generate(request(PromptKind.PROP, BackgroundPolicy.MAGENTA), a)
    service.generate(request(PromptKind.PROP, BackgroundPolicy.MAGENTA), b)
    assert a.read_bytes() == b.read_bytes()


def test_describe_cycles_through_the_supplied_labels(tmp_path):
    service = FakeImageService(descriptions=["Wooden Chair", "Round Rug"])
    assert service.describe(Path("x.png"), "name it") == "Wooden Chair"
    assert service.describe(Path("y.png"), "name it") == "Round Rug"
    assert service.describe(Path("z.png"), "name it") == "Wooden Chair"


def test_describe_answers_text_only_questions_too():
    service = FakeImageService(descriptions=["Cosy Den"])
    assert service.describe(None, "name this style") == "Cosy Den"


def test_every_prompt_kind_that_draws_has_a_template():
    """`PARTS_SHEET`, `TILESET_SHEET`, `PATCH_SHEET` and `SURFACE_BLOCK` are left out: they
    only exist so an old request.json still loads, and nothing draws with them any more."""
    legacy = {
        PromptKind.PARTS_SHEET, PromptKind.TILESET_SHEET, PromptKind.PATCH_SHEET,
        PromptKind.SURFACE_BLOCK,
    }
    for kind in set(PromptKind) - legacy:
        assert "{subject}" in load_template(kind)
