import io
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import BackgroundPolicy, GenerationRequest, PromptKind
from pixelmagic.providers.base import ServiceError
from pixelmagic.providers.gemini import (
    API_KEY_ENV,
    DEFAULT_MODEL,
    MAX_REFERENCES,
    GeminiImageService,
    aspect_ratio,
    build_config,
    build_contents,
    image_size_tier,
)

NOW = "2026-09-04T14:49:55+00:00"


def png_bytes(size=(1024, 1024)) -> bytes:
    """Gemini answers with encoded image bytes, so the fake must too."""
    from io import BytesIO

    buffer = BytesIO()
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (10, 20, 200)
    Image.fromarray(pixels, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def request(size=(1024, 1024), background=BackgroundPolicy.MAGENTA, model=None):
    return GenerationRequest(
        id="20260904-144955", created=NOW, subject="a wooden chair",
        kind=PromptKind.PROP, prompt="Draw a wooden chair.", service="gemini",
        model=model, size=size, background=background,
    )


def reply(data: bytes | None = None, *, finish_reason="STOP", text=None,
          block_reason=None):
    """One `generate_content` reply, shaped the way the SDK shapes it."""
    parts = []
    if data is not None:
        parts.append(
            SimpleNamespace(
                inline_data=SimpleNamespace(data=data, mime_type="image/png"),
                text=None,
            )
        )
    if text is not None:
        parts.append(SimpleNamespace(inline_data=None, text=text))
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason=finish_reason,
                content=SimpleNamespace(parts=parts),
            )
        ],
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
        text=text,
        response_id="resp-1",
    )


class FakeClient:
    """Stands in for `genai.Client`, recording every call it is handed."""

    def __init__(self, *replies):
        self.calls: list[dict] = []
        self._replies = list(replies)
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        answer = self._replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_transparency_is_not_advertised_because_gemini_has_none():
    """Verified 2026-09-04: no flag, no alpha, and the docs say to ask for white.

    This is the whole reason locked decision 6 exists — `choose_background`
    reads this and asks for magenta instead — so a hopeful True here would
    quietly break the key-colour fallback for every Gemini request.
    """
    capabilities = GeminiImageService(client=FakeClient()).capabilities()
    assert capabilities.transparent_background is False
    assert capabilities.reference_images is True
    assert capabilities.vision is True
    assert capabilities.sizes == []


@pytest.mark.parametrize(
    ("size", "expected"),
    [((1024, 1024), "1:1"), ((1536, 1024), "3:2"), ((1024, 1536), "2:3"),
     ((1920, 1080), "16:9"), ((1080, 1920), "9:16"), ((2100, 900), "21:9")],
)
def test_a_pixel_size_becomes_the_closest_ratio_gemini_offers(size, expected):
    assert aspect_ratio(size) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [((512, 512), "1K"), ((1024, 1024), "1K"), ((1536, 1024), "2K"),
     ((3840, 2160), "4K")],
)
def test_a_pixel_size_becomes_a_tier(size, expected):
    assert image_size_tier(size) == expected


def test_the_config_asks_for_an_image_and_nothing_else():
    config = build_config(request(size=(1536, 1024)))
    assert config["response_modalities"] == ["IMAGE"]
    assert config["image_config"] == {"aspect_ratio": "3:2", "image_size": "2K"}
    assert config["automatic_function_calling"] == {"disable": True}


def test_references_travel_inline_after_the_prompt(tmp_path):
    first = tmp_path / "a.png"
    first.write_bytes(png_bytes(size=(8, 8)))
    contents = build_contents("Draw a chair.", [first])
    assert contents[0] == "Draw a chair."
    assert contents[1]["inline_data"]["mime_type"] == "image/png"
    assert contents[1]["inline_data"]["data"] == first.read_bytes()


def test_a_missing_reference_file_is_a_service_error_naming_the_path(tmp_path):
    """`build_contents` reads reference bytes itself, so a lost file must not
    escape as a raw `OSError` — that is finding 1 of the guardian's PR #16
    review: it left a generation folder behind with no picture and no verdict.
    """
    missing = tmp_path / "nope.png"
    with pytest.raises(ServiceError) as caught:
        build_contents("x", [missing])
    assert caught.value.code == "REFERENCE_UNREADABLE"
    assert str(missing) in str(caught.value)


def test_generate_with_a_missing_reference_raises_a_service_error(tmp_path):
    """The same failure, reached through `generate`, must not write `out`."""
    missing = tmp_path / "nope.png"
    out = tmp_path / "image.png"
    client = FakeClient(reply(png_bytes()))
    with pytest.raises(ServiceError) as caught:
        GeminiImageService(client=client).generate(
            request(), out, reference_paths=[missing]
        )
    assert caught.value.code == "REFERENCE_UNREADABLE"
    assert not out.exists()


def test_a_drawn_picture_is_saved_and_recorded(tmp_path):
    client = FakeClient(reply(png_bytes()))
    out = tmp_path / "image.png"
    result = GeminiImageService(client=client).generate(request(), out)
    assert out.exists()
    assert result.service == "gemini"
    assert result.model == DEFAULT_MODEL
    assert result.response_id == "resp-1"
    assert client.calls[0]["model"] == DEFAULT_MODEL


def test_the_direction_can_name_a_different_model(tmp_path):
    client = FakeClient(reply(png_bytes()))
    GeminiImageService(client=client).generate(
        request(model="gemini-3-pro-image"), tmp_path / "image.png"
    )
    assert client.calls[0]["model"] == "gemini-3-pro-image"


def test_a_watermark_is_mentioned_on_every_result(tmp_path):
    """Every Gemini image carries SynthID and there is no opt-out."""
    result = GeminiImageService(client=FakeClient(reply(png_bytes()))).generate(
        request(), tmp_path / "image.png"
    )
    assert any("SynthID" in warning for warning in result.warnings)


def test_a_blocked_prompt_is_a_service_error_naming_the_reason(tmp_path):
    client = FakeClient(reply(None, finish_reason="STOP", block_reason="SAFETY"))
    with pytest.raises(ServiceError) as caught:
        GeminiImageService(client=client).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "SAFETY"


def test_a_reply_with_no_image_is_a_service_error(tmp_path):
    """`NO_IMAGE` is a 200 with words in it, which is the easiest case to miss."""
    client = FakeClient(reply(None, finish_reason="NO_IMAGE", text="I can't draw that."))
    with pytest.raises(ServiceError) as caught:
        GeminiImageService(client=client).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "NO_IMAGE"
    assert "I can't draw that." in str(caught.value)


def test_an_sdk_error_becomes_a_service_error(tmp_path):
    class ApiError(Exception):
        code = 429

    client = FakeClient(ApiError("resource exhausted"))
    with pytest.raises(ServiceError) as caught:
        GeminiImageService(client=client).generate(request(), tmp_path / "image.png")
    assert caught.value.service == "gemini"
    assert "resource exhausted" in str(caught.value)
    assert caught.value.code == "429"


def test_describe_asks_for_text_and_returns_it(tmp_path):
    picture = tmp_path / "cutout.png"
    picture.write_bytes(png_bytes(size=(8, 8)))
    client = FakeClient(reply(None, text="Wooden Chair"))
    answer = GeminiImageService(client=client).describe(picture, "What is this?")
    assert answer == "Wooden Chair"
    assert client.calls[0]["config"]["response_modalities"] == ["TEXT"]
    assert client.calls[0]["config"]["automatic_function_calling"] == {"disable": True}

def test_a_missing_key_is_a_service_error_naming_the_variable(monkeypatch):
    """The most common real failure, and the one that must not be a traceback.

    Note what this costs: `importorskip` means the test does not run at all
    unless the SDK is installed, so on the default offline suite — the run
    anybody actually does — the contract it defends has no coverage. The same
    is true of the OpenAI twin in Task 6. Leaving them ungated is worse (an
    `ImportError` in a suite that is meant to pass with no provider installed),
    so the gap is recorded here rather than closed.

    With no client injected the SDK is constructed for real, and with no key
    anywhere it raises a bare `ValueError` — which `stages.generate` does not
    catch, so the generation folder would be left behind, and which `cli.run`
    already reports as a *refiner* problem. Neither is acceptable, so the
    construction is wrapped.
    """
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Also the Vertex route: on a machine with application-default credentials
    # `genai.Client()` succeeds without either key, and `describe` would then
    # make a real call from what is supposed to be an offline test.
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    pytest.importorskip("google.genai")
    with pytest.raises(ServiceError) as caught:
        GeminiImageService().describe(None, "anything")
    assert caught.value.code == "NO_KEY"
    assert API_KEY_ENV in str(caught.value)


def test_a_client_that_will_not_build_is_not_mistaken_for_a_missing_key(monkeypatch):
    """A bad proxy or a malformed key is not the same problem as no key at all.

    Only the SDK's documented no-credentials failure is `ValueError`; anything
    else the constructor raises deserves its own code so a person is not sent
    off to set a variable that was already set.
    """
    genai = pytest.importorskip("google.genai")

    def _raise(*args, **kwargs):
        raise RuntimeError("proxy down")

    monkeypatch.setattr(genai, "Client", _raise)
    with pytest.raises(ServiceError) as caught:
        GeminiImageService().describe(None, "anything")
    assert caught.value.code == "NOT_STARTED"
    assert "proxy down" in str(caught.value)


def test_a_candidate_with_no_content_is_a_service_error(tmp_path):
    """A `MAX_TOKENS` finish is not a blocking reason, so it reaches the parts.

    Real candidates arrive with `content=None`, and reaching straight for
    `.content.parts` there is an `AttributeError` — not a `ServiceError`, so it
    escapes the failure contract and leaves the generation folder behind.
    """
    client = FakeClient(
        SimpleNamespace(
            candidates=[SimpleNamespace(content=None, finish_reason="MAX_TOKENS")],
            prompt_feedback=None,
            text=None,
            response_id="resp-2",
        )
    )
    with pytest.raises(ServiceError) as caught:
        GeminiImageService(client=client).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "NO_IMAGE"


def test_a_reply_that_is_not_a_picture_is_a_service_error(tmp_path):
    """Gemini's `inline_data` is not promised to be a PNG, or to be an image."""
    client = FakeClient(reply(b"this is not a picture"))
    with pytest.raises(ServiceError) as caught:
        GeminiImageService(client=client).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "UNREADABLE_IMAGE"


def test_a_webp_reply_is_saved_as_a_real_png(tmp_path):
    """The file is called image.png, so it has to be one."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 200)).save(buffer, format="WEBP")
    out = tmp_path / "image.png"
    GeminiImageService(client=FakeClient(reply(buffer.getvalue()))).generate(
        request(), out
    )
    with Image.open(out) as image:
        assert image.format == "PNG"


def test_a_size_that_is_not_what_was_asked_for_is_a_warning(tmp_path):
    """Gemini takes a ratio and a tier, so this is the normal case, not a fault."""
    result = GeminiImageService(client=FakeClient(reply(png_bytes(size=(64, 64))))).generate(
        request(size=(1024, 1024)), tmp_path / "image.png"
    )
    assert any("1024x1024" in warning and "64x64" in warning for warning in result.warnings)


def test_a_picture_too_large_to_decode_is_unreadable_not_a_crash(tmp_path, monkeypatch):
    """Guardian finding 2: `Image.DecompressionBombError` derives from
    `Exception`, not `OSError`, so it was uncaught. It stays `UNREADABLE_IMAGE`
    — the decode failed, so there is nothing to write yet.
    """
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    client = FakeClient(reply(png_bytes(size=(64, 64))))
    with pytest.raises(ServiceError) as caught:
        GeminiImageService(client=client).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "UNREADABLE_IMAGE"


@pytest.mark.skipif(
    os.name == "nt", reason="chmod-based read-only guard is not portable to Windows"
)
def test_a_directory_that_will_not_take_the_file_is_output_not_written(tmp_path):
    """Guardian finding 2: only the decode was guarded; a write failure — a
    full disk, a read-only directory — raised a bare `OSError` out of
    `generate` and orphaned the generation folder. This is the write half.
    """
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    out = readonly / "image.png"
    readonly.chmod(0o500)
    try:
        client = FakeClient(reply(png_bytes()))
        with pytest.raises(ServiceError) as caught:
            GeminiImageService(client=client).generate(request(), out)
        assert caught.value.code == "OUTPUT_NOT_WRITTEN"
        assert str(out) in str(caught.value)
    finally:
        readonly.chmod(0o700)


def test_more_than_the_reference_limit_keeps_the_newest_and_warns(tmp_path):
    """Guardian finding 3: Gemini's documented limit is 14 references. The
    fifteenth approved generation in a direction would otherwise send every
    later request over that limit with no warning to anyone.
    """
    paths = []
    for index in range(16):
        path = tmp_path / f"ref-{index}.png"
        pixels = np.full((2, 2, 3), index, dtype=np.uint8)
        Image.fromarray(pixels, "RGB").save(path)
        paths.append(path)
    client = FakeClient(reply(png_bytes()))
    result = GeminiImageService(client=client).generate(
        request(), tmp_path / "image.png", reference_paths=paths
    )
    contents = client.calls[0]["contents"]
    assert len(contents) == 1 + MAX_REFERENCES
    kept = [part["inline_data"]["data"] for part in contents[1:]]
    assert kept == [path.read_bytes() for path in paths[-MAX_REFERENCES:]]
    assert (
        "gemini takes at most 14 reference images; the 2 oldest of 16 were "
        "left out"
    ) in result.warnings


#: Set this, and a key, to run the two tests here that spend credits.
LIVE_ENV = "PIXELMAGIC_LIVE_GEMINI"


@pytest.mark.skipif(
    not (os.environ.get(LIVE_ENV) and
         (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))),
    reason=f"set {LIVE_ENV}=1 and a Gemini key to draw one real picture",
)
def test_gemini_really_draws_and_really_has_no_alpha(tmp_path):
    """Opt-in, and the only test here that spends anything.

    It is the gate on the call shape this module assumes — dict config, dict
    parts, raw bytes on `inline_data` — and on the negative fact everything
    else rests on: what comes back has no alpha channel, which is why
    `capabilities().transparent_background` is False.
    """
    out = tmp_path / "image.png"
    result = GeminiImageService().generate(
        request(size=(1024, 1024), background=BackgroundPolicy.MAGENTA), out
    )
    assert out.exists()
    assert result.model == DEFAULT_MODEL
    with Image.open(out) as image:
        assert image.mode in {"RGB", "RGBA"}
        if image.mode == "RGBA":
            # If this ever fires, Gemini grew transparency and the register,
            # the capability and this test all need revisiting.
            assert image.getchannel("A").getextrema()[0] == 255

@pytest.mark.skipif(
    not (os.environ.get(LIVE_ENV) and
         (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))),
    reason=f"set {LIVE_ENV}=1 and a Gemini key to draw one real picture",
)
def test_gemini_really_takes_a_reference_image(tmp_path):
    """Opt-in, and the shape the pipeline actually sends.

    `build_config` is proved by the Step 2 gate; the *reference part* is not,
    and after the first approved generation every request in a direction
    carries at least one, because `promote` adds one each time. So this sends
    a prompt and a reference together — the composed call — rather than the
    prompt alone, which is the only one a plain draw test exercises.
    """
    reference = tmp_path / "swatch.png"
    Image.new("RGB", (64, 64), (10, 20, 200)).save(reference)
    out = tmp_path / "image.png"
    GeminiImageService().generate(
        request(size=(1024, 1024), background=BackgroundPolicy.MAGENTA),
        out,
        reference_paths=[reference],
    )
    assert out.exists()
