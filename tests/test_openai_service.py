import base64
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import BackgroundPolicy, GenerationRequest, PromptKind
from pixelmagic.providers.base import ServiceError
from pixelmagic.providers.openai_images import (
    API_KEY_ENV,
    DEFAULT_MODEL,
    MAX_REFERENCE_IMAGES,
    TRANSPARENT_MODEL,
    OpenAIImageService,
    build_edit_kwargs,
    build_generate_kwargs,
    legal_size,
)

NOW = "2026-09-04T14:49:55+00:00"


def encoded(size=(1024, 1024), mode="RGBA", alpha=0) -> str:
    """One base64 PNG, the only shape a GPT image model ever answers with.

    `mode="LA"` exists for Minor 4: a grayscale-plus-alpha reply is a real
    shape a model could send, and it must not be mistaken for opaque just
    because it is not `RGBA` before conversion.
    """
    buffer = BytesIO()
    if mode == "RGBA":
        pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        pixels[1:-1, 1:-1] = (10, 20, 200, 255)
        pixels[0, 0] = (0, 0, 0, alpha)
    elif mode == "LA":
        pixels = np.full((size[1], size[0], 2), 200, dtype=np.uint8)
        pixels[0, 0, 1] = alpha
    else:
        pixels = np.full((size[1], size[0], 3), (10, 20, 200), dtype=np.uint8)
    Image.fromarray(pixels, mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def request(size=(1024, 1024), background=BackgroundPolicy.TRANSPARENT, model=None):
    return GenerationRequest(
        id="20260904-144955", created=NOW, subject="a wooden chair",
        kind=PromptKind.PROP, prompt="Draw a wooden chair.", service="openai",
        model=model, size=size, background=background,
    )


def reply(b64=None, *, background="transparent", request_id="req-1"):
    """Stand in for an `ImagesResponse`.

    `_request_id` mirrors what the installed SDK actually attaches: every
    parsed `BaseModel` gets one via `openai._response.add_request_id`, which
    is verified against `openai==3.8.0` in the fix report — `ImagesResponse`
    carries no `id` field, but every reply the SDK hands back carries this.
    """
    return SimpleNamespace(
        created=1770000000,
        data=[SimpleNamespace(b64_json=b64 or encoded())],
        background=background,
        _request_id=request_id,
    )


class FakeClient:
    """Stands in for `openai.OpenAI`, recording what each endpoint was given."""

    def __init__(self, *replies):
        self.generate_calls: list[dict] = []
        self.edit_calls: list[dict] = []
        self._replies = list(replies)
        self.images = SimpleNamespace(generate=self._generate, edit=self._edit)

    def _answer(self):
        answer = self._replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def _generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self._answer()

    def _edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        return self._answer()


def test_transparency_is_advertised_because_gpt_image_2_supports_it():
    capabilities = OpenAIImageService(client=FakeClient()).capabilities()
    assert capabilities.transparent_background is True
    assert capabilities.reference_images is True
    assert capabilities.edit is True
    assert capabilities.vision is False
    assert (1024, 1024) in capabilities.sizes


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ((1024, 1024), "1024x1024"),
        # Not on the 16-pixel grid: moved onto the nearest line of it.
        ((1000, 1000), "1008x1008"),
        # A big square is perfectly legal and is kept.
        ((2048, 2048), "2048x2048"),
        # Under the pixel floor: grown to the smallest legal square.
        ((256, 256), "816x816"),
        # Over the pixel ceiling: brought to the biggest legal square.
        ((5000, 5000), "2880x2880"),
        # Anything that is not a square gets the nearest standard shape.
        ((1536, 1024), "1536x1024"),
        ((1024, 1536), "1024x1536"),
        ((5000, 2000), "1536x1024"),
    ],
)
def test_a_size_is_snapped_to_one_the_api_accepts(asked, expected):
    assert legal_size(asked) == expected


def test_a_transparent_request_asks_for_png_and_transparency():
    kwargs = build_generate_kwargs(request(), DEFAULT_MODEL)
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["background"] == "transparent"
    assert kwargs["output_format"] == "png"
    assert kwargs["size"] == "1024x1024"
    assert kwargs["n"] == 1
    # Never sent to a GPT image model: it always answers base64, and the
    # parameter is rejected.
    assert "response_format" not in kwargs
    # Only valid with webp or jpeg, and this is always png.
    assert "output_compression" not in kwargs


def test_a_keyed_request_asks_for_an_opaque_background():
    kwargs = build_generate_kwargs(request(background=BackgroundPolicy.MAGENTA),
                                   DEFAULT_MODEL)
    assert kwargs["background"] == "opaque"


def test_an_edit_never_sends_input_fidelity_to_gpt_image_2():
    """The guide says gpt-image-2 refuses to have it changed."""
    kwargs = build_edit_kwargs(request(), DEFAULT_MODEL)
    assert "input_fidelity" not in kwargs
    assert "mask" not in kwargs


def test_a_plain_request_uses_generate(tmp_path):
    client = FakeClient(reply())
    OpenAIImageService(client=client).generate(request(), tmp_path / "image.png")
    assert len(client.generate_calls) == 1
    assert client.edit_calls == []


def test_a_request_with_references_uses_edit(tmp_path):
    first = tmp_path / "a.png"
    first.write_bytes(base64.b64decode(encoded(size=(8, 8))))
    client = FakeClient(reply())
    OpenAIImageService(client=client).generate(
        request(), tmp_path / "image.png", reference_paths=[first]
    )
    assert client.generate_calls == []
    assert len(client.edit_calls) == 1
    assert len(client.edit_calls[0]["image"]) == 1


def test_more_references_than_the_api_takes_keeps_the_newest_and_warns(tmp_path):
    """Mirrors `gemini.py`: degrade and say so, rather than refuse outright.

    A mature direction accumulates one reference per approved generation
    (`stages.generate.promote`), so a hard refusal at 17 would make the
    direction permanently un-drawable on OpenAI. Seventeen references in, the
    newest sixteen — in order — must be what `client.images.edit` receives,
    and the warning must name exactly one dropped out of seventeen.
    """
    paths = []
    for index in range(MAX_REFERENCE_IMAGES + 1):
        path = tmp_path / f"{index}.png"
        path.write_bytes(base64.b64decode(encoded(size=(8, 8))))
        paths.append(path)
    client = FakeClient(reply())
    result = OpenAIImageService(client=client).generate(
        request(), tmp_path / "image.png", reference_paths=paths
    )
    sent = client.edit_calls[0]["image"]
    assert len(sent) == MAX_REFERENCE_IMAGES
    # The oldest (index 0) is dropped; the newest sixteen travel, in order.
    assert [Path(handle.name).name for handle in sent] == [
        f"{index}.png" for index in range(1, MAX_REFERENCE_IMAGES + 1)
    ]
    assert any("1 oldest of 17" in warning for warning in result.warnings)


def test_a_reference_that_will_not_open_names_itself_and_calls_nothing(tmp_path):
    """A bad path is a mistake in the request, not the SDK being unreachable."""
    missing = tmp_path / "does-not-exist.png"
    client = FakeClient(reply())
    out = tmp_path / "image.png"
    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=client).generate(
            request(), out, reference_paths=[missing]
        )
    assert caught.value.code == "REFERENCE_UNREADABLE"
    assert str(missing) in str(caught.value)
    assert not out.exists()
    assert client.edit_calls == []


def test_the_bytes_are_decoded_and_saved(tmp_path):
    out = tmp_path / "image.png"
    result = OpenAIImageService(client=FakeClient(reply())).generate(request(), out)
    assert out.exists()
    with Image.open(out) as image:
        assert image.mode == "RGBA"
    assert result.service == "openai"
    assert result.model == DEFAULT_MODEL
    # `_request_id` is what the installed SDK actually attaches to every
    # parsed reply; there is no `id` field on `ImagesResponse` to read.
    assert result.response_id == "req-1"


def test_an_opaque_answer_to_a_transparent_request_is_a_warning(tmp_path):
    """Transparency on gpt-image-2 is in preview and can quietly not happen."""
    client = FakeClient(reply(encoded(mode="RGB")))
    result = OpenAIImageService(client=client).generate(
        request(), tmp_path / "image.png"
    )
    assert any("no transparency" in warning for warning in result.warnings)


def test_an_la_reply_with_real_alpha_is_not_falsely_warned_about(tmp_path):
    """Minor 4: `transparent` must be measured after the RGBA conversion.

    An `LA` reply carries real alpha but is not `mode == "RGBA"` before
    `image.convert("RGBA")` runs — measuring beforehand would call a picture
    with a fully-transparent pixel "no transparency in it".
    """
    client = FakeClient(reply(encoded(mode="LA", alpha=0)))
    result = OpenAIImageService(client=client).generate(
        request(), tmp_path / "image.png"
    )
    assert not any("no transparency" in warning for warning in result.warnings)


def test_a_size_the_api_would_not_take_is_recorded(tmp_path):
    """The warning compares against the *measured* size, not the discarded one.

    The reply here (1024x1024) disagrees with both what was asked for
    (1000x1000) and the size that was actually sent after `legal_size` snapped
    it (1008x1008) — so this only holds down the intended behaviour if the
    measured size from `_save_as_png` is what gets compared, not thrown away.
    """
    client = FakeClient(reply(encoded(size=(1024, 1024))))
    result = OpenAIImageService(client=client).generate(
        request(size=(1000, 1000)), tmp_path / "image.png"
    )
    assert any(
        "1000x1000" in warning and "1024x1024" in warning
        for warning in result.warnings
    )


def test_a_moderation_refusal_is_a_service_error_that_names_itself(tmp_path):
    class BadRequest(Exception):
        code = "moderation_blocked"
        body = {"moderation_details": {"moderation_stage": "input",
                                       "categories": ["harassment"]}}

    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=FakeClient(BadRequest("blocked"))).generate(
            request(), tmp_path / "image.png"
        )
    assert caught.value.code == "moderation_blocked"
    assert "harassment" in str(caught.value)
    # A retry with the same prompt gets the same answer, and the message says so.
    assert "change the prompt" in str(caught.value)


def test_a_moderation_refusal_in_the_standard_envelope_is_read_too(tmp_path):
    """Minor 5: `_rejected_param` reads the nested `error` envelope; this must too.

    Some responses carry `moderation_details` nested inside `error`, not flat
    on the body — the same shape `_rejected_param` already handles for `param`.
    """
    class BadRequest(Exception):
        code = "moderation_blocked"
        body = {"error": {"moderation_details": {"moderation_stage": "input",
                                                  "categories": ["violence"]}}}

    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=FakeClient(BadRequest("blocked"))).generate(
            request(), tmp_path / "image.png"
        )
    assert "violence" in str(caught.value)


def test_a_refused_transparent_parameter_names_the_other_model(tmp_path):
    """Only when the API says which parameter it refused."""
    class BadRequest(Exception):
        code = "unknown_parameter"
        param = "background"

        def __str__(self):
            return "Unknown parameter: 'background'."

    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=FakeClient(BadRequest())).generate(
            request(), tmp_path / "image.png"
        )
    assert TRANSPARENT_MODEL in str(caught.value)
    assert "2026-12-01" in str(caught.value)


def test_the_parameter_name_is_read_out_of_the_body_when_it_has_to_be(tmp_path):
    """Some SDK versions leave `param` off the exception and only in the body."""
    class BadRequest(Exception):
        code = "unknown_parameter"
        body = {"error": {"param": "background", "message": "Unknown parameter."}}

    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=FakeClient(BadRequest())).generate(
            request(), tmp_path / "image.png"
        )
    assert TRANSPARENT_MODEL in str(caught.value)


def test_an_error_that_merely_mentions_a_background_is_not_a_model_problem(tmp_path):
    """The old check was a substring match, and this is what it got wrong.

    Telling somebody to change models — and to change to one with a shutdown
    date on it — because a sentence happened to contain the word "background"
    is expensive advice given for no reason.
    """
    class ServerError(Exception):
        code = "server_error"

        def __str__(self):
            return "Internal error while rendering the background layer."

    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=FakeClient(ServerError())).generate(
            request(), tmp_path / "image.png"
        )
    assert TRANSPARENT_MODEL not in str(caught.value)
    assert caught.value.code == "server_error"


def test_a_reply_that_is_not_base64_is_a_service_error(tmp_path):
    """`binascii.Error` is a ValueError, so unguarded it escapes the contract."""
    client = FakeClient(SimpleNamespace(data=[SimpleNamespace(b64_json="!!!not base64!!!")]))
    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=client).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "UNREADABLE_IMAGE"


def test_a_reply_that_decodes_but_is_not_a_picture_leaves_no_file(tmp_path):
    """Base64 can decode to anything; only Pillow proves it is a picture.

    Proving that in memory, before the bytes ever reach `out_path`, is what
    keeps a broken `image.png` from being left behind for a later stage to
    trip over.
    """
    junk = base64.b64encode(b"not a png").decode("ascii")
    client = FakeClient(SimpleNamespace(data=[SimpleNamespace(b64_json=junk)]))
    out = tmp_path / "image.png"
    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=client).generate(request(), out)
    assert caught.value.code == "UNREADABLE_IMAGE"
    assert not out.exists()


def test_a_missing_key_is_a_service_error_naming_the_variable(monkeypatch):
    """The most common real failure, and it must not be a traceback.

    `importorskip`-gated, so it is skipped on the default offline suite — the
    same caveat Task 5's Gemini twin records, and for the same reason.
    """
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    pytest.importorskip("openai")
    with pytest.raises(ServiceError) as caught:
        OpenAIImageService().generate(request(), Path("unused.png"))
    assert caught.value.code == "NO_KEY"
    assert API_KEY_ENV in str(caught.value)


def test_a_client_that_will_not_build_for_a_reason_other_than_a_key_is_not_started(
    monkeypatch,
):
    """Important 2: only `OpenAIError` means a missing key; anything else does not.

    A malformed `OPENAI_BASE_URL`, a proxy refusing the connection, or an SDK
    signature change are not "no key" — `openai.OpenAIError` is importable
    here (the SDK is provably installed, since `_ensure_client` has already
    imported it), so only that documented class earns `NO_KEY`. Everything
    else is `NOT_STARTED`, quoting the SDK's own words.
    """
    pytest.importorskip("openai")
    import openai

    def _raise(*args, **kwargs):
        raise RuntimeError("proxy down")

    monkeypatch.setattr(openai, "OpenAI", _raise)
    with pytest.raises(ServiceError) as caught:
        OpenAIImageService().generate(request(), Path("unused.png"))
    assert caught.value.code == "NOT_STARTED"
    assert "proxy down" in str(caught.value)


def test_any_other_sdk_error_is_still_a_service_error(tmp_path):
    class RateLimited(Exception):
        code = "rate_limit_exceeded"

    with pytest.raises(ServiceError) as caught:
        OpenAIImageService(client=FakeClient(RateLimited("slow down"))).generate(
            request(), tmp_path / "image.png"
        )
    assert caught.value.service == "openai"
    assert caught.value.code == "rate_limit_exceeded"


def test_an_empty_reply_is_a_service_error(tmp_path):
    with pytest.raises(ServiceError, match="no image"):
        OpenAIImageService(client=FakeClient(SimpleNamespace(data=[]))).generate(
            request(), tmp_path / "image.png"
        )


def test_a_reply_with_a_url_but_no_base64_is_a_service_error(tmp_path):
    """Minor 9: the other half of the `NO_IMAGE` branch, not just the empty list.

    GPT image models always answer base64; a `url` with no `b64_json` is
    precisely the shape a model would send if it ignored that, and `image_bytes`
    deliberately treats it the same as no data at all.
    """
    client = FakeClient(
        SimpleNamespace(
            data=[SimpleNamespace(b64_json=None, url="https://example.com/x.png")]
        )
    )
    with pytest.raises(ServiceError, match="no image") as caught:
        OpenAIImageService(client=client).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "NO_IMAGE"


def test_describe_says_it_cannot(tmp_path):
    """The Images API draws; it does not look. Naming falls back to `item`."""
    answer = OpenAIImageService(client=FakeClient()).describe(None, "What is this?")
    assert answer == ""


#: Set this, and a key, to run the two tests here that cost money.
LIVE_ENV = "PIXELMAGIC_LIVE_OPENAI"


@pytest.mark.skipif(
    not (os.environ.get(LIVE_ENV) and os.environ.get("OPENAI_API_KEY")),
    reason=f"set {LIVE_ENV}=1 and OPENAI_API_KEY to draw one real picture",
)
def test_openai_really_draws_a_transparent_picture(tmp_path):
    """Opt-in, and the only test here that costs money. It draws one picture.

    It is the gate on the one thing this module cannot check offline: that
    `background="transparent"` on gpt-image-2 — a preview feature — really
    comes back with an alpha channel.
    """
    out = tmp_path / "image.png"
    result = OpenAIImageService().generate(
        GenerationRequest(
            id="live", created=NOW, subject="a red circle", kind=PromptKind.PROP,
            prompt="A single flat red circle on a fully transparent background.",
            service="openai", size=(1024, 1024),
            background=BackgroundPolicy.TRANSPARENT,
        ),
        out,
    )
    assert out.exists()
    assert result.model == DEFAULT_MODEL
    with Image.open(out) as image:
        assert image.mode == "RGBA"
        assert image.getchannel("A").getextrema()[0] == 0
    assert result.warnings == []


@pytest.mark.skipif(
    not (os.environ.get(LIVE_ENV) and os.environ.get("OPENAI_API_KEY")),
    reason=f"set {LIVE_ENV}=1 and OPENAI_API_KEY to draw one real picture",
)
def test_openai_really_takes_a_reference_through_the_edit_endpoint(tmp_path):
    """Opt-in, and the shape the pipeline actually sends.

    Every offline test proves the kwargs this module builds; nothing proves
    the API accepts them on `images.edit`, which is the register's one
    ungated row — `background`, `output_format`, `size` and `n` are documented
    for `images.generate` and the `edit` reference lists only `image` and
    `mask`. And `edit` is not the rare path: after the first approved
    generation every request in a direction carries a reference, because
    `promote` adds one each time, so `edit` is what almost every real call
    goes through.

    If this fails on a rejected parameter, drop that one parameter from
    `build_edit_kwargs` alone — never from `build_generate_kwargs` — and record
    it in the deviations register.
    """
    reference = tmp_path / "swatch.png"
    Image.new("RGBA", (64, 64), (10, 20, 200, 255)).save(reference)
    out = tmp_path / "image.png"
    OpenAIImageService().generate(
        GenerationRequest(
            id="live-ref", created=NOW, subject="a red circle", kind=PromptKind.PROP,
            prompt="A single flat red circle on a fully transparent background.",
            service="openai", size=(1024, 1024),
            background=BackgroundPolicy.TRANSPARENT,
        ),
        out,
        reference_paths=[reference],
    )
    assert out.exists()
