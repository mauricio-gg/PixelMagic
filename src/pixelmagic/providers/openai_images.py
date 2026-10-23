"""Draw with the OpenAI Images API.

Third of the three (locked decision 1) because it is the only one billed per
picture: Codex comes out of a subscription and Gemini out of credits, and this
one out of a card.

CONTRACT — checked on 2026-09-04 against `developers.openai.com/api/docs` and
the generated types in `openai-python`.

**`gpt-image-2` does support a transparent background.** The spec says it does
not; that was true when the spec was written and stopped being true on
2026-08-20, when the guide gained: "Transparent backgrounds are available in
preview for `gpt-image-2`. Set `background: 'transparent'` to request one. Use
`png` (the default) or `webp`; `jpeg` isn't supported with transparent
backgrounds." The guide writes that value in double quotes; it is single-quoted
here because a backslash inside a docstring is `D301`, and the alternative —
escaping it — is exactly the kind of thing a later editor silently "fixes" by
changing the quoted contract. Since every request in this pipeline asks for
transparency first,
following the spec literally would mean never using the current model at all.
Spec §2 carries a dated note saying so; locked decision 1 — the order of the
three services — is untouched.

`gpt-image-1.5` is the previous model. No official page says it accepts
`background="transparent"`, so this module never falls back to it on its own: a
refused `background` parameter becomes an error that names `--model
gpt-image-1.5` as the thing to try. Falling back quietly would spend a second
picture's money on a guess.

The calls:

    client.images.generate(model=…, prompt=…, size="WxH", background=…,
                           output_format="png", n=1)
    client.images.edit(model=…, image=[file, …], prompt=…, size=…,
                       background=…, output_format="png", n=1)

`edit` is what reference images go through — up to 16 of them. When a
direction has more, `generate` keeps the newest 16 and warns about the rest,
mirroring `providers/gemini.py`'s own cap — a direction should not become
permanently un-drawable on this service just because it has matured past
sixteen approvals. `mask` is optional and this pipeline never sends one,
because it composites references rather than repainting a region.

Things that must never be sent:
  response_format      GPT image models always answer base64; it is rejected
  output_compression   only valid with webp or jpeg, and this is always png
  input_fidelity       gpt-image-2 refuses to have it changed
  stream/partial_images  a synchronous call wants neither
And `model` must always be sent: `generate` defaults to `dall-e-2` and `edit`
defaults to `gpt-image-1.5`, neither of which is what anyone here wants.

`size` is `"WIDTHxHEIGHT"` with both edges multiples of 16, an aspect ratio
between 1:3 and 3:1, a longest edge of at most 3840, and a total between
655,360 and 8,294,400 pixels. The three standard sizes `1024x1024`,
`1536x1024` and `1024x1536` are always legal. `legal_size` snaps a request
onto all of that — see its docstring for why it treats a square and a
non-square differently.

The bytes come back base64 in `response.data[0].b64_json`; `url` is
unsupported for these models. The reply echoes `background` and `size` back,
but `ImagesResponse.size` is typed as three fixed literals and cannot carry an
arbitrary `gpt-image-2` resolution — so the size that was actually produced is
measured off the decoded picture, in memory, before anything is written to
disk.

A content-policy refusal is HTTP 400 with `error.code == "moderation_blocked"`
and optional `moderation_details` inside `error.body`. It is never retried:
the same prompt gets the same answer.

The key is `OPENAI_API_KEY`, read by the SDK from the environment. A brand-new
organisation may get 403 until API Organization Verification is done.
"""

from __future__ import annotations

import base64
import binascii
import io
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from pixelmagic.keys import OPENAI_API_KEY_ENV, find_key
from pixelmagic.model import BackgroundPolicy, GenerationRequest, GenerationResult
from pixelmagic.providers.base import ImageCapabilities, ServiceError

#: The current flagship, and the only model this service picks on its own.
DEFAULT_MODEL = "gpt-image-2"

#: The previous model. Named in an error message, never chosen automatically —
#: whether it takes `background="transparent"` is not documented anywhere, and
#: guessing with it costs a picture.
TRANSPARENT_MODEL = "gpt-image-1.5"

API_KEY_ENV = OPENAI_API_KEY_ENV

SERVICE_NAME = "openai"

#: The guide says a complex prompt can take two minutes. The SDK's own default
#: is ten, which is long enough to hide a hung generate or redraw call.
DEFAULT_TIMEOUT = 300.0

#: `client.images.edit` takes at most this many images.
MAX_REFERENCE_IMAGES = 16

#: The error code a content-policy refusal carries. The guide says to branch on
#: this first and treat the details as extra.
MODERATION_CODE = "moderation_blocked"

#: The size rules, from the image-generation guide: both edges on a 16-pixel
#: grid, a shape between 1:3 and 3:1, a longest edge of at most 3840, and a
#: total between 655,360 and 8,294,400 pixels.
_EDGE_MULTIPLE = 16

#: The smallest and largest legal squares: 800x800 is 640,000 pixels, under the
#: floor, and 816x816 is 665,856, over it; 2880x2880 is 8,294,400, the ceiling
#: exactly. Both are on the 16-pixel grid and both are 1:1, so no other rule
#: can bite.
_MIN_SQUARE_EDGE = 816
_MAX_SQUARE_EDGE = 2880

#: The three sizes every GPT image model takes. They are what `capabilities`
#: advertises and what any non-square request is snapped to.
STANDARD_SIZES: list[tuple[int, int]] = [(1024, 1024), (1536, 1024), (1024, 1536)]


def _to_multiple(value: int) -> int:
    """The nearest multiple of 16, rounding halves up.

    Not `round`, which rounds halves to even: `round(1000 / 16)` is 62 and not
    63, so a request for 1000 would quietly become 992 rather than 1008. That
    is invisible until somebody compares two runs.
    """
    return int(value / _EDGE_MULTIPLE + 0.5) * _EDGE_MULTIPLE


def legal_size(size: tuple[int, int]) -> str:
    """The `"WIDTHxHEIGHT"` to ask for, given the one that was asked for.

    Four rules interact here — the 16-pixel grid, the 1:3..3:1 shape, the 3840
    edge ceiling and the pixel band — and a general snapper that satisfies all
    four at once is easy to get subtly wrong in a way no test would notice.
    So this splits the problem where the pipeline actually splits it.

    Every face asks for a **square**: `generate --size N` and `run --size N`
    are one number, and a square only has to satisfy the grid and the two area
    bounds, which is arithmetic anyone can check. A square is therefore kept,
    put on the grid, and clamped between the smallest and largest legal ones.

    Anything else — which nothing in PixelMagic asks for today, and which only
    a hand-built `GenerationRequest` can produce — gets the standard size
    closest in shape. That is always legal by construction.

    Snapping rather than refusing: a request one pixel off a multiple of
    sixteen should draw a picture rather than stop a run. The size that was
    asked for is recorded as a warning whenever it had to move.
    """
    width, height = size
    if width == height:
        edge = min(max(_to_multiple(width), _MIN_SQUARE_EDGE), _MAX_SQUARE_EDGE)
        return f"{edge}x{edge}"
    wanted = width / height
    nearest = min(
        STANDARD_SIZES, key=lambda pair: abs(pair[0] / pair[1] - wanted)
    )
    return f"{nearest[0]}x{nearest[1]}"


def _background_value(request: GenerationRequest) -> str:
    """`transparent` or `opaque`, never `auto`.

    `auto` lets the model decide, and this pipeline has already decided:
    `stages.generate.choose_background` picked transparency or a key colour,
    and a key colour with an `auto` background comes back with a drop shadow
    on it as often as not.
    """
    if request.background is BackgroundPolicy.TRANSPARENT:
        return "transparent"
    return "opaque"


def build_generate_kwargs(request: GenerationRequest, model: str) -> dict:
    """Everything `client.images.generate` is given for one drawing call."""
    return {
        "model": model,
        "prompt": request.prompt,
        "size": legal_size(request.size),
        "background": _background_value(request),
        # Required whenever background is transparent, and harmless otherwise;
        # sending it always is one fewer branch to get wrong.
        "output_format": "png",
        "n": 1,
    }


def build_edit_kwargs(request: GenerationRequest, model: str) -> dict:
    """Everything `client.images.edit` is given, except the open files.

    The caller adds `image=[…]`: the files have to be opened and closed around
    the call, and a function that returns open handles is a function that leaks
    them when the call raises.
    """
    return build_generate_kwargs(request, model)


def image_bytes(response: object) -> bytes:
    """The decoded PNG out of a reply, or a `ServiceError` saying there is none.

    The decode is guarded because `binascii.Error` is a `ValueError`, which is
    not a `ServiceError` and would therefore escape the failure contract — the
    generation folder would be left behind and `cli.run` would report a
    truncated reply as a *refiner* problem, since it already catches
    `ValueError` for that.
    """
    data = getattr(response, "data", None) or []
    encoded = getattr(data[0], "b64_json", None) if data else None
    if not encoded:
        raise ServiceError(
            "openai answered with no image at all",
            service=SERVICE_NAME,
            code="NO_IMAGE",
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ServiceError(
            f"openai answered with something that is not base64 ({error})",
            service=SERVICE_NAME,
            code="UNREADABLE_IMAGE",
        ) from error


def _save_as_png(data: bytes, out_path: Path, *, service: str) -> tuple[tuple[int, int], bool]:
    """Decode the reply as a real picture, measure it, and only then save it.

    Mirrors `providers/gemini.py`'s own `_save_as_png`, for the same reason:
    proving the bytes are a picture in memory, before anything touches disk,
    rather than four stages later when the slicer opens a file that turns out
    to be corrupt. `image_bytes` only proves the reply decoded from base64 —
    nothing rules out the base64 having decoded to something that is not a
    picture at all, and writing those bytes straight to `out_path` before
    checking would leave a broken `image.png` behind for the failure to trip
    over next.

    Decoding and writing are two different ways to fail and are told apart by
    two `try` blocks: a reply that is not a picture is `UNREADABLE_IMAGE`
    (openai's fault), while a full disk or a directory this process cannot
    write to is `OUTPUT_NOT_WRITTEN` (the local filesystem's fault) —  both
    are `ServiceError`, but the words a person reads should point at the right
    place. `Image.DecompressionBombError` derives from `Exception`, not
    `OSError`, so it is named explicitly or it would escape both.

    The size and the alpha channel are read here, off the image actually
    decoded, rather than off the file after it is written: `generate` needs
    both to build its warnings, and by the time the file exists on disk the
    image object that could answer either cheaply is already gone.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            size = image.size
            # Measured off `picture` — the RGBA-converted image — rather than
            # off `image.mode` beforehand: an `LA` reply, or a `P` reply with
            # a `tRNS` chunk, carries real alpha and converts to it, but is
            # not `mode == "RGBA"` before the conversion runs. Checking the
            # raw mode would call a picture with real transparency "no
            # transparency in it".
            picture = image.convert("RGBA")
            transparent = picture.getchannel("A").getextrema()[0] < 255
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ServiceError(
            f"openai answered with {len(data)} bytes that are not a picture",
            service=service,
            code="UNREADABLE_IMAGE",
        ) from error
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        picture.save(out_path, format="PNG")
    except (OSError, ValueError) as error:
        raise ServiceError(
            f"openai's picture could not be written to {out_path}: {error}",
            service=service,
            code="OUTPUT_NOT_WRITTEN",
        ) from error
    return size, transparent


def _rejected_param(error: Exception) -> str | None:
    """The parameter name a 400 blamed, when it named one.

    The SDK puts it on `.param`; when the exception does not carry the
    attribute it is still inside `error.body["error"]["param"]`. Reading both
    is what lets `_wrap` branch on *which* parameter was refused rather than on
    whether a word appeared somewhere in a sentence.
    """
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict):
            param = inner.get("param")
            if isinstance(param, str):
                return param
    return None


def _moderation_detail(error: Exception) -> str:
    """The categories and stage a refusal carries, if it carried any.

    Read both shapes, exactly as `_rejected_param` does for `.param`: the flat
    `body["moderation_details"]` and the standard envelope's
    `body["error"]["moderation_details"]`. Which one a given SDK version uses
    is not settled by the guide, so both are checked rather than guessed at.
    """
    body = getattr(error, "body", None)
    details = None
    if isinstance(body, dict):
        details = body.get("moderation_details")
        if not isinstance(details, dict):
            inner = body.get("error")
            if isinstance(inner, dict):
                details = inner.get("moderation_details")
    if not isinstance(details, dict):
        return ""
    categories = ", ".join(str(item) for item in details.get("categories") or [])
    stage = details.get("moderation_stage") or ""
    parts = [part for part in (stage, categories) if part]
    return f" ({'; '.join(parts)})" if parts else ""


class OpenAIImageService:
    """An `ImageService` backed by the OpenAI Images API."""

    name = SERVICE_NAME

    def __init__(
        self,
        *,
        client: object | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Bind the client, the default model and the per-call timeout.

        `client` is injected by every test here, which is what lets the suite
        run with `openai` not installed.
        """
        self._client = client
        self._model = model
        self._timeout = timeout

    def _ensure_client(self) -> object:
        """Build the SDK client on first use, importing the SDK only here.

        The import is guarded so this module still imports with `openai`
        absent, but the construction is guarded by a *named* class:
        `OpenAI()` with no `OPENAI_API_KEY` anywhere raises
        `openai.OpenAIError: Missing credentials`, and by the time
        construction runs `openai.OpenAIError` is provably importable — the
        `from openai import OpenAI` four lines above already succeeded. So
        only that documented class is treated as a missing key; a malformed
        `OPENAI_BASE_URL`, a proxy that refuses the connection, or an SDK
        signature change are a different problem than a missing variable, and
        telling someone to set a variable that is already set wastes their
        time. Left unguarded, either failure escapes as something
        `stages.generate` does not catch, so the folder it had just made
        stays behind holding a request and no picture, and the person gets a
        traceback rather than the sentence `cli._service_failed` exists to
        print.
        """
        if self._client is None:
            try:
                from openai import OpenAI, OpenAIError
            except ImportError as error:
                raise ServiceError(
                    "the openai service needs the openai package: run "
                    "`uv sync --extra openai`",
                    service=self.name,
                    code="NOT_INSTALLED",
                ) from error
            found = find_key(self.name)
            try:
                # With no key found the SDK is still asked, and says so itself.
                # max_retries is left at the SDK's 2: those retries are on
                # connection errors and rate limits, which cost nothing when
                # they fail.
                self._client = OpenAI(api_key=found.key if found else None, timeout=self._timeout)
            except OpenAIError as error:
                raise ServiceError(
                    f"openai has no API key: set {API_KEY_ENV} in the environment, or "
                    f"save a key on the Services page — {error}",
                    service=self.name,
                    code="NO_KEY",
                ) from error
            except Exception as error:
                raise ServiceError(
                    f"openai's client could not be built: {error}",
                    service=self.name,
                    code="NOT_STARTED",
                ) from error
        return self._client

    def capabilities(self) -> ImageCapabilities:
        """Describe what this service supports.

        `vision` is False: this is the Images API, which draws and does not
        look, so `naming.describe_cutout` falls back to `item` and nobody has
        to reach for a second API to name a chair.
        """
        return ImageCapabilities(
            transparent_background=True,
            reference_images=True,
            edit=True,
            vision=False,
            sizes=list(STANDARD_SIZES),
        )

    def _wrap(self, error: Exception) -> ServiceError:
        """Turn one SDK exception into the only error a service may raise.

        The SDK's classes cannot be imported for an `except` clause, because
        this module must import with `openai` absent — so the branching is on
        `error.code`, which is a plain attribute the guide tells callers to
        read first.
        """
        code = getattr(error, "code", None)
        code = str(code) if code is not None else None
        if code == MODERATION_CODE:
            return ServiceError(
                f"openai refused the prompt{_moderation_detail(error)} — "
                f"change the prompt; the same words get the same answer",
                service=self.name,
                code=code,
            )
        # Narrow on purpose. The word "background" can appear in any message —
        # a prompt that mentions one, a validation error listing every
        # parameter, a sentence echoed back — and this branch tells the
        # operator to change models, which is expensive advice to give for the
        # wrong reason. So it fires only when the API itself named `background`
        # as the parameter it refused. `param` is the SDK's field for that;
        # `error.body` carries it when the exception object does not.
        if (getattr(error, "param", None) or _rejected_param(error)) == "background":
            return ServiceError(
                f"openai would not take the transparent background "
                f"({error}). Transparency on {DEFAULT_MODEL} is a preview "
                f"feature; try a direction whose model is {TRANSPARENT_MODEL} "
                f"— which OpenAI shuts down on 2026-12-01 — or ask for a key "
                f"colour instead, which `choose_background` already handles.",
                service=self.name,
                code=code,
            )
        return ServiceError(
            f"openai returned an error: {error}", service=self.name, code=code
        )

    def _draw(self, request: GenerationRequest, model: str,
              reference_paths: Sequence[Path]) -> object:
        """One call to whichever endpoint the references decide.

        The reference files are opened before the client is even built, and
        in their own guard: a path that will not open is a mistake in the
        request, not an SDK failure, and left inside the general `except
        Exception` below it would come out of `_wrap` as "openai returned an
        error" — true, but pointing at the wrong party, and with no code a
        face could branch on.
        """
        handles: list = []
        try:
            for path in reference_paths:
                try:
                    handles.append(Path(path).open("rb"))
                except OSError as error:
                    raise ServiceError(
                        f"openai could not read the reference image {path}: "
                        f"{error}",
                        service=self.name,
                        code="REFERENCE_UNREADABLE",
                    ) from error
            # Built inside the `try` on purpose: constructing the client is
            # where a missing key is discovered, and a raw SDK exception from
            # there would escape the failure contract.
            client = self._ensure_client()
            if not handles:
                return client.images.generate(**build_generate_kwargs(request, model))
            return client.images.edit(
                image=handles, **build_edit_kwargs(request, model)
            )
        except ServiceError:
            raise
        except Exception as error:  # the SDK's own errors, which we cannot name
            raise self._wrap(error) from error
        finally:
            for handle in handles:
                handle.close()

    def generate(
        self,
        request: GenerationRequest,
        out_path: Path,
        *,
        reference_paths: Sequence[Path] = (),
    ) -> GenerationResult:
        """Draw `request` and save the PNG at `out_path`.

        When more than `MAX_REFERENCE_IMAGES` are given, only the newest
        travel — mirroring `providers/gemini.py`'s own cap — and the result's
        warnings say how many older ones were left out. A mature direction
        gaining one reference per approved generation
        (`stages.generate.promote`) must not become permanently un-drawable
        on this service just for having outlived the limit.
        """
        started = datetime.now(UTC)
        model = request.model or self._model
        warnings: list[str] = []
        references = list(reference_paths)
        if len(references) > MAX_REFERENCE_IMAGES:
            total = len(references)
            dropped = total - MAX_REFERENCE_IMAGES
            references = references[-MAX_REFERENCE_IMAGES:]
            warnings.append(
                f"openai takes at most {MAX_REFERENCE_IMAGES} reference images; "
                f"the {dropped} oldest of {total} were left out"
            )
        response = self._draw(request, model, references)
        size, transparent = _save_as_png(
            image_bytes(response), out_path, service=self.name
        )

        requested = f"{request.size[0]}x{request.size[1]}"
        asked = legal_size(request.size)
        measured = f"{size[0]}x{size[1]}"
        if measured != requested:
            # The measured size — what actually came back — is what the
            # slicer works from, so it is what belongs in the warning; `asked`
            # (what was sent, after `legal_size` snapped it) is folded in only
            # when it differs from what was requested, since that is a second,
            # separate reason the two could disagree.
            if asked != requested:
                warnings.append(
                    f"asked for {requested}, which the API does not accept; "
                    f"sent {asked} instead, and openai returned {measured}"
                )
            else:
                warnings.append(
                    f"asked for {requested}, but openai returned {measured}"
                )
        if request.background is BackgroundPolicy.TRANSPARENT and not transparent:
            # Transparency on gpt-image-2 is in preview, so this is a real
            # outcome rather than an impossible one. Decompose's own dirty
            # check catches it for sheets; this says it for everything else.
            warnings.append(
                "asked for a transparent background and got a picture with no "
                "transparency in it"
            )

        finished = datetime.now(UTC)
        return GenerationResult(
            id=request.id,
            created=finished,
            service=self.name,
            model=model,
            # `ImagesResponse` has no `id` field — verified against the
            # installed SDK (openai 3.8.0): `background, created, data,
            # output_format, quality, size, usage`, nothing named `id`. But
            # every reply the SDK hands back is a parsed `pydantic` model, and
            # `openai._response` stamps `_request_id` onto every one of those
            # from the HTTP response's own request-id header — the same id
            # OpenAI support and the usage dashboard key on. This is the only
            # service billed per picture, so it is the thing to have when a
            # line on an invoice has to be matched to a folder on disk. The
            # SDK does not promise the attribute on every object, hence
            # `getattr`; it is `None` when the SDK does not attach one — a
            # fake client in a test, for instance.
            response_id=getattr(response, "_request_id", None),
            elapsed_seconds=round((finished - started).total_seconds(), 3),
            warnings=warnings,
        )

    def describe(self, image_path: Path | None, question: str) -> str:
        """Answer nothing: the Images API draws and does not look.

        `capabilities().vision` is False, so `naming.describe_cutout` never
        calls this and cutouts get the `item` fallback. It exists because the
        protocol has it, and returning an empty string is the honest answer —
        reaching for a chat model to name a chair would be a second API, a
        second bill and a second thing to keep current.
        """
        return ""
