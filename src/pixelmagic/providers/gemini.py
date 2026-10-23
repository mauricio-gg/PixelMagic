"""Draw with Google's Gemini image models.

CONTRACT — checked on 2026-09-04 against Google's current documentation and by
introspecting `google-genai==2.22.0`. Three things about this service shape
everything below.

**There is no transparent background, and there is no flag for one.** The
image-generation docs never mention transparency, alpha or RGBA; `ImageConfig`
has no such field; and Google's own sticker examples say "The background must
be white." The models emit flat RGB. So `capabilities().transparent_background`
is False, `stages.generate.choose_background` downgrades every transparent
request to the magenta key colour, and `stages.slice.to_alpha` removes it — which
is exactly what locked decision 6 was written for. Nothing here works around
that decision; it is the decision working.

**There is no pixel size.** `image_config` takes an `aspect_ratio` out of a
fixed list and an `image_size` tier of `1K`, `2K` or `4K`. A request's pixel
size is mapped to the closest of each and the result records what was actually
asked for.

**Every image carries a SynthID watermark**, with no opt-out on this API. That
is recorded as a warning on every result rather than hidden: PixelRefiner
removes Gemini watermarks downstream (spec §2), and a person reading a sidecar
should know why that step exists.

The call. Google's documentation has moved to a new `client.interactions.create()`
surface; the migration page says `generateContent` "remains fully supported"
and gives no sunset date. This module uses `client.models.generate_content`
because it is the typed, validated, stable surface, and because the whole
request is then plain data this repo can test with nothing installed:

    client.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=[prompt, {"inline_data": {"mime_type": …, "data": …}}, …],
        config={"response_modalities": ["IMAGE"],
                "image_config": {"aspect_ratio": "1:1", "image_size": "1K"}},
    )

The reply is not flat either. The bytes are `raw`, not base64, and they live on
a part: `response.candidates[0].content.parts[n].inline_data.data`. A refusal
is not an exception — it is HTTP 200 with `candidates[0].finish_reason` set to
one of the blocking reasons, or `prompt_feedback.block_reason` set, and no
image part at all.

Up to `MAX_REFERENCES` reference images travel inline with the prompt.
`generate` keeps the most recent and warns about the rest; `build_contents`
itself stays a pure "send what you are given" function so it can be tested
against any length without knowing about the limit.

The key is read from the environment: `GEMINI_API_KEY`, or `GOOGLE_API_KEY`,
which the SDK prefers when both are set (and warns about). Never a literal,
never a sidecar, never an argument on a command line.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from pixelmagic.keys import GEMINI_API_KEY_ENV, find_key
from pixelmagic.model import GenerationRequest, GenerationResult
from pixelmagic.providers.base import ImageCapabilities, ServiceError

#: The flagship image model on 2026-09-04. `gemini-2.5-flash-image` is legacy;
#: `gemini-3-pro-image` and `gemini-3.1-flash-lite-image` are the other two
#: current ids and a direction may name any of them through `Direction.model`.
#: `imagen-4.0-*` was shut down on 2026-08-17 and is not an option at all.
DEFAULT_MODEL = "gemini-3.1-flash-image"

#: The key. `GOOGLE_API_KEY` also works — the SDK reads it first — but the
#: README names one variable so nobody sets both and gets a warning.
API_KEY_ENV = GEMINI_API_KEY_ENV

SERVICE_NAME = "gemini"

#: Google's documented limit on inline reference images per request (checked
#: 2026-09-04 against the current image-generation docs). `generate` enforces
#: it by keeping the most recent references — `stages.generate.promote`
#: appends one per approved generation, so the newest are the most relevant —
#: and warning about the ones left out; `build_contents` itself sends whatever
#: it is handed, uncapped, so it stays a pure function to test.
MAX_REFERENCES = 14

#: The ratios `ImageConfig.aspect_ratio` documents, with their values. The
#: docs page also lists `4:5` and `5:4`; the installed SDK's docstring does
#: not, so they are left out rather than guessed at.
ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("1:1", 1.0),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:3", 4 / 3),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("21:9", 21 / 9),
)

#: A `finish_reason` in this set means the model refused rather than drew.
#: `NO_IMAGE` is in here on purpose: it is a perfectly successful call that
#: came back with words instead of a picture, and it is the easiest failure in
#: this API to mistake for a success.
BLOCKING_REASONS: frozenset[str] = frozenset(
    {
        "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
        "IMAGE_OTHER", "NO_IMAGE", "SAFETY", "PROHIBITED_CONTENT",
        "BLOCKLIST", "SPII", "RECITATION",
    }
)

#: Said on every result, because it is true of every result.
WATERMARK_WARNING = (
    "every Gemini image carries a SynthID watermark; PixelRefiner removes it "
    "during refine"
)


def aspect_ratio(size: tuple[int, int]) -> str:
    """The ratio Gemini offers that is closest to the one that was asked for."""
    width, height = size
    wanted = width / height
    return min(ASPECT_RATIOS, key=lambda pair: abs(pair[1] - wanted))[0]


def image_size_tier(size: tuple[int, int]) -> str:
    """The `1K`/`2K`/`4K` tier that covers the requested pixels.

    Rounding up rather than down: a picture bigger than asked for can be
    scaled, and the slicer works in whatever it is given, while one smaller
    than asked for has lost detail that cannot come back.
    """
    longest = max(size)
    if longest <= 1024:
        return "1K"
    if longest <= 2048:
        return "2K"
    return "4K"


def build_config(request: GenerationRequest) -> dict:
    """The `config` object for one drawing call, as plain data.

    A dict rather than `types.GenerateContentConfig` so this module imports —
    and this function is tested — without `google-genai` installed.
    """
    return {
        "response_modalities": ["IMAGE"],
        "image_config": {
            "aspect_ratio": aspect_ratio(request.size),
            "image_size": image_size_tier(request.size),
        },
        # We never pass tools, so automatic function calling has nothing to
        # do — leaving it on just makes the SDK print an advisory on every
        # single call.
        "automatic_function_calling": {"disable": True},
    }


def build_contents(prompt: str, reference_paths: Sequence[Path]) -> list[object]:
    """The prompt, then each reference image inline, in the order given.

    This function sends whatever it is given and does not enforce
    `MAX_REFERENCES` itself — `generate` is where the cap and the warning
    belong, since only it has a `GenerationResult` to put the warning on.
    Every reference in this repo is a PNG `promote` wrote, so the mime type is
    always `image/png` — an invariant of the store, not a guess.

    Reading each file is guarded here rather than left to raise bare, because
    a missing or unreadable reference is not exotic: `direction.references` is
    just a list of names in `direction.json`, and the files behind them can be
    deleted, renamed or lost when a project folder is copied.
    """
    contents: list[object] = [prompt]
    for path in reference_paths:
        try:
            data = Path(path).read_bytes()
        except OSError as error:
            raise ServiceError(
                f"gemini reference image is not readable: {path}",
                service=SERVICE_NAME,
                code="REFERENCE_UNREADABLE",
            ) from error
        contents.append({"inline_data": {"mime_type": "image/png", "data": data}})
    return contents


def _refusal(response: object) -> tuple[str, str] | None:
    """The reason the model gave for not drawing, if it gave one.

    Returns a `(code, detail)` pair. A blocked *prompt* shows up on
    `prompt_feedback`; a blocked or empty *answer* shows up as the candidate's
    `finish_reason`. Both are HTTP 200, which is why neither can be caught.
    """
    blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if blocked:
        return str(blocked), "the prompt was blocked before anything was drawn"
    candidates = getattr(response, "candidates", None) or []
    # A non-empty list whose one entry is itself `None` is as good as no
    # candidate at all — guarded here so it reads as NO_CANDIDATES rather
    # than an AttributeError further down where the candidate is used.
    if not candidates or candidates[0] is None:
        return "NO_CANDIDATES", "the reply carried no candidate at all"
    reason = str(getattr(candidates[0], "finish_reason", "") or "")
    # The enum may arrive as `FinishReason.NO_IMAGE`; only the name matters.
    reason = reason.rsplit(".", 1)[-1]
    if reason in BLOCKING_REASONS:
        said = getattr(response, "text", None) or ""
        return reason, said.strip() or "no picture came back"
    return None


def image_bytes(response: object) -> bytes:
    """The raw PNG bytes out of a reply, or a `ServiceError` saying why not.

    The bytes are raw here, not base64 — that is the one difference from the
    newer `interactions` surface, and getting it wrong writes a file that looks
    like a picture and is not.
    """
    refused = _refusal(response)
    if refused is not None:
        code, detail = refused
        raise ServiceError(
            f"gemini did not draw anything ({code}): {detail}",
            service=SERVICE_NAME,
            code=code,
        )
    # A candidate can arrive with `content` set to None — a `MAX_TOKENS` finish
    # is the common way — and `_refusal` deliberately lets that through,
    # because it is not one of the blocking reasons. Reaching straight for
    # `.content.parts` there is an AttributeError, which is not a ServiceError
    # and so escapes the whole failure contract.
    content = getattr(response.candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        data = getattr(inline, "data", None) if inline is not None else None
        if not data:
            continue
        if not isinstance(data, bytes):
            # `types.Blob.data` is documented as raw bytes today, but the
            # newer `interactions` surface returns base64 text — and this
            # module does not know which encoding a str would be, so it is
            # reported rather than guessed at, to avoid writing a file that
            # looks like a picture and is not.
            raise ServiceError(
                f"gemini returned a {type(data).__name__} image payload, "
                "not bytes",
                service=SERVICE_NAME,
                code="UNREADABLE_IMAGE",
            )
        return data
    raise ServiceError(
        "gemini answered without an image part, and without saying why",
        service=SERVICE_NAME,
        code="NO_IMAGE",
    )


def _save_as_png(data: bytes, out_path: Path, *, service: str) -> tuple[int, int]:
    """Write the reply's bytes as a real PNG and return the size they were.

    Two jobs, both at the boundary, for the reason `providers/codex.py` gives
    for its own version: proving the file is a picture here rather than four
    stages later when the slicer opens it. Gemini's `inline_data` carries a
    `mime_type` that is not promised to be `image/png`, so writing the bytes
    straight into a file called `image.png` can leave a WebP behind under a
    PNG name; opening through Pillow and saving normalises it. And the size
    has to be measured, because there is nothing in the reply to trust.

    Decoding and writing are two different ways to fail and are told apart by
    two `try` blocks, mirroring `providers/codex.py`'s own `_save_as_png`: a
    reply that is not a picture is `UNREADABLE_IMAGE` (Gemini's fault), while
    a full disk or a directory this process cannot write to is
    `OUTPUT_NOT_WRITTEN` (the local filesystem's fault) — both are
    `ServiceError`, but the words a person reads should point at the right
    place. `Image.DecompressionBombError` derives from `Exception`, not
    `OSError`, so it is named explicitly or it would escape both.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            picture = image.convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ServiceError(
            f"gemini returned {len(data)} bytes that are not a picture",
            service=service,
            code="UNREADABLE_IMAGE",
        ) from error
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        picture.save(out_path, format="PNG")
    except (OSError, ValueError) as error:
        raise ServiceError(
            f"gemini's picture could not be written to {out_path}: {error}",
            service=service,
            code="OUTPUT_NOT_WRITTEN",
        ) from error
    return picture.size


class GeminiImageService:
    """An `ImageService` backed by the Gemini image models."""

    name = SERVICE_NAME

    def __init__(
        self,
        *,
        client: object | None = None,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        """Bind the client, the default model and — only if given — a key.

        `client` is injected by every test in this repo, which is what lets the
        whole suite run with `google-genai` not installed. In real use it is
        None and one is built on first use, so importing this module costs
        nothing and needs nothing.
        """
        self._client = client
        self._model = model
        self._api_key = api_key

    def _ensure_client(self) -> object:
        """Build the SDK client on first use, importing the SDK only here.

        Both the import and the construction are guarded. Constructing a client
        with no key anywhere raises a bare `ValueError` out of the SDK — which
        is the single most common real failure, and the one the failure
        contract exists for: an unguarded one would escape as something
        `stages.generate` does not catch, so the folder it had just made would
        be left behind holding a request and no picture, and the person would
        get a traceback instead of a sentence. Worse for this service in
        particular, a `ValueError` is what `cli.refine` and `cli.run` already
        catch and report as a *refiner* problem, so a missing Gemini key could
        be announced as something else entirely.

        Only that documented no-credentials `ValueError` is reported as
        `NO_KEY`. Anything else the constructor raises — a malformed key, a
        proxy that refuses the connection, a bad credentials path — is a
        different problem than a missing variable, and telling someone to set
        a variable that is already set wastes their time. Those become
        `NOT_STARTED` instead, quoting the SDK's own words rather than
        guessing at a cause.
        """
        if self._client is None:
            try:
                from google import genai
            except ImportError as error:
                raise ServiceError(
                    "the gemini service needs the google-genai package: run "
                    "`uv sync --extra gemini`",
                    service=self.name,
                    code="NOT_INSTALLED",
                ) from error
            found = find_key(self.name)
            key = self._api_key or (found.key if found else None)
            try:
                # `genai.Client()` with no key falls back to GOOGLE_API_KEY
                # itself, so an absent GEMINI_API_KEY is not an error here —
                # but an absent *both* is, and the SDK says so with a
                # ValueError rather than an APIError.
                self._client = genai.Client(api_key=key) if key else genai.Client()
            except ValueError as error:
                raise ServiceError(
                    f"gemini has no API key: set {API_KEY_ENV} (or GOOGLE_API_KEY) in the "
                    f"environment, or save a key on the Services page — {error}",
                    service=self.name,
                    code="NO_KEY",
                ) from error
            except Exception as error:
                raise ServiceError(
                    f"the gemini client could not be built: {error}",
                    service=self.name,
                    code="NOT_STARTED",
                ) from error
        return self._client

    def capabilities(self) -> ImageCapabilities:
        """Describe what this service supports.

        `transparent_background` is False and that is a verified fact, not
        caution: Gemini has no such parameter and its models emit flat RGB.
        `sizes` is empty because there is no pixel size to promise.
        """
        return ImageCapabilities(
            transparent_background=False,
            reference_images=True,
            edit=False,
            vision=True,
            sizes=[],
        )

    def _call(self, **kwargs) -> object:
        """One SDK call, with every way it can fail turned into a ServiceError.

        The exception classes cannot be imported for an `except` clause,
        because this module has to import with `google-genai` absent — so the
        catch is broad and narrow at the same time: broad in what it accepts,
        narrow in what it wraps, which is one call and nothing else.
        """
        try:
            client = self._ensure_client()
            return client.models.generate_content(**kwargs)
        except ServiceError:
            # Raised by `_ensure_client` for a missing package or a missing
            # key. It is already the right shape; re-wrapping it would bury a
            # precise message inside a vague one. This clause is reachable
            # only because the client is built inside the `try` — which is the
            # point, since building it is where the key is checked.
            raise
        except Exception as error:  # the SDK's own errors, which we cannot name
            code = getattr(error, "code", None)
            raise ServiceError(
                f"gemini could not be reached: {error}",
                service=self.name,
                code=str(code) if code is not None else None,
            ) from error

    def generate(
        self,
        request: GenerationRequest,
        out_path: Path,
        *,
        reference_paths: Sequence[Path] = (),
    ) -> GenerationResult:
        """Draw `request` and save the PNG at `out_path`.

        When more than `MAX_REFERENCES` are given, only the most recent travel
        — `stages.generate.promote` appends one per approved generation, so
        the tail of the list is the newest — and the result's warnings say how
        many older ones were left out. `build_contents` stays a pure function
        that sends whatever it is handed; the cap belongs here because only
        `generate` has a `GenerationResult` to put the warning on.
        """
        started = datetime.now(UTC)
        model = request.model or self._model
        warnings: list[str] = []
        references = list(reference_paths)
        if len(references) > MAX_REFERENCES:
            total = len(references)
            dropped = total - MAX_REFERENCES
            references = references[-MAX_REFERENCES:]
            warnings.append(
                f"gemini takes at most {MAX_REFERENCES} reference images; "
                f"the {dropped} oldest of {total} were left out"
            )
        response = self._call(
            model=model,
            contents=build_contents(request.prompt, references),
            config=build_config(request),
        )
        data = image_bytes(response)
        measured = _save_as_png(data, out_path, service=self.name)

        finished = datetime.now(UTC)
        warnings.append(WATERMARK_WARNING)
        if measured != request.size:
            # Gemini takes an aspect ratio and a 1K/2K/4K tier, never a pixel
            # size, so the returned picture almost never matches what was
            # asked for. The slicer works from whatever came back, so this is
            # a warning and not a failure — but it belongs on the sidecar,
            # because a person comparing two directions will otherwise wonder
            # why one is 1024 wide and the other 1408.
            warnings.append(
                f"asked for {request.size[0]}x{request.size[1]} but gemini "
                f"returned {measured[0]}x{measured[1]}; gemini takes an aspect "
                f"ratio and a size tier, not a pixel size"
            )
        return GenerationResult(
            id=request.id,
            created=finished,
            service=self.name,
            model=model,
            response_id=getattr(response, "response_id", None),
            elapsed_seconds=round((finished - started).total_seconds(), 3),
            warnings=warnings,
        )

    def describe(self, image_path: Path | None, question: str) -> str:
        """Answer a short question, optionally about one picture.

        The same model answers in words when the modality asks for text, so
        there is no second model id to keep current.
        """
        contents = build_contents(
            question, [image_path] if image_path is not None else []
        )
        response = self._call(
            model=self._model,
            contents=contents,
            # Same reason as `build_config`: we never pass tools, so this
            # silences the SDK's per-call automatic-function-calling advisory.
            config={
                "response_modalities": ["TEXT"],
                "automatic_function_calling": {"disable": True},
            },
        )
        return (getattr(response, "text", None) or "").strip()
