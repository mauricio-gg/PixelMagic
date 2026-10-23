"""A refiner that does nothing, honestly.

This stand-in is the default refiner; the real PixelRefiner CLI is opt-in
with `--refiner cli`. It copies the input to
the output and returns a report in exactly the contract's shape, so every
stage downstream can be built and tested with no Node and no subprocess. It
reports the `preserve` route and the `FALLBACK_TO_PRESERVE` warning, which is
the literal truth — that way a `refine.json` produced by the stand-in is never
mistaken for real pixel art.

It reads a pitch of 1.0 off every picture and ignores a request's `target`
(spec M3, locked decision 9): a stand-in bundle carries target sizes its
images do not match, and `manifest.json`'s `refiner_tool` is what says so.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from pixelmagic.model import RefinerSettings
from pixelmagic.refiner.base import (
    EXIT_REFUSED_IO,
    EXIT_USAGE,
    PRESETS,
    QUICK_KEYS,
    Analysis,
    BatchItem,
    RefineFailure,
    RefineOutput,
    RefineRequest,
    RefinerError,
    RefineResult,
    RefinerOptions,
    validate_settings,
)

#: How many colours the synthetic palette lists at most.
PALETTE_LIMIT = 32

#: The four cohesion/clean-up keys `StubRefiner` must never claim it ran,
#: however they were requested (M4 spec, "The stand-in"). `refine`'s echo of
#: `request.settings` into `effective_options` is otherwise a straight copy,
#: and correct for every other key -- this is the one, named exception.
_COHESION_KEYS = frozenset({"cohesion", "cohesionFamilies", "cohesionTones", "cleanup"})


def _opaque_colour_counts(image: Image.Image) -> Counter:
    """How many opaque pixels of each RGB colour one image has.

    Shared by `_palette` (one image) and `fit_palette` (pooled across
    several), so the one rule for "what counts as a colour here" — opaque
    pixels only, alpha dropped once it has done its job of excluding
    transparent ones — lives in one place rather than two copies that could
    drift apart.
    """
    pixels = [pixel for pixel in image.convert("RGBA").get_flattened_data() if pixel[3] > 0]
    return Counter(pixel[:3] for pixel in pixels)


def _ordered_hex(counts: Counter) -> list[str]:
    """The colours in `counts` as `#rrggbb` strings, commonest first."""
    return [f"#{red:02x}{green:02x}{blue:02x}" for (red, green, blue), _ in counts.most_common()]


def _palette(image: Image.Image) -> tuple[list[str], int]:
    """The opaque colours in an image as hex strings, commonest first."""
    counts = _opaque_colour_counts(image)
    return _ordered_hex(counts)[:PALETTE_LIMIT], len(counts)


class StubRefiner:
    """Stand-in `Refiner` used by default; the real CLI is opt-in (`--refiner cli`)."""

    name = "stub"
    # Locked decision 9: the stand-in changes no pixels and ignores the
    # target on purpose, so a result at another size is expected, not a fault.
    honours_forced_size = False

    def options(self) -> RefinerOptions:
        """Report the contract's presets and quick keys.

        `palettes` is empty on purpose: the contract's palette ids come from
        the real engine and the stand-in does not know them. Returning the
        `reductionMode` values here would look like an answer and be wrong.
        """
        return RefinerOptions(
            presets=sorted(PRESETS),
            palettes=[],
            advanced={
                key: (
                    {"kind": "enum", "values": sorted(values), "default": None}
                    if values
                    else {"kind": "string", "default": None}
                )
                for key, values in QUICK_KEYS.items()
            },
        )

    def analyze(
        self, input_path: Path, settings: RefinerSettings | None = None
    ) -> Analysis:
        """Report a pitch of 1.0 — one source pixel per art pixel — for anything.

        The stand-in changes no pixels, so the honest reading is that the
        picture already is the pixel art: the scene's target then equals its
        own box (spec §2, locked decision 9). Confidence 1.0 because that
        answer is exactly as sure as the copy `refine` makes. A missing file
        is refused the way the real CLI refuses one, for the reason
        `refine` gives.
        """
        validate_settings(settings or RefinerSettings())
        if not input_path.exists():
            raise RefinerError(
                f"{input_path} does not exist",
                exit_code=EXIT_REFUSED_IO,
                code="INPUT_NOT_FOUND",
            )
        return Analysis(pitch=1.0, confidence=1.0, warnings=[], grid_candidates=[])

    def refine(self, request: RefineRequest) -> RefineResult:
        """Copy the input to the output and describe what was copied.

        `effective_options` echoes the request's settings except for the
        four cohesion/clean-up keys (`_COHESION_KEYS`): the stand-in changes
        no pixels, so claiming either stage ran, however it was asked for,
        would be a lie a `CohesionRecord` built off this report would then
        repeat.
        """
        validate_settings(request.settings)
        if not request.input_path.exists():
            # The real CLI reports a missing input as INPUT_NOT_FOUND and exits
            # 3, so the stand-in raises the same `RefinerError` rather than a
            # bare `FileNotFoundError`. A caller that catches only what the
            # seam documents would otherwise crash on the stand-in and cope
            # with the real tool — which is the one difference the stand-in
            # exists to rule out.
            raise RefinerError(
                f"{request.input_path} does not exist",
                exit_code=EXIT_REFUSED_IO,
                code="INPUT_NOT_FOUND",
            )
        # The real CLI refuses to clobber an output without --overwrite; the
        # stand-in has to refuse the same way or it hides the failure mode.
        if request.output_path.exists() and not request.overwrite:
            raise RefinerError(
                f"{request.output_path} already exists",
                exit_code=EXIT_REFUSED_IO,
                code="OUTPUT_EXISTS",
            )
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(request.input_path, request.output_path)

        with Image.open(request.output_path) as image:
            width, height = image.size
            palette, colour_count = _palette(image)

        effective: dict[str, object] = {}
        if request.settings.preset:
            effective["preset"] = request.settings.preset
        effective.update(request.settings.quick)
        effective.update(
            {
                key: value
                for key, value in request.settings.advanced.items()
                if key not in _COHESION_KEYS
            }
        )
        if request.settings.scale:
            effective["scale"] = request.settings.scale

        return RefineResult(
            output=RefineOutput(
                path=request.output_path,
                width=width,
                height=height,
                color_count=colour_count,
                palette=palette,
            ),
            route="preserve",
            confidence=1.0,
            classification="stub",
            warnings=["FALLBACK_TO_PRESERVE"],
            effective_options=effective,
        )

    def fit_palette(
        self,
        input_path: Path,
        count: int,
        settings: RefinerSettings | None = None,
    ) -> list[str] | RefineFailure:
        """Answer with the picture's own opaque colours, most common first.

        The stand-in changes no pixels, so there is nothing to fit: this
        returns the same opaque colours a stub `refine` already builds its
        reported palette from, capped at `count` rather than `PALETTE_LIMIT`,
        so a caller asking for a 24-colour palette from the stand-in gets at
        most 24 colours back, the same shape the real refiner would answer
        with. A picture that does not exist comes back as a `RefineFailure`,
        the same shape `PixelRefinerCli.fit_palette` uses for the same case.
        """
        validate_settings(settings or RefinerSettings())
        if not input_path.exists():
            return RefineFailure(
                input_path=input_path,
                code="INPUT_NOT_FOUND",
                message=f"{input_path} does not exist",
            )
        with Image.open(input_path) as image:
            counts = _opaque_colour_counts(image)
        return _ordered_hex(counts)[:PALETTE_LIMIT][:count]

    def batch(self, requests: Sequence[RefineRequest]) -> list[BatchItem]:
        """Refine exactly one request, honouring the protocol's output contract.

        Any other count is refused with `EXIT_USAGE` before anything is
        written, as the real adapter refuses it: a stand-in that accepted
        several requests would let a caller pass every test and then fail the
        moment the seam is swapped. Writing straight to the request's
        `output_path` satisfies the output contract trivially. Bad settings
        and an unwanted clobber (`OUTPUT_EXISTS`, `overwrite` off) are caught
        before anything is written. A missing `input_path` comes back as a
        `RefineFailure` (`INPUT_NOT_FOUND`), as the real CLI reports it, with
        nothing written.
        """
        if len(requests) != 1:
            raise RefinerError(
                f"pixel-refiner's batch takes exactly one request, got {len(requests)}",
                exit_code=EXIT_USAGE,
            )
        [request] = requests
        validate_settings(request.settings)
        if request.output_path.exists() and not request.overwrite:
            raise RefinerError(
                f"{request.output_path} already exists",
                exit_code=EXIT_REFUSED_IO,
                code="OUTPUT_EXISTS",
            )
        if not request.input_path.exists():
            return [
                RefineFailure(
                    input_path=request.input_path,
                    code="INPUT_NOT_FOUND",
                    message=f"{request.input_path} does not exist",
                )
            ]
        return [self.refine(request)]
