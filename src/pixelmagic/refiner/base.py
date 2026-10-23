"""The PixelRefiner seam, and the contract it is coded against.

PixelRefiner is the tool that turns a
piece of concept art into true pixel art. PixelMagic never reimplements any of
its image maths — it shells out to the CLI and reads JSON back.

`stub.StubRefiner` changes no pixels and is the default; `--refiner cli` turns
on `cli_adapter.PixelRefinerCli`, which drives the real command.

CONTRACT — read off the shipped CLI on 2026-09-04 (PixelRefiner's own
`packages/mcp` bin script, Node >= 24) by running every verb and keeping the
replies. Nothing below is guessed.

The command is a list of argv tokens: `pixel-refiner <verb> …` from the PATH,
or whatever `PIXELMAGIC_REFINER_BIN` says, split the way a shell would.

Verbs. JSON goes to stdout, logs go to stderr; `--compact` puts the JSON on one
line, which we do not need:

  refine <input> [--output PATH] [--candidate-id ID] [--scale 1..32]
      [--overwrite] [--no-preview] [--detail summary|full] <settings flags>
  analyze <input> [--detail summary|full] <settings flags>
                                              — reports only, writes nothing
  batch <input> [<input> …]                   — 1..64 inputs, POSITIONAL
      [--output-dir DIR] [--suffix S] [--shared-palette] [--palette-colors N]
      [--palette-dither MODE] [--palette-dither-strength N] [--preview]
      [--scale N] [--overwrite] [--detail …] <settings flags>
  options [--section all|presets|palettes|quick|advanced]

Settings flags, shared by refine/analyze/batch:
  --preset <id>            one of PRESETS
  --quick <key=value>      repeatable, one key each; quick values are ALWAYS strings
  --advanced <key=value>   repeatable, one key each; coerced by the key's kind,
                           and the literal `null` unsets a key
  --settings <file.json>   {preset, quick, advanced, gridDetection}; the flags win
  --grid off|auto|hint:WxH|force:WxH
                           W and H are the OUTPUT SIZE in art pixels — not the
                           pitch — and whole numbers only (force:5.9x5.8 is a
                           usage error, exit 2). Measured 2026-09-06.

Settings layer preset -> quick -> advanced; a later layer overrides only the
keys it actually sets. Quick keys and their values:
  processingMode  auto | refine | convert | preserve
  detailLevel     smallest | small | coarse | balanced | detailed
  cellScale       quarter | half | same | double | quadruple
  reductionMode   auto | none | 8 | 16 | 24 | 32 | mono | gb_legacy | gb_pocket
                  | gb_light | pico8 | nes | pc98 | msx | c64 | arne16
                  | sfc_sprite | sfc_bg
  background      keep | auto | pick
  backgroundColor "#rrggbb"; REQUIRED when background is pick, ignored otherwise.
                  It is a QUICK key — sending it as --advanced is rejected as an
                  unknown option.
  dithering       off | subtle | strong

Forcing a size (measured on the local checkout, 2026-09-06 — the evidence
behind spec M3 §3):
  --advanced forcePixelsW=<w> --advanced forcePixelsH=<h>
                           the same as --grid force:WxH, as two advanced keys
                           so they compose with a settings block; hintPixelsW/H
                           are the hint:WxH equivalents. Whole numbers only.
  A forced refine takes route "convert" at confidence 1.0 and carries no
  LOW_GRID_CONFIDENCE. The batch verb applies its flags to EVERY input, so it
  could not force a different size per input; `Refiner.batch` therefore takes
  exactly one request. `--grid force:WxH`
  and an advanced `forcePixelsW`/`forcePixelsH` cannot be sent on the same
  call — the CLI exits 2 with INVALID_SETTINGS ("gridDetection cannot be
  combined with the advanced option") when both are present, measured on the
  local checkout on 2026-09-23 — so a batch that forces a size, and any
  caller settings that already carry `forcePixelsW/H` of their own, must go
  out as one set of advanced keys, never both forms at once.
  `options --section advanced` gives forcePixelsW/H a range of 1..1024
  (measured on the local checkout, 2026-09-23) — see `MAX_FORCE_PIXELS`. A
  value above 1024 is not refused: measured the same day, a 1536x1024 source
  forced to (1536, 1024) came back as a 1024x1024 file at exit 0, with no
  warning that anything was clamped. PixelMagic must never ask for a size the
  CLI would silently shrink, so `stages.refine.validate_target` refuses any
  side above `MAX_FORCE_PIXELS` before a request is ever built, for a refiner
  that honours a forced size. The stand-in ignores the target, so it is not
  held to that limit.
  batch --shared-palette   adds `sharedPalette: ["#rrggbb", …]` to the reply.
  --advanced reduceColorMode=fixed --advanced fixedPalette=<JSON list>
                           pins a single refine to that palette; the result's
                           palette was a subset of the shared one.
  analyze <in> --detail full
                           prints {input: {path}, report: {… gridCandidates[],
                           selectedCandidateIndex, confidence, warnings}}; the
                           adapter's pitch is the selected candidate's cellW
                           and cellH averaged. analyze over a whole sheet is
                           useless (preserve route, confidence 0); measure the
                           pieces and take the median.

Cohesion and clean-up (M4.1, merged to PixelRefiner's private `main` as
ed55da0 on 2026-09-25): four advanced keys, `cohesion` (bool),
`cohesionFamilies` (int), `cohesionTones` (int), `cleanup` (bool).
`options --section advanced`, read off that build, reports `default: true`
for `cohesion` and `cleanup` and `default: 14` / `default: 2` for
`cohesionFamilies` / `cohesionTones` — a call that never mentions one of
these four gets it on, at that value. PixelMagic never relies on that
default staying what it is today and always sends its own explicit values
(`stages.refine.HOUSE_BASELINE`), overridden per-asset the same way any
other advanced key is. `batch --shared-palette --palette-colors N` is how
`PixelRefinerCli.fit_palette` (M4.2) fits a colour palette on one
untouched picture ahead of cohesion — the kept scene's; it reads the reply's top-level
`sharedPalette`, never that one image's own `report.output.palette` —
measured on the same build on 2026-09-25, a one-image shared-palette batch's
`output.palette` holds the same colours in a different order. A `--quick background=keep`
sent alongside never shows up in `effectiveOptions` as a `background` key
at all (measured the same day, a plain refine call): it only ever appears
as the derived keys `preRemoveBackground`/`postRemoveBackground` (both
`false`), `bgExtractionMethod` (`none`) and `bgRemovalScope` (`off`).

Exit codes: 0 ok; 1 engine failure, and a batch in which ANY item failed; 2
usage error or INVALID_SETTINGS; 3 refused I/O (INPUT_NOT_FOUND,
UNSUPPORTED_INPUT, INPUT_TOO_LARGE, OUTPUT_EXISTS). Exit 1 out of a batch does
NOT mean nothing was written: the items that succeeded are on disk and in the
reply.

A failed run prints, instead of a result:
  {"failure": {"code": …, "message": …, "hint": …}}

refine prints:
  output {path, width, height, scale, bytes}      ← the file that was written
  report {input {width, height, format},
          classification {kind, confidence, reasons, features}   (may be absent)
          route             refine | convert | preserve
          confidence        0..1
          warnings[]        LOW_GRID_CONFIDENCE, BACKGROUND_UNCERTAIN,
                            BACKGROUND_REMOVAL_SKIPPED, CONTENT_LOSS_RISK,
                            ONE_AXIS_DETECTION_FAILED, EXTREME_OUTPUT_SIZE,
                            NO_CONTENT, FALLBACK_TO_PRESERVE
          gridCandidates[]  {outW, outH, cellW, cellH, offset {x, y}, method,
                             totalScore, confidence,
                             subscores {…} — only with --detail full}
                            cellW/cellH are FLOATS: a sprite upscaled by a
                            non-integer factor has a fractional cell size.
                            offset.x/y are FLOATS for the same reason — that
                            same sprite has a fractional grid origin.
          selectedCandidateIndex, autoResultCandidateIndex
          background {confidence}, smallComponentRemoval {…}, contentLoss {…}
          output {width, height, colorCount, palette[]}   ← the pixel numbers
          candidates[] {id, kind, recommended, processingMode, outW, outH}
          effectiveOptions {…}  only with --detail full; re-sending it as
                                --advanced reproduces the result exactly
          resolved {…}          only with --detail full
          adjustments[]}

Mind the split: the top-level `output` is the file; the route, the confidence,
the warnings, the grid candidates and the colour count all live under `report`.
That is why the adapter always asks for `--detail full`.

`candidates[]` carries the ids `refine --candidate-id` accepts — that is how a
person picks a different grid. M1 records the candidates; offering the re-pick
is M2.1's screen.

analyze prints {input {path}, report {…}} with the same report, and writes
nothing.

batch prints:
  {items: [{id, status "done", output {…}, report {…}, needsAttention}
           | {id, status "error", error {code, message, hint}}],
   summary: {done, failed},
   sharedPalette: [ "#rrggbb", … ]        only with --shared-palette}
`id` is the input path resolved to absolute — the CLI resolves whatever it was
given — so results are matched to requests by that path, not by position. Each
output lands at <output-dir>/<stem><suffix>.png, and `--suffix` defaults to
`.refined`.

options prints lists of specs, not name->spec maps:
  {presets: [{id, quickSettings}],
   palettes: [{id, name, colorCount, colors}],
   quick:    [{key, values?, default?, description}],
   advanced: [{key, kind, min?, max?, values?, default?, description}]}

Inputs: .png .jpg .jpeg .webp .gif .avif, at most 64 MiB and 64 megapixels, at
most 64 in one batch. The output path must end in .png. RGBA true alpha is
fine; there is NO magenta-key support — use background pick. `--scale` is a
nearest-neighbour export upscale. The CLI refuses to overwrite an existing
output without `--overwrite` (OUTPUT_EXISTS, exit 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from pixelmagic.model import GridCandidate, RefineCandidate, RefinerSettings
from pixelmagic.stages.slice import SEAM_VISIBLE

EXIT_OK = 0
EXIT_ENGINE_FAILURE = 1
EXIT_USAGE = 2
EXIT_REFUSED_IO = 3

#: Not a CLI exit code: the process never ran, or ran and said something we
#: could not read. Callers branch on `RefinerError.exit_code`, so a failure
#: must never be able to report success.
EXIT_NOT_RUN = -1

LOW_GRID_CONFIDENCE = "LOW_GRID_CONFIDENCE"

#: The most `scale` PixelRefiner will take. The contract at the top of this
#: file quotes the CLI's own flag: `--scale 1..32`. Checked in
#: `validate_settings` so a `scale` outside that range can never be written
#: into a direction's stored defaults — the same argument `stages/export.py`'s
#: `MAX_SCALE` makes for the export upscale, a different number for a
#: different knob.
MAX_REFINE_SCALE = 32

#: The most a forced side (`forcePixelsW`/`forcePixelsH`) can be. Measured on
#: the local checkout on 2026-09-23: `options --section advanced` reports
#: `min: 1, max: 1024` for both keys, and a value over 1024 is not refused —
#: the CLI clamps it to 1024 and exits 0, so a caller that did not check this
#: itself would be told a forced 1536x1024 refine succeeded and get a
#: 1024x1024 file back. `stages.refine.validate_target` refuses any target
#: side above this for a refiner that honours a forced size (the stand-in is
#: exempt: it ignores the target), on top of the `MAX_IMAGE_SIZE` bound every other picture
#: gets — both must hold, and this one is usually the tighter of the two.
MAX_FORCE_PIXELS = 1024

#: Every warning code the CLI may return.
WARNING_CODES: frozenset[str] = frozenset(
    {
        LOW_GRID_CONFIDENCE,
        "BACKGROUND_UNCERTAIN",
        "BACKGROUND_REMOVAL_SKIPPED",
        "CONTENT_LOSS_RISK",
        "ONE_AXIS_DETECTION_FAILED",
        "EXTREME_OUTPUT_SIZE",
        "NO_CONTENT",
        "FALLBACK_TO_PRESERVE",
    }
)

#: The warning code every tile of a block once got when its cohesion-on centre
#: seam scored worse than the same nine tiles refined again with cohesion and
#: clean-up off (ticket #81). Nothing produces it any more, since tile blocks
#: are no longer refined (#179); it stays so an old `refine.json` that carries
#: it still loads. Defined here rather than in `stages/refine.py` because that
#: module already imports from this one.
COHESION_FALLBACK = "COHESION_FALLBACK"

#: Added to a receipt when a saved setting that would have changed a forced
#: output size (`scale`, `makeSquare`, `keepAspectRatio`, `outlineStyle`; each
#: measured on the real CLI, ticket #34) was left out of the call. The
#: receipt's `requested` holds what was actually sent.
SIZE_SETTING_DROPPED = "SIZE_SETTING_DROPPED"

#: Warning codes PixelMagic adds to a report itself, after the refiner has
#: run — as opposed to `stages.refine.PIXELMAGIC_REFUSAL_CODES`, which means
#: PixelMagic refused the asset before the refiner ever saw it and which
#: lands in `RefineReport.error_code`, never in `warnings`. Kept apart from
#: `WARNING_CODES` — the refiner's own closed vocabulary — so a test can
#: still prove the adapter never invents a code, and so a code of ours can
#: never shadow one of theirs. `RefineReport.warnings` holds codes from
#: either set and nothing else.
PIXELMAGIC_WARNING_CODES: frozenset[str] = frozenset(
    {SEAM_VISIBLE, COHESION_FALLBACK, SIZE_SETTING_DROPPED}
)

#: Every preset the CLI ships.
PRESETS: frozenset[str] = frozenset(
    {
        "auto",
        "crisp-sprite",
        "keep-fine-details",
        "photo-to-pixel",
        "transparent-icon",
        "retro-game",
        "background-art",
        "illustration-to-pixel-art",
    }
)

#: Allowed values for each quick key.
QUICK_KEYS: dict[str, frozenset[str]] = {
    "processingMode": frozenset({"auto", "refine", "convert", "preserve"}),
    "detailLevel": frozenset({"smallest", "small", "coarse", "balanced", "detailed"}),
    "cellScale": frozenset({"quarter", "half", "same", "double", "quadruple"}),
    "reductionMode": frozenset(
        {
            "auto", "none", "8", "16", "24", "32", "mono", "gb_legacy", "gb_pocket",
            "gb_light", "pico8", "nes", "pc98", "msx", "c64", "arne16",
            "sfc_sprite", "sfc_bg",
        }
    ),
    "background": frozenset({"keep", "auto", "pick"}),
    #: Free-form "#rrggbb". The CLI lists no values for it, and an empty set
    #: says that rather than pretending there is a vocabulary to check.
    "backgroundColor": frozenset(),
    "dithering": frozenset({"off", "subtle", "strong"}),
}


class RefinerError(RuntimeError):
    """The refiner refused or failed.

    `exit_code` is the CLI's own exit code when the CLI actually ran, and
    `EXIT_NOT_RUN` when it did not (missing binary, timeout, unreadable
    output). Callers branch on it, so a failure must never be able to carry
    `EXIT_OK`.
    """

    def __init__(self, message: str, *, exit_code: int, code: str | None = None) -> None:
        """Record the exit code and, when the CLI gave one, the error code."""
        super().__init__(message)
        self.exit_code = exit_code
        self.code = code


class RefineRequest(BaseModel):
    """One image to refine and where the result should land."""

    input_path: Path
    output_path: Path
    settings: RefinerSettings = Field(default_factory=RefinerSettings)
    overwrite: bool = True
    #: Ask for one of the alternatives from an earlier reply's `candidates[]`
    #: instead of letting the refiner choose. Not part of `RefinerSettings`:
    #: it is a one-off answer about one image, not a layer a direction can
    #: hold as a default — and `batch` has no per-input equivalent, so this
    #: only ever travels on a single `refine`.
    candidate_id: str | None = None
    #: The output size to force, in art pixels, `(w, h)` (spec locked
    #: decision 2). PixelMagic decides the number and the refiner does the
    #: work: the CLI adapter turns it into `forcePixelsW/H`, the stand-in
    #: ignores it. `None` lets the refiner guess, which is what an M2 asset
    #: with no target on its sidecar still gets.
    target: tuple[int, int] | None = None


class RefineOutput(BaseModel):
    """What the refiner actually produced."""

    path: Path
    width: int
    height: int
    color_count: int = 0
    palette: list[str] = Field(default_factory=list)


class RefineResult(BaseModel):
    """One successful refine, in the CLI's own vocabulary."""

    output: RefineOutput
    route: str
    confidence: float
    classification: str | None = None
    warnings: list[str] = Field(default_factory=list)
    grid_candidates: list[GridCandidate] = Field(default_factory=list)
    #: The alternatives `--candidate-id` accepts. Empty from a refiner that
    #: does not detect grids at all, which is the honest answer for the
    #: stand-in.
    candidates: list[RefineCandidate] = Field(default_factory=list)
    effective_options: dict[str, object] = Field(default_factory=dict)


class Analysis(BaseModel):
    """What `analyze` read off one picture, with nothing written.

    `pitch` is source pixels per art pixel: the mean of the selected grid
    candidate's `cell_w` and `cell_h`, or `None` when the refiner selected
    no candidate at all. The rest is the same report `refine` gives, minus
    the output — which is the point: a reading costs a subprocess and no
    file, so decompose can take one per piece before anything is refined.
    """

    pitch: float | None
    confidence: float
    warnings: list[str] = Field(default_factory=list)
    grid_candidates: list[GridCandidate] = Field(default_factory=list)


class RefineFailure(BaseModel):
    """One item of a batch the CLI refused, without failing the whole batch."""

    input_path: Path
    code: str
    message: str = ""


#: What a batch hands back per input.
BatchItem = RefineResult | RefineFailure


class RefinerOptions(BaseModel):
    """The answer to `pixel-refiner options` — used to build a settings UI."""

    presets: list[str] = Field(default_factory=list)
    palettes: list[str] = Field(default_factory=list)
    #: The quick keys and their allowed values, keyed by name. These are the
    #: "quick knobs" the web page offers; M1 dropped this section because
    #: nothing read it.
    quick: dict[str, dict] = Field(default_factory=dict)
    advanced: dict[str, dict] = Field(default_factory=dict)


def validate_settings(settings: RefinerSettings) -> None:
    """Catch the settings mistakes the CLI would reject outright.

    `background: pick` without a `backgroundColor` is refused by the CLI
    (INVALID_SETTINGS, exit 2), so checking it here turns a subprocess
    round-trip into an immediate, readable error. Both keys live in the quick
    layer: `backgroundColor` sent as an advanced option is rejected as an
    unknown key.

    `scale`, when given, must be a whole number the CLI's own `--scale 1..32`
    would accept. Every caller that can set a direction's stored defaults —
    `stages.refine.set_refiner_defaults` and every refine path built on it —
    runs its settings through this function first, so a
    `scale` outside that range can never be written into `direction.json` and
    quietly break every later run.
    """
    if settings.scale is not None and (
        isinstance(settings.scale, bool) or not (1 <= settings.scale <= MAX_REFINE_SCALE)
    ):
        raise ValueError(
            f"scale must be a whole number from 1 to {MAX_REFINE_SCALE}, "
            f"not {settings.scale!r}"
        )
    if settings.quick.get("background") != "pick":
        return
    if not settings.quick.get("backgroundColor"):
        raise ValueError(
            "background 'pick' needs a quick backgroundColor like '#ff00ff'"
        )


class Refiner(Protocol):
    """Anything that can turn concept art into pixel art."""

    name: str
    #: Whether this refiner produces exactly the `RefineRequest.target` it is
    #: given. The core checks a result's size against the target only when
    #: this is True; a refiner that ignores the target on purpose says False.
    honours_forced_size: bool

    def options(self) -> RefinerOptions:
        """List the presets, palettes and advanced keys this refiner accepts."""
        ...

    def analyze(
        self, input_path: Path, settings: RefinerSettings | None = None
    ) -> Analysis:
        """Read the pixel grid off one image without writing anything.

        The same settings flags `refine` takes, so a reading can be made
        under a direction's house style if a caller wants it; decompose
        does not pass any. Raises `RefinerError` the way `refine` does,
        including `INPUT_NOT_FOUND` for a file that is not there.
        """
        ...

    def refine(self, request: RefineRequest) -> RefineResult:
        """Refine one image. Raises `RefinerError` when the refiner refuses."""
        ...

    def batch(self, requests: Sequence[RefineRequest]) -> list[BatchItem]:
        """Refine exactly one image through the CLI's `batch` verb.

        It takes a sequence but exactly one request: any other count (zero, or
        two or more) is refused with `RefinerError(exit_code=EXIT_USAGE)`
        before anything runs or is written, by every implementation. The kept
        scene's refine is the only caller. The CLI's batch verb is kept, not
        swapped for `refine`, because the own-palette recipe was measured on
        it and its failures come back per item rather than as a top-level
        failure.

        OUTPUT CONTRACT — every implementation must honour this:

        * When this returns a `RefineResult`, the refined image is at the
          request's own `output_path`, and the result's `output.path` equals
          that path.
        * A request that comes back as a `RefineFailure` leaves its
          `output_path` untouched.
        * The returned list has one item.
        * An existing `output_path` with `overwrite` off is refused.
        * `candidate_id` is ignored. The CLI's `batch` verb has no per-input
          candidate flag, so a re-pick is always a single `refine`.
        * A result names a file the refiner *wrote*, not one it linked to
          something it was given. An implementation that stages a copy of the
          input must check that what came back is not that copy under another
          name — the path, the type and the directory all look right in that
          case, and only the inode does not.
        * A raise means nothing was written. An input the refiner refused
          comes back as a `RefineFailure`, not as an exception — including a
          missing `input_path` (`INPUT_NOT_FOUND`).
        * `target` is honoured. The stand-in ignores it altogether (spec
          locked decision 9).

        The CLI names its outputs after its inputs and writes them into one
        `--output-dir`, and the input in this pipeline is always called
        `source.png`. Honouring this contract is the adapter's job (it stages
        the input in a scratch directory and moves the result into place), not
        the caller's.
        """
        ...

    def fit_palette(
        self,
        input_path: Path,
        count: int,
        settings: RefinerSettings | None = None,
    ) -> list[str] | RefineFailure:
        """Fit one colour palette of at most `count` colours on one untouched picture.

        This is the first half of the two-step own-palette recipe the kept
        scene gets (`stages.refine`). Fit the palette *before* cohesion or
        clean-up ever runs, on the picture exactly as generate/decompose left
        it, so the palette describes the art itself rather than whatever
        cohesion's own colour grouping already did to it. `settings` travels
        exactly as given — this method does not turn cohesion or clean-up off
        itself, the caller does that before calling it — and there is no
        `target` parameter: a palette fit is a reading, never a resize, so it
        is never sent with a forced size.

        Returns the fitted colours as `#rrggbb` strings, at most `count` of
        them. It comes back as a `RefineFailure` instead of a list only when
        no usable palette resulted at all: the picture was missing or refused
        as its own item (with the refiner's own code and message — the same
        shape one item of a `batch` call takes), the call itself was refused as
        a whole with an I/O code such as INPUT_TOO_LARGE or UNSUPPORTED_INPUT,
        or the call succeeded but still returned no usable `sharedPalette` (a
        fully transparent source is the real case this covers). Raises
        `RefinerError` the way `batch` does when the call itself did not
        happen: the CLI not running, bad settings, or an unreadable reply.

        `PixelRefinerCli.fit_palette` gets the colours from a `batch
        --shared-palette --palette-colors N` call's own top-level
        `sharedPalette`, never from the image's `report.output.palette`
        — measured on the local checkout on 2026-09-25 (M4.1 acceptance): a
        one-image shared-palette batch's `output.palette` holds the same
        colours in a *different order*, and the M4.1 spike's own recipe this
        recipe reproduces reads `sharedPalette` for exactly that reason.
        """
        ...
