"""Talk to the real `pixel-refiner` command.

Everything this module knows about the CLI is written down in
`pixelmagic.refiner.base`'s docstring, which was checked against the built
binary on 2026-09-04 by running every verb and reading what came back. The
adapter's whole job is turning a `RefineRequest` into arguments, running the
process, and turning its JSON back into models — no image maths, no retries, no
cleverness.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from pixelmagic.model import GridCandidate, GridOffset, RefineCandidate, RefinerSettings
from pixelmagic.refiner.base import (
    EXIT_ENGINE_FAILURE,
    EXIT_NOT_RUN,
    EXIT_OK,
    EXIT_REFUSED_IO,
    EXIT_USAGE,
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

#: Overrides the command, split the way a shell would split it — set this to
#: point at any other PixelRefiner build.
BINARY_ENV = "PIXELMAGIC_REFINER_BIN"

#: The command used when `PIXELMAGIC_REFINER_BIN` is not set: a plain
#: `pixel-refiner` found on the PATH. The real refiner is opt-in
#: (`--refiner cli`), so this is only looked up when a person asked for it.
PATH_COMMAND = "pixel-refiner"

DEFAULT_TIMEOUT = 300

#: Appended to a `RefinerError` that means the CLI itself never ran — not
#: installed, refused to start, timed out, or exited with no JSON reply to
#: read. Never appended to a failure the tool reported through its own JSON
#: `failure` object: that kind is about the picture (an existing output, bad
#: settings), and the fix for it is never one of these two flags.
_WAY_OUT = (
    f" — carry on without it (leave out --refiner cli, or use --refiner stub), "
    f"or set {BINARY_ENV} to your PixelRefiner command"
)


def default_command() -> list[str]:
    """The command to run: the environment's, else `pixel-refiner` on the PATH.

    `PIXELMAGIC_REFINER_BIN` is split the way a shell would split it, so it can
    be a script with its interpreter in front, a packaged binary, anything a
    shell could run. With it unset the command is the plain `pixel-refiner`
    name; a machine without one finds out on first use, as a `RefinerError`
    that names the way out.
    """
    override = os.environ.get(BINARY_ENV)
    if override:
        return shlex.split(override)
    return [PATH_COMMAND]


def _advanced_value(value: object) -> str:
    """Render one advanced value the way the CLI's `key=value` parser reads it.

    The CLI coerces by the key's kind: `null` unsets a key, `true`/`false` are
    booleans, a bare integer is a number, anything starting with `[` or `{` is
    parsed as JSON, and everything else stays a string.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _settings_args(settings: RefinerSettings) -> list[str]:
    """The preset/quick/advanced/scale flags for one settings block.

    Quick and advanced are one flag per key, repeated — the CLI has no
    whole-object form on the command line. Sorting the keys keeps the command
    stable, which is what makes it readable in a log and comparable in a test.
    """
    args: list[str] = []
    if settings.preset:
        args += ["--preset", settings.preset]
    for key in sorted(settings.quick):
        args += ["--quick", f"{key}={settings.quick[key]}"]
    for key in sorted(settings.advanced):
        args += ["--advanced", f"{key}={_advanced_value(settings.advanced[key])}"]
    if settings.scale:
        args += ["--scale", str(settings.scale)]
    return args


def _forced_settings(
    settings: RefinerSettings, target: tuple[int, int] | None
) -> RefinerSettings:
    """A copy of `settings` whose advanced layer carries the forced output size.

    `forcePixelsW/H` are the advanced keys that mean "produce exactly this
    many art pixels" (checked against the local checkout on 2026-09-06 — they
    are the output size, not the pitch). Folding them into the dictionary
    rather than appending two more flags means exactly one flag per key
    reaches the CLI whatever the caller's own settings held, so the target
    wins without relying on which of two repeated flags the tool honours.
    The advanced keys are used rather than `--grid force`, so a settings
    block that also sets advanced keys composes and `effectiveOptions`
    records the size (spec §3).
    """
    if target is None:
        return settings
    width, height = target
    return settings.model_copy(
        update={
            "advanced": {**settings.advanced, "forcePixelsW": width, "forcePixelsH": height}
        }
    )


def build_refine_args(request: RefineRequest) -> list[str]:
    """The full argument list for one `refine` call, after the command tokens.

    Paths are resolved to absolute because a relative path would otherwise be
    read against the CLI's own working directory. `--no-preview` keeps a
    base64 PNG out of the reply we do not use, and `--detail full` is what
    makes `effectiveOptions` come back at all.
    """
    args = [
        "refine",
        str(request.input_path.resolve()),
        "--output",
        str(request.output_path.resolve()),
    ]
    if request.candidate_id:
        args += ["--candidate-id", request.candidate_id]
    args += _settings_args(_forced_settings(request.settings, request.target))
    if request.overwrite:
        args.append("--overwrite")
    args += ["--no-preview", "--detail", "full"]
    return args


def build_analyze_args(input_path: Path, settings: RefinerSettings) -> list[str]:
    """The full argument list for one `analyze` call, after the command tokens.

    The same settings flags `refine` takes, no `--output` because nothing is
    written, and `--detail full` so the grid candidates come back with their
    cell sizes — which is the whole reason to call it.
    """
    return [
        "analyze",
        str(input_path.resolve()),
        *_settings_args(settings),
        "--detail",
        "full",
    ]


#: Our field name -> the CLI's own key. Checked against the shipped binary;
#: none of these is a guess, so a missing one means the contract moved.
_GRID_KEYS: dict[str, str] = {
    "out_w": "outW",
    "out_h": "outH",
    "cell_w": "cellW",
    "cell_h": "cellH",
    "method": "method",
    "total_score": "totalScore",
    "confidence": "confidence",
}


def _grid_candidate(raw: Mapping[str, object]) -> GridCandidate:
    """Read one grid candidate, or say plainly that the contract moved.

    A candidate we cannot read is not something to drop quietly: these are the
    alternatives offered to a person when the refiner is unsure of the grid,
    and half of one is worse than none.

    A `ValidationError` counts as unreadable too: the CLI is free to send a
    value our models will not take — a non-numeric `cellW`, a null where a
    score belongs — and a pydantic traceback out of a subprocess adapter tells
    the caller nothing about which of the two sides moved.
    """
    try:
        offset = raw["offset"]
        return GridCandidate(
            offset=GridOffset(x=offset["x"], y=offset["y"]),
            **{name: raw[key] for name, key in _GRID_KEYS.items()},
        )
    except (KeyError, TypeError, ValidationError) as error:
        raise RefinerError(
            f"pixel-refiner returned a grid candidate this build cannot read "
            f"({error}): {raw!r}",
            exit_code=EXIT_NOT_RUN,
        ) from error


def _candidate(raw: Mapping[str, object]) -> RefineCandidate:
    """Read one pickable candidate, or say plainly that the contract moved.

    Only `id` is required, because `id` is the whole point: it is what
    `--candidate-id` takes, and a candidate without one is an alternative
    nobody could ever ask for. The rest is what a screen shows next to it, so
    a missing field there is a gap in the label, not a broken offer.
    """
    identifier = raw.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise RefinerError(
            f"pixel-refiner returned a candidate this build cannot read "
            f"(no id): {dict(raw)!r}",
            exit_code=EXIT_NOT_RUN,
        )
    return RefineCandidate(
        id=identifier,
        kind=str(raw.get("kind", "")),
        recommended=bool(raw.get("recommended", False)),
        processing_mode=str(raw.get("processingMode", "")),
        out_w=int(raw.get("outW", 0) or 0),
        out_h=int(raw.get("outH", 0) or 0),
    )


def _result(raw: Mapping[str, object]) -> RefineResult:
    """Turn one `refine` reply — or one finished batch item — into a result.

    The reply is not flat. The top-level `output` is the file that was written;
    the route, the confidence, the warnings, the grid candidates and the colour
    count all live under `report`.
    """
    try:
        written = raw["output"]
        report = raw["report"]
        measured = report["output"]
        classification = report.get("classification") or {}
        return RefineResult(
            output=RefineOutput(
                path=Path(written["path"]),
                width=written["width"],
                height=written["height"],
                color_count=measured.get("colorCount", 0),
                palette=list(measured.get("palette", [])),
            ),
            route=report["route"],
            confidence=report["confidence"],
            classification=classification.get("kind"),
            warnings=list(report.get("warnings", [])),
            grid_candidates=[
                _grid_candidate(item) for item in report.get("gridCandidates", [])
            ],
            candidates=[_candidate(item) for item in report.get("candidates", [])],
            # effectiveOptions is kept verbatim: re-sending it reproduces the
            # result, and editing it here would break that promise.
            effective_options=dict(report.get("effectiveOptions", {})),
        )
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        # ValueError covers `dict(effectiveOptions)` on something that is not a
        # mapping of pairs, and ValidationError with it — pydantic's subclasses
        # ValueError, so naming both would only suggest the two are different
        # kinds of wrong. RefinerError extends RuntimeError, so the one
        # `_grid_candidate` raises inside this try still travels out unwrapped.
        raise RefinerError(
            f"pixel-refiner returned a reply this build cannot read ({error}): "
            f"{str(raw)[:200]}",
            exit_code=EXIT_NOT_RUN,
        ) from error


def _analysis(raw: Mapping[str, object]) -> Analysis:
    """Turn one `analyze` reply into a reading.

    The pitch is the selected grid candidate's `cellW` and `cellH` averaged
    (spec §2); no selected candidate — an empty list, an index off the end,
    or no index at all — is honestly `None`, not a guess. Everything else is
    read the way `_result` reads a `refine` report, and an unreadable reply
    is a `RefinerError` for the same reason.
    """
    try:
        report = raw["report"]
        candidates = [_grid_candidate(item) for item in report.get("gridCandidates", [])]
        selected = report.get("selectedCandidateIndex")
        pitch = None
        if (
            isinstance(selected, int)
            and not isinstance(selected, bool)
            and 0 <= selected < len(candidates)
        ):
            chosen = candidates[selected]
            pitch = (chosen.cell_w + chosen.cell_h) / 2
        return Analysis(
            pitch=pitch,
            confidence=report.get("confidence", 0.0),
            warnings=list(report.get("warnings", [])),
            grid_candidates=candidates,
        )
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise RefinerError(
            f"pixel-refiner returned an analysis this build cannot read ({error}): "
            f"{str(raw)[:200]}",
            exit_code=EXIT_NOT_RUN,
        ) from error


def _failure_message(failure: Mapping[str, object]) -> tuple[str | None, str]:
    """Read one already-parsed `failure` object into a code and a message.

    The CLI's own `hint` is kept: it is the sentence that says what to do
    about the failure, and dropping it makes the error harder to act on.
    """
    message = str(failure.get("message", "")).strip()
    hint = str(failure.get("hint", "")).strip()
    code = failure.get("code")
    return (str(code) if code is not None else None), f"{message} {hint}".strip()


def _first_line(text: str) -> str:
    """The one line of some text that a person should read, or `""`.

    A line starting with `Error:` is preferred, because that is the line
    Node's own crash dump actually names the problem on — checked against the
    real binary on 2026-09-05: an unhandled `MODULE_NOT_FOUND` prints
    `node:internal/modules/cjs/loader:1573`, `  throw err;` and `  ^` *before*
    `Error: Cannot find module '...'`, so blindly keeping the first non-empty
    line would keep the file-and-line-number preamble and throw away the one
    line that names the missing path. Falling back to the first non-empty line
    covers every other shape of failure, where there is no such preamble to
    skip past.
    """
    lines = [line.strip() for line in text.splitlines()]
    for line in lines:
        if line.startswith("Error:"):
            return line
    for line in lines:
        if line:
            return line
    return ""


def _failure(stdout: str, stderr: str) -> tuple[str | None, str]:
    """The `{"failure": ...}` object a failed run prints, if it printed one.

    Returns the CLI's error code and a readable message. Falling back to the
    raw output matters: a usage error the argument parser rejects never reaches
    the JSON layer. When there is no JSON `failure` object to fall back to,
    only one line of stderr (or stdout, if stderr had nothing) is kept via
    `_first_line`, not the whole thing: a run that fails before the CLI's own
    error handling gets a chance — Node unable to find the script at all, say —
    prints a dozen lines of `node:internal/modules/cjs/loader` preamble and
    `at Module._resolveFilename (...)` stack frames, and CLAUDE.md's rule that
    no page ever shows a traceback applies just as much to a Node one as a
    Python one.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None
    failure = payload.get("failure") if isinstance(payload, dict) else None
    if not isinstance(failure, dict):
        return None, _first_line(stderr) or _first_line(stdout)
    return _failure_message(failure)


def _exit_failure(completed: subprocess.CompletedProcess) -> RefinerError:
    """The `RefinerError` for a run that exited without a reply worth reading.

    Covers a plain non-zero exit and a tolerated one whose stdout turned out
    to be empty or unparseable after all — in both cases the CLI did not hand
    back something a caller can use. `_WAY_OUT` is appended only when the CLI
    printed no JSON `failure` object of its own (`code` is then `None`): a
    printed `failure` means the tool ran and is reporting something about the
    picture, which `--refiner stub` and `PIXELMAGIC_REFINER_BIN` do nothing
    for.
    """
    code, detail = _failure(completed.stdout, completed.stderr)
    message = f"pixel-refiner exited {completed.returncode}: {detail}"
    if code is None:
        message += _WAY_OUT
    return RefinerError(message, exit_code=completed.returncode, code=code)


def _missing_input(request: RefineRequest) -> RefineFailure:
    """The per-item answer for a request whose input is not on disk.

    The CLI reports a missing input as an item's error, so `Refiner.batch`'s
    contract says an implementation must too: a `RefineFailure`, not a raise.
    Here the check has to happen before the CLI is called at all: the CLI only
    ever sees the staged copy, so a gone input would otherwise fail the copy.
    The message is worded the way the CLI words its own.
    """
    return RefineFailure(
        input_path=request.input_path,
        code="INPUT_NOT_FOUND",
        message=f"No readable file at {request.input_path}.",
    )


def _check_produced(
    result: RefineResult,
    request: RefineRequest,
    output_dir: Path,
    given_ids: set[tuple[int, int]],
) -> None:
    """Refuse a finished item that does not name a file the CLI just wrote.

    This is a raise, so it belongs to `batch`'s reading step: checking the item
    before moving it is what keeps "a raise means nothing was written" true.

    Four things, because the next line moves whatever this admits. A *file*,
    not merely something that exists — an empty `output.path` is `Path(".")`,
    which exists, and `shutil.move` on a directory is copytree plus rmtree
    across filesystems. Not a *symlink*, which `is_file()` would follow: moving
    the link would leave the caller holding a pointer out of the tree, and the
    next stage to write there would write through it. And *under the
    `--output-dir` we gave the CLI*, which is the half of "a PNG it wrote" that
    type alone never proves: without it, a reply naming any file on the machine
    has that file renamed onto the caller's `output_path`.

    Both sides are resolved before comparing, because on macOS the scratch
    directory is `/var/folders/…` while a path out of the CLI resolves to
    `/private/var/folders/…`; comparing one against the other would refuse
    every real run. `strict=True` makes a dangling link raise here rather than
    needing its own branch.

    And *not one of the files we gave it*, under another name — the copy
    staged in the scratch directory, or the caller's own `input_path` behind
    that copy. A hardlink from the output directory back to either one is a
    real file, is not a symlink, and resolves inside `--output-dir`, so all
    three checks above pass — and the reply is still claiming the refiner
    produced a picture it merely pointed at. `(st_dev, st_ino)` is what tells
    them apart, because a hardlink has no path of its own to compare. The
    guard stops there: a link between two files the CLI genuinely wrote is its
    own business and is accepted, and a file the CLI wrote from bytes it
    invented is a file it wrote whatever those bytes are.
    """
    produced = result.output.path
    try:
        resolved = produced.resolve(strict=True)
        inside = resolved.is_relative_to(output_dir.resolve())
    except OSError:
        inside = False
    if produced.is_symlink() or not produced.is_file() or not inside:
        raise RefinerError(
            f"pixel-refiner named {produced} for {request.input_path} "
            f"but that is not a file it wrote under {output_dir}",
            exit_code=EXIT_NOT_RUN,
        )
    # Past that raise, the resolve succeeded, so `resolved` is bound.
    info = resolved.stat()
    identity = (info.st_dev, info.st_ino)
    if identity in given_ids:
        raise RefinerError(
            f"pixel-refiner named {produced} for {request.input_path}, but "
            f"that is a file it was given — it did not write a new file",
            exit_code=EXIT_NOT_RUN,
        )


def _move_into_place(result: RefineResult, request: RefineRequest) -> None:
    """Move the batch result from the scratch directory to the caller's path.

    This is what makes `Refiner.batch`'s output contract true for the real CLI.
    The CLI wrote into a scratch directory, so its own OUTPUT_EXISTS check
    never fired on the caller's path; `batch` checks the caller's path
    before it writes anything, which is what keeps `overwrite` meaning the same
    thing in a batch as it does in a single refine.

    Nothing here can refuse: `_check_produced` has already run for the item.
    """
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(result.output.path), str(request.output_path))
    result.output.path = request.output_path


class _EmptyPalette(RefinerError):
    """`sharedPalette` was missing or empty -- `fit_palette`'s one expected failure.

    Distinct from a malformed reply. `fit_palette` catches this specific
    subclass to turn it into a `RefineFailure` naming the input it was given.
    """


def _shared_palette(payload: Mapping[str, object]) -> list[str]:
    """The `sharedPalette` a `batch --shared-palette` reply carries.

    A missing or empty `sharedPalette` -- the real case is a fully
    transparent source giving the refiner nothing to fit -- raises
    `_EmptyPalette` (ticket #70 review round 2): a reply that answered but
    had nothing to offer, which `fit_palette` treats as a picture's problem
    rather than the whole call's. A `sharedPalette` that is present
    but not a list of colour strings is a different, unplanned shape nothing
    here understands, and raises the plain `RefinerError` instead, which is
    left to travel out of every caller as a failure of the call itself.
    """
    palette = payload.get("sharedPalette")
    if palette is None or palette == []:
        raise _EmptyPalette(
            "pixel-refiner was asked for a shared palette and returned none",
            exit_code=EXIT_NOT_RUN,
        )
    if not isinstance(palette, list) or not all(
        isinstance(colour, str) for colour in palette
    ):
        raise RefinerError(
            f"pixel-refiner returned a sharedPalette that is not a list of "
            f"colour strings: {palette!r}",
            exit_code=EXIT_NOT_RUN,
        )
    return list(palette)


def _item_failure(item: Mapping[str, object], input_path: Path) -> RefineFailure:
    """Read one already-refused batch item into that input's own `RefineFailure`.

    Shared by `_read_batch` and `fit_palette`: both read a `status == "error"`
    item the same way, and a sentence kept in two places drifts the first
    time one of them changes. The CLI's own `hint` is kept on the message via
    `_failure_message` — a refused input is where the operator has the least
    context, so it is the last place to drop it. An `error` object that is
    not a mapping is not this input's own answer, it is the whole reply
    being unreadable, so that still raises rather than being swallowed into
    a made-up failure.
    """
    reported = item.get("error", {})
    if not isinstance(reported, Mapping):
        raise RefinerError(
            f"pixel-refiner returned an unreadable error for "
            f"{input_path}: {str(reported)[:200]}",
            exit_code=EXIT_NOT_RUN,
        )
    code, message = _failure_message(reported)
    return RefineFailure(input_path=input_path, code=code or "UNKNOWN", message=message)


def _stage_input(path: Path, staging: Path) -> Path:
    """Copy the input into `staging` and return the copy; the path must exist.

    The CLI is never handed the caller's own file, and it writes into our
    scratch `out/` folder, never beside the asset. That keeps
    `_check_produced`'s path and inode checks meaningful (a reply cannot
    alias the caller's picture) and leaves nothing at the caller's
    `output_path` if the call raises.

    The copy is named "<parent folder><suffix>", legible in a log. The input's
    own suffix is kept because the CLI reads six formats and decides by
    extension: renaming a .webp to .png would hand it a file that lies about
    itself.
    """
    staged = staging / f"{path.parent.name}{path.suffix or '.png'}"
    shutil.copyfile(path, staged)
    return staged


def _given_identities(
    request: RefineRequest, staged: Path
) -> set[tuple[int, int]]:
    """`(st_dev, st_ino)` of the staged copy handed to the CLI and the original behind it.

    A reply that links back to either one cannot then be mistaken for a
    result (see `_check_produced`).
    """
    return {
        (info.st_dev, info.st_ino)
        for info in (path.stat() for path in (staged, request.input_path))
    }


def _batch_args(
    input_path: Path,
    written_to: Path,
    settings: RefinerSettings,
    *,
    shared_palette: bool,
    force: tuple[int, int] | None,
) -> list[str]:
    """The argument list for one `batch` call.

    `force` is the output size the input gets. It is folded into the
    advanced layer by `_forced_settings`, the same way a single `refine`
    forces its size, rather than sent as `--grid force:WxH`: the real CLI
    exits 2 with "gridDetection cannot be combined with advanced option
    forcePixelsW" when both are present on one call (checked against the
    local checkout on 2026-09-23), and a caller's own settings are free to
    carry `forcePixelsW/H` of their own — folding into the dict is what lets
    the target overwrite those rather than collide with them.
    """
    args = ["batch", str(input_path)]
    args += ["--output-dir", str(written_to)]
    args += _settings_args(_forced_settings(settings, force))
    if shared_palette:
        args.append("--shared-palette")
    args += ["--detail", "full"]
    return args


def _read_batch(
    payload: Mapping[str, object],
    request: RefineRequest,
    staged: Path,
    written_to: Path,
    given_ids: set[tuple[int, int]],
) -> BatchItem:
    """Read a batch reply for its one request into a result or a failure.

    This only reads and checks; the caller moves the file afterwards, so a
    reply that goes wrong never leaves anything at the caller's `output_path`:
    that is what makes "a raise means nothing was written" true.

    The item names the input it came from, resolved to an absolute path — the
    same string we handed over — so it is matched by name. Keyed as a string
    because JSON is free to put an array there: an id that cannot be a dict
    key would otherwise take the whole call down before any item was looked
    at, where a stringified one simply fails to match and gets the accurate
    "returned no item for …" refusal instead.
    """
    by_input = {str(item.get("id")): item for item in payload["items"]}
    item = by_input.get(str(staged))
    if item is None:
        raise RefinerError(
            f"pixel-refiner returned no item for {request.input_path}",
            exit_code=EXIT_NOT_RUN,
        )
    if item.get("status") == "error":
        # A refused input is this request's answer, not the call's: its
        # output_path is left untouched and the caller decides what to do.
        return _item_failure(item, request.input_path)
    result = _result(item)
    _check_produced(result, request, written_to, given_ids)
    return result


class PixelRefinerCli:
    """A `Refiner` backed by the `pixel-refiner` command-line tool."""

    name = "pixel-refiner"
    # `forcePixelsW/H` produce exactly that size (measured on the real CLI),
    # provided no size-changing setting rides along (`stages.refine` drops those).
    honours_forced_size = True

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Bind the command to run and the function used to run it."""
        self._command = list(command) if command else default_command()
        self._label = shlex.join(self._command)
        self._runner = runner
        self._timeout = timeout

    def _run(
        self, args: Sequence[str], *, tolerate: Sequence[int] = ()
    ) -> tuple[int, dict]:
        """Run one CLI command and return its exit code and parsed JSON.

        Every way this can go wrong — a non-zero exit, output that is not JSON,
        the command not being installed — becomes a `RefinerError` carrying an
        exit code, so callers have one thing to catch and can never mistake a
        failure for a success. `tolerate` lists exit codes whose output is
        still worth reading: a batch exits 1 when any single item failed, and
        the items that succeeded are in that reply — but only when there
        actually is a reply; a tolerated exit code with an empty or unparseable
        stdout is reported the same way a non-tolerated one would be, rather
        than as a hollow "printed output that is not JSON" that hides the real
        failure underneath it.

        `_WAY_OUT` names the fix for every one of these — the CLI itself never
        ran, so leaving out `--refiner cli` (or `--refiner stub`) or setting
        `PIXELMAGIC_REFINER_BIN` is genuinely the way out — except the one place
        the tool ran and is reporting something about the picture through its
        own JSON `failure` object; that case is left for `_exit_failure` to tell apart by
        whether a `code` came back.
        """
        command = [*self._command, *args]
        try:
            completed = self._runner(
                command, capture_output=True, text=True, check=False, timeout=self._timeout
            )
        except FileNotFoundError as error:
            raise RefinerError(
                f"{self._label} is not installed{_WAY_OUT}",
                exit_code=EXIT_NOT_RUN,
            ) from error
        except OSError as error:
            # Every other way the operating system can refuse to start the
            # process — no execute permission, a directory where the binary
            # should be — is still "the CLI did not run", not a result.
            raise RefinerError(
                f"{self._label} could not be started: {error}{_WAY_OUT}",
                exit_code=EXIT_NOT_RUN,
            ) from error
        except subprocess.SubprocessError as error:
            # SubprocessError covers TimeoutExpired, which is not an OSError.
            raise RefinerError(
                f"{self._label} failed to run: {error}{_WAY_OUT}",
                exit_code=EXIT_NOT_RUN,
            ) from error

        if completed.returncode != EXIT_OK and completed.returncode not in tolerate:
            raise _exit_failure(completed)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            if completed.returncode != EXIT_OK:
                # A tolerated exit code only means "read the reply anyway" when
                # there is a reply. Node exiting 1 with an empty stdout because
                # it could not find the script is not a reply this build failed
                # to parse — it is the failure `tolerate` was never meant to
                # hide, so it is reported the same way a non-tolerated exit is.
                raise _exit_failure(completed) from error
            raise RefinerError(
                f"pixel-refiner printed output that is not JSON: "
                f"{completed.stdout[:200]!r}",
                exit_code=EXIT_NOT_RUN,
            ) from error
        if not isinstance(payload, dict):
            # Every reply in the contract is an object. A list or a bare value
            # is JSON we can parse and still cannot read, and every caller
            # below indexes it — so say so here rather than raising an
            # AttributeError three frames away.
            raise RefinerError(
                f"pixel-refiner printed an unreadable reply, expected a JSON "
                f"object: {completed.stdout[:200]!r}",
                exit_code=EXIT_NOT_RUN,
            )
        return completed.returncode, payload

    def options(self) -> RefinerOptions:
        """Ask the CLI what settings it accepts.

        The CLI answers with lists of specs, each carrying its own `id` or
        `key`; `RefinerOptions` wants ids and a key-to-spec map, so the shapes
        are converted here rather than everywhere they are read.
        """
        _, raw = self._run(["options"])
        return RefinerOptions(
            presets=[preset["id"] for preset in raw.get("presets", [])],
            palettes=[palette["id"] for palette in raw.get("palettes", [])],
            quick={spec["key"]: dict(spec) for spec in raw.get("quick", [])},
            advanced={spec["key"]: dict(spec) for spec in raw.get("advanced", [])},
        )

    def analyze(
        self, input_path: Path, settings: RefinerSettings | None = None
    ) -> Analysis:
        """Read the pixel grid off one image; the CLI writes nothing for this verb."""
        chosen = settings or RefinerSettings()
        validate_settings(chosen)
        _, raw = self._run(build_analyze_args(input_path, chosen))
        return _analysis(raw)

    def refine(self, request: RefineRequest) -> RefineResult:
        """Refine one image."""
        validate_settings(request.settings)
        _, raw = self._run(build_refine_args(request))
        return _result(raw)

    def batch(self, requests: Sequence[RefineRequest]) -> list[BatchItem]:
        """Refine exactly one image, honouring `Refiner.batch`'s contract.

        Any other count is refused with `EXIT_USAGE` before anything runs or
        is written. The kept scene's refine is the only caller.

        The staging is the point. The CLI takes its input as a positional path
        and names the output after it inside one `--output-dir`; every input
        in this pipeline is called `source.png`. So the input is copied into a
        scratch `in/` directory, the CLI writes into a scratch `out/`
        directory, and the result is moved to the request's own `output_path`
        before this returns. The target goes out as `--advanced
        forcePixelsW/H` (not `--grid force:WxH`: the CLI refuses the two
        together).

        A batch where the item failed exits 1. That is a per-item outcome, so
        it is returned as a `RefineFailure`, and the run does not raise. This
        raises only when the call did not happen and nothing was written: the
        count is not one, the settings are unusable, an existing `output_path`
        has `overwrite` off, the command would not run, the reply is
        unreadable, it carries a top-level `failure` instead of items, it is
        missing the item, or a finished item names something other than a
        file the CLI just wrote into our scratch directory. Every one of
        those is checked before the result is moved into place.
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
            # Nothing to ask the CLI: the answer is already known.
            return [_missing_input(request)]

        with tempfile.TemporaryDirectory(prefix="pixelmagic-batch-") as scratch:
            staging = Path(scratch) / "in"
            written_to = Path(scratch) / "out"
            staging.mkdir(parents=True)
            written_to.mkdir(parents=True)

            staged = _stage_input(request.input_path, staging)
            payload = self._batch_payload(
                _batch_args(
                    staged,
                    written_to,
                    request.settings,
                    shared_palette=False,
                    force=request.target,
                )
            )
            item = _read_batch(
                payload, request, staged, written_to, _given_identities(request, staged)
            )
            if isinstance(item, RefineResult):
                _move_into_place(item, request)
            return [item]

    def fit_palette(
        self,
        input_path: Path,
        count: int,
        settings: RefinerSettings | None = None,
    ) -> list[str] | RefineFailure:
        """Fit one palette on one untouched picture, in one scratch directory.

        Runs the M4.1 spike's own palette-fitting call — `batch --shared-
        palette --palette-colors N` over the one input, no forced size — and
        reads the answer off the reply's top-level `sharedPalette`
        (`_shared_palette`), never off the image's `report.output.palette`
        (see `Refiner.fit_palette`'s docstring for why the two can disagree).
        The input is staged in a scratch `in/` directory
        before it is sent (`_stage_input`, the same rule `batch` stages its
        own input with), and the item is matched back by the *staged* path,
        since that is what the CLI was actually told the input was called. The
        whole scratch tree — staging and the `out/` the CLI writes into — is a
        throwaway `tempfile.TemporaryDirectory` that is never the caller's
        asset folder and is removed as soon as this returns; nothing this call
        writes is ever read by anything downstream, only the palette that comes
        back with it. Reuses `_batch_args` and `_batch_payload` — the same two
        things that build and run every other `batch` call — rather than
        opening a second way to talk to the CLI.

        A missing input never reaches the CLI at all, the same way `batch`'s
        own `_missing_input` keeps one from reaching it: it is returned as a
        `RefineFailure`. An input the CLI itself refuses (found in the reply's
        own `items`) is read the same way `_read_batch` reads a refused batch
        item (`_item_failure`), with its own code and message.

        Two more failures are turned into that same `RefineFailure` shape
        rather than left to raise, so one bad call never costs the direction
        a palette is fitted ahead of (ticket #70 round 1): a `RefinerError`
        the call itself carries a refiner error code for, at an exit code
        other than usage — an I/O refusal such as INPUT_TOO_LARGE or
        UNSUPPORTED_INPUT reported about the whole call — telling a picture's
        own refusal apart from the whole call being unusable (`error.code is None` or
        `exit_code == EXIT_USAGE` still raises: that is the CLI never having
        run, or the call itself being malformed, not the picture); and a call
        that succeeded with no usable `sharedPalette` at all — a fully
        transparent source is the real case this catches — which
        `_shared_palette` raises as `_EmptyPalette` for exactly this reason
        (ticket #70 round 2). A `sharedPalette` that came back some other,
        malformed shape is a different problem — the CLI answered with
        something nothing here understands, not "no colours" — and
        `_shared_palette`'s plain `RefinerError` for that case is left to
        raise past this call rather than being folded into a `RefineFailure`.
        """
        chosen = settings or RefinerSettings()
        validate_settings(chosen)
        if not input_path.exists():
            return RefineFailure(
                input_path=input_path,
                code="INPUT_NOT_FOUND",
                message=f"No readable file at {input_path}.",
            )
        with tempfile.TemporaryDirectory(prefix="pixelmagic-palette-") as scratch:
            staging = Path(scratch) / "in"
            written_to = Path(scratch) / "out"
            staging.mkdir(parents=True)
            written_to.mkdir(parents=True)
            staged = _stage_input(input_path, staging)
            args = _batch_args(
                staged,
                written_to,
                chosen,
                shared_palette=True,
                force=None,
            )
            args += ["--palette-colors", str(count)]
            try:
                payload = self._batch_payload(args)
            except RefinerError as error:
                if error.code is None or error.exit_code == EXIT_USAGE:
                    raise
                return RefineFailure(
                    input_path=input_path, code=error.code, message=str(error)
                )
            # Matched by the exact string sent on the command line, the same
            # way `_read_batch` matches (`str(path)`, not `.resolve()`): the
            # CLI echoes back the id we handed it unresolved, and on macOS the
            # scratch directory resolves through a /var -> /private/var
            # symlink, so resolving here and not there would fail to match
            # every real run.
            by_input = {str(item.get("id")): item for item in payload.get("items") or []}
            item = by_input.get(str(staged))
            if item is None:
                raise RefinerError(
                    f"pixel-refiner returned no item for {input_path}",
                    exit_code=EXIT_NOT_RUN,
                )
            if item.get("status") == "error":
                return _item_failure(item, input_path)
            try:
                return _shared_palette(payload)
            except _EmptyPalette:
                return RefineFailure(
                    input_path=input_path,
                    code="EMPTY_PALETTE",
                    message=(
                        f"pixel-refiner returned no usable colours for {input_path} "
                        f"(a fully transparent source gives it nothing to fit)."
                    ),
                )

    def _batch_payload(self, args: Sequence[str]) -> dict:
        """Run one `batch` command and hand back a reply that has readable items.

        `items` is the one field the whole batch is read out of, so a reply
        where it is missing, or a string, or a list of them, must say so
        here. Indexing it anyway would surface as an AttributeError from
        inside a comprehension, which tells the caller nothing.
        """
        exit_code, payload = self._run(args, tolerate=[EXIT_ENGINE_FAILURE])
        items = payload.get("items")
        if items is None:
            failure = payload.get("failure")
            detail = (
                _failure_message(failure)[1]
                if isinstance(failure, Mapping)
                else str(payload)[:200]
            )
            raise RefinerError(
                f"pixel-refiner exited {exit_code} without a batch reply: {detail}",
                exit_code=exit_code or EXIT_NOT_RUN,
            )
        if not isinstance(items, list) or not all(
            isinstance(item, Mapping) for item in items
        ):
            raise RefinerError(
                f"pixel-refiner returned an unreadable reply, expected "
                f"items to be a list of objects: {str(items)[:200]}",
                exit_code=EXIT_NOT_RUN,
            )
        return payload
