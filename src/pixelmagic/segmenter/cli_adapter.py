"""Talk to the real `pixelmagic-segmenter` helper (Segment Anything, ticket #112).

The helper is a separate program in `segmenter/` at the repo root, with its own
environment; PixelMagic never imports it. Its verbs and JSON are written down in
`pixelmagic.segmenter.base`'s docstring. This adapter's whole job is turning
requests into a file and arguments, running the process, and turning its JSON
and mask PNGs back into `Mask`s. A mask's PNG is placed at the box the helper
reports and is never resized.

Nothing here starts a process or reads a file until a call is made, so naming
this segmenter (`--segmenter cli`) works on a machine without the helper; the
failure comes on first use, as a `SegmenterError`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from pixelmagic.segmenter.base import (
    EXIT_ENGINE_FAILURE,
    EXIT_NOT_RUN,
    EXIT_OK,
    EXIT_USAGE,
    Mask,
    SegmenterError,
    SegmentFailure,
    SegmentRequest,
)

#: What the helper reports as its own name; it goes into `decomposition.json`
#: and the manifest.
SEGMENTER_NAME = "sam2.1-small-mlx"

#: Seconds, as the refiner's.
DEFAULT_TIMEOUT = 300

#: `src/pixelmagic/segmenter/cli_adapter.py` is four levels under the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Appended to an error that means the helper never ran.
_WAY_OUT = (
    " — set it up with `cd segmenter && uv sync` and then "
    "`uv run pixelmagic-segmenter fetch-model`, or use --segmenter stub"
)


def default_command() -> list[str]:
    """The helper's own environment's command, found from this file's place in the repo."""
    return [str(REPO_ROOT / "segmenter" / ".venv" / "bin" / "pixelmagic-segmenter")]


def _failure_of(completed: subprocess.CompletedProcess) -> SegmenterError:
    """The `SegmenterError` for a run that exited without a reply worth reading.

    The helper's own `failure` object gives the code and the sentence; when it
    printed none (a crash before its error handling), one line of stderr stands
    in and the way out is named, since the helper then did not really run —
    except after a usage error (`EXIT_USAGE`), which an installed helper gives
    for a bad request and which setup would not fix. A failure never carries
    `EXIT_OK`: a helper that exits 0 yet prints a `failure` is `EXIT_NOT_RUN`.
    """
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        payload = None
    failure = payload.get("failure") if isinstance(payload, dict) else None
    if isinstance(failure, dict):
        message = f"{failure.get('message', '')} {failure.get('hint', '')}".strip()
        code = failure.get("code")
        return SegmenterError(
            f"pixelmagic-segmenter exited {completed.returncode}: {message}",
            exit_code=completed.returncode or EXIT_NOT_RUN,
            code=str(code) if code is not None else None,
        )
    lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
    detail = lines[-1] if lines else "no output"
    way_out = "" if completed.returncode == EXIT_USAGE else _WAY_OUT
    return SegmenterError(
        f"pixelmagic-segmenter exited {completed.returncode}: {detail}{way_out}",
        exit_code=completed.returncode,
    )


def _read_mask(
    out_dir: Path, name: str, box: Sequence[int], score: float | None, exit_code: int
) -> Mask:
    """One mask PNG, read as a `Mask` at the box the helper reported.

    The helper ran, so an error here carries its real `exit_code`.
    """
    x0, y0, x1, y1 = (int(value) for value in box)
    path = out_dir / name
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L")) > 0
    except OSError as error:
        raise SegmenterError(
            f"pixelmagic-segmenter reported the mask {name} but it could not be read: {error}",
            exit_code=exit_code,
        ) from error
    if pixels.shape != (y1 - y0, x1 - x0):
        raise SegmenterError(
            f"pixelmagic-segmenter's mask {name} is {pixels.shape[1]} x {pixels.shape[0]} "
            f"but its box {[x0, y0, x1, y1]} is {x1 - x0} x {y1 - y0}",
            exit_code=exit_code,
        )
    return Mask(box=(x0, y0, x1, y1), pixels=pixels, score=score)


def _entry_mask(entry: dict, out_dir: Path, exit_code: int) -> Mask:
    """Read a reply entry that carries `box`, `mask` and (optionally) `score`."""
    try:
        return _read_mask(out_dir, entry["mask"], entry["box"], entry.get("score"), exit_code)
    except (KeyError, TypeError, ValueError) as error:
        raise SegmenterError(
            f"pixelmagic-segmenter printed an unreadable entry {entry!r}: {error!r}",
            exit_code=exit_code,
        ) from error


class SegmenterCli:
    """A `Segmenter` backed by the `pixelmagic-segmenter` helper."""

    name = SEGMENTER_NAME

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Bind the command to run and the function used to run it; start nothing."""
        self._command = list(command) if command else default_command()
        self._runner = runner
        self._timeout = timeout

    def _run(self, args: Sequence[str], *, tolerate: Sequence[int] = ()) -> tuple[dict, int]:
        """Run one helper command and return its parsed JSON reply and exit code.

        Every way this can go wrong — not installed, cannot start, timeout, a
        non-zero exit, output that is not a JSON object, a `failure` object —
        becomes a `SegmenterError` carrying an exit code, so a failure can never
        read as "no objects on the picture". `tolerate` lists exit codes whose
        reply is still worth reading (a `masks` call exits 1 when any request
        failed), but only when a reply with no `failure` is actually there.
        """
        command = [*self._command, *args]
        try:
            completed = self._runner(
                command, capture_output=True, text=True, check=False, timeout=self._timeout
            )
        except FileNotFoundError as error:
            raise SegmenterError(
                f"pixelmagic-segmenter is not installed at {self._command[0]}{_WAY_OUT}",
                exit_code=EXIT_NOT_RUN,
            ) from error
        except OSError as error:
            raise SegmenterError(
                f"pixelmagic-segmenter could not be started: {error}{_WAY_OUT}",
                exit_code=EXIT_NOT_RUN,
            ) from error
        except subprocess.SubprocessError as error:
            # Covers TimeoutExpired, which is not an OSError.
            raise SegmenterError(
                f"pixelmagic-segmenter failed to run: {error}", exit_code=EXIT_NOT_RUN
            ) from error

        if completed.returncode != EXIT_OK and completed.returncode not in tolerate:
            raise _failure_of(completed)
        try:
            payload = json.loads(completed.stdout)
        except ValueError as error:
            if completed.returncode != EXIT_OK:
                raise _failure_of(completed) from error
            raise SegmenterError(
                f"pixelmagic-segmenter printed output that is not JSON: {completed.stdout[:200]!r}",
                exit_code=EXIT_NOT_RUN,
            ) from error
        if not isinstance(payload, dict):
            raise SegmenterError(
                f"pixelmagic-segmenter printed an unreadable reply, expected a JSON object: "
                f"{completed.stdout[:200]!r}",
                exit_code=EXIT_NOT_RUN,
            )
        if "failure" in payload:
            raise _failure_of(completed)
        # The name goes into decomposition.json and the manifest; it must be
        # what actually ran, not what this adapter assumes.
        ran = payload.get("segmenter")
        if ran is not None and ran != self.name:
            raise SegmenterError(
                f"pixelmagic-segmenter says it ran {ran!r} but this adapter records "
                f"{self.name!r}",
                exit_code=EXIT_NOT_RUN,
            )
        return payload, completed.returncode

    def suggest(self, image_path: Path) -> list[Mask]:
        """Every object the helper finds on its own (about 15 seconds on a full picture)."""
        scratch = Path(tempfile.mkdtemp(prefix="pixelmagic-segmenter-"))
        try:
            reply, code = self._run(["suggest", str(image_path), "--out-dir", str(scratch)])
            entries = reply.get("suggestions")
            if not isinstance(entries, list):
                raise SegmenterError(
                    "pixelmagic-segmenter's reply has no `suggestions` list",
                    exit_code=EXIT_NOT_RUN,
                )
            return [_entry_mask(entry, scratch, code) for entry in entries]
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def segment(
        self, image_path: Path, requests: Sequence[SegmentRequest]
    ) -> list[Mask | SegmentFailure]:
        """Answer each request, in order; a request the helper refused is a `SegmentFailure`."""
        if not requests:
            return []
        scratch = Path(tempfile.mkdtemp(prefix="pixelmagic-segmenter-"))
        try:
            requests_file = scratch / "requests.json"
            requests_file.write_text(json.dumps([_request_json(item) for item in requests]))
            out_dir = scratch / "masks"
            out_dir.mkdir()
            reply, code = self._run(
                [
                    "masks",
                    str(image_path),
                    "--requests",
                    str(requests_file),
                    "--out-dir",
                    str(out_dir),
                ],
                tolerate=[EXIT_ENGINE_FAILURE],
            )
            results = reply.get("results")
            if not isinstance(results, list) or len(results) != len(requests):
                got = len(results) if isinstance(results, list) else "no"
                raise SegmenterError(
                    f"pixelmagic-segmenter answered {got} of {len(requests)} requests",
                    exit_code=EXIT_NOT_RUN,
                )
            answers: list[Mask | SegmentFailure] = []
            for request, entry in zip(requests, results, strict=True):
                if not isinstance(entry, dict) or entry.get("id") != request.id:
                    raise SegmenterError(
                        f"pixelmagic-segmenter answered {entry!r} where request "
                        f"{request.id!r} was expected",
                        exit_code=EXIT_NOT_RUN,
                    )
                if "error" in entry:
                    answers.append(SegmentFailure(id=request.id, message=str(entry["error"])))
                else:
                    answers.append(_entry_mask(entry, out_dir, code))
            return answers
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _request_json(request: SegmentRequest) -> dict:
    """One request as the helper's JSON: points as `[x, y, label]`, a box as four numbers."""
    item: dict = {"id": request.id}
    if request.points:
        item["points"] = [[point.x, point.y, point.label] for point in request.points]
    if request.box is not None:
        item["box"] = list(request.box)
    return item
