"""The `pixelmagic-segmenter` command: three verbs, JSON on stdout, logs on stderr.

The contract (verbs, reply shapes, exit codes) is the module docstring of
`pixelmagic.segmenter.base` in the main project. Anything a library prints to
stdout is sent to stderr instead, so stdout holds exactly one JSON object.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sam_helper import auto
from sam_helper.model import (
    LABEL_BOX_BOTTOM_RIGHT,
    LABEL_BOX_TOP_LEFT,
    SEGMENTER_NAME,
    WEIGHTS_DIR,
    WEIGHTS_FILE,
    WEIGHTS_PATH,
    WEIGHTS_REPO,
    WEIGHTS_SHA256,
    ImageUnreadable,
    Sam,
    read_rgb,
    sha256_of,
)

EXIT_OK = 0
EXIT_ENGINE_FAILURE = 1
EXIT_REFUSED_IO = 3

FETCH_HINT = "run pixelmagic-segmenter fetch-model"


class Refusal(Exception):
    """A run that stops with a `failure` object and an exit code."""

    def __init__(self, code: str, message: str, hint: str = "", exit_code: int = EXIT_REFUSED_IO):
        """Record what the `failure` object and the exit code will say."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.exit_code = exit_code


def _parser() -> argparse.ArgumentParser:
    """The command line: three verbs."""
    parser = argparse.ArgumentParser(prog="pixelmagic-segmenter", description=__doc__)
    verbs = parser.add_subparsers(dest="verb", required=True)

    suggest = verbs.add_parser("suggest", help="find every object on a picture by itself")
    suggest.add_argument("image", type=Path)
    suggest.add_argument("--out-dir", type=Path, required=True)
    suggest.add_argument("--spacing", type=int, default=auto.DEFAULT_SPACING)
    suggest.add_argument("--min-area", type=int, default=auto.DEFAULT_MIN_AREA)
    suggest.add_argument("--max-fraction", type=float, default=auto.DEFAULT_MAX_FRACTION)

    masks = verbs.add_parser("masks", help="the outline under some clicks or inside a box")
    masks.add_argument("image", type=Path)
    masks.add_argument("--requests", type=Path, required=True)
    masks.add_argument("--out-dir", type=Path, required=True)

    verbs.add_parser("fetch-model", help="download the model weights")
    return parser


def _load_picture(image: Path) -> np.ndarray:
    """Read the picture or refuse with the contract's codes."""
    if not image.is_file():
        raise Refusal("IMAGE_NOT_FOUND", f"No file at {image}.")
    try:
        return read_rgb(image)
    except ImageUnreadable as error:
        raise Refusal("IMAGE_UNREADABLE", str(error)) from error


def _load_model() -> Sam:
    """Build the model, or refuse when the weights are not there (nothing downloads here)."""
    if not WEIGHTS_PATH.is_file():
        raise Refusal("MODEL_NOT_FOUND", f"No model weights at {WEIGHTS_PATH}.", FETCH_HINT)
    return Sam(WEIGHTS_PATH)


def run_suggest(args: argparse.Namespace) -> tuple[dict, int]:
    """The `suggest` verb: the reply and its exit code."""
    rgb = _load_picture(args.image)
    sam = _load_model()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suggestions = auto.suggest(
        sam,
        rgb,
        args.out_dir,
        spacing=args.spacing,
        min_area=args.min_area,
        max_fraction=args.max_fraction,
    )
    height, width = rgb.shape[:2]
    reply = {
        "segmenter": SEGMENTER_NAME,
        "picture": {"width": width, "height": height},
        "suggestions": suggestions,
    }
    return reply, EXIT_OK


def _points_of(request: dict) -> tuple[list[tuple[float, float]], list[int]]:
    """A request's clicks and then its box corners, as SAM wants them.

    Raises `ValueError` naming what is wrong with the request.
    """
    coords: list[tuple[float, float]] = []
    labels: list[int] = []
    for point in request.get("points") or []:
        if len(point) != 3 or point[2] not in (0, 1):
            raise ValueError(f"a point must be [x, y, 0 or 1], got {point!r}")
        coords.append((point[0], point[1]))
        labels.append(int(point[2]))
    box = request.get("box")
    if box is not None:
        if len(box) != 4:
            raise ValueError(f"a box must be [x0, y0, x1, y1], got {box!r}")
        coords += [(box[0], box[1]), (box[2], box[3])]
        labels += [LABEL_BOX_TOP_LEFT, LABEL_BOX_BOTTOM_RIGHT]
    if not coords:
        raise ValueError("the request has neither points nor a box")
    return coords, labels


def _answer(sam: Sam, request: dict, index: int, out_dir: Path) -> dict:
    """One request's result entry; a bad request becomes its own `error` entry."""
    request_id = request.get("id") if isinstance(request, dict) else None
    try:
        coords, labels = _points_of(request)
        mask, score = sam.best_mask(coords, labels)
        box = auto.mask_box(mask)
        if box is None:
            raise ValueError("the model found nothing there (an empty mask)")
    except Exception as error:  # one bad request must not cost the others
        print(f"masks: request {request_id!r} failed: {error}", file=sys.stderr)
        return {"id": request_id, "error": str(error)}
    # The file is named by position, never by the caller's id, so an id can
    # hold any characters without touching the file system.
    name = f"{index:04d}.png"
    auto.write_mask(mask, box, out_dir / name)
    return {
        "id": request_id,
        "box": list(box),
        "area": int(mask.sum()),
        "score": score,
        "mask": name,
    }


def run_masks(args: argparse.Namespace) -> tuple[dict, int]:
    """The `masks` verb: one encoding answers every request."""
    rgb = _load_picture(args.image)
    try:
        requests = json.loads(args.requests.read_text())
    except (OSError, ValueError) as error:
        raise Refusal(
            "REQUESTS_UNREADABLE", f"Could not read the requests file {args.requests}: {error}"
        ) from error
    if not isinstance(requests, list):
        raise Refusal("REQUESTS_UNREADABLE", "The requests file must hold a JSON list.")
    sam = _load_model()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sam.set_image(rgb)
    results = [_answer(sam, request, index, args.out_dir) for index, request in enumerate(requests)]
    failed = any("error" in result for result in results)
    reply = {"segmenter": SEGMENTER_NAME, "results": results}
    return reply, EXIT_ENGINE_FAILURE if failed else EXIT_OK


def run_fetch_model(_args: argparse.Namespace) -> tuple[dict, int]:
    """The `fetch-model` verb: download the weights and check their hash."""
    if WEIGHTS_PATH.is_file() and sha256_of(WEIGHTS_PATH) == WEIGHTS_SHA256:
        print(f"{WEIGHTS_PATH} is already here and its hash is right.", file=sys.stderr)
        return {"weights": str(WEIGHTS_PATH), "sha256": WEIGHTS_SHA256, "downloaded": False}, 0
    from huggingface_hub import hf_hub_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE, local_dir=WEIGHTS_DIR))
    found = sha256_of(downloaded)
    if found != WEIGHTS_SHA256:
        downloaded.unlink()
        raise Refusal(
            "MODEL_HASH_MISMATCH",
            f"The downloaded file's SHA-256 is {found}, not {WEIGHTS_SHA256}; it was deleted.",
            "try fetch-model again, and report it if it keeps happening",
            exit_code=EXIT_ENGINE_FAILURE,
        )
    return {"weights": str(downloaded), "sha256": found, "downloaded": True}, 0


VERBS = {"suggest": run_suggest, "masks": run_masks, "fetch-model": run_fetch_model}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one verb, print its one JSON object, return the exit code."""
    args = _parser().parse_args(argv)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # whatever a library prints must not reach the JSON channel
    try:
        reply, code = VERBS[args.verb](args)
    except Refusal as refusal:
        reply = {
            "failure": {"code": refusal.code, "message": refusal.message, "hint": refusal.hint}
        }
        code = refusal.exit_code
    except Exception as error:
        traceback.print_exc()
        reply = {"failure": {"code": "ENGINE_FAILURE", "message": str(error), "hint": ""}}
        code = EXIT_ENGINE_FAILURE
    finally:
        sys.stdout = real_stdout
    print(json.dumps(reply))
    return code


if __name__ == "__main__":
    sys.exit(main())
