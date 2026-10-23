"""The segmenter seam, and the contract the real helper is coded against.

A segmenter finds the outline of objects on a picture. The real one is Segment
Anything (SAM 2.1 small, MLX build), which runs as a separate local command in
its own folder and environment (`segmenter/` at the repo root, ticket #112).
PixelMagic never imports it — it shells out and reads JSON back, exactly as it
does with PixelRefiner. `stub.StubSegmenter` is the offline stand-in.

Every rule about what to do with outlines (duplicate removal, ordering, paging,
the 12 px guard, the lasso rule) is a PixelMagic rule and lives in the core,
never in the helper.

CONTRACT (plan 2026-09-30-object-cutting.md, "The helper, said plainly"). JSON
goes to stdout, logs go to stderr:

  pixelmagic-segmenter suggest <image> --out-dir DIR
      [--spacing 32] [--min-area 200] [--max-fraction 0.35]
  pixelmagic-segmenter masks   <image> --requests FILE.json --out-dir DIR
  pixelmagic-segmenter fetch-model

`suggest` clicks the model on a grid of points 32 px apart over the whole
picture, keeps a mask whose area is at least 200 px and at most 0.35 of the
picture, and removes NO duplicates. Each mask is written to DIR as an 8-bit PNG
cropped to its own box (white inside), as it is produced. Reply:

  {"segmenter": "sam2.1-small-mlx", "picture": {"width": W, "height": H},
   "suggestions": [{"id": 0, "box": [x0, y0, x1, y1], "area": N,
                    "score": 0.93, "mask": "0000.png"}, ...]}

`box` is right and bottom exclusive.

`masks` reads a list of requests, each with an `id` and points, a box, or both
(label 1 = this object, 0 = not this):

  [{"id": "a", "points": [[x, y, 1], [x, y, 0]]},
   {"id": "b", "box": [x0, y0, x1, y1], "points": [[x, y, 0]]}]

and answers every request from one encoding, the best of the model's three
masks by its own score, a box sent as two points labelled 2 and 3 placed after
any points. Reply:

  {"segmenter": ..., "results": [{"id": "a", "box": [...], "area": N,
                                  "score": 0.9, "mask": "a.png"},
                                 {"id": "b", "error": "..."}]}

A failed run prints `{"failure": {"code", "message", "hint"}}` instead of a
result. Exit codes: 0 ok; 1 engine failure, and a `masks` call in which any
request failed (the reply is still on stdout); 2 usage; 3 refused I/O
(IMAGE_NOT_FOUND, IMAGE_UNREADABLE, MODEL_NOT_FOUND with the hint "run
pixelmagic-segmenter fetch-model").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from pixelmagic.model import Point

EXIT_OK = 0
EXIT_ENGINE_FAILURE = 1
EXIT_USAGE = 2
EXIT_REFUSED_IO = 3

#: `SegmenterError.exit_code` when the helper never ran (missing binary,
#: timeout, unreadable output). A failure must never be able to carry `EXIT_OK`.
EXIT_NOT_RUN = -1


class SegmenterError(RuntimeError):
    """The segmenter refused or failed.

    A `RuntimeError` on purpose: the MCP server's `readable_errors()` already
    turns it into words. `exit_code` is the helper's own exit code when it ran,
    `EXIT_NOT_RUN` when it did not; `code` is the helper's error code, if any.
    """

    def __init__(self, message: str, *, exit_code: int, code: str | None = None) -> None:
        """Record the exit code and, when the helper gave one, the error code."""
        super().__init__(message)
        self.exit_code = exit_code
        self.code = code


@dataclass(eq=False)
class Mask:
    """One object's outline on the original picture."""

    #: `(x0, y0, x1, y1)` on the picture; right and bottom exclusive.
    box: tuple[int, int, int, int]
    #: Boolean array of shape `(y1 - y0, x1 - x0)`: True inside the object.
    pixels: np.ndarray
    #: The segmenter's own confidence, when it gives one.
    score: float | None = None

    @property
    def area(self) -> int:
        """How many pixels are inside the object."""
        return int(self.pixels.sum())


@dataclass
class SegmentRequest:
    """One question for `segment`: the object under some clicks, or inside a box."""

    id: str
    #: Clicks on the picture: label 1 = this object, 0 = not this.
    points: Sequence[Point] = field(default_factory=tuple)
    #: `(x0, y0, x1, y1)`, right and bottom exclusive.
    box: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        """Refuse a request that asks nothing."""
        if not self.points and self.box is None:
            raise ValueError(f"request {self.id!r} has neither points nor a box")


@dataclass
class SegmentFailure:
    """One request the segmenter could not answer; the others still were."""

    id: str
    message: str


class Segmenter(Protocol):
    """Finds object outlines on a picture."""

    name: str

    def suggest(self, image_path: Path) -> list[Mask]:
        """Every object the segmenter finds on its own, duplicates included."""
        ...

    def segment(
        self, image_path: Path, requests: Sequence[SegmentRequest]
    ) -> list[Mask | SegmentFailure]:
        """Answer each request, in order; a bad request costs only its own answer."""
        ...
