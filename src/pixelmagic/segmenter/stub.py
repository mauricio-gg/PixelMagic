"""A segmenter that needs no model: it reads the picture's own transparency.

`--segmenter stub` asks for this. An object is one 8-connected blob of opaque
pixels, the same sense `slice.label_parts` gives it, so a picture of separate
things on transparency is segmented exactly the way a person would point at
them. It is deterministic. A picture with no transparency is one blob, which
is honest: the stand-in cannot see edges inside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from pixelmagic.segmenter.base import Mask, SegmentFailure, SegmentRequest
from pixelmagic.stages.slice import MIN_PART_AREA, label_parts


def _mask_of(labels: np.ndarray, chosen: set[int]) -> Mask | None:
    """The union of the chosen blobs as one cropped mask, or None when empty."""
    inside = np.isin(labels, list(chosen))
    if not inside.any():
        return None
    rows = np.flatnonzero(inside.any(axis=1))
    columns = np.flatnonzero(inside.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    return Mask(box=(left, top, right, bottom), pixels=inside[top:bottom, left:right])


class StubSegmenter:
    """Stand-in `Segmenter`: one object per connected blob of opaque pixels."""

    name = "stub"

    def suggest(self, image_path: Path) -> list[Mask]:
        """Each blob of at least `MIN_PART_AREA` pixels, as its own mask."""
        labels = label_parts(Image.open(image_path))
        areas = np.bincount(labels.ravel())
        masks: list[Mask] = []
        for label in range(1, len(areas)):
            if areas[label] >= MIN_PART_AREA:
                mask = _mask_of(labels, {label})
                if mask is not None:
                    masks.append(mask)
        return masks

    def segment(
        self, image_path: Path, requests: Sequence[SegmentRequest]
    ) -> list[Mask | SegmentFailure]:
        """Answer each request from the picture's blobs.

        A box takes the blobs lying wholly inside it; a "this" point adds the
        blob under it; a "not this" point removes the blob under it. Nothing
        left means a `SegmentFailure` for that request alone.
        """
        labels = label_parts(Image.open(image_path))
        height, width = labels.shape
        answers: list[Mask | SegmentFailure] = []
        for request in requests:
            chosen: set[int] = set()
            removed: set[int] = set()
            if request.box is not None:
                x0, y0, x1, y1 = request.box
                for index, slices in enumerate(ndimage.find_objects(labels), start=1):
                    if slices is None:
                        continue
                    rows, columns = slices
                    if (
                        columns.start >= x0 and columns.stop <= x1
                        and rows.start >= y0 and rows.stop <= y1
                    ):
                        chosen.add(index)
            for point in request.points:
                under = (
                    int(labels[point.y, point.x])
                    if 0 <= point.x < width and 0 <= point.y < height
                    else 0
                )
                if under:
                    (chosen if point.label == 1 else removed).add(under)
            mask = _mask_of(labels, chosen - removed) if chosen - removed else None
            answers.append(mask or SegmentFailure(request.id, "no object found there"))
        return answers
