"""The automatic grid: click the model all over the picture, keep what is an object.

It removes no duplicates, on purpose: that is a PixelMagic rule, in the core.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from sam_helper.model import Sam

DEFAULT_SPACING = 32
DEFAULT_MIN_AREA = 200
DEFAULT_MAX_FRACTION = 0.35


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """The box round a mask's True pixels, right and bottom exclusive; None if empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(mask.any(axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def write_mask(mask: np.ndarray, box: tuple[int, int, int, int], path: Path) -> None:
    """Write a mask cropped to its box as an 8-bit PNG: white inside, black outside."""
    x0, y0, x1, y1 = box
    crop = mask[y0:y1, x0:x1]
    Image.fromarray(np.where(crop, 255, 0).astype(np.uint8), "L").save(path)


def grid_points(width: int, height: int, spacing: int) -> Iterator[tuple[int, int]]:
    """Points `spacing` px apart over the whole picture, the first half a step in."""
    for y in range(spacing // 2, height, spacing):
        for x in range(spacing // 2, width, spacing):
            yield x, y


def suggest(
    sam: Sam,
    rgb: np.ndarray,
    out_dir: Path,
    *,
    spacing: int = DEFAULT_SPACING,
    min_area: int = DEFAULT_MIN_AREA,
    max_fraction: float = DEFAULT_MAX_FRACTION,
) -> list[dict]:
    """Click every grid point and write each mask worth keeping to `out_dir`.

    Each mask goes to disk the moment it is made, so the many full-size masks
    are never held in memory together. Returns the reply's `suggestions`.
    """
    height, width = rgb.shape[:2]
    biggest = max_fraction * width * height
    sam.set_image(rgb)
    suggestions: list[dict] = []
    for x, y in grid_points(width, height, spacing):
        mask, score = sam.best_mask([(x, y)], [1])
        area = int(mask.sum())
        if area < min_area or area > biggest:
            continue
        box = mask_box(mask)
        name = f"{len(suggestions):04d}.png"
        write_mask(mask, box, out_dir / name)
        suggestions.append(
            {"id": len(suggestions), "box": list(box), "area": area, "score": score, "mask": name}
        )
    print(f"suggest: {len(suggestions)} masks kept", file=sys.stderr)
    return suggestions
