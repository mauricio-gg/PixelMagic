"""Small image helpers shared by the object road: keying, boxes, blob labels.

Nothing here touches the store or a service — it takes a Pillow image and
returns Pillow images, boxes and arrays, so it can be tested against a
picture drawn in four lines of numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from pixelmagic.model import KEY_COLOURS, BackgroundPolicy

#: How far a pixel may sit from the key colour and still count as background.
#: Distance is plain Euclidean distance in RGB.
KEY_TOLERANCE = 40

#: The fringe pass uses a looser tolerance, but only on pixels that touch the
#: background. A model's anti-aliased edge blends the key colour into the
#: subject; those blended pixels are background, a pinkish detail in the middle
#: of the subject is not.
FRINGE_MULTIPLIER = 2.0

#: Anything below this alpha is treated as background; the output is always
#: fully opaque or fully transparent, never a halo in between.
ALPHA_THRESHOLD = 128


def _distance_to(rgb: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """Euclidean RGB distance from every pixel to `colour`."""
    difference = rgb.astype(np.float32) - np.asarray(colour, dtype=np.float32)
    return np.sqrt((difference**2).sum(axis=2))


def to_alpha(
    image: Image.Image,
    background: BackgroundPolicy,
    *,
    tolerance: int = KEY_TOLERANCE,
) -> Image.Image:
    """Return `image` as RGBA with a clean, binary alpha channel.

    For a transparent source this is just a threshold. For a keyed source the
    key colour is removed with `tolerance`, then one pass of fringe erosion
    drops the anti-aliased halo: pixels within twice the tolerance of the key
    that touch an already-transparent pixel.
    """
    rgba = image.convert("RGBA")
    array = np.asarray(rgba).copy()
    alpha = array[:, :, 3]

    if background is BackgroundPolicy.TRANSPARENT:
        array[:, :, 3] = np.where(alpha >= ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
        return Image.fromarray(array, "RGBA")

    key = KEY_COLOURS[background]
    distance = _distance_to(array[:, :, :3], key)
    transparent = (distance <= tolerance) | (alpha < ALPHA_THRESHOLD)

    # Grow the background by one pixel and keep only the overlap with the
    # near-key band: that is exactly the halo ring, and nothing inside the
    # subject can be reached by a single dilation from outside it.
    grown = ndimage.binary_dilation(transparent, structure=np.ones((3, 3), dtype=bool))
    fringe = grown & ~transparent & (distance <= tolerance * FRINGE_MULTIPLIER)

    array[:, :, 3] = np.where(transparent | fringe, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGBA")


def opaque_fraction(image: Image.Image) -> float:
    """What share of the image is opaque. Used to spot a keying that ate everything."""
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    return float((alpha > 0).mean())


#: A part smaller than this many opaque pixels is a stray speck, not an object.
MIN_PART_AREA = 64

#: 8-connectivity: pieces that touch only at a corner are still one piece.
_CONNECTIVITY = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class SliceBox:
    """A crop rectangle. `right` and `bottom` are exclusive, as Pillow wants."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return self.bottom - self.top

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        """The box as the 4-tuple `Image.crop` and `Cutout.bbox` both use."""
        return (self.left, self.top, self.right, self.bottom)


def label_parts(image: Image.Image) -> np.ndarray:
    """Number every connected blob of opaque pixels on a picture.

    Returns an integer array the shape of the image — `(height, width)` —
    that is 0 on every transparent pixel and the same positive number on
    every pixel of one blob. Blobs are 8-connected.
    """
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3] > 0
    labelled, _count = ndimage.label(alpha, structure=_CONNECTIVITY)
    return labelled


#: The warning code the removed seam check put on a centre tile. Nothing
#: produces it any more; it stays so an old `refine.json` that carries it
#: still loads (`refiner.base` lists it in `PIXELMAGIC_WARNING_CODES`) and
#: export can still explain it.
SEAM_VISIBLE = "SEAM_VISIBLE"
