"""Loading SAM 2.1 small (MLX build) and asking it for masks.

One `Sam` is one loaded model with one encoded picture. The helper runs one
fresh process per call, so there is no cache: a picture is encoded every time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

#: What the helper reports as its own name; it ends up in `decomposition.json`
#: and in the manifest, so a bundle says which model cut it.
SEGMENTER_NAME = "sam2.1-small-mlx"

#: The one weights file, downloaded by `fetch-model` from this Hugging Face repo.
WEIGHTS_REPO = "eisneim/sam2.1_mlx"
WEIGHTS_FILE = "sam2.1_hiera_small.safetensors"
WEIGHTS_SHA256 = "0b3a0198309acc9d1879d0a78aedc4adea10cf2fa9ff6ab173872fc6f4ae3eef"

#: `segmenter/weights/`: this file is `segmenter/src/sam_helper/model.py`.
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / WEIGHTS_FILE

#: SAM 2's convention for sending a box as two points.
LABEL_BOX_TOP_LEFT = 2
LABEL_BOX_BOTTOM_RIGHT = 3


class ImageUnreadable(Exception):
    """The file is there but is not a picture Pillow can open."""


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, read in pieces so a 184 MB file is never held whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    """Read a picture as an `H x W x 3` uint8 array.

    A picture with transparency is laid on white first: dropping the alpha
    channel would expose whatever colour the transparent pixels happen to hold,
    often black, and the model would see an object that is not there.
    """
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGBA")
    except Exception as error:
        raise ImageUnreadable(f"{path} could not be read as a picture: {error}") from error
    flat = Image.new("RGBA", image.size, (255, 255, 255, 255))
    flat.alpha_composite(image)
    return np.asarray(flat.convert("RGB"))


class Sam:
    """SAM 2.1 small with one picture encoded, ready to answer clicks."""

    def __init__(self, weights: Path = WEIGHTS_PATH) -> None:
        """Build the model from the weights file (the heavy imports happen here)."""
        from sam2.build import build_model
        from sam2.predictor import SAM2ImagePredictor

        self._predictor = SAM2ImagePredictor(build_model("small", str(weights)))

    def set_image(self, rgb: np.ndarray) -> None:
        """Encode the picture once; every later call answers from this encoding."""
        self._predictor.set_image(rgb)

    def best_mask(
        self, coords: list[tuple[float, float]], labels: list[int]
    ) -> tuple[np.ndarray, float]:
        """The model's best of three masks for these points, and its own score.

        `coords` are `(x, y)` on the picture; `labels` say what each point is
        (1 this object, 0 not this, 2 and 3 a box's corners). The mask is a
        boolean array the size of the picture.
        """
        masks, scores, _ = self._predictor.predict(
            np.array(coords, np.float32), np.array(labels, np.int32)
        )
        best = int(np.argmax(scores))
        # Logits above zero are probabilities above one half.
        return masks[best] > 0, float(scores[best])
