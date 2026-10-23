"""PixelMagic's own stand-in for the few OpenCV calls inside `vendor/sam2`.

This is NOT OpenCV. The vendored MLX port of SAM 2.1 does `import cv2` and uses
one thing from it on the path the object finder takes: `cv2.resize` (linear
and cubic). The vendored files must stay byte-for-byte unchanged (see
`vendor/NOTICE`), and shipping all of OpenCV for that drags in FFmpeg and a
long list of video libraries (see THIRD_PARTY.md), so this small module is
installed under the name `cv2` and answers that call with numpy.

The vendored video code also calls `cv2.connectedComponentsWithStats`
(`SAM2ImagePredictor.refine_init_with_click_cc`). The object finder never
calls it, so it is not provided: using it raises an `AttributeError`.

It reproduces OpenCV's own arithmetic, not merely "a resize": half-pixel
centres, borders repeated (never wrapped or mirrored), no antialiasing when
shrinking (Pillow's BILINEAR does antialias, so it is not used), and cubic
convolution with a = -0.75 (Pillow uses -0.5). Anything beyond that
raises, on purpose: a missing name is an `AttributeError`, an unsupported
argument a `ValueError`, so nothing quietly does something different.
"""

from __future__ import annotations

import numpy as np

INTER_LINEAR = 1
INTER_CUBIC = 2

#: OpenCV's cubic-convolution constant.
_CUBIC_A = -0.75


def _cubic_kernel(x: np.ndarray) -> np.ndarray:
    """OpenCV's cubic convolution kernel (a = -0.75) at distances `x`."""
    x = np.abs(x)
    a = _CUBIC_A
    near = ((a + 2) * x - (a + 3)) * x * x + 1
    far = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return np.where(x <= 1, near, np.where(x < 2, far, 0.0))


def _taps(src_size: int, dst_size: int, interpolation: int) -> tuple[np.ndarray, np.ndarray]:
    """For each output index, the source indices to read and their weights.

    Returns `(index, weight)`, both `dst_size x taps` (2 taps linear, 4 cubic).
    Indices past either end are clamped, which repeats the border pixel.
    """
    source = (np.arange(dst_size) + 0.5) * (src_size / dst_size) - 0.5
    first = np.floor(source).astype(np.int64)
    fraction = source - first
    if interpolation == INTER_LINEAR:
        offsets = np.array([0, 1])
        weight = np.stack([1 - fraction, fraction], axis=1)
    else:
        offsets = np.array([-1, 0, 1, 2])
        weight = _cubic_kernel(fraction[:, None] - offsets[None, :])
    index = np.clip(first[:, None] + offsets[None, :], 0, src_size - 1)
    return index, weight


def _resize_axis(data: np.ndarray, axis: int, size: int, interpolation: int) -> np.ndarray:
    """Resample `data` along one axis to `size`, in float64."""
    index, weight = _taps(data.shape[axis], size, interpolation)
    moved = np.moveaxis(data, axis, 0)
    shape = (size, index.shape[1]) + (1,) * (moved.ndim - 1)
    out = (moved[index] * weight.reshape(shape)).sum(axis=1)
    return np.moveaxis(out, 0, axis)


def resize(
    src: np.ndarray, dsize: tuple[int, int], interpolation: int = INTER_LINEAR
) -> np.ndarray:
    """Resize like `cv2.resize`: `dsize` is `(width, height)`; dtype is kept.

    Only `INTER_LINEAR` and `INTER_CUBIC` on uint8 or float32 arrays of shape
    `H x W` or `H x W x C` are supported. uint8 results are rounded and clipped.
    """
    if interpolation not in (INTER_LINEAR, INTER_CUBIC):
        raise ValueError(f"cv2 stand-in: interpolation {interpolation!r} is not supported")
    if src.dtype not in (np.uint8, np.float32) or src.ndim not in (2, 3):
        raise ValueError(f"cv2 stand-in: cannot resize {src.dtype} with shape {src.shape}")
    width, height = dsize
    data = src.astype(np.float64)
    data = _resize_axis(data, 1, width, interpolation)
    data = _resize_axis(data, 0, height, interpolation)
    if src.dtype == np.uint8:
        return np.clip(np.rint(data), 0, 255).astype(np.uint8)
    return data.astype(np.float32)
