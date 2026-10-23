import numpy as np
from PIL import Image

from pixelmagic.model import BackgroundPolicy
from pixelmagic.stages.slice import opaque_fraction, to_alpha


def test_the_key_colour_becomes_transparent_and_the_subject_stays():
    pixels = np.full((8, 8, 3), (255, 0, 255), dtype=np.uint8)
    pixels[3:5, 3:5] = (20, 40, 80)
    out = to_alpha(Image.fromarray(pixels, "RGB"), BackgroundPolicy.MAGENTA)
    alpha = np.asarray(out)[:, :, 3]
    assert alpha[0, 0] == 0
    assert alpha[3, 3] == 255


def test_a_near_key_fringe_pixel_touching_the_background_is_eroded():
    pixels = np.full((8, 8, 3), (255, 0, 255), dtype=np.uint8)
    pixels[3:5, 3:5] = (20, 40, 80)
    # A magenta-tinted halo pixel sitting directly above the blob: 60 away from
    # the key, so the plain tolerance of 40 keeps it but the fringe pass (2x)
    # removes it because it touches transparent background.
    pixels[2, 3] = (215, 20, 215)
    out = to_alpha(Image.fromarray(pixels, "RGB"), BackgroundPolicy.MAGENTA)
    assert np.asarray(out)[2, 3, 3] == 0


def test_a_near_key_pixel_enclosed_by_the_subject_survives():
    pixels = np.full((10, 10, 3), (255, 0, 255), dtype=np.uint8)
    pixels[2:8, 2:8] = (20, 40, 80)
    pixels[5, 5] = (215, 20, 215)  # a genuinely pinkish detail, fully enclosed
    out = to_alpha(Image.fromarray(pixels, "RGB"), BackgroundPolicy.MAGENTA)
    assert np.asarray(out)[5, 5, 3] == 255


def test_lime_is_keyed_too():
    pixels = np.full((6, 6, 3), (0, 255, 0), dtype=np.uint8)
    pixels[2:4, 2:4] = (20, 40, 80)
    alpha = np.asarray(to_alpha(Image.fromarray(pixels, "RGB"), BackgroundPolicy.LIME))[:, :, 3]
    assert alpha[0, 0] == 0
    assert alpha[2, 2] == 255


def test_transparent_input_is_thresholded_to_binary_alpha():
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[1:3, 1:3] = (10, 20, 30, 200)
    pixels[0, 0] = (10, 20, 30, 4)
    out = to_alpha(Image.fromarray(pixels, "RGBA"), BackgroundPolicy.TRANSPARENT)
    alpha = np.asarray(out)[:, :, 3]
    assert set(np.unique(alpha).tolist()) == {0, 255}
    assert alpha[1, 1] == 255
    assert alpha[0, 0] == 0


def test_opaque_fraction_reports_how_much_survived():
    pixels = np.zeros((10, 10, 4), dtype=np.uint8)
    pixels[0:5, :] = (1, 2, 3, 255)
    assert opaque_fraction(Image.fromarray(pixels, "RGBA")) == 0.5
