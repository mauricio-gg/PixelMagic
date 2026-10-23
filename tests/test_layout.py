import pytest

from pixelmagic.model import MAX_PITCH, MIN_PITCH
from pixelmagic.stages.layout import (
    check_anchor,
    check_pitch,
    default_anchor,
    footprint_for,
    target_for,
)
from pixelmagic.stages.slice import SliceBox


def test_a_target_is_the_box_divided_by_the_pitch_rounded():
    # The candle from the spec: a 68 x 128 box at pitch 4 is 17 x 32 art pixels.
    assert target_for(SliceBox(0, 0, 68, 128), 4.0) == (17, 32)
    # 115 / 4 = 28.75 -> 29; 61 / 4 = 15.25 -> 15.
    assert target_for(SliceBox(10, 10, 125, 71), 4.0) == (29, 15)


def test_a_target_is_never_below_one_on_either_side():
    assert target_for(SliceBox(0, 0, 3, 2), 8.0) == (1, 1)
    assert target_for(SliceBox(0, 0, 40, 2), 8.0) == (5, 1)


def test_a_pitch_of_one_keeps_the_box_size():
    """What the stand-in produces: every prop's target is its own box."""
    assert target_for(SliceBox(4, 4, 28, 28), 1.0) == (24, 24)


def test_a_footprint_rounds_up_to_whole_cells():
    assert footprint_for((17, 32), 16) == (2, 2)
    assert footprint_for((16, 16), 16) == (1, 1)
    assert footprint_for((33, 16), 16) == (3, 1)
    assert footprint_for((1, 1), 16) == (1, 1)


def test_the_default_anchor_is_bottom_centre():
    assert default_anchor((24, 40)) == (12, 40)
    assert default_anchor((16, 16)) == (8, 16)


@pytest.mark.parametrize("value", [MIN_PITCH - 0.01, 0.0, -1.0, MAX_PITCH + 0.01, 1000.0])
def test_a_pitch_outside_the_bounds_is_refused(value):
    with pytest.raises(ValueError, match="pitch"):
        check_pitch(value)


@pytest.mark.parametrize("value", [MIN_PITCH, 1.0, 4.25, MAX_PITCH])
def test_a_pitch_inside_the_bounds_is_accepted(value):
    check_pitch(value)


@pytest.mark.parametrize("anchor", [(0, 0), (17, 32), (8, 32), (0, 32), (17, 0)])
def test_an_anchor_anywhere_on_or_inside_the_target_is_accepted(anchor):
    check_anchor(anchor, (17, 32))


@pytest.mark.parametrize("anchor", [(18, 32), (8, 33), (-1, 0), (0, -1)])
def test_an_anchor_outside_the_target_is_refused(anchor):
    with pytest.raises(ValueError, match="anchor"):
        check_anchor(anchor, (17, 32))
