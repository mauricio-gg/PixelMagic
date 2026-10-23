"""The lasso keeps the object inside the line, and only that, whatever runs under the line."""

from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import Direction, Reviewer
from pixelmagic.segmenter.base import Mask, SegmentFailure
from pixelmagic.stages.generate import import_picture
from pixelmagic.stages.lasso import fix_lasso, pick_lasso
from pixelmagic.stages.pick import load_picks
from pixelmagic.stages.suggest import suggest_objects
from pixelmagic.store import Store

NOW = datetime(2026, 9, 30, 9, 0, 0, tzinfo=UTC)
SIZE = 200
#: The line drawn in every test: a square from (40, 40) to (160, 160).
LOOP = [(40, 40), (160, 40), (160, 160), (40, 160)]


def rect(x0, y0, x1, y1):
    """A picture-sized region: True for x0 <= x < x1 and y0 <= y < y1."""
    region = np.zeros((SIZE, SIZE), dtype=bool)
    region[y0:y1, x0:x1] = True
    return region


class Regions:
    """A segmenter double: a point is answered with the smallest region that holds it.

    Regions may overlap (a floor under a rug), which the stand-in segmenter cannot model.
    A point in no region is a failure. `calls` records how many requests each call carried.
    """

    name = "regions"

    def __init__(self, *regions):
        """Keep the regions, smallest first."""
        self.regions = sorted(regions, key=lambda region: region.sum())
        self.calls = []

    def suggest(self, image_path):
        """Suggest nothing: the picture only needs to exist."""
        return []

    def segment(self, image_path, requests):
        """Answer every request with the region under its first point."""
        self.calls.append(len(requests))
        answers = []
        for request in requests:
            point = request.points[0]
            region = next((r for r in self.regions if r[point.y, point.x]), None)
            if region is None:
                answers.append(SegmentFailure(request.id, "nothing there"))
                continue
            rows, columns = np.flatnonzero(region.any(axis=1)), np.flatnonzero(region.any(axis=0))
            box = (int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)
            answers.append(Mask(box, region[box[1]:box[3], box[0]:box[2]].copy()))
        return answers


@pytest.fixture
def scene(tmp_path):
    """A project holding a 200 x 200 picture with a finished `suggest` step."""
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    original = tmp_path / "scene.png"
    Image.new("RGBA", (SIZE, SIZE), (200, 120, 60, 255)).save(original)
    record = import_picture(store, direction, original, reviewer=Reviewer.HUMAN, now=NOW)
    made = suggest_objects(store, direction, record, Regions(), now=NOW)
    return store, direction, made.id


def saved_mask(store, direction, decomposition_id, pick):
    """A pick's saved mask back on a picture-sized canvas."""
    canvas = np.zeros((SIZE, SIZE), dtype=bool)
    x0, y0, x1, y1 = pick.picture_box
    folder = store.picks_dir(direction.name, decomposition_id)
    canvas[y0:y1, x0:x1] = np.asarray(Image.open(folder / pick.mask)) > 0
    return canvas


def test_lasso_drops_what_the_line_cuts_through_keeps_the_object_and_never_leaves_the_loop(scene):
    store, direction, decomposition_id = scene
    inside_line = rect(40, 40, 161, 161)
    bed = rect(70, 70, 130, 130)
    mug = rect(80, 135, 120, 155)
    # Hangs 5 columns over the line: 13.9% outside, just under the 15% rule.
    near_edge = rect(130, 100, 166, 130)
    # Hangs 5 columns over the line too, but it is only 30 wide: 16.7% outside, just over.
    crosses = rect(136, 140, 166, 150)
    assert (near_edge & ~inside_line).sum() / near_edge.sum() < 0.15
    assert (crosses & ~inside_line).sum() / crosses.sum() > 0.15
    # The floor runs under the line all round and is everything else.
    floor = ~(bed | mug | near_edge | crosses)
    segmenter = Regions(floor, bed, mug, near_edge, crosses)

    pick = pick_lasso(store, direction, decomposition_id, LOOP, segmenter=segmenter)

    kept = saved_mask(store, direction, decomposition_id, pick)
    # The object and the mug stay; the piece just under 15% out stays but is clipped to the line.
    assert np.array_equal(kept, bed | mug | (near_edge & inside_line))
    assert not (kept & ~inside_line).any()
    assert pick.area == int(kept.sum())
    assert len(segmenter.calls) == 1


def test_a_piece_mostly_covered_by_background_is_dropped_the_middle_piece_survives_15_pct(scene):
    store, direction, decomposition_id = scene
    inside_line = rect(40, 40, 161, 161)
    # The piece under the loop's middle hangs far over the line (31% outside) and must be kept.
    bed = rect(70, 70, 200, 130)
    assert (bed & ~inside_line).sum() / bed.sum() > 0.15
    # Two strips wholly inside the line, sitting on the floor: one 83% floor, one 73%.
    covered = rect(45, 135, 75, 155)
    uncovered = rect(45, 45, 75, 65)
    floor = ~bed & ~rect(45, 135, 50, 155) & ~rect(67, 45, 75, 65)
    assert (covered & floor).sum() / covered.sum() > 0.8
    assert (uncovered & floor).sum() / uncovered.sum() < 0.8
    segmenter = Regions(floor, bed, covered, uncovered)

    pick = pick_lasso(store, direction, decomposition_id, LOOP, segmenter=segmenter)

    kept = saved_mask(store, direction, decomposition_id, pick)
    assert np.array_equal(kept, (bed & inside_line) | uncovered)


def test_fix_ups_add_past_the_line_remove_and_re_add_a_piece_and_ignore_a_repeat_click(scene):
    store, direction, decomposition_id = scene
    bed = rect(70, 70, 130, 130)
    mug = rect(80, 135, 120, 155)
    lamp = rect(170, 90, 190, 110)  # wholly outside the line
    segmenter = Regions(~(bed | mug | lamp), bed, mug, lamp)
    pick = pick_lasso(store, direction, decomposition_id, LOOP, segmenter=segmenter)
    calls_after_pick = len(segmenter.calls)

    added = fix_lasso(store, direction, decomposition_id, pick.number, 180, 100,
                      add=True, segmenter=segmenter)
    # Not clipped to the line: the lamp is added whole, and the box grows to hold it.
    assert np.array_equal(saved_mask(store, direction, decomposition_id, added),
                          bed | mug | lamp)
    assert added.picture_box[2] == 190

    # 6 px from the first click, same kind: nothing changes and the segmenter is not asked.
    again = fix_lasso(store, direction, decomposition_id, pick.number, 184, 104,
                      add=True, segmenter=segmenter)
    assert len(segmenter.calls) == calls_after_pick + 1
    assert [(f.x, f.y, f.label) for f in again.fixes] == [(180, 100, 1)]

    removed = fix_lasso(store, direction, decomposition_id, pick.number, 100, 100,
                        add=False, segmenter=segmenter)
    assert np.array_equal(saved_mask(store, direction, decomposition_id, removed), mug | lamp)

    # Remove the lamp, then click where it was added first: it comes back, because the
    # earlier add is behind a remove and no longer guards that spot.
    fix_lasso(store, direction, decomposition_id, pick.number, 180, 100,
              add=False, segmenter=segmenter)
    back = fix_lasso(store, direction, decomposition_id, pick.number, 180, 100,
                     add=True, segmenter=segmenter)
    assert np.array_equal(saved_mask(store, direction, decomposition_id, back), mug | lamp)
    stored = load_picks(store, direction.name, decomposition_id).picks[0]
    assert stored.area == int((mug | lamp).sum())
