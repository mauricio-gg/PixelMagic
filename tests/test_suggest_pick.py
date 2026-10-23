"""Suggestions are tidied and numbered once; picks keep their numbers for good."""

from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import Direction, PickMethod, Point, Reviewer
from pixelmagic.segmenter.base import Mask, SegmentFailure
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.stages.decompose import missing_run_sentence
from pixelmagic.stages.generate import import_picture
from pixelmagic.stages.pick import (
    add_clicks,
    fix_pick,
    load_picks,
    pick_box,
    pick_points,
    pick_suggestions,
    rename_pick,
    render_suggestion_map,
    suggestions_under,
    undo_last,
    unpick,
)
from pixelmagic.stages.suggest import (
    load_suggestions,
    suggest_objects,
    suggestions_on,
)
from pixelmagic.store import Store

NOW = datetime(2026, 9, 30, 9, 0, 0, tzinfo=UTC)


class ManyMasks:
    """A segmenter double: 45 separate squares, each also returned once slightly shifted."""

    name = "double"

    def suggest(self, image_path):
        """Return 90 masks, the shifted copies first, so order alone cannot save the result."""
        squares, copies = [], []
        for index in range(45):
            side = 12 + index % 5
            x, y = (index % 16) * 20, (index // 16) * 20
            pixels = np.ones((side, side), dtype=bool)
            squares.append(Mask((x, y, x + side, y + side), pixels))
            copies.append(Mask((x + 1, y, x + 1 + side, y + side), pixels))
        return copies + squares


@pytest.fixture
def suggested(tmp_path):
    """A project with an approved picture and its suggestions from `ManyMasks`."""
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    original = tmp_path / "scene.png"
    Image.new("RGBA", (320, 220), (200, 120, 60, 255)).save(original)
    record = import_picture(store, direction, original, reviewer=Reviewer.HUMAN, now=NOW)
    made = suggest_objects(store, direction, record, ManyMasks(), now=NOW)
    return store, direction, made.id


def test_duplicates_are_removed_largest_first_and_numbers_do_not_move_between_pages(suggested):
    store, direction, decomposition_id = suggested
    found = load_suggestions(store, direction.name, decomposition_id)

    # 90 masks came in; each square's shifted copy is the same object and is gone.
    assert found.raw_count == 90
    assert len(found.suggestions) == 45
    areas = [suggestion.area for suggestion in found.suggestions]
    assert areas == sorted(areas, reverse=True)
    assert [s.number for s in found.suggestions] == list(range(1, 46))

    # Two pages; number 41 is the first thing on page 2.
    assert [s.number for s in suggestions_on(found, 2)] == [41, 42, 43, 44, 45]
    folder = store.suggestions_dir(direction.name, decomposition_id)
    assert (folder / "contact-01.png").is_file() and (folder / "contact-02.png").is_file()
    assert not (folder / "contact-03.png").exists()


class RoomAndThings:
    """A 100 x 100 picture: a floor a little over a fifth of it, a slab exactly a fifth, a mug."""

    name = "room"

    def suggest(self, image_path):
        """Return the floor (2100 px), the slab (2000 px) and a mug sitting on the slab."""
        floor = Mask((0, 0, 100, 21), np.ones((21, 100), dtype=bool))
        slab = Mask((0, 50, 100, 70), np.ones((20, 100), dtype=bool))
        mug = Mask((10, 55, 20, 65), np.ones((10, 10), dtype=bool))
        return [mug, floor, slab]


@pytest.fixture
def room(tmp_path):
    """A project whose suggestions come from `RoomAndThings`."""
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    original = tmp_path / "scene.png"
    Image.new("RGBA", (100, 100), (200, 120, 60, 255)).save(original)
    record = import_picture(store, direction, original, reviewer=Reviewer.HUMAN, now=NOW)
    made = suggest_objects(store, direction, record, RoomAndThings(), now=NOW)
    return store, direction, made.id


def test_a_suggestion_over_a_fifth_of_the_picture_is_dropped_as_background(room):
    """Catches: the room, walls and floor coming back as suggestions and burying every
    real object."""
    store, direction, decomposition_id = room
    found = load_suggestions(store, direction.name, decomposition_id)

    # The 2100 px floor is over a fifth of 10000 px and is gone; the 2000 px slab
    # is exactly a fifth and stays. Numbers are 1-based and biggest first.
    assert [(s.number, s.area) for s in found.suggestions] == [(1, 2000), (2, 100)]
    assert found.raw_count == 3


def test_the_suggestion_map_shows_the_smallest_suggestion_and_leaves_picked_ones_out(room):
    """Catches: the hover and click picking the slab instead of the mug on it, or
    offering an object that is already picked."""
    store, direction, decomposition_id = room

    def number_at(x, y):
        red, green, _blue, alpha = render_suggestion_map(
            store, direction, decomposition_id
        ).getpixel((x, y))
        assert alpha == 255
        return red * 256 + green

    assert number_at(15, 60) == 2  # the mug, not the slab under it
    assert number_at(50, 60) == 1  # the slab
    assert number_at(50, 10) == 0  # bare picture
    pick_suggestions(store, direction, decomposition_id, [2])
    assert number_at(15, 60) == 1  # the mug is picked: the cursor falls through to the slab


def test_a_pick_number_is_never_reused_after_an_unpick(suggested):
    store, direction, decomposition_id = suggested
    first, second = pick_suggestions(store, direction, decomposition_id, [5, 9])
    unpick(store, direction, decomposition_id, second.number)
    (third,) = pick_suggestions(store, direction, decomposition_id, [12])

    # Reusing 2 would let a later redraw overwrite the piece the old pick 2 made.
    assert (first.number, second.number, third.number) == (1, 2, 3)
    assert [pick.number for pick in load_picks(store, direction.name, decomposition_id).picks] == [
        1,
        3,
    ]


class RecordingSegmenter:
    """A segmenter double that notes every request it is sent and answers with a fixed square."""

    name = "recording"

    def __init__(self):
        self.requests = []
        self.fail = False

    def segment(self, image_path, requests):
        """Record the requests; answer a 10 x 10 square at (90, 90), or a failure when told to."""
        self.requests.extend(requests)
        if self.fail:
            return [SegmentFailure(request.id, "no object found there") for request in requests]
        return [Mask((90, 90, 100, 100), np.ones((10, 10), dtype=bool)) for _ in requests]


def test_a_point_within_12_px_of_an_earlier_same_label_point_is_ignored_the_other_label_is_kept(
    suggested,
):
    # Catches: repeated near-identical points reaching Segment Anything (it jumps to another
    # object), or the guard eating a legitimate "not this" click at the same spot.
    store, direction, decomposition_id = suggested
    double = RecordingSegmenter()
    clicks = [
        Point(x=100, y=100, label=1),
        Point(x=112, y=100, label=1),  # exactly 12 away: dropped
        Point(x=113, y=100, label=1),  # 13 away: kept
        Point(x=100, y=100, label=0),  # same spot, other label: kept
        Point(x=103, y=104, label=0),  # 5 from the earlier "not this": dropped
    ]
    expected = [(100, 100, 1), (113, 100, 1), (100, 100, 0)]

    made = pick_points(store, direction, decomposition_id, clicks, segmenter=double)
    assert [(p.x, p.y, p.label) for p in double.requests[0].points] == expected
    assert [(p.x, p.y, p.label) for p in made.points] == expected

    # A fix builds its point list the same way.
    fix_pick(store, direction, decomposition_id, made.number, points=clicks, segmenter=double)
    assert [(p.x, p.y, p.label) for p in double.requests[1].points] == expected


def test_a_shift_click_outside_a_suggestion_pick_box_grows_the_pick_past_it(suggested):
    # Catches: an Add-a-spot click on a pick made from a suggestion being clipped to that
    # suggestion's box (a bed picked as its pillow could never grow to the quilt), a
    # "not this" click widening the prompt, and a repeat within 12 px reaching the segmenter.
    store, direction, decomposition_id = suggested
    double = RecordingSegmenter()
    (chosen,) = pick_suggestions(store, direction, decomposition_id, [3], name="lamp")
    x0, y0, x1, y1 = chosen.picture_box
    assert x1 < 300  # the click below is well outside the suggestion's own box

    made = add_clicks(store, direction, decomposition_id, chosen.number,
                      [Point(x=300, y=200, label=1)], segmenter=double)
    sent = double.requests[0]
    sent_x0, sent_y0, sent_x1, sent_y1 = sent.box
    assert (sent_x0, sent_y0) <= (x0, y0)           # still holds the original object
    assert sent_x1 > 300 and sent_y1 > 200          # and reaches the click
    assert [(p.x, p.y, p.label) for p in sent.points] == [(300, 200, 1)]
    assert made.box == sent.box                     # the saved pick records what was asked
    assert (made.number, made.name) == (chosen.number, "lamp")

    # "Not this" goes through and does not widen the box; a repeat near an earlier
    # "this" is dropped.
    add_clicks(store, direction, decomposition_id, chosen.number,
               [Point(x=300, y=200, label=1), Point(x=10, y=10, label=0)], segmenter=double)
    again = double.requests[1]
    assert again.box == sent.box
    assert [(p.x, p.y, p.label) for p in again.points] == [(300, 200, 1), (10, 10, 0)]


def test_suggestions_under_a_point_come_smallest_first_by_mask_and_leave_out_picked(tmp_path):
    # Catches: scroll stepping to an object that is not under the cursor (a suggestion whose
    # box covers the point but whose mask has a hole there), skipping a bigger one
    # (the whole bed), or offering one that is already picked.
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    original = tmp_path / "scene.png"
    Image.new("RGBA", (100, 100), (200, 120, 60, 255)).save(original)
    record = import_picture(store, direction, original, reviewer=Reviewer.HUMAN, now=NOW)

    class Nested:
        name = "nested"

        def suggest(self, image_path):
            ring = np.ones((40, 40), dtype=bool)
            ring[10:30, 10:30] = False  # a hole where the point will be
            return [
                Mask((20, 20, 60, 60), np.ones((40, 40), dtype=bool)),  # whole: 1600
                Mask((30, 30, 40, 40), np.ones((10, 10), dtype=bool)),  # pillow: 100
                Mask((25, 25, 45, 45), np.ones((20, 20), dtype=bool)),  # quilt: 400
                Mask((20, 20, 60, 60), ring),                           # 1200: box only
                Mask((70, 70, 90, 90), np.ones((20, 20), dtype=bool)),  # elsewhere
            ]

    made = suggest_objects(store, direction, record, Nested(), now=NOW)
    found = load_suggestions(store, direction.name, made.id)
    area_of = {s.number: s.area for s in found.suggestions}

    def under(x, y):
        return [area_of[n] for n in suggestions_under(store, direction, made.id, x, y)]

    assert under(35, 35) == [100, 400, 1600]   # the ring has a hole at 35,35
    assert under(22, 22) == [1200, 1600]
    assert under(5, 5) == []

    smallest = suggestions_under(store, direction, made.id, 35, 35)[0]
    pick_suggestions(store, direction, made.id, [smallest])
    assert [area_of[n] for n in suggestions_under(store, direction, made.id, 35, 35)] == [400, 1600]


@pytest.fixture
def six_objects(tmp_path, object_picture):
    """The synthetic picture of separate objects, suggested with the offline stand-in."""
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    record = import_picture(store, direction, object_picture, reviewer=Reviewer.HUMAN, now=NOW)
    made = suggest_objects(store, direction, record, StubSegmenter(), now=NOW)
    return store, direction, made.id


def test_a_box_over_two_close_neighbours_with_a_not_this_point_picks_only_the_other(six_objects):
    # Catches: a "not this" click being ignored when a box is given, so the pick holds both
    # neighbours (objects 1 and 2 sit six pixels apart) and redraws as one lump.
    store, direction, decomposition_id = six_objects

    made = pick_box(
        store, direction, decomposition_id, (0, 0, 100, 50),
        points=[Point(x=70, y=25, label=0)], segmenter=StubSegmenter(),
    )

    assert made.method is PickMethod.BOX and made.box == (0, 0, 100, 50)
    assert made.picture_box == (10, 10, 50, 40)
    assert made.area == 40 * 30


def test_a_fix_replaces_the_outline_in_place_and_a_failed_fix_leaves_the_pick_alone(six_objects):
    # Catches: a fix that adds a second pick or renumbers, keeps the old outline, or (when the
    # segmenter finds nothing) wipes the pick that was there.
    store, direction, decomposition_id = six_objects
    (first,) = pick_suggestions(store, direction, decomposition_id, [3], name="lamp")
    folder = store.picks_dir(direction.name, decomposition_id)
    old_mask = (folder / first.mask).read_bytes()

    with pytest.raises(ValueError, match="no object found"):
        fix_pick(
            store, direction, decomposition_id, first.number,
            points=[Point(x=400, y=300, label=1)], segmenter=StubSegmenter(),
        )
    assert (folder / first.mask).read_bytes() == old_mask
    assert load_picks(store, direction.name, decomposition_id).picks == [first]

    fixed = fix_pick(
        store, direction, decomposition_id, first.number,
        points=[Point(x=70, y=25, label=1)], segmenter=StubSegmenter(),
    )

    (only,) = load_picks(store, direction.name, decomposition_id).picks
    assert only == fixed
    assert (only.number, only.name) == (first.number, "lamp")
    assert only.picture_box == (56, 10, 90, 44) and only.suggestion is None
    assert only.method is PickMethod.POINTS
    assert np.asarray(Image.open(folder / only.mask)).shape == (34, 34)


def test_undoing_a_fix_gives_back_the_old_outline_and_keeps_a_later_name(six_objects):
    # Catches: Undo leaving the pick on the outline the person just replaced, losing the
    # name they gave it since, or a refused fix leaving an entry that undoes nothing.
    store, direction, decomposition_id = six_objects
    (first,) = pick_suggestions(store, direction, decomposition_id, [3])
    folder = store.picks_dir(direction.name, decomposition_id)
    old_mask = (folder / first.mask).read_bytes()
    entries = len(load_picks(store, direction.name, decomposition_id).undo)

    with pytest.raises(ValueError, match="no object found"):
        fix_pick(
            store, direction, decomposition_id, first.number,
            points=[Point(x=400, y=300, label=1)], segmenter=StubSegmenter(),
        )
    assert len(load_picks(store, direction.name, decomposition_id).undo) == entries

    fix_pick(
        store, direction, decomposition_id, first.number,
        points=[Point(x=70, y=25, label=1)], segmenter=StubSegmenter(),
    )
    rename_pick(store, direction, decomposition_id, first.number, "lamp")
    undo_last(store, direction, decomposition_id)

    (back,) = load_picks(store, direction.name, decomposition_id).picks
    assert (folder / back.mask).read_bytes() == old_mask
    assert (back.method, back.suggestion, back.picture_box, back.area) == (
        first.method, first.suggestion, first.picture_box, first.area
    )
    assert back.name == "lamp"


def test_undoing_a_remove_brings_the_pick_back_under_its_own_number_with_its_mask(six_objects):
    # Catches: a removed pick coming back renumbered or at the end of the list (a later
    # redraw would then name the wrong piece), or coming back with no mask file.
    store, direction, decomposition_id = six_objects
    one, two, three = pick_suggestions(store, direction, decomposition_id, [1, 2, 3])
    folder = store.picks_dir(direction.name, decomposition_id)
    two_mask = (folder / two.mask).read_bytes()
    unpick(store, direction, decomposition_id, two.number)

    undo_last(store, direction, decomposition_id)

    found = load_picks(store, direction.name, decomposition_id)
    assert [pick.number for pick in found.picks] == [one.number, two.number, three.number]
    assert (folder / two.mask).read_bytes() == two_mask


def test_undoing_a_new_pick_removes_it_and_its_number_is_not_handed_out_again(six_objects):
    # Catches: Undo of a new pick leaving its mask behind, or rewinding last_number so the
    # next pick takes the number of a pick a redraw may already know.
    store, direction, decomposition_id = six_objects
    (first,) = pick_suggestions(store, direction, decomposition_id, [3])
    folder = store.picks_dir(direction.name, decomposition_id)
    undo_last(store, direction, decomposition_id)

    assert load_picks(store, direction.name, decomposition_id).picks == []
    assert not (folder / first.mask).exists()
    (next_pick,) = pick_suggestions(store, direction, decomposition_id, [3])
    assert next_pick.number == first.number + 1
    undo_last(store, direction, decomposition_id)
    with pytest.raises(ValueError, match="nothing to undo"):
        undo_last(store, direction, decomposition_id)


def test_undoing_a_new_pick_works_when_an_earlier_fix_already_made_the_undo_folder(six_objects):
    # Catches: undoing a new pick (which has no mask copy) trying to delete the undo folder
    # itself, so the person gets an "Operation not permitted" error for an undo that happened.
    store, direction, decomposition_id = six_objects
    (first,) = pick_suggestions(store, direction, decomposition_id, [3])
    fix_pick(
        store, direction, decomposition_id, first.number,
        points=[Point(x=70, y=25, label=1)], segmenter=StubSegmenter(),
    )
    (second,) = pick_suggestions(store, direction, decomposition_id, [1])

    assert undo_last(store, direction, decomposition_id) == second.number

    assert [p.number for p in load_picks(store, direction.name, decomposition_id).picks] == [
        first.number
    ]


def test_a_picture_id_typed_as_a_run_id_is_named_and_pointed_to_suggest(suggested):
    """Catches: an agent that passes the picture's id to a picking step being told only that
    the run is missing, with no hint that it must suggest objects first."""
    store, direction, _run_id = suggested
    picture_id = store.list_generations(direction.name)[0]

    mixed_up = missing_run_sentence(store, direction.name, picture_id, "suggest {id}")
    assert "picture id" in mixed_up and f"suggest {picture_id}" in mixed_up

    unknown = missing_run_sentence(store, direction.name, "20200101-000000", "suggest {id}")
    assert "picture id" not in unknown and "no cut-up run" in unknown
