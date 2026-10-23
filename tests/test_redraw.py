"""Redrawing picks in groups: a failed group costs its own objects; sizes come from the original."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pixelmagic.model import (
    BackgroundPolicy,
    Decomposition,
    Direction,
    Group,
    PitchSource,
    PromptKind,
    Reviewer,
)
from pixelmagic.providers.base import ServiceError
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.stages.decompose import load_decomposition, save_decomposition
from pixelmagic.stages.generate import import_picture
from pixelmagic.stages.pick import load_picks, pick_suggestions, rename_pick, undo_last
from pixelmagic.stages.redraw import (
    SCALE_PHRASE,
    TARGET_PHRASE,
    redraw_picks,
    set_prop_scale,
    unowned_notes,
)
from pixelmagic.stages.suggest import suggest_objects
from pixelmagic.store import Store
from test_rebuild import draw_chunky_piece

NOW = datetime(2026, 9, 30, 9, 0, 0, tzinfo=UTC)


def _project_with_picks(tmp_path, boxes, size):
    """A project whose imported picture holds one separate blob per box, all picked."""
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    picture = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(picture)
    for index, box in enumerate(boxes):
        colour = (40 + index * 20, 90, 200, 255)
        draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=colour)
    path = tmp_path / "scene.png"
    picture.save(path)
    record = import_picture(store, direction, path, reviewer=Reviewer.HUMAN, now=NOW)
    made = suggest_objects(store, direction, record, StubSegmenter(), now=NOW)
    pick_suggestions(store, direction, made.id, list(range(1, len(boxes) + 1)))
    return store, direction, made.id


class TroubledService(FakeImageService):
    """The fake, but group-01's transparent reply comes back opaque and group-02 always fails."""

    def generate(self, request, out_path, *, reference_paths=()):
        """Fail or spoil a group's reply according to the folder its reference sits in."""
        if request.kind is PromptKind.GROUP_REDRAW:
            group = reference_paths[0].parent.name
            if group == "group-02":
                raise ServiceError("no picture today", service="fake")
            if group == "group-01" and request.background is BackgroundPolicy.TRANSPARENT:
                result = super().generate(request, out_path, reference_paths=reference_paths)
                Image.new("RGBA", request.size, (10, 200, 10, 255)).save(out_path)
                return result
        return super().generate(request, out_path, reference_paths=reference_paths)


def test_a_dirty_reply_is_retried_on_the_key_colour_and_a_failed_group_costs_only_its_own_objects(
    tmp_path,
):
    boxes = [
        (col * 120 + 10, row * 180 + 10, col * 120 + 10 + 30 + 9 * (row * 4 + col),
         row * 180 + 10 + 30 + 6 * (row * 4 + col))
        for row in range(2) for col in range(4)
    ]
    store, direction, decomposition_id = _project_with_picks(tmp_path, boxes, (480, 360))

    result = redraw_picks(
        store, direction, decomposition_id, service=TroubledService()
    )

    picks = load_picks(store, direction.name, decomposition_id).picks
    by_pick = {c.origin.pick: c for c in result.cutouts}
    # 8 picks make two groups of 4, smallest first: group-01 the four smallest objects,
    # group-02 the four biggest.
    smallest = {p.number for p in sorted(picks, key=lambda p: p.area)[:4]}
    biggest = {p.number for p in picks} - smallest
    # The retried group's four pieces exist; the failed group's do not.
    assert set(by_pick) == smallest
    assert len({c.bbox for c in by_pick.values()}) == 4, "the opaque reply was not cut as one blob"
    group = store.read_model(
        store.group_dir(direction.name, decomposition_id, "group-01") / "group.json",
        Group,
    )
    assert [(a.background, a.ok) for a in group.attempts] == [
        (BackgroundPolicy.TRANSPARENT, False),
        (BackgroundPolicy.MAGENTA, True),
    ]
    named = {p.number: p.name for p in picks}
    for number in biggest:
        assert any(named[number] in w and "not redrawn" in w for w in result.warnings)
    assert {p.number for p in picks} == smallest | biggest, "failed objects stay picked"

    # Running it again redraws only what has no piece: one group, four calls' worth of pieces.
    plain = FakeImageService()
    again = redraw_picks(store, direction, decomposition_id, service=plain)
    assert {c.origin.pick for c in again.cutouts} == smallest | biggest
    assert len(plain.reference_paths_seen) == 1
    assert not any("not redrawn" in warning for warning in again.warnings)


def _chunky_cells(request):
    """A group reply with one chunky 9 x 7 cell piece (6.5 x 6.5 px per cell) in each cell."""
    image = Image.new("RGBA", request.size, (0, 0, 0, 0))
    cell_w, cell_h = request.size[0] // request.group.cols, request.size[1] // request.group.rows
    for index in range(request.group.count):
        row, col = divmod(index, request.group.cols)
        piece, _, _ = draw_chunky_piece(cells=(9, 7), pitch=(6.5, 6.5))
        image.paste(
            piece,
            (
                col * cell_w + (cell_w - piece.width) // 2,
                row * cell_h + (cell_h - piece.height) // 2,
            ),
        )
    return image


def _noise_cells(request):
    """A group reply with a big block of random pixels in each cell: no grid anywhere."""
    rng = np.random.default_rng(1)
    image = Image.new("RGBA", request.size, (0, 0, 0, 0))
    cell_w, cell_h = request.size[0] // request.group.cols, request.size[1] // request.group.rows
    for index in range(request.group.count):
        row, col = divmod(index, request.group.cols)
        inset_x, inset_y = cell_w // 8, cell_h // 8
        noise = rng.integers(
            0, 256, (cell_h - 2 * inset_y, cell_w - 2 * inset_x, 4), dtype=np.uint8
        )
        noise[..., 3] = 255
        image.paste(
            Image.fromarray(noise, "RGBA"), (col * cell_w + inset_x, row * cell_h + inset_y)
        )
    return image


def test_a_rebuilt_pieces_target_is_its_native_size_and_its_pitch_is_its_own(
    tmp_path,
):
    boxes = [(10, 10, 90, 50), (120, 10, 160, 130), (200, 10, 264, 74), (10, 160, 170, 256)]
    store, direction, decomposition_id = _project_with_picks(tmp_path, boxes, (400, 300))

    result = redraw_picks(
        store, direction, decomposition_id,
        service=FakeImageService(painter=_chunky_cells),
    )

    assert len(result.cutouts) == 4
    for cutout in result.cutouts:
        left, top, right, bottom = cutout.origin.picture_box
        longer = max(right - left, bottom - top)
        # The piece is saved at the grid's native size (the 9 x 7 cells the painter drew),
        # which is also its target.
        assert cutout.grid_rebuild.native_size == (9, 7)
        assert cutout.target == (9, 7)
        with Image.open(
            store.cutout_dir(direction.name, decomposition_id, cutout.name) / "source.png"
        ) as saved:
            assert saved.size == cutout.target
        # Pitch is the object's longer side over the native size.
        assert cutout.pitch == pytest.approx(longer / max(cutout.target))
        assert cutout.origin.scale == cutout.pitch
        assert cutout.pitch_source is PitchSource.MEASURED
        assert cutout.grid_rebuild is not None


def _big_chunky_cells(request):
    """Like `_chunky_cells`, but each piece blown up 4x (nearest neighbour) so it is big enough
    to count in a lone object's wide cell. The rebuild then reads the 4 px sub-grid of the
    blown-up picture, not the 9 x 7 blocks, so native sizes are about 59 x 49."""
    image = Image.new("RGBA", request.size, (0, 0, 0, 0))
    cell_w, cell_h = request.size[0] // request.group.cols, request.size[1] // request.group.rows
    for index in range(request.group.count):
        row, col = divmod(index, request.group.cols)
        piece, _, _ = draw_chunky_piece(cells=(9, 7), pitch=(6.5, 6.5))
        piece = piece.resize((piece.width * 4, piece.height * 4), Image.Resampling.NEAREST)
        image.paste(
            piece,
            (
                col * cell_w + (cell_w - piece.width) // 2,
                row * cell_h + (cell_h - piece.height) // 2,
            ),
        )
    return image


def _noise_in_first_cell(request):
    """A group reply: noise (no grid) in cell 0, a good chunky piece in every other cell."""
    image = _big_chunky_cells(request)
    noise = _noise_cells(request)
    cell_w, cell_h = request.size[0] // request.group.cols, request.size[1] // request.group.rows
    box = (0, 0, cell_w, cell_h)
    image.paste(noise.crop(box), box[:2])
    return image


def test_an_object_with_no_grid_stays_picked_is_retried_and_never_costs_a_good_piece(tmp_path):
    # Catches: a failed second try deleting a good first piece, or a no-grid object dropped
    # from the picks or never picked up again by a plain redraw.
    boxes = [(10, 10, 60, 60), (100, 10, 150, 60), (200, 10, 250, 60)]
    store, direction, decomposition_id = _project_with_picks(tmp_path, boxes, (300, 200))

    first = redraw_picks(
        store, direction, decomposition_id,
        service=FakeImageService(painter=_noise_in_first_cell),
    )

    names = [p.name for p in load_picks(store, direction.name, decomposition_id).picks]
    assert len(names) == 3, "every object stays picked"
    landed = {c.name for c in first.cutouts}
    (bad,) = set(names) - landed
    assert len(landed) == 2
    refused = [w for w in first.warnings if w.startswith(f"{bad}: ")]
    assert len(refused) == 1
    assert float(refused[0].split("(confidence ")[1].split(" ")[0]) < 0.030

    def source_bytes(name):
        """The saved rebuilt picture of one piece, as bytes."""
        folder = store.cutout_dir(direction.name, decomposition_id, name)
        return (folder / "source.png").read_bytes()

    good_before = {name: source_bytes(name) for name in landed}

    # A plain redraw picks up only the object with no piece, and the good painter lands it.
    second = redraw_picks(
        store, direction, decomposition_id,
        service=FakeImageService(painter=_big_chunky_cells),
    )
    assert {c.name for c in second.cutouts} == set(names)
    assert {name: source_bytes(name) for name in landed} == good_before
    assert not any(w.startswith(f"{bad}: ") for w in second.warnings), "stale warnings stay"
    landed_bytes = source_bytes(bad)
    cutout_json = store.cutout_dir(direction.name, decomposition_id, bad) / "cutout.json"
    cutout_before = cutout_json.read_bytes()
    entry_before = next(c for c in second.cutouts if c.name == bad)

    # Asked again, the object comes back with no grid: its piece stays, only a warning is added.
    third = redraw_picks(
        store, direction, decomposition_id, only=[bad],
        service=FakeImageService(painter=_noise_in_first_cell),
    )
    assert source_bytes(bad) == landed_bytes
    assert cutout_json.read_bytes() == cutout_before
    assert next(c for c in third.cutouts if c.name == bad) == entry_before
    assert len([w for w in third.warnings if w.startswith(f"{bad}: no pixel grid")]) == 1

    # A second failed try leaves one current warning for it, not two.
    fourth = redraw_picks(
        store, direction, decomposition_id, only=[bad],
        service=FakeImageService(painter=_noise_in_first_cell),
    )
    assert len([w for w in fourth.warnings if w.startswith(f"{bad}: no pixel grid")]) == 1


def _double_blocks_in_first_cell(request):
    """A group reply: cell 0 holds a piece drawn with twice the blocks (blown up 2x) of the
    others. The rebuild reads sub-grids, so its native size comes out about twice the others'."""
    image = _big_chunky_cells(request)
    cell_w, cell_h = request.size[0] // request.group.cols, request.size[1] // request.group.rows
    piece, _, _ = draw_chunky_piece(cells=(18, 14), pitch=(6.5, 6.5))
    piece = piece.resize((piece.width * 2, piece.height * 2), Image.Resampling.NEAREST)
    box = (0, 0, cell_w, cell_h)
    image.paste(Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0)), box[:2])
    image.paste(piece, ((cell_w - piece.width) // 2, (cell_h - piece.height) // 2))
    return image


def test_a_piece_drawn_at_another_scale_is_flagged_until_it_is_redrawn_and_two_pieces_flag_nothing(
    tmp_path,
):
    # Catches: a piece drawn at the wrong scale going into the bundle unflagged, and a stale
    # scale warning outliving its fix (or surviving when too few pieces remain to compare).
    boxes = [(10 + 55 * i, 10, 60 + 55 * i, 60) for i in range(5)]
    store, direction, decomposition_id = _project_with_picks(tmp_path, boxes, (300, 100))

    result = redraw_picks(
        store, direction, decomposition_id,
        service=FakeImageService(painter=_double_blocks_in_first_cell),
    )

    assert len(result.cutouts) == 5
    flagged = [w for w in result.warnings if SCALE_PHRASE in w]
    assert len(flagged) == 1
    big = next(c for c in result.cutouts if c.origin.cell == (0, 0))
    assert flagged[0].startswith(f"{big.name}: "), "the flagged piece is not the planted one"
    before = {c.name: c for c in result.cutouts}

    # Redrawn by name at the usual scale, the piece is back in line and its warning is gone.
    again = redraw_picks(
        store, direction, decomposition_id, only=[big.name],
        service=FakeImageService(painter=_big_chunky_cells),
    )
    assert [c.name for c in again.cutouts] == [c.name for c in result.cutouts]
    assert len({c.grid_rebuild.native_size for c in again.cutouts}) == 1
    assert not any(SCALE_PHRASE in w for w in again.warnings)
    assert all(c == before[c.name] for c in again.cutouts if c.name != big.name)

    # With two rebuilt pieces there is nothing to compare: an earlier scale warning is
    # cleared, and a warning of any other kind is left alone.
    small = tmp_path / "small"
    small.mkdir()
    store2, direction2, id2 = _project_with_picks(small, boxes[:2], (300, 100))
    stale = load_decomposition(store2, direction2.name, id2)
    stale.warnings += [f"x: drawn about 2.0x too big {SCALE_PHRASE} (old)", "x: another note"]
    save_decomposition(store2, direction2.name, stale)
    done = redraw_picks(
        store2, direction2, id2,
        service=FakeImageService(painter=_double_blocks_in_first_cell),
    )
    assert len(done.cutouts) == 2
    assert not any(SCALE_PHRASE in w for w in done.warnings)
    assert "x: another note" in done.warnings


def test_a_retry_redraw_does_not_redraw_an_object_already_inside_a_merged_piece(tmp_path):
    boxes = [(10, 10, 60, 60), (100, 10, 140, 80), (200, 20, 280, 60)]
    store, direction, decomposition_id = _project_with_picks(tmp_path, boxes, (300, 200))
    made = redraw_picks(
        store, direction, decomposition_id, service=FakeImageService()
    )
    # An older M5 decomposition holds a merged piece like this: one cutout carrying both picks.
    first, second = made.cutouts[:2]
    joined = first.model_copy(
        update={
            "name": "joined",
            "origin": first.origin.model_copy(
                update={"merged_picks": [first.origin.pick, second.origin.pick]}
            ),
        }
    )
    made.cutouts = [joined, *made.cutouts[2:]]
    save_decomposition(store, direction.name, made)

    plain = FakeImageService()
    with pytest.raises(ValueError, match="already has a piece"):
        redraw_picks(store, direction, decomposition_id, service=plain)
    # Redrawing them would spend a call and bring back a duplicate of the merged piece.
    assert plain.reference_paths_seen == []


def test_renaming_a_pick_that_already_has_a_piece_is_refused(tmp_path):
    # Catches: a pick and its piece silently disagreeing on the name, so a later redraw by
    # name misses the piece.
    store, direction, decomposition_id = _project_with_picks(
        tmp_path, [(10, 10, 60, 60), (100, 10, 140, 80)], (300, 200)
    )
    redraw_picks(
        store, direction, decomposition_id, service=FakeImageService()
    )
    before = load_picks(store, direction.name, decomposition_id).picks[0]

    with pytest.raises(ValueError, match="already has a piece"):
        rename_pick(store, direction, decomposition_id, before.number, "something-else")
    assert load_picks(store, direction.name, decomposition_id).picks[0].name == before.name


def test_a_redraw_empties_undo_so_an_old_outline_cannot_come_back_under_a_piece(tmp_path):
    # Catches: Undo after a redraw putting back an outline the redrawn piece was not made
    # from, leaving the pick and its piece disagreeing.
    store, direction, decomposition_id = _project_with_picks(
        tmp_path, [(10, 10, 60, 60), (100, 10, 140, 80)], (300, 200)
    )
    assert load_picks(store, direction.name, decomposition_id).undo

    redraw_picks(store, direction, decomposition_id, service=FakeImageService())

    assert load_picks(store, direction.name, decomposition_id).undo == []
    with pytest.raises(ValueError, match="nothing to undo"):
        undo_last(store, direction, decomposition_id)


def test_a_hand_named_pick_takes_the_name_its_piece_was_given(tmp_path):
    # Catches: a pick named "lamp" whose piece became "lamp-2" (the name was taken) keeping
    # the old name, so its failure notes, written under "lamp-2", never reach its row.
    store, direction, decomposition_id = _project_with_picks(
        tmp_path, [(10, 10, 60, 60)], (300, 200)
    )
    taken = store.asset_dir(direction.name, "lamp")
    taken.mkdir(parents=True)
    Image.new("RGBA", (4, 4), (1, 2, 3, 255)).save(taken / "source.png")
    rename_pick(store, direction, decomposition_id, 1, "lamp")

    cut = redraw_picks(store, direction, decomposition_id, service=FakeImageService())

    (pick,) = load_picks(store, direction.name, decomposition_id).picks
    assert pick.name == "lamp-2"
    assert [c.name for c in cut.cutouts] == ["lamp-2"]


def test_a_hand_named_pick_never_takes_another_picks_name(tmp_path):
    # Catches: pick "bed" (name taken by an asset) being renamed "bed-2" while another pick
    # already holds "bed-2", leaving two picks with one name so redraws by name and notes cross.
    store, direction, decomposition_id = _project_with_picks(
        tmp_path, [(10, 10, 60, 60), (100, 10, 140, 80)], (300, 200)
    )
    taken = store.asset_dir(direction.name, "bed")
    taken.mkdir(parents=True)
    Image.new("RGBA", (4, 4), (1, 2, 3, 255)).save(taken / "source.png")
    rename_pick(store, direction, decomposition_id, 1, "bed")
    rename_pick(store, direction, decomposition_id, 2, "bed-2")

    redraw_picks(store, direction, decomposition_id, service=FakeImageService(), only=["1"])

    names = [p.name for p in load_picks(store, direction.name, decomposition_id).picks]
    assert len(set(names)) == 2
    assert names[1] == "bed-2"


def test_notes_no_pick_owns_are_returned_and_a_longer_name_does_not_claim_a_shorter_ones():
    """Catches: a note about a removed object, or one with no name, never being shown."""
    cut = Decomposition(
        id="d", created=NOW, source_generation="g",
        warnings=[
            "bed: no pixel grid found (score 0.2)",
            "bed-2: no pixel grid found (score 0.1)",
            "lamp: not redrawn after two tries",
            "the picture is small",
        ],
    )
    assert unowned_notes(cut, ["bed"]) == [
        (True, "bed-2: no pixel grid found (score 0.1)"),
        (True, "lamp: not redrawn after two tries"),
        (False, "the picture is small"),
    ]


def test_the_prop_scale_comes_from_one_picks_height_and_bad_input_is_refused_unsaved(tmp_path):
    # Catches: a wrong scale saved (every prop target in the direction would be off), or a
    # refused request still changing the direction.
    boxes = [(10, 10, 90, 50), (120, 10, 160, 130)]
    store, direction, decomposition_id = _project_with_picks(tmp_path, boxes, (400, 300))
    tall = next(p for p in load_picks(store, direction.name, decomposition_id).picks
                if p.picture_box[3] - p.picture_box[1] == 120)

    saved = set_prop_scale(store, direction, decomposition_id, str(tall.number), 60)

    assert saved.prop_scale == pytest.approx(2.0)
    assert store.load_direction(direction.name).prop_scale == pytest.approx(2.0)
    for bad_pick, bad_height in [(str(tall.number), 0.5), ("99", 60), (str(tall.number), 1)]:
        with pytest.raises(ValueError):
            set_prop_scale(store, direction, decomposition_id, bad_pick, bad_height)
    assert store.load_direction(direction.name).prop_scale == pytest.approx(2.0)


def test_with_a_prop_scale_each_piece_is_checked_against_its_own_target_and_the_prompt_asks_for_it(
    tmp_path,
):
    # Catches: a piece far off its own target going unflagged (or an on-target one flagged),
    # earlier warnings of either kind surviving a run, and the size-true prompt not reaching
    # the service.
    boxes = [(10, 10, 90, 50), (120, 10, 160, 130), (200, 10, 264, 74), (10, 160, 170, 256)]
    store, direction, decomposition_id = _project_with_picks(tmp_path, boxes, (400, 300))
    tall = next(p for p in load_picks(store, direction.name, decomposition_id).picks
                if p.picture_box[3] - p.picture_box[1] == 120)
    # The painter draws every piece 9 x 7, so a 9-tall target is exactly right for the tall box.
    direction = set_prop_scale(store, direction, decomposition_id, str(tall.number), 9)
    stale = load_decomposition(store, direction.name, decomposition_id)
    stale.warnings += [
        f"x: drawn about 2.0x too big {SCALE_PHRASE} (old)",
        f"x: drawn about 2.0x too big {TARGET_PHRASE} (old)",
        "x: another note",
    ]
    save_decomposition(store, direction.name, stale)

    result = redraw_picks(
        store, direction, decomposition_id, service=FakeImageService(painter=_chunky_cells)
    )

    flagged = {w.split(":")[0]: w for w in result.warnings if TARGET_PHRASE in w}
    by_box = {c.origin.picture_box: c.name for c in result.cutouts}
    assert set(flagged) == {by_box[boxes[0]], by_box[boxes[2]], by_box[boxes[3]]}
    assert "too big" in flagged[by_box[boxes[0]]]
    assert "too small" in flagged[by_box[boxes[3]]]
    assert not any(SCALE_PHRASE in w for w in result.warnings)
    assert "x: another note" in result.warnings
    assert not any(w.startswith("x: drawn") for w in result.warnings)
    prompts = [
        read_prompt for folder in store.generations_dir(direction.name).iterdir()
        if (read_prompt := _prompt_of(folder)) is not None
    ]
    assert prompts and all("art pixels" in p and "SAME art-pixel size" in p for p in prompts)


def _prompt_of(folder):
    """The prompt stored in a generation folder's request.json, or None when there is none."""
    path = folder / "request.json"
    return json.loads(path.read_text()).get("prompt") if path.exists() else None


def test_with_no_prop_scale_the_group_prompt_keeps_its_relative_size_line(tmp_path):
    # Catches: the size-true wording leaking into a redraw of a direction with no scale.
    store, direction, decomposition_id = _project_with_picks(
        tmp_path, [(10, 10, 90, 50), (120, 10, 160, 130)], (400, 300)
    )

    redraw_picks(
        store, direction, decomposition_id, service=FakeImageService(painter=_chunky_cells)
    )

    prompts = [p for f in store.generations_dir(direction.name).iterdir() if (p := _prompt_of(f))]
    group_prompts = [p for p in prompts if "grid of" in p]
    assert group_prompts
    for prompt in group_prompts:
        assert (
            "- Keep each object at the same size relative to the others as in the first image.\n"
            in prompt
        )
        assert "art pixels" not in prompt


class CountingService(FakeImageService):
    """The fake, counting every call it is asked to make."""

    calls = 0

    def generate(self, request, out_path, *, reference_paths=()):
        """Count the call, then answer as the fake does."""
        self.calls += 1
        return super().generate(request, out_path, reference_paths=reference_paths)


def test_a_pick_too_big_for_its_cell_is_refused_before_anything_is_spent_or_cleared(tmp_path):
    # Catches: the too-big refusal arriving after paid naming calls, the wiped undo history
    # and a half-written group folder.
    store, direction, decomposition_id = _project_with_picks(
        tmp_path, [(0, 0, 400, 250)], (1000, 1000)
    )
    direction.prop_scale = 0.5
    undo_before = load_picks(store, direction.name, decomposition_id).undo
    assert undo_before, "the picks should have an undo history to protect"
    service = CountingService(painter=_chunky_cells)

    with pytest.raises(ValueError, match="too big"):
        redraw_picks(store, direction, decomposition_id, service=service)

    assert service.calls == 0
    assert load_picks(store, direction.name, decomposition_id).undo == undo_before
    assert not store.group_dir(direction.name, decomposition_id, "group-01").exists()
