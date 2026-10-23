"""Which step of a scene is done, current, next or waiting, read off real folders."""

import shutil
from datetime import UTC, datetime

import pytest

from pixelmagic.journey import (
    StepState,
    choose_scene,
    direction_summary,
    followed_scene,
    latest_scene,
    order_for_home,
    scene_steps,
)
from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    Cutout,
    Decomposition,
    Direction,
    GenerationRequest,
    GenerationResult,
    GridRebuild,
    Manifest,
    ManifestEntry,
    PromptKind,
    RecipeRecord,
    RefineReport,
    Review,
    Reviewer,
    ReviewStatus,
    SheetKind,
)
from pixelmagic.store import Store

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)
GEN = "20260903-191233"
DEN = "cosy-den"


@pytest.fixture
def store(tmp_path):
    """A project with one direction and nothing drawn."""
    store = Store(tmp_path)
    store.create_project("Game", now=NOW)
    store.create_direction(Direction(name=DEN, created=NOW, style="warm"))
    return store


def add_generation(store, gen_id=GEN, *, status=ReviewStatus.APPROVED, kind=PromptKind.SCENE):
    """Write a generation's three sidecars."""
    folder = store.generation_dir(DEN, gen_id)
    store.write_model(
        folder / "request.json",
        GenerationRequest(id=gen_id, created=NOW, subject=f"s-{gen_id}", kind=kind, size=(8, 8)),
    )
    store.write_model(folder / "result.json", GenerationResult(id=gen_id, created=NOW))
    store.write_model(folder / "review.json", Review(status=status, reviewer=Reviewer.HUMAN))


def add_scene_asset(store, *, refined=True, error_code=None, name="den", generation=GEN):
    """Write a kept scene asset, optionally refined (or refused)."""
    folder = store.asset_dir(DEN, name)
    folder.mkdir(parents=True)
    (folder / "source.png").write_bytes(b"x")
    store.write_model(
        folder / "asset.json",
        AssetRecord(name=name, kind=AssetKind.SCENE, source_generation=generation),
    )
    if refined:
        (folder / "refined.png").write_bytes(b"x")
        store.write_model(folder / "refine.json", RefineReport(asset=name, error_code=error_code))


def add_decomposition(store, dec_id, cutouts, *, picks=False, generation=GEN):
    """Write a decomposition from GEN holding the given cutouts."""
    store.write_model(
        store.decomposition_dir(DEN, dec_id) / "decomposition.json",
        Decomposition(id=dec_id, created=NOW, source_generation=generation, cutouts=cutouts),
    )
    if picks:
        store.picks_dir(DEN, dec_id).mkdir(parents=True)


def prop_cutout():
    """A cutout that was redrawn and rebuilt on its own grid."""
    return Cutout(
        name="chair", sheet="props", kind=SheetKind.PROPS, bbox=(0, 0, 4, 4),
        grid_rebuild=GridRebuild(
            pitch_x=1, pitch_y=1, phase_x=0, phase_y=0, confidence=1, native_size=(4, 4)
        ),
    )


def tile_cutout():
    """A cutout that is a recipe tile."""
    return Cutout(
        name="floor-v1-nw", sheet="tiles", kind=SheetKind.TILES, bbox=(0, 0, 16, 16),
        recipe=RecipeRecord(kind="wood", layout="block", surface="floor"),
    )


def add_bundle(store, stamp, *, direction=DEN, entries=("den",)):
    """Write an export bundle's manifest listing the given scene entries."""
    store.write_model(
        store.export_dir(stamp) / "manifest.json",
        Manifest(
            project="Game", direction=direction, exported=NOW,
            entries=[
                ManifestEntry(name=n, kind=AssetKind.SCENE, width=8, height=8) for n in entries
            ],
        ),
    )


def states(steps):
    """The step states by key, as plain strings."""
    return {s.key.value: s.state.value for s in steps}


def test_unapproved_picture_makes_approve_current_and_keep_next(store):
    add_generation(store, status=ReviewStatus.PENDING)
    assert states(scene_steps(store, DEN, GEN)) == {
        "start": "done", "approve": "current", "keep": "next", "pick": "waiting",
        "surfaces": "waiting", "refine": "waiting", "export": "waiting",
    }
    # catches: a pending picture shown as approved, or "Keep scene" offered before approval.


def test_whole_road_done_has_no_current_step(store):
    add_generation(store)
    add_scene_asset(store)
    add_decomposition(store, "20260903-200000", [prop_cutout(), tile_cutout()], picks=True)
    add_bundle(store, "bundle")
    steps = scene_steps(store, DEN, GEN)
    assert all(s.state is StepState.DONE for s in steps)
    # catches: a finished scene still showing a step to do, or a step that can never turn done.


def test_surfaces_without_props_leaves_pick_current_and_surfaces_done(store):
    add_generation(store)
    add_scene_asset(store, refined=False)
    add_decomposition(store, "20260903-200000", [tile_cutout()], picks=True)
    steps = scene_steps(store, DEN, GEN)
    assert states(steps)["pick"] == "current"
    assert states(steps)["surfaces"] == "done"
    assert states(steps)["refine"] == "waiting"
    pick = next(s for s in steps if s.key.value == "pick")
    assert pick.decomposition == "20260903-200000"
    # catches: surfaces drawn first being hidden because a later step is "behind" the pick step.


def test_pick_links_to_newest_decomposition_even_before_a_redraw(store):
    add_generation(store)
    add_scene_asset(store, refined=False)
    add_decomposition(store, "20260903-200000", [], picks=True)
    add_decomposition(store, "20260903-210000", [])  # no suggestions or picks
    pick = next(s for s in scene_steps(store, DEN, GEN) if s.key.value == "pick")
    assert pick.state is StepState.CURRENT
    assert pick.decomposition == "20260903-200000"
    # catches: the pick screen link pointing at a decomposition with nothing to pick on.


def test_refine_with_error_code_is_not_refined(store):
    add_generation(store)
    add_scene_asset(store, error_code="TARGET_NOT_USABLE")
    assert states(scene_steps(store, DEN, GEN))["refine"] != "done"
    # catches: a refused refine (image-2 case) shown as done because refined.png exists.


def test_broken_manifest_does_not_raise_or_count(store):
    add_generation(store)
    add_scene_asset(store)
    broken = store.export_dir("broken")
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{not json")
    assert states(scene_steps(store, DEN, GEN))["export"] != "done"
    add_bundle(store, "good")
    assert states(scene_steps(store, DEN, GEN))["export"] == "done"
    # catches: one corrupt manifest breaking the scene page, or hiding a real export.


def test_export_of_another_direction_does_not_count(store):
    add_generation(store)
    add_scene_asset(store)
    add_bundle(store, "other", direction="other-den")
    assert states(scene_steps(store, DEN, GEN))["export"] != "done"
    # catches: a same-named scene exported from a different direction marking this one exported.


def test_missing_generation_raises_and_none_starts_at_start(store):
    with pytest.raises(FileNotFoundError):
        scene_steps(store, DEN, "20990101-000000")
    steps = scene_steps(store, DEN, None)
    assert states(steps)["start"] == "current"
    assert states(steps)["approve"] == "next"
    # catches: an unknown id producing a half-empty timeline instead of a 404.


def test_latest_scene_skips_rejected_and_non_scene(store):
    add_generation(store, "20260903-100000")
    add_generation(store, "20260903-110000", kind=PromptKind.GROUP_REDRAW)
    add_generation(store, "20260903-120000", status=ReviewStatus.REJECTED)
    assert latest_scene(store, DEN) == "20260903-100000"
    # catches: the home card opening a group-redraw reply or a rejected try as "the scene".


def test_refine_and_export_follow_the_kept_scene_not_a_newer_pending_one(store):
    add_generation(store)
    add_scene_asset(store)
    add_generation(store, "20260905-090000", status=ReviewStatus.PENDING)
    assert latest_scene(store, DEN) == "20260905-090000"
    assert followed_scene(store, DEN) == GEN
    # catches: a new attempt hiding where the kept scene stands on the Refine and Export pages.


OLD, NEW = "20260901-100000", "20260902-100000"


def two_kept_scenes(store, *, old_has_pieces):
    """An older kept scene `old` and a newer bare one `new`; the older may hold a prop."""
    for gen, name in ((OLD, "old"), (NEW, "new")):
        add_generation(store, gen)
        add_scene_asset(store, name=name, generation=gen)
    if old_has_pieces:
        add_decomposition(store, "20260903-200000", [prop_cutout()], generation=OLD)


def test_default_follows_the_newest_kept_scene_that_has_pieces(store):
    two_kept_scenes(store, old_has_pieces=True)
    assert followed_scene(store, DEN) == OLD
    # catches: a newer bare import taking Export over so the older scene's room never shows.


def test_default_with_no_pieces_anywhere_is_the_newest_kept_scene(store):
    two_kept_scenes(store, old_has_pieces=False)
    assert followed_scene(store, DEN) == NEW
    # catches: the piece rule hiding every scene, or picking the oldest, when none has pieces.


def test_a_stored_choice_beats_the_default(store):
    two_kept_scenes(store, old_has_pieces=True)
    choose_scene(store, DEN, NEW)
    assert followed_scene(store, DEN) == NEW
    # catches: the person's choice being ignored in favour of the default.


def test_a_stale_or_unreadable_choice_falls_back_without_raising(store):
    two_kept_scenes(store, old_has_pieces=True)
    choose_scene(store, DEN, NEW)
    shutil.rmtree(store.asset_dir(DEN, "new"))
    assert followed_scene(store, DEN) == OLD
    store.scene_choice_file(DEN).write_text("{not json")
    assert followed_scene(store, DEN) == OLD
    # catches: deleting the chosen scene (or a damaged sidecar) breaking Refine and Export pages.


def test_choosing_something_that_is_not_a_kept_scene_is_refused_and_writes_nothing(store):
    two_kept_scenes(store, old_has_pieces=False)
    add_generation(store, "20260905-090000", status=ReviewStatus.PENDING)
    with pytest.raises(FileNotFoundError):
        choose_scene(store, DEN, "20260905-090000")
    with pytest.raises(FileNotFoundError):
        choose_scene(store, DEN, "20990101-000000")
    assert not store.scene_choice_file(DEN).exists()
    # catches: pointing Refine and Export at a picture with nothing kept to refine.


def test_summary_counts_and_ordering_for_home(store):
    add_generation(store, "20260903-100000")
    add_generation(store, "20260903-110000-2")
    add_decomposition(store, "20260904-090000", [])
    summary = direction_summary(store, DEN)
    assert summary.scene_count == 2
    assert summary.latest_scene == "20260903-110000-2"
    assert summary.last_edited == datetime(2026, 9, 4, 9, 0, 0, tzinfo=UTC)

    created = datetime(2027, 1, 1, tzinfo=UTC)
    store.create_direction(Direction(name="empty", created=created, style="x"))
    empty = direction_summary(store, "empty")
    assert empty.latest_scene is None
    assert empty.last_edited == datetime(2027, 1, 1, tzinfo=UTC)
    assert [s.name for s in order_for_home([empty, summary])] == [DEN, "empty"]
    # catches: a direction with no picture leading the home page, or a -2 suffix
    # breaking the edit time.
