import logging
import os
import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pixelmagic.approval import AutoApprove, PendingApproval
from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    Cutout,
    Decomposition,
    Direction,
    GridRebuild,
    PitchSource,
    PromptKind,
    RefinerSettings,
    Reviewer,
    ReviewStatus,
    SheetKind,
)
from pixelmagic.naming import Surface, UnusableName
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.refiner.base import EXIT_NOT_RUN, LOW_GRID_CONFIDENCE, Analysis, RefinerError
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.stages.decompose import (
    PITCH_ANALYZE_SETTINGS,
    NoPicks,
    NotApproved,
    OldPieceRefused,
    decompose,
    delete_cutout,
    draw_surface,
    drawn_surfaces,
    keep_scene,
    layout_cutout,
    load_decomposition,
    pitch_note,
    promote_cutouts,
    promote_generation,
    read_box_shades,
    remove_surface,
    rename_cutout,
    rename_scene,
    save_decomposition,
    scene_asset_name,
    set_anchor,
)
from pixelmagic.stages.export import export
from pixelmagic.stages.generate import generate, import_picture
from pixelmagic.stages.layout import default_anchor
from pixelmagic.stages.pick import pick_suggestions
from pixelmagic.stages.recipes import RecipeRefused
from pixelmagic.stages.redraw import redraw_picks
from pixelmagic.stages.refine import load_report, refine_asset
from pixelmagic.stages.suggest import suggest_objects
from pixelmagic.store import AlreadyExists, Store
from test_recipes import FLOOR_COLOURS, floor_and_block_scene

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    source = generate(
        store, direction, "a cosy den",
        kind=PromptKind.SCENE, service=FakeImageService(), approval=AutoApprove(),
        size=(96, 96), now=NOW,
    )
    return store, direction, source


def test_promote_cutouts_stages_assets_with_their_provenance(tmp_path):
    store, direction, decomposition = cut_up(tmp_path)
    names = promote_cutouts(store, direction, decomposition)
    assert set(names) == {"wooden-chair", "round-rug", "small-lamp"}
    folder = store.asset_dir("cosy-den", "wooden-chair")
    assert (folder / "source.png").exists()
    record = store.read_model(folder / "asset.json", AssetRecord)
    assert record.kind is AssetKind.PROP
    assert record.source_generation == decomposition.source_generation
    assert record.grid_rebuild is not None


def test_provenance_names_the_service_that_drew_the_piece(tmp_path):
    """Catches: an asset claiming the direction's configured service ("codex") instead of
    the one that actually drew the redraw ("fake")."""
    store, direction, decomposition = cut_up(tmp_path, service="codex")
    promote_cutouts(store, direction, decomposition)
    record = store.read_model(
        store.asset_dir("cosy-den", "wooden-chair") / "asset.json", AssetRecord
    )
    assert record.service == "fake"
    assert store.load_direction("cosy-den").service == "codex"


def test_promote_generation_stages_an_approved_scene_as_an_asset(setup):
    store, direction, source = setup
    name = promote_generation(store, direction, source, refiner=StubRefiner())
    assert name == "cosy-den"
    folder = store.asset_dir("cosy-den", name)
    assert (folder / "source.png").read_bytes() == source.image_path.read_bytes()
    record = store.read_model(folder / "asset.json", AssetRecord)
    assert record.kind is AssetKind.SCENE
    assert record.source_generation == source.request.id
    assert record.source_prompt == source.request.prompt
    assert record.service == "fake"


def test_an_imported_picture_can_be_kept_as_the_scene(setup, tmp_path):
    store, direction, _ = setup
    Image.new("RGB", (64, 48), (90, 60, 40)).save(tmp_path / "den.png")
    imported = import_picture(store, direction, tmp_path / "den.png", reviewer=Reviewer.AGENT)
    name = keep_scene(store, direction, imported, refiner=StubRefiner())
    folder = store.asset_dir("cosy-den", name)
    record = store.read_model(folder / "asset.json", AssetRecord)
    assert record.kind is AssetKind.SCENE
    assert record.source_generation == imported.request.id
    assert (folder / "source.png").read_bytes() == imported.image_path.read_bytes()


def test_keeping_an_unapproved_picture_as_the_scene_is_refused_and_writes_nothing(setup):
    store, direction, _ = setup
    pending = generate(
        store, direction, "a rainy yard", kind=PromptKind.SCENE,
        service=FakeImageService(), approval=PendingApproval(), size=(96, 96), now=NOW,
    )
    before = store.list_assets("cosy-den")
    with pytest.raises(NotApproved):
        keep_scene(store, direction, pending, refiner=StubRefiner())
    assert store.list_assets("cosy-den") == before


def test_keeping_the_same_picture_as_the_scene_twice_leaves_one_scene(setup):
    store, direction, source = setup
    name = keep_scene(store, direction, source, refiner=StubRefiner())
    with pytest.raises(AlreadyExists, match=name):
        keep_scene(store, direction, source, refiner=StubRefiner())
    assert store.list_assets("cosy-den") == [name]


def test_keeping_a_second_picture_with_the_same_prompt_as_the_scene_numbers_its_name(setup):
    """Two different pictures of the same prompt both reach the asset list, the
    second one numbered, so keeping it cannot overwrite the first scene."""
    store, direction, first = setup
    second = generate(
        store, direction, first.request.subject, kind=PromptKind.SCENE,
        service=FakeImageService(), approval=AutoApprove(), size=(96, 96), now=NOW,
    )
    assert keep_scene(store, direction, first, refiner=StubRefiner()) == "cosy-den"
    assert keep_scene(store, direction, second, refiner=StubRefiner()) == "cosy-den-2"


def test_a_typed_scene_name_is_the_name_the_scene_is_filed_under(setup):
    """Catches: the typed name being ignored and the suggested one used."""
    store, direction, source = setup
    assert keep_scene(store, direction, source, refiner=StubRefiner(), name="main-room") == (
        "main-room"
    )
    assert store.list_assets("cosy-den") == ["main-room"]


@pytest.mark.parametrize(
    ("typed", "way_forward"),
    [("Main Room", "main-room"), ("../escape", "escape"), ("cosy--den", "cosy-den"),
     ("a" * 300, None), ("one two three four five", "one-two-three-four")],
)
def test_an_unsafe_scene_name_is_refused_with_a_way_forward_and_writes_nothing(
    setup, typed, way_forward
):
    """Catches: a typed name becoming a folder path outside the direction, one too long for
    the file system (a crash after the refiner ran), or a refusal that names no way forward."""
    store, direction, source = setup
    with pytest.raises(UnusableName) as refused:
        keep_scene(store, direction, source, refiner=StubRefiner(), name=typed)
    if way_forward is not None:
        assert f"try '{way_forward}'" in str(refused.value)
    assert store.list_assets("cosy-den") == []


def test_a_scene_name_used_by_an_asset_or_an_unkept_cutout_is_refused_not_numbered(tmp_path):
    """Catches: a typed name silently becoming `-2`, or a scene landing on a name a later
    keep of the cut-up run would overwrite."""
    store, direction, decomposition = cut_up(tmp_path)
    picture = generate(
        store, direction, "a cosy den", kind=PromptKind.SCENE,
        service=FakeImageService(), approval=AutoApprove(), size=(96, 96), now=NOW,
    )
    # "wooden-chair" is a cut-out of an unkept run: no asset exists under it yet.
    assert "wooden-chair" not in store.list_assets("cosy-den")
    with pytest.raises(AlreadyExists, match="wooden-chair"):
        keep_scene(store, direction, picture, refiner=StubRefiner(), name="wooden-chair")
    promote_cutouts(store, direction, decomposition)
    with pytest.raises(AlreadyExists, match="small-lamp"):
        keep_scene(store, direction, picture, refiner=StubRefiner(), name="small-lamp")
    assert store.list_assets("cosy-den") == ["round-rug", "small-lamp", "wooden-chair"]


def test_a_scene_name_may_not_be_a_recipe_surfaces_bare_name(recipe_setup):
    """Catches: a scene taking `wood-floor`, which then blocks re-running that surface and
    clashes with the surface's `wood-floor.aseprite` in an export."""
    store, direction, source = recipe_setup
    _recipe_decompose(store, direction, source)
    with pytest.raises(AlreadyExists, match="wood-floor"):
        keep_scene(store, direction, source, refiner=StubRefiner(), name="wood-floor")
    assert scene_asset_name(store, direction, source) != "wood-floor"
    assert store.list_assets("cosy-den") == []


def test_renaming_a_refined_scene_leaves_nothing_on_the_old_name_and_exports_the_new(
    setup, monkeypatch
):
    """Catches: a rename that leaves a folder, `asset.json` or `refine.json` on the old
    name (the scene then vanishes from refine and export), or that rewrites a bundle
    already exported."""
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None, raising=False)
    store, direction, source = setup
    keep_scene(store, direction, source, refiner=StubRefiner())
    refine_asset(store, direction, "cosy-den", refiner=StubRefiner())
    earlier = export(store, direction, stamp="first", now=NOW)

    assert rename_scene(store, direction, "cosy-den", "main-room") == "main-room"

    assert store.list_assets("cosy-den") == ["main-room"]
    assert not store.asset_dir("cosy-den", "cosy-den").exists()
    folder = store.asset_dir("cosy-den", "main-room")
    assert store.read_model(folder / "asset.json", AssetRecord).name == "main-room"
    assert load_report(store, "cosy-den", "main-room").asset == "main-room"
    later = export(store, direction, stamp="second", now=NOW)
    assert [entry.name for entry in later.manifest.entries] == ["main-room"]
    assert (later.directory / "main-room.png").exists()
    assert [entry.name for entry in earlier.manifest.entries] == ["cosy-den"]
    assert (earlier.directory / "cosy-den.png").exists()


def test_renaming_a_prop_or_to_a_taken_name_is_refused_and_moves_nothing(setup, tmp_path):
    """Catches: a rename moving a folder before it noticed the refusal, and a prop being
    renamed through the scene road (its decomposition would no longer find it)."""
    store, direction, source = setup
    keep_scene(store, direction, source, refiner=StubRefiner())
    other = generate(
        store, direction, "a rainy yard", kind=PromptKind.SCENE,
        service=FakeImageService(), approval=AutoApprove(), size=(96, 96), now=NOW,
    )
    promote_generation(store, direction, other, refiner=StubRefiner(), kind=AssetKind.PROP)
    with pytest.raises(ValueError, match="not a scene"):
        rename_scene(store, direction, "rainy-yard", "yard")
    with pytest.raises(AlreadyExists):
        rename_scene(store, direction, "cosy-den", "rainy-yard")
    with pytest.raises(AlreadyExists, match="already called"):
        rename_scene(store, direction, "cosy-den", " cosy-den ")
    with pytest.raises(UnusableName):
        rename_scene(store, direction, "cosy-den", "a" * 300)
    with pytest.raises(ValueError, match="no asset"):
        rename_scene(store, direction, "nothing", "yard")
    assert store.list_assets("cosy-den") == ["cosy-den", "rainy-yard"]


def redraw_run(store, direction, scratch, boxes, descriptions, *, size=(300, 200)):
    """Import a picture with one blob per box, pick them all and redraw them.

    The pieces come out of the object road (the fake service's chunky board,
    rebuilt on its own pixel grid), named from `descriptions` in pick order.
    """
    picture = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(picture)
    for index, box in enumerate(boxes):
        colour = (40 + index * 20, 90, 200, 255)
        draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=colour)
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / "scene.png"
    picture.save(path)
    record = import_picture(store, direction, path, reviewer=Reviewer.HUMAN, now=NOW)
    made = suggest_objects(store, direction, record, StubSegmenter(), now=NOW)
    pick_suggestions(store, direction, made.id, list(range(1, len(boxes) + 1)))
    return redraw_picks(
        store, direction, made.id, service=FakeImageService(descriptions=descriptions)
    )


CUT_UP_BOXES = [(10, 10, 60, 60), (100, 10, 140, 80), (200, 20, 280, 60)]
CUT_UP_NAMES = ["Wooden Chair", "Round Rug", "Small Lamp"]


def cut_up(tmp_path, *, service="fake"):
    """A direction whose three picked objects have been redrawn into three rebuilt props.

    The pieces are named `wooden-chair`, `round-rug` and `small-lamp`.
    """
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.", service=service)
    )
    decomposition = redraw_run(
        store, direction, tmp_path / "scratch", CUT_UP_BOXES, CUT_UP_NAMES
    )
    assert {cutout.name for cutout in decomposition.cutouts} == {
        "wooden-chair", "round-rug", "small-lamp"
    }
    return store, direction, decomposition


def test_renaming_a_cutout_moves_its_folder_and_its_sidecar(tmp_path):
    store, direction, decomposition = cut_up(tmp_path)
    rename_cutout(store, direction, decomposition, "wooden-chair", "Grandad's Armchair")
    names = [cutout.name for cutout in decomposition.cutouts]
    assert "grandads-armchair" in names
    assert "wooden-chair" not in names
    folder = store.cutout_dir("cosy-den", decomposition.id, "grandads-armchair")
    assert (folder / "source.png").exists()
    assert store.read_model(folder / "cutout.json", Cutout).name == "grandads-armchair"
    assert not store.cutout_dir("cosy-den", decomposition.id, "wooden-chair").exists()


def test_a_rename_survives_a_reload_of_the_sidecar(tmp_path):
    store, direction, decomposition = cut_up(tmp_path)
    rename_cutout(store, direction, decomposition, "round-rug", "big rug")
    reloaded = load_decomposition(store, "cosy-den", decomposition.id)
    assert "big-rug" in [cutout.name for cutout in reloaded.cutouts]


def test_renaming_onto_a_name_in_use_gets_a_number(tmp_path):
    store, direction, decomposition = cut_up(tmp_path)
    rename_cutout(store, direction, decomposition, "round-rug", "wooden chair")
    assert "wooden-chair-2" in [cutout.name for cutout in decomposition.cutouts]


def test_renaming_a_cutout_that_is_not_there_says_so(tmp_path):
    store, direction, decomposition = cut_up(tmp_path)
    with pytest.raises(ValueError, match="nope"):
        rename_cutout(store, direction, decomposition, "nope", "whatever")


def test_renaming_a_cutout_to_nothing_usable_is_refused(tmp_path):
    """A person's rename goes through `slug`, not `kebab_case` — an unusable

    name is refused rather than quietly becoming `item` (M2.11's coordination
    note on the cutout-rename path). The piece must be untouched: still under
    its old name, still on disk under its old folder.
    """
    store, direction, decomposition = cut_up(tmp_path)
    with pytest.raises(ValueError):
        rename_cutout(store, direction, decomposition, "wooden-chair", "日本語")
    names = [cutout.name for cutout in decomposition.cutouts]
    assert "wooden-chair" in names
    assert store.cutout_dir("cosy-den", decomposition.id, "wooden-chair").exists()


def test_renaming_a_cutout_to_a_name_too_long_is_refused_not_shortened(tmp_path):
    """Catches: a typed name losing its fifth word (`bath-mat-tiles-corner-nw` quietly
    becoming `bath-mat-tiles-corner`) and the piece being moved under a name
    the person never chose."""
    store, direction, decomposition = cut_up(tmp_path)
    with pytest.raises(UnusableName, match="bath-mat-tiles-corner"):
        rename_cutout(
            store, direction, decomposition, "wooden-chair", "bath mat tiles corner nw"
        )
    assert "wooden-chair" in [cutout.name for cutout in decomposition.cutouts]
    assert store.cutout_dir("cosy-den", decomposition.id, "wooden-chair").exists()


def test_deleting_a_cutout_removes_the_folder_and_the_entry(tmp_path):
    store, direction, decomposition = cut_up(tmp_path)
    delete_cutout(store, direction, decomposition, "small-lamp")
    assert "small-lamp" not in [cutout.name for cutout in decomposition.cutouts]
    assert not store.cutout_dir("cosy-den", decomposition.id, "small-lamp").exists()
    assert "small-lamp" not in promote_cutouts(store, direction, decomposition)


def test_a_folder_that_cannot_be_removed_leaves_the_sidecar_alone(tmp_path):
    """A real removal failure must raise rather than be swallowed.

    `shutil.rmtree(..., ignore_errors=True)` used to report success even when
    a folder could not actually be removed, so the sidecar would be rewritten
    as if the piece were gone while its folder stayed on disk. Making the
    *parent* of the cutout folder unwritable is what actually blocks the
    removal: deleting a directory entry needs write permission on its parent,
    not on the directory being deleted.
    """
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("directory permissions do not block root or Windows")
    store, direction, decomposition = cut_up(tmp_path)
    name = decomposition.cutouts[0].name
    cutouts_root = store.cutout_dir("cosy-den", decomposition.id, name).parent
    cutouts_root.chmod(0o500)
    try:
        with pytest.raises(OSError):
            delete_cutout(store, direction, decomposition, name)
    finally:
        cutouts_root.chmod(0o700)
    reloaded = load_decomposition(store, "cosy-den", decomposition.id)
    assert name in [cutout.name for cutout in reloaded.cutouts]


def test_a_rename_cannot_walk_onto_an_earlier_runs_unpromoted_name(tmp_path):
    """The collision `_all_cutout_names` exists to stop, on the rename path.

    Two redraw runs in one direction: the second run's pieces are numbered `-2`
    so they cannot clash. Renaming one of them back onto the first run's
    still-unpromoted name would undo that, and promoting both would then copy
    one piece over the other and leave five assets where six pieces were.
    """
    store, direction, first = cut_up(tmp_path)
    second = redraw_run(
        store, direction, tmp_path / "again", CUT_UP_BOXES, CUT_UP_NAMES
    )
    assert "wooden-chair-2" in [cutout.name for cutout in second.cutouts]

    rename_cutout(store, direction, second, "wooden-chair-2", "wooden chair")
    assert "wooden-chair" not in [cutout.name for cutout in second.cutouts]

    promoted = set(promote_cutouts(store, direction, first))
    promoted |= set(promote_cutouts(store, direction, second))
    assert len(promoted) == len(first.cutouts) + len(second.cutouts)


def test_a_rejected_source_cannot_be_cut_up(setup):
    """Controller ruling 2: the check that used to live only in the web
    route (Guardian minor 3) now lives here, so the command line and the MCP
    server refuse a picture nobody approved without inventing their own copy
    of the rule."""
    store, direction, source = setup
    rejected = source.model_copy(
        update={"review": source.review.model_copy(update={"status": ReviewStatus.REJECTED})}
    )
    with pytest.raises(ValueError, match="rejected"):
        decompose(
            store, direction, rejected,
            surfaces=[Surface(text="wood floor", name="wood-floor", kind="wood", **FLOOR_AREA)],
            now=NOW,
        )
    # Nothing was written: no decomposition folder, no cutout —
    # `decompositions_dir` itself is created empty by `create_direction`, so
    # what proves nothing ran is that nothing sits inside it.
    assert store.list_decompositions("cosy-den") == []
    assert list(store.decompositions_dir("cosy-den").iterdir()) == []


class SettingsRecordingRefiner(StubRefiner):
    """A stand-in that remembers every settings block `analyze` was asked with."""

    def __init__(self):
        self.seen: list[RefinerSettings | None] = []

    def analyze(self, input_path, settings=None):
        self.seen.append(settings)
        return super().analyze(input_path, settings)


def test_a_promoted_scenes_pitch_reading_asks_for_cohesion_and_cleanup_off(setup):
    store, direction, source = setup
    refiner = SettingsRecordingRefiner()
    promote_generation(store, direction, source, refiner=refiner)
    assert refiner.seen == [PITCH_ANALYZE_SETTINGS]


def test_promote_generation_measures_the_scene_and_writes_its_target(setup):
    store, direction, source = setup

    class FixedPitchRefiner(StubRefiner):
        honours_forced_size = True  # stands for the real refiner, whose reading is a measurement

        def analyze(self, input_path, settings=None):
            return Analysis(pitch=2.0, confidence=0.9, warnings=[], grid_candidates=[])

    name = promote_generation(store, direction, source, refiner=FixedPitchRefiner())
    record = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert record.pitch == 2.0
    assert record.pitch_source is PitchSource.MEASURED
    assert pitch_note(store, "cosy-den", name) is None
    assert record.target == (48, 48)
    assert (record.footprint, record.anchor, record.role) == (None, None, None)


@pytest.mark.parametrize("reading", [None, 200.0], ids=["no reading", "out of range"])
def test_a_scene_with_no_usable_pitch_reading_is_filed_as_a_fallback(setup, caplog, reading):
    """Catches: a guessed pitch of 1.0 (or an absurd refiner reading) filed as
    MEASURED, so the scene is refined at the wrong size with nothing telling
    the person. A kept scene has no decomposition sidecar to carry a warning,
    so the case is also said through `logging` (never printed: the MCP
    server's standard output is its protocol)."""
    store, direction, source = setup

    class UnusableReadingRefiner(StubRefiner):
        def analyze(self, input_path, settings=None):
            return Analysis(pitch=reading, confidence=0.0, warnings=[], grid_candidates=[])

    with caplog.at_level(logging.WARNING):
        name = promote_generation(store, direction, source, refiner=UnusableReadingRefiner())
    record = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert record.pitch == 1.0
    assert record.pitch_source is PitchSource.FALLBACK
    assert record.target == (96, 96)
    assert pitch_note(store, "cosy-den", name)
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "cosy-den" in warning.message
    assert source.request.id in warning.message


def test_the_stand_in_refiner_files_no_measured_pitch(setup):
    """Catches: the stand-in's stock answer (1.0, full confidence) filed as MEASURED
    when a scene is kept by default, so a later real refine forces the full picture
    size and gives a full-size copy with no warning."""
    store, direction, source = setup
    name = promote_generation(store, direction, source, refiner=StubRefiner())
    record = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert record.pitch_source is not PitchSource.MEASURED
    assert pitch_note(store, "cosy-den", name)


def test_a_pitch_the_refiner_flagged_as_low_confidence_is_kept_but_filed_as_unsure(setup):
    """Catches: a reading the refiner itself doubted (LOW_GRID_CONFIDENCE) filed
    as MEASURED, so nobody is told to check it."""
    store, direction, source = setup

    class DoubtfulRefiner(StubRefiner):
        def analyze(self, input_path, settings=None):
            return Analysis(
                pitch=2.0, confidence=0.3, warnings=[LOW_GRID_CONFIDENCE], grid_candidates=[]
            )

    name = promote_generation(store, direction, source, refiner=DoubtfulRefiner())
    record = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert record.pitch == 2.0
    assert record.pitch_source is PitchSource.UNSURE
    assert record.target == (48, 48)
    assert pitch_note(store, "cosy-den", name)


def test_a_refiner_that_fails_to_analyze_leaves_no_half_staged_asset(setup):
    store, direction, source = setup

    class DeadRefiner(StubRefiner):
        def analyze(self, input_path, settings=None):
            raise RefinerError("pixel-refiner is not installed", exit_code=EXIT_NOT_RUN)

    with pytest.raises(RefinerError):
        promote_generation(store, direction, source, refiner=DeadRefiner())
    assert store.list_assets("cosy-den") == []


def test_promote_cutouts_copies_the_grid_fields_onto_the_asset(tmp_path):
    """Catches: a kept prop losing its pitch, target, footprint or anchor on the way to
    `asset.json`, so refine and export would read a prop with no size."""
    store, direction, decomposition = cut_up(tmp_path)
    promote_cutouts(store, direction, decomposition)
    for cutout in decomposition.cutouts:
        record = store.read_model(
            store.asset_dir("cosy-den", cutout.name) / "asset.json", AssetRecord
        )
        assert (
            record.pitch, record.pitch_source, record.target, record.footprint, record.anchor,
        ) == (
            cutout.pitch, cutout.pitch_source, cutout.target, cutout.footprint, cutout.anchor,
        )
        assert record.target is not None and record.footprint is not None


def empty_run(tmp_path, *, cell_size=16):
    """A direction with one decomposition that holds no pieces yet."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(
            name="cosy-den", created=NOW, style="Warm SNES interiors.", cell_size=cell_size
        )
    )
    run = Decomposition(
        id="20260903-191233", created=NOW, source_generation="20260903-191200"
    )
    save_decomposition(store, direction.name, run)
    return store, direction, run


def plant(store, direction, run, name, width, height):
    """Add a hand-built rebuilt prop to a run: `width` x `height` native pixels.

    The grid's native size is the piece's target, as the redraw makes it; its
    `bbox` stays where the piece sat on the group's reply, four times as big, so
    it does not match the saved picture. Footprint and anchor are laid out the
    way the redraw lays them out.
    """
    folder = store.cutout_dir(direction.name, run.id, name)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((height, width, 4), (120, 80, 40, 255), dtype=np.uint8), "RGBA"
    ).save(folder / "source.png")
    cutout = layout_cutout(
        Cutout(
            name=name, sheet="20260903-191240", kind=SheetKind.PROPS,
            bbox=(0, 0, width * 4, height * 4), pitch=4.0, pitch_source=PitchSource.MEASURED,
            target=(width, height),
            grid_rebuild=GridRebuild(
                pitch_x=4.0, pitch_y=4.0, phase_x=0.0, phase_y=0.0, confidence=0.9,
                native_size=(width, height),
            ),
        ),
        direction.cell_size,
    )
    run.cutouts = [c for c in run.cutouts if c.name != name] + [cutout]
    store.write_model(folder / "cutout.json", cutout)
    save_decomposition(store, direction.name, run)
    return cutout


def one_prop(tmp_path, *, cell_size=16):
    """A direction with one rebuilt prop, 62 x 129 art pixels: the M3.2 plan's worked example."""
    store, direction, run = empty_run(tmp_path, cell_size=cell_size)
    plant(store, direction, run, "gramophone-cabinet", 62, 129)
    return store, direction, run


@pytest.mark.parametrize(("cell_size", "footprint"), [(16, (6, 8)), (32, (3, 4))])
def test_a_redrawn_prop_gets_a_footprint_in_cells_and_a_bottom_centre_anchor(
    tmp_path, cell_size, footprint
):
    """Catches: the redraw leaving a prop with no footprint or anchor, or counting cells
    with the wrong cell size. The fake board's one piece is 96 x 128 art pixels."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="x", cell_size=cell_size)
    )
    run = redraw_run(
        store, direction, tmp_path / "scratch", CUT_UP_BOXES[:1], CUT_UP_NAMES[:1]
    )
    (piece,) = run.cutouts
    assert piece.target == (96, 128)
    assert piece.footprint == footprint
    assert piece.anchor == (48, 128)  # bottom centre; cells do not move it


def test_layout_cutout_fills_a_prop_that_has_a_target():
    cutout = Cutout(
        name="cabinet", sheet="20260906-120000", kind=SheetKind.PROPS,
        bbox=(0, 0, 62, 129), pitch=1.0, target=(62, 129),
    )
    laid_out = layout_cutout(cutout, 16)
    assert (laid_out.footprint, laid_out.anchor) == ((4, 9), (31, 129))
    assert cutout.footprint is None  # a copy, not an edit in place


def test_layout_cutout_leaves_a_piece_without_a_target_and_a_tile_alone():
    bare = Cutout(name="old", sheet="20260101-000000", kind=SheetKind.PROPS, bbox=(0, 0, 8, 8))
    assert layout_cutout(bare, 16) is bare
    tile = Cutout(
        name="floor", sheet="20260101-000000", kind=SheetKind.TILES,
        bbox=(0, 0, 16, 16), target=(16, 16),
    )
    assert layout_cutout(tile, 16) is tile


def test_set_anchor_rewrites_the_assets_sidecar(tmp_path):
    store, direction, decomposition = one_prop(tmp_path)
    promote_cutouts(store, direction, decomposition)
    updated = set_anchor(store, "cosy-den", "gramophone-cabinet", (20, 100))
    assert updated.anchor == (20, 100)
    on_disk = store.read_model(
        store.asset_dir("cosy-den", "gramophone-cabinet") / "asset.json", AssetRecord
    )
    assert on_disk.anchor == (20, 100)
    assert on_disk.footprint == (4, 9)  # nothing else moved
    assert on_disk.target == (62, 129)


def test_re_keeping_a_prop_keeps_a_hand_moved_anchor(tmp_path):
    """`set_anchor` only ever writes the AssetRecord, never the decomposition's
    own `Cutout` — so before this fix, `promote_cutouts` handed a re-kept
    prop the cutout's own untouched anchor (still the fresh default) and
    silently put a hand-moved one back (ticket #37). Locked decision 6: an
    anchor is editable, never guessed back to a default.

    A rebuilt piece keeps `bbox` on the group's reply but saves a smaller,
    native-size `source.png`; `_asset_matches_piece` compares the native size,
    or the moved anchor would be lost again.

    Mutation check: pass `cutout.anchor` straight through in `promote_cutouts`
    (today's code, before this fix) and this fails — `on_disk.anchor` comes
    back as the fresh default (31, 129), not the moved (20, 100).
    """
    store, direction, decomposition = one_prop(tmp_path)
    promote_cutouts(store, direction, decomposition)
    set_anchor(store, "cosy-den", "gramophone-cabinet", (20, 100))

    promote_cutouts(store, direction, decomposition)

    on_disk = store.read_model(
        store.asset_dir("cosy-den", "gramophone-cabinet") / "asset.json", AssetRecord
    )
    assert on_disk.anchor == (20, 100)



@pytest.mark.parametrize(
    "old",
    [
        # A prop cut on the props sheet, and one redrawn before the grid rebuild: no grid_rebuild.
        Cutout(name="sheet-lamp", sheet="s", kind=SheetKind.PROPS, bbox=(0, 0, 8, 8)),
        # A tile of a bordered surface: no recipe.
        Cutout(name="floor-centre", sheet="s", kind=SheetKind.TILES, bbox=(0, 0, 8, 8)),
    ],
)
def test_keeping_a_run_that_holds_an_old_piece_is_refused_before_anything_is_written(
    tmp_path, old
):
    """Catches: a run from a road that is gone being kept anyway, so its pieces become assets
    that refine would then refuse for ever (the person told only after the fact), or the
    refusal arriving half way with some assets already written. Dropping the old piece lets
    the rest of the run be kept, so a person is not stranded."""
    store, direction, run = empty_run(tmp_path)
    plant(store, direction, run, "rebuilt-lamp", 8, 4)
    folder = store.cutout_dir(direction.name, run.id, old.name)
    folder.mkdir(parents=True)
    Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(folder / "source.png")
    store.write_model(folder / "cutout.json", old)
    run.cutouts.append(old)
    save_decomposition(store, direction.name, run)

    with pytest.raises(OldPieceRefused, match=old.name):
        promote_cutouts(store, direction, run)
    assert store.list_assets(direction.name) == []

    delete_cutout(store, direction, run, old.name)
    assert promote_cutouts(store, direction, run) == ["rebuilt-lamp"]


def test_a_tile_or_scene_sharing_a_props_name_gets_the_fresh_default_anchor(tmp_path):
    """`_asset_matches_piece` refuses an asset that is not a prop before it
    ever compares sizes — a scene or a tile filed under the exact name a
    prop cutout is about to reuse has no anchor of its own to hand down.

    Mutation check: drop the `record.kind is not AssetKind.PROP` guard from
    `_asset_matches_piece` and this fails — the crafted scene's own (5, 5)
    survives the re-key instead of the fresh default.
    """
    store, direction, decomposition = one_prop(tmp_path)
    promote_cutouts(store, direction, decomposition)
    asset_file = store.asset_dir("cosy-den", "gramophone-cabinet") / "asset.json"
    record = store.read_model(asset_file, AssetRecord)
    store.write_model(
        asset_file, record.model_copy(update={"kind": AssetKind.SCENE, "anchor": (5, 5)})
    )

    promote_cutouts(store, direction, decomposition)

    on_disk = store.read_model(asset_file, AssetRecord)
    assert on_disk.anchor == default_anchor(on_disk.target) != (5, 5)


def test_an_unreadable_assets_sidecar_does_not_crash_re_keying(tmp_path):
    """`promote_cutouts` is about to overwrite this `asset.json` anyway, so a
    copy that will not parse is not raised on — it is treated exactly as if
    there were nothing there to inherit an anchor from.

    Mutation check: drop the `except (OSError, ValidationError)` in
    `_reused_anchor` and this fails with a `ValidationError` instead of
    keeping, since the corrupted sidecar propagates uncaught.
    """
    store, direction, decomposition = one_prop(tmp_path)
    promote_cutouts(store, direction, decomposition)
    asset_file = store.asset_dir("cosy-den", "gramophone-cabinet") / "asset.json"
    asset_file.write_text("not valid json at all")

    promote_cutouts(store, direction, decomposition)  # must not raise

    on_disk = store.read_model(asset_file, AssetRecord)
    assert on_disk.anchor == default_anchor(on_disk.target)


@pytest.mark.parametrize("anchor", [(63, 100), (20, 130), (-1, 100), (20, -1)])
def test_an_anchor_outside_the_target_is_refused_and_nothing_is_written(tmp_path, anchor):
    store, direction, decomposition = one_prop(tmp_path)
    promote_cutouts(store, direction, decomposition)
    with pytest.raises(ValueError):
        set_anchor(store, "cosy-den", "gramophone-cabinet", anchor)
    on_disk = store.read_model(
        store.asset_dir("cosy-den", "gramophone-cabinet") / "asset.json", AssetRecord
    )
    assert on_disk.anchor == (31, 129)  # still the default


def test_an_asset_without_a_target_cannot_take_an_anchor(tmp_path, write_asset):
    """An M2 asset: cut before target sizes existed. Refused by name."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    store.create_direction(Direction(name="cosy-den", created=NOW, style="Warm SNES interiors."))
    write_asset(store, "old-chair", AssetKind.PROP)
    with pytest.raises(ValueError, match="old-chair"):
        set_anchor(store, "cosy-den", "old-chair", (4, 8))


def test_a_tile_cannot_take_an_anchor(recipe_setup):
    """Locked decision 6 gives anchors to props; a tile is the grid."""
    store, direction, source = recipe_setup
    result = _recipe_decompose(store, direction, source)
    first, *_rest = promote_cutouts(store, direction, result)
    with pytest.raises(ValueError, match="only a prop"):
        set_anchor(store, "cosy-den", first, (8, 16))


def test_an_asset_that_is_not_there_is_refused_by_name(tmp_path):
    store, direction, decomposition = one_prop(tmp_path)
    promote_cutouts(store, direction, decomposition)
    with pytest.raises(ValueError, match="no asset called 'nope'"):
        set_anchor(store, "cosy-den", "nope", (0, 0))


def test_a_hand_built_plain_surface_is_refused_and_leaves_no_folder(setup):
    """Catches: a surface with neither a kind nor a border being drawn some other way (the
    old plain-patch road, now deleted), or leaving a half-made decomposition folder behind
    when it is refused."""
    store, direction, source = setup
    with pytest.raises(RecipeRefused, match="needs a recipe kind, a box and shapes"):
        decompose(
            store, direction, source,
            surfaces=[Surface(text="wood floor", name="wood-floor")],
            now=NOW,
        )
    assert not list(store.decompositions_dir("cosy-den").iterdir())


def test_props_are_refused_with_nothing_picked_and_a_typed_surface_makes_tiles_only(
    recipe_setup,
):
    """Catches: props arriving by any road but picking (the old props sheet is gone), or a
    typed surface also asking for props."""
    store, direction, source = recipe_setup
    with pytest.raises(NoPicks):
        decompose(store, direction, source, now=NOW)

    tiles_only = decompose(
        store, direction, source,
        surfaces=[Surface(text="wood floor", name="wood-floor", kind="wood", **FLOOR_AREA)],
        now=NOW,
    )
    assert tiles_only.cutouts
    assert all(c.kind is SheetKind.TILES for c in tiles_only.cutouts)


#: Textured floors for the block kinds, dark to light, with each colour's share. The
#: shares put one of the finder's fixed starting points (10%, 30%, ... of the sorted
#: pixels) in each colour, so no colour group comes out empty. Stone's lightest colour
#: is its grout, well above the 1% bar; plaster's four each clear 3%.
STONE_COLOURS = [
    ((70, 70, 74), 0.20),
    ((104, 104, 108), 0.20),
    ((138, 138, 140), 0.20),
    ((170, 168, 164), 0.27),
    ((214, 212, 204), 0.13),
]
PLASTER_COLOURS = [
    ((196, 180, 156), 0.25),
    ((210, 196, 172), 0.25),
    ((224, 210, 188), 0.25),
    ((238, 226, 206), 0.25),
]
RECIPE_FLOORS = {"wood": FLOOR_COLOURS, "stone": STONE_COLOURS, "plaster": PLASTER_COLOURS}


def _scene_with_floor(colours):
    """The shared scene with its floor repainted at random from `colours` (rgb, share)."""
    scene = floor_and_block_scene()
    rng = random.Random(11)
    picture = scene.picture.copy()
    rgbs = [rgb for rgb, _ in colours]
    weights = [weight for _, weight in colours]
    for y, x in zip(*np.nonzero(scene.floor), strict=True):
        picture[y, x, :3] = rng.choices(rgbs, weights)[0]
    return picture


#: The shared scene's floor as one rectangle, with a 40 x 40 box on a patch of it.
FLOOR_AREA = {"shapes": (((0, 0), (100, 0), (100, 88), (0, 88)),), "box": (10, 10, 50, 50)}


@pytest.fixture
def recipe_setup(setup, tmp_path):
    """A direction holding the shared synthetic scene (a floor beside a blue block)."""
    store, direction, _ = setup
    scene = floor_and_block_scene()
    path = tmp_path / "scene.png"
    Image.fromarray(scene.picture, "RGBA").save(path)
    imported = import_picture(store, direction, path, reviewer=Reviewer.HUMAN, now=NOW)
    return store, direction, imported


def _recipe_decompose(store, direction, source, kind="wood"):
    """Decompose a floor of `kind` drawn as a shape and a box, no image service."""
    return decompose(
        store, direction, source,
        surfaces=[Surface(text=f"{kind} floor", name=f"{kind}-floor", kind=kind, **FLOOR_AREA)],
        now=NOW,
    )


def test_a_failed_cleanup_does_not_hide_the_refusal(recipe_setup, monkeypatch, caplog):
    """Catches: a folder that will not delete replacing the sentence the person should see."""
    store, direction, source = recipe_setup

    def refuse(*_args):
        raise RecipeRefused("the box is too flat")

    def stuck(_folder):
        raise PermissionError("locked")

    monkeypatch.setattr("pixelmagic.stages.decompose._draw_recipe_tiles", refuse)
    monkeypatch.setattr("pixelmagic.stages.decompose._remove_cutout_folder", stuck)
    with pytest.raises(RecipeRefused, match="the box is too flat"):
        _recipe_decompose(store, direction, source)
    assert "could not remove decomposition" in caplog.text


@pytest.mark.parametrize("kind", ["wood", "stone", "plaster"])
def test_recipe_tiles_are_exactly_cell_size(setup, tmp_path, kind):
    """Catches: a recipe drawing an off-by-one tile, a wrong count (32), or a wrong block
    place name."""
    store, direction, _ = setup
    path = tmp_path / "scene.png"
    Image.fromarray(_scene_with_floor(RECIPE_FLOORS[kind]), "RGBA").save(path)
    source = import_picture(store, direction, path, reviewer=Reviewer.HUMAN, now=NOW)
    result = _recipe_decompose(store, direction, source, kind)
    names = [cutout.name for cutout in result.cutouts]
    expected = [
        f"{kind}-floor-v{n}-{place}" for n in range(1, 9) for place in ("nw", "ne", "sw", "se")
    ]
    assert names == expected
    for name in names:
        with Image.open(store.cutout_dir("cosy-den", result.id, name) / "source.png") as tile:
            assert tile.size == (direction.cell_size, direction.cell_size)


def test_retyping_a_surface_with_a_master_file_says_it_is_kept(recipe_setup):
    """Catches: a person typing the surface again and not being told that export will still
    take its tiles from the master file they edited, not from the new run."""
    store, direction, source = recipe_setup
    assert _recipe_decompose(store, direction, source).warnings == []
    master = store.surface_master("cosy-den", "wood-floor")
    master.write_bytes(b"a person's master")
    again = _recipe_decompose(store, direction, source)
    assert len(again.warnings) == 1
    assert "wood-floor.aseprite already exists and is kept" in again.warnings[0]
    assert master.read_bytes() == b"a person's master"


def test_recipe_colours_come_from_the_box_and_the_outline_is_the_joined_shapes(recipe_setup):
    """Catches: colours read from the whole drawn area instead of the box (a shape that also
    covers the blue block would blend blue into the tiles), a shape lost when two are joined,
    or the saved outline covering pixels no shape drew."""
    store, direction, source = recipe_setup
    # Corners are filled inclusively, so these cover columns 0-39 and 110-149.
    floor_left = ((0, 0), (39, 0), (39, 87), (0, 87))
    block = ((110, 20), (149, 20), (149, 59), (110, 59))
    result = decompose(
        store, direction, source,
        surfaces=[Surface(text="wood floor", name="wood-floor", kind="wood",
                          shapes=(floor_left, block), box=(10, 10, 50, 50))],
        now=NOW,
    )
    floor_colours = {rgb for rgb, _ in FLOOR_COLOURS}
    recipe = result.cutouts[0].recipe
    recorded = {tuple(int(shade[i : i + 2], 16) for i in (1, 3, 5)) for shade in recipe.shades}
    assert recorded == floor_colours
    assert recipe.point is None and recipe.segmenter is None
    assert recipe.box == (10, 10, 50, 50)
    assert [[tuple(c) for c in shape] for shape in recipe.shapes] == [list(floor_left), list(block)]

    cutout = result.cutouts[0]
    with Image.open(
        store.surface_dir("cosy-den", result.id, "wood-floor") / "outline.png"
    ) as picture:
        outline = np.asarray(picture) > 0
    x0, y0, x1, y1 = cutout.bbox
    assert (x0, y0, x1, y1) == (0, 0, 150, 88)
    expected = np.zeros((96, 160), dtype=bool)
    expected[0:88, 0:40] = True
    expected[20:60, 110:150] = True
    assert np.array_equal(outline, expected[y0:y1, x0:x1])


def test_a_box_outside_the_picture_is_refused_before_any_folder_is_made(recipe_setup):
    """Catches: a box hanging off the picture being read as if it were all there (short of
    pixels, or numpy slicing quietly clipping it) after a decomposition folder was already
    left behind."""
    store, direction, source = recipe_setup
    with pytest.raises(RecipeRefused, match="wholly inside"):
        decompose(
            store, direction, source,
            surfaces=[Surface(text="wood floor", name="wood-floor", kind="wood",
                              shapes=FLOOR_AREA["shapes"], box=(130, 10, 170, 50))],
            now=NOW,
        )
    assert not store.decompositions_dir("cosy-den").exists() or not list(
        store.decompositions_dir("cosy-den").iterdir()
    )


def test_a_recipe_record_from_before_drawn_surfaces_still_loads():
    """Catches: making `shapes` and `box` required, which would stop every older cutout,
    asset and manifest sidecar (made with a `point` and a `segmenter`) from being read."""
    from pixelmagic.model import RecipeRecord

    old = RecipeRecord.model_validate_json(
        '{"kind": "wood", "layout": "block", "place": "nw", "surface": "floor",'
        ' "point": [412, 630], "segmenter": "cli", "shades": ["#102030"], "shares": [1.0]}'
    )
    assert old.point == (412, 630) and old.segmenter == "cli"
    assert old.shapes is None and old.box is None


def test_a_scene_never_takes_the_name_of_a_cutout_that_is_not_kept_yet(tmp_path):
    """A later `keep` writes into `assets/<cutout name>` without asking, so a
    scene named like a waiting cut-out would be silently overwritten by it."""
    store, direction, decomposition = cut_up(tmp_path)
    chair_picture = generate(
        store, direction, "a wooden chair", kind=PromptKind.SCENE,
        service=FakeImageService(), approval=AutoApprove(), size=(96, 96), now=NOW,
    )
    scene = keep_scene(store, direction, chair_picture, refiner=StubRefiner())
    assert scene != "wooden-chair"
    promote_cutouts(store, direction, decomposition)
    kept = store.read_model(store.asset_dir("cosy-den", scene) / "asset.json", AssetRecord)
    assert kept.kind is AssetKind.SCENE


def _floor(name="wood floor"):
    """The shared floor as a surface called `name` (words as typed; the slug is its name)."""
    return Surface(
        text=name, name=name.replace(" ", "-"), kind="wood", **FLOOR_AREA
    )


def test_the_colour_reader_gives_what_decompose_records_and_refuses_as_it_does(recipe_setup):
    """Catches: the live colour readout showing colours the drawn tiles do not use, or a box
    (or a direction) the page accepts while the real draw refuses it with another sentence."""
    store, direction, source = recipe_setup
    read = read_box_shades(direction, source, "wood", FLOOR_AREA["box"], "wood floor")
    drawn = decompose(store, direction, source, surfaces=[_floor()], now=NOW)
    recorded = drawn.cutouts[0].recipe
    assert [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in read] == recorded.shades
    assert [share for _, share in read] == recorded.shares

    hanging = (130, 10, 170, 50)
    with pytest.raises(RecipeRefused) as from_reader:
        read_box_shades(direction, source, "wood", hanging, "wood floor")
    with pytest.raises(RecipeRefused) as from_decompose:
        decompose(
            store, direction, source,
            surfaces=[Surface(text="wood floor", name="wood-floor", kind="wood",
                              shapes=FLOOR_AREA["shapes"], box=hanging)],
            now=NOW + timedelta(seconds=1),
        )
    assert str(from_reader.value) == str(from_decompose.value)

    coarse = direction.model_copy(update={"cell_size": 8})
    with pytest.raises(RecipeRefused) as from_reader:
        read_box_shades(coarse, source, "wood", FLOOR_AREA["box"], "wood floor")
    with pytest.raises(RecipeRefused) as from_decompose:
        decompose(store, coarse, source, surfaces=[_floor()], now=NOW + timedelta(seconds=2))
    assert str(from_reader.value) == str(from_decompose.value)


def test_drawing_a_surface_twice_leaves_one_set_of_tiles_and_32_assets(recipe_setup):
    """Catches: an Edit re-draw leaving the old run's 32 tiles behind as a duplicate, or
    taking the assets away with them."""
    store, direction, source = recipe_setup
    first = draw_surface(store, direction, source, _floor(), now=NOW)
    second = draw_surface(
        store, direction, source, _floor(), replaces="wood-floor", now=NOW + timedelta(seconds=1)
    )
    assert store.list_decompositions("cosy-den") == [second.id]
    assert not store.decomposition_dir("cosy-den", first.id).exists()
    assert len(store.list_assets("cosy-den")) == 32
    (listed,) = drawn_surfaces(store, "cosy-den", source.request.id)
    assert listed.name == "wood-floor"


def test_a_refused_redraw_keeps_the_surface_as_it_was(recipe_setup):
    """Catches: removing the old tiles before the new draw is known to have worked, which
    would lose a person's surface to one bad edit."""
    store, direction, source = recipe_setup
    first = draw_surface(store, direction, source, _floor(), now=NOW)
    off_picture = Surface(
        text="wood floor", name="wood-floor", kind="wood",
        shapes=FLOOR_AREA["shapes"], box=(130, 10, 170, 50),
    )
    with pytest.raises(RecipeRefused):
        draw_surface(
            store, direction, source, off_picture, replaces="wood-floor",
            now=NOW + timedelta(seconds=1),
        )
    assert store.list_decompositions("cosy-den") == [first.id]
    assert len(load_decomposition(store, "cosy-den", first.id).cutouts) == 32
    assert len(store.list_assets("cosy-den")) == 32


def test_removing_a_surface_leaves_other_surfaces_and_the_master_file(recipe_setup):
    """Catches: removing a surface throwing away another surface's tiles or assets, or the
    person's Aseprite touch-ups in its master file."""
    store, direction, source = recipe_setup
    draw_surface(store, direction, source, _floor("wood floor"), now=NOW)
    draw_surface(store, direction, source, _floor("oak floor"), now=NOW + timedelta(seconds=1))
    master = store.surface_master("cosy-den", "wood-floor")
    master.write_bytes(b"a person's touch-ups")

    remove_surface(store, direction, source.request.id, "wood-floor")

    assert [d.name for d in drawn_surfaces(store, "cosy-den", source.request.id)] == ["oak-floor"]
    assert store.list_assets("cosy-den") == sorted(
        f"oak-floor-v{n}-{place}" for n in range(1, 9) for place in ("nw", "ne", "sw", "se")
    )
    assert master.read_bytes() == b"a person's touch-ups"
    with pytest.raises(FileNotFoundError, match="wood-floor"):
        remove_surface(store, direction, source.request.id, "wood-floor")


def test_editing_a_surface_under_a_new_name_leaves_one_and_a_new_draw_cannot_take_a_name(
    recipe_setup,
):
    """Catches: drawing a new "wood floor" silently overwriting the one already drawn, and an
    Edit that renames a surface leaving the old name behind as a second surface."""
    store, direction, source = recipe_setup
    draw_surface(store, direction, source, _floor("wood floor"), now=NOW)

    with pytest.raises(AlreadyExists, match="That name is taken"):
        draw_surface(
            store, direction, source, _floor("wood floor"), now=NOW + timedelta(seconds=1)
        )
    assert len(store.list_decompositions("cosy-den")) == 1
    assert len(store.list_assets("cosy-den")) == 32

    with pytest.raises(FileNotFoundError, match="marble"):
        draw_surface(
            store, direction, source, _floor("oak floor"), replaces="marble",
            now=NOW + timedelta(seconds=2),
        )

    draw_surface(
        store, direction, source, _floor("oak floor"), replaces="wood-floor",
        now=NOW + timedelta(seconds=3),
    )
    assert [d.name for d in drawn_surfaces(store, "cosy-den", source.request.id)] == ["oak-floor"]
    assert all(name.startswith("oak-floor") for name in store.list_assets("cosy-den"))
    assert len(store.list_assets("cosy-den")) == 32


def test_a_surface_name_drawn_on_another_picture_is_refused_not_taken_over(recipe_setup, tmp_path):
    """Catches: drawing the same name on a second picture silently replacing the first
    picture's tiles and assets."""
    store, direction, first = recipe_setup
    draw_surface(store, direction, first, _floor(), now=NOW)
    path = tmp_path / "second.png"
    Image.fromarray(floor_and_block_scene().picture, "RGBA").save(path)
    second = import_picture(
        store, direction, path, reviewer=Reviewer.HUMAN, now=NOW + timedelta(seconds=1)
    )
    with pytest.raises(RecipeRefused, match="different name"):
        draw_surface(store, direction, second, _floor(), now=NOW + timedelta(seconds=2))
    assert [d.name for d in drawn_surfaces(store, "cosy-den", first.request.id)] == ["wood-floor"]
    assert len(store.list_assets("cosy-den")) == 32
