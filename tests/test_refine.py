from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    CohesionRecord,
    Direction,
    GridCandidate,
    GridOffset,
    GridRebuild,
    PitchSource,
    RecipeRecord,
    RefineReport,
    RefinerSettings,
    TileRole,
)
from pixelmagic.refiner.base import (
    EXIT_USAGE,
    LOW_GRID_CONFIDENCE,
    MAX_FORCE_PIXELS,
    PIXELMAGIC_WARNING_CODES,
    SIZE_SETTING_DROPPED,
    WARNING_CODES,
    RefineFailure,
    RefineRequest,
    RefinerError,
    RefineResult,
)
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.stages.decompose import set_scene_pitch
from pixelmagic.stages.generate import MAX_IMAGE_SIZE
from pixelmagic.stages.recipes import RecipeTileRefused
from pixelmagic.stages.refine import (
    ASSET_RECORD_UNREADABLE,
    OLD_ITEM_REFUSED,
    OLD_ITEM_ROUTE,
    PICTURE_UNREADABLE,
    PIXELMAGIC_REFUSAL_CODES,
    REBUILD_REFINER,
    REBUILD_ROUTE,
    RECIPE_REFINER,
    RECIPE_ROUTE,
    REFINED_FILE,
    REPORT_FILE,
    TARGET_NOT_USABLE,
    WRONG_SIZE,
    load_report,
    needs_attention,
    refine_asset,
    refine_direction,
    require_asset,
    resolve_settings,
    set_refiner_defaults,
    validate_target,
)
from pixelmagic.stages.slice import SEAM_VISIBLE
from pixelmagic.store import Store

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


class RecordingRefiner(StubRefiner):
    """A stub that remembers the settings it was handed."""

    def __init__(self):
        self.seen: list[RefinerSettings] = []
        self.batch_calls = 0

    def refine(self, request: RefineRequest) -> RefineResult:
        self.seen.append(request.settings)
        return super().refine(request)

    def batch(self, requests):
        self.batch_calls += 1
        return super().batch(requests)


class WarningRefiner(StubRefiner):
    """A stub that always reports a shaky grid, with candidates."""

    def refine(self, request: RefineRequest) -> RefineResult:
        result = super().refine(request)
        result.warnings = [LOW_GRID_CONFIDENCE]
        # A real GridCandidate rather than a dict laundered through
        # model_validate: the dict round-trip works but makes pydantic warn on
        # every call, and the warnings drown the ones worth reading.
        result.grid_candidates = [
            GridCandidate(
                out_w=32,
                out_h=32,
                cell_w=8.0,
                cell_h=8.0,
                offset=GridOffset(x=0, y=0),
                method="autocorrelation",
                total_score=0.7,
                confidence=0.4,
            )
        ]
        return result


def add_asset(store, name):
    """Put one 8x8 source.png where the refine stage expects to find it."""
    folder = store.asset_dir("cosy-den", name)
    folder.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[2:6, 2:6] = (10, 20, 200, 255)
    Image.fromarray(pixels, "RGBA").save(folder / "source.png")


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(
            name="cosy-den",
            created=NOW,
            style="Warm SNES interiors.",
            refiner_defaults=RefinerSettings(
                preset="crisp-sprite", quick={"detailLevel": "balanced", "dithering": "off"}
            ),
        )
    )
    for name in ("wooden-chair", "round-rug"):
        add_asset(store, name)
    return store, direction


class BatchRecordingRefiner(StubRefiner):
    """The stand-in, plus a record of how the work was split into batches.

    Not `RecordingRefiner`: `tests/test_refine.py` has had a class of that name
    since M1 — it records `settings`, not batches — and a second definition
    would be `F811` *and* would silently replace the first for every test
    defined above it, because they resolve the global at call time. Four M1
    tests fail that way, all with `AttributeError: … has no attribute 'seen'`.
    """

    def __init__(self):
        super().__init__()
        self.batches: list[list[str]] = []

    def batch(self, requests):
        # The asset folder's name is the only thing that identifies a request.
        self.batches.append([request.input_path.parent.name for request in requests])
        return super().batch(requests)


def test_layers_merge_key_by_key():
    merged = resolve_settings(
        RefinerSettings(
            preset="crisp-sprite", quick={"detailLevel": "balanced", "dithering": "off"}
        ),
        RefinerSettings(quick={"dithering": "subtle"}, advanced={"trim": True}),
    )
    assert merged.preset == "crisp-sprite"
    assert merged.quick == {"detailLevel": "balanced", "dithering": "subtle"}
    assert merged.advanced == {"trim": True}


def test_a_later_preset_replaces_an_earlier_one():
    merged = resolve_settings(
        RefinerSettings(preset="auto", scale=2), RefinerSettings(preset="retro-game")
    )
    assert merged.preset == "retro-game"
    assert merged.scale == 2


def test_none_layers_are_skipped():
    merged = resolve_settings(RefinerSettings(preset="auto"), None)
    assert merged.preset == "auto"


def test_new_defaults_layer_onto_the_ones_already_stored(setup):
    store, direction = setup
    saved = set_refiner_defaults(
        store, direction, RefinerSettings(quick={"dithering": "subtle"}, scale=2)
    )
    assert saved.refiner_defaults.preset == "crisp-sprite"
    assert saved.refiner_defaults.quick == {"detailLevel": "balanced", "dithering": "subtle"}
    assert saved.refiner_defaults.scale == 2
    # And it is on disk, not just in the object that was handed in.
    assert store.load_direction("cosy-den").refiner_defaults.scale == 2


def test_defaults_the_refiner_would_refuse_are_never_written(setup):
    """They would refuse every later run, curable only by editing the sidecar."""
    store, direction = setup
    with pytest.raises(RefinerError) as raised:
        set_refiner_defaults(store, direction, RefinerSettings(quick={"background": "pick"}))
    assert raised.value.code == "INVALID_SETTINGS"
    assert raised.value.exit_code == EXIT_USAGE
    assert store.load_direction("cosy-den").refiner_defaults.quick == {
        "detailLevel": "balanced", "dithering": "off"
    }


def test_an_existing_directions_settings_still_pick_up_the_house_baseline(setup):
    """`setup`'s direction has a preset and two quick keys, but nothing about
    reductionMode or cohesion -- exactly the shape a pre-M4 direction.json
    has. Refining it must still send the M4 baseline, with no file
    rewritten."""
    store, direction = setup
    refiner = StubRefiner()
    report = refine_asset(store, direction, "wooden-chair", refiner=refiner)
    assert report.requested.quick["reductionMode"] == "24"
    assert report.requested.advanced == {
        "cohesion": True, "cohesionFamilies": 14, "cohesionTones": 2, "cleanup": True,
    }
    # The direction's own file is untouched -- the baseline is layered at
    # refine time, never written back.
    assert store.load_direction("cosy-den").refiner_defaults.quick == {
        "detailLevel": "balanced", "dithering": "off"
    }


def test_the_report_names_the_tool_that_ran(setup):
    store, direction = setup
    report = refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())
    assert report.refiner == "stub"


def test_a_refused_asset_records_an_error_code_not_a_warning(setup):
    store, direction = setup

    class RefusingRefiner(StubRefiner):
        def batch(self, requests):
            return [
                RefineFailure(
                    input_path=request.input_path,
                    code="INVALID_SETTINGS",
                    message="bad cellScale",
                )
                for request in requests
            ]

    reports = refine_direction(store, direction, refiner=RefusingRefiner())
    assert [report.error_code for report in reports] == ["INVALID_SETTINGS"] * 2
    assert all(report.warnings == [] for report in reports)
    assert all(not needs_attention(report) for report in reports)
    # It took no route at all; saying "preserve" would read as "copied through".
    assert all(report.route == "" for report in reports)


def test_a_refusal_s_message_reaches_the_sidecar(setup):
    """The CLI's own hint has to survive the round-trip, not just its code."""
    store, direction = setup

    class RefusingRefiner(StubRefiner):
        def batch(self, requests):
            return [
                RefineFailure(
                    input_path=request.input_path,
                    code="INVALID_SETTINGS",
                    message="bad cellScale, try lower",
                )
                for request in requests
            ]

    refine_direction(store, direction, refiner=RefusingRefiner())
    reloaded = load_report(store, "cosy-den", "wooden-chair")
    assert reloaded.error_message == "bad cellScale, try lower"


def test_one_refused_asset_costs_only_itself(setup):
    """Every asset still gets a receipt, and the good ones still get refined."""
    store, direction = setup

    class PickyRefiner(StubRefiner):
        def batch(self, requests):
            items = super().batch(requests)
            for index, request in enumerate(requests):
                if request.input_path.parent.name == "round-rug":
                    items[index] = RefineFailure(
                        input_path=request.input_path,
                        code="UNSUPPORTED_INPUT",
                        message="not an image",
                    )
            return items

    reports = {r.asset: r for r in refine_direction(store, direction, refiner=PickyRefiner())}
    assert reports["round-rug"].error_code == "UNSUPPORTED_INPUT"
    assert reports["wooden-chair"].error_code is None
    for name in ("round-rug", "wooden-chair"):
        assert (store.asset_dir("cosy-den", name) / "refine.json").exists()
    assert (store.asset_dir("cosy-den", "wooden-chair") / "refined.png").exists()
    assert not (store.asset_dir("cosy-den", "round-rug") / "refined.png").exists()


def test_a_refusal_takes_the_refined_png_an_earlier_run_left(setup):
    """A stale PNG beside a refusal is the one arrangement that reads as success.

    Export skips an asset on `refined.png` being absent, not on the report's
    verdict, so leaving last run's file would put last run's pixels in a bundle
    whose manifest describes settings that never produced them.
    """
    store, direction = setup

    class RefusingRefiner(StubRefiner):
        def batch(self, requests):
            return [
                RefineFailure(
                    input_path=request.input_path,
                    code="INVALID_SETTINGS",
                    message="bad cellScale",
                )
                for request in requests
            ]

    refine_direction(store, direction, refiner=StubRefiner())
    refined = [
        store.asset_dir("cosy-den", name) / "refined.png"
        for name in ("round-rug", "wooden-chair")
    ]
    assert all(path.exists() for path in refined)

    reports = refine_direction(store, direction, refiner=RefusingRefiner())

    assert [report.error_code for report in reports] == ["INVALID_SETTINGS"] * 2
    assert not any(path.exists() for path in refined)


def test_a_refused_single_asset_loses_its_stale_refined_png_too(setup):
    """The one-asset path refuses in its own branch, so it has to clear it too."""
    store, direction = setup
    folder = store.asset_dir("cosy-den", "wooden-chair")

    refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())
    assert (folder / "refined.png").exists()

    (folder / "source.png").unlink()
    report = refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())

    assert report.error_code == "INPUT_NOT_FOUND"
    assert not (folder / "refined.png").exists()


def test_a_batch_that_loses_an_item_is_an_error(setup):
    store, direction = setup

    class ForgetfulRefiner(StubRefiner):
        def batch(self, requests):
            return []

    with pytest.raises(ValueError):
        refine_direction(store, direction, refiner=ForgetfulRefiner())


def test_refine_asset_writes_refined_png_and_refine_json(setup):
    store, direction = setup
    report = refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())
    folder = store.asset_dir("cosy-den", "wooden-chair")
    assert (folder / "refined.png").exists()
    assert (folder / "refine.json").exists()
    assert report.asset == "wooden-chair"
    assert report.route == "preserve"
    assert report.width == 8


def test_a_missing_source_is_refused_not_raised(setup):
    """The real StubRefiner never even sees a request for a missing source."""
    store, direction = setup
    folder = store.asset_dir("cosy-den", "wooden-chair")
    (folder / "source.png").unlink()

    report = refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())

    assert report.error_code == "INPUT_NOT_FOUND"
    assert "source.png" in report.error_message
    assert report.route == ""
    assert not (folder / "refined.png").exists()
    assert load_report(store, "cosy-den", "wooden-chair").error_code == "INPUT_NOT_FOUND"


def test_the_report_records_the_settings_that_were_asked_for(setup):
    store, direction = setup
    refiner = RecordingRefiner()
    report = refine_asset(
        store, direction, "wooden-chair",
        refiner=refiner, override=RefinerSettings(quick={"dithering": "subtle"}),
    )
    assert refiner.seen[0].preset == "crisp-sprite"
    assert refiner.seen[0].quick["dithering"] == "subtle"
    assert report.requested.quick["dithering"] == "subtle"
    assert report.effective_options["dithering"] == "subtle"


def test_a_report_round_trips_off_disk(setup):
    store, direction = setup
    refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())
    reloaded = load_report(store, "cosy-den", "wooden-chair")
    assert reloaded.asset == "wooden-chair"
    assert reloaded.palette == ["#0a14c8"]


def test_low_grid_confidence_is_surfaced_with_its_candidates(setup):
    store, direction = setup
    report = refine_asset(store, direction, "wooden-chair", refiner=WarningRefiner())
    assert needs_attention(report) is True
    assert report.grid_candidates[0].cell_w == 8.0


def test_a_clean_report_needs_no_attention(setup):
    store, direction = setup
    report = refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())
    assert needs_attention(report) is False


def test_the_candidates_reach_the_receipt_on_disk(setup, candidate_refiner):
    store, direction = setup
    refine_asset(store, direction, "wooden-chair", refiner=candidate_refiner)
    report = load_report(store, "cosy-den", "wooden-chair")
    assert [candidate.id for candidate in report.candidates] == [
        "auto-result:40x22", "grid:20x11"
    ]
    assert report.candidate_id is None
    assert needs_attention(report)


def test_asking_for_a_candidate_is_passed_on_and_written_down(setup, candidate_refiner):
    store, direction = setup
    refine_asset(
        store, direction, "wooden-chair",
        refiner=candidate_refiner, candidate_id="grid:20x11",
    )
    assert candidate_refiner.asked == ["grid:20x11"]
    assert load_report(store, "cosy-den", "wooden-chair").candidate_id == "grid:20x11"


def test_a_refused_asset_records_no_candidate_it_never_used(setup):
    """A report that offered a re-pick here would offer one that cannot work."""
    store, direction = setup
    (store.asset_dir("cosy-den", "wooden-chair") / "source.png").unlink()
    report = refine_asset(
        store, direction, "wooden-chair", refiner=StubRefiner(), candidate_id="grid:20x11"
    )
    assert report.error_code == "INPUT_NOT_FOUND"
    assert report.candidates == []
    assert report.candidate_id is None


def test_require_asset_lets_a_real_one_through(setup):
    store, direction = setup
    assert require_asset(store, "cosy-den", "wooden-chair") == "wooden-chair"


def test_require_asset_names_the_ones_that_exist(setup):
    store, direction = setup
    with pytest.raises(ValueError, match="wooden-chair"):
        require_asset(store, "cosy-den", "sofa")


def test_require_asset_names_the_deleted_source_png_not_a_missing_asset(setup):
    """Catches: a hand-deleted `source.png` being reported as "no asset called ...",
    which sends the person looking for a typo instead of a missing picture."""
    store, direction = setup
    (store.asset_dir("cosy-den", "wooden-chair") / "source.png").unlink()
    with pytest.raises(ValueError, match="source.png") as refusal:
        require_asset(store, "cosy-den", "wooden-chair")
    assert "no asset called" not in str(refusal.value)


def test_require_asset_refuses_a_name_that_could_leave_the_folder(setup):
    store, direction = setup
    with pytest.raises(ValueError, match="not a name PixelMagic uses"):
        require_asset(store, "cosy-den", "../escape")


@pytest.mark.parametrize(
    "target", [(0, 5), (5, 0), (MAX_IMAGE_SIZE + 1, 5), (5, MAX_IMAGE_SIZE + 1), (-3, 5)]
)
def test_a_target_outside_the_image_bounds_is_refused(target):
    with pytest.raises(ValueError, match="target"):
        validate_target(target)


@pytest.mark.parametrize("target", [(1, 1), (17, 32), (MAX_FORCE_PIXELS, MAX_FORCE_PIXELS)])
def test_a_target_inside_the_image_bounds_is_accepted(target):
    validate_target(target)


def test_a_target_that_is_not_whole_numbers_is_refused():
    with pytest.raises(ValueError, match="target"):
        validate_target((17.5, 32))
    with pytest.raises(ValueError, match="target"):
        validate_target((True, 32))


class TargetRecordingRefiner(StubRefiner):
    """A stub that remembers the target on every request it was handed."""

    def __init__(self):
        self.targets: list[tuple[int, int] | None] = []
        self.batch_calls = 0
        self._recording_batch = False

    def refine(self, request: RefineRequest) -> RefineResult:
        if not self._recording_batch:
            self.targets.append(request.target)
        return super().refine(request)

    def batch(self, requests):
        self.batch_calls += 1
        self.targets.extend(request.target for request in requests)
        self._recording_batch = True
        try:
            return super().batch(requests)
        finally:
            self._recording_batch = False


class ForcedSizeRefiner(TargetRecordingRefiner):
    """A recording stand-in that claims to honour a forced size, like the real
    refiner, so the 1024-per-side rule applies to it."""

    honours_forced_size = True


def stamp_target(store, name, target, *, kind=AssetKind.SCENE):
    """Write an asset.json for a hand-staged asset, carrying `target`.

    A scene by default: it is the one kind a refiner is still asked to refine."""
    store.write_model(
        store.asset_dir("cosy-den", name) / "asset.json",
        AssetRecord(
            name=name, kind=kind, source_generation="20260903-191233",
            source_prompt="draw a wooden chair", service="fake", target=target,
        ),
    )


class RefusesRebuiltRefiner(StubRefiner):
    """A stand-in that raises when asked anything about the rebuilt asset's pictures."""

    def _check(self, paths):
        """Fail loudly if any of ``paths`` belongs to the rebuilt asset."""
        if any("rebuilt-lamp" in str(path) for path in paths):
            raise AssertionError("the rebuilt asset was sent to the refiner")

    def refine(self, request):
        """Refuse the rebuilt asset, otherwise refine as the stand-in does."""
        self._check([request.input_path])
        return super().refine(request)

    def batch(self, requests):
        """Refuse a batch holding the rebuilt asset, otherwise as the stand-in does."""
        self._check([r.input_path for r in requests])
        return super().batch(requests)

    def fit_palette(self, path, *args, **kwargs):
        """Refuse a palette fit over the rebuilt asset, otherwise as the stand-in does."""
        self._check([path])
        return super().fit_palette(path, *args, **kwargs)

    def analyze(self, *args, **kwargs):
        """Refining never measures a pitch, so any call here is a bug."""
        raise AssertionError("nothing is analysed when refining")


def test_a_rebuilt_prop_is_kept_as_it_is_and_never_sent_to_the_refiner(setup):
    """A rebuilt prop is already one pixel per art pixel; PixelRefiner resampling
    it would undo the rebuild. It must still get `refined.png` and `refine.json`,
    or export drops it from the bundle."""
    store, direction = setup
    add_asset(store, "rebuilt-lamp")
    stamp_target(store, "wooden-chair", (8, 8))
    stamp_target(store, "rebuilt-lamp", (8, 8))
    path = store.asset_dir("cosy-den", "rebuilt-lamp") / "asset.json"
    store.write_model(
        path,
        store.read_model(path, AssetRecord).model_copy(
            update={
                "grid_rebuild": GridRebuild(
                    pitch_x=4.0, pitch_y=4.0, phase_x=0.0, phase_y=0.0,
                    confidence=0.1, native_size=(8, 8),
                )
            }
        ),
    )
    folder = store.asset_dir("cosy-den", "rebuilt-lamp")
    chair = store.asset_dir("cosy-den", "wooden-chair")
    refiner = RefusesRebuiltRefiner()

    reports = refine_direction(store, direction, refiner=refiner)

    assert [r.asset for r in reports] == ["rebuilt-lamp", "round-rug", "wooden-chair"]
    assert (folder / REFINED_FILE).read_bytes() == (folder / "source.png").read_bytes()
    report = store.read_model(folder / REPORT_FILE, RefineReport)
    assert (report.refiner, report.route) == (REBUILD_REFINER, REBUILD_ROUTE)
    assert (report.width, report.height, report.color_count) == (8, 8, 1)
    assert report.palette == ["#0a14c8"]
    assert (chair / REFINED_FILE).exists()
    assert store.read_model(chair / REPORT_FILE, RefineReport).refiner == "stub"

    # A corrupt source.png costs only that asset: a receipt on disk, no stale
    # refined.png, and a rebuilt prop handled after it (the rebuilt ones are
    # processed in name order) still gets its files.
    add_asset(store, "rebuilt-lamp-2")
    stamp_target(store, "rebuilt-lamp-2", (8, 8))
    later = store.asset_dir("cosy-den", "rebuilt-lamp-2")
    later_path = later / "asset.json"
    store.write_model(
        later_path,
        store.read_model(later_path, AssetRecord).model_copy(
            update={"grid_rebuild": store.read_model(path, AssetRecord).grid_rebuild}
        ),
    )
    (folder / "source.png").write_bytes(b"not a png")
    reports = refine_direction(store, direction, refiner=refiner)
    by_name = {r.asset: r for r in reports}
    assert by_name["rebuilt-lamp"].error_code == PICTURE_UNREADABLE
    assert not (folder / REFINED_FILE).exists()
    stored = load_report(store, "cosy-den", "rebuilt-lamp")
    assert stored.error_code == PICTURE_UNREADABLE
    assert (later / REFINED_FILE).read_bytes() == (later / "source.png").read_bytes()
    assert load_report(store, "cosy-den", "rebuilt-lamp-2").error_code is None


class RefusesRecipeRefiner(RefusesRebuiltRefiner):
    """A stand-in that raises when asked anything about the recipe tile's pictures."""

    def _check(self, paths):
        """Fail loudly if any of ``paths`` belongs to the recipe tile."""
        if any("wood-floor-v1" in str(path) for path in paths):
            raise AssertionError("the recipe tile was sent to the refiner")


def test_a_recipe_tile_is_kept_as_it_is_and_never_sent_to_the_refiner(setup):
    """A recipe tile is drawn crisp at 16 x 16; PixelRefiner resampling it would
    destroy the recipe's shades. It must still get `refined.png` and `refine.json`,
    or export drops it from the bundle, and a setting aimed at it is refused."""
    store, direction = setup
    add_asset(store, "wood-floor-v1")
    stamp_target(store, "wooden-chair", (8, 8))
    folder = store.asset_dir("cosy-den", "wood-floor-v1")
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[:, :8] = (120, 80, 40, 255)
    pixels[:, 8:] = (150, 100, 50, 255)
    Image.fromarray(pixels, "RGBA").save(folder / "source.png")
    path = folder / "asset.json"
    store.write_model(
        path,
        AssetRecord(
            name="wood-floor-v1", kind=AssetKind.TILE, source_generation="20260903-191233",
            recipe=RecipeRecord(
                kind="wood", layout="block", place="nw", surface="wood floor",
                point=(5, 5), segmenter="stub",
            ),
        ),
    )
    chair = store.asset_dir("cosy-den", "wooden-chair")
    refiner = RefusesRecipeRefiner()

    reports = refine_direction(store, direction, refiner=refiner)

    assert [r.asset for r in reports] == ["round-rug", "wood-floor-v1", "wooden-chair"]
    assert (folder / REFINED_FILE).read_bytes() == (folder / "source.png").read_bytes()
    report = store.read_model(folder / REPORT_FILE, RefineReport)
    assert (report.refiner, report.route) == (RECIPE_REFINER, RECIPE_ROUTE)
    assert (report.width, report.height, report.color_count) == (16, 16, 2)
    assert (chair / REFINED_FILE).exists()
    assert store.read_model(chair / REPORT_FILE, RefineReport).refiner == "stub"

    # A setting or a re-pick aimed at the tile is refused and writes nothing.
    (folder / REFINED_FILE).unlink()
    (folder / REPORT_FILE).unlink()
    setting = RefinerSettings(preset="clean")
    with pytest.raises(RecipeTileRefused, match="wood-floor-v1 is a recipe tile"):
        refine_asset(store, direction, "wood-floor-v1", refiner=refiner, override=setting)
    with pytest.raises(RecipeTileRefused):
        refine_asset(store, direction, "wood-floor-v1", refiner=refiner, candidate_id="c1")
    assert not (folder / REFINED_FILE).exists()
    assert not (folder / REPORT_FILE).exists()


def test_refine_asset_puts_the_assets_target_on_the_request(setup):
    store, direction = setup
    stamp_target(store, "wooden-chair", (17, 32))
    refiner = TargetRecordingRefiner()
    refine_asset(store, direction, "wooden-chair", refiner=refiner)
    assert refiner.targets == [(17, 32)]


def test_refine_direction_puts_each_assets_own_target_on_its_request(setup):
    store, direction = setup
    stamp_target(store, "round-rug", (40, 24))
    stamp_target(store, "wooden-chair", (17, 32))
    refiner = TargetRecordingRefiner()
    refine_direction(store, direction, refiner=refiner)
    assert refiner.targets == [(40, 24), (17, 32)]
    # Both are scenes (stamp_target's default kind), so the default call
    # refines each on its own: two solo batch-of-one calls, not one shared batch.
    assert refiner.batch_calls == 2


def test_an_asset_without_a_target_is_refined_at_the_refiners_own_guess(setup):
    """`add_asset` writes no asset.json: the M2 shape, and a hand-staged folder."""
    store, direction = setup
    refiner = TargetRecordingRefiner()
    refine_asset(store, direction, "wooden-chair", refiner=refiner)
    refine_direction(store, direction, refiner=refiner)
    assert refiner.targets == [None, None, None]


def test_an_m2_asset_json_with_no_target_field_is_refined_unforced(setup):
    store, direction = setup
    (store.asset_dir("cosy-den", "wooden-chair") / "asset.json").write_text(
        '{"name": "wooden-chair", "kind": "scene", "source_generation": "20260903-191233", '
        '"source_prompt": "a chair", "service": "fake", "cell": null}'
    )
    refiner = TargetRecordingRefiner()
    refine_asset(store, direction, "wooden-chair", refiner=refiner)
    assert refiner.targets == [None]


def test_a_corrupt_asset_json_names_the_asset_rather_than_a_bare_traceback(setup):
    """`refine_asset`, asked to refine one named asset, still lets the
    `ValueError` `_target_of` builds travel out of it rather than swallow a
    broken file as if it were the refiner refusing an asset: `cli.py`'s own
    `except (RefinerError, ValueError)` already knows how to turn that into
    one readable line, naming the asset and the file, instead of a bare
    pydantic traceback naming neither."""
    store, direction = setup
    (store.asset_dir("cosy-den", "wooden-chair") / "asset.json").write_text("not json")
    refiner = TargetRecordingRefiner()
    with pytest.raises(ValueError, match="wooden-chair") as caught:
        refine_asset(store, direction, "wooden-chair", refiner=refiner)
    assert not isinstance(caught.value, ValidationError)
    assert "asset.json" in str(caught.value)
    assert refiner.targets == []


@pytest.mark.parametrize("broken", [b"not json", b"\xff\xfe"], ids=["not-json", "not-utf8"])
def test_a_corrupt_asset_json_costs_only_that_asset_in_the_direction(setup, broken):
    """`refine_direction` must not let one broken sidecar stop every other
    asset (CLAUDE.md: "one broken asset never costs the bundle"): the corrupt
    `wooden-chair` gets an `ASSET_RECORD_UNREADABLE` receipt naming it and the
    file, `round-rug` still refines, and nothing raises. Bytes that are not
    UTF-8 used to escape as a bare `UnicodeDecodeError` naming neither."""
    store, direction = setup
    (store.asset_dir("cosy-den", "wooden-chair") / "asset.json").write_bytes(broken)
    refiner = TargetRecordingRefiner()
    refine_direction(store, direction, refiner=refiner)
    stored = load_report(store, "cosy-den", "wooden-chair")
    assert stored.error_code == ASSET_RECORD_UNREADABLE
    assert "wooden-chair" in stored.error_message
    assert "asset.json" in stored.error_message
    assert load_report(store, "cosy-den", "round-rug").error_code is None
    assert (store.asset_dir("cosy-den", "round-rug") / REFINED_FILE).exists()
    assert refiner.targets == [None]
    assert refiner.batch_calls == 1


@pytest.mark.parametrize("target", [(0, 5), (5, 0), (MAX_IMAGE_SIZE + 1, 5), (-3, 5)])
def test_an_out_of_range_target_names_the_asset_and_does_not_stop_the_direction(setup, target):
    """`refine_asset`, asked for one named asset, still raises — but the
    message must name it. `refine_direction` no longer refuses the whole
    direction over one asset's bad target: it gives that asset a receipt and
    still refines everything else (CLAUDE.md: "one broken asset never costs
    the bundle")."""
    store, direction = setup
    stamp_target(store, "wooden-chair", target)
    refiner = TargetRecordingRefiner()
    with pytest.raises(ValueError, match="wooden-chair"):
        refine_asset(store, direction, "wooden-chair", refiner=refiner)
    assert refiner.targets == []

    refine_direction(store, direction, refiner=refiner)
    stored = load_report(store, "cosy-den", "wooden-chair")
    assert stored.error_code == TARGET_NOT_USABLE
    assert "wooden-chair" in stored.error_message
    assert "target" in stored.error_message
    assert load_report(store, "cosy-den", "round-rug").error_code is None
    assert (store.asset_dir("cosy-den", "round-rug") / REFINED_FILE).exists()
    # Only the good asset ever reached the refiner: the bad one's request was
    # never built.
    assert refiner.targets == [None]
    assert refiner.batch_calls == 1


@pytest.mark.parametrize("target", [(MAX_FORCE_PIXELS + 1, 16), (1536, 1024)])
def test_a_target_above_the_force_limit_names_the_asset_and_does_not_stop_the_direction(
    setup, target
):
    """A side over MAX_FORCE_PIXELS is well inside MAX_IMAGE_SIZE, so this is
    a different failure than the one above: the real CLI does not refuse a
    forcePixelsW/H over 1024, it clamps to 1024 and exits 0 (measured
    2026-09-23), so this bound has to be enforced here instead, for a refiner
    that honours a forced size (the stand-in does not; see the test below). Same recovery
    as the image-bound case: `refine_asset` names the asset it refuses,
    `refine_direction` gives it a receipt and keeps refining the rest — this
    is the M3.1 review's reproduction (`--size 1536` stopped the whole
    direction on the stand-in, whose pitch is always 1.0)."""
    store, direction = setup
    stamp_target(store, "wooden-chair", target)
    refiner = ForcedSizeRefiner()
    with pytest.raises(ValueError, match="wooden-chair"):
        refine_asset(store, direction, "wooden-chair", refiner=refiner)
    assert refiner.targets == []

    reports = refine_direction(store, direction, refiner=refiner)
    by_name = {report.asset: report for report in reports}
    assert by_name["wooden-chair"].error_code == TARGET_NOT_USABLE
    assert "wooden-chair" in by_name["wooden-chair"].error_message
    assert by_name["round-rug"].error_code is None
    assert refiner.batch_calls == 1


def test_the_stand_in_refines_a_scene_over_the_force_limit_at_its_own_size(setup):
    """Catches: #295, a bare offline run (stand-in refiner, pitch falling back
    to 1.0) refusing every scene over 1024 on a side with TARGET_NOT_USABLE,
    though the stand-in ignores the target and the 1024 limit is only the real
    refiner's."""
    store, direction = setup
    add_scene(store, "the-den", width=1536, height=1024)

    reports = {r.asset: r for r in refine_direction(store, direction, refiner=StubRefiner())}

    assert reports["the-den"].error_code is None
    assert reports["the-den"].refiner == "stub"
    assert (store.asset_dir("cosy-den", "the-den") / REFINED_FILE).exists()


def test_the_stand_in_ignores_the_target_and_the_report_says_so(setup):
    """Locked decision 9, seen from the stage: the receipt is honest about it."""
    store, direction = setup
    stamp_target(store, "wooden-chair", (2, 2))
    report = refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())
    assert (report.width, report.height) == (8, 8)
    assert report.error_code is None
    assert report.route == "preserve"
    assert report.refiner == "stub"


class SizedRefiner(StubRefiner):
    """A stand-in that claims to honour a forced size, and writes a picture of
    one chosen size whatever it is asked for, remembering what it was sent."""

    honours_forced_size = True

    def __init__(self, size):
        self.size = size
        self.requests: list[RefineRequest] = []

    def refine(self, request: RefineRequest) -> RefineResult:
        self.requests.append(request)
        result = super().refine(request)
        with Image.open(request.output_path) as picture:
            picture.resize(self.size, Image.Resampling.NEAREST).save(request.output_path)
        result.output.width, result.output.height = self.size
        return result


def test_a_result_at_the_wrong_size_is_not_a_success(setup):
    """Catches: a refiner that returns another size than the forced one being
    reported as refined and exported, with its refined.png left on disk. One
    such asset costs only itself; asked for by name, the receipt comes back."""
    store, direction = setup
    for name, target in (("wooden-chair", (4, 4)), ("round-rug", (5, 5))):
        stamp_target(store, name, target)
    refiner = SizedRefiner((5, 5))

    reports = {r.asset: r for r in refine_direction(store, direction, refiner=refiner)}

    chair = store.asset_dir("cosy-den", "wooden-chair")
    assert reports["wooden-chair"].error_code == WRONG_SIZE
    assert not (chair / REFINED_FILE).exists()
    on_disk = load_report(store, "cosy-den", "wooden-chair")
    assert on_disk.error_code == WRONG_SIZE
    assert "wooden-chair" in on_disk.error_message
    assert "5 x 5" in on_disk.error_message and "4 x 4" in on_disk.error_message
    assert reports["round-rug"].error_code is None
    assert (store.asset_dir("cosy-den", "round-rug") / REFINED_FILE).exists()

    # reductionMode "none" turns the own-palette recipe off, so this takes the
    # plain single `refine` call, whose size check is what is under test here.
    by_name = refine_asset(
        store, direction, "wooden-chair", refiner=refiner,
        override=RefinerSettings(quick={"reductionMode": "none"}),
    )
    assert by_name.error_code == WRONG_SIZE
    assert not (chair / REFINED_FILE).exists()
    assert load_report(store, "cosy-den", "wooden-chair").error_code == WRONG_SIZE


@pytest.mark.parametrize("by_name", [False, True], ids=["direction", "by name"])
def test_saved_settings_that_would_change_the_forced_size_are_not_sent(setup, by_name):
    """Catches: a direction's saved scale / makeSquare / keepAspectRatio /
    outlineStyle reaching the refiner alongside a forced size and quietly
    winning, so the scene comes back at a size nobody asked for. The receipt
    must say it left them out and record what was actually sent."""
    store, direction = setup
    stamp_target(store, "wooden-chair", (5, 5))
    direction.refiner_defaults = RefinerSettings(
        scale=2,
        quick={"dithering": "off"},
        advanced={
            "makeSquare": True,
            "keepAspectRatio": True,
            "outlineStyle": "sharp",
            "cleanup": True,
        },
    )
    refiner = SizedRefiner((5, 5))

    if by_name:
        # reductionMode "none" turns the own-palette recipe off, so this takes
        # the plain single `refine` call (where the dropped settings are applied).
        report = refine_asset(
            store, direction, "wooden-chair", refiner=refiner,
            override=RefinerSettings(quick={"reductionMode": "none"}),
        )
    else:
        [report] = [
            r for r in refine_direction(store, direction, refiner=refiner)
            if r.asset == "wooden-chair"
        ]

    [request] = [r for r in refiner.requests if r.input_path.parent.name == "wooden-chair"]
    assert request.target == (5, 5)
    assert request.settings.scale is None
    assert request.settings.advanced.get("cleanup") is True
    for key in ("makeSquare", "keepAspectRatio", "outlineStyle"):
        assert key not in request.settings.advanced
    assert request.settings.quick["dithering"] == "off"
    on_disk = load_report(store, "cosy-den", "wooden-chair")
    assert on_disk.error_code is None
    assert SIZE_SETTING_DROPPED in on_disk.warnings
    assert on_disk.requested.scale is None
    assert "makeSquare" not in on_disk.requested.advanced


class CohesionReportingRefiner(StubRefiner):
    """A stand-in that claims a chosen `effectiveOptions` for cohesion, so a
    test can prove the record is read off it -- not off what was asked."""

    def __init__(self, effective: dict[str, object]):
        self._effective = effective

    def refine(self, request):
        result = super().refine(request)
        result.effective_options = dict(self._effective)
        return result


def test_the_cohesion_record_is_read_off_what_the_refiner_says_ran(setup):
    store, direction = setup
    refiner = CohesionReportingRefiner(
        {"cohesion": True, "cohesionFamilies": 14, "cohesionTones": 2, "cleanup": True}
    )
    report = refine_asset(store, direction, "wooden-chair", refiner=refiner)
    assert report.cohesion.cohesion is True
    assert report.cohesion.cohesion_families == 14
    assert report.cohesion.cohesion_tones == 2
    assert report.cohesion.cleanup is True
    assert load_report(store, "cosy-den", "wooden-chair").cohesion == report.cohesion


def test_the_stub_never_claims_cohesion_or_cleanup_ran_even_when_requested(setup):
    """The house baseline (Task 1) always requests cohesion and clean-up on;
    the stand-in changes no pixels, so its own record must say neither ran.
    This is the one test that proves both halves of "Defaults and the
    record" at once: what was asked for (`report.requested`, on) and what a
    non-cohesive refiner honestly reports back (`report.cohesion`, off) can
    disagree, and the record follows the report -- not the request."""
    store, direction = setup
    report = refine_asset(store, direction, "wooden-chair", refiner=StubRefiner())
    assert report.requested.advanced["cohesion"] is True
    assert report.requested.advanced["cleanup"] is True
    assert report.cohesion == CohesionRecord()
    assert load_report(store, "cosy-den", "wooden-chair").cohesion == CohesionRecord()


def test_an_old_refine_json_with_no_cohesion_key_still_loads():
    """A `refine.json` written before this ticket has never heard of
    `cohesion` at all; `Field(default_factory=CohesionRecord)` must fill it
    with the truthful "did not run" answer rather than refuse to parse. No
    existing test loads a raw pre-M4 `RefineReport` sidecar off disk (unlike
    `ManifestEntry`, which `test_an_m2_decomposition_direction_and_manifest_
    still_parse` in `tests/test_model.py` already exercises through
    `M2_MANIFEST`, itself missing the `cohesion` key), so this is that check
    for `RefineReport`."""
    old = '{"asset": "wooden-chair", "refiner": "stub", "requested": {}}'
    report = RefineReport.model_validate_json(old)
    assert report.cohesion == CohesionRecord()


def test_pixelmagic_s_own_warning_codes_never_shadow_the_refiner_s():
    """`SEAM_VISIBLE` is added by PixelMagic after the refiner ran.

    It must be a member of our own vocabulary, and that vocabulary must not
    overlap the refiner's: the adapter tests prove the CLI never invents a
    code by checking against `WARNING_CODES`, and a code of ours inside that
    set would make that proof silently weaker.
    """
    assert SEAM_VISIBLE in PIXELMAGIC_WARNING_CODES
    assert PIXELMAGIC_WARNING_CODES.isdisjoint(WARNING_CODES)


#: Every kind of piece whose road is gone, as (kind, extra `AssetRecord` fields): a bordered
#: tile, a plain-patch tile, a sheet-road prop, and a prop kept from an object-road redraw
#: made before the grid rebuild existed.
OLD_ITEMS = {
    "floor-centre": (AssetKind.TILE, {"role": TileRole.CENTRE}),
    "mat-r0c0": (AssetKind.TILE, {"position": (0, 0)}),
    "sheet-lamp": (AssetKind.PROP, {"pitch": 4.0, "footprint": (1, 1), "anchor": (2, 4)}),
    "picked-lamp": (AssetKind.PROP, {"segmenter": "cli", "decomposition": "20260930-100000"}),
}


def add_old_item(store, name):
    """Stage one 4 x 4 piece from `OLD_ITEMS`: not a recipe tile, not a rebuilt prop."""
    kind, extra = OLD_ITEMS[name]
    folder = store.asset_dir("cosy-den", name)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 4), 255, dtype=np.uint8), "RGBA").save(folder / "source.png")
    store.write_model(
        folder / "asset.json",
        AssetRecord(
            name=name, kind=kind, source_generation="20260903-191233",
            source_prompt="a sheet", service="fake", target=(4, 4), **extra,
        ),
    )


def test_an_old_item_never_refined_is_refused_and_the_rest_still_refine(setup):
    """Catches: a tile or prop whose road is gone (bordered, plain-patch, sheet-road, or
    kept before the grid rebuild) being sent to the refiner and coming back looking
    refined, one such piece stopping the whole direction, or its refusal sentence sending a
    person to the wrong fix."""
    store, direction = setup
    for name in OLD_ITEMS:
        add_old_item(store, name)

    reports = {r.asset: r for r in refine_direction(store, direction, refiner=StubRefiner())}

    for name, (kind, _extra) in OLD_ITEMS.items():
        assert reports[name].error_code == OLD_ITEM_REFUSED
        assert name in reports[name].error_message
        assert not (store.asset_dir("cosy-den", name) / REFINED_FILE).exists()
        way_forward = (
            "type the surface again as a recipe surface"
            if kind is AssetKind.TILE
            else "pick and redraw the object again"
        )
        assert way_forward in reports[name].error_message
        with pytest.raises(ValueError, match=f"{name}.*{way_forward}"):
            refine_asset(store, direction, name, refiner=StubRefiner())
    assert OLD_ITEM_REFUSED in PIXELMAGIC_REFUSAL_CODES
    for name in ("wooden-chair", "round-rug"):
        assert reports[name].error_code is None
        assert (store.asset_dir("cosy-den", name) / REFINED_FILE).exists()


@pytest.mark.parametrize("name", list(OLD_ITEMS))
def test_an_old_item_already_refined_is_left_exactly_as_it_is(setup, name):
    """Catches: refining a direction wiping the picture of an old tile or prop that was
    refined before its road went, which export still ships, or counting it as refined now."""
    store, direction = setup
    add_old_item(store, name)
    folder = store.asset_dir("cosy-den", name)
    Image.fromarray(np.full((4, 4, 4), 90, dtype=np.uint8), "RGBA").save(folder / REFINED_FILE)
    store.write_model(
        folder / REPORT_FILE,
        RefineReport(
            asset=name, refiner="pixel-refiner", route="refine", warnings=[LOW_GRID_CONFIDENCE]
        ),
    )
    before = ((folder / REFINED_FILE).read_bytes(), (folder / REPORT_FILE).read_bytes())

    reports = {r.asset: r for r in refine_direction(store, direction, refiner=StubRefiner())}

    assert reports[name].error_code is None
    assert reports[name].route == OLD_ITEM_ROUTE  # so no face counts it as refined
    assert ((folder / REFINED_FILE).read_bytes(), (folder / REPORT_FILE).read_bytes()) == before


def add_scene(store, name, *, width, height, pitch=1.0):
    """Stage a kept scene the way `file_promotion` leaves one whose own
    `analyze` reading failed: pitch 1.0, so the target is the whole picture.
    That is precisely what makes a scene over 1024 px on a side unusable
    until `set_scene_pitch` gives it a real one (ticket #34, spec §3(c))."""
    folder = store.asset_dir("cosy-den", name)
    folder.mkdir(parents=True, exist_ok=True)
    pixels = np.full((height, width, 4), 255, dtype=np.uint8)
    Image.fromarray(pixels, "RGBA").save(folder / "source.png")
    store.write_model(
        folder / "asset.json",
        AssetRecord(
            name=name, kind=AssetKind.SCENE, source_generation="20260903-191233",
            source_prompt="a cosy den", service="fake",
            pitch=pitch, pitch_source=PitchSource.MEASURED, target=(width, height),
        ),
    )


def test_a_grid_candidate_on_the_kept_scene_is_refused_before_anything_is_touched(
    setup, candidate_refiner
):
    """Catches: the scene being sent to the refiner with a candidate, which the real
    refiner lets win over the forced size, so a slow call ends in `WRONG_SIZE` and
    the person's refined scene is thrown away for nothing."""
    store, direction = setup
    add_scene(store, "the-den", width=40, height=20)
    folder = store.asset_dir("cosy-den", "the-den")
    (folder / "refined.png").write_bytes(b"earlier result")

    with pytest.raises(ValueError, match="the-den.*Set the scene's pitch"):
        refine_asset(
            store, direction, "the-den", refiner=candidate_refiner, candidate_id="grid:1x1"
        )

    assert candidate_refiner.asked == []
    assert not (folder / "refine.json").exists()
    assert (folder / "refined.png").read_bytes() == b"earlier result"


def test_refine_direction_refines_each_asset_on_its_own(setup):
    """The default call never lets two assets share a palette — each gets its own
    batch-of-one call so its own colours are fit to it alone (spec "Palettes")."""
    store, direction = setup  # two bare folders
    add_scene(store, "the-den", width=40, height=20)

    refiner = BatchRecordingRefiner()
    reports = refine_direction(store, direction, refiner=refiner)

    assert len(reports) == 3
    batches = [frozenset(names) for names in refiner.batches]
    assert frozenset({"wooden-chair"}) in batches
    assert frozenset({"round-rug"}) in batches
    assert frozenset({"the-den"}) in batches
    assert len(batches) == 3  # three solo calls


class PaletteRecordingRefiner(StubRefiner):
    """Records every `fit_palette` and `batch` call it is given, and answers
    a made-up palette that is easy to tell apart from anything `_palette()`
    would measure off the source itself -- what proves the final refine was
    pinned to what the fit actually returned, not to a stand-in coincidence.
    """

    def __init__(self):
        self.palette_calls: list[tuple] = []
        self.batch_calls: list[list] = []

    def fit_palette(self, input_path, count, settings=None):
        self.palette_calls.append((input_path, count, settings))
        return [f"#{index:02x}0000" for index in range(count)]

    def batch(self, requests):
        self.batch_calls.append(list(requests))
        return super().batch(requests)


def test_the_scene_is_refined_with_the_two_step_own_palette_recipe(setup):
    """`HOUSE_BASELINE`'s `reductionMode: "24"` is a colour count, so the
    default call must fit the scene's own palette on its untouched source
    -- cohesion and clean-up off, no reductionMode -- before the pinned
    refine that follows asks for cohesion and exactly the colours the fit
    returned (M4.1 acceptance recipe, 2026-09-25). No `background` flag goes
    on either call: the scene's `source.png` is the generator's raw frame and
    needs the refiner's ordinary background removal (sending `keep` once left a
    den export with an opaque surround)."""
    store, direction = setup
    add_asset(store, "the-den")
    stamp_target(store, "the-den", (8, 8))
    refiner = PaletteRecordingRefiner()

    refine_direction(store, direction, refiner=refiner)

    [(input_path, count, palette_settings)] = [
        call for call in refiner.palette_calls
        if call[0].parent.name == "the-den"
    ]
    assert input_path == store.asset_dir("cosy-den", "the-den") / "source.png"
    assert "background" not in palette_settings.quick
    assert count == 24
    assert "reductionMode" not in palette_settings.quick
    assert palette_settings.advanced["cohesion"] is False
    assert palette_settings.advanced["cleanup"] is False

    [wooden_chair_batch] = [
        requests for requests in refiner.batch_calls
        if requests[0].input_path.parent.name == "the-den"
    ]
    [request] = wooden_chair_batch
    fitted = [f"#{index:02x}0000" for index in range(24)]
    assert request.settings.advanced["reduceColorMode"] == "fixed"
    assert request.settings.advanced["fixedPalette"] == fitted
    assert request.settings.advanced["cohesion"] is True
    assert "background" not in request.settings.quick
    assert "reductionMode" not in request.settings.quick


class RefusingPaletteRefiner(StubRefiner):
    """Refuses the palette-fitting pass for one named asset only, and fits
    normally for any other -- proving one bad palette fit costs only its
    own asset, not the direction."""

    def __init__(self, refuse: str):
        self._refuse = refuse

    def fit_palette(self, input_path, count, settings=None):
        if input_path.parent.name == self._refuse:
            return RefineFailure(
                input_path=input_path, code="UNSUPPORTED_INPUT", message="too weird"
            )
        return super().fit_palette(input_path, count, settings)


def test_a_palette_pass_refusal_costs_only_that_asset(setup):
    store, direction = setup
    stamp_target(store, "wooden-chair", (8, 8))
    stamp_target(store, "round-rug", (8, 8))
    refiner = RefusingPaletteRefiner("wooden-chair")

    reports = refine_direction(store, direction, refiner=refiner)
    by_name = {report.asset: report for report in reports}

    assert by_name["wooden-chair"].error_code == "UNSUPPORTED_INPUT"
    assert by_name["wooden-chair"].error_message == "too weird"
    assert not (store.asset_dir("cosy-den", "wooden-chair") / REFINED_FILE).exists()
    assert by_name["round-rug"].error_code is None
    assert (store.asset_dir("cosy-den", "round-rug") / REFINED_FILE).exists()


def test_refine_asset_on_the_scene_runs_the_palette_pass_then_the_pinned_call(setup):
    """`refine_asset`, asked for the scene by name, must run the identical M4.2
    two-step recipe `refine_direction` gives it by default (ticket
    #70 review round 1) -- not the single `reductionMode` call M4.1 measured
    as failing (91% different)."""
    store, direction = setup
    add_asset(store, "the-den")
    stamp_target(store, "the-den", (8, 8))
    refiner = PaletteRecordingRefiner()

    report = refine_asset(store, direction, "the-den", refiner=refiner)

    [(input_path, count, palette_settings)] = refiner.palette_calls
    assert input_path == store.asset_dir("cosy-den", "the-den") / "source.png"
    assert count == 24
    assert palette_settings.advanced["cohesion"] is False

    [request] = refiner.batch_calls[0]
    fitted = [f"#{index:02x}0000" for index in range(24)]
    assert request.settings.advanced["fixedPalette"] == fitted
    assert "background" not in request.settings.quick
    assert report.requested.advanced["fixedPalette"] == fitted


def test_a_palette_pass_refusal_on_refine_asset_raises_naming_the_asset(setup):
    """`refine_asset`'s existing contract for a refiner refusal is to raise,
    not to write a receipt (see `test_an_out_of_range_target_names_the_asset_
    and_does_not_stop_the_direction`'s sibling case for the plain path). A
    palette-pass refusal reached through the two-step recipe has to raise the
    same way, naming the asset -- `fit_palette`'s own failure only names the
    file -- rather than silently falling back to the receipt
    `refine_direction` writes for the same failure."""
    store, direction = setup
    stamp_target(store, "wooden-chair", (8, 8))
    refiner = RefusingPaletteRefiner("wooden-chair")

    with pytest.raises(RefinerError, match="wooden-chair") as raised:
        refine_asset(store, direction, "wooden-chair", refiner=refiner)

    assert raised.value.code == "UNSUPPORTED_INPUT"
    assert not (store.asset_dir("cosy-den", "wooden-chair") / REPORT_FILE).exists()


class RefusingFinalCallRefiner(StubRefiner):
    """Fits the palette normally, then refuses the final pinned call."""

    def batch(self, requests):
        return [
            RefineFailure(
                input_path=request.input_path, code="INVALID_SETTINGS", message="bad palette"
            )
            for request in requests
        ]


def test_a_final_call_refusal_on_refine_asset_also_raises_naming_the_asset(setup):
    """The two-step recipe's *second* call is a refiner refusal too, and
    `refine_asset` must treat it exactly like the palette-pass refusal above
    (ticket #70 review round 2): raise, naming the asset, write no receipt,
    and leave whatever `refined.png` was already there untouched --
    `refine_direction` keeps writing a receipt for this same failure."""
    store, direction = setup
    stamp_target(store, "wooden-chair", (8, 8))
    refiner = RefusingFinalCallRefiner()

    with pytest.raises(RefinerError, match="wooden-chair") as raised:
        refine_asset(store, direction, "wooden-chair", refiner=refiner)

    assert raised.value.code == "INVALID_SETTINGS"
    assert not (store.asset_dir("cosy-den", "wooden-chair") / REPORT_FILE).exists()
    assert not (store.asset_dir("cosy-den", "wooden-chair") / REFINED_FILE).exists()

    # `refine_direction`, refining a whole direction, still answers the same
    # refusal with a receipt rather than stopping the run.
    reports = refine_direction(store, direction, refiner=refiner)
    assert {r.asset: r.error_code for r in reports}["wooden-chair"] == "INVALID_SETTINGS"


def test_set_scene_pitch_lifts_the_target_not_usable_refusal(setup):
    """Ticket #34 / the 2026-09-23 amendment to spec §3(c): a kept scene over
    1024 px on a side whose pitch could not be read is refused with
    `TARGET_NOT_USABLE` and never reaches the refiner. Once `set_scene_pitch`
    gives it a real pitch, its target shrinks under the cap and
    `refine_direction` refines it with no failure code, asking the refiner
    for exactly the target `set_scene_pitch` computed. (The first refiner
    honours a forced size, like the real one, so the limit applies; the stand-in
    has no such limit. The second is a plain recording stand-in, since this
    test is about the target, not the size check.)"""
    store, direction = setup
    add_scene(store, "the-den", width=1100, height=100)
    refiner = ForcedSizeRefiner()
    before = {r.asset: r for r in refine_direction(store, direction, refiner=refiner)}
    assert before["the-den"].error_code == TARGET_NOT_USABLE
    # Only the two bare folders reached the refiner, and with no forced size.
    assert refiner.targets == [None, None]

    updated = set_scene_pitch(store, "cosy-den", "the-den", 2.0)
    assert updated.target == (550, 50)
    assert updated.pitch == 2.0
    assert updated.pitch_source is PitchSource.SET

    refiner = TargetRecordingRefiner()
    after = {r.asset: r for r in refine_direction(store, direction, refiner=refiner)}
    assert after["the-den"].error_code is None
    assert (550, 50) in refiner.targets


def test_set_scene_pitch_refuses_a_pitch_whose_target_is_too_big_and_writes_nothing(setup):
    """Catches: a small pitch on a big scene being saved, then every refine of
    the scene failing with `TARGET_NOT_USABLE` though the person just set a pitch."""
    store, direction = setup
    add_scene(store, "the-den", width=1100, height=100)
    path = store.asset_dir("cosy-den", "the-den") / "asset.json"
    before = store.read_model(path, AssetRecord)
    with pytest.raises(ValueError, match="1100 x 100"):
        set_scene_pitch(store, "cosy-den", "the-den", 1.0)
    assert store.read_model(path, AssetRecord) == before


@pytest.mark.parametrize("kind", [AssetKind.PROP, AssetKind.TILE])
def test_set_scene_pitch_refuses_a_non_scene_and_writes_nothing(setup, kind):
    """A rebuilt prop's pitch is its own, and a tile has no pitch at all —
    `set_scene_pitch` refuses either, writing nothing."""
    store, direction = setup
    stamp_target(store, "wooden-chair", (17, 32), kind=kind)
    before = store.read_model(
        store.asset_dir("cosy-den", "wooden-chair") / "asset.json", AssetRecord
    )
    with pytest.raises(ValueError, match="only a scene's pitch"):
        set_scene_pitch(store, "cosy-den", "wooden-chair", 2.0)
    after = store.read_model(
        store.asset_dir("cosy-den", "wooden-chair") / "asset.json", AssetRecord
    )
    assert after == before


@pytest.mark.parametrize("value", [0.1, 100.0])
def test_set_scene_pitch_refuses_an_out_of_range_value_and_writes_nothing(setup, value):
    store, direction = setup
    add_scene(store, "the-den", width=200, height=100)
    before = store.read_model(
        store.asset_dir("cosy-den", "the-den") / "asset.json", AssetRecord
    )
    with pytest.raises(ValueError, match="pitch"):
        set_scene_pitch(store, "cosy-den", "the-den", value)
    after = store.read_model(
        store.asset_dir("cosy-den", "the-den") / "asset.json", AssetRecord
    )
    assert after == before
