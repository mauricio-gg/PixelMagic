import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from pixelmagic.approval import AutoApprove
from pixelmagic.cli import app
from pixelmagic.model import (
    AssetRecord,
    PitchSource,
    RefinerSettings,
)
from pixelmagic.providers.base import ImageCapabilities, ServiceError
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.refiner.base import LOW_GRID_CONFIDENCE, RefineFailure, RefinerError
from pixelmagic.refiner.cli_adapter import BINARY_ENV
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.services import NO_SERVICE_SENTENCE
from pixelmagic.stages.decompose import load_decomposition
from pixelmagic.stages.layout import default_anchor
from pixelmagic.stages.refine import load_report
from pixelmagic.store import Store
from test_recipes import floor_and_block_scene

runner = CliRunner()


def printed(result) -> str:
    """Everything the command printed, wherever this click version put it.

    Some click versions keep stderr separate from stdout and some do not; a
    test that only reads `result.output` passes or fails on that difference
    rather than on the behaviour it is checking. The deliberate cost: no test
    that uses this helper says which stream the text came out of, so nothing
    here catches the loss of `err=True` in `_refiner_failed`.
    """
    try:
        return result.output + (result.stderr or "")
    except ValueError:
        return result.output


@pytest.fixture(autouse=True)
def no_aseprite(monkeypatch):
    """Keep the CLI tests off any Aseprite the machine happens to have."""
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None)


def invoke(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result


#: A wood floor typed the recipe way, on the shared synthetic scene's floor.
FLOOR_SURFACE = "wood floor:wood; box 10,10 50,50; shape 0,0 100,0 100,88 0,88"


def import_floor(tmp_path):
    """Import the shared synthetic scene (a floor beside a block); hand back its id."""
    path = tmp_path / "den.png"
    Image.fromarray(floor_and_block_scene().picture, "RGBA").save(path)
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "import", str(path))
    return Store(tmp_path).list_generations("cosy-den")[0]


def new_project(tmp_path):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    invoke("--root", str(tmp_path), "project", "new", "DemoGame")


def new_direction(tmp_path, name="cosy-den"):
    invoke(
        "--root", str(tmp_path), "direction", "new",
        "--name", name, "--style", "Warm SNES interiors.", "--dont", "no gradients",
    )


def cut_by_picking(tmp_path, object_picture):
    """Give the cosy-den direction a mix of assets, all by the command line.

    Two assets the refiner works on (the imported picture kept as the scene,
    `objects`, and a drawn picture kept whole, `cosy-den`) and three rebuilt
    props, made by suggesting with the stand-in segmenter, picking objects 1 to
    3 and redrawing them with the fake service. Rebuilt props never reach the
    refiner, so the tests that look at what the refiner did need the other two.
    """
    root = ["--root", str(tmp_path), "--direction", "cosy-den"]
    invoke(*root, "import", str(object_picture))
    generation_id = Store(tmp_path).list_generations("cosy-den")[0]
    invoke(*root, "keep-scene", generation_id, "--refiner", "stub")
    invoke(
        *root, "--service", "fake", "generate", "a cosy den", "--size", "96",
        "--auto-approve", "--as-asset", "--refiner", "stub",
    )
    invoke(*root, "suggest", generation_id, "--segmenter", "stub")
    decomposition_id = Store(tmp_path).list_decompositions("cosy-den")[0]
    invoke(*root, "pick", decomposition_id, "1", "2", "3")
    invoke(*root, "--service", "fake", "redraw", decomposition_id)


def test_project_new_creates_the_sidecar(tmp_path):
    new_project(tmp_path)
    assert (tmp_path / "pixelmagic.json").exists()
    assert Store(tmp_path).load_project().name == "DemoGame"


def test_project_list_names_what_it_finds(tmp_path):
    new_project(tmp_path)
    assert "DemoGame" in invoke("--root", str(tmp_path), "project", "list").output


def test_direction_new_with_a_name_creates_it(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    direction = Store(tmp_path).load_direction("cosy-den")
    assert direction.style == "Warm SNES interiors."
    assert direction.donts == ["no gradients"]


def test_direction_new_without_a_name_asks_the_service(tmp_path):
    new_project(tmp_path)
    invoke(
        "--root", str(tmp_path), "--service", "fake",
        "direction", "new", "--style", "Warm SNES interiors.",
    )
    assert Store(tmp_path).list_directions() == ["wooden-chair"]


def test_direction_list_prints_every_direction(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path, "cosy-den")
    new_direction(tmp_path, "cold-cave")
    output = invoke("--root", str(tmp_path), "direction", "list").output
    assert "cosy-den" in output and "cold-cave" in output


def test_generate_with_auto_approve_writes_a_generation(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    output = invoke(
        "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
        "generate", "a cosy den", "--kind", "scene", "--size", "96", "--auto-approve",
    ).output
    generations = list((tmp_path / "directions" / "cosy-den" / "generations").iterdir())
    assert len(generations) == 1
    assert generations[0].name in output


def test_generate_with_no_service_chosen_is_refused_and_draws_nothing(tmp_path):
    """Catches: a draw that falls back to the stand-in (or a paid service) when nobody
    chose one, instead of stopping with the sentence that says where to choose."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "generate", "a cosy den",
         "--auto-approve"],
    )
    assert result.exit_code == 2
    # The usage error is drawn in a box that wraps the sentence, so check its opening.
    assert NO_SERVICE_SENTENCE.split(".")[0] in printed(result)
    assert "Drawing with" not in printed(result)
    assert Store(tmp_path).list_generations("cosy-den") == []


def test_generate_with_wait_asks_the_terminal(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        [
            "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
            "generate", "a cosy den", "--size", "96", "--wait",
        ],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "keep it?" in result.output


def test_an_unknown_service_lists_the_ones_that_exist(tmp_path):
    """Two different problems get two different sentences."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "nonesuch", "--direction", "cosy-den",
         "generate", "a cosy den", "--auto-approve"],
    )
    assert result.exit_code == 2
    assert "unknown service" in printed(result)
    assert "fake" in printed(result)


def test_generate_can_keep_the_picture_as_an_asset(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    output = invoke(
        "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
        "generate", "a cosy den", "--size", "96", "--auto-approve", "--as-asset",
        "--refiner", "stub",
    ).output
    assert "kept the whole picture as asset cosy-den" in output
    assert Store(tmp_path).list_assets("cosy-den") == ["cosy-den"]


def test_generate_names_the_approved_picture_when_the_refiner_wont_keep_it_as_an_asset(
    tmp_path, monkeypatch
):
    """`generate --as-asset` approves the picture before it ever calls the
    refiner (`stages.decompose.promote_generation`'s own docstring says so),
    so a refiner refusal here must not read like an ordinary
    `_refiner_failed` — the picture is not lost, just not kept, and the
    message has to name the generation id so a person knows which one."""
    new_project(tmp_path)
    new_direction(tmp_path)

    class RefusingRefiner(StubRefiner):
        def analyze(self, input_path, settings=None):
            raise RefinerError(
                "the refiner would not read it", exit_code=1, code="ENGINE_FAILURE"
            )

    monkeypatch.setattr("pixelmagic.cli.load_refiner", lambda name: RefusingRefiner())
    result = runner.invoke(
        app,
        [
            "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
            "generate", "a cosy den", "--size", "96", "--auto-approve", "--as-asset",
            "--refiner", "stub",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    output = printed(result)
    assert "approved" in output
    assert "not kept as an asset" in output
    assert "the refiner would not read it" in output
    generation_id = Store(tmp_path).list_generations("cosy-den")[0]
    assert generation_id in output
    assert Store(tmp_path).list_assets("cosy-den") == []


def test_refine_and_export_can_be_run_one_at_a_time_after_a_cut(tmp_path, object_picture):
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "stub")
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "export", "--scale", "1")

    exports = list((tmp_path / "exports").iterdir())
    manifest = json.loads((exports[0] / "manifest.json").read_text())
    assert len(manifest["entries"]) == 5  # two kept pictures and three rebuilt props


def test_decompose_with_no_surface_names_both_ways_forward_and_writes_nothing(tmp_path):
    """A person who types no surface must be told both ways forward: suggest, pick and redraw
    with the real generation id for props, or --surface for tiles; and nothing is written."""
    new_project(tmp_path)
    new_direction(tmp_path)
    generation_id = invoke(
        "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
        "generate", "a cosy den", "--kind", "scene", "--size", "96", "--auto-approve",
    ).output.split()[-1].strip()

    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "decompose", generation_id],
    )

    assert result.exit_code == 1
    output = printed(result)
    assert f"pixelmagic suggest {generation_id}" in output
    assert "pixelmagic pick" in output
    assert "pixelmagic redraw" in output
    assert "--surface" in output
    assert Store(tmp_path).list_decompositions("cosy-den") == []


def test_picked_props_through_the_command_line_and_redraw_before_any_pick_is_refused(
    tmp_path, object_picture
):
    """import, suggest, pick and redraw chained the way the README says, by their option names."""
    new_project(tmp_path)
    new_direction(tmp_path)
    root = ["--root", str(tmp_path), "--direction", "cosy-den"]
    imported = invoke(*root, "import", str(object_picture)).output
    generation_id = imported.split()[3]
    assert f"pixelmagic suggest {generation_id}" in imported

    suggested = invoke(*root, "suggest", generation_id, "--segmenter", "stub").output
    assert "8 shown" in suggested
    decomposition_id = Store(tmp_path).list_decompositions("cosy-den")[0]

    early = runner.invoke(app, [*root, "--service", "fake", "redraw", decomposition_id])
    assert early.exit_code == 1
    assert "nothing is picked yet" in printed(early)

    invoke(*root, "pick", decomposition_id, "1", "2", "3")
    redrawn = invoke(*root, "--service", "fake", "redraw", decomposition_id).output
    assert "3 piece(s)" in redrawn
    assert len(Store(tmp_path).list_assets("cosy-den")) == 3


def test_refine_reports_assets_that_need_a_look(tmp_path, monkeypatch, object_picture):
    """The LOW_GRID_CONFIDENCE branch, driven through the real command."""
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)

    class UncertainRefiner(StubRefiner):
        def refine(self, request):
            result = super().refine(request)
            result.warnings = [LOW_GRID_CONFIDENCE]
            return result

    monkeypatch.setattr("pixelmagic.cli.load_refiner", lambda name: UncertainRefiner())
    output = invoke("--root", str(tmp_path), "--direction", "cosy-den", "refine").output
    assert "pixel grid is uncertain" in output


def test_refining_an_asset_that_does_not_exist_says_so(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den",
         "refine", "--asset", "no-such-thing", "--refiner", "stub"],
    )
    assert result.exit_code != 0
    assert "no-such-thing" in printed(result)


def test_a_refiner_that_will_not_run_is_a_message_not_a_traceback(
    tmp_path, monkeypatch, object_picture
):
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    monkeypatch.setenv(BINARY_ENV, "/nowhere/pixel-refiner")
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "cli"],
    )
    assert result.exit_code == 1
    # A clean exit, not a RefinerError escaping into a stack trace.
    assert isinstance(result.exception, SystemExit), result.exception
    assert "is not installed" in printed(result)


def test_refine_with_no_refiner_flag_uses_the_stand_in_and_says_so(tmp_path, object_picture):
    """The real PixelRefiner is opt-in: a stranger's bare `refine` must not
    try to run a tool they do not have, and the output names what ran."""
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    assert "with stub" in invoke(
        "--root", str(tmp_path), "--direction", "cosy-den", "refine"
    ).output


def test_a_per_asset_override_leaves_the_direction_defaults_alone(tmp_path, object_picture):
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    invoke(
        "--root", str(tmp_path), "--direction", "cosy-den",
        "refine", "--preset", "crisp-sprite", "--scale", "2", "--refiner", "stub",
    )
    asset = Store(tmp_path).list_assets("cosy-den")[0]
    invoke(
        "--root", str(tmp_path), "--direction", "cosy-den",
        "refine", "--asset", asset, "--preset", "retro-game", "--refiner", "stub",
    )
    direction = Store(tmp_path).load_direction("cosy-den")
    assert direction.refiner_defaults.preset == "crisp-sprite"
    assert direction.refiner_defaults.scale == 2
    report = json.loads((Store(tmp_path).asset_dir("cosy-den", asset) / "refine.json").read_text())
    assert report["requested"]["preset"] == "retro-game"


def test_setting_a_preset_does_not_wipe_a_stored_scale(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    invoke(
        "--root", str(tmp_path), "--direction", "cosy-den", "refine", "--scale", "4",
        "--refiner", "stub",
    )
    invoke(
        "--root", str(tmp_path), "--direction", "cosy-den", "refine", "--preset", "retro-game",
        "--refiner", "stub",
    )
    defaults = Store(tmp_path).load_direction("cosy-den").refiner_defaults
    assert defaults.scale == 4
    assert defaults.preset == "retro-game"


def test_export_warns_when_aseprite_is_missing(tmp_path, object_picture):
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    output = invoke("--root", str(tmp_path), "--direction", "cosy-den", "export").output
    assert "Aseprite" in output


def test_export_repeats_a_low_grid_confidence_warning(tmp_path, object_picture):
    """The CLI already prints bundle warnings; this is the one that was missing."""
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "stub")
    store = Store(tmp_path)
    name = store.list_assets("cosy-den")[0]
    report = load_report(store, "cosy-den", name)
    report.warnings = [LOW_GRID_CONFIDENCE]
    store.write_model(store.asset_dir("cosy-den", name) / "refine.json", report)

    output = invoke(
        "--root", str(tmp_path), "--direction", "cosy-den", "export"
    ).output

    assert name in output and "grid" in output


# --- the controller's rulings ---------------------------------------------


def test_a_run_nobody_approves_is_a_message_not_a_traceback(tmp_path):
    """Ruling 1: `ApprovalExhausted` names the last attempt and exits non-zero."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "generate", "a cosy den", "--size", "96", "--wait", "--rounds", "1"],
        input="n\nmake it warmer\n",
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.exception
    generations = list((tmp_path / "directions" / "cosy-den" / "generations").iterdir())
    assert generations[0].name in printed(result)
    assert "rejected" in printed(result)


def test_walking_away_from_the_prompt_is_not_a_traceback(tmp_path):
    """Ruling 4: Ctrl-D at the approval prompt exits cleanly."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "generate", "a cosy den", "--size", "96", "--wait"],
        input="",
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.exception
    assert "end of input" in printed(result)


def test_settings_the_refiner_refuses_are_a_message_not_a_traceback(tmp_path, object_picture):
    """Ruling 2: a `ValueError` out of the refine stage is a clean exit."""
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den",
         "refine", "--quick", '{"background": "pick"}', "--refiner", "stub"],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "backgroundColor" in printed(result)


def test_settings_the_refiner_refuses_never_become_the_defaults(tmp_path, object_picture):
    """A refused settings block would otherwise refuse every later run too."""
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    # Read before the refused call, rather than pin the exact shape a new
    # direction's stored defaults happen to have today (M4's HOUSE_BASELINE
    # seeds a new direction's refiner_defaults -- see stages/refine.py): the
    # thing this test guards is that the refused call changes nothing, not
    # what those defaults are.
    before = Store(tmp_path).load_direction("cosy-den").refiner_defaults
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den",
         "refine", "--quick", '{"background": "pick"}', "--refiner", "stub"],
    )
    assert result.exit_code == 1
    after = Store(tmp_path).load_direction("cosy-den").refiner_defaults
    assert after == before
    assert "background" not in after.quick
    # And the direction still refines, rather than being stuck until someone
    # edits direction.json by hand.
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "stub")


def test_refine_says_when_pixelmagic_itself_did_not_send_an_asset(tmp_path, object_picture):
    """A broken `asset.json` is PixelMagic's refusal, not the refiner's: the
    line names the asset and must not blame the refiner."""
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    (Store(tmp_path).asset_dir("cosy-den", "objects") / "asset.json").write_text("not json")
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "stub"],
    )
    assert result.exit_code == 1
    assert "objects: PixelMagic did not send it to the refiner" in result.output
    assert "the refiner refused it" not in result.output


def test_refine_with_refusals_names_them_on_the_summary_and_exits_non_zero(
    tmp_path, monkeypatch, object_picture
):
    """Ruling 6: a refused asset is visible, not silently missing, and a script sees exit 1."""
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)

    class RefusingRefiner(StubRefiner):
        def batch(self, requests):
            return [
                RefineFailure(
                    input_path=request.input_path,
                    code="UNSUPPORTED_INPUT",
                    message="the refiner would not read it",
                )
                for request in requests
            ]

    monkeypatch.setattr("pixelmagic.cli.load_refiner", lambda name: RefusingRefiner())
    result = runner.invoke(
        app, ["--root", str(tmp_path), "--direction", "cosy-den", "refine"]
    )
    first_line = result.output.splitlines()[0]
    assert result.exit_code == 1
    assert first_line.startswith("refined 0 asset(s)")
    assert first_line.endswith("; 2 refused: cosy-den, objects")
    assert "the refiner refused it (UNSUPPORTED_INPUT)" in result.output


def test_settings_that_are_not_json_are_a_usage_error(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "refine", "--quick", "nonsense",
         "--refiner", "stub"],
    )
    assert result.exit_code != 0
    assert "not valid JSON" in printed(result)


def test_settings_that_are_not_a_json_object_are_a_usage_error(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "refine", "--quick", "[1, 2]",
         "--refiner", "stub"],
    )
    assert result.exit_code != 0
    assert "JSON object" in printed(result)


def test_settings_with_the_wrong_shape_are_a_usage_error(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den",
         "refine", "--quick", '{"detailLevel": 3}', "--refiner", "stub"],
    )
    assert result.exit_code != 0
    assert "detailLevel" in printed(result)


def test_rounds_must_be_at_least_one(tmp_path):
    """Ruling 3: `--rounds 0` is refused at the boundary."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "generate", "a cosy den", "--rounds", "0", "--auto-approve"],
    )
    assert result.exit_code != 0
    assert "rounds" in printed(result)


def test_export_writes_the_default_scales(tmp_path):
    """Ruling 10: the default comes from `DEFAULT_SCALES`, so 1, 2 and 4 land."""
    new_project(tmp_path)
    new_direction(tmp_path)
    invoke(
        "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
        "generate", "a cosy den", "--size", "96", "--auto-approve", "--as-asset",
        "--refiner", "stub",
    )
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "stub")
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "export")
    bundle = next(iter((tmp_path / "exports").iterdir()))
    assert {path.name for path in bundle.glob("cosy-den*.png")} == {
        "cosy-den.png", "cosy-den@2x.png", "cosy-den@4x.png"
    }


def test_export_scale_2_still_writes_the_native_png(tmp_path):
    """Issue #236: the command line used to write only what `--scale` named, so
    `--scale 2` gave a manifest naming a native file that was not there."""
    new_project(tmp_path)
    new_direction(tmp_path)
    invoke(
        "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
        "generate", "a cosy den", "--size", "96", "--auto-approve", "--as-asset",
        "--refiner", "stub",
    )
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "stub")
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "export", "--scale", "2")
    bundle = next(iter((tmp_path / "exports").iterdir()))
    assert {path.name for path in bundle.glob("cosy-den*.png")} == {
        "cosy-den.png", "cosy-den@2x.png"
    }


def test_a_scale_of_zero_is_refused_before_anything_is_written(tmp_path):
    """Ruling 11: Pillow must never be the thing that reports `--scale 0`."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "export", "--scale", "0"],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "exports").exists()
    assert "scale" in printed(result)


def test_export_refuses_a_scale_that_is_too_big(tmp_path):
    """Guardian I1: `--scale` had a floor (`min=1`) but no ceiling, so a
    fat-fingered value reached `stages.export.export` and would have asked
    Pillow for a resize large enough to exhaust memory. The core now refuses
    it, and the CLI must report that refusal as one line, not a traceback."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "export", "--scale", "999"],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "exports").exists()
    output = printed(result)
    assert "Traceback" not in output
    assert "999" in output


def test_a_size_of_zero_is_refused(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "generate", "a cosy den", "--size", "0", "--auto-approve"],
    )
    assert result.exit_code != 0
    assert "size" in printed(result)


def test_a_size_over_the_maximum_is_refused(tmp_path):
    """Guardian M2.2 #1: `--size` had a floor (`min=1`) but no ceiling, so a
    fat-fingered value would have reached the core and drawn a picture Pillow
    cannot even open. `--size` now mirrors `MAX_IMAGE_SIZE` as its own
    `max=`, so Typer's own range check refuses it — exit 2, the same usage
    error `--size 0` gets from its `min=1` — one line, no traceback, and
    nothing is ever drawn."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "generate", "a cosy den", "--size", "99999", "--auto-approve"],
    )
    assert result.exit_code == 2
    output = printed(result)
    assert "Traceback" not in output
    assert "99999" in output
    assert Store(tmp_path).list_generations("cosy-den") == []


def test_a_refiner_scale_of_zero_is_refused(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "refine", "--scale", "0",
         "--refiner", "stub"],
    )
    assert result.exit_code != 0
    assert "scale" in printed(result)


def test_a_direction_that_does_not_exist_says_so(tmp_path):
    new_project(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "nowhere", "export"],
    )
    assert result.exit_code != 0
    assert "nowhere" in printed(result)


def test_a_project_without_a_default_direction_says_so(tmp_path):
    new_project(tmp_path)
    result = runner.invoke(app, ["--root", str(tmp_path), "export"])
    assert result.exit_code != 0
    assert "no direction" in printed(result)


def test_a_folder_that_is_not_a_project_says_so(tmp_path):
    result = runner.invoke(app, ["--root", str(tmp_path), "export"])
    assert result.exit_code != 0
    assert "no project" in printed(result)


def test_an_explicit_direction_does_not_skip_the_project_check(tmp_path):
    """The project is read even when `--direction` says which one to use.

    Otherwise a command that draws a picture would pay for it before a later
    stage reads the missing sidecar and raises.
    """
    result = runner.invoke(
        app,
        [
            "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
            "generate", "a cosy den", "--size", "120", "--auto-approve",
        ],
    )
    assert result.exit_code == 2, printed(result)
    assert "no project" in printed(result)
    assert result.exception is None or isinstance(result.exception, SystemExit)
    # Nothing was drawn: the refusal came before the first stage call.
    assert not (tmp_path / "directions").exists()


def test_one_unreadable_project_does_not_hide_the_others(tmp_path):
    """A half-copied sidecar names itself; every other project still lists."""
    new_project(tmp_path / "good-one")
    broken = tmp_path / "broken-one"
    broken.mkdir()
    (broken / "pixelmagic.json").write_text('{"name": "half a fi')
    output = invoke("--root", str(tmp_path), "project", "list").output
    assert "DemoGame" in output
    assert "unreadable pixelmagic.json" in output
    assert str(broken) in output


def test_a_sidecar_that_is_not_even_text_does_not_hide_the_others(tmp_path):
    """Not UTF-8: this fails on the decode, before pydantic sees the file."""
    new_project(tmp_path / "good-one")
    broken = tmp_path / "broken-one"
    broken.mkdir()
    (broken / "pixelmagic.json").write_bytes(b"\xff\xfe{\x00")
    output = invoke("--root", str(tmp_path), "project", "list").output
    assert "DemoGame" in output
    assert "unreadable pixelmagic.json" in output


def test_generate_will_not_draw_a_sheet_that_has_no_scene_to_describe(tmp_path):
    """`--kind` offers only what `generate` can honestly draw on its own."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        [
            "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
            "generate", "a cosy den", "--kind", "parts-sheet", "--auto-approve",
        ],
    )
    assert result.exit_code == 2
    assert "parts-sheet" in printed(result)
    assert "scene|prop" in flat_help("generate")


def test_generate_still_takes_the_two_kinds_it_documents(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    for kind in ("scene", "prop"):
        invoke(
            "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
            "generate", "a wooden chair", "--kind", kind, "--size", "64", "--auto-approve",
        )


def flat_help(command: str) -> str:
    """One command's `--help` as a single line, with the box drawing removed.

    Typer wraps help text to the terminal width and puts a border between the
    columns, so a sentence can arrive split across rows. Stripping the borders
    and collapsing whitespace is what lets a test assert on the sentence
    rather than on today's terminal width.
    """
    text = invoke(command, "--help").output
    for border in "│╭╮╰╯─":
        text = text.replace(border, " ")
    return " ".join(text.split())


def test_decompose_describes_every_flag_it_offers():
    """Pristine output: no command prints a bare flag name with no explanation, and the
    flags of the removed props sheet (it asked a service and waited for approval) are gone."""
    help_text = flat_help("decompose")
    assert "says what to do instead" in help_text
    for gone in ("--kind", "--wait", "--rounds", "--auto-approve", "--size", "--refiner"):
        assert gone not in help_text


def test_a_direction_cannot_be_created_outside_a_project(tmp_path):
    """Nothing is written: a direction tree in a non-project is litter."""
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "direction", "new", "--name", "cosy-den",
         "--style", "Warm SNES interiors."],
    )
    assert result.exit_code == 2
    assert "no project" in printed(result)
    assert not (tmp_path / "directions").exists()


def test_a_direction_that_already_exists_is_refused(tmp_path):
    """Re-creating one would silently throw away its style and don'ts."""
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "direction", "new", "--name", "cosy-den",
         "--style", "Something else entirely."],
    )
    assert result.exit_code == 2
    assert "already exists" in printed(result)
    kept = Store(tmp_path).load_direction("cosy-den")
    assert kept.style == "Warm SNES interiors."
    assert kept.donts == ["no gradients"]


def test_a_sidecar_that_will_not_parse_stays_a_traceback(tmp_path, monkeypatch):
    """A `ValidationError` is a `ValueError`; it must not be dressed up as one line.

    Driven through a patched stage rather than a corrupt file on disk: no
    sidecar is parsed inside `refine`'s guarded block today, so there is no
    file this command would read to produce one. The guard is what stops the
    next stage that does read one from having its failure mislabelled.
    """
    new_project(tmp_path)
    new_direction(tmp_path)

    def explode(*args, **kwargs):
        RefinerSettings.model_validate({"scale": "not a number"})

    monkeypatch.setattr("pixelmagic.cli.refine_direction", explode)
    result = runner.invoke(
        app, ["--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "stub"]
    )
    assert isinstance(result.exception, ValidationError), result.exception


def test_a_project_that_already_exists_is_refused(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(app, ["--root", str(tmp_path), "project", "new", "SomethingElse"])
    assert result.exit_code == 2
    assert "already exists" in printed(result)
    # The name and the default direction the first project recorded survive.
    project = Store(tmp_path).load_project()
    assert project.name == "DemoGame"
    assert project.default_direction == "cosy-den"


def test_decomposing_a_generation_that_does_not_exist_says_so(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "decompose", "20200101-000000"],
    )
    assert result.exit_code != 0
    assert "20200101-000000" in printed(result)


def test_cutting_up_a_rejected_picture_is_a_message_not_a_traceback(tmp_path):
    """Controller ruling 2: `decompose` (the stage) now refuses a `source`
    nobody approved on its own, so this command needs its own `ValueError`
    handler rather than letting it escape as a traceback."""
    new_project(tmp_path)
    new_direction(tmp_path)
    rejected = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "generate", "a cosy den", "--size", "96", "--wait", "--rounds", "1"],
        input="n\nmake it warmer\n",
    )
    assert isinstance(rejected.exception, SystemExit), rejected.exception
    generation_id = list(
        (tmp_path / "directions" / "cosy-den" / "generations").iterdir()
    )[0].name
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "decompose", generation_id, "--surface", FLOOR_SURFACE],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "rejected" in printed(result)
    assert "Traceback" not in printed(result)


def test_an_unknown_refiner_says_which_ones_exist(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "refine", "--refiner", "magic"],
    )
    assert result.exit_code != 0
    assert "stub" in printed(result)


def test_an_unknown_service_says_which_ones_exist(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "crayons", "--direction", "cosy-den",
         "generate", "a cosy den", "--auto-approve"],
    )
    assert result.exit_code != 0
    assert "crayons" in printed(result)


def test_asking_for_both_wait_and_auto_approve_is_refused(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
         "generate", "a cosy den", "--wait", "--auto-approve"],
    )
    assert result.exit_code != 0
    assert "not both" in printed(result)


def test_without_a_flag_a_script_approves_and_a_terminal_asks():
    """Locked decision 7: one hook, and the default follows where it is running."""
    from pixelmagic.approval import TerminalApproval
    from pixelmagic.cli import load_approval

    assert isinstance(load_approval(False, False, interactive=False), AutoApprove)
    assert isinstance(load_approval(False, False, interactive=True), TerminalApproval)


def test_find_projects_looks_one_level_down(tmp_path):
    from pixelmagic.cli import find_projects

    for name in ("beta", "alpha"):
        Store(tmp_path / name).create_project(name)
    (tmp_path / "not-a-project").mkdir()
    assert find_projects(tmp_path) == [tmp_path / "alpha", tmp_path / "beta"]
    assert find_projects(tmp_path / "alpha") == [tmp_path / "alpha"]
    assert find_projects(tmp_path / "missing") == []


def test_a_direction_can_be_renamed_from_the_terminal(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app, ["--root", str(tmp_path), "direction", "rename", "cosy-den", "warm-den"]
    )
    assert result.exit_code == 0
    assert "warm-den" in printed(result)
    assert Store(tmp_path).list_directions() == ["warm-den"]


def test_renaming_onto_a_name_that_is_taken_is_a_usage_error(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    runner.invoke(
        app, ["--root", str(tmp_path), "direction", "new", "--name", "cold-cave",
              "--style", "Cold."],
    )
    result = runner.invoke(
        app, ["--root", str(tmp_path), "direction", "rename", "cosy-den", "cold-cave"]
    )
    assert result.exit_code == 2
    assert "already exists" in printed(result)
    assert Store(tmp_path).list_directions() == ["cold-cave", "cosy-den"]


def test_a_direction_json_that_will_not_parse_stays_a_traceback_on_rename(tmp_path):
    """`rename` used to catch bare `ValueError`, which is also what a corrupt
    `direction.json` raises on load — so the read failure came out as a
    one-line usage error with a pydantic dump in it, instead of the traceback
    `test_a_sidecar_that_will_not_parse_stays_a_traceback` requires for every
    other stage that reads a sidecar.
    """
    new_project(tmp_path)
    new_direction(tmp_path)
    Store(tmp_path).direction_file("cosy-den").write_text("{not json at all")
    result = runner.invoke(
        app, ["--root", str(tmp_path), "direction", "rename", "cosy-den", "warm-den"]
    )
    assert isinstance(result.exception, ValidationError), result.exception
    assert Store(tmp_path).list_directions() == ["cosy-den"]


def test_a_service_that_cannot_draw_is_a_message_not_a_traceback(tmp_path, monkeypatch):
    """The CLI owns one job: turning a stage's exception into a sentence."""

    class Refusing(FakeImageService):
        name = "refusing"

        def generate(self, request, out_path, *, reference_paths=()):
            raise ServiceError("no key on this machine", service=self.name)

    new_project(tmp_path)
    new_direction(tmp_path)
    monkeypatch.setattr("pixelmagic.cli.load_service", lambda name: Refusing())
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den", "generate",
         "a chair", "--auto-approve"],
    )
    assert result.exit_code == 1
    assert "refusing: no key on this machine" in printed(result)
    assert "Traceback" not in printed(result)


def test_a_name_with_nothing_usable_in_it_is_refused(tmp_path):
    """Better a usage error than a direction called `item` nobody asked for."""
    new_project(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "direction", "new", "--name", "日本語",
         "--style", "Warm SNES interiors."],
    )
    assert result.exit_code != 0
    assert "日本語" in printed(result)
    assert Store(tmp_path).list_directions() == []


def test_an_accented_name_keeps_its_letters(tmp_path):
    new_project(tmp_path)
    invoke(
        "--root", str(tmp_path), "direction", "new",
        "--name", "Café Noir", "--style", "Warm SNES interiors.",
    )
    assert Store(tmp_path).list_directions() == ["cafe-noir"]


def test_the_web_command_refuses_a_host_that_is_not_this_machine(tmp_path):
    new_project(tmp_path)
    result = runner.invoke(app, ["--root", str(tmp_path), "web", "--host", "0.0.0.0"])
    assert result.exit_code == 2
    assert "local-only" in printed(result)


def test_the_web_command_says_when_there_is_no_project_here(tmp_path):
    result = runner.invoke(app, ["--root", str(tmp_path), "web"])
    assert result.exit_code == 2
    assert "no project" in printed(result)


def test_the_mcp_command_says_when_there_is_no_project_here(tmp_path):
    result = runner.invoke(app, ["--root", str(tmp_path), "mcp"])
    assert result.exit_code == 2
    assert "no project" in printed(result)


def test_the_mcp_command_prints_nothing_to_stdout_before_it_serves(tmp_path, monkeypatch):
    """stdout is the protocol. One stray line and the client cannot parse it."""
    new_project(tmp_path)
    new_direction(tmp_path)
    served = []
    monkeypatch.setattr(
        "pixelmagic.mcp_server.MCPServer.run", lambda self, *a, **k: served.append(self)
    )
    result = runner.invoke(app, ["--root", str(tmp_path), "mcp"])
    assert result.exit_code == 0, result.output
    assert served, "the server never ran"
    # `printed()`'s own docstring says not to lean on click keeping stdout and
    # stderr apart, and this does. It is kept because the thing being proved
    # is exactly that: `pixelmagic mcp` speaks the protocol on stdout, so one
    # stray `typer.echo` there corrupts every message. It passes on typer
    # 0.27.2 (measured: `stdout=""`, `stderr="to stderr\n"`), and if a future
    # click merges the streams it fails loudly rather than silently.
    assert result.stdout == ""


# --- M3.6 Task 2: sitting on the grid, from the terminal --------------------


def test_direction_new_takes_a_cell_size(tmp_path):
    new_project(tmp_path)
    output = invoke(
        "--root", str(tmp_path), "direction", "new",
        "--name", "cosy-den", "--style", "Warm SNES interiors.", "--cell-size", "32",
    ).output
    assert "cell size 32" in output
    assert Store(tmp_path).load_direction("cosy-den").cell_size == 32


def test_direction_new_defaults_the_cell_size_to_sixteen(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    assert Store(tmp_path).load_direction("cosy-den").cell_size == 16


def test_a_cell_size_outside_the_bound_is_a_usage_error_and_writes_nothing(tmp_path):
    """The bound is `Direction`'s; `--cell-size` mirrors it with the same two
    constants, so Typer refuses before the core is called — exit 2, one line,
    and no direction folder."""
    new_project(tmp_path)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "direction", "new", "--name", "cosy-den",
         "--style", "Warm SNES interiors.", "--cell-size", "3"],
    )
    assert result.exit_code == 2
    assert "cell-size" in printed(result)
    assert "128" in printed(result)
    assert "Traceback" not in printed(result)
    assert Store(tmp_path).list_directions() == []


def cut_up_unkept(tmp_path, object_picture):
    """Import a picture, suggest, pick three objects and redraw them, leaving the pieces as pieces.

    `redraw --no-keep` leaves the review window between cutting and keeping open.
    """
    new_project(tmp_path)
    new_direction(tmp_path)
    root = ["--root", str(tmp_path), "--direction", "cosy-den"]
    invoke(*root, "import", str(object_picture))
    generation_id = Store(tmp_path).list_generations("cosy-den")[0]
    invoke(*root, "suggest", generation_id, "--segmenter", "stub")
    decomposition_id = Store(tmp_path).list_decompositions("cosy-den")[0]
    invoke(*root, "pick", decomposition_id, "1", "2", "3")
    invoke(*root, "--service", "fake", "redraw", decomposition_id, "--no-keep")
    return Store(tmp_path), decomposition_id


def test_decompose_no_keep_leaves_the_pieces_as_pieces(tmp_path):
    """`--no-keep` through the command line: nothing under assets/, every piece on disk."""
    new_project(tmp_path)
    new_direction(tmp_path)
    generation_id = import_floor(tmp_path)
    invoke(
        "--root", str(tmp_path), "--direction", "cosy-den",
        "decompose", generation_id, "--surface", FLOOR_SURFACE,
        "--no-keep",
    )
    store = Store(tmp_path)
    assert store.list_assets("cosy-den") == []
    (decomposition_id,) = store.list_decompositions("cosy-den")
    cut = load_decomposition(store, "cosy-den", decomposition_id)
    assert len(cut.cutouts) == 32
    for cutout in cut.cutouts:
        folder = store.cutout_dir("cosy-den", decomposition_id, cutout.name)
        assert (folder / "source.png").is_file()
        assert (folder / "cutout.json").is_file()


def test_decompose_still_keeps_by_default(tmp_path):
    """Today's behaviour is the default; `--no-keep` is the opt-out."""
    new_project(tmp_path)
    new_direction(tmp_path)
    generation_id = import_floor(tmp_path)
    invoke(
        "--root", str(tmp_path), "--direction", "cosy-den",
        "decompose", generation_id, "--surface", FLOOR_SURFACE,
    )
    assert len(Store(tmp_path).list_assets("cosy-den")) == 32


def test_keep_turns_the_pieces_into_assets(tmp_path, object_picture):
    store, decomposition_id = cut_up_unkept(tmp_path, object_picture)
    output = invoke(
        "--root", str(tmp_path), "--direction", "cosy-den", "keep", decomposition_id
    ).output
    assert "kept 3 asset(s)" in output
    assert len(store.list_assets("cosy-den")) == 3


def test_keeping_a_run_that_does_not_exist_is_a_usage_error(tmp_path):
    new_project(tmp_path)
    new_direction(tmp_path)
    result = runner.invoke(
        app, ["--root", str(tmp_path), "--direction", "cosy-den", "keep", "20200101-000000"]
    )
    assert result.exit_code == 2
    assert "20200101-000000" in printed(result)
    assert "Traceback" not in printed(result)


def test_set_scene_pitch_changes_the_kept_scenes_target(tmp_path):
    """`generate --as-asset` is the CLI's way to a kept scene; its whole
    picture is 96 x 96, so a pitch of 2.0 halves it to 48 x 48."""
    new_project(tmp_path)
    new_direction(tmp_path)
    invoke(
        "--root", str(tmp_path), "--service", "fake", "--direction", "cosy-den",
        "generate", "a cosy den", "--size", "96", "--auto-approve", "--as-asset",
        "--refiner", "stub",
    )
    store = Store(tmp_path)
    output = invoke(
        "--root", str(tmp_path), "--direction", "cosy-den",
        "set-scene-pitch", "cosy-den", "2.0",
    ).output
    assert "2.0" in output and "48" in output
    after = store.read_model(store.asset_dir("cosy-den", "cosy-den") / "asset.json", AssetRecord)
    assert after.pitch == 2.0
    assert after.pitch_source is PitchSource.SET
    assert after.target == (48, 48)


def test_set_scene_pitch_on_a_prop_is_refused(tmp_path, object_picture):
    """A rebuilt prop's pitch is its own; the refusal says only a scene's can be set."""
    store, decomposition_id = cut_up_unkept(tmp_path, object_picture)
    invoke("--root", str(tmp_path), "--direction", "cosy-den", "keep", decomposition_id)
    name = store.list_assets("cosy-den")[0]
    before = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "set-scene-pitch", name, "2.0"],
    )
    assert result.exit_code == 2
    assert "only a scene's pitch" in printed(result)
    assert "Traceback" not in printed(result)
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after == before


# --- M3.6 Task 2, Steps 10-15: anchor from the terminal ----------


def a_prop(store):
    """The name and record of one asset that has an anchor — a prop."""
    for name in store.list_assets("cosy-den"):
        record = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
        if record.anchor is not None:
            return name, record
    raise AssertionError("the fake pipeline made no prop with an anchor")


def test_anchor_moves_where_a_prop_stands(tmp_path, object_picture):
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    store = Store(tmp_path)
    name, record = a_prop(store)
    # default: bottom centre
    assert record.anchor == default_anchor(record.target)
    # x and y are given distinct values on purpose: a command that swapped
    # them would still land inside the (square) target and go unnoticed.
    output = invoke(
        "--root", str(tmp_path), "--direction", "cosy-den", "anchor", name, "1", "5"
    ).output
    assert "(1, 5)" in output
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.anchor == (1, 5)


def test_an_anchor_outside_the_target_is_a_usage_error(tmp_path, object_picture):
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    store = Store(tmp_path)
    name, before = a_prop(store)
    outside_x, outside_y = before.target[0] + 1000, before.target[1] + 1000
    result = runner.invoke(
        app,
        ["--root", str(tmp_path), "--direction", "cosy-den", "anchor", name,
         str(outside_x), str(outside_y)],
    )
    assert result.exit_code == 2
    assert str(outside_x) in printed(result)
    assert "Traceback" not in printed(result)
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.anchor == before.anchor


def test_anchoring_an_asset_nobody_has_names_the_ones_that_do(tmp_path, object_picture):
    new_project(tmp_path)
    new_direction(tmp_path)
    cut_by_picking(tmp_path, object_picture)
    store = Store(tmp_path)
    name, _ = a_prop(store)
    result = runner.invoke(
        app, ["--root", str(tmp_path), "--direction", "cosy-den", "anchor", "nope", "1", "1"]
    )
    assert result.exit_code == 2
    assert "nope" in printed(result)
    assert name in printed(result)
    assert "Traceback" not in printed(result)


def test_the_grid_commands_describe_every_flag_they_offer():
    """Pristine output, as for decompose and run: no bare flag names."""
    assert "--no-keep" in flat_help("decompose")
    assert "art pixels" in flat_help("anchor")


def test_a_run_id_that_climbs_to_a_real_run_is_refused_not_followed(tmp_path):
    """An id like `../generations/<real id>` resolves to a real folder, so the old code followed
    it and succeeded. It must be refused, naming the id, and must not act on the real target."""
    new_project(tmp_path)
    new_direction(tmp_path)
    root = ["--root", str(tmp_path), "--direction", "cosy-den"]
    generation_id = import_floor(tmp_path)
    invoke(*root, "suggest", generation_id, "--segmenter", "stub")
    store = Store(tmp_path)
    run_id = store.list_decompositions("cosy-den")[0]

    climbing_generation = f"../generations/{generation_id}"
    result = runner.invoke(app, [*root, "suggest", climbing_generation, "--segmenter", "stub"])
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)  # a clean exit, not a traceback
    assert repr(climbing_generation) in printed(result)
    assert store.list_decompositions("cosy-den") == [run_id]  # no run was made from it

    climbing_run = f"../decompositions/{run_id}"
    result = runner.invoke(app, [*root, "picks", climbing_run])
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert repr(climbing_run) in printed(result)


def test_pick_with_a_picture_id_says_it_is_a_picture_id_and_suggest_is_next(
    tmp_path, object_picture
):
    """Catches: the command line still saying only "has no cut-up run" when the id typed is a
    picture's."""
    new_project(tmp_path)
    new_direction(tmp_path)
    root = ["--root", str(tmp_path), "--direction", "cosy-den"]
    generation_id = invoke(*root, "import", str(object_picture)).output.split()[3]

    refused = runner.invoke(app, [*root, "pick", generation_id, "1"])
    assert refused.exit_code != 0
    assert "picture id" in printed(refused)
    assert f"pixelmagic suggest {generation_id}" in printed(refused)


@pytest.mark.parametrize(
    ("case", "sentence"),
    [
        ("pixel-cut run", "cut straight from its own pixels"),
        ("nothing picked", "nothing is picked yet"),
        ("no reference images", "cannot take reference images"),
    ],
)
def test_redraw_that_will_be_refused_does_not_say_it_is_drawing(
    case, sentence, tmp_path, monkeypatch, object_picture
):
    """Catches: `redraw` printing "Drawing with <service>." (which tells a person their key is
    being used) and only then refusing: a pixel-cut run, a run with nothing picked, or a
    service that cannot take reference images."""
    new_project(tmp_path)
    new_direction(tmp_path)
    root = ["--root", str(tmp_path), "--direction", "cosy-den"]
    invoke(*root, "import", str(object_picture))
    generation_id = Store(tmp_path).list_generations("cosy-den")[0]
    if case == "pixel-cut run":
        invoke(*root, "pixel-cut", generation_id, "--size", "1")
    else:
        invoke(*root, "suggest", generation_id, "--segmenter", "stub")
    run_id = Store(tmp_path).list_decompositions("cosy-den")[0]
    if case == "pixel-cut run":
        invoke(*root, "pick", run_id, "--box", "0,0,60,50", "--segmenter", "stub")
    elif case == "no reference images":
        invoke(*root, "pick", run_id, "1", "--segmenter", "stub")
        monkeypatch.setattr(
            "pixelmagic.cli.load_service",
            lambda name: FakeImageService(
                capabilities=ImageCapabilities(
                    transparent_background=True, reference_images=False, vision=True
                )
            ),
        )

    refused = runner.invoke(app, ["--service", "fake", *root, "redraw", run_id])

    assert refused.exit_code == 1
    assert sentence in printed(refused)
    assert "Drawing with" not in printed(refused)


def test_suggest_closes_with_numbers_that_were_suggested(tmp_path, object_picture):
    """Catches: the closing hint offering invented numbers (it once said "3 7 12") that an
    agent then tried to pick, whatever was actually found."""
    new_project(tmp_path)
    new_direction(tmp_path)
    root = ["--root", str(tmp_path), "--direction", "cosy-den"]
    generation_id = invoke(*root, "import", str(object_picture)).output.split()[3]
    suggested = invoke(*root, "suggest", generation_id, "--segmenter", "stub").output
    run_id = Store(tmp_path).list_decompositions("cosy-den")[0]
    assert f"pick with: pixelmagic pick {run_id} 1 2 3\n" in suggested


def test_pixel_cut_road_from_import_to_a_manifest_that_says_the_props_were_not_refined(
    tmp_path, object_picture
):
    """The whole command-line road for a picture that is already pixel art.

    Each step is a thin face over the core, so what can break here is the wiring: a
    command that does not reach the core, a cut prop that is not kept, or a refine and
    export that do not recognise it as pixel-cut.
    """
    new_project(tmp_path)
    new_direction(tmp_path)
    root = ["--root", str(tmp_path), "--direction", "cosy-den"]
    invoke(*root, "import", str(object_picture))
    generation_id = Store(tmp_path).list_generations("cosy-den")[0]
    invoke(*root, "pixel-cut", generation_id, "--size", "1")
    run_id = Store(tmp_path).list_decompositions("cosy-den")[0]
    invoke(*root, "pick", run_id, "--box", "0,0,60,50", "--segmenter", "stub")
    invoke(*root, "pick", run_id, "--box", "130,0,210,80", "--segmenter", "stub")
    cut = invoke(*root, "cut", run_id)
    assert "2 piece(s)" in cut.output
    refined = invoke(*root, "refine", "--refiner", "stub")
    assert "2 pixel-cut prop(s) were not sent to the refiner" in refined.output
    invoke(*root, "export")
    manifest_file = next((tmp_path / "exports").glob("*/manifest.json"))
    entries = json.loads(manifest_file.read_text())["entries"]
    props = [entry for entry in entries if entry["route"] == "pixel-cut"]
    assert len(props) == 2
    assert {entry["refiner_tool"] for entry in props} == {"pixelmagic-pixel-cut"}
