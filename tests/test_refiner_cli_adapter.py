import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import RefinerSettings
from pixelmagic.refiner.base import (
    EXIT_ENGINE_FAILURE,
    EXIT_NOT_RUN,
    EXIT_REFUSED_IO,
    EXIT_USAGE,
    LOW_GRID_CONFIDENCE,
    WARNING_CODES,
    Analysis,
    RefineFailure,
    RefineRequest,
    RefinerError,
    RefineResult,
)
from pixelmagic.refiner.cli_adapter import (
    BINARY_ENV,
    PixelRefinerCli,
    build_analyze_args,
    build_refine_args,
    default_command,
)
from pixelmagic.refiner.stub import StubRefiner

#: A stand-in for the two-token real command, so the tests never depend on
#: anyone's checkout path.
FAKE_COMMAND = ["node", "/opt/pixel-refiner/bin/pixel-refiner.mjs"]

#: A trimmed copy of a real `refine --no-preview --detail full` reply
#: (a real PixelRefiner run, 2026-09-04). Keys we never read are dropped; every
#: key we do read is spelled exactly the way the CLI spells it — note that the
#: written file is the top-level `output` while route, confidence, warnings,
#: grid candidates and the colour count all live under `report`.
REFINE_PAYLOAD = {
    "output": {
        "path": "/tmp/out.png",
        "width": 36,
        "height": 19,
        "scale": 1,
        "bytes": 1206,
    },
    "report": {
        "input": {"width": 1200, "height": 630, "format": "png"},
        "classification": {"kind": "scaled-pixel", "confidence": 0.6198, "reasons": []},
        "route": "refine",
        "confidence": 0.4333,
        "warnings": ["LOW_GRID_CONFIDENCE"],
        "gridCandidates": [
            {
                "outW": 217,
                "outH": 114,
                "cellW": 5.505050505050505,
                "cellH": 5.504761904761905,
                "offset": {"x": 0, "y": 0},
                "method": "trimmed-reconstruction-fast",
                "totalScore": 0.6573,
                "confidence": 0.4610,
                "subscores": {"coverage": 0.9248},
            }
        ],
        "selectedCandidateIndex": 8,
        "autoResultCandidateIndex": 8,
        "output": {
            "width": 36,
            "height": 19,
            "colorCount": 153,
            "palette": ["#efefef", "#000000"],
        },
        "candidates": [
            {
                "id": "auto-result:40x22",
                "kind": "auto-result",
                "recommended": True,
                "processingMode": "auto",
                "outW": 40,
                "outH": 22,
            }
        ],
        "effectiveOptions": {
            "processingMode": "auto",
            "detailLevel": "balanced",
            "ditherMode": "none",
            "outlineColor": {"r": 255, "g": 255, "b": 255},
        },
        "adjustments": [],
    },
}

#: What the CLI prints instead of a result when a run fails.
FAILURE_PAYLOAD = {
    "failure": {
        "code": "OUTPUT_EXISTS",
        "message": "/tmp/out.png already exists.",
        "hint": "Pass overwrite: true, or choose another output path.",
    }
}

#: A trimmed `analyze --detail full` reply (a real PixelRefiner run,
#: 2026-09-06): the same `report` as `refine`, beside `input` instead of
#: `output`, and nothing written. Two candidates so a test can tell "the
#: selected one" from "the first one".
ANALYZE_PAYLOAD = {
    "input": {"path": "/tmp/in.png"},
    "report": {
        "route": "refine",
        "confidence": 0.71,
        "warnings": ["LOW_GRID_CONFIDENCE"],
        "gridCandidates": [
            {
                "outW": 34,
                "outH": 64,
                "cellW": 2.0,
                "cellH": 2.0,
                "offset": {"x": 0, "y": 0},
                "method": "autocorrelation",
                "totalScore": 0.5,
                "confidence": 0.3,
            },
            {
                "outW": 17,
                "outH": 32,
                "cellW": 4.0,
                "cellH": 4.5,
                "offset": {"x": 0.5, "y": 0},
                "method": "trimmed-reconstruction-fast",
                "totalScore": 0.8,
                "confidence": 0.71,
            },
        ],
        "selectedCandidateIndex": 1,
        "autoResultCandidateIndex": 1,
    },
}


def advanced_flags(args) -> list[str]:
    """Every `key=value` that follows an `--advanced` in an argv, in order."""
    return [args[index + 1] for index, part in enumerate(args) if part == "--advanced"]


class FakeRunner:
    """Stands in for subprocess.run and records the command it was handed."""

    def __init__(self, payload, returncode=0, stderr=""):
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=json.dumps(self.payload),
            stderr=self.stderr,
        )


def cli(runner) -> PixelRefinerCli:
    return PixelRefinerCli(FAKE_COMMAND, runner=runner)


def request(tmp_path, settings=None, overwrite=True) -> RefineRequest:
    return RefineRequest(
        input_path=tmp_path / "in.png",
        output_path=tmp_path / "out.png",
        settings=settings or RefinerSettings(),
        overwrite=overwrite,
    )


def test_the_plain_command_matches_the_contract(tmp_path):
    args = build_refine_args(request(tmp_path))
    assert args == [
        "refine",
        str((tmp_path / "in.png").resolve()),
        "--output",
        str((tmp_path / "out.png").resolve()),
        "--overwrite",
        "--no-preview",
        "--detail",
        "full",
    ]


def test_every_settings_layer_becomes_its_own_repeated_flag(tmp_path):
    settings = RefinerSettings(
        preset="crisp-sprite",
        quick={"dithering": "off", "detailLevel": "balanced"},
        advanced={"colorCount": 24, "trimToContent": True},
        scale=2,
    )
    args = build_refine_args(request(tmp_path, settings))
    assert args[args.index("--preset") + 1] == "crisp-sprite"
    quick = [args[index + 1] for index, part in enumerate(args) if part == "--quick"]
    assert quick == ["detailLevel=balanced", "dithering=off"]
    advanced = [args[index + 1] for index, part in enumerate(args) if part == "--advanced"]
    assert advanced == ["colorCount=24", "trimToContent=true"]
    assert args[args.index("--scale") + 1] == "2"


def test_advanced_values_are_written_the_way_the_cli_reads_them(tmp_path):
    settings = RefinerSettings(
        advanced={
            "backgroundDehalo": False,
            "bgRgb": "#ff00ff",
            "outlineColor": None,
            "outlineStyle": "none",
        }
    )
    args = build_refine_args(request(tmp_path, settings))
    advanced = [args[index + 1] for index, part in enumerate(args) if part == "--advanced"]
    assert advanced == [
        "backgroundDehalo=false",
        "bgRgb=#ff00ff",
        "outlineColor=null",
        "outlineStyle=none",
    ]


def test_overwrite_is_omitted_when_not_asked_for(tmp_path):
    assert "--overwrite" not in build_refine_args(request(tmp_path, overwrite=False))


def test_paths_are_always_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_refine_args(
        RefineRequest(input_path=Path("in.png"), output_path=Path("out.png"))
    )
    assert Path(args[1]).is_absolute()
    assert Path(args[3]).is_absolute()


def test_refine_reads_the_report_not_the_top_level(tmp_path):
    result = cli(FakeRunner(REFINE_PAYLOAD)).refine(request(tmp_path))
    assert isinstance(result, RefineResult)
    assert result.route == "refine"
    assert result.confidence == pytest.approx(0.4333)
    assert result.classification == "scaled-pixel"
    assert result.warnings == ["LOW_GRID_CONFIDENCE"]
    # The written file's size comes from the top-level output …
    assert (result.output.width, result.output.height) == (36, 19)
    # … the colour numbers only exist under report.output.
    assert result.output.color_count == 153
    assert result.output.palette == ["#efefef", "#000000"]


def test_grid_candidates_keep_the_clis_own_field_names(tmp_path):
    result = cli(FakeRunner(REFINE_PAYLOAD)).refine(request(tmp_path))
    candidate = result.grid_candidates[0]
    assert (candidate.out_w, candidate.out_h) == (217, 114)
    # Cell sizes are fractional whenever the sprite was scaled by a non-integer.
    assert candidate.cell_w == pytest.approx(5.505050505050505)
    assert candidate.method == "trimmed-reconstruction-fast"
    assert candidate.total_score == pytest.approx(0.6573)
    assert (candidate.offset.x, candidate.offset.y) == (0, 0)


def test_a_grid_candidate_missing_a_key_is_a_contract_break(tmp_path):
    payload = json.loads(json.dumps(REFINE_PAYLOAD))
    payload["report"]["gridCandidates"] = [{"tileW": 8, "tileH": 8}]
    with pytest.raises(RefinerError, match="grid candidate"):
        cli(FakeRunner(payload)).refine(request(tmp_path))


def test_a_fractional_grid_offset_is_read_not_refused(tmp_path):
    """A real reply carries fractional offsets, so the models must take them.

    Straight out of a real `refine` reply (a real PixelRefiner run, 2026-09-04):
    4 of that run's 12 candidates had a fractional offset, because a sprite
    upscaled by a non-integer factor has a fractional grid origin.
    """
    payload = json.loads(json.dumps(REFINE_PAYLOAD))
    payload["report"]["gridCandidates"][0]["offset"] = {"x": 3.25, "y": 1.5}
    result = cli(FakeRunner(payload)).refine(request(tmp_path))
    offset = result.grid_candidates[0].offset
    assert (offset.x, offset.y) == pytest.approx((3.25, 1.5))


def test_a_grid_candidate_our_models_reject_is_a_contract_break_too(tmp_path):
    """Whichever side moved, the caller gets a RefinerError, not a traceback.

    A pydantic ValidationError escaping a subprocess adapter says nothing about
    which of the two contracts changed; this says which candidate and why.
    """
    payload = json.loads(json.dumps(REFINE_PAYLOAD))
    payload["report"]["gridCandidates"][0]["offset"] = {"x": "left", "y": 0}
    with pytest.raises(RefinerError, match="grid candidate") as caught:
        cli(FakeRunner(payload)).refine(request(tmp_path))
    assert caught.value.exit_code == EXIT_NOT_RUN


def test_effective_options_are_kept_verbatim(tmp_path):
    result = cli(FakeRunner(REFINE_PAYLOAD)).refine(request(tmp_path))
    assert result.effective_options == REFINE_PAYLOAD["report"]["effectiveOptions"]


def test_the_configured_command_leads_the_argv(tmp_path):
    runner = FakeRunner(REFINE_PAYLOAD)
    cli(runner).refine(request(tmp_path))
    assert runner.commands[0][:2] == FAKE_COMMAND
    assert runner.commands[0][2] == "refine"


def test_the_command_can_be_swapped_from_the_environment(monkeypatch):
    """With nothing set the command is `pixel-refiner` from the PATH, never a
    hard-coded location; the environment variable can point at any build."""
    monkeypatch.delenv(BINARY_ENV, raising=False)
    assert default_command() == ["pixel-refiner"]
    other = ["node", "/opt/other-checkout/pixel-refiner.mjs"]
    monkeypatch.setenv(BINARY_ENV, shlex.join(other))
    assert default_command() == other


def test_background_pick_without_a_colour_never_spawns_a_process(tmp_path):
    runner = FakeRunner(REFINE_PAYLOAD)
    with pytest.raises(ValueError, match="backgroundColor"):
        cli(runner).refine(request(tmp_path, RefinerSettings(quick={"background": "pick"})))
    assert runner.commands == []


def test_a_target_becomes_the_two_force_pixels_keys(tmp_path):
    """The mutation check spec §8 demands: delete the one line in
    `build_refine_args` that folds the target in, and this goes red."""
    asked = request(tmp_path).model_copy(update={"target": (17, 32)})
    assert advanced_flags(build_refine_args(asked)) == ["forcePixelsH=32", "forcePixelsW=17"]


def test_no_target_sends_no_force_pixels_keys(tmp_path):
    flags = advanced_flags(build_refine_args(request(tmp_path)))
    assert not any(flag.startswith("forcePixels") for flag in flags)


def test_the_target_wins_over_a_force_key_already_in_the_settings(tmp_path):
    """One flag per key reaches the CLI, whatever the caller's settings held —
    so the target wins without depending on which repeated flag the CLI honours."""
    asked = request(
        tmp_path, RefinerSettings(advanced={"forcePixelsW": 99, "trimToContent": True})
    ).model_copy(update={"target": (17, 32)})
    assert advanced_flags(build_refine_args(asked)) == [
        "forcePixelsH=32",
        "forcePixelsW=17",
        "trimToContent=true",
    ]


def test_a_forced_refine_reaches_the_process_with_the_keys(tmp_path):
    runner = FakeRunner(REFINE_PAYLOAD)
    cli(runner).refine(request(tmp_path).model_copy(update={"target": (17, 32)}))
    assert "forcePixelsW=17" in advanced_flags(runner.commands[0])
    assert "forcePixelsH=32" in advanced_flags(runner.commands[0])


def test_analyze_runs_the_verb_with_full_detail_and_no_output(tmp_path):
    runner = FakeRunner(ANALYZE_PAYLOAD)
    source = tmp_path / "in.png"
    cli(runner).analyze(source)
    argv = runner.commands[0]
    assert argv[:2] == FAKE_COMMAND
    assert argv[2:] == ["analyze", str(source.resolve()), "--detail", "full"]
    assert build_analyze_args(source, RefinerSettings()) == argv[2:]


def test_analyze_passes_the_settings_flags(tmp_path):
    runner = FakeRunner(ANALYZE_PAYLOAD)
    cli(runner).analyze(
        tmp_path / "in.png",
        RefinerSettings(preset="crisp-sprite", quick={"dithering": "off"}),
    )
    argv = runner.commands[0]
    assert argv[argv.index("--preset") + 1] == "crisp-sprite"
    assert argv[argv.index("--quick") + 1] == "dithering=off"


def test_analyze_reads_the_selected_candidates_cell_size_as_the_pitch(tmp_path):
    reading = cli(FakeRunner(ANALYZE_PAYLOAD)).analyze(tmp_path / "in.png")
    assert isinstance(reading, Analysis)
    # (4.0 + 4.5) / 2 — the selected candidate, not the first one.
    assert reading.pitch == pytest.approx(4.25)
    assert reading.confidence == pytest.approx(0.71)
    assert reading.warnings == [LOW_GRID_CONFIDENCE]
    assert [candidate.cell_w for candidate in reading.grid_candidates] == [2.0, 4.0]


def test_analyze_with_no_candidates_reports_no_pitch(tmp_path):
    payload = json.loads(json.dumps(ANALYZE_PAYLOAD))
    payload["report"]["gridCandidates"] = []
    payload["report"]["selectedCandidateIndex"] = 0
    assert cli(FakeRunner(payload)).analyze(tmp_path / "in.png").pitch is None


def test_analyze_with_an_index_off_the_list_reports_no_pitch(tmp_path):
    payload = json.loads(json.dumps(ANALYZE_PAYLOAD))
    payload["report"]["selectedCandidateIndex"] = 7
    assert cli(FakeRunner(payload)).analyze(tmp_path / "in.png").pitch is None


def test_analyze_without_a_report_never_reports_success(tmp_path):
    with pytest.raises(RefinerError, match="cannot read") as caught:
        cli(FakeRunner({"input": {"path": "/tmp/in.png"}})).analyze(tmp_path / "in.png")
    assert caught.value.exit_code == EXIT_NOT_RUN


def test_background_pick_without_a_colour_never_spawns_analyze_either(tmp_path):
    runner = FakeRunner(ANALYZE_PAYLOAD)
    with pytest.raises(ValueError, match="backgroundColor"):
        cli(runner).analyze(
            tmp_path / "in.png", RefinerSettings(quick={"background": "pick"})
        )
    assert runner.commands == []


@pytest.mark.parametrize("code", [EXIT_ENGINE_FAILURE, EXIT_USAGE, EXIT_REFUSED_IO])
def test_a_failed_run_raises_with_the_exit_code_and_the_failure_code(tmp_path, code):
    runner = FakeRunner(FAILURE_PAYLOAD, returncode=code, stderr="warn OUTPUT_EXISTS")
    with pytest.raises(RefinerError) as caught:
        cli(runner).refine(request(tmp_path))
    assert caught.value.exit_code == code
    assert caught.value.code == "OUTPUT_EXISTS"
    assert "already exists" in str(caught.value)
    # The hint is the CLI's own advice; losing it makes the error harder to act on.
    assert "overwrite" in str(caught.value)


def test_unreadable_json_never_reports_success(tmp_path):
    def bad(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="not json", stderr="")

    with pytest.raises(RefinerError) as caught:
        cli(bad).refine(request(tmp_path))
    assert caught.value.exit_code == EXIT_NOT_RUN


def test_a_reply_that_is_not_an_object_never_reports_success(tmp_path):
    def a_list(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(["nope"]), stderr="")

    with pytest.raises(RefinerError, match="unreadable reply") as caught:
        cli(a_list).refine(request(tmp_path))
    assert caught.value.exit_code == EXIT_NOT_RUN


def test_a_reply_our_models_reject_never_reports_success(tmp_path):
    """A null where a number belongs is unreadable, not a crash.

    `_run` promises callers one thing to catch, and the CLI's own JSON is the
    least trustworthy input this module has.
    """
    payload = json.loads(json.dumps(REFINE_PAYLOAD))
    payload["report"]["confidence"] = None
    with pytest.raises(RefinerError, match="cannot read") as caught:
        cli(FakeRunner(payload)).refine(request(tmp_path))
    assert caught.value.exit_code == EXIT_NOT_RUN


def test_effective_options_that_are_not_an_object_never_report_success(tmp_path):
    """`dict()` on a string raises ValueError, which is still unreadable JSON."""
    payload = json.loads(json.dumps(REFINE_PAYLOAD))
    payload["report"]["effectiveOptions"] = "nope"
    with pytest.raises(RefinerError, match="cannot read") as caught:
        cli(FakeRunner(payload)).refine(request(tmp_path))
    assert caught.value.exit_code == EXIT_NOT_RUN


def test_a_batch_item_whose_error_is_not_an_object_never_reports_success(tmp_path):
    """The refused branch is the least-exercised shape in the CLI's output."""
    requests = batch_requests(tmp_path, 1)
    runner = FakeBatchRunner()

    def malformed(command, **kwargs):
        completed = runner(command, **kwargs)
        staged = command[3]
        return SimpleNamespace(
            returncode=EXIT_ENGINE_FAILURE,
            stdout=json.dumps(
                {
                    "items": [{"id": staged, "status": "error", "error": "boom"}],
                    "summary": {"done": 0, "failed": 1},
                }
            ),
            stderr=completed.stderr,
        )

    with pytest.raises(RefinerError, match="unreadable error") as caught:
        cli(malformed).batch(requests)
    assert caught.value.exit_code == EXIT_NOT_RUN
    assert not requests[0].output_path.exists()


def done_item_naming(path) -> object:
    """A batch runner whose single done item claims it wrote `path`."""

    def runner(command, **kwargs):
        item = json.loads(json.dumps(REFINE_PAYLOAD))
        item["id"] = command[3]
        item["status"] = "done"
        item["output"]["path"] = str(path)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"items": [item], "summary": {"done": 1, "failed": 0}}),
            stderr="",
        )

    return runner


def batch_reply(staged: Path, produced: Path) -> dict:
    """One finished batch item, spelled the way the CLI spells it."""
    item = json.loads(json.dumps(REFINE_PAYLOAD))
    item["id"] = str(staged)
    item["status"] = "done"
    item["output"]["path"] = str(produced)
    return {"items": [item], "summary": {"done": 1, "failed": 0}}


def test_a_file_outside_the_output_dir_is_never_moved_off_its_own_path(tmp_path):
    """`is_file()` proves a file; it does not prove the CLI wrote it.

    Without the containment half, a reply naming any file on the machine has
    that file renamed onto the caller's output_path — the incident that cost
    this branch a checkout, narrowed to one file at a time.
    """
    requests = batch_requests(tmp_path, 1)
    outsider = tmp_path / "precious" / "id_ed25519"
    outsider.parent.mkdir()
    outsider.write_text("PRIVATE KEY MATERIAL")

    with pytest.raises(RefinerError, match="not a file it wrote") as caught:
        cli(done_item_naming(outsider)).batch(requests)

    assert caught.value.exit_code == EXIT_NOT_RUN
    assert outsider.read_text() == "PRIVATE KEY MATERIAL"
    assert not requests[0].output_path.exists()


def test_a_symlink_in_the_output_dir_is_refused_and_its_target_left_alone(tmp_path):
    """`is_file()` follows links, so containment alone would still admit one.

    Moving the link hands the caller a pointer out of the tree, and the next
    stage to write to that output path writes through it.
    """
    requests = batch_requests(tmp_path, 1)
    target = tmp_path / "precious" / "hosts"
    target.parent.mkdir()
    target.write_text("127.0.0.1 localhost")

    def runner(command, **kwargs):
        output_dir = Path(command[command.index("--output-dir") + 1])
        link = output_dir / "0000-asset0.refined.png"
        link.symlink_to(target)
        return done_item_naming(link)(command, **kwargs)

    with pytest.raises(RefinerError, match="not a file it wrote") as caught:
        cli(runner).batch(requests)

    assert caught.value.exit_code == EXIT_NOT_RUN
    assert target.read_text() == "127.0.0.1 localhost"
    assert not requests[0].output_path.exists()


def test_a_symlink_is_refused_even_when_it_points_inside_the_output_dir(tmp_path):
    """Containment alone would admit this one, and a link is never what we want.

    The real CLI writes plain PNGs, so refusing links outright costs nothing
    and takes the follow-the-link question out of the guard entirely.
    """
    requests = batch_requests(tmp_path, 1)

    def runner(command, **kwargs):
        output_dir = Path(command[command.index("--output-dir") + 1])
        real = output_dir / "real.png"
        real.write_bytes(b"refined")
        link = output_dir / "0000-asset0.refined.png"
        link.symlink_to(real)
        return done_item_naming(link)(command, **kwargs)

    with pytest.raises(RefinerError, match="not a file it wrote"):
        cli(runner).batch(requests)
    assert not requests[0].output_path.exists()


def test_a_directory_inside_the_output_dir_is_refused_too(tmp_path):
    """The `is_file()` half, in the one place containment cannot cover it."""
    requests = batch_requests(tmp_path, 1)

    def runner(command, **kwargs):
        output_dir = Path(command[command.index("--output-dir") + 1])
        folder = output_dir / "0000-asset0.refined.png"
        folder.mkdir()
        (folder / "keep.txt").write_text("still here")
        return done_item_naming(folder)(command, **kwargs)

    with pytest.raises(RefinerError, match="not a file it wrote"):
        cli(runner).batch(requests)
    assert not requests[0].output_path.exists()


def test_a_reply_naming_a_link_to_our_own_staged_input_is_refused(tmp_path):
    """A linked input is not a file the refiner wrote, whatever the reply says.

    This is the case the provenance guard could not see: the path is a real
    file, it is not a symlink, and it resolves inside --output-dir, so every
    existing check passes. Only the inode says it is the picture we handed
    over rather than a new one.
    """
    folder = tmp_path / "asset0"
    folder.mkdir()
    write_tiny_png(folder / "source.png")
    request_one = RefineRequest(
        input_path=folder / "source.png", output_path=folder / "refined.png"
    )

    def link_the_input(command, **kwargs):
        staged = Path(command[command.index("batch") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        produced = output_dir / "0000-asset0.refined.png"
        os.link(staged, produced)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(batch_reply(staged, produced)),
            stderr="",
        )

    with pytest.raises(RefinerError, match="did not write"):
        cli(link_the_input).batch([request_one])
    assert not request_one.output_path.exists()


def test_a_reply_naming_a_link_to_the_callers_original_is_refused(tmp_path):
    """A link to the caller's own `input_path` is refused, not only a link to
    the copy we staged.

    The CLI never sees `request_one.input_path` — it only ever sees the
    staged copy under a scratch name — but a hand-built reply could still
    name a file hardlinked back to that original. The inode set has to catch
    that too, or a reply that aliases the caller's own picture would pass
    every check the staged-copy guard applies.
    """
    folder = tmp_path / "asset0"
    folder.mkdir()
    write_tiny_png(folder / "source.png")
    request_one = RefineRequest(
        input_path=folder / "source.png", output_path=folder / "refined.png"
    )

    def link_the_original(command, **kwargs):
        staged = Path(command[command.index("batch") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        produced = output_dir / "0000-asset0.refined.png"
        os.link(request_one.input_path, produced)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(batch_reply(staged, produced)),
            stderr="",
        )

    with pytest.raises(RefinerError, match="did not write"):
        cli(link_the_original).batch([request_one])
    assert not request_one.output_path.exists()


def test_a_link_to_a_file_the_refiner_really_wrote_is_still_accepted(tmp_path):
    """The guard stops at our inputs, and says so: a preserve route that links
    its own output to another of its own outputs is not our business."""
    folder = tmp_path / "asset0"
    folder.mkdir()
    write_tiny_png(folder / "source.png")
    request_one = RefineRequest(
        input_path=folder / "source.png", output_path=folder / "refined.png"
    )

    def link_its_own(command, **kwargs):
        staged = Path(command[command.index("batch") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        first = output_dir / "written.png"
        write_tiny_png(first)
        produced = output_dir / "0000-asset0.refined.png"
        os.link(first, produced)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(batch_reply(staged, produced)),
            stderr="",
        )

    results = cli(link_its_own).batch([request_one])
    assert request_one.output_path.exists()
    assert results[0].output.path == request_one.output_path


def test_a_batch_item_naming_a_directory_never_moves_it(tmp_path):
    """An empty output.path is `Path(".")`, which exists — and is not a file.

    Trusting `exists()` here hands a directory to shutil.move, which across
    filesystems is copytree plus rmtree. A reply must never be able to do that.
    """
    requests = batch_requests(tmp_path, 1)
    a_directory = tmp_path / "not-a-png"
    a_directory.mkdir()
    (a_directory / "keep.txt").write_text("still here")

    def malformed(command, **kwargs):
        item = json.loads(json.dumps(REFINE_PAYLOAD))
        item["id"] = command[3]
        item["status"] = "done"
        item["output"]["path"] = str(a_directory)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"items": [item], "summary": {"done": 1, "failed": 0}}),
            stderr="",
        )

    with pytest.raises(RefinerError, match="not a file it wrote") as caught:
        cli(malformed).batch(requests)
    assert caught.value.exit_code == EXIT_NOT_RUN
    assert not requests[0].output_path.exists()
    # The directory it named is still a directory, with its contents.
    assert a_directory.is_dir()
    assert (a_directory / "keep.txt").read_text() == "still here"


def test_an_empty_output_path_is_not_the_working_directory(tmp_path, monkeypatch):
    """`Path("")` resolves to the cwd; the guard has to refuse that too."""
    monkeypatch.chdir(tmp_path)
    requests = batch_requests(tmp_path, 1)

    def malformed(command, **kwargs):
        item = json.loads(json.dumps(REFINE_PAYLOAD))
        item["id"] = command[3]
        item["status"] = "done"
        item["output"]["path"] = ""
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"items": [item], "summary": {"done": 1, "failed": 0}}),
            stderr="",
        )

    before = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(RefinerError, match="not a file it wrote") as caught:
        cli(malformed).batch(requests)
    assert caught.value.exit_code == EXIT_NOT_RUN
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_a_batch_item_whose_id_is_not_a_string_never_reports_success(tmp_path):
    """An unhashable id must not blow up the by-input index."""
    requests = batch_requests(tmp_path, 1)

    def malformed(command, **kwargs):
        staged = command[3]
        return SimpleNamespace(
            returncode=EXIT_ENGINE_FAILURE,
            stdout=json.dumps(
                {
                    "items": [
                        {"id": [staged], "status": "error", "error": {"code": "X"}}
                    ],
                    "summary": {"done": 0, "failed": 1},
                }
            ),
            stderr="",
        )

    with pytest.raises(RefinerError, match="no item for") as caught:
        cli(malformed).batch(requests)
    assert caught.value.exit_code == EXIT_NOT_RUN
    assert not requests[0].output_path.exists()


def test_a_batch_whose_items_are_not_a_list_of_objects_never_reports_success(tmp_path):
    runner = FakeRunner({"items": "boom", "summary": {"done": 0, "failed": 0}})
    with pytest.raises(RefinerError, match="list of objects") as caught:
        cli(runner).batch(batch_requests(tmp_path, 1))
    assert caught.value.exit_code == EXIT_NOT_RUN


def test_a_missing_binary_never_reports_success(tmp_path):
    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])

    with pytest.raises(RefinerError, match="pixel-refiner") as caught:
        cli(missing).refine(request(tmp_path))
    assert caught.value.exit_code == EXIT_NOT_RUN
    # (d) Not-installed is a "the CLI itself never ran" failure, so the
    # message says what to type next, not only that something went wrong.
    message = str(caught.value)
    assert "--refiner stub" in message
    assert BINARY_ENV in message


#: A trimmed but genuine-shaped Node crash dump — the loader preamble, then
#: the line that names the real problem, then several stack frames. This is
#: what stderr looks like when Node cannot find the script at all, before the
#: CLI's own JSON error handling ever gets a turn (checked against the real
#: binary on 2026-09-05).
NODE_STACK_STDERR = (
    "node:internal/modules/cjs/loader:1573\n"
    "  throw err;\n"
    "  ^\n"
    "\n"
    "Error: Cannot find module '/x/y.mjs'\n"
    "    at Module._resolveFilename (node:internal/modules/cjs/loader:1569:15)\n"
    "    at Module._load (node:internal/modules/cjs/loader:1113:27)\n"
    "    at Function.executeUserEntryPoint [as runMain] "
    "(node:internal/modules/run_main:174:12)\n"
)


def test_a_tolerated_exit_with_no_reply_is_reported_as_the_failure_it_is(tmp_path):
    """(a) Finding 1: an empty stdout at a tolerated exit code is not a
    hollow success — batch must report it the same way a non-tolerated exit
    would, with the way out named."""
    requests = batch_requests(tmp_path, 1)

    def crashed(command, **kwargs):
        return SimpleNamespace(
            returncode=EXIT_ENGINE_FAILURE, stdout="", stderr="Cannot find pixel-refiner.mjs"
        )

    with pytest.raises(RefinerError) as caught:
        cli(crashed).batch(requests)
    message = str(caught.value)
    assert f"exited {EXIT_ENGINE_FAILURE}" in message
    assert "Cannot find pixel-refiner.mjs" in message
    assert "--refiner stub" in message


def test_a_node_crash_keeps_only_its_first_line(tmp_path):
    """(b) Finding 2: a Node stack trace is cut down to the line that names
    the actual problem — the `at ...` frames never reach the message."""

    def crashed(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr=NODE_STACK_STDERR)

    with pytest.raises(RefinerError) as caught:
        cli(crashed).refine(request(tmp_path))
    message = str(caught.value)
    assert "Cannot find module '/x/y.mjs'" in message
    assert "at " not in message
    assert "--refiner stub" in message


def test_options_against_a_crashed_cli_gets_the_same_shape():
    """(c) `options()` goes through the same `_run`, so a Node crash there is
    cut down and pointed at the way out too."""

    def crashed(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr=NODE_STACK_STDERR)

    with pytest.raises(RefinerError) as caught:
        cli(crashed).options()
    message = str(caught.value)
    assert "Cannot find module '/x/y.mjs'" in message
    assert "at " not in message
    assert "--refiner stub" in message


def test_a_failure_the_cli_itself_reports_names_no_way_out(tmp_path):
    """Finding 3's other half: a genuine JSON `failure` object is about the
    picture, not the install, so `--refiner stub` would be bad advice here."""
    runner = FakeRunner(FAILURE_PAYLOAD, returncode=EXIT_REFUSED_IO)
    with pytest.raises(RefinerError) as caught:
        cli(runner).refine(request(tmp_path))
    assert "--refiner stub" not in str(caught.value)


def test_a_command_the_os_refuses_to_start_never_reports_success(tmp_path):
    """Not-installed is one refusal of many; every OSError is "it did not run"."""

    def refused(command, **kwargs):
        raise PermissionError(13, "Permission denied", command[0])

    with pytest.raises(RefinerError, match="could not be started") as caught:
        cli(refused).refine(request(tmp_path))
    assert caught.value.exit_code == EXIT_NOT_RUN


def test_a_timeout_never_reports_success(tmp_path):
    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 300)

    with pytest.raises(RefinerError) as caught:
        cli(slow).refine(request(tmp_path))
    assert caught.value.exit_code == EXIT_NOT_RUN


def batch_requests(tmp_path, count):
    """Requests shaped like the real pipeline: every input is named source.png."""
    requests = []
    for index in range(count):
        folder = tmp_path / "assets" / f"asset{index}"
        folder.mkdir(parents=True)
        (folder / "source.png").write_bytes(f"source-{index}".encode())
        requests.append(
            RefineRequest(
                input_path=folder / "source.png", output_path=folder / "refined.png"
            )
        )
    return requests


class FakeBatchRunner:
    """Behaves like the real CLI's `batch`.

    Inputs are positional, each result is written to
    `<output-dir>/<stem>.refined.png`, every item is keyed by the input path,
    and a refused item makes the whole call exit 1, which is exactly the case
    the adapter has to survive.
    """

    def __init__(self, *, error_stems=(), drop_last=False):
        self.error_stems = set(error_stems)
        self.drop_last = drop_last
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        inputs = []
        for part in command[command.index("batch") + 1 :]:
            if part.startswith("--"):
                break
            inputs.append(part)
        out_dir = Path(command[command.index("--output-dir") + 1])
        items = []
        failed = 0
        for source in inputs:
            stem = Path(source).stem
            if stem in self.error_stems:
                failed += 1
                items.append(
                    {
                        "id": source,
                        "status": "error",
                        "error": {
                            "code": "UNSUPPORTED_INPUT",
                            "message": f"{source} is not an image.",
                            "hint": "Convert it first.",
                        },
                    }
                )
                continue
            written = out_dir / f"{stem}.refined.png"
            written.parent.mkdir(parents=True, exist_ok=True)
            written.write_bytes(f"refined-{stem}".encode())
            item = json.loads(json.dumps(REFINE_PAYLOAD))
            item["id"] = source
            item["status"] = "done"
            item["output"]["path"] = str(written)
            item["needsAttention"] = False
            items.append(item)
        if self.drop_last:
            items = items[:-1]
        return SimpleNamespace(
            returncode=EXIT_ENGINE_FAILURE if failed else 0,
            stdout=json.dumps(
                {
                    "items": items,
                    "summary": {"done": len(inputs) - failed, "failed": failed},
                }
            ),
            stderr="",
        )


def test_a_batch_with_a_target_forces_the_size_once(tmp_path):
    """The size travels as `--advanced forcePixelsW/H`, never `--grid force:WxH`:
    the real CLI refuses the two together (base.py's contract), so sending
    both would break every batch with a target.
    """
    runner = FakeBatchRunner()
    [request_] = batch_requests(tmp_path, 1)
    request_ = request_.model_copy(update={"target": (16, 16)})
    results = cli(runner).batch([request_])
    assert [command[2] for command in runner.commands] == ["batch"]
    command = runner.commands[0]
    assert "--grid" not in command
    flags = advanced_flags(command)
    assert flags.count("forcePixelsW=16") == 1
    assert flags.count("forcePixelsH=16") == 1
    assert isinstance(results[0], RefineResult)
    assert request_.output_path.exists()
    assert results[0].output.path == request_.output_path


def test_a_batch_with_no_target_sends_no_forced_size(tmp_path):
    """No target, no `forcePixelsW`/`forcePixelsH` advanced flag: the refiner still guesses."""
    runner = FakeBatchRunner()
    cli(runner).batch(batch_requests(tmp_path, 1))
    assert not any(
        flag.startswith("forcePixelsW") or flag.startswith("forcePixelsH")
        for flag in advanced_flags(runner.commands[0])
    )


def test_batch_sends_one_call_with_the_input_as_a_positional(tmp_path):
    runner = FakeBatchRunner()
    results = cli(runner).batch(batch_requests(tmp_path, 1))
    assert len(runner.commands) == 1
    command = runner.commands[0]
    assert command[2] == "batch"
    # The input comes straight after the verb, with no flag in front of it.
    assert not command[3].startswith("--")
    assert command[4] == "--output-dir"
    assert "--shared-palette" not in command
    assert len(results) == 1


def test_batch_staging_keeps_the_inputs_own_extension(tmp_path):
    """The CLI decides a format by extension, so a .webp must stay a .webp."""
    requests = batch_requests(tmp_path, 1)
    source = requests[0].input_path.with_suffix(".webp")
    requests[0].input_path.rename(source)
    requests[0] = requests[0].model_copy(update={"input_path": source})
    runner = FakeBatchRunner()
    cli(runner).batch(requests)
    assert Path(runner.commands[0][3]).suffix == ".webp"


def test_a_refused_item_comes_back_as_its_requests_failure(tmp_path):
    """A refused input is a `RefineFailure` and nothing is written."""
    requests = batch_requests(tmp_path, 1)
    results = cli(FakeBatchRunner(error_stems={"asset0"})).batch(requests)
    assert [type(item) for item in results] == [RefineFailure]
    assert results[0].code == "UNSUPPORTED_INPUT"
    assert results[0].input_path == requests[0].input_path
    assert not requests[0].output_path.exists()


def test_a_missing_input_is_a_failure_and_never_spawns_a_process(tmp_path):
    """The CLI never sees the caller's input path, only the staged copy, so the
    adapter is the only place that can notice the file is gone."""
    runner = FakeBatchRunner()
    requests = batch_requests(tmp_path, 1)
    requests[0].input_path.unlink()
    results = cli(runner).batch(requests)
    assert [type(item) for item in results] == [RefineFailure]
    assert results[0].code == "INPUT_NOT_FOUND"
    assert results[0].input_path == requests[0].input_path
    assert not requests[0].output_path.exists()
    assert runner.commands == []


def test_a_batch_will_not_clobber_an_output_without_overwrite(tmp_path):
    """The CLI wrote to a scratch dir, so only the adapter can refuse this."""
    requests = batch_requests(tmp_path, 1)
    requests[0] = requests[0].model_copy(update={"overwrite": False})
    requests[0].output_path.write_bytes(b"already here")
    runner = FakeBatchRunner()
    with pytest.raises(RefinerError) as caught:
        cli(runner).batch(requests)
    assert caught.value.exit_code == EXIT_REFUSED_IO
    assert requests[0].output_path.read_bytes() == b"already here"
    assert runner.commands == []


def test_a_batch_with_no_items_at_all_still_raises(tmp_path):
    """A top-level failure is the whole call failing, not one item failing."""

    def refused(command, **kwargs):
        return SimpleNamespace(
            returncode=EXIT_ENGINE_FAILURE,
            stdout=json.dumps(FAILURE_PAYLOAD),
            stderr="",
        )

    with pytest.raises(RefinerError) as caught:
        cli(refused).batch(batch_requests(tmp_path, 1))
    assert caught.value.exit_code == EXIT_ENGINE_FAILURE
    assert "already exists" in str(caught.value)


def test_a_reply_with_no_item_for_the_request_is_an_error_not_a_silent_loss(tmp_path):
    requests = batch_requests(tmp_path, 1)
    with pytest.raises(RefinerError, match="no item"):
        cli(FakeBatchRunner(drop_last=True)).batch(requests)
    assert not requests[0].output_path.exists()


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("real_adapter", [True, False], ids=["adapter", "stub"])
def test_a_batch_of_any_other_count_than_one_is_refused_before_anything_runs(
    tmp_path, real_adapter, count
):
    """Several requests with different targets would be silently forced to one
    size, because the CLI's batch verb applies its flags to every input. Zero
    is a caller bug with nothing to refine."""
    runner = FakeBatchRunner()
    refiner = cli(runner) if real_adapter else StubRefiner()
    requests = batch_requests(tmp_path, count)
    with pytest.raises(RefinerError) as caught:
        refiner.batch(requests)
    assert caught.value.exit_code == EXIT_USAGE
    assert runner.commands == []
    assert not any(request_.output_path.exists() for request_ in requests)


def test_options_reads_the_lists_of_specs():
    payload = {
        "presets": [
            {"id": "auto", "quickSettings": {"processingMode": "auto"}},
            {"id": "crisp-sprite", "quickSettings": {"detailLevel": "detailed"}},
        ],
        "palettes": [
            {"id": "gb_legacy", "name": "Game Boy (Legacy)", "colorCount": 4, "colors": []}
        ],
        "quick": [{"key": "dithering", "values": ["off"], "default": "off"}],
        "advanced": [{"key": "colorCount", "kind": "int", "min": 2, "max": 256, "default": 32}],
    }
    options = PixelRefinerCli(FAKE_COMMAND, runner=FakeRunner(payload)).options()
    assert options.presets == ["auto", "crisp-sprite"]
    assert options.palettes == ["gb_legacy"]
    assert options.advanced["colorCount"]["kind"] == "int"


def test_the_candidate_ids_that_refine_would_accept_come_back(tmp_path):
    result = cli(FakeRunner(REFINE_PAYLOAD)).refine(request(tmp_path))
    assert [candidate.id for candidate in result.candidates] == [
        item["id"] for item in REFINE_PAYLOAD["report"]["candidates"]
    ]
    first = result.candidates[0]
    assert first.kind == REFINE_PAYLOAD["report"]["candidates"][0]["kind"]
    assert first.recommended is REFINE_PAYLOAD["report"]["candidates"][0]["recommended"]


def test_a_candidate_with_no_id_is_a_loud_error(tmp_path):
    """The id is the only part `--candidate-id` needs; a candidate without one
    is an alternative nobody could ever pick."""
    payload = json.loads(json.dumps(REFINE_PAYLOAD))
    payload["report"]["candidates"] = [{"kind": "auto-result", "outW": 40}]
    with pytest.raises(RefinerError, match="cannot read"):
        cli(FakeRunner(payload)).refine(request(tmp_path))


def test_a_reply_with_no_candidates_at_all_is_fine(tmp_path):
    payload = json.loads(json.dumps(REFINE_PAYLOAD))
    payload["report"].pop("candidates", None)
    assert cli(FakeRunner(payload)).refine(request(tmp_path)).candidates == []


def test_asking_for_a_candidate_puts_it_on_the_command_line(tmp_path):
    runner = FakeRunner(REFINE_PAYLOAD)
    asked = request(tmp_path)
    asked.candidate_id = "auto-result:40x22"
    cli(runner).refine(asked)
    argv = runner.commands[0]
    assert "--candidate-id" in argv
    assert argv[argv.index("--candidate-id") + 1] == "auto-result:40x22"


def test_not_asking_for_a_candidate_sends_no_flag(tmp_path):
    runner = FakeRunner(REFINE_PAYLOAD)
    cli(runner).refine(request(tmp_path))
    assert "--candidate-id" not in runner.commands[0]


def test_options_keeps_the_quick_vocabulary_the_settings_screen_needs():
    payload = {
        "presets": [{"id": "auto", "quickSettings": {}}],
        "palettes": [{"id": "gb_legacy", "name": "Game Boy", "colorCount": 4, "colors": []}],
        "quick": [
            {"key": "dithering", "values": ["off", "subtle", "strong"], "default": "off"},
            {"key": "backgroundColor", "default": "#ff00ff"},
        ],
        "advanced": [{"key": "colorCount", "kind": "int", "min": 2, "max": 256, "default": 32}],
    }
    options = PixelRefinerCli(FAKE_COMMAND, runner=FakeRunner(payload)).options()
    assert options.quick["dithering"]["values"] == ["off", "subtle", "strong"]
    assert options.quick["backgroundColor"]["default"] == "#ff00ff"
    assert options.presets == ["auto"]
    assert options.advanced["colorCount"]["kind"] == "int"


def write_tiny_png(path: Path) -> Path:
    """A 32x32 RGBA square, big enough for the refiner to find a grid in."""
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[8:24, 8:24] = (10, 20, 200, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def test_fit_palette_sends_shared_palette_and_palette_colors_with_no_forced_size(tmp_path):
    """The M4.1 recipe's own call, offline: a one-image shared-palette batch
    with no forced size, reading the reply's top-level `sharedPalette` even
    when the item's own `report.output.palette` lists the same colours in a
    different order — exactly the disagreement M4.1 measured on the real
    build on 2026-09-25."""
    source = write_tiny_png(tmp_path / "asset0" / "source.png")
    shared = ["#112233", "#445566", "#778899"]
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(list(command))
        item = json.loads(json.dumps(REFINE_PAYLOAD))
        item["id"] = command[command.index("batch") + 1]
        item["status"] = "done"
        item["report"]["output"]["palette"] = list(reversed(shared))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "items": [item],
                    "summary": {"done": 1, "failed": 0},
                    "sharedPalette": shared,
                }
            ),
            stderr="",
        )

    palette = cli(runner).fit_palette(source, 24)

    assert palette == shared
    command = commands[0]
    assert command[2] == "batch"
    assert "--shared-palette" in command
    assert command[command.index("--palette-colors") + 1] == "24"
    assert not any(flag.startswith("forcePixels") for flag in advanced_flags(command))


def test_fit_palette_returns_a_refine_failure_for_a_refused_item(tmp_path):
    """A single input the CLI itself refuses costs that asset, not a raise —
    the same contract `batch` keeps for a refused item."""
    source = write_tiny_png(tmp_path / "asset0" / "source.png")

    def runner(command, **kwargs):
        item = {
            "id": command[command.index("batch") + 1],
            "status": "error",
            "error": {
                "code": "UNSUPPORTED_INPUT",
                "message": f"{source} is not an image.",
                "hint": "Convert it first.",
            },
        }
        return SimpleNamespace(
            returncode=EXIT_ENGINE_FAILURE,
            stdout=json.dumps({"items": [item], "summary": {"done": 0, "failed": 1}}),
            stderr="",
        )

    result = cli(runner).fit_palette(source, 24)

    assert isinstance(result, RefineFailure)
    assert result.code == "UNSUPPORTED_INPUT"
    assert result.input_path == source


def test_fit_palette_returns_a_refine_failure_when_the_reply_has_no_palette(tmp_path):
    """Round 1 fix: a fully transparent source succeeds with no colour to
    fit at all. Before this fix `_shared_palette`'s `RefinerError` escaped
    `fit_palette` uncaught and cost the whole call it was fitted ahead of;
    now it names this one picture and lets the rest carry on."""
    source = write_tiny_png(tmp_path / "asset0" / "source.png")

    def runner(command, **kwargs):
        item = json.loads(json.dumps(REFINE_PAYLOAD))
        item["id"] = command[command.index("batch") + 1]
        item["status"] = "done"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"items": [item], "summary": {"done": 1, "failed": 0}, "sharedPalette": []}
            ),
            stderr="",
        )

    result = cli(runner).fit_palette(source, 24)

    assert isinstance(result, RefineFailure)
    assert result.input_path == source
    assert result.code
    assert result.message


def test_fit_palette_returns_a_refine_failure_for_an_io_refusal_exit_code(tmp_path):
    """Round 1 fix: an I/O refusal such as INPUT_TOO_LARGE is this picture's
    own answer, not the whole call's — a code and a non-usage exit tell a
    picture's own refusal apart from the call itself never having run."""
    source = write_tiny_png(tmp_path / "asset0" / "source.png")

    def runner(command, **kwargs):
        return SimpleNamespace(
            returncode=EXIT_REFUSED_IO,
            stdout=json.dumps(
                {
                    "failure": {
                        "code": "INPUT_TOO_LARGE",
                        "message": f"{source} is too large.",
                        "hint": "Resize it first.",
                    }
                }
            ),
            stderr="",
        )

    result = cli(runner).fit_palette(source, 24)

    assert isinstance(result, RefineFailure)
    assert result.code == "INPUT_TOO_LARGE"
    assert result.input_path == source


@pytest.mark.skipif(
    not os.environ.get(BINARY_ENV),
    reason=f"set {BINARY_ENV} to the pixel-refiner command to run this",
)
def test_the_real_cli_batches_one_asset_through_its_staging(tmp_path):
    """Opt-in: the only test that proves the staging against the binary.

    Matching the reply item by the staged path rests on the CLI echoing the path
    we handed it *unresolved*, and on macOS that path runs through a
    /var/folders symlink into /private/var. No fake runner can check that
    assumption; this is what will say when it stops holding.
    """
    folder = tmp_path / "asset0"
    folder.mkdir()
    write_tiny_png(folder / "source.png")
    request_ = RefineRequest(
        input_path=folder / "source.png", output_path=folder / "refined.png"
    )

    [result] = PixelRefinerCli().batch([request_])

    assert isinstance(result, RefineResult)
    assert request_.output_path.exists()
    assert result.output.path == request_.output_path
    assert set(result.warnings) <= set(WARNING_CODES)


@pytest.mark.skipif(
    not os.environ.get(BINARY_ENV),
    reason=f"set {BINARY_ENV} to the pixel-refiner command to run this",
)
def test_the_real_cli_refines_a_tiny_png(tmp_path):
    """Opt-in smoke test: the only test that proves the argv against the binary."""
    source = write_tiny_png(tmp_path / "in.png")
    target = tmp_path / "out.png"
    result = PixelRefinerCli().refine(
        RefineRequest(input_path=source, output_path=target)
    )
    assert target.exists()
    assert result.output.path == target
    assert result.route in {"refine", "convert", "preserve"}
    assert set(result.warnings) <= set(WARNING_CODES)


@pytest.mark.skipif(
    not os.environ.get(BINARY_ENV),
    reason=f"set {BINARY_ENV} to the pixel-refiner command to run this",
)
def test_the_real_cli_accepts_a_candidate_id_it_offered(tmp_path):
    """Opt-in: the only test that proves the re-pick against the binary.

    The ids come out of one reply and go straight back into the next command,
    so a change to either half breaks the offer with no other symptom than a
    button that does nothing.
    """
    source = write_tiny_png(tmp_path / "in.png")
    refiner = PixelRefinerCli()
    first = refiner.refine(
        RefineRequest(input_path=source, output_path=tmp_path / "a.png")
    )
    assert first.candidates, "the CLI offered no candidates to re-pick"
    chosen = first.candidates[-1].id
    again = refiner.refine(
        RefineRequest(
            input_path=source, output_path=tmp_path / "b.png", candidate_id=chosen
        )
    )
    assert (tmp_path / "b.png").exists()
    assert again.output.path == (tmp_path / "b.png").resolve()


@pytest.mark.skipif(
    not os.environ.get(BINARY_ENV),
    reason=f"set {BINARY_ENV} to the pixel-refiner command to run this",
)
def test_the_real_cli_lists_a_quick_vocabulary():
    """The settings screen is built from this; an empty answer means no knobs."""
    options = PixelRefinerCli().options()
    assert options.quick, "the CLI listed no quick settings"
    assert "dithering" in options.quick


def write_piece_png(path: Path, size=(68, 128)) -> Path:
    """A blocky two-colour piece, the shape and size of the spec's candle.

    Big enough (68 x 128 source pixels) to be forced down to 17 x 32 art
    pixels at a pitch of 4, with a hard inner block so the refiner has
    edges to keep.
    """
    width, height = size
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[8:-8, 8:-8] = (200, 40, 40, 255)
    pixels[24:-24, 24:-24] = (40, 40, 200, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, "RGBA").save(path)
    return path


@pytest.mark.skipif(
    not os.environ.get(BINARY_ENV),
    reason=f"set {BINARY_ENV} to the pixel-refiner command to run this",
)
def test_the_real_cli_forces_a_piece_to_the_size_it_was_asked_for(tmp_path):
    """Opt-in: the spec's own measurement, kept as a test.

    A forced piece comes back at exactly the size asked for, on the convert
    route at confidence 1.0, with no grid warning (spec "Context", measured
    2026-09-06). If the meaning of `forcePixelsW/H` ever moves — to a pitch,
    say — this is what says so.
    """
    source = write_piece_png(tmp_path / "in.png")
    target = tmp_path / "out.png"
    result = PixelRefinerCli().refine(
        RefineRequest(input_path=source, output_path=target, target=(17, 32))
    )
    assert (result.output.width, result.output.height) == (17, 32)
    with Image.open(target) as image:
        assert image.size == (17, 32)
    assert result.route == "convert"
    assert result.confidence == pytest.approx(1.0)
    assert LOW_GRID_CONFIDENCE not in result.warnings


@pytest.mark.skipif(
    not os.environ.get(BINARY_ENV),
    reason=f"set {BINARY_ENV} to the pixel-refiner command to run this",
)
def test_the_real_cli_fits_a_palette_of_at_most_n_colours(tmp_path):
    """Opt-in: proves `fit_palette` reads a real `sharedPalette`, capped at
    the count asked for, off a genuine one-image batch reply."""
    source = write_tiny_png(tmp_path / "in.png")
    palette = PixelRefinerCli().fit_palette(source, 8)
    assert isinstance(palette, list)
    assert palette
    assert len(palette) <= 8
    assert all(
        isinstance(colour, str) and len(colour) == 7 and colour.startswith("#")
        for colour in palette
    )


@pytest.mark.skipif(
    not os.environ.get(BINARY_ENV),
    reason=f"set {BINARY_ENV} to the pixel-refiner command to run this",
)
def test_the_real_cli_batch_forces_the_target_over_the_callers_own_size(tmp_path):
    """Opt-in: reproduces the failure against the real binary.

    Sending the caller's own `forcePixelsW/H` alongside `--grid force:WxH`
    on one `batch` call exits 2 with INVALID_SETTINGS ("gridDetection cannot
    be combined with the advanced option forcePixelsW") — checked on the
    local checkout on 2026-09-23. `_batch_args` must fold the target into the
    same advanced keys the caller's settings already carry, so the target
    overwrites them and only one size instruction goes out; the
    batch then succeeds and the output lands at the target, not the
    caller's stale size.
    """
    settings = RefinerSettings(advanced={"forcePixelsW": 99, "forcePixelsH": 99})
    request_ = RefineRequest(
        input_path=write_piece_png(tmp_path / "in.png"),
        output_path=tmp_path / "out.png",
        settings=settings,
        target=(17, 32),
    )

    [result] = PixelRefinerCli().batch([request_])

    assert isinstance(result, RefineResult)
    assert request_.output_path.exists()
    assert (result.output.width, result.output.height) == (17, 32)
    with Image.open(request_.output_path) as image:
        assert image.size == (17, 32)


@pytest.mark.skipif(
    not os.environ.get(BINARY_ENV),
    reason=f"set {BINARY_ENV} to the pixel-refiner command to run this",
)
def test_the_real_cli_analyzes_without_writing(tmp_path):
    """Opt-in: `analyze` answers a reading and leaves the folder alone."""
    source = write_piece_png(tmp_path / "in.png")
    reading = PixelRefinerCli().analyze(source)
    assert 0.0 <= reading.confidence <= 1.0
    assert reading.pitch is None or reading.pitch > 0
    assert set(reading.warnings) <= set(WARNING_CODES)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["in.png"]
