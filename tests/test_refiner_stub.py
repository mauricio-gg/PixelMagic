import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import GridCandidate, GridOffset, RefinerSettings
from pixelmagic.refiner.base import (
    EXIT_REFUSED_IO,
    LOW_GRID_CONFIDENCE,
    PRESETS,
    WARNING_CODES,
    Analysis,
    RefineFailure,
    RefineRequest,
    RefinerError,
    RefineResult,
    validate_settings,
)
from pixelmagic.refiner.stub import StubRefiner


def make_png(path, size=(8, 8)):
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[2:6, 2:6] = (10, 20, 200, 255)
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def test_the_stub_copies_the_input_to_the_output(tmp_path):
    source = make_png(tmp_path / "in.png")
    target = tmp_path / "out.png"
    StubRefiner().refine(
        RefineRequest(input_path=source, output_path=target, settings=RefinerSettings())
    )
    assert target.read_bytes() == source.read_bytes()


def test_the_stub_reports_the_preserve_route_and_says_so_in_a_warning(tmp_path):
    source = make_png(tmp_path / "in.png")
    result = StubRefiner().refine(
        RefineRequest(
            input_path=source, output_path=tmp_path / "out.png", settings=RefinerSettings()
        )
    )
    assert isinstance(result, RefineResult)
    assert result.route == "preserve"
    assert "FALLBACK_TO_PRESERVE" in result.warnings
    assert set(result.warnings) <= WARNING_CODES


def test_the_stub_measures_the_real_output(tmp_path):
    source = make_png(tmp_path / "in.png", size=(12, 9))
    result = StubRefiner().refine(
        RefineRequest(
            input_path=source, output_path=tmp_path / "out.png", settings=RefinerSettings()
        )
    )
    assert (result.output.width, result.output.height) == (12, 9)
    assert result.output.color_count == 1
    assert result.output.palette == ["#0a14c8"]


def test_the_stub_echoes_the_requested_settings_as_effective_options(tmp_path):
    source = make_png(tmp_path / "in.png")
    settings = RefinerSettings(
        preset="crisp-sprite", quick={"dithering": "off"}, advanced={"trim": True}
    )
    result = StubRefiner().refine(
        RefineRequest(input_path=source, output_path=tmp_path / "out.png", settings=settings)
    )
    assert result.effective_options["preset"] == "crisp-sprite"
    assert result.effective_options["dithering"] == "off"
    assert result.effective_options["trim"] is True


def test_the_stub_honours_the_batch_output_contract(tmp_path):
    """The request's own output_path holds the file, and the result agrees."""
    request = RefineRequest(
        input_path=make_png(tmp_path / "in.png"),
        output_path=tmp_path / "out" / "asset" / "refined.png",
        settings=RefinerSettings(),
    )
    [result] = StubRefiner().batch([request])
    assert isinstance(result, RefineResult)
    assert request.output_path.exists()
    assert result.output.path == request.output_path


def test_the_stub_refuses_to_clobber_without_overwrite(tmp_path):
    source = make_png(tmp_path / "in.png")
    target = tmp_path / "out.png"
    target.write_bytes(b"already here")
    with pytest.raises(RefinerError) as caught:
        StubRefiner().refine(
            RefineRequest(input_path=source, output_path=target, overwrite=False)
        )
    assert caught.value.exit_code == EXIT_REFUSED_IO
    # The same code the real CLI names, so a caller can branch on it either way.
    assert caught.value.code == "OUTPUT_EXISTS"
    assert target.read_bytes() == b"already here"


def test_the_stub_refuses_to_clobber_in_a_batch_either(tmp_path):
    """The adapter refuses here too; a stand-in that did not would hide it."""
    target = tmp_path / "out.png"
    target.write_bytes(b"already here")
    with pytest.raises(RefinerError) as caught:
        StubRefiner().batch(
            [
                RefineRequest(
                    input_path=make_png(tmp_path / "in.png"),
                    output_path=target,
                    overwrite=False,
                )
            ]
        )
    assert caught.value.exit_code == EXIT_REFUSED_IO
    assert caught.value.code == "OUTPUT_EXISTS"
    assert target.read_bytes() == b"already here"


def test_the_stub_admits_it_knows_no_palette_ids():
    assert StubRefiner().options().palettes == []


def test_the_stub_lists_the_contract_presets():
    assert set(StubRefiner().options().presets) == PRESETS


def test_a_missing_input_raises_the_same_error_the_real_cli_would(tmp_path):
    with pytest.raises(RefinerError) as raised:
        StubRefiner().refine(
            RefineRequest(
                input_path=tmp_path / "nope.png",
                output_path=tmp_path / "out.png",
                settings=RefinerSettings(),
            )
        )
    assert raised.value.code == "INPUT_NOT_FOUND"
    assert raised.value.exit_code == EXIT_REFUSED_IO


def test_a_missing_input_in_a_batch_is_a_per_item_failure(tmp_path):
    """A missing input comes back as a failure, as the real CLI reports it, not a raise."""
    missing_target = tmp_path / "out" / "missing.png"
    results = StubRefiner().batch(
        [RefineRequest(input_path=tmp_path / "nope.png", output_path=missing_target)]
    )
    assert len(results) == 1
    assert isinstance(results[0], RefineFailure)
    assert results[0].code == "INPUT_NOT_FOUND"
    assert not missing_target.exists()


def test_background_pick_without_a_colour_is_rejected():
    with pytest.raises(ValueError, match="backgroundColor"):
        validate_settings(RefinerSettings(quick={"background": "pick"}))


def test_background_pick_with_a_colour_is_accepted():
    validate_settings(
        RefinerSettings(quick={"background": "pick", "backgroundColor": "#112233"})
    )


def test_low_grid_confidence_is_a_known_warning_code():
    assert LOW_GRID_CONFIDENCE in WARNING_CODES


@pytest.mark.parametrize("scale", [0, -4, 33])
def test_an_out_of_range_scale_is_refused(scale):
    """Guardian M2.2 #2: `--scale 1..32` is the CLI's own contract (quoted at
    the top of `refiner/base.py`); a scale outside it must never reach
    `direction.json` as a stored default."""
    with pytest.raises(ValueError, match="32"):
        validate_settings(RefinerSettings(scale=scale))


@pytest.mark.parametrize("scale", [1, 32])
def test_the_floor_and_the_ceiling_of_scale_are_accepted(scale):
    validate_settings(RefinerSettings(scale=scale))


def test_a_request_has_no_target_unless_one_is_asked_for(tmp_path):
    plain = RefineRequest(input_path=tmp_path / "in.png", output_path=tmp_path / "out.png")
    assert plain.target is None
    forced = RefineRequest(
        input_path=tmp_path / "in.png", output_path=tmp_path / "out.png", target=(17, 32)
    )
    assert forced.target == (17, 32)


def test_an_analysis_is_a_reading_with_nothing_written():
    reading = Analysis(
        pitch=4.25,
        confidence=0.71,
        warnings=[LOW_GRID_CONFIDENCE],
        grid_candidates=[
            GridCandidate(
                out_w=17, out_h=32, cell_w=4.0, cell_h=4.5, offset=GridOffset(x=0, y=0),
                method="autocorrelation", total_score=0.8, confidence=0.71,
            )
        ],
    )
    assert reading.pitch == 4.25
    assert reading.grid_candidates[0].cell_h == 4.5
    # A reading may honestly have no pitch at all.
    assert Analysis(pitch=None, confidence=0.0, warnings=[], grid_candidates=[]).pitch is None


def test_the_stub_reads_a_pitch_of_one_off_anything(tmp_path):
    """Locked decision 9 in practice: offline, every prop's target is its own box."""
    source = make_png(tmp_path / "in.png", size=(68, 128))
    reading = StubRefiner().analyze(source)
    assert reading == Analysis(pitch=1.0, confidence=1.0, warnings=[], grid_candidates=[])


def test_the_stub_analyze_writes_nothing(tmp_path):
    source = make_png(tmp_path / "in.png")
    StubRefiner().analyze(source, RefinerSettings(preset="crisp-sprite"))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["in.png"]


def test_the_stub_analyze_refuses_a_missing_input_the_way_the_cli_would(tmp_path):
    with pytest.raises(RefinerError) as caught:
        StubRefiner().analyze(tmp_path / "nowhere.png")
    assert caught.value.exit_code == EXIT_REFUSED_IO
    assert caught.value.code == "INPUT_NOT_FOUND"


def test_the_stub_analyze_checks_the_settings_first(tmp_path):
    source = make_png(tmp_path / "in.png")
    with pytest.raises(ValueError, match="backgroundColor"):
        StubRefiner().analyze(source, RefinerSettings(quick={"background": "pick"}))


def test_the_stub_ignores_a_target_and_copies_the_pixels_through(tmp_path):
    """The stand-in is not taught to resize (locked decision 9)."""
    source = make_png(tmp_path / "in.png", size=(12, 9))
    target = tmp_path / "out.png"
    result = StubRefiner().refine(
        RefineRequest(input_path=source, output_path=target, target=(4, 3))
    )
    assert (result.output.width, result.output.height) == (12, 9)
    assert target.read_bytes() == source.read_bytes()
    assert result.route == "preserve"


def test_the_stub_batch_ignores_targets_too(tmp_path):
    request = RefineRequest(
        input_path=make_png(tmp_path / "in.png", size=(12, 9)),
        output_path=tmp_path / "out" / "refined.png",
        target=(4, 3),
    )
    [result] = StubRefiner().batch([request])
    assert (result.output.width, result.output.height) == (12, 9)
