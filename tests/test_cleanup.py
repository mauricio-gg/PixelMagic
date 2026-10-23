"""The clean-up job: an enlarged sprite comes back at native size, and repeat runs are cached."""

from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import Direction, Reviewer
from pixelmagic.stages import cleanup, export
from pixelmagic.stages import rebuild as rb
from pixelmagic.stages.cleanup import clean_up
from pixelmagic.stages.generate import import_picture, load_generation
from pixelmagic.store import Store

REAL_FIND_ASEPRITE = export.find_aseprite
NOW = datetime(2026, 10, 20, 9, 0, 0, tzinfo=UTC)
PALETTE = np.array([(200, 40, 40), (40, 160, 60), (50, 70, 200), (240, 220, 80)], np.uint8)


@pytest.fixture(autouse=True)
def no_aseprite(monkeypatch):
    """Do not touch whatever Aseprite the machine happens to have."""
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None)


def _sprite() -> np.ndarray:
    """A 20 x 16 sprite: four colours in a fixed pattern, transparent two-pixel surround."""
    rng = np.random.default_rng(3)
    sprite = np.zeros((16, 20, 4), np.uint8)
    inner = PALETTE[rng.integers(0, 4, (12, 16))]
    sprite[2:14, 2:18, :3] = inner
    sprite[2:14, 2:18, 3] = 255
    return sprite


def _setup(tmp_path, factor=4):
    """A project with the sprite enlarged by `factor` imported as a picture."""
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    big = np.kron(_sprite(), np.ones((factor, factor, 1), np.uint8))
    source = tmp_path / "sprite.png"
    Image.fromarray(big, "RGBA").save(source)
    record = import_picture(store, direction, source, reviewer=Reviewer.HUMAN, now=NOW)
    return store, direction, record.request.id


def test_an_enlarged_sprite_comes_back_pixel_exact_at_native_size(tmp_path):
    store, direction, gid = _setup(tmp_path)
    result = clean_up(store, direction, gid, now=NOW)

    cleaned = np.asarray(Image.open(result.png_path).convert("RGBA"))
    assert np.array_equal(cleaned, _sprite())
    assert result.record.found_pitch_x == pytest.approx(4, abs=0.1)
    assert result.record.found_pitch_y == pytest.approx(4, abs=0.1)
    assert result.record.size_source == "found" and not result.record.unsure
    assert result.record.grid.native_size == (20, 16)
    assert result.aseprite_path is None and "Aseprite was not found" in result.record.notes[0]


def test_a_reading_at_the_top_of_the_finders_range_is_flagged_unsure_but_kept(tmp_path):
    """A reading at the top of the range is flagged even on a true enlargement there.

    That is the agreed cost: on AI pictures the finder cannot tell it from the wrong peak.
    The found size is kept, and a big in-range enlargement (12x) stays sure.
    """
    readings = {}
    for factor in (12, 13.9):
        store = Store(tmp_path / f"project-{factor}")
        store.create_project("Game", now=NOW)
        direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
        size = (round(20 * factor), round(16 * factor))
        source = tmp_path / f"sprite-{factor}.png"
        Image.fromarray(_sprite(), "RGBA").resize(size, Image.NEAREST).save(source)
        gid = import_picture(store, direction, source, reviewer=Reviewer.HUMAN, now=NOW).request.id
        readings[factor] = clean_up(store, direction, gid, now=NOW).record

    assert not readings[12].unsure
    top = readings[13.9]
    assert top.unsure and top.size_source == "found"
    assert top.pixel_size == top.found_pitch_x == pytest.approx(13.875, abs=0.1)


def test_a_typed_size_overrides_the_found_one_but_the_found_one_is_still_reported(tmp_path):
    store, direction, gid = _setup(tmp_path)
    result = clean_up(store, direction, gid, size=8, now=NOW)

    assert result.record.size_source == "typed"
    assert result.record.grid.pitch_x == 8
    assert result.record.found_pitch_x == pytest.approx(4, abs=0.1)
    assert result.record.grid.native_size == (10, 8)


def test_a_typed_true_size_gives_the_sprite_pixel_exact(tmp_path):
    store, direction, gid = _setup(tmp_path)
    result = clean_up(store, direction, gid, size=4, now=NOW)

    cleaned = np.asarray(Image.open(result.png_path).convert("RGBA"))
    assert np.array_equal(cleaned, _sprite())


def test_a_run_stopped_after_the_new_picture_is_not_served_as_the_old_settings(
    tmp_path, monkeypatch
):
    store, direction, gid = _setup(tmp_path)
    a = clean_up(store, direction, gid, now=NOW)
    a_bytes = a.png_path.read_bytes()

    def stop(*args):
        raise KeyboardInterrupt  # not caught by the Aseprite handler: the run just ends

    with monkeypatch.context() as stopped:
        stopped.setattr(cleanup, "_write_aseprite", stop)
        with pytest.raises(KeyboardInterrupt):
            clean_up(store, direction, gid, colours=2, now=NOW)

    again = clean_up(store, direction, gid, now=NOW)
    assert not again.cache_hit
    assert again.png_path.read_bytes() == a_bytes


def test_a_colour_count_is_honoured(tmp_path):
    store, direction, gid = _setup(tmp_path)
    result = clean_up(store, direction, gid, colours=2, now=NOW)

    assert result.record.colour_count == 2
    cleaned = np.asarray(Image.open(result.png_path).convert("RGBA"))
    assert len(np.unique(cleaned[cleaned[..., 3] == 255][:, :3], axis=0)) == 2


def test_a_repeat_run_is_a_cache_hit_and_a_changed_setting_is_not(tmp_path, monkeypatch):
    store, direction, gid = _setup(tmp_path)
    first = clean_up(store, direction, gid, now=NOW)
    first_bytes = first.png_path.read_bytes()

    # If the second run computed anything, this would raise.
    monkeypatch.setattr(cleanup, "_sample_cells", lambda *a: pytest.fail("recomputed"))
    second = clean_up(store, direction, gid, now=NOW)
    assert second.cache_hit and not first.cache_hit
    assert second.png_path.read_bytes() == first_bytes
    assert second.record == first.record

    with pytest.raises(pytest.fail.Exception):  # a changed setting must reach the maths
        clean_up(store, direction, gid, colours=2, now=NOW)


@pytest.mark.parametrize("size", [0.5, 10_000])
def test_an_impossible_size_is_refused_before_anything_is_written(tmp_path, size):
    store, direction, gid = _setup(tmp_path)
    with pytest.raises(ValueError, match="pixel size"):
        clean_up(store, direction, gid, size=size, now=NOW)
    assert not store.cleanup_dir(direction.name, gid).exists()


def _palette_file(tmp_path, monkeypatch):
    """Make a real .aseprite of the true sprite; skip when this machine has no Aseprite.

    Leaves the real `find_aseprite` in place, so `clean_up` can read the file.
    """
    binary = REAL_FIND_ASEPRITE()
    if binary is None:
        pytest.skip("Aseprite is not installed")
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: binary)
    png = tmp_path / "true.png"
    Image.fromarray(_sprite(), "RGBA").save(png)
    out = tmp_path / "true.aseprite"
    export.AsepriteRunner(binary).build(png, out, cell=16)
    return out


def _nudged_setup(tmp_path):
    """The sprite enlarged, every art pixel's colour nudged by a few levels (blurry source)."""
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    sprite = _sprite()
    noise = np.random.default_rng(5).integers(-6, 7, sprite.shape[:2] + (3,))
    sprite[..., :3] = np.clip(sprite[..., :3].astype(int) + noise, 0, 255)
    big = np.kron(sprite, np.ones((4, 4, 1), np.uint8))
    source = tmp_path / "nudged.png"
    Image.fromarray(big, "RGBA").save(source)
    record = import_picture(store, direction, source, reviewer=Reviewer.HUMAN, now=NOW)
    return store, direction, record.request.id


def test_a_palette_file_gives_back_exactly_its_colours(tmp_path, monkeypatch):
    palette_file = _palette_file(tmp_path, monkeypatch)
    store, direction, gid = _nudged_setup(tmp_path)

    plain = clean_up(store, direction, gid, now=NOW)
    assert plain.record.colour_count > len(PALETTE)  # the plain reduction keeps the wrong colours

    result = clean_up(store, direction, gid, palette_file=palette_file, now=NOW)

    cleaned = np.asarray(Image.open(result.png_path).convert("RGBA"))
    assert np.array_equal(cleaned, _sprite())
    assert result.record.palette_file == "true.aseprite"
    assert result.record.palette == sorted(f"{r:02x}{g:02x}{b:02x}" for r, g, b in PALETTE)
    assert not result.cache_hit  # a palette is part of the cache key


def test_reusing_the_last_palette_repeats_a_palette_run_without_the_file(tmp_path, monkeypatch):
    """The page cannot refill a file input, so its repeat sends no file; the core must cope.

    The palette is handed over by a stand-in reader: what is under test is the reuse, not
    Aseprite, so this runs on any machine.
    """
    monkeypatch.setattr(cleanup, "read_palette", lambda _file: PALETTE)
    store, direction, gid = _nudged_setup(tmp_path)
    first = clean_up(store, direction, gid, palette_file=tmp_path / "mine.aseprite", now=NOW)

    again = clean_up(store, direction, gid, reuse_last_palette=True, now=NOW)
    assert again.cache_hit
    assert again.png_path.read_bytes() == first.png_path.read_bytes()

    resized = clean_up(store, direction, gid, size=4, reuse_last_palette=True, now=NOW)
    assert not resized.cache_hit
    assert resized.record.palette_file == "mine.aseprite"
    assert resized.record.palette == first.record.palette
    shown = {tuple(int(h[i : i + 2], 16) for i in (0, 2, 4)) for h in resized.record.shown_colours}
    assert shown <= {tuple(int(c) for c in colour) for colour in PALETTE}


def test_a_palette_and_a_colour_count_are_refused_before_anything_is_written(tmp_path):
    store, direction, gid = _setup(tmp_path)
    with pytest.raises(ValueError, match="either a colour count or a palette"):
        clean_up(store, direction, gid, colours=3, palette_file=tmp_path / "x.aseprite", now=NOW)
    with pytest.raises(ValueError, match="Aseprite is needed"):  # autouse: no Aseprite found
        clean_up(store, direction, gid, palette_file=tmp_path / "x.aseprite", now=NOW)
    assert not store.cleanup_dir(direction.name, gid).exists()


def _soft_tall_cells_setup(tmp_path):
    """A 16 x 12 sprite drawn in cells 5 wide and 7 tall, softened sideways.

    A 3-tap blur along x leaves the row edges sharp, so the finder reads 4.975 (score 0.59)
    across and 7.0 (score 1.0) down: both above `rebuild.FOUND_BAR`, readings 1.41 apart.
    The picture is 100 wide and 112 tall.
    """
    rng = np.random.default_rng(0)
    cells = PALETTE[rng.integers(0, 4, (12, 16))]
    for x in range(1, 16):  # runs along x, so the column edges are weak
        keep = rng.random(12) < 0.6
        cells[keep, x] = cells[keep, x - 1]
    sprite = np.zeros((16, 20, 4), np.uint8)
    sprite[2:14, 2:18, :3], sprite[2:14, 2:18, 3] = cells, 255
    big = np.kron(sprite, np.ones((7, 5, 1), np.uint8)).astype(float)
    padded = np.pad(big[..., :3], ((0, 0), (1, 1), (0, 0)), mode="edge")
    big[..., :3] = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3
    source = tmp_path / "soft.png"
    Image.fromarray(big.round().astype(np.uint8), "RGBA").save(source)
    store = Store(tmp_path / "project")
    store.create_project("Game", now=NOW)
    direction = store.create_direction(Direction(name="den", created=NOW, style="x"))
    return store, direction, import_picture(
        store, direction, source, reviewer=Reviewer.HUMAN, now=NOW
    ).request.id


def test_readings_that_differ_give_square_pixels_at_the_smaller_reading(tmp_path):
    """catches: a squashed result (non-square pixels) when the two axes read differently."""
    store, direction, gid = _soft_tall_cells_setup(tmp_path)
    record = clean_up(store, direction, gid, now=NOW).record

    assert record.found_pitch_x == pytest.approx(4.975, abs=0.05)
    assert record.found_pitch_y == pytest.approx(7.0, abs=0.05)
    assert record.found_score_y > record.found_score_x
    assert record.pixel_size == pytest.approx(record.found_pitch_x)
    assert record.grid.pitch_x == record.grid.pitch_y == record.pixel_size
    # Both sides of the 100 x 112 picture are divided by the one size (20.1 x 22.5), so the
    # proportions hold; each side may gain one cell from the grid's phase.
    width, height = record.grid.native_size
    assert width in (20, 21) and height in (22, 23)


def test_readings_that_disagree_are_recorded_unsure_even_when_both_scores_are_good(tmp_path):
    """catches: readings far apart (here 4.975 vs 7.0) with good scores being recorded as sure."""
    store, direction, gid = _soft_tall_cells_setup(tmp_path)
    record = clean_up(store, direction, gid, now=NOW).record

    assert min(record.found_score_x, record.found_score_y) > rb.FOUND_BAR
    assert max(record.found_pitch_x, record.found_pitch_y) > (
        cleanup.READINGS_DISAGREE * min(record.found_pitch_x, record.found_pitch_y)
    )
    assert record.unsure


def test_a_picture_with_no_solid_pixel_is_refused_and_leaves_nothing_behind(tmp_path):
    store, direction, gid = _setup(tmp_path)
    record = load_generation(store, direction.name, gid)
    Image.fromarray(np.zeros((16, 20, 4), np.uint8), "RGBA").save(record.image_path)
    with pytest.raises(ValueError, match="no solid pixels"):
        clean_up(store, direction, gid, now=NOW)
    assert not store.cleanup_dir(direction.name, gid).exists()
