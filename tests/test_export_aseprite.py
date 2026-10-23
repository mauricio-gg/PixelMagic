import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from pixelmagic.model import AssetKind, AssetRecord, Direction, RecipeRecord, RefineReport
from pixelmagic.stages.export import (
    ASEPRITE_ENV_VAR,
    AsepriteRunner,
    export,
    find_aseprite,
)
from pixelmagic.store import Store

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


class FakeAsepriteRun:
    """Stands in for subprocess.run, writing the output file Aseprite would."""

    def __init__(self, returncode=0, write_output=True):
        self.returncode = returncode
        self.write_output = write_output
        self.commands: list[list[str]] = []
        self.environments: list[dict] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        self.environments.append(kwargs.get("env") or {})
        if self.write_output:
            for part in command:
                if part.startswith("job="):
                    job = json.loads(Path(part[4:]).read_text())
                    Path(job["output"]).write_bytes(b"fake-aseprite")
        return SimpleNamespace(returncode=self.returncode, stdout="", stderr="nope")


@pytest.fixture
def setup(tmp_path, write_asset):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    write_asset(store, "wooden-chair", AssetKind.PROP)
    return store, direction


def test_find_aseprite_prefers_the_environment_variable(tmp_path, monkeypatch):
    binary = tmp_path / "aseprite"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv(ASEPRITE_ENV_VAR, str(binary))
    assert find_aseprite() == binary


def test_find_aseprite_ignores_a_path_that_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setenv(ASEPRITE_ENV_VAR, str(tmp_path / "nope"))
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("pixelmagic.stages.export.MACOS_APP_BINARY", tmp_path / "also-nope")
    assert find_aseprite() is None


def test_the_runner_builds_the_documented_command(tmp_path):
    run = FakeAsepriteRun()
    AsepriteRunner(Path("/bin/aseprite"), runner=run).build(
        tmp_path / "in.png", tmp_path / "out.aseprite", cell=16
    )
    command = run.commands[0]
    assert command[0] == "/bin/aseprite"
    assert command[1] == "-b"
    assert command[2] == "--script-param"
    assert command[3].startswith("job=")
    # Every --script-param must come before --script or Aseprite drops it.
    assert command[4:6] == ["--script-param", "cell=16"]
    assert command[6] == "--script"
    assert command[7].endswith("single_sprite.lua")


def test_the_runner_points_aseprite_at_a_scratch_user_folder(tmp_path):
    run = FakeAsepriteRun()
    AsepriteRunner(Path("/bin/aseprite"), runner=run).build(
        tmp_path / "in.png", tmp_path / "out.aseprite", cell=16
    )
    assert "ASEPRITE_USER_FOLDER" in run.environments[0]


def test_a_non_zero_exit_raises(tmp_path):
    run = FakeAsepriteRun(returncode=1, write_output=False)
    with pytest.raises(RuntimeError, match="nope"):
        AsepriteRunner(Path("/bin/aseprite"), runner=run).build(
            tmp_path / "in.png", tmp_path / "out.aseprite", cell=16
        )


def test_a_silently_missing_output_raises(tmp_path):
    run = FakeAsepriteRun(write_output=False)
    with pytest.raises(RuntimeError, match="wrote no file"):
        AsepriteRunner(Path("/bin/aseprite"), runner=run).build(
            tmp_path / "in.png", tmp_path / "out.aseprite", cell=16
        )


def test_export_writes_an_aseprite_file_per_asset(setup, tmp_path):
    store, direction = setup
    runner = AsepriteRunner(Path("/bin/aseprite"), runner=FakeAsepriteRun())
    bundle = export(store, direction, stamp="s", now=NOW, aseprite=runner)
    assert (bundle.directory / "wooden-chair.aseprite").exists()
    entry = bundle.manifest.entries[0]
    assert entry.files["aseprite"] == "wooden-chair.aseprite"
    assert bundle.warnings == []


def test_export_without_aseprite_warns_once_and_still_ships_the_pngs(setup, monkeypatch):
    store, direction = setup
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None)
    bundle = export(store, direction, stamp="s", now=NOW)
    assert (bundle.directory / "wooden-chair.png").exists()
    assert not (bundle.directory / "wooden-chair.aseprite").exists()
    assert len([w for w in bundle.warnings if "Aseprite" in w]) == 1
    assert "aseprite" not in bundle.manifest.entries[0].files


def test_an_aseprite_failure_warns_rather_than_failing_the_export(setup, write_asset):
    store, direction = setup
    # A second asset, so "the others still export" is actually asserted: with
    # one asset a raise and a warning look identical from the bundle.
    write_asset(store, "round-rug", AssetKind.PROP)
    runner = AsepriteRunner(
        Path("/bin/aseprite"), runner=FakeAsepriteRun(returncode=1, write_output=False)
    )
    bundle = export(store, direction, stamp="s", now=NOW, aseprite=runner)
    assert (bundle.directory / "wooden-chair.png").exists()
    assert (bundle.directory / "round-rug.png").exists()
    assert [entry.name for entry in bundle.manifest.entries] == ["round-rug", "wooden-chair"]
    assert any("wooden-chair" in warning for warning in bundle.warnings)
    assert any("round-rug" in warning for warning in bundle.warnings)


def test_export_hands_aseprite_the_direction_s_cell_size(tmp_path, write_asset):
    """The runner gets the direction's own cell size, not a default, and it
    arrives as a script param before --script (Aseprite drops params given
    after it)."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    direction = store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.", cell_size=24)
    )
    write_asset(store, "wooden-chair", AssetKind.PROP)
    run = FakeAsepriteRun()
    runner = AsepriteRunner(Path("/bin/aseprite"), runner=run)
    export(store, direction, stamp="s", now=NOW, aseprite=runner)
    command = run.commands[0]
    assert "cell=24" in command
    assert command.index("cell=24") < command.index("--script")


def test_a_half_written_aseprite_file_is_removed_from_the_bundle(setup):
    """A file the manifest does not list must not sit in the bundle."""
    store, direction = setup

    class HalfWrite(FakeAsepriteRun):
        """Writes a truncated .aseprite and then reports failure, as a crash would."""

        def __call__(self, command, **kwargs):
            for part in command:
                if part.startswith("job="):
                    job = json.loads(Path(part[4:]).read_text())
                    Path(job["output"]).write_bytes(b"\x00\x00")
            return SimpleNamespace(returncode=1, stdout="", stderr="crashed")

    runner = AsepriteRunner(Path("/bin/aseprite"), runner=HalfWrite())
    bundle = export(store, direction, stamp="s", now=NOW, aseprite=runner)
    assert not (bundle.directory / "wooden-chair.aseprite").exists()
    assert "aseprite" not in bundle.manifest.entries[0].files
    assert (bundle.directory / "wooden-chair.png").exists()


# --- the master .aseprite, with the real program ---------------------------

real_aseprite = pytest.mark.skipif(find_aseprite() is None, reason="Aseprite is not installed")

PLACES = ("nw", "ne", "sw", "se")


def stage_surface(store):
    """Stage a recipe surface's 32 tiles, each a different flat colour with one marker pixel."""
    for i in range(32):
        variant = i // 4 + 1
        place = PLACES[i % 4]
        name = f"floor-v{variant}-{place}"
        folder = store.asset_dir("cosy-den", name)
        folder.mkdir(parents=True)
        tile = Image.new("RGBA", (16, 16), (i * 7, 40 + i, 90, 255))
        tile.putpixel((i % 16, 0), (255, 255, 255, 255))
        tile.save(folder / "source.png")
        tile.save(folder / "refined.png")
        store.write_model(
            folder / "asset.json",
            AssetRecord(
                name=name, kind=AssetKind.TILE, source_generation="20260903-191233",
                variant=variant,
                recipe=RecipeRecord(
                    kind="wood", layout="block", place=place,
                    surface="floor", point=(1, 1), segmenter="stub",
                ),
            ),
        )
        store.write_model(
            folder / "refine.json",
            RefineReport(
                asset=name, route="recipe", confidence=1.0, refiner="recipe",
                palette=["#000000"], width=16, height=16, color_count=1,
            ),
        )


def edit_master(master, tile_index, script_dir, *, add_tile=False):
    """Change two pixels of one tile of the master's tileset (or add a tile), in real Aseprite."""
    script = script_dir / "edit.lua"
    script.write_text(
        f"""
local sprite = app.open("{master}")
local tileset
for _, layer in ipairs(sprite.layers) do
  if layer.isTilemap then tileset = layer.tileset end
end
if {str(add_tile).lower()} then
  sprite:newTile(tileset, #tileset)
else
  local tile = tileset:tile({tile_index})
  local image = tile.image:clone()
  image:putPixel(3, 5, app.pixelColor.rgba(1, 2, 3, 255))
  image:putPixel(4, 5, app.pixelColor.rgba(4, 5, 6, 255))
  tile.image = image
end
sprite:saveAs("{master}")
"""
    )
    subprocess.run(
        [str(find_aseprite()), "-b", "--script", str(script)],
        check=True,
        env={**os.environ, "ASEPRITE_USER_FOLDER": str(script_dir)},
    )


def pixels(path):
    with Image.open(path) as image:
        return list(image.convert("RGBA").get_flattened_data())


@real_aseprite
def test_an_edit_in_the_master_reaches_the_bundle_and_only_that_edit(setup, tmp_path):
    """The master's tiles, not the recipe's, make the bundle: two pixels painted in one tile
    come out at every scale in exactly that tile (the second block's proves the order)."""
    store, direction = setup
    edited = "floor-v2-ne"
    stage_surface(store)
    first = export(store, direction, stamp="one", now=NOW)
    master = store.surface_master("cosy-den", "floor")
    assert master.exists()
    edit_master(master, 6, tmp_path)  # block 2's ne is the sixth tile

    second = export(store, direction, stamp="two", now=NOW)

    for entry in first.manifest.entries:
        if entry.recipe is None:
            continue
        old = pixels(first.directory / entry.files["native"])
        new = pixels(second.directory / entry.files["native"])
        if entry.name == edited:
            changed = {i for i in range(256) if old[i] != new[i]}
            assert changed == {5 * 16 + 3, 5 * 16 + 4}
            assert new[5 * 16 + 3] == (1, 2, 3, 255)
            assert pixels(second.directory / entry.files["x2"])[(10 * 32) + 6] == (1, 2, 3, 255)
        else:
            assert old == new
            assert pixels(first.directory / entry.files["x4"]) == pixels(
                second.directory / entry.files["x4"]
            )
    assert (second.directory / "floor.aseprite").read_bytes() == master.read_bytes()
    new_entry = next(e for e in second.manifest.entries if e.name == edited)
    assert "#010203" in new_entry.palette


@real_aseprite
def test_a_master_with_the_wrong_tile_count_warns_and_the_bundle_uses_the_recipe_tiles(
    setup, tmp_path
):
    """A master with a ninth tile must not put a mismatched picture in the bundle:
    one warning names the file and the count, and the pictures are the recipe's own."""
    store, direction = setup
    stage_surface(store)
    first = export(store, direction, stamp="one", now=NOW)
    master = store.surface_master("cosy-den", "floor")
    edit_master(master, 0, tmp_path, add_tile=True)
    broken = master.read_bytes()

    second = export(store, direction, stamp="two", now=NOW)

    assert [w for w in second.warnings if str(master) in w and "holds 33 tiles, expected 32" in w]
    for entry in first.manifest.entries:
        assert pixels(first.directory / entry.files["native"]) == pixels(
            second.directory / entry.files["native"]
        )
    assert master.read_bytes() == broken
