"""The room file export writes beside a kept scene: what it holds and where."""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    Cutout,
    CutoutOrigin,
    Direction,
    RecipeRecord,
    RefineReport,
    SheetKind,
)
from pixelmagic.stages.export import AsepriteRunner, export, find_aseprite
from pixelmagic.stages.recipes import PLACES, room_layout
from pixelmagic.stages.room_preview import preview_room, room_from_bundle
from pixelmagic.store import Store

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)
PICTURE = "20260903-191233"
DECOMPOSITION = "20260903-200000"
CELL = 16
#: The scene is 128 x 128 and its props were drawn at 2 original pixels per room pixel,
#: so the room is 64 x 64: a 4 x 4 map of 16-pixel cells.
#: Deliberately not the real ratio (box over picture is 2): the room must ignore it.
STALE_ORIGIN_SCALE = 3.0


class CapturingRun:
    """Stands in for subprocess.run: keeps each job file and writes the output Aseprite would."""

    def __init__(self):
        self.jobs: list[dict] = []

    def __call__(self, command, **kwargs):
        for part in command:
            if part.startswith("job="):
                job = json.loads(Path(part[4:]).read_text())
                self.jobs.append(job)
                Path(job["output"]).write_bytes(b"fake-aseprite")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def room(self) -> dict | None:
        """The one room job, or None when no room was asked for."""
        rooms = [job for job in self.jobs if "surfaces" in job]
        assert len(rooms) <= 1
        return rooms[0] if rooms else None


def stage(store, name, kind, *, size=(8, 8), recipe=None, variant=None, decomposition=None,
          picture=PICTURE, source_size=None):
    """Write one finished asset (and its refine report) the way export finds it."""
    folder = store.asset_dir("cosy-den", name)
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, (10, 20, 200, 255)).save(folder / "refined.png")
    Image.new("RGBA", source_size or size, (10, 20, 200, 255)).save(folder / "source.png")
    store.write_model(
        folder / "asset.json",
        AssetRecord(
            name=name, kind=kind, source_generation=picture, recipe=recipe, variant=variant,
            decomposition=decomposition, pitch=3.0 if kind is AssetKind.SCENE else None,
        ),
    )
    store.write_model(
        folder / "refine.json",
        RefineReport(
            asset=name, route="preserve", confidence=1.0, refiner="stub",
            palette=["#0a14c8"], width=size[0], height=size[1], color_count=1,
        ),
    )


def stage_prop(store, name, box, **kwargs):
    """A kept object-road prop whose decomposition says where it sat on the picture."""
    stage(store, name, AssetKind.PROP, decomposition=DECOMPOSITION, **kwargs)
    cutout = Cutout(
        name=name, sheet="props", kind=SheetKind.PROPS, bbox=box,
        origin=CutoutOrigin(
            pick=1, group="group-01", cell=(0, 0), picture_box=box,
            scale=STALE_ORIGIN_SCALE, segmenter="stub",
        ),
    )
    store.write_model(store.cutout_dir("cosy-den", DECOMPOSITION, name) / "cutout.json", cutout)


def stage_floor(store, box, hole=None):
    """A whole recipe surface "floor" whose outline fills `box` on the picture.

    `hole` is a box (in picture pixels) the outline leaves out, as it does around a rug.
    """
    x0, y0, x1, y1 = box
    for i in range(32):
        place = PLACES[i % 4]
        name = f"floor-v{i // 4 + 1}-{place}"
        recipe = RecipeRecord(
            kind="wood", layout="block", place=place, surface="floor", point=(x0, y0),
            segmenter="stub",
        )
        stage(
            store, name, AssetKind.TILE, size=(CELL, CELL), recipe=recipe, variant=i // 4 + 1,
            decomposition=DECOMPOSITION,
        )
    # Every tile's cutout carries the outline's box, as `decompose` writes them.
    first = store.cutout_dir("cosy-den", DECOMPOSITION, "floor-v1-nw")
    store.write_model(
        first / "cutout.json",
        Cutout(name="floor-v1-nw", sheet="floor", kind=SheetKind.TILES, bbox=box),
    )
    outline = store.surface_dir("cosy-den", DECOMPOSITION, "floor")
    outline.mkdir(parents=True)
    mask = Image.new("L", (x1 - x0, y1 - y0), 255)
    if hole:
        mask.paste(0, (hole[0] - x0, hole[1] - y0, hole[2] - x0, hole[3] - y0))
    mask.save(outline / "outline.png")


@pytest.fixture
def store(tmp_path):
    """A project with a direction holding a kept 128 x 128 scene."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    store.create_direction(Direction(name="cosy-den", created=NOW, style="Warm.", cell_size=CELL))
    stage(store, "den", AssetKind.SCENE, source_size=(128, 128))
    return store


def plan_for(store):
    """The room plan export hands to Aseprite for the kept scene, caught on its way."""
    plans = []

    class Catching(AsepriteRunner):
        def build_room(self, plan, output_path, *, cell):
            plans.append(plan)
            super().build_room(plan, output_path, cell=cell)

    export(
        store, store.load_direction("cosy-den"), stamp="s", now=NOW,
        aseprite=Catching(Path("/bin/aseprite"), runner=CapturingRun()),
    )
    [plan] = plans
    return plan


def run_export(store, **kwargs):
    """Export the direction with a runner that records its jobs."""
    runner = CapturingRun()
    bundle = export(
        store, store.load_direction("cosy-den"), stamp="s", now=NOW,
        aseprite=AsepriteRunner(Path("/bin/aseprite"), runner=runner), **kwargs,
    )
    return bundle, runner


def test_the_room_puts_a_prop_where_it_sat_and_paints_whole_blocks_only_inside_the_outline(store):
    """A prop lands bottom-centre on its box; tiles cover the outline and what it encloses."""
    stage_prop(store, "chair", (24, 16, 40, 32))
    # The outline's box spans picture x 0-96, y 64-128: cell centres at x 16, 48, 80 and y 80, 112.
    # A hole the outline fully encloses (a rug the finder left out) around cell (3, 1)'s centre.
    stage_floor(store, (0, 64, 96, 128), hole=(40, 100, 60, 124))

    bundle, runner = run_export(store)

    room = runner.room()
    assert (room["width"], room["height"]) == (64, 64)
    scene = next(e for e in bundle.manifest.entries if e.name == "den")
    assert scene.files["room"] == "den-room.aseprite"
    # 8 x 8 prop in a 16 x 16 box: scale 2. Centre 32 / 2 = 16 -> left 12; bottom 16 -> top 8.
    assert [(p["name"], p["x"], p["y"]) for p in room["props"]] == [("chair", 12, 8)]
    [floor] = room["surfaces"]
    assert (floor["columns"], floor["rows"]) == (4, 4)
    painted = {(n // 4, n % 4) for n, index in enumerate(floor["cells"]) if index}
    assert painted == {(row, column) for row in (2, 3) for column in (0, 1, 2)}
    picks = room_layout("floor", 4)
    for row, column in painted:
        index = floor["cells"][row * 4 + column]
        block = picks[(row // 2) * 2 + column // 2]
        assert index == (block - 1) * 4 + (row % 2) * 2 + column % 2 + 1


def test_the_structure_layer_is_the_shrunk_original_minus_what_props_and_surfaces_cover(store):
    """Only the uncovered, opaque part of the scene's own picture reaches the structure layer."""
    # The original picture (not the small refined scene): red left half, green right half,
    # and a fully transparent block top-right (original x 96-128, y 0-32).
    original = Image.new("RGBA", (128, 128), (200, 0, 0, 255))
    original.paste((0, 200, 0, 255), (64, 0, 128, 128))
    original.paste((0, 0, 0, 0), (96, 0, 128, 32))
    original.save(store.asset_dir("cosy-den", "den") / "source.png")
    stage_prop(store, "chair", (24, 16, 40, 32))
    stage_floor(store, (0, 64, 96, 128))

    plan = plan_for(store)

    structure = plan.structure
    assert structure.size == (plan.width, plan.height) == (64, 64)
    pixels = structure.load()
    # Uncovered pixels are the shrunk original: red at x 2, green at x 40.
    assert pixels[2, 2] == (200, 0, 0, 255)
    assert pixels[40, 10] == (0, 200, 0, 255)
    # Under the chair (room 12-20 x 8-16), under painted floor cells (rows 2-3, columns 0-2).
    assert pixels[15, 10][3] == 0
    assert pixels[20, 40][3] == 0
    assert pixels[47, 63][3] == 0
    # A floor-row pixel right of the painted cells is still structure.
    assert pixels[50, 40] == (0, 200, 0, 255)
    # The picture's own transparent block stays transparent (room x 48-64, y 0-16).
    assert pixels[55, 5][3] == 0


def test_a_piece_with_no_place_on_the_picture_is_left_out_with_a_note(store):
    """A prop kept before places were recorded must not be put somewhere invented."""
    stage_prop(store, "chair", (24, 16, 40, 32))
    stage(store, "old-lamp", AssetKind.PROP)

    bundle, runner = run_export(store)

    assert [p["name"] for p in runner.room()["props"]] == ["chair"]
    assert any(
        "old-lamp: left out of the room" in warning for warning in bundle.warnings
    )


def test_an_export_asked_only_for_2x_still_writes_the_room_and_the_tileset(store):
    """Room and tileset files are built from native pictures, so native size is always exported."""
    stage_prop(store, "chair", (24, 16, 40, 32))
    stage_floor(store, (0, 64, 96, 128))

    bundle, runner = run_export(store, scales=(2,))

    scene = next(e for e in bundle.manifest.entries if e.name == "den")
    assert scene.files["room"] == "den-room.aseprite"
    assert runner.room() is not None
    floor_entries = [e for e in bundle.manifest.entries if e.recipe is not None]
    assert floor_entries
    assert all(e.files.get("aseprite") == "floor.aseprite" for e in floor_entries)
    assert (bundle.directory / "den.png").exists()
    assert not any("native size" in w for w in bundle.warnings)


def test_the_preview_before_export_shows_the_prop_and_the_painted_cells_where_the_plan_puts_them(
    store,
):
    """The preview holds the prop on its planned pixels and tiles only on painted cells."""
    stage_prop(store, "chair", (24, 16, 40, 32))
    red = Image.new("RGBA", (8, 8), (200, 0, 0, 255))
    red.save(store.asset_dir("cosy-den", "chair") / "refined.png")
    stage_floor(store, (0, 64, 96, 128))

    preview = preview_room(store, store.load_direction("cosy-den"), "den")

    layers = preview.layers
    assert layers.tiles.size == layers.props.size == layers.structure.size == (64, 64)
    # The chair is 8 x 8 and lands at room x 12, y 8 (see the placement test above).
    opaque = {(x, y) for x in range(64) for y in range(64) if layers.props.getpixel((x, y))[3]}
    assert opaque == {(x, y) for x in range(12, 20) for y in range(8, 16)}
    assert layers.props.getpixel((12, 8)) == (200, 0, 0, 255)
    # The floor paints rows 2-3, columns 0-2 (16-pixel cells): tile blue there, none above.
    assert layers.tiles.getpixel((0, 32)) == (10, 20, 200, 255)
    assert layers.tiles.getpixel((47, 63)) == (10, 20, 200, 255)
    assert layers.tiles.getpixel((48, 63))[3] == 0
    assert layers.tiles.getpixel((0, 31))[3] == 0


def test_the_preview_before_export_leaves_nothing_in_the_projects_exports(store):
    """A preview is not an export: the project must not gain a bundle folder from looking."""
    stage_prop(store, "chair", (24, 16, 40, 32))
    stage_floor(store, (0, 64, 96, 128))

    preview_room(store, store.load_direction("cosy-den"), "den")

    assert not store.export_dir("anything").parent.exists()


def test_a_scene_picture_gone_since_the_export_gives_a_sentence_not_a_crash(store, tmp_path):
    """The export screen reads the room of a bundle; a missing picture must not take it down.

    If this raised, the export page would be an error page and nobody could export.
    """
    stage_prop(store, "chair", (24, 16, 40, 32))
    bundle = export(store, store.load_direction("cosy-den"), into=tmp_path / "bundle")
    (store.asset_dir("cosy-den", "den") / "source.png").unlink()

    preview = room_from_bundle(store, "cosy-den", bundle.directory, "den")

    assert preview.layers is None
    assert len(preview.notes) == 1
    assert "could not be built" in preview.notes[0]


real_aseprite = pytest.mark.skipif(find_aseprite() is None, reason="Aseprite is not installed")


@real_aseprite
def test_the_room_file_opens_in_aseprite_with_its_layers_in_place(store, tmp_path):
    """The real program reads the file back: tile layer under prop layers, prop at its place."""
    stage_prop(store, "chair", (24, 16, 40, 32))
    stage_prop(store, "rug", (24, 40, 40, 56))
    stage_floor(store, (0, 64, 96, 128))

    bundle = export(store, store.load_direction("cosy-den"), stamp="s", now=NOW)

    result = tmp_path / "read.json"
    script = tmp_path / "read.lua"
    script.write_text(
        f"""
local sprite = app.open("{bundle.directory / "den-room.aseprite"}")
local out = {{ size = {{ sprite.width, sprite.height }}, layers = {{}} }}
for _, layer in ipairs(sprite.layers) do
  local cel = layer:cel(1)
  local at = cel.position
  local entry = {{ name = layer.name, map = layer.isTilemap, x = at.x, y = at.y }}
  if layer.isTilemap then
    entry.tiles = #layer.tileset - 1
    entry.cells = {{}}
    for it in cel.image:pixels() do
      table.insert(entry.cells, app.pixelColor.tileI(it()))
    end
  end
  table.insert(out.layers, entry)
end
local file = io.open("{result}", "w")
file:write(json.encode(out))
file:close()
"""
    )
    subprocess.run(
        [str(find_aseprite()), "-b", "--script", str(script)],
        check=True, env={**os.environ, "ASEPRITE_USER_FOLDER": str(tmp_path)},
    )
    read = json.loads(result.read_text())

    assert read["size"] == [64, 64]
    # Bottom to top: the floor, the structure, then props by box bottom (chair 32, rug 56).
    assert [layer["name"] for layer in read["layers"]] == ["floor", "structure", "chair", "rug"]
    floor, structure, chair, rug = read["layers"]
    assert floor["map"] and floor["tiles"] == 32
    assert sum(1 for index in floor["cells"] if index) == 6
    assert (structure["x"], structure["y"]) == (0, 0)
    assert (chair["x"], chair["y"]) == (12, 8)
    assert (rug["x"], rug["y"]) == (12, 28 - 8)


@real_aseprite
def test_the_preview_is_the_room_file_pixel_for_pixel(store, tmp_path):
    """Aseprite flattens the written room file; the page's preview must be the same picture.

    No tolerance: the layers are opaque flat colours or fully transparent, so no blending
    rounding can differ. Catches the preview showing a room the file does not hold.
    """
    stage_prop(store, "chair", (24, 16, 40, 32))
    stage_prop(store, "rug", (24, 40, 40, 56))
    stage_floor(store, (0, 64, 96, 128))
    bundle = export(store, store.load_direction("cosy-den"), stamp="s", now=NOW)
    flat = tmp_path / "flat.png"
    subprocess.run(
        [str(find_aseprite()), "-b", str(bundle.directory / "den-room.aseprite"),
         "--save-as", str(flat)],
        check=True, env={**os.environ, "ASEPRITE_USER_FOLDER": str(tmp_path)},
    )

    preview = room_from_bundle(store, "cosy-den", bundle.directory, "den")

    with Image.open(flat) as from_file:
        assert from_file.convert("RGBA").tobytes() == preview.layers.flatten().tobytes()
