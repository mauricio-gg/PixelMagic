"""Pack a direction's finished assets into a bundle a game can read.

An export is a flat folder: one PNG per asset at its real size, optional
nearest-neighbour copies at two and four times, and a `manifest.json` that
names every file and says where it came from. PixelMagic never writes into a
game's asset folders — the game reads the bundle (spec, out of scope for v1).
"""

from __future__ import annotations

import importlib.resources
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    Direction,
    Manifest,
    ManifestEntry,
    Pitch,
)
from pixelmagic.refiner.base import COHESION_FALLBACK, LOW_GRID_CONFIDENCE, SIZE_SETTING_DROPPED
from pixelmagic.stages.recipes import (
    PLACES,
    SAMPLE_COLUMNS,
    SAMPLE_ROWS,
    VARIANTS,
    sample_layout,
)
from pixelmagic.stages.refine import (
    PIXELMAGIC_REFUSAL_CODES,
    REFINED_FILE,
    REPORT_FILE,
    WRONG_SIZE,
    load_report,
    wrong_size_sentence,
)
from pixelmagic.stages.room import RoomPlan, TileSets, plan_room
from pixelmagic.stages.slice import SEAM_VISIBLE
from pixelmagic.store import Store

#: Native size plus the two upscales a pixel-art game usually wants.
DEFAULT_SCALES: tuple[int, ...] = (1, 2, 4)

#: The largest upscale factor `export` will honour. A 16x upscale of a
#: 2880-pixel picture is already a 46k-pixel-wide image; anything larger is a
#: mistake, not a request. Guardian I1: the command line had a floor
#: (`min=1`) but no ceiling, and the web form had neither, so a fat-fingered
#: `scale=20000` reached Pillow and could exhaust memory. The bound lives
#: here, in the core, so every face — the CLI, the web page, and the MCP
#: server still to come — refuses the same values the same way.
MAX_SCALE = 16

ASSET_FILE = "asset.json"
#: The index file every bundle holds; the room preview and the web page read it by this name.
MANIFEST_FILE = "manifest.json"

#: Fields the models keep so an old manifest or sidecar still loads, but that
#: only the removed roads ever filled, so a new `manifest.json` does not write
#: them at all (not even as null): `role` and `position` (bordered and
#: plain-patch tiles), `grid` (the sheet road's cell), and the two seam scores
#: on an entry's `cohesion` (tile blocks). There is no `pitches.props` either:
#: `sheet_pitches` no longer makes one.
MANIFEST_DROPPED_FIELDS = {
    "entries": {
        "__all__": {
            "role": True,
            "position": True,
            "grid": True,
            "cohesion": {"centre_seam_with_cohesion", "centre_seam_without_cohesion"},
        }
    }
}

#: The warning codes an operator can act on, in plain words.
#: `LOW_GRID_CONFIDENCE`: the refiner guessed the pixel grid and was not sure —
#: `refine.json` holds the candidates it considered, and M2.1's screen offers
#: them. `SEAM_VISIBLE` is ours, not the refiner's: a centre tile whose opposite
#: edges disagree. Nothing writes it now (#179), but an old `refine.json` may
#: carry it, so its explanation stays.
#:
#: Public on purpose: `export` uses it below to explain a warning in a
#: bundle's notes, and `web.refining` reuses the same table to show the
#: refine screen's warnings in plain words, rather than holding a second
#: copy of the same two sentences (CLAUDE.md: a rule every face needs lives
#: in the core, not repeated in a face).
ACTIONABLE_WARNINGS = {
    LOW_GRID_CONFIDENCE: "the refiner was not sure of the pixel grid",
    SEAM_VISIBLE: "this tile's edges do not meet; laid in a row it will show a seam",
    COHESION_FALLBACK: (
        "this tile's block refined with a worse seam under cohesion, so it "
        "shipped with cohesion and clean-up off instead"
    ),
    SIZE_SETTING_DROPPED: (
        "a saved setting that would have changed the forced size (scale, makeSquare, "
        "keepAspectRatio or outlineStyle) was not sent; the receipt's requested "
        "settings show what was"
    ),
}


#: How many asset names a collapsed note spells out before "…". Three is
#: enough to recognise which sheet the note is about without turning a
#: forty-three-tile warning back into forty-three lines.
MAX_NAMED_ASSETS = 3


def collapse_notes(notes: Sequence[tuple[str, str]]) -> list[str]:
    """Turn `(asset name, note)` pairs into the lines a face prints.

    Identical notes become one line: "<text> — <n> assets: a, b, c, …", with
    the first `MAX_NAMED_ASSETS` names and "…" only when there are more. A
    note that concerns one asset gets the "<name>: <text>" shape. Lines come
    out in the order a note first appeared, so the first thing that went
    wrong is still the first thing said.

    Spec M3 §6: a tile sheet whose every centre tile shows a seam used to
    print one line per tile, which buried the one note that was different.
    """
    grouped: dict[str, list[str]] = {}
    for name, text in notes:
        grouped.setdefault(text, []).append(name)
    lines: list[str] = []
    for text, names in grouped.items():
        if len(names) == 1:
            lines.append(f"{names[0]}: {text}")
            continue
        shown = names[:MAX_NAMED_ASSETS]
        if len(names) > MAX_NAMED_ASSETS:
            shown.append("…")
        lines.append(f"{text} — {len(names)} assets: {', '.join(shown)}")
    return lines


class ExportBundle(BaseModel):
    """Where an export landed and what it contains."""

    directory: Path
    manifest: Manifest
    warnings: list[str] = Field(default_factory=list)


def scale_nearest(image: Image.Image, factor: int) -> Image.Image:
    """Blow an image up by a whole number without softening its edges.

    Nearest-neighbour is the only correct resampling for pixel art; anything
    that interpolates invents colours that are not in the palette.
    """
    if factor == 1:
        return image
    width, height = image.size
    return image.resize((width * factor, height * factor), Image.Resampling.NEAREST)


def bundle_folder(store: Store, direction_name: str, stamp: str) -> Path:
    """Pick an export folder that does not already exist.

    Two exports inside the same second would otherwise share
    `<direction>-YYYYmmdd-HHMMSS` and the second would silently overwrite the
    first. Same `-2`, `-3` rule as `store.new_id`.
    """
    base = f"{direction_name}-{stamp}"
    if not store.export_dir(base).exists():
        return store.export_dir(base)
    counter = 2
    while store.export_dir(f"{base}-{counter}").exists():
        counter += 1
    return store.export_dir(f"{base}-{counter}")


def _file_key(factor: int) -> str:
    """The manifest key for one scale."""
    return "native" if factor == 1 else f"x{factor}"


def _file_name(name: str, factor: int) -> str:
    """The file name for one asset at one scale."""
    return f"{name}.png" if factor == 1 else f"{name}@{factor}x.png"


#: Where to look first for the Aseprite program.
ASEPRITE_ENV_VAR = "PIXELMAGIC_ASEPRITE"

#: The standard macOS install location (Steam and direct builds agree on it).
MACOS_APP_BINARY = Path("/Applications/Aseprite.app/Contents/MacOS/aseprite")

#: The one Lua script we ship, as package data. Resolved once at import: this
#: is a real file path for any normal install, and would need
#: `importlib.resources.as_file` if PixelMagic were ever run from a zipped
#: wheel — which nothing does today (it is a local tool on one Mac).
LUA_SCRIPT = Path(
    str(importlib.resources.files("pixelmagic.aseprite").joinpath("single_sprite.lua"))
)

#: The read-back script: a tileset file's tiles out as PNGs, plus what to check.
READ_TILESET_LUA_SCRIPT = Path(
    str(importlib.resources.files("pixelmagic.aseprite").joinpath("read_tileset.lua"))
)

#: The frame read-back script: every frame of a sprite as a flat RGB picture.
READ_FRAMES_LUA_SCRIPT = Path(
    str(importlib.resources.files("pixelmagic.aseprite").joinpath("read_frames.lua"))
)

#: The tileset script: one tilemap layer and a tileset, for a recipe surface.
TILESET_LUA_SCRIPT = Path(
    str(importlib.resources.files("pixelmagic.aseprite").joinpath("tileset_sprite.lua"))
)

#: The room script: a kept scene's layered room, tile layers under prop layers.
ROOM_LUA_SCRIPT = Path(
    str(importlib.resources.files("pixelmagic.aseprite").joinpath("room_sprite.lua"))
)

ASEPRITE_TIMEOUT = 300


def find_aseprite() -> Path | None:
    """Locate the Aseprite program: env var, then PATH, then the macOS app.

    Aseprite is not redistributable, so PixelMagic only ever detects an
    existing installation — it never downloads or bundles it.
    """
    configured = os.environ.get(ASEPRITE_ENV_VAR)
    if configured and Path(configured).exists():
        return Path(configured)
    found = shutil.which("aseprite")
    if found:
        return Path(found)
    if MACOS_APP_BINARY.exists():
        return MACOS_APP_BINARY
    return None


class AsepriteRunner:
    """Drives a detected Aseprite in batch mode to write one `.aseprite`."""

    def __init__(
        self,
        binary: Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        timeout: int = ASEPRITE_TIMEOUT,
    ) -> None:
        """Bind the program to run and the function used to run it."""
        self._binary = binary
        self._runner = runner
        self._timeout = timeout

    def build(self, image_path: Path, output_path: Path, *, cell: int) -> None:
        """Turn one PNG into one `.aseprite` with a `cell` x `cell` grid.

        Raises RuntimeError on any failure. The grid is the direction's cell
        size (spec M3 §6), so a piece opened in Aseprite shows the cells the
        engine lays; it goes through its own `--script-param` rather than the
        job file simply because the Lua script reads it that way (see
        `single_sprite.lua`'s own header comment for the invocation it expects).

        Two headless gotchas are handled here: every `--script-param` must come
        before `--script` or Aseprite drops it silently, and Aseprite needs a
        writable user folder — a user's installed extensions can otherwise
        error on load and poison the run, so it is pointed at a scratch dir.
        """
        self._run_script(
            LUA_SCRIPT,
            {
                "source_image": str(image_path.resolve()),
                "output": str(output_path.resolve()),
            },
            output_path,
            cell=cell,
        )

    def build_tileset(
        self,
        tile_paths: Sequence[Path],
        tile_data: Sequence[str],
        layout: Sequence[int],
        output_path: Path,
        *,
        cell: int,
        columns: int,
        rows: int,
    ) -> None:
        """Write a tileset `.aseprite`: the tiles, plus a sample floor laid from them.

        `tile_paths` are in tileset order (the first is tile index 1, index 0
        being Aseprite's empty tile); `layout` is the sample floor as tile
        indexes in reading order, `columns` x `rows` of them. `tile_data` is
        one string per tile (`variant=N;place=nw`).
        Raises RuntimeError on any failure, run exactly as `build` runs its
        script (parameters before `--script`, a scratch user folder).
        """
        self._run_script(
            TILESET_LUA_SCRIPT,
            {
                "tiles": [str(path.resolve()) for path in tile_paths],
                "tile_data": list(tile_data),
                "layout": list(layout),
                "columns": columns,
                "rows": rows,
                "output": str(output_path.resolve()),
            },
            output_path,
            cell=cell,
        )

    def build_room(self, plan: RoomPlan, output_path: Path, *, cell: int) -> None:
        """Write a room `.aseprite`: a tilemap layer per surface, the structure, a layer per prop.

        `plan` says what goes where (`stages.room.plan_room`); surfaces are
        listed bottom to top, then the `structure` image layer (the shrunk
        original picture), then props back to front.
        The structure picture goes through a scratch file, never the bundle.
        Raises RuntimeError on any failure, run exactly as `build` runs its script.
        """
        with tempfile.TemporaryDirectory(prefix="pixelmagic-structure-") as scratch:
            structure = Path(scratch) / "structure.png"
            plan.structure.save(structure)
            self._run_script(
                ROOM_LUA_SCRIPT,
                {
                    "width": plan.width,
                    "height": plan.height,
                    "surfaces": [
                        {
                            "name": surface.name,
                            "tiles": [str(path.resolve()) for path in surface.tiles],
                            "tile_data": surface.tile_data,
                            "columns": surface.columns,
                            "rows": surface.rows,
                            "cells": surface.cells,
                        }
                        for surface in plan.surfaces
                    ],
                    "structure": str(structure.resolve()),
                    "props": [
                        {
                            "name": prop.name,
                            "image": str(prop.image.resolve()),
                            "x": prop.x,
                            "y": prop.y,
                        }
                        for prop in plan.props
                    ],
                    "output": str(output_path.resolve()),
                },
                output_path,
                cell=cell,
            )

    def read_tileset(self, file: Path, out_dir: Path) -> dict:
        """Write a tileset file's tiles into `out_dir` as `tile-<i>.png` and say what is in it.

        Returns `{"has_tilemap": bool, "count": int, "sizes": [[w, h], ...]}`
        (tile index 0, Aseprite's empty tile, is neither written nor counted).
        Raises RuntimeError when Aseprite cannot open or read the file.
        """
        result_path = out_dir / "result.json"
        self._run_script(
            READ_TILESET_LUA_SCRIPT,
            {"file": str(file.resolve()), "out_dir": str(out_dir.resolve())},
            result_path,
        )
        return json.loads(result_path.read_text(encoding="utf-8"))

    def read_frames(self, file: Path, out_dir: Path) -> int:
        """Write every frame of `file` into `out_dir` as flat RGB `frame-<n>.png`; return the count.

        Frames are flattened as they look (visible layers), and an indexed
        sprite comes out in RGB. Raises RuntimeError when Aseprite cannot open
        or read the file.
        """
        result_path = out_dir / "result.json"
        self._run_script(
            READ_FRAMES_LUA_SCRIPT,
            {"file": str(file.resolve()), "out_dir": str(out_dir.resolve())},
            result_path,
        )
        return int(json.loads(result_path.read_text(encoding="utf-8"))["frames"])

    def _run_script(
        self, script: Path, job: dict, output_path: Path, *, cell: int | None = None
    ) -> None:
        """Run one Lua `script` headless over a `job` and check it wrote `output_path`.

        Raises RuntimeError on a non-zero exit or when no file appears. The job
        goes through a file in a scratch folder, which is also Aseprite's user
        folder (see `build` for why).
        """
        with tempfile.TemporaryDirectory(prefix="pixelmagic-aseprite-") as scratch:
            job_path = Path(scratch) / "job.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            command = [
                str(self._binary),
                "-b",
                "--script-param",
                f"job={job_path}",
                *(["--script-param", f"cell={cell}"] if cell is not None else []),
                "--script",
                str(script),
            ]
            environment = {**os.environ, "ASEPRITE_USER_FOLDER": scratch}
            completed = self._runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._timeout,
                env=environment,
            )
            if getattr(completed, "returncode", 1) != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"aseprite exited {completed.returncode}: {detail}")
        if not output_path.exists():
            raise RuntimeError(f"aseprite reported success but wrote no file at {output_path}")


def _check_scales(scales: Sequence[int]) -> None:
    """Refuse a bad `scales` sequence before any folder is created.

    Guardian finding I1: `scale=0` used to reach `scale_nearest` and write a
    bundle with zero entries, whose page then blamed the refiner for an empty
    bundle that was really an empty request; `scale=20000` reached Pillow and
    could exhaust memory, since `export`'s per-asset `except (OSError,
    ValueError)` cannot catch a `MemoryError`. Both are refused here, by name,
    before `bundle_folder` ever picks a path. An empty sequence is fine: `export`
    adds native size to whatever is asked for.
    """
    for factor in scales:
        if not isinstance(factor, int) or isinstance(factor, bool) or not (
            1 <= factor <= MAX_SCALE
        ):
            raise ValueError(
                f"{factor!r} is not a scale export will use; give a whole "
                f"number from 1 to {MAX_SCALE}"
            )


def _agreed_pitch(
    records: Sequence[AssetRecord],
) -> tuple[AssetRecord | None, list[AssetRecord]]:
    """Pick the scene a bundle agrees to trust for its pitch, and name who does not.

    When scenes from two pictures sit side by side, the newest
    `source_generation` wins — ids are timestamp slugs, so the largest is the
    latest. `max` breaks a tie by keeping the first one in `records`' own order,
    which callers build from `Store.list_assets`, so the winner is whichever
    asset name sorts first. Every disagreeing record still gets its note either
    way. Returns the chosen record (or `None` if nobody carries a pitch) and
    every other record whose pitch disagrees with it; `sheet_pitches` turns
    those into notes once it knows whether the chosen pitch will actually be
    recorded.
    """
    carrying = [record for record in records if record.pitch is not None]
    if not carrying:
        return None, []
    chosen = max(carrying, key=lambda record: record.source_generation)
    disagreeing = [record for record in carrying if record.pitch != chosen.pitch]
    return chosen, disagreeing


def sheet_pitches(
    records: Sequence[AssetRecord],
) -> tuple[dict[str, Pitch], list[tuple[str, str]]]:
    """The manifest's pitches, read off the assets that went in.

    `"scene"` is the kept scene's pitch; it is absent when no scene carries one
    (an M2 direction, or a bundle without a scene). There is no props pitch:
    a rebuilt prop carries its own pitch on its manifest entry. Only assets that
    made it into the bundle are consulted: a skipped asset must not set the
    pitch of a bundle it is not in. Returns the dict and the `(asset name, note)`
    pairs for any scene whose pitch disagrees with the one recorded, or whose
    winning pitch has no known source.

    A record can hold a pitch with no `pitch_source` only if a sidecar was
    hand-edited — no writer of ours produces one. `Pitch.source` is required,
    so there is nothing to put in the dict; the key is left out entirely and a
    note says why. A disagreeing scene's own note is worded to match: when the
    chosen pitch is recorded it names the value the manifest holds, and when it
    is not — because the chosen asset's own source is unknown — it says so
    instead of pointing at a value the manifest never wrote down.
    """
    pitches: dict[str, Pitch] = {}
    notes: list[tuple[str, str]] = []
    chosen, disagreeing = _agreed_pitch(
        [record for record in records if record.kind is AssetKind.SCENE]
    )
    if chosen is None:
        return pitches, notes
    if chosen.pitch_source is None:
        notes.append(
            (
                chosen.name,
                f"has pitch {chosen.pitch} but its source is unknown, so the "
                f"manifest records no scene pitch",
            )
        )
        notes += [
            (
                record.name,
                f"has pitch {record.pitch}; the manifest records no scene pitch, "
                f"since {chosen.name}'s source is unknown",
            )
            for record in disagreeing
        ]
        return pitches, notes
    pitches["scene"] = Pitch(value=chosen.pitch, source=chosen.pitch_source)
    notes += [
        (
            record.name,
            f"has pitch {record.pitch}, not the {chosen.pitch} the manifest "
            f"records for the scene",
        )
        for record in disagreeing
    ]
    return pitches, notes


def _tile_order(surface: str) -> tuple[list[tuple[int, str]], list[str], list[int]]:
    """A surface's tile order, each tile's data string, and its sample floor.

    Tiles are `(variant, place)` keys in tileset order: block 1's `nw`, `ne`, `sw`,
    `se`, then block 2's, and so on, so block `v`'s tile at place `p` is at index
    `(v - 1) * 4 + position of p in PLACES + 1`. The floor is the sample map in
    reading order, SAMPLE_COLUMNS x SAMPLE_ROWS tile indexes: 5 x 3 blocks, each
    laid with its four tiles in place.
    """
    picks = sample_layout(surface)
    keys = [(v, place) for v in range(1, VARIANTS + 1) for place in PLACES]
    data = [f"variant={v};place={place}" for v, place in keys]
    blocks_across = SAMPLE_COLUMNS // 2
    floor = []
    for y in range(SAMPLE_ROWS):
        for x in range(SAMPLE_COLUMNS):
            block = picks[(y // 2) * blocks_across + x // 2]
            floor.append((block - 1) * len(PLACES) + (y % 2) * 2 + x % 2 + 1)
    return keys, data, floor


def _palette(image: Image.Image) -> list[str]:
    """A tile's opaque colours as `#rrggbb`, commonest first, as a refine report lists them."""
    counts = Counter(p[:3] for p in image.convert("RGBA").get_flattened_data() if p[3] > 0)
    return [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in counts.most_common()]


def _take_from_master(
    aseprite: AsepriteRunner,
    master: Path,
    ordered: Sequence[ManifestEntry],
    folder: Path,
    cell: int,
) -> str | None:
    """Rewrite the surface's bundle pictures from its master file's tiles.

    Returns None when the master was used, else what is wrong with it (the
    caller warns and falls back to the recipe's own tiles). Nothing in the
    bundle changes until the master has passed every check. The tile at index i
    goes to the entry the build order put at index i: its PNG at every scale
    and its manifest palette; the master itself is copied in as the surface's
    `.aseprite` by the caller.
    """
    with tempfile.TemporaryDirectory(prefix="pixelmagic-master-") as scratch:
        try:
            read = aseprite.read_tileset(master, Path(scratch))
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            return f"could not be read ({error})"
        if not read["has_tilemap"]:
            return "has no tilemap layer"
        if read["count"] != len(ordered):
            return f"holds {read['count']} tiles, expected {len(ordered)}"
        for index, size in enumerate(read["sizes"], start=1):
            if size != [cell, cell]:
                return f"tile {index} is {size[0]} x {size[1]}, expected {cell} x {cell}"
        for index, entry in enumerate(ordered, start=1):
            with Image.open(Path(scratch) / f"tile-{index}.png") as tile:
                tile.load()
                entry.palette = _palette(tile)
                for key, file_name in entry.files.items():
                    if key != "aseprite":
                        factor = 1 if key == "native" else int(key[1:])
                        scale_nearest(tile.convert("RGBA"), factor).save(
                            folder / file_name, format="PNG"
                        )
    return None


def _surface_entries(manifest: Manifest) -> dict[str, list[ManifestEntry]]:
    """The manifest's recipe tiles grouped by the surface they belong to."""
    surfaces: dict[str, list[ManifestEntry]] = {}
    for entry in manifest.entries:
        if entry.recipe is not None:
            surfaces.setdefault(entry.recipe.surface, []).append(entry)
    return surfaces


def _ordered_tiles(
    entries: Sequence[ManifestEntry], keys: Sequence[tuple[int, str]]
) -> list[ManifestEntry] | None:
    """A surface's tiles in `keys` order, or None unless it holds exactly that whole set."""
    by_key = {(entry.variant, entry.recipe.place): entry for entry in entries}
    if len(entries) != len(keys) or set(by_key) != set(keys):
        return None
    return [by_key[key] for key in keys]


def surface_tile_sets(manifest: Manifest) -> TileSets:
    """Every whole recipe surface in the manifest: its tiles in tileset order and their data.

    A surface that does not hold exactly its whole set is left out, which is how
    `plan_room` knows its tile set is not whole. Shared by the room file and the
    room preview so the two can never disagree.
    """
    tile_sets = {}
    for surface, entries in _surface_entries(manifest).items():
        keys, tile_data, _ = _tile_order(surface)
        ordered = _ordered_tiles(entries, keys)
        if ordered is not None:
            tile_sets[surface] = (ordered, tile_data)
    return tile_sets


def _write_rooms(
    aseprite: AsepriteRunner,
    store: Store,
    direction: str,
    folder: Path,
    manifest: Manifest,
    records: dict[str, AssetRecord],
    cell: int,
    notes: list[tuple[str, str]],
) -> None:
    """Write `<scene>-room.aseprite` for every exported scene that can have one.

    The room is planned by `stages.room.plan_room` from the bundle's own
    pictures, so it is written after the tilesets. A name already taken in the
    bundle is not overwritten and gets a note. The file is listed on the
    scene's entry only once it exists; a failed run deletes what it wrote and
    adds a note, and never costs the bundle its PNGs or manifest.
    """
    tile_sets = surface_tile_sets(manifest)
    for scene in [entry for entry in manifest.entries if entry.kind is AssetKind.SCENE]:
        room_name = f"{scene.name}-room.aseprite"
        if (folder / room_name).exists():
            notes.append(
                (scene.name, f"no room written, {room_name} is already taken by another piece")
            )
            continue
        try:
            plan = plan_room(
                store, direction, folder, records[scene.name], manifest.entries, records,
                tile_sets, cell, notes,
            )
            if plan is None:
                continue
            aseprite.build_room(plan, folder / room_name, cell=cell)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            (folder / room_name).unlink(missing_ok=True)
            notes.append((scene.name, f"could not write the room .aseprite ({error})"))
            continue
        scene.files["room"] = room_name


def _write_tilesets(
    aseprite: AsepriteRunner,
    store: Store,
    direction: str,
    folder: Path,
    manifest: Manifest,
    cell: int,
    notes: list[tuple[str, str]],
) -> None:
    """Write one tileset `.aseprite` per recipe surface that exported whole.

    Tiles are found by the `recipe.surface`, `variant` and `place` on the
    manifest entries, never by name. A surface that holds 32 tiles is whole;
    any other count gets a note and no file. A file name already taken by a prop's
    own `.aseprite` is never overwritten and gets a note.

    The surface's master file (`Store.surface_master`) decides the pictures.
    No master: the file is built from the recipe tiles and a copy is saved as
    the master. A master that passes `_take_from_master`'s checks replaces the
    surface's pictures and is copied into the bundle. A master that fails
    them gets one note naming it and what is wrong, the recipe tiles are used,
    and the master is left as it is. Only after the file exists is it listed
    on the surface's entries, so the bundle never lists a file it does not hold.
    """
    for surface, entries in _surface_entries(manifest).items():
        keys, tile_data, floor = _tile_order(surface)
        ordered = _ordered_tiles(entries, keys)
        if ordered is None:
            notes.append(
                (
                    surface,
                    f"not written, the surface holds {len(entries)} tiles where {len(keys)} "
                    "are expected; tiles kept from before the block layout may be among them "
                    "and can be deleted",
                )
            )
            continue
        aseprite_name = f"{surface}.aseprite"
        if (folder / aseprite_name).exists():
            # A prop renamed to the surface's name already owns this file.
            notes.append(
                (surface, f"not written, {aseprite_name} is already taken by another piece")
            )
            continue
        master = store.surface_master(direction, surface)
        try:
            if master.exists():
                problem = _take_from_master(aseprite, master, ordered, folder, cell)
                if problem is None:
                    shutil.copyfile(master, folder / aseprite_name)
                else:
                    notes.append(
                        (surface, f"master file {master} {problem}; used the recipe's own tiles")
                    )
            if not (folder / aseprite_name).exists():
                aseprite.build_tileset(
                    [folder / entry.files["native"] for entry in ordered],
                    tile_data,
                    floor,
                    folder / aseprite_name,
                    cell=cell,
                    columns=SAMPLE_COLUMNS,
                    rows=SAMPLE_ROWS,
                )
                if not master.exists():
                    shutil.copyfile(folder / aseprite_name, master)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            (folder / aseprite_name).unlink(missing_ok=True)
            notes.append((surface, f"could not write .aseprite ({error})"))
            continue
        for entry in entries:
            entry.files["aseprite"] = aseprite_name


def export(
    store: Store,
    direction: Direction,
    *,
    scales: Sequence[int] = DEFAULT_SCALES,
    stamp: str | None = None,
    now: datetime | None = None,
    aseprite: AsepriteRunner | None = None,
    into: Path | None = None,
) -> ExportBundle:
    """Write every refined asset of a direction into a fresh bundle folder.

    `scales` is checked first, against `MAX_SCALE`, before anything on disk
    changes (see `_check_scales`) — locked decision 7 puts the bound here
    rather than in any one face, so the command line, the web page and the
    MCP server that comes next all refuse the same values the same way.

    Native size (scale 1) is always written, whatever `scales` asks for: the
    manifest names a native file for every asset, and the tileset and room
    `.aseprite` files are built from the native pictures. So the scales
    written are the requested ones plus 1, sorted, without duplicates, and an
    empty `scales` means native only. This is decided here once, for every face.

    Assets that have not been refined yet are skipped with a warning rather
    than exported half-finished — a bundle is meant to be safe to drop into a
    game as it stands. An asset the refiner refused is skipped the same way,
    even if a `refined.png` from an earlier, successful run is still sitting
    in its folder: the report's `error_code` is what decides, not the file's
    mere presence.

    A broken asset is skipped rather than raised, too — a sidecar that will
    not parse, a `refined.png` that is not a readable image — because one
    damaged folder must not cost the other nineteen assets their bundle. Each
    one adds a note naming the asset and what went wrong; `collapse_notes`
    folds any notes that read identically into one line, so a sheet of tiles
    that all hit the same problem is one line, not one per tile.

    `into` names the folder to write the bundle in instead of a fresh one under
    the project's exports. It is for the room preview's scratch bundle: Aseprite is
    not looked for at all, so no `.aseprite` is written and no warning about it is made.
    """
    _check_scales(scales)
    scales = sorted({1, *scales})
    moment = now or datetime.now(UTC)
    folder = into or bundle_folder(
        store, direction.name, stamp or moment.strftime("%Y%m%d-%H%M%S")
    )
    folder.mkdir(parents=True, exist_ok=True)

    project = store.load_project()
    manifest = Manifest(
        project=project.name,
        direction=direction.name,
        exported=moment,
        cell_size=direction.cell_size,
    )
    #: Bundle-wide lines (today only the missing-Aseprite one). Per-asset
    #: notes go in `notes` as `(asset, text)` pairs and are folded by
    #: `collapse_notes` at the end, so forty-three tiles that share one
    #: warning are one line, not forty-three.
    warnings: list[str] = []
    notes: list[tuple[str, str]] = []
    #: The sidecars of every asset that made it into the bundle;
    #: `sheet_pitches` reads the manifest's `pitches` off these once the loop
    #: below is done, never off a skipped asset or a decomposition file.
    exported_records: list[AssetRecord] = []

    # Aseprite is optional (spec locked decision 4 names it, but a machine
    # without it must still get its PNGs and manifest). One warning, not one
    # per asset.
    if aseprite is None and into is None:
        binary = find_aseprite()
        if binary is None:
            warnings.append(
                "Aseprite was not found, so no .aseprite files were written. "
                f"Set {ASEPRITE_ENV_VAR} to its full path if you want them."
            )
        else:
            aseprite = AsepriteRunner(binary)

    for name in store.list_assets(direction.name):
        asset_folder = store.asset_dir(direction.name, name)
        refined = asset_folder / REFINED_FILE
        report_path = asset_folder / REPORT_FILE

        # A refusal is read from the report, not guessed from the PNG: the
        # refine stage unlinks `refined.png` on refusal, so the realistic
        # refused asset has no PNG at all, and checking the file first would
        # bury the refiner's reason under a generic "not refined yet".
        try:
            report = (
                load_report(store, direction.name, name) if report_path.exists() else None
            )
        except (OSError, ValidationError) as error:
            notes.append((name, f"{REPORT_FILE} could not be read ({error}) — skipped"))
            continue
        if report is not None and report.error_code:
            reason = report.error_message or report.error_code
            if report.error_code in PIXELMAGIC_REFUSAL_CODES:
                # PixelMagic never sent this one to the refiner — it refused
                # its own target or could not read its own sidecar — so the
                # note must not say the refiner was the one that said no.
                notes.append(
                    (name, f"PixelMagic did not send it to the refiner ({reason}) — skipped")
                )
            elif report.error_code == WRONG_SIZE:
                notes.append((name, f"{wrong_size_sentence(report)} — skipped"))
            else:
                notes.append((name, f"was refused by the refiner ({reason}) — skipped"))
            continue
        if report is None or not refined.exists():
            notes.append((name, "has not been refined yet — skipped"))
            continue

        try:
            record = store.read_model(asset_folder / ASSET_FILE, AssetRecord)
        except (OSError, ValidationError) as error:
            notes.append((name, f"{ASSET_FILE} could not be read ({error}) — skipped"))
            continue

        files: dict[str, str] = {}
        try:
            with Image.open(refined) as image:
                native = image.convert("RGBA")
                width, height = native.size
                for factor in scales:
                    file_name = _file_name(name, factor)
                    # Recorded before the save, not after: a save that writes
                    # some bytes and then fails still leaves a file, and the
                    # handler below can only delete names it knows about.
                    # `files` is thrown away on failure, so listing a name
                    # that never got written costs nothing.
                    files[_file_key(factor)] = file_name
                    scale_nearest(native, factor).save(folder / file_name, format="PNG")
        except (OSError, ValueError) as error:
            # One unreadable PNG must not cost the other nineteen assets their
            # bundle. Every file this asset touched is removed, so the folder
            # never holds a file the manifest does not list.
            for written in files.values():
                (folder / written).unlink(missing_ok=True)
            notes.append((name, f"could not export {REFINED_FILE} ({error}) — skipped"))
            continue

        # A recipe tile has no `.aseprite` of its own: its surface gets one
        # tileset file after the loop.
        if aseprite is not None and record.recipe is None:
            aseprite_name = f"{name}.aseprite"
            try:
                aseprite.build(refined, folder / aseprite_name, cell=direction.cell_size)
                files["aseprite"] = aseprite_name
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                # Never fail an export over the optional format: the PNGs and
                # the manifest are the primary artifacts. A half-written file
                # is deleted, so the bundle holds only what the manifest lists.
                (folder / aseprite_name).unlink(missing_ok=True)
                notes.append((name, f"could not write .aseprite ({error})"))

        if report.warnings:
            # Warnings survive the run they happened in. Until now they lived
            # only on screen while `refine` was running, which meant a grid the
            # refiner was unsure about was invisible by the time anyone opened
            # the bundle.
            explained = [
                ACTIONABLE_WARNINGS[code]
                for code in report.warnings
                if code in ACTIONABLE_WARNINGS
            ]
            detail = f" — {'; '.join(explained)}" if explained else ""
            notes.append(
                (name, f"was refined with warnings: {', '.join(report.warnings)}{detail}")
            )

        exported_records.append(record)
        manifest.entries.append(
            ManifestEntry(
                name=name,
                kind=record.kind,
                width=width,
                height=height,
                palette=report.palette,
                source_prompt=record.source_prompt,
                service=record.service,
                refiner=report.requested,
                refiner_tool=report.refiner,
                route=report.route,
                segmenter=record.segmenter,
                warnings=list(report.warnings),
                files=files,
                pitch=record.pitch,
                pitch_source=record.pitch_source,
                target=record.target,
                footprint=record.footprint,
                anchor=record.anchor,
                cohesion=report.cohesion,
                variant=record.variant,
                recipe=record.recipe,
            )
        )

    if aseprite is not None:
        _write_tilesets(
            aseprite, store, direction.name, folder, manifest, direction.cell_size, notes
        )
        _write_rooms(
            aseprite, store, direction.name, folder, manifest,
            {
                entry.name: record
                for entry, record in zip(manifest.entries, exported_records, strict=True)
            },
            direction.cell_size, notes,
        )

    manifest.pitches, pitch_notes = sheet_pitches(exported_records)
    notes += pitch_notes

    store.write_model(folder / MANIFEST_FILE, manifest, exclude=MANIFEST_DROPPED_FIELDS)
    return ExportBundle(
        directory=folder, manifest=manifest, warnings=warnings + collapse_notes(notes)
    )
