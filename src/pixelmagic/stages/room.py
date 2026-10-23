"""Plan a room: where a kept scene's surfaces and props sit, drawn at one scale.

A room is the layered `.aseprite` export writes beside a kept scene: one tile
layer per recipe surface, painted over the area the surface covers in the
original picture, one layer per prop, placed where its object sits in that
picture, and between them one `structure` layer: the original picture shrunk to
the room, keeping only what no prop and no painted cell already covers (the
frame, beams and arches that are neither). It is a starting point a person edits
in Aseprite; nothing flows back.

This module is the planning only, with no Aseprite in it: which pieces belong,
the scale, the canvas, each prop's place and each surface's map of tile
indexes. Pieces that cannot be placed are left out with a note, never guessed.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import ValidationError
from scipy import ndimage

from pixelmagic.model import AssetKind, AssetRecord, Cutout, ManifestEntry
from pixelmagic.stages.recipes import PLACES, room_layout
from pixelmagic.store import Store

#: Tiles in one block of a recipe surface (2 x 2).
BLOCK_SPAN = 2

#: The structure layer drops shrunk pixels less opaque than this (of 255). The den's
#: dark glow around the room averages down to a faint brown halo outside the frame;
#: half opaque and above keeps the frame's own outline and drops the halo.
STRUCTURE_MIN_ALPHA = 128


@dataclass
class RoomProp:
    """One prop layer: its picture and where its top-left corner lands in the room."""

    name: str
    image: Path
    x: int
    y: int
    #: Bottom edge of the object's box on the original picture; the layer order.
    bottom: int


@dataclass
class RoomSurface:
    """One tile layer: a surface's tiles and its map of tile indexes."""

    name: str
    #: The tiles' native pictures in tileset order (the first is tile index 1).
    tiles: list[Path]
    tile_data: list[str]
    columns: int
    rows: int
    #: Tile indexes in reading order, `columns` x `rows`; 0 is an unpainted cell.
    cells: list[int]

    @property
    def painted(self) -> int:
        """How many cells hold a tile."""
        return sum(1 for index in self.cells if index)


@dataclass
class RoomPlan:
    """Everything the Aseprite script needs to build one room."""

    width: int
    height: int
    #: Bottom to top.
    surfaces: list[RoomSurface] = field(default_factory=list)
    #: Back to front, above every surface.
    props: list[RoomProp] = field(default_factory=list)
    #: The scene's original picture shrunk to `width` x `height` (RGBA), with every pixel
    #: a placed prop or a painted cell covers made transparent. Drawn between the
    #: surfaces and the props. Set by `plan_room` once the props and surfaces are placed.
    structure: Image.Image = field(init=False)


#: A recipe surface's tiles in tileset order and their data strings, as export builds them.
TileSets = Mapping[str, tuple[Sequence[ManifestEntry], Sequence[str]]]


def _leave_out(notes: list[tuple[str, str]], name: str, why: str) -> None:
    """Note that a piece is not in the room and why."""
    notes.append((name, f"left out of the room: {why}"))


def _read_cutout(store: Store, direction: str, decomposition: str, name: str) -> Cutout | None:
    """The cutout record a piece was kept from, or None when it is gone or unreadable."""
    path = store.cutout_dir(direction, decomposition, name) / "cutout.json"
    try:
        return store.read_model(path, Cutout)
    except (OSError, ValidationError):
        return None


def _plan_prop(
    store: Store,
    direction: str,
    folder: Path,
    record: AssetRecord,
    entry: ManifestEntry,
    notes: list[tuple[str, str]],
) -> tuple[Path, tuple[int, int, int, int]] | None:
    """A prop's native picture and its box on the original picture, or None.

    None leaves the prop out of the room with one note saying why.
    """
    if record.decomposition is None:
        _leave_out(notes, record.name, "it was kept before its place on the picture was recorded")
        return None
    cutout = _read_cutout(store, direction, record.decomposition, record.name)
    if cutout is None:
        _leave_out(notes, record.name, "its cutout record in the decomposition is missing")
        return None
    if cutout.origin is None:
        _leave_out(notes, record.name, "it was cut from a sheet, so it has no place on the picture")
        return None
    return folder / entry.files["native"], cutout.origin.picture_box


def _plan_surface(
    store: Store,
    direction: str,
    folder: Path,
    name: str,
    entries: Sequence[ManifestEntry],
    tile_data: Sequence[str],
    decomposition: str | None,
    width: int,
    height: int,
    scale: float,
    cell: int,
    notes: list[tuple[str, str]],
) -> RoomSurface | None:
    """A surface's tile layer: which cells its outline covers and the block laid in each.

    A cell is painted when the original-picture pixel at its centre lies
    inside the outline, holes it encloses included. Each 2 x 2 block of the map
    takes one block of the surface's tiles, so a painted cell gets that block's tile for its place.
    Returns None, with a note, when the outline or its place is missing.
    """
    if decomposition is None:
        _leave_out(notes, name, "it was kept before its place on the picture was recorded")
        return None
    cutout = _read_cutout(store, direction, decomposition, entries[0].name)
    outline_path = store.surface_dir(direction, decomposition, name) / "outline.png"
    if cutout is None or not outline_path.exists():
        _leave_out(notes, name, "its outline is missing from the decomposition")
        return None
    x0, y0, x1, y1 = cutout.bbox
    try:
        with Image.open(outline_path) as picture:
            # A floor runs under what stands on it: the finder leaves a bed or a rug
            # out of the outline, so holes the outline fully encloses count as inside.
            inside = ndimage.binary_fill_holes(np.asarray(picture.convert("L")) > 0)
    except OSError:
        _leave_out(notes, name, "its outline could not be read")
        return None
    if inside.shape != (y1 - y0, x1 - x0):
        _leave_out(notes, name, "its outline does not fit its box on the picture")
        return None

    columns = math.ceil(width / cell)
    rows = math.ceil(height / cell)
    blocks_across = math.ceil(columns / BLOCK_SPAN)
    picks = room_layout(name, blocks_across * math.ceil(rows / BLOCK_SPAN))
    cells: list[int] = []
    for row in range(rows):
        for column in range(columns):
            x = int((column + 0.5) * cell * scale) - x0
            y = int((row + 0.5) * cell * scale) - y0
            if not (0 <= y < inside.shape[0] and 0 <= x < inside.shape[1] and inside[y, x]):
                cells.append(0)
                continue
            block = picks[(row // BLOCK_SPAN) * blocks_across + column // BLOCK_SPAN]
            place = (row % BLOCK_SPAN) * BLOCK_SPAN + column % BLOCK_SPAN
            cells.append((block - 1) * len(PLACES) + place + 1)
    surface = RoomSurface(
        name=name,
        tiles=[folder / entry.files["native"] for entry in entries],
        tile_data=list(tile_data),
        columns=columns,
        rows=rows,
        cells=cells,
    )
    if not surface.painted:
        _leave_out(notes, name, "its outline covers no cell of the room")
        return None
    return surface


def _plan_structure(
    picture: Image.Image, plan: RoomPlan, cell: int
) -> Image.Image:
    """The original `picture` shrunk to the room, minus what props and surfaces cover.

    The shrink averages areas (so a thin line survives as a blend, not a gap). A pixel is
    covered when it lies inside an opaque pixel of a placed prop's picture at its
    planned place, or inside a painted cell of any surface; covered pixels become
    transparent so the layer never draws twice over something a person moves.
    """
    small = np.array(
        picture.convert("RGBA").resize((plan.width, plan.height), Image.Resampling.BOX)
    )
    covered = np.zeros((plan.height, plan.width), dtype=bool)
    for prop in plan.props:
        with Image.open(prop.image) as image:
            opaque = np.asarray(image.convert("RGBA"))[..., 3] > 0
        # Clip the prop's rectangle to the canvas, then take the matching part of its mask.
        top, left = max(prop.y, 0), max(prop.x, 0)
        bottom = min(prop.y + opaque.shape[0], plan.height)
        right = min(prop.x + opaque.shape[1], plan.width)
        if bottom > top and right > left:
            covered[top:bottom, left:right] |= opaque[
                top - prop.y : bottom - prop.y, left - prop.x : right - prop.x
            ]
    for surface in plan.surfaces:
        for n, index in enumerate(surface.cells):
            if index:
                row, column = divmod(n, surface.columns)
                covered[row * cell : (row + 1) * cell, column * cell : (column + 1) * cell] = True
    small[covered | (small[..., 3] < STRUCTURE_MIN_ALPHA)] = 0
    return Image.fromarray(small, "RGBA")


def plan_room(
    store: Store,
    direction: str,
    folder: Path,
    scene: AssetRecord,
    entries: Sequence[ManifestEntry],
    records: Mapping[str, AssetRecord],
    tile_sets: TileSets,
    cell: int,
    notes: list[tuple[str, str]],
) -> RoomPlan | None:
    """Plan the room of one kept scene from the pieces cut from the same picture.

    `entries` are the bundle's manifest entries, `records` their sidecars by
    name, `folder` the bundle (every picture in the room is the bundle's own
    native PNG), `tile_sets` each whole surface's tiles in tileset order with
    their data strings. The scale is the median of the placed props' own
    scales (box over picture), or the scene's `pitch` when no prop is placed. Returns None, with a
    note, when there is no scale or nothing could be placed.
    """
    same_picture = [
        entry
        for entry in entries
        if entry.kind is not AssetKind.SCENE
        and records[entry.name].source_generation == scene.source_generation
    ]
    #: Each placed prop: its entry, native picture, native size, and box on the picture.
    placed: list[tuple[ManifestEntry, Path, tuple[int, int], tuple[int, int, int, int]]] = []
    surfaces: set[str] = set()
    #: Notes about pieces left out; kept only if the picture has a room to write at all.
    left_out: list[tuple[str, str]] = []
    for entry in same_picture:
        record = records[entry.name]
        if record.recipe is not None:
            surfaces.add(record.recipe.surface)
        elif entry.kind is AssetKind.PROP:
            found = _plan_prop(store, direction, folder, record, entry, left_out)
            if found is not None:
                with Image.open(found[0]) as image:
                    placed.append((entry, found[0], image.size, found[1]))
        else:
            _leave_out(left_out, entry.name, "an older kind of tile has no place on the picture")
    if not placed and not surfaces:
        # An ordinary bundle of sheet-road props and a scene: nothing to say.
        return None
    notes += left_out

    # Each prop's scale is its box over the picture actually placed, longer side over
    # longer side; the room is drawn at the median of them.
    scale = (
        statistics.median(
            max(right - left, bottom - top) / max(size)
            for _, _, size, (left, top, right, bottom) in placed
        )
        if placed
        else scene.pitch
    )
    if not scale:
        notes.append(
            (scene.name, "no room written: no prop could be placed and the scene has no pitch")
        )
        return None
    with Image.open(store.asset_dir(direction, scene.name) / "source.png") as picture:
        original = picture.copy()
    plan = RoomPlan(
        width=max(1, round(original.width / scale)),
        height=max(1, round(original.height / scale)),
    )

    for entry, image_path, (image_width, image_height), (left, _, right, bottom) in placed:
        centre = round((left + right) / 2 / scale)
        base = round(bottom / scale)
        plan.props.append(
            RoomProp(
                name=entry.name,
                image=image_path,
                x=centre - image_width // 2,
                y=base - image_height,
                bottom=bottom,
            )
        )
    plan.props.sort(key=lambda prop: (prop.bottom, prop.name))

    for surface in sorted(surfaces):
        if surface not in tile_sets:
            _leave_out(notes, surface, "its tile set is not whole")
            continue
        ordered, tile_data = tile_sets[surface]
        planned = _plan_surface(
            store, direction, folder, surface, ordered, tile_data,
            records[ordered[0].name].decomposition,
            plan.width, plan.height, scale, cell, notes,
        )
        if planned is not None:
            plan.surfaces.append(planned)
    # Most painted cells first: the widest surface sits at the bottom.
    plan.surfaces.sort(key=lambda surface: (-surface.painted, surface.name))

    if not plan.props and not plan.surfaces:
        notes.append((scene.name, "no room written: nothing cut from this picture could be placed"))
        return None
    plan.structure = _plan_structure(original, plan, cell)
    return plan


@dataclass
class RoomLayers:
    """A planned room as three RGBA pictures of the room's size, so a page can show or hide each."""

    #: Every surface's painted cells, bottom surface first.
    tiles: Image.Image
    #: The plan's structure layer.
    structure: Image.Image
    #: Every prop, back to front.
    props: Image.Image

    def flatten(self) -> Image.Image:
        """The three layers stacked in the room file's order: tiles, structure, props."""
        flat = self.tiles.copy()
        flat.alpha_composite(self.structure)
        flat.alpha_composite(self.props)
        return flat


def _paste_clipped(layer: Image.Image, picture: Image.Image, left: int, top: int) -> None:
    """Alpha-composite `picture` onto `layer` at `left`, `top`, clipped at the edges."""
    cut_left, cut_top = max(-left, 0), max(-top, 0)
    cut_right = min(picture.width, layer.width - left)
    cut_bottom = min(picture.height, layer.height - top)
    if cut_right > cut_left and cut_bottom > cut_top:
        layer.alpha_composite(
            picture.crop((cut_left, cut_top, cut_right, cut_bottom)),
            (left + cut_left, top + cut_top),
        )


def compose_room(plan: RoomPlan, cell: int) -> RoomLayers:
    """Paste a plan's layers the way `aseprite/room_sprite.lua` places them.

    A painted cell of a surface's map holds its tile's native picture with its
    top-left corner at `column * cell`, `row * cell`, as a tilemap cel at 0,0 does;
    surfaces go in the plan's order, bottom first. Each prop is composited at its
    planned x, y, back to front, and anything past the canvas edge is cut off.
    """
    size = (plan.width, plan.height)
    tiles = Image.new("RGBA", size)
    for surface in plan.surfaces:
        pictures: dict[int, Image.Image] = {}
        for n, index in enumerate(surface.cells):
            if not index:
                continue
            if index not in pictures:
                with Image.open(surface.tiles[index - 1]) as tile:
                    pictures[index] = tile.convert("RGBA")
            row, column = divmod(n, surface.columns)
            _paste_clipped(tiles, pictures[index], column * cell, row * cell)
    props = Image.new("RGBA", size)
    for prop in plan.props:
        with Image.open(prop.image) as picture:
            _paste_clipped(props, picture.convert("RGBA"), prop.x, prop.y)
    return RoomLayers(tiles=tiles, structure=plan.structure.convert("RGBA"), props=props)
