"""The Export step (step 7): the room as it will be, what goes in the bundle, and the export.

Before an export the screen shows the kept scene's room, three pictures stacked in
the room file's order, and lists exactly what an export would write; both come from
the core's `preview_room`, whose scratch export is the list. After an export the
same screen is reached by a redirect that names the bundle, and then shows the
room from that bundle (`room_from_bundle`), the folder it went to, the files in its
manifest and how to open them in Aseprite. This module only reads, groups and
displays; what goes in a bundle, what is left out and every sentence saying why
are the core's.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import Response
from PIL import Image
from pydantic import Field

from pixelmagic import journey
from pixelmagic.journey import StepKey
from pixelmagic.model import Manifest, ManifestEntry
from pixelmagic.naming import check_name
from pixelmagic.stages.decompose import scene_holding
from pixelmagic.stages.export import MANIFEST_FILE, export
from pixelmagic.stages.room_preview import preview_room, room_from_bundle
from pixelmagic.web.deps import Workbench, page, require_direction, see_other, workbench
from pixelmagic.web.timeline import kept_scene_context

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]


def _data_url(picture: Image.Image) -> str:
    """A picture as a PNG data URL, so one page view makes one preview."""
    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def bundle_items(direction_name: str, entries: list[ManifestEntry]) -> list[dict[str, str]]:
    """The rows of "In the bundle": one per asset, and one per recipe surface.

    A recipe surface's tiles are one tileset row, grouped by the surface name the
    entries carry in their `recipe`. Thumbnails are the assets' refined pictures.
    """
    tile_counts: dict[str, int] = {}
    for entry in entries:
        if entry.recipe is not None:
            tile_counts[entry.recipe.surface] = tile_counts.get(entry.recipe.surface, 0) + 1
    # A tileset shows the tile whose recipe says variant 1, place nw, else its first tile.
    covers: dict[str, str] = {}
    for entry in entries:
        if entry.recipe is not None and (
            entry.recipe.surface not in covers
            or (entry.variant == 1 and entry.recipe.place == "nw")
        ):
            covers[entry.recipe.surface] = entry.name
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        shown = covers[entry.recipe.surface] if entry.recipe is not None else entry.name
        thumb = f"/directions/{direction_name}/assets/{shown}/refined.png"
        size = f"{entry.width} x {entry.height} px"
        if entry.recipe is None:
            kind = entry.kind.value.capitalize()
            items.append({"name": entry.name, "kind": kind, "size": size, "thumb": thumb})
        elif entry.recipe.surface not in seen:
            seen.add(entry.recipe.surface)
            items.append(
                {
                    "name": entry.recipe.surface,
                    "kind": "Tileset",
                    "size": f"{tile_counts[entry.recipe.surface]} tiles, {size} each",
                    "thumb": thumb,
                }
            )
    return items


def written_files(manifest: Manifest) -> list[str]:
    """Every file the bundle holds, as its manifest lists them, plus the manifest itself."""
    names = {name for entry in manifest.entries for name in entry.files.values()}
    return [MANIFEST_FILE, *sorted(names)]


def aseprite_files(manifest: Manifest, scene: str | None) -> tuple[list[str], list[str]]:
    """Every room file and each recipe surface's tileset file the manifest lists.

    The kept scene's room file comes first when it has one.
    """
    rooms = sorted(
        (e.name != scene, e.files["room"]) for e in manifest.entries if "room" in e.files
    )
    tilesets = sorted(
        {
            e.files["aseprite"]
            for e in manifest.entries
            if e.recipe is not None and "aseprite" in e.files
        }
    )
    return [name for _, name in rooms], tilesets


@router.get("/directions/{direction}/export")
def screen(
    request: Request,
    direction: str,
    bench: Bench,
    bundle: str | None = None,
    note: Annotated[list[str] | None, Query()] = None,
) -> Response:
    """The room and the bundle list, or, with `?bundle=`, the bundle just written.

    `write` redirects here rather than rendering, so a browser reload re-runs this
    read instead of posting the form again. The manifest and the room are read back
    off the bundle's folder, so the page shows what is really in it. `note` carries
    the export's own sentences across that redirect (they are kept nowhere else).
    """
    found = require_direction(bench, direction)
    generation = journey.followed_scene(bench.store, found.name)
    scene = scene_holding(bench.store, found.name, generation) if generation else None

    folder = None
    manifest = None
    if bundle is not None:
        folder = bench.store.export_dir(check_name(bundle, "bundle"))
        if not (folder / MANIFEST_FILE).exists():
            raise FileNotFoundError(
                f"There is no bundle called {bundle} in this project's exports."
            )
        manifest = bench.store.read_model(folder / MANIFEST_FILE, Manifest)

    preview = None
    if scene is not None:
        preview = (
            room_from_bundle(bench.store, found.name, folder, scene)
            if folder is not None and manifest is not None
            else preview_room(bench.store, found, scene)
        )

    room = None
    if preview is not None and preview.layers is not None:
        layers = preview.layers
        room = {
            "width": layers.tiles.width,
            "height": layers.tiles.height,
            "tiles": _data_url(layers.tiles),
            "structure": _data_url(layers.structure),
            "props": _data_url(layers.props),
        }

    written = None
    if folder is not None and manifest is not None:
        room_files, tilesets = aseprite_files(manifest, scene)
        written = {
            "path": str(folder.resolve()),
            "files": written_files(manifest),
            "room_files": room_files,
            "tilesets": tilesets,
        }

    # Each sentence once, in order: the export's own (carried in `note` after an export, or
    # the scratch export's before it), then the room's when there is a room to show them under.
    given = note if written is not None else (preview.warnings if preview is not None else [])
    shown = preview.notes if preview is not None and room is not None else []
    return page(
        request,
        "export.html",
        direction=found,
        **kept_scene_context(bench, found.name, StepKey.EXPORT, "Export"),
        scene=scene,
        room=room,
        room_notes=preview.notes if preview is not None else [],
        items=bundle_items(found.name, preview.entries if preview is not None else []),
        notes=list(dict.fromkeys([*(given or []), *shown])),
        written=written,
    )


@router.post("/directions/{direction}/export")
def write(
    direction: str,
    bench: Bench,
    scale: Annotated[list[Annotated[int, Field(ge=1)]] | None, Form()] = None,
) -> Response:
    """Write the bundle, then redirect to the screen that shows it.

    `Field(ge=1)` on the list's *element* type turns `scale=0` and a negative value
    into a 400 straight out of request validation; the upper bound, and the real
    source of truth for both ends, is `stages.export._check_scales`, which `export`
    calls before writing anything.

    Native size is always written whether or not the "2x" switch is on; the core
    adds it. `scale or []` rather than bare `tuple(scale)` because a POST with no
    `scale` at all (curl can send one) is a native-only bundle, not a `TypeError`.

    Post, redirect, get: a reload of the result re-runs the GET, never a second
    export. The export's own warnings ride along as `note`, since nothing else keeps
    them.
    """
    found = require_direction(bench, direction)
    bundle = export(bench.store, found, scales=tuple(scale or []))
    query = urlencode(
        [("bundle", bundle.directory.name), *(("note", w) for w in bundle.warnings)]
    )
    return see_other(f"/directions/{found.name}/export?{query}")
