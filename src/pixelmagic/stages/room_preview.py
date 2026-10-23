"""A picture of a kept scene's room for the export screen, with no Aseprite in it.

The room file export writes is built by Aseprite from a `RoomPlan`. The page
needs to see the same room as pictures: `room_from_bundle` reads one from a
finished bundle, and `preview_room` makes one before any export by writing a
scratch bundle. Both plan the room with the same code export uses and paste it
with `room.compose_room`, so what is shown is what the room file will hold.
Every sentence a person reads is made here; the page only displays them.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from pixelmagic.model import AssetRecord, Direction, Manifest, ManifestEntry
from pixelmagic.stages.export import (
    ASSET_FILE,
    MANIFEST_FILE,
    collapse_notes,
    export,
    surface_tile_sets,
)
from pixelmagic.stages.room import RoomLayers, compose_room, plan_room
from pixelmagic.store import Store


@dataclass
class RoomPreview:
    """What the export screen gets: the room's layers, or none, and plain sentences."""

    #: None when there is no room to show; `notes` then says why.
    layers: RoomLayers | None
    #: Pieces left out of the room, or the reasons there is no room.
    notes: list[str] = field(default_factory=list)
    #: What an export would write, as the scratch export's own manifest entries. Filled only
    #: by `preview_room`; a preview made from a finished bundle has the bundle's manifest instead.
    entries: list[ManifestEntry] = field(default_factory=list)
    #: The scratch export's own sentences for anything it left out or flagged. Filled only
    #: by `preview_room`.
    warnings: list[str] = field(default_factory=list)


def room_from_bundle(store: Store, direction: str, bundle: Path, scene: str) -> RoomPreview:
    """The room of the kept scene `scene` as pictures, from a bundle folder holding `manifest.json`.

    A failure to read or draw the room comes back as a preview with no layers and one sentence.

    Pictures are read from the bundle itself, so a surface's touch-ups from its master
    file show once the bundle has been exported with them.
    """
    manifest = store.read_model(bundle / MANIFEST_FILE, Manifest)
    entries = {entry.name: entry for entry in manifest.entries}
    if scene not in entries:
        return RoomPreview(
            None,
            [
                f"{scene} is not in this bundle (it has to be refined first), "
                "so there is no room to show."
            ],
        )
    notes: list[tuple[str, str]] = []
    # A room that cannot be shown must not cost the page its export button: a record that
    # will not read, or a picture that is gone since the export, becomes a sentence.
    try:
        records = {
            name: store.read_model(store.asset_dir(direction, name) / ASSET_FILE, AssetRecord)
            for name in entries
        }
        plan = plan_room(
            store, direction, bundle, records[scene], manifest.entries, records,
            surface_tile_sets(manifest), manifest.cell_size, notes,
        )
        sentences = collapse_notes(notes)
        if plan is None:
            return RoomPreview(
                None, sentences or ["Nothing cut from this picture can be placed in a room yet."]
            )
        return RoomPreview(compose_room(plan, manifest.cell_size), sentences)
    except (OSError, KeyError, ValueError) as error:
        return RoomPreview(None, [f"The room could not be built from this bundle: {error}"])


def preview_room(store: Store, direction: Direction, scene: str) -> RoomPreview:
    """The room of `scene` as pictures before anything is exported.

    A scratch bundle at scale 1 is written in a temporary folder outside the project,
    always deleted, and read with `room_from_bundle`. It holds the recipe's own tiles:
    a surface's master-file touch-ups are read through Aseprite, which this never runs,
    so they show only after a real export. The scratch export's manifest entries and warnings
    come back on the preview, so a page lists exactly what an export would write.
    """
    with tempfile.TemporaryDirectory(prefix="pixelmagic-room-preview-") as scratch:
        bundle = Path(scratch)
        written = export(store, direction, scales=(1,), into=bundle)
        preview = room_from_bundle(store, direction.name, bundle, scene)
        return replace(preview, entries=written.manifest.entries, warnings=written.warnings)
