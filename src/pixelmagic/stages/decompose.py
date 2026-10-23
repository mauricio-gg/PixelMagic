"""Turn one approved picture into named pieces, and keep pieces and scenes as assets.

`decompose` draws a recipe surface's tiles from the picture's own colours; props
come only from the object road (`suggest`, `pick`, `redraw`). The rest of this
module is what happens to pieces afterwards: keeping them as assets, renaming and
deleting them, and the scene's pitch and the props' anchors.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ValidationError

from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    Cutout,
    Decomposition,
    Direction,
    GenerationRecord,
    PitchSource,
    RecipeRecord,
    RefineReport,
    RefinerSettings,
    ReviewStatus,
    SheetKind,
)
from pixelmagic.naming import (
    Surface,
    UnusableName,
    check_name,
    kebab_case,
    kindless_surface_sentence,
    scene_name,
    typed_slug,
    unique_name,
)
from pixelmagic.refiner.base import LOW_GRID_CONFIDENCE, Refiner
from pixelmagic.segmenter.base import Mask
from pixelmagic.stages.generate import NotApproved
from pixelmagic.stages.layout import (
    box_outside_sentence,
    check_anchor,
    check_pitch,
    default_anchor,
    footprint_for,
    target_for,
)
from pixelmagic.stages.recipes import (
    BLOCK_TILES,
    PLACES,
    RECIPE_CELL_SIZE,
    VARIANTS,
    RecipeRefused,
    RecipeTileRefused,
    Shade,
    check_seams,
    make_tiles,
    read_shades,
)
from pixelmagic.stages.refine import (
    REPORT_FILE,
    old_pieces_sentence,
    require_asset,
    validate_target,
)
from pixelmagic.stages.slice import SliceBox
from pixelmagic.store import AlreadyExists, Store, claim_id

_LOGGER = logging.getLogger(__name__)

#: Cohesion and clean-up must never run for a pitch reading (spec locked
#: decision 19): the point of `analyze` is to measure the picture's own grid,
#: not PixelRefiner's cohesion-processed redraw of it.
PITCH_ANALYZE_SETTINGS = RefinerSettings(advanced={"cohesion": False, "cleanup": False})

#: What each sheet kind becomes once promoted to an asset.
SHEET_ASSET_KINDS: dict[SheetKind, AssetKind] = {
    SheetKind.PROPS: AssetKind.PROP,
    SheetKind.TILES: AssetKind.TILE,
}


class NoPicks(ValueError):
    """No object is picked, so there is nothing to cut out or redraw.

    Raised in two places: `redraw_picks` when nothing is picked yet, and
    `decompose` when no surface is typed, since props are cut out object by
    object from picks a person makes and a surface is all `decompose` draws.

    A `ValueError`, so a face that catches that family at its boundary still
    catches this. The message says only what is wrong and names the way
    forward, so a face can print it as it is.
    """


def _all_cutout_names(store: Store, direction_name: str) -> set[str]:
    """Every cutout name ever sliced in this direction, promoted or not.

    `promote_cutouts` deliberately defers promotion — a person may rename
    or delete pieces first — so decompose, decompose again, then review
    and promote both is a supported order. Seeding `taken` from
    `store.list_assets` alone only stops a name collision when promotion
    happens *between* the two runs; it lists promoted assets only. Walking
    every decomposition's `cutouts/` folder catches a same-named cutout from
    an earlier, still-unpromoted run too, so the two runs never choose the
    same name and one promotion cannot silently overwrite the other's asset.
    """
    names: set[str] = set()
    root = store.decompositions_dir(direction_name)
    if not root.is_dir():
        return names
    for decomposition_folder in root.iterdir():
        cutouts_root = decomposition_folder / "cutouts"
        if not cutouts_root.is_dir():
            continue
        names.update(folder.name for folder in cutouts_root.iterdir() if folder.is_dir())
    return names


def layout_cutout(cutout: Cutout, cell_size: int) -> Cutout:
    """Fill in how a prop sits on the grid: its footprint in cells and its anchor.

    Locked decision 6 of the M3 spec. The footprint is the target size divided
    by the direction's cell size, rounded up on both axes; the anchor — the
    point that meets the floor — defaults to bottom centre. Both come from the
    target and nothing else: the tool never guesses an anchor from what the
    prop is. Tiles have neither (a tile is the grid), and a piece with no
    target yet cannot have either, so those come back exactly as they are.
    The redraw is the one caller: it gives each rebuilt piece its target, then
    comes through here. A scene gets its target in `file_promotion` and again,
    later, in `set_scene_pitch`, and never has a footprint or an anchor — this
    is prop-only, and scenes never reach here.
    """
    if cutout.kind is not SheetKind.PROPS or cutout.target is None:
        return cutout
    return cutout.model_copy(
        update={
            "footprint": footprint_for(cutout.target, cell_size),
            "anchor": default_anchor(cutout.target),
        }
    )


#: A recipe surface's tile-name suffixes, `-v1-nw` to `-v8-se`, in `PLACES` order.
_RECIPE_SUFFIXES = [f"-v{number}-{place}" for number in range(1, VARIANTS + 1) for place in PLACES]


def _recipe_owners_of(
    store: Store, direction_name: str, name: str
) -> set[tuple[str | None, str | None]]:
    """Who owns every sidecar that carries `name`, as `(recipe surface, source picture)` pairs.

    Looks at the kept asset's `asset.json` and at every earlier decomposition's
    `cutout.json` for that name (the picture of a cutout is its decomposition's
    `source_generation`). A part that is not a recipe tile has `None` for the
    surface; a sidecar that will not parse is `(None, None)`, so the name is
    treated as someone else's rather than guessed at.
    """
    found: set[tuple[str | None, str | None]] = set()
    asset_file = store.asset_dir(direction_name, name) / "asset.json"
    if asset_file.exists():
        try:
            record = store.read_model(asset_file, AssetRecord)
            found.add((record.recipe.surface if record.recipe else None, record.source_generation))
        except (OSError, ValidationError):
            found.add((None, None))
    root = store.decompositions_dir(direction_name)
    for cutout_file in root.glob(f"*/cutouts/{name}/cutout.json") if root.is_dir() else ():
        try:
            recorded = store.read_model(cutout_file, Cutout)
            run = load_decomposition(store, direction_name, cutout_file.parents[2].name)
            found.add((recorded.recipe.surface if recorded.recipe else None, run.source_generation))
        except (OSError, ValidationError):
            found.add((None, None))
    return found


def _reserve_recipe_names(
    store: Store,
    direction_name: str,
    generation: str,
    surface: Surface,
    taken: set[str],
    reserved: set[str],
) -> None:
    """Refuse a recipe surface whose names belong to something else, then reserve them.

    The checked names are the bare surface name (export writes
    `<surface>.aseprite` under it, which would overwrite a piece of that name)
    and the 32 tile names. A name already in `taken` is a clash unless every
    sidecar carrying it shows a recipe tile of this same surface drawn on this
    same picture (`generation`): that is a re-run, and re-runs are allowed. The
    same name on another picture would take that picture's tiles and assets
    over, so it is a clash. A name reserved earlier in this same call
    is always a clash (the same surface typed twice). Old patch and bordered
    names of the same base (`-r0c0`, `-centre`) are not checked and never block.
    """
    for name in [surface.name] + [f"{surface.name}{suffix}" for suffix in _RECIPE_SUFFIXES]:
        if name not in taken:
            continue
        # A name reserved earlier in this call is a clash before any sidecar is
        # read: it is the same surface typed twice, never a re-run.
        if name not in reserved and name != surface.name and _recipe_owners_of(
            store, direction_name, name
        ) == {(surface.name, generation)}:
            continue
        if name == surface.name:
            raise RecipeRefused(
                f"surface {surface.text!r}: {name} is already used by another piece, and "
                f"its .aseprite file would overwrite that piece's; type a different name"
            )
        raise RecipeRefused(
            f"surface {surface.text!r}: the name {name} is already used by something "
            f"that is not this surface; type a different name"
        )
    reserved.update(f"{surface.name}{suffix}" for suffix in _RECIPE_SUFFIXES)
    reserved.add(surface.name)
    taken.update(reserved)


def _write_outline(mask: Mask, path: Path) -> None:
    """Save an outline as an 8-bit PNG, white inside, the way pick masks are saved.

    Written to a scratch file and renamed into place so an interrupted write
    never leaves half an outline. (`pick._write_mask_png` does the same, but
    `pick` imports this module, so the few lines are repeated here.)
    """
    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    Image.fromarray(mask.pixels.astype(np.uint8) * 255, "L").save(scratch, format="PNG")
    os.replace(scratch, path)


def _surface_outline(width: int, height: int, surface: Surface) -> Mask:
    """The joined shapes of a recipe surface as a mask, cropped to what it covers.

    Every shape is filled as a polygon onto a mask the size of the picture, so
    whatever lies past the edge is clipped and corners outside the picture are
    fine. The mask is cropped to the box around the covered pixels, which is the
    form `outline.png` and the tiles' `bbox` are saved in. Raises `RecipeRefused`
    when the shapes cover no pixel of the picture.
    """
    canvas = Image.new("L", (width, height), 0)
    drawing = ImageDraw.Draw(canvas)
    for shape in surface.shapes:
        drawing.polygon(list(shape), fill=255)
    covered = np.asarray(canvas) > 0
    rows = np.flatnonzero(covered.any(axis=1))
    columns = np.flatnonzero(covered.any(axis=0))
    if not len(rows):
        raise RecipeRefused(
            f"surface {surface.text!r}: its shapes cover no pixel of the "
            f"{width} × {height} picture; draw them over the surface"
        )
    x0, y0, x1, y1 = columns[0], rows[0], columns[-1] + 1, rows[-1] + 1
    return Mask(box=(int(x0), int(y0), int(x1), int(y1)), pixels=covered[y0:y1, x0:x1])


def _check_cell_size(direction: Direction, text: str) -> None:
    """Refuse a recipe surface in a direction whose cell size is not `RECIPE_CELL_SIZE`.

    One check for `decompose` and the colour reader, so they cannot disagree.
    `text` is only there for the sentence.
    """
    if direction.cell_size != RECIPE_CELL_SIZE:
        raise RecipeRefused(
            f"surface {text!r}: recipe tiles are made at "
            f"{RECIPE_CELL_SIZE} × {RECIPE_CELL_SIZE}, and this direction's cell "
            f"size is {direction.cell_size}"
        )


def _check_box_inside(
    text: str, box: tuple[int, int, int, int], width: int, height: int
) -> None:
    """Refuse a colour box that is not wholly inside a `width` × `height` picture.

    Shared by `decompose` and the colour reader for the same reason as
    `_check_cell_size`. `text` is only there for the sentence.
    """
    outside = box_outside_sentence(box, width, height)
    if outside is not None:
        raise RecipeRefused(f"surface {text!r}: its {outside}")


def _check_surface_areas(
    source: GenerationRecord, surfaces: Sequence[Surface]
) -> dict[int, Mask]:
    """Refuse a recipe surface whose box is not inside the picture or whose shapes miss it.

    Pure arithmetic on the picture's size, so `decompose` runs it before a
    decomposition folder is made: a typo leaves no empty record behind. The
    result is each surface's outline, keyed by `id(surface)` so two surfaces
    typed alike still get one each.
    """
    with Image.open(source.image_path) as picture:
        width, height = picture.size
    outlines: dict[int, Mask] = {}
    for surface in surfaces:
        assert surface.box is not None
        _check_box_inside(surface.text, surface.box, width, height)
        outlines[id(surface)] = _surface_outline(width, height, surface)
    return outlines


def _save_outlines(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    surfaces: Sequence[Surface],
    outlines: dict[int, Mask],
) -> None:
    """Save each recipe surface's joined shapes as `surfaces/<name>/outline.png`."""
    for surface in surfaces:
        folder = store.surface_dir(direction.name, decomposition_id, surface.name)
        folder.mkdir(parents=True, exist_ok=True)
        _write_outline(outlines[id(surface)], folder / "outline.png")


def _read_box(
    source: GenerationRecord, kind: str, box: tuple[int, int, int, int], text: str
) -> list[Shade]:
    """Read a recipe's colours from the picture's own pixels inside `box`, and nowhere else.

    Every shade refusal (`RecipeRefused`) comes from `read_shades`. `text` is
    only there for the sentences.
    """
    with Image.open(source.image_path) as picture:
        scene = np.asarray(picture.convert("RGBA"))
    inside = np.zeros(scene.shape[:2], dtype=bool)
    x0, y0, x1, y1 = box
    inside[y0:y1, x0:x1] = True
    return read_shades(kind, scene, inside, surface=text)


def read_box_shades(
    direction: Direction,
    source: GenerationRecord,
    kind: str,
    box: tuple[int, int, int, int],
    name: str,
) -> list[Shade]:
    """The colours a recipe would read from `box`, without drawing any tile.

    For a face that shows the colours while a person moves the box. It refuses
    exactly as `decompose` would for the same parts and with the same sentences:
    a direction whose cell size is not `RECIPE_CELL_SIZE`, a box not wholly
    inside the picture, and everything `read_shades` refuses (an unknown kind,
    a box that is too small, too few colours, a shade that is too thin). `box`
    is `(x0, y0, x1, y1)`, right and bottom exclusive, corners already in
    order (`naming.parse_box`). `name` is only there for the sentences.
    """
    _check_cell_size(direction, name)
    with Image.open(source.image_path) as picture:
        width, height = picture.size
    _check_box_inside(name, box, width, height)
    return _read_box(source, kind, box, name)


#: A recipe surface's drawn tiles and the shades they were drawn from.
_DrawnTiles = tuple[list[np.ndarray], list[Shade]]


def _draw_recipe_tiles(source: GenerationRecord, surface: Surface) -> _DrawnTiles:
    """Read a recipe surface's colours and draw its tiles, writing nothing.

    The colours come from the picture's own pixels inside the surface's box and
    nowhere else. The seam check runs on the finished set, so a broken recipe
    (`RecipeSeamBroken`) and every shade refusal happen here, before any tile
    is written.
    """
    assert surface.kind is not None and surface.box is not None
    shades = _read_box(source, surface.kind, surface.box, surface.text)
    tiles = make_tiles(surface.kind, shades, surface.name)
    check_seams(tiles, kind=surface.kind, surface=surface.name)
    return tiles, shades


def _write_recipe_tiles(
    store: Store,
    direction: Direction,
    decomposition: Decomposition,
    surface: Surface,
    outline: Mask,
    drawn: _DrawnTiles,
) -> None:
    """Write a drawn recipe surface's tiles as cutouts.

    The image service is never asked. No sheet exists, so `tile_sheets` gets
    no entry; the cutouts' `sheet` is only the surface's name, a label shared
    by all of them.
    """
    assert surface.kind is not None and surface.box is not None
    tiles, shades = drawn
    decomposition_id = decomposition.id
    master = store.surface_master(direction.name, surface.name)
    if master.exists():
        decomposition.warnings.append(
            f"{surface.name}: its master file {master.name} already exists and is kept, so "
            f"export takes this surface's tiles from it and not from this new run; "
            f"delete the file to start fresh"
        )
    for index, (suffix, tile) in enumerate(zip(_RECIPE_SUFFIXES, tiles, strict=True)):
        name = f"{surface.name}{suffix}"
        cutout_folder = store.cutout_dir(direction.name, decomposition_id, name)
        cutout_folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(tile, "RGBA").save(cutout_folder / "source.png", format="PNG")
        cutout = Cutout(
            name=name, ai_name=surface.text, sheet=surface.name, kind=SheetKind.TILES,
            bbox=outline.box, target=(RECIPE_CELL_SIZE, RECIPE_CELL_SIZE),
            variant=index // BLOCK_TILES + 1,
            recipe=RecipeRecord(
                kind=surface.kind, layout="block",
                place=PLACES[index % BLOCK_TILES],
                surface=surface.name,
                shapes=[list(shape) for shape in surface.shapes], box=surface.box,
                shades=[f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in shades],
                shares=[share for _, share in shades],
            ),
        )
        store.write_model(cutout_folder / "cutout.json", cutout)
        decomposition.cutouts.append(cutout)


def decompose(
    store: Store,
    direction: Direction,
    source: GenerationRecord,
    *,
    surfaces: Sequence[Surface] = (),
    now: datetime | None = None,
) -> Decomposition:
    """Draw one set of tiles per typed recipe surface from an approved picture.

    Props are cut out object by object from picks a person makes (`suggest`,
    `pick`, `redraw`), never here, so with no `surfaces` typed there is nothing
    to draw and `NoPicks` is raised. The rule lives here so every face agrees
    on it.

    A surface is typed with a recipe kind, shapes and a box (`naming.parse_surface`).
    No image service is asked and no object finder is used: the shapes joined
    are its outline (saved as `outline.png`), its colours are read from the box,
    and the kind's recipe draws its tiles (32 `<name>-vN-nw` ... `-se`: 8 blocks
    of four). A recipe surface in a direction whose cell size is not 16, with a
    box not wholly inside the picture, or whose shapes cover no pixel of it, is
    refused with `RecipeRefused` before anything is made.

    Only an approved `source` may be used. This rule used to live in the web
    page alone (Guardian minor 3: the "cut it up" button just did not appear on
    a picture that was not approved, but nothing stopped a stale tab or a
    hand-made POST from reaching the route anyway), which meant every other
    face had to invent the same check or go without it. CLAUDE.md's rule
    that something every face needs belongs in the core, not in one of them,
    is why it is checked here, first, before anything is written: a picture
    nobody has said yes to yet is not raw material for an asset.
    """
    if not surfaces:
        raise NoPicks(
            "no surface was typed to tile, and props are not made here "
            "(they are picked object by object)"
        )
    if source.review.status is not ReviewStatus.APPROVED:
        raise NotApproved(
            f"only an approved picture can be cut up; this one is "
            f"{source.review.status}"
        )
    # Only a hand-built `Surface` can be kindless here (`parse_surface`
    # refuses it first); the old plain road stays shut for every caller,
    # before a folder is made.
    for surface in surfaces:
        if surface.kind is None:
            raise RecipeRefused(kindless_surface_sentence(surface.text))
    _check_cell_size(direction, surfaces[0].text)
    outlines = _check_surface_areas(source, surfaces)
    moment = now or datetime.now(UTC)
    # Seeded from every asset already staged plus every cutout any earlier
    # decompose run in this direction ever sliced (see `_all_cutout_names`):
    # a second decompose run that also draws a "rug" must not choose the
    # same name as the first one's, whether or not the first has been
    # promoted yet.
    taken: set[str] = set(store.list_assets(direction.name))
    taken |= _all_cutout_names(store, direction.name)
    reserved_recipe_names: set[str] = set()

    # Claimed atomically: two draws in the same second must not share a folder.
    decomposition_id = claim_id(store.decompositions_dir(direction.name), now=moment)
    folder = store.decomposition_dir(direction.name, decomposition_id)
    decomposition = Decomposition(
        id=decomposition_id, created=moment, source_generation=source.request.id
    )

    try:
        _save_outlines(store, direction, decomposition_id, surfaces, outlines)
        # Every recipe refusal (a name in use, a shade the picture cannot give,
        # a broken seam) happens here, before any tile is written.
        drawn: dict[int, _DrawnTiles] = {}
        for surface in surfaces:
            drawn[id(surface)] = _draw_recipe_tiles(source, surface)
            _reserve_recipe_names(
                store, direction.name, source.request.id, surface, taken, reserved_recipe_names
            )
        for surface in surfaces:
            _write_recipe_tiles(
                store, direction, decomposition, surface,
                outlines[id(surface)], drawn[id(surface)],
            )
    except BaseException:
        if decomposition.cutouts:
            # A refusal partway through still leaves any earlier surface's finished
            # tiles on disk; the sidecar is the only thing that makes them
            # reachable again, so it is written before the refusal goes on.
            store.write_model(folder / "decomposition.json", decomposition)
        else:
            # Refused before any tile was written: the folder holds only an
            # outline and no record, so it goes rather than lingering empty.
            try:
                _remove_cutout_folder(folder)
            except OSError:
                # The refusal the person should see must win over a cleanup
                # failure, but the leftover folder is reported, not hidden.
                _LOGGER.exception(
                    "could not remove decomposition %s after a refusal: %s",
                    decomposition_id,
                    folder,
                )
        raise
    store.write_model(folder / "decomposition.json", decomposition)
    return decomposition


class PreparedPromotion(BaseModel):
    """What `prepare_promotion` reads off a picture before anything is filed.

    Split out of `promote_generation` so a caller that must record an
    approval decision in between can run the refiner call first:
    `prepare_promotion` is the only part of the old function that
    can raise, so running it before the decision is written means a refiner
    failure leaves the generation pending rather than approved with no asset
    to show for it — the caller can retry the whole approval once the refiner
    is reachable again.
    """

    pitch: float
    target: tuple[int, int]
    #: `MEASURED`, `UNSURE` or `FALLBACK`: how far the pitch can be trusted.
    source: PitchSource = PitchSource.MEASURED


def prepare_promotion(
    refiner: Refiner, direction_name: str, generation_id: str, image_path: Path
) -> PreparedPromotion:
    """Read the pitch and target a promoted picture will need, writing nothing.

    The reading is filed honestly (`PreparedPromotion.source`): no pitch, or
    one `check_pitch` refuses, is not used and the pitch is 1.0 labelled
    `FALLBACK`; so is any reading from a refiner that does not honour a
    forced size (the stand-in), which measures nothing; a usable pitch the
    refiner flagged with `LOW_GRID_CONFIDENCE` (its own threshold, none of
    ours) is kept and labelled `UNSURE`; anything else is `MEASURED`. No
    reading, an out-of-range reading and an `UNSURE` one are each logged at
    warning level, saying which case it was; the stand-in's is not, since it
    is the expected default and `pitch_note` already tells the person. Never
    printed — `mcp_server.py` uses standard output as its protocol, so this
    goes through `logging`, which lands on stderr. `pitch_note` is the sentence
    a face shows the person.

    Raises `RefinerError` when `refiner.analyze` does; the caller has written
    nothing yet, so the picture this was asked about stays exactly as it was.
    """
    reading = refiner.analyze(image_path, PITCH_ANALYZE_SETTINGS)
    pitch = 1.0
    source = PitchSource.FALLBACK
    if reading.pitch is None:
        _LOGGER.warning(
            "could not read a pixel size for the scene from generation %s in "
            "direction %r (no reading); its pitch is 1.0 (one source pixel "
            "per art pixel)",
            generation_id, direction_name,
        )
    else:
        try:
            check_pitch(reading.pitch)
        except ValueError:
            _LOGGER.warning(
                "the pixel size read for the scene from generation %s in "
                "direction %r (%r) is out of range; its pitch is 1.0 (one "
                "source pixel per art pixel)",
                generation_id, direction_name, reading.pitch,
            )
        else:
            pitch = reading.pitch
            # A stand-in's answer is a stock number, not a measurement of
            # this picture, so it is filed as "no reading".
            source = (
                PitchSource.MEASURED
                if refiner.honours_forced_size
                else PitchSource.FALLBACK
            )
            if LOW_GRID_CONFIDENCE in reading.warnings:
                source = PitchSource.UNSURE
                _LOGGER.warning(
                    "the refiner is unsure of the pixel size it read for the "
                    "scene from generation %s in direction %r (%s, confidence "
                    "%.2f); its pitch is kept",
                    generation_id, direction_name, pitch, reading.confidence,
                )
    with Image.open(image_path) as picture:
        width, height = picture.size
    target = target_for(SliceBox(left=0, top=0, right=width, bottom=height), pitch)
    return PreparedPromotion(pitch=pitch, target=target, source=source)


def pitch_note(store: Store, direction_name: str, asset_name: str) -> str | None:
    """A plain sentence for a kept scene whose pitch should be checked, else None.

    Reads the asset's `asset.json` (so a face passes only the name
    `file_promotion` returned).

    Said for `FALLBACK` (PixelMagic had no usable reading and used 1.0) and `UNSURE` (the
    refiner flagged its own reading). Every face that keeps a scene shows it
    after its success line; the web page shows `pitch_source` on the refine
    screen instead.
    """
    record = store.read_model(
        store.asset_dir(direction_name, asset_name) / "asset.json", AssetRecord
    )
    if record.pitch_source is PitchSource.FALLBACK:
        what = (
            "PixelMagic could not use a reading of the scene's pixel size from "
            "the refiner, so it used 1.0."
        )
    elif record.pitch_source is PitchSource.UNSURE:
        what = (
            f"The refiner was not sure of the scene's pixel size (pitch {record.pitch})."
        )
    else:
        return None
    return f"{what} Check it, and set the scene's pitch before refining."


def scene_asset_name(store: Store, direction: Direction, record: GenerationRecord) -> str:
    """The name a whole picture would be filed under as an asset right now.

    `naming.scene_name` on the picture's `request.subject` (what the person
    typed, or the imported file's stem), numbered with `unique_name` against
    the direction's assets and every cut-out name, kept or not: a later `keep`
    of a decomposition writes into `assets/<cutout name>` without asking, so a
    scene sharing a still-unkept cut-out's name would be overwritten.
    `file_promotion` uses it, and a face can call it to say the name before
    the person keeps the picture.
    """
    taken = (
        set(store.list_assets(direction.name))
        | _all_cutout_names(store, direction.name)
        | _recipe_surface_names(store, direction.name)
    )
    return unique_name(scene_name(record.request.subject), taken)


def file_promotion(
    store: Store,
    direction: Direction,
    record: GenerationRecord,
    prepared: PreparedPromotion,
    *,
    kind: AssetKind = AssetKind.SCENE,
    name: str | None = None,
) -> str:
    """Write a picture already read by `prepare_promotion` as an asset.

    Locked decision 3 lists scenes alongside props and tilesets, so an
    approved scene has to be able to reach an export bundle without being cut
    up first. The picture is copied unchanged and refine and export then
    treat it like any other asset. The pitch is filed with the source
    `prepare_promotion` gave it (`MEASURED`, `UNSURE` or `FALLBACK`). Returns
    the name it was filed under: the
    typed `name` when one is given (already passed by `check_scene_name`, which
    `keep_scene` runs first), else `scene_asset_name`'s suggestion.

    Takes no path that can raise for a reason worth telling a caller apart
    from an ordinary I/O failure: everything that can refuse (the refiner
    call) already ran, in `prepare_promotion`.
    """
    if name is None:
        name = scene_asset_name(store, direction, record)
    folder = store.asset_dir(direction.name, name)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source.png").write_bytes(Path(record.image_path).read_bytes())
    store.write_model(
        folder / "asset.json",
        AssetRecord(
            name=name,
            kind=kind,
            source_generation=record.request.id,
            source_prompt=record.request.prompt,
            service=record.request.service,
            pitch=prepared.pitch,
            pitch_source=prepared.source,
            target=prepared.target,
        ),
    )
    return name


def promote_generation(
    store: Store,
    direction: Direction,
    record: GenerationRecord,
    *,
    refiner: Refiner,
    kind: AssetKind = AssetKind.SCENE,
    name: str | None = None,
) -> str:
    """Stage a whole approved picture as an asset in its own right.

    A thin wrapper over `prepare_promotion` and `file_promotion`, for a caller
    that has no approval decision to write in between the two. The command
    line's `generate --as-asset` is exactly that
    caller — it approves inline, as part of generating, before this ever
    runs — but that does not make a refiner failure here free: the picture is
    already approved by the time `prepare_promotion` can raise, and this
    function has no decision left to protect, so a refusal leaves that
    picture approved with no asset. `cli.py`'s `_asset_not_kept` is the one
    that has to say so, by name, rather than reporting the refiner's words
    alone. A face that does have a decision to record between the risky
    read and the write should call the two halves itself instead, in that
    order — which is what buys the web page and the MCP server their
    "still pending, not stranded" retry.
    """
    prepared = prepare_promotion(refiner, direction.name, record.request.id, record.image_path)
    return file_promotion(store, direction, record, prepared, kind=kind, name=name)


def scene_holding(store: Store, direction_name: str, generation_id: str) -> str | None:
    """The name of the scene asset already made from this generation, if any.

    An asset folder whose `asset.json` is missing or will not parse is not
    this one, so it is skipped rather than allowed to stop the check.
    """
    for name in store.list_assets(direction_name):
        try:
            found = store.read_model(
                store.asset_dir(direction_name, name) / "asset.json", AssetRecord
            )
        except (OSError, ValueError):
            continue
        if found.kind is AssetKind.SCENE and found.source_generation == generation_id:
            return name
    return None


def check_scene_name(store: Store, direction: Direction, name: str) -> str:
    """Return `name` if a scene may be filed under it, or refuse it with a sentence.

    The one check for a name a person typed, used by `keep_scene` and
    `rename_scene`. `UnusableName` when it is not a safe name (`naming.check_name`'s
    rule); the sentence offers `naming.slug`'s form of it when there is one.
    A name is safe only in the form the suggested names take (`kebab_case`: at
    most four words, 60 characters), as for `Store.rename_direction`. Surrounding
    spaces are trimmed here, so every face accepts the same input and the
    trimmed name is what is returned.
    `AlreadyExists` when an asset of the direction has the name, or any cut-out
    does, kept or not (`_all_cutout_names`: a later keep of a decomposition
    writes into `assets/<cutout name>` without asking), or a recipe surface
    does (its bare name is its export file). A clash is refused, never
    numbered: the person typed that name on purpose.
    """
    name = name.strip()
    try:
        check_name(name, "scene name")
        if name != kebab_case(name, fallback=""):
            raise UnusableName(
                f"scene name {name!r} is too long: at most four words, 60 characters"
            )
    except UnusableName as error:
        suggestion = kebab_case(name, fallback="")
        way = f"; try {suggestion!r}" if suggestion else ""
        raise UnusableName(f"{error}{way}") from error
    if name in store.list_assets(direction.name):
        raise AlreadyExists(f"{direction.name} already has an asset called {name!r}")
    if name in _all_cutout_names(store, direction.name):
        raise AlreadyExists(
            f"{name!r} is already the name of a cut-out in {direction.name}; "
            "type a different name"
        )
    if name in _recipe_surface_names(store, direction.name):
        raise AlreadyExists(
            f"{name!r} is already a floor or wall surface of {direction.name}, whose "
            "export file would clash; type a different name"
        )
    return name


def _recipe_surface_names(store: Store, direction_name: str) -> set[str]:
    """The `recipe.surface` of every cut-out of every readable decomposition.

    A surface's bare name is its export file name and is never numbered, so a
    scene may not take it. A decomposition that will not read is skipped, as
    in `scene_holding`.
    """
    names: set[str] = set()
    for decomposition_id in store.list_decompositions(direction_name):
        try:
            decomposition = load_decomposition(store, direction_name, decomposition_id)
        except (OSError, ValueError):
            continue
        names.update(c.recipe.surface for c in decomposition.cutouts if c.recipe is not None)
    return names


def keep_scene(
    store: Store,
    direction: Direction,
    record: GenerationRecord,
    *,
    refiner: Refiner,
    name: str | None = None,
) -> str:
    """Keep an already-approved picture, imported or drawn, as the scene asset.

    This is the "keep it as the scene" step for a picture that was approved
    earlier, so it did not pass through `generate --as-asset`. It reuses
    `promote_generation` for the work. Refusals come first, before the
    refiner is called and before anything is written: `NotApproved` when the
    picture has not been approved, the store's `AlreadyExists` when a
    scene asset already holds this generation (keeping twice would file the
    same picture under a second name), and, when a typed `name` is given,
    whatever `check_scene_name` refuses. With no `name` the scene is filed
    under `scene_asset_name`'s suggestion. Returns the asset name.
    """
    if record.review.status is not ReviewStatus.APPROVED:
        raise NotApproved(
            f"generation {record.request.id} is {record.review.status.value}; "
            "approve it before keeping it as the scene"
        )
    held = scene_holding(store, direction.name, record.request.id)
    if held is not None:
        raise AlreadyExists(
            f"generation {record.request.id} is already kept as the scene, "
            f"asset {held}"
        )
    if name is not None and name.strip():
        name = check_scene_name(store, direction, name)
    else:
        name = None
    return promote_generation(
        store, direction, record, refiner=refiner, kind=AssetKind.SCENE, name=name
    )


def rename_scene(store: Store, direction: Direction, old: str, new: str) -> str:
    """Rename a kept scene: its folder, its `asset.json` and its `refine.json`.

    Allowed at any time, before or after export, as `Store.rename_direction`
    is: a bundle already on disk keeps the old name, because a bundle is a
    record of what was exported, not a live link; the next export writes the
    new one (manifest entry and `<new>-room.aseprite`). Nothing else names a
    scene by name: decompositions and picks name the generation id, and the
    room is planned from the asset's folder at export time.

    Refused before anything moves: `ValueError` for an asset that does not
    exist, `ValueError` for one that is not a scene, `AlreadyExists` for the
    scene's own name, whatever `check_scene_name` refuses for `new`, and an
    `asset.json` or `refine.json` that will not read. Returns the trimmed new name.

    The window is `rename_direction`'s: the folder moves before `asset.json`
    and `refine.json` are rewritten, so a failed write there leaves the folder
    on the new name with a sidecar still naming the old.
    """
    require_asset(store, direction.name, old)
    folder = store.asset_dir(direction.name, old)
    record = store.read_model(folder / "asset.json", AssetRecord)
    if record.kind is not AssetKind.SCENE:
        raise ValueError(f"{old!r} is a {record.kind.value}, not a scene; only a scene is renamed")
    if new.strip() == old:
        raise AlreadyExists(f"the scene is already called {old!r}")
    new = check_scene_name(store, direction, new)
    report = None
    if (folder / REPORT_FILE).exists():
        report = store.read_model(folder / REPORT_FILE, RefineReport)
    record.name = new
    folder.rename(store.asset_dir(direction.name, new))
    moved = store.asset_dir(direction.name, new)
    store.write_model(moved / "asset.json", record)
    if report is not None:
        report.asset = new
        store.write_model(moved / REPORT_FILE, report)
    return new


def load_decomposition(
    store: Store, direction_name: str, decomposition_id: str
) -> Decomposition:
    """Read a decomposition sidecar back off disk."""
    path = store.decomposition_dir(direction_name, decomposition_id) / "decomposition.json"
    return store.read_model(path, Decomposition)


def missing_run_sentence(
    store: Store,
    direction_name: str,
    run_id: str,
    suggest_step: str,
    making: str = "Suggesting objects on the picture",
) -> str:
    """The sentence for a cut-up run id this direction does not have.

    A picture (generation) id is the usual wrong thing to type here, so when a
    generation of the direction has that id the sentence says so and names the
    step that makes a cut-up run. `suggest_step` is that step as a face writes
    it, with `{id}` where the picture id goes (`pixelmagic suggest {id}` on the
    command line, `suggest_objects(generation_id={id!r})` in MCP), so the rule
    is here and each face only supplies its own words. `making` is the opening
    words naming that step (a pixel cut says "Starting a pixel cut on the picture"
    instead). Only the generation's
    `request.json` is looked for; nothing is loaded.
    """
    if (store.generation_dir(direction_name, run_id) / "request.json").exists():
        return (
            f"{run_id!r} is a picture id, not a cut-up run id. {making} "
            f"({suggest_step.format(id=run_id)}) makes a cut-up run and gives its id."
        )
    return f"{direction_name} has no cut-up run {run_id!r}"


def _reused_anchor(
    store: Store, direction: Direction, cutout: Cutout, folder: Path
) -> tuple[int, int] | None:
    """The anchor a re-kept prop should carry, read off the asset it replaces.

    `promote_cutouts` used to hand every prop `cutout.anchor` outright, which
    is always the fresh default `layout_cutout` gave the cutout — `set_anchor`
    only ever writes a hand-moved anchor onto `asset.json`, never back onto the
    decomposition's own `Cutout`, so keeping a prop a second time silently put
    the default back and threw away whatever a person had moved it to (ticket
    #37; locked decision 6 says an anchor is editable, never guessed at). The
    fix reads whatever `asset.json` is about to be overwritten and keeps its
    anchor.

    "The asset it replaces" only counts when `_asset_matches_piece` says so —
    a prop is still there, its own saved picture exactly the native size of
    `cutout`'s rebuilt piece. That also means its target is the cutout's, so
    the anchor still sits inside it unchanged. This is called before that
    picture is overwritten, so the size it reads is still the old one. A
    different size, a scene or a tile sharing the name fails the check, and the
    piece gets the fresh default rather than an anchor measured against a shape
    it never had. An `asset.json` — or its `source.png` — that will not read at
    all is left alone rather than raised on: `promote_cutouts` is about to
    overwrite both anyway, so this falls back to the fresh default exactly as if
    nothing were there to inherit.

    A tile, or a piece with no target yet, has no anchor to settle at all —
    `cutout.anchor` is already `None` in both cases — and that is handed back
    unchanged. (Only a rebuilt or pixel-cut prop reaches this: `promote_cutouts` refuses an
    old prop.)
    """
    if (cutout.grid_rebuild is None and cutout.pixel_cut is None) or cutout.target is None:
        return cutout.anchor
    asset_file = folder / "asset.json"
    if not asset_file.exists():
        return cutout.anchor
    try:
        existing = store.read_model(asset_file, AssetRecord)
        matches = _asset_matches_piece(store, direction, cutout, existing)
    except (OSError, ValidationError):
        return cutout.anchor
    if not matches or existing.anchor is None:
        return cutout.anchor
    return existing.anchor


class OldPieceRefused(ValueError):
    """A cut-up run still holds pieces from a road that is gone, so it cannot be kept.

    A `ValueError`, so every face that already turns that family into a
    sentence turns this one into a sentence too. Nothing has been written when
    it is raised.
    """


def promote_cutouts(
    store: Store, direction: Direction, decomposition: Decomposition
) -> list[str]:
    """Copy each cutout into `assets/<name>/` with the provenance export needs.

    Decomposition output is still editable — a person may rename or
    delete pieces first. Promotion is the moment a piece becomes an asset, so it
    is a separate call rather than something `decompose` does on its own. The
    grid fields are copied across untouched, except that the anchor is not
    always one of them. Re-keying a prop whose `asset.json` already carries a
    hand-moved anchor for this same piece (`_reused_anchor`) keeps that anchor,
    unchanged, instead of overwriting it with
    the cutout's own fresh default — see `_reused_anchor` for exactly when an
    anchor counts as the same piece's own. A first keep, a tile, a scene, and
    a piece whose asset has no moved anchor all still get the cutout's anchor
    exactly as before.

    A run that still holds an old piece — a prop that is neither a rebuilt nor a
    pixel-cut one (cut on the props sheet, or redrawn before props were rebuilt on
    their own pixel grid) or a tile with no recipe (a bordered or plain-patch surface's) — is
    refused with `OldPieceRefused` before anything is written, because their
    road is gone and nothing could refine them. Reading and deleting such a
    run still work, so a person can drop the old pieces and keep the rest.
    """
    old = [
        cutout.name
        for cutout in decomposition.cutouts
        if (
            cutout.kind is SheetKind.PROPS
            and cutout.grid_rebuild is None
            and cutout.pixel_cut is None
        )
        or (cutout.kind is SheetKind.TILES and cutout.recipe is None)
    ]
    if old:
        raise OldPieceRefused(old_pieces_sentence(old))
    names: list[str] = []
    for cutout in decomposition.cutouts:
        source = store.cutout_dir(direction.name, decomposition.id, cutout.name) / "source.png"
        if not source.exists():
            continue  # deleted during review
        folder = store.asset_dir(direction.name, cutout.name)
        anchor = _reused_anchor(store, direction, cutout, folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "source.png").write_bytes(source.read_bytes())
        store.write_model(
            folder / "asset.json",
            AssetRecord(
                name=cutout.name,
                kind=SHEET_ASSET_KINDS[cutout.kind],
                source_generation=decomposition.source_generation,
                # No image service drew a recipe tile, so none is claimed.
                source_prompt=(
                    None if cutout.recipe else decomposition.sheet_prompts.get(cutout.sheet, "")
                ),
                # Nor did one draw a pixel-cut prop: it is the picture's own pixels.
                service=(
                    None
                    if cutout.recipe or cutout.pixel_cut
                    else decomposition.sheet_services.get(cutout.sheet, direction.service)
                ),
                variant=cutout.variant,
                recipe=cutout.recipe,
                pitch=cutout.pitch,
                pitch_source=cutout.pitch_source,
                target=cutout.target,
                footprint=cutout.footprint,
                anchor=anchor,
                segmenter=cutout.origin.segmenter if cutout.origin else None,
                grid_rebuild=cutout.grid_rebuild,
                pixel_cut=cutout.pixel_cut,
                decomposition=decomposition.id,
            ),
        )
        names.append(cutout.name)
    return names


def set_anchor(
    store: Store, direction_name: str, asset: str, anchor: tuple[int, int]
) -> AssetRecord:
    """Move a prop's anchor — the point that meets the floor — and write it down.

    The default is bottom centre (`layout_cutout`), and this is the one way
    to change it: a person clicks the refined preview, an agent gives two
    numbers, and the command line takes them as arguments — all three land
    here (M3.6; spec §5, §7). The anchor is in art pixels from the piece's
    top-left and may sit anywhere from (0, 0) to (w, h) inclusive — the far
    corner is a real place to stand.

    Three refusals, all `ValueError`s so every face already knows how to say
    them. An asset that is not a prop has no anchor (locked decision 6: a
    tile is the grid, and the kept scene is not placed on it), and writing
    one would put a number in the manifest that means nothing. An asset with
    no target size was cut before target sizes existed (an M2 sidecar); there
    is nothing to bound the anchor against, so it says which asset and what
    to do. A point outside the target is `check_anchor`'s refusal. Nothing is
    written on any of the three. The lookup is `require_asset`'s, so a typo
    is refused with the known names, exactly as `refine` refuses it.
    """
    name = require_asset(store, direction_name, asset)
    path = store.asset_dir(direction_name, name) / "asset.json"
    record = store.read_model(path, AssetRecord)
    if record.kind is not AssetKind.PROP:
        raise ValueError(f"{name!r} is a {record.kind}, and only a prop has an anchor")
    if record.target is None:
        raise ValueError(
            f"{name!r} has no target size, so there is nothing to anchor to; it was cut "
            "before target sizes existed — pick and redraw the object again to give it one"
        )
    check_anchor(anchor, record.target)
    updated = record.model_copy(update={"anchor": anchor})
    store.write_model(path, updated)
    return updated


def set_scene_pitch(
    store: Store, direction_name: str, asset: str, value: float
) -> AssetRecord:
    """Replace a kept scene's pitch with a number a person or agent chose.

    Locked decision 1 lets a person replace a pitch with one number "from any
    face". A kept scene is filed straight from a generation by
    `file_promotion`, never through `decompose`. Ticket #34 is the gap that
    leaves open: when a scene's own reading fails, its pitch falls back to 1.0
    and its target becomes the whole picture, and the 2026-09-23 amendment to
    spec §3(c) refuses a scene over 1024 px on a side with `TARGET_NOT_USABLE`
    in that case — "until M3.4's set-pitch exists". This is that set-pitch,
    for a scene.

    `check_pitch` runs first, before anything is read off disk, so a value
    outside the spec's bound leaves nothing written. The lookup then goes
    through `require_asset`, exactly as `set_anchor` uses it, so a typo is
    refused with the known names rather than a bare file-not-found. Only a
    scene has a pitch this function may touch: a rebuilt prop's pitch is its
    own, set by the redraw, and a tile has no pitch to set at all (its size is
    always the direction's cell size) — either one is refused, and nothing is
    written.

    The target is recomputed from the whole `source.png`, at the new pitch,
    exactly as `prepare_promotion` computes it the first time; the pixels
    are never touched. A target the refiner could not be asked for
    (`refine.validate_target`: over `MAX_FORCE_PIXELS` on a side) is refused here,
    before `asset.json` is written, rather than at refine time as
    `TARGET_NOT_USABLE`. A scene has no footprint or anchor — locked decision
    6 gives those to a prop only — so this never adds either.
    """
    check_pitch(value)
    name = require_asset(store, direction_name, asset)
    path = store.asset_dir(direction_name, name) / "asset.json"
    record = store.read_model(path, AssetRecord)
    if record.kind is not AssetKind.SCENE:
        raise ValueError(
            f"{name!r} is a {record.kind}, not a scene; only a scene's pitch can "
            f"be set (a prop's is its own, and a tile has none)"
        )
    with Image.open(store.asset_dir(direction_name, name) / "source.png") as picture:
        width, height = picture.size
    target = target_for(SliceBox(left=0, top=0, right=width, bottom=height), value)
    try:
        validate_target(target)
    except ValueError as error:
        raise ValueError(
            f"a pitch of {value} would make {name!r} {target[0]} x {target[1]} art "
            f"pixels, which the refiner cannot be asked for: {error}. Set a larger pitch"
        ) from error
    updated = record.model_copy(
        update={"pitch": value, "pitch_source": PitchSource.SET, "target": target}
    )
    store.write_model(path, updated)
    return updated


def save_decomposition(
    store: Store, direction_name: str, decomposition: Decomposition
) -> Decomposition:
    """Rewrite `decomposition.json` after an edit.

    Every edit below goes through this rather than writing the path itself:
    the sidecar is the only thing that makes a folder of cutouts reachable
    again, so it must never be the step somebody forgot.
    """
    store.write_model(
        store.decomposition_dir(direction_name, decomposition.id) / "decomposition.json",
        decomposition,
    )
    return decomposition


def _cutout_index(decomposition: Decomposition, name: str) -> int:
    """Where `name` sits in the decomposition's cutout list."""
    for index, cutout in enumerate(decomposition.cutouts):
        if cutout.name == name:
            return index
    raise ValueError(
        f"decomposition {decomposition.id} has no piece called {name!r}"
    )


def _taken_names(
    store: Store, direction: Direction, decomposition: Decomposition, freeing: set[str]
) -> set[str]:
    """Every name a new cutout may not take, minus the ones being given up.

    A cutout name becomes an asset folder name on promotion, so it has to be
    unique against the other cutouts *and* against everything already staged
    — otherwise `promote_cutouts` would copy one piece over another.

    `_all_cutout_names` is in here for the reason its own docstring gives:
    `store.list_assets` lists *promoted* assets only, so it stops a collision
    only when promotion happened between two decompose runs. Renaming is
    exactly where a person can walk a piece of run B onto the still-unpromoted
    name of a piece of run A, and promoting both would then silently leave one
    asset where there should be two. `decompose` seeds its own `taken` from
    both sets; `rename_cutout`, whose whole job is renaming, must not seed
    its own from less.

    `- freeing` comes after the union, not before it: `_all_cutout_names`
    reads folder names off disk, and at rename time the old folder is still
    sitting there under the name being given up.
    """
    taken = (
        {cutout.name for cutout in decomposition.cutouts}
        | _all_cutout_names(store, direction.name)
        | set(store.list_assets(direction.name))
    )
    return taken - freeing


def _remove_cutout_folder(folder: Path) -> None:
    """Remove one cutout's folder, or raise if it will not go.

    `shutil.rmtree(..., ignore_errors=True)` swallows a real removal failure
    — a permission error, a file locked by another process — and reports
    success anyway, so a caller that then rewrote the sidecar would leave
    `decomposition.json` claiming a folder gone that is still sitting on
    disk. This propagates that failure instead. `delete_cutout` removes the
    one folder it has before it touches `decomposition.cutouts` at all, so a
    raise here leaves that sidecar exactly as it was. Doing nothing when the
    folder is already absent keeps a second delete of the same name from
    failing on that account.

    `shutil.rmtree` is not atomic — a failure partway through can leave a
    folder emptied of its files but still present as an empty directory —
    so this only ever promises the sidecar and disk agree on *which pieces
    exist*, not that a folder standing after a failure holds anything. A
    retry of the same call finishes whatever removal did not complete.
    """
    if folder.exists():
        shutil.rmtree(folder)


def _refuse_recipe_tile(cutout: Cutout, doing: str) -> None:
    """Refuse an edit that would break a recipe surface's set of tiles.

    The `.aseprite` and the room find a tile by its name, variant and
    place, so a renamed tile would no longer belong to its set.
    """
    if cutout.recipe is not None:
        raise RecipeTileRefused(
            f"{cutout.name} is a recipe tile; {doing} it would break the set of "
            f"{cutout.recipe.surface} tiles. Type the surface again to redraw it, "
            f"or, once it has been exported, edit its master file, "
            f"<direction>/assets/{cutout.recipe.surface}.aseprite, in Aseprite"
        )


def rename_cutout(
    store: Store,
    direction: Direction,
    decomposition: Decomposition,
    name: str,
    new_name: str,
) -> Decomposition:
    """Give one piece a different name, and move its folder to match.

    The name a service invented is a guess (spec §3), and this is where a
    person fixes it before the piece becomes an asset. `new_name` came from a
    person rather than a service, so it is squeezed through `typed_slug` — which
    refuses a name with nothing usable in it or too long to keep whole — rather
    than `kebab_case`, which would silently paper over that with `item` (M2.11's
    coordination note on this path). Refusing happens before anything moves.
    """
    index = _cutout_index(decomposition, name)
    _refuse_recipe_tile(decomposition.cutouts[index], "renaming")
    squeezed = typed_slug(new_name, "piece name")
    if squeezed is None:
        raise ValueError(f"{new_name!r} has nothing in it that can be a piece name")
    chosen = unique_name(squeezed, _taken_names(store, direction, decomposition, {name}))
    old_folder = store.cutout_dir(direction.name, decomposition.id, name)
    new_folder = store.cutout_dir(direction.name, decomposition.id, chosen)
    if chosen != name and old_folder.is_dir():
        old_folder.rename(new_folder)
    cutout = decomposition.cutouts[index].model_copy(update={"name": chosen})
    decomposition.cutouts[index] = cutout
    store.write_model(new_folder / "cutout.json", cutout)
    return save_decomposition(store, direction.name, decomposition)


def delete_cutout(
    store: Store, direction: Direction, decomposition: Decomposition, name: str
) -> Decomposition:
    """Drop one piece, folder and all, before anything is promoted.

    A piece the slicer found in a shadow should not reach the bundle. The
    folder goes as well as the entry: `_all_cutout_names` reads folders off
    disk to keep two decompose runs from choosing the same name, so a folder
    left behind would go on reserving a name nobody wants.
    """
    index = _cutout_index(decomposition, name)
    _remove_cutout_folder(store.cutout_dir(direction.name, decomposition.id, name))
    decomposition.cutouts.pop(index)
    return save_decomposition(store, direction.name, decomposition)


def decompositions_of(store: Store, direction_name: str, generation: str) -> list[Decomposition]:
    """Readable decompositions cut from `generation`, newest first.

    A decomposition whose sidecar will not read is skipped, never raised on.
    """
    found: list[Decomposition] = []
    for decomposition_id in store.list_decompositions(direction_name):
        try:
            record = load_decomposition(store, direction_name, decomposition_id)
        except (OSError, ValueError):
            continue
        if record.source_generation == generation:
            found.append(record)
    return found


def _tiles_of_surface(decomposition: Decomposition, name: str) -> list[Cutout]:
    """The recipe tiles of the surface called `name` that this decomposition holds."""
    return [
        cutout for cutout in decomposition.cutouts
        if cutout.recipe is not None and cutout.recipe.surface == name
    ]


def is_props_run(store: Store, direction_name: str, decomposition: Decomposition) -> bool:
    """Whether the decomposition is an object-by-object (props) run.

    It is one once it has a suggestions folder or a picks folder, or is a pixel
    cut (it carries `pixel_grid`, and makes neither folder until the first
    pick), whether or not it holds any piece yet.
    """
    return (
        decomposition.pixel_grid is not None
        or store.suggestions_dir(direction_name, decomposition.id).exists()
        or store.picks_dir(direction_name, decomposition.id).exists()
    )


def _drop_surface_tiles(
    store: Store, direction: Direction, decomposition: Decomposition, name: str
) -> None:
    """Delete one surface's tiles and outline folder from one decomposition.

    The folders go before the entries, as in `delete_cutout`, and the sidecar
    is written once at the end. A decomposition left with nothing in it (no
    piece, no suggestions, no picks) is a recipe-only run with no reason to
    stay, so its whole folder goes.
    """
    for cutout in _tiles_of_surface(decomposition, name):
        _remove_cutout_folder(store.cutout_dir(direction.name, decomposition.id, cutout.name))
    _remove_cutout_folder(store.surface_dir(direction.name, decomposition.id, name))
    decomposition.cutouts = [
        cutout for cutout in decomposition.cutouts
        if cutout.recipe is None or cutout.recipe.surface != name
    ]
    if (
        not decomposition.cutouts
        and not is_props_run(store, direction.name, decomposition)
    ):
        _remove_cutout_folder(store.decomposition_dir(direction.name, decomposition.id))
    else:
        save_decomposition(store, direction.name, decomposition)


def draw_surface(
    store: Store,
    direction: Direction,
    source: GenerationRecord,
    surface: Surface,
    *,
    replaces: str | None = None,
    now: datetime | None = None,
) -> Decomposition:
    """Draw one recipe surface from an approved picture and keep its tiles as assets.

    For the web page's surfaces screen, where one surface is drawn at a time.
    `decompose` makes the tiles and the outline, `promote_cutouts` files them
    as assets, and only then are the same surface's tiles removed from older
    decompositions of the same picture: drawing it again (an Edit) would
    otherwise leave a duplicate there. The assets of the same names were just
    overwritten by the new ones, so they stay. Every refusal `decompose` or
    `promote_cutouts` raises propagates unchanged, and nothing is removed if
    the draw failed.

    `replaces` is the slug of the drawn surface this one is an edit of, or
    None for a new surface. A name already drawn on this picture that is not
    `replaces` is refused with `AlreadyExists` before anything is drawn, so a
    new surface can never silently overwrite another. A `replaces` that names
    nothing drawn here is a `FileNotFoundError`, also before drawing. When the
    edit renamed the surface, the replaced one is taken off the picture after
    the new one is drawn, by `remove_surface` (which keeps its master file).
    """
    drawn_here = {drawn.name for drawn in drawn_surfaces(store, direction.name, source.request.id)}
    if replaces is not None and replaces not in drawn_here:
        raise FileNotFoundError(f"no surface called {replaces!r} is drawn on this picture")
    if surface.name in drawn_here and surface.name != replaces:
        raise AlreadyExists("That name is taken. Pick another.")
    drawn = decompose(store, direction, source, surfaces=[surface], now=now)
    promote_cutouts(store, direction, drawn)
    for older in decompositions_of(store, direction.name, source.request.id):
        if older.id != drawn.id and _tiles_of_surface(older, surface.name):
            _drop_surface_tiles(store, direction, older, surface.name)
    if replaces is not None and replaces != surface.name:
        remove_surface(store, direction, source.request.id, replaces)
    return drawn


@dataclass(frozen=True)
class DrawnSurface:
    """One recipe surface drawn on a picture, as its tiles record it."""

    #: The surface's slug, which groups its tiles.
    name: str
    #: The words as typed.
    text: str
    kind: str
    shapes: list[list[tuple[int, int]]]
    box: tuple[int, int, int, int]
    #: The colours drawn from, as `#rrggbb`, dark to light.
    shades: list[str]


def drawn_surfaces(store: Store, direction_name: str, generation: str) -> list[DrawnSurface]:
    """The recipe surfaces drawn on this picture, one per name, by name.

    Read from the tiles' `RecipeRecord` in every decomposition of the picture;
    where a name appears in more than one, the newest run wins. A tile from
    before surfaces were drawn by hand (no shapes or box) is not a surface the
    page can show, and is left out.
    """
    found: dict[str, DrawnSurface] = {}
    for decomposition in decompositions_of(store, direction_name, generation):
        for cutout in decomposition.cutouts:
            recipe = cutout.recipe
            if (
                recipe is None or recipe.box is None or recipe.shapes is None
                or recipe.surface in found
            ):
                continue
            # Tiles drawn before the typed words were kept hold only the slug in
            # `ai_name`; show that as words (hyphens to spaces), never rewrite the file.
            text = cutout.ai_name or recipe.surface
            if text == recipe.surface:
                text = text.replace("-", " ")
            found[recipe.surface] = DrawnSurface(
                name=recipe.surface,
                text=text,
                kind=recipe.kind,
                shapes=[[tuple(corner) for corner in shape] for shape in recipe.shapes],
                box=tuple(recipe.box),
                shades=list(recipe.shades),
            )
    return [found[name] for name in sorted(found)]


def remove_surface(
    store: Store, direction: Direction, generation: str, name: str
) -> None:
    """Take a drawn surface off a picture: its tiles, its outline and its assets.

    The tiles and the outline folder go from every decomposition of the
    picture, and so do the assets whose record is a recipe tile of this surface
    from this picture. The surface's master file (`Store.surface_master`) is NOT
    deleted: it holds the person's own touch-ups in Aseprite, and removing a
    surface must never throw those away. Raises `FileNotFoundError` when no
    tile of that name is drawn on this picture.
    """
    holding = [
        decomposition
        for decomposition in decompositions_of(store, direction.name, generation)
        if _tiles_of_surface(decomposition, name)
    ]
    if not holding:
        raise FileNotFoundError(f"no surface called {name!r} is drawn on this picture")
    for decomposition in holding:
        _drop_surface_tiles(store, direction, decomposition, name)
    for asset in store.list_assets(direction.name):
        try:
            record = store.read_model(
                store.asset_dir(direction.name, asset) / "asset.json", AssetRecord
            )
        except (OSError, ValueError):
            continue
        if (
            record.recipe is not None
            and record.recipe.surface == name
            and record.source_generation == generation
        ):
            _remove_cutout_folder(store.asset_dir(direction.name, asset))


def _asset_matches_piece(
    store: Store,
    direction: Direction,
    cutout: Cutout,
    record: AssetRecord,
) -> bool:
    """Whether `assets/<cutout.name>` is a prop holding this rebuilt piece's picture.

    Asked by `_reused_anchor` before it decides whether a re-kept asset's own
    anchor is worth carrying forward. A shared name is not proof by itself. A
    kept scene can be filed under a still-unpromoted cutout's name —
    `file_promotion` checks a scene's name only against other assets, never
    against a decomposition's own cutouts — and a `redraw --only` can land a
    new box under an already-kept piece's name. `source_generation` is no help
    either: every cutout of one decompose run shares it. Carrying an anchor
    across onto a target measured from a picture the asset was never cut from
    would be wrong, so two things have to agree before this counts an asset as
    `cutout`'s own: it must be a prop (a scene or a tile sharing the name is
    never this cutout), and its saved `source.png` — an exact copy of the
    cutout's, made by `promote_cutouts` — must still be exactly the native size
    the rebuilt or pixel-cut piece has. `cutout` must be one (`grid_rebuild` or
    `pixel_cut` set); a rebuilt piece's `bbox` stays where it sat on the group's
    reply, so it is the record's native size that is compared.
    """
    if record.kind is not AssetKind.PROP:
        return False
    expected = tuple((cutout.grid_rebuild or cutout.pixel_cut).native_size)
    with Image.open(store.asset_dir(direction.name, cutout.name) / "source.png") as picture:
        size = picture.size
    return size == expected
