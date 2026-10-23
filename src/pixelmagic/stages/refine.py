"""Hand every asset to PixelRefiner and keep the receipt.

Settings arrive in layers: an art direction sets the house style once, an
individual asset overrides the one or two knobs it needs. Merging them key by
key — rather than replacing whole objects — is what makes "this one tile wants
a bigger cell" a one-line override instead of a copied settings block.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from pixelmagic.model import (
    AssetKind,
    AssetRecord,
    CohesionRecord,
    Direction,
    RefineReport,
    RefinerSettings,
)
from pixelmagic.naming import check_name
from pixelmagic.refiner.base import (
    EXIT_ENGINE_FAILURE,
    EXIT_USAGE,
    LOW_GRID_CONFIDENCE,
    MAX_FORCE_PIXELS,
    SIZE_SETTING_DROPPED,
    RefineFailure,
    Refiner,
    RefineRequest,
    RefinerError,
    RefineResult,
    validate_settings,
)
from pixelmagic.stages.generate import MAX_IMAGE_SIZE
from pixelmagic.stages.rebuild import RebuiltPieceRefused
from pixelmagic.stages.recipes import RecipeTileRefused
from pixelmagic.store import Store

SOURCE_FILE = "source.png"
REFINED_FILE = "refined.png"
REPORT_FILE = "refine.json"
ASSET_FILE = "asset.json"

#: The receipt code written when an asset's own stored target fails
#: `validate_target`. Not a code the real refiner ever returns — the request
#: is refused here, before any subprocess runs — so it lives with the stage's
#: other synthetic codes rather than the CLI contract in `refiner/base.py`.
#: `refine_direction` writes this receipt for the one asset and keeps
#: refining everything else (CLAUDE.md: "one broken asset never costs the
#: bundle"); `refine_asset`, asked to refine that one asset by name, still
#: raises instead — the caller named exactly one asset, so the error belongs
#: to that call, not a receipt it has to go read.
TARGET_NOT_USABLE = "TARGET_NOT_USABLE"

#: The receipt code written when an asset's own `asset.json` will not parse.
#: Not a code the real refiner ever returns, for the same reason
#: `TARGET_NOT_USABLE` is not: the request is refused here, before any
#: subprocess runs. `refine_direction` writes this receipt for the one asset
#: and keeps refining everything else (CLAUDE.md: "one broken asset never
#: costs the bundle"); `refine_asset`, asked to refine that one asset by
#: name, still raises the `ValueError` `_target_of` builds, because the
#: caller named exactly one asset and that error already names it.
ASSET_RECORD_UNREADABLE = "ASSET_RECORD_UNREADABLE"

#: The receipt code written when a rebuilt prop's own `source.png` will not
#: open as a picture. A rebuilt prop never reaches the refiner, so this is
#: PixelMagic's own finding, never the refiner's. `refine_direction` writes
#: it for the one asset and keeps going; `refine_asset` raises instead.
PICTURE_UNREADABLE = "PICTURE_UNREADABLE"

#: The receipt code written for an old tile (a bordered block's, or a plain
#: patch's) or an old prop (cut on the props sheet, or kept before props were
#: rebuilt on their own pixel grid) that was never refined. Its road is gone,
#: so there is nothing to refine it with; the sentence says what to do instead
#: (`old_item_sentence`). PixelMagic's own refusal, like `TARGET_NOT_USABLE`.
OLD_ITEM_REFUSED = "OLD_ITEM_REFUSED"

#: The receipt code written when the refiner reported success but the picture
#: it wrote is not the size the asset's target asked for. Unlike the codes
#: above it is not in `PIXELMAGIC_REFUSAL_CODES`: the refiner did see the
#: picture. The receipt replaces the result and the `refined.png` is deleted,
#: so a wrong-sized scene can never be exported as if it had been refined.
WRONG_SIZE = "WRONG_SIZE"

#: Codes that mean PixelMagic itself refused an asset, before the refiner
#: ever saw it — as opposed to a code the refiner returned after looking at
#: the picture. A face (`cli.py`, `export.py`, `web/templates/refine.html`)
#: must read this set rather than repeat the list of codes, so the two
#: refusal messages ("PixelMagic did not send it" vs "the refiner refused
#: it") never drift out of step with which codes actually mean which thing.
PIXELMAGIC_REFUSAL_CODES = frozenset(
    {TARGET_NOT_USABLE, ASSET_RECORD_UNREADABLE, PICTURE_UNREADABLE, OLD_ITEM_REFUSED}
)

#: What `refine.json` names as the tool for a rebuilt prop (ticket #134). The
#: picture was rebuilt one pixel per art pixel before it was ever kept, so
#: PixelRefiner never sees it; export copies this into `refiner_tool`.
REBUILD_REFINER = "pixelmagic-grid-rebuild"

#: The route `refine.json` names for a rebuilt prop (the others are refine,
#: convert, preserve, and the stand-in's "stub").
REBUILD_ROUTE = "rebuild"

#: What `refine.json` names as the tool for a prop cut straight from a pixel-art
#: picture's own pixels. Like a rebuilt prop it is never sent to PixelRefiner; export
#: copies this into `refiner_tool`.
PIXEL_CUT_REFINER = "pixelmagic-pixel-cut"

#: The route `refine.json` names for a prop cut straight from a pixel-art picture.
PIXEL_CUT_ROUTE = "pixel-cut"

#: What `refine.json` names as the tool for a recipe tile (M7). The tile was
#: drawn crisp at its final size, so PixelRefiner never sees it; export copies
#: this into `refiner_tool`.
RECIPE_REFINER = "pixelmagic-recipe"

#: The route `refine.json` names for a recipe tile.
RECIPE_ROUTE = "recipe"

#: The route `refine_direction` puts on the receipt it hands back for an old
#: tile or prop that was refined before its road went. Only the returned copy
#: carries it, so a face can tell "left as it is" from "refined now";
#: `refine.json` on disk is not rewritten.
OLD_ITEM_ROUTE = "old-item"

#: The house style the kept scene refines with unless the direction or an
#: override says otherwise (M4 spec, "Defaults and the record"). Layered *underneath*
#: `direction.refiner_defaults` wherever settings are resolved for an actual
#: refine call; `set_refiner_defaults` never folds it into what it writes, so
#: an existing direction whose stored defaults predate this ticket (the den's
#: does) still refines with these four values without `direction.json` being
#: rewritten. A *new* direction is the one exception — `directions.py` seeds
#: a fresh `refiner_defaults` with a deep copy of this exact block, so it has
#: its own stored copy from the start; either way, any value the direction or
#: an override *does* set still wins key by key (`resolve_settings`'s
#: existing rule). PixelMagic sends these explicitly
#: rather than relying on PixelRefiner's own CLI/MCP default-on backfill for
#: an unmentioned key (M4.1, PM decision 3 on ticket #69): that backfill is
#: a stopgap for a caller that never asks, not a value PixelMagic wants to
#: depend on staying what it is today.
HOUSE_BASELINE = RefinerSettings(
    quick={"reductionMode": "24"},
    advanced={
        "cohesion": True,
        "cohesionFamilies": 14,
        "cohesionTones": 2,
        "cleanup": True,
    },
)

def validate_target(
    target: tuple[int, int], *, honours_forced_size: bool = True
) -> None:
    """Refuse a forced output size the pipeline could never use.

    Each side must be a whole number of art pixels from 1 to the tighter of
    two ceilings. `MAX_IMAGE_SIZE` is the bound `generate` and `decompose`
    put on a picture, because a piece refined to a size Pillow will not open
    is a piece nothing downstream can export. `MAX_FORCE_PIXELS` is the real
    refiner's own limit on `forcePixelsW`/`forcePixelsH` — measured at 1024,
    see the constant's docstring — and matters here because the CLI does not
    refuse a bigger request: it silently clamps to 1024 and exits 0, so a
    target above that would be reported as refined at a size it never
    reached. That second ceiling applies only to a refiner that honours a
    forced size (`honours_forced_size`, the default, since a person setting a
    pitch is setting it for the real one); the stand-in ignores the target
    (locked decision 9), so it refines a big picture at its own size and only
    `MAX_IMAGE_SIZE` bounds it. Checked here, in the core, so every face
    inherits the bound (spec §3), and checked before any subprocess runs so
    a bad target costs nothing but the sentence.
    """
    limit = (
        min(MAX_IMAGE_SIZE, MAX_FORCE_PIXELS) if honours_forced_size else MAX_IMAGE_SIZE
    )
    for value in target:
        if isinstance(value, bool) or not isinstance(value, int) or not (
            1 <= value <= limit
        ):
            raise ValueError(
                f"target size {target} is not usable — each side must be a whole "
                f"number of art pixels from 1 to {limit} (the refiner clamps a "
                f"forced size above {MAX_FORCE_PIXELS} instead of refusing it), "
                f"not {value!r}"
            )


#: The advanced keys that move a forced output size (measured, ticket #34).
_SIZE_ADVANCED_KEYS = ("makeSquare", "keepAspectRatio", "outlineStyle")


def wrong_size_sentence(report: RefineReport) -> str:
    """The one truthful sentence for a `WRONG_SIZE` receipt, for every face.

    The refiner reported success; PixelMagic found the picture was not the
    forced size and did not keep it. The receipt's own message holds the sizes.
    """
    return (
        "the refiner returned the wrong size, so PixelMagic did not keep it: "
        f"{report.error_message}"
    )


def _size_changing_settings(settings: RefinerSettings) -> list[str]:
    """Names of the settings in `settings` that would move a forced output size.

    Each was measured on the real CLI with `forcePixelsW/H` set (ticket #34):
    `scale` multiplies the size, `makeSquare` and `keepAspectRatio` pad it,
    `outlineStyle` (other than "none") grows it by the outline. A key that is
    present but switched off (false, "none", a scale of 1) does nothing and is
    not named. No other quick or advanced key, and no preset, moved the size.
    """
    names: list[str] = []
    if settings.scale and settings.scale != 1:
        names.append("scale")
    for key in ("makeSquare", "keepAspectRatio"):
        if settings.advanced.get(key) not in (None, False, "false", "null"):
            names.append(key)
    if settings.advanced.get("outlineStyle") not in (None, "none", "null"):
        names.append("outlineStyle")
    return names


def _settings_for_target(
    settings: RefinerSettings, target: tuple[int, int] | None
) -> tuple[RefinerSettings, bool]:
    """The settings to send with a forced size, and whether any were left out.

    With a target, the size-changing settings (`_size_changing_settings`) are
    removed rather than refused: a person saved them, and refusing a saved
    setting is not ours to decide, but sending them would let them quietly beat
    the forced size. The returned settings are exactly what is sent, so the
    receipt's `requested` tells the truth. Without a target nothing changes.
    """
    if target is None:
        return settings, False
    if not _size_changing_settings(settings):
        return settings, False
    kept = settings.model_copy(
        update={
            "scale": None,
            "advanced": {
                key: value
                for key, value in settings.advanced.items()
                if key not in _SIZE_ADVANCED_KEYS
            },
        }
    )
    return kept, True


def _target_of(store: Store, direction_name: str, name: str) -> tuple[int, int] | None:
    """The forced output size an asset's own sidecar asks for, if any.

    Assets from before M3 either have no sidecar or no target on it, and stay
    unforced so the refiner uses its normal grid detection. A present target
    is deliberately not defaulted: callers validate it before refining.

    A sidecar that will not parse is named here rather than left to raise its
    own `ValidationError`: a `ValueError` naming the asset and the file says
    the same thing — a broken file, not the refiner saying no — in a
    sentence a person can act on. `refine_asset`, asked to refine one named
    asset, lets that `ValueError` travel out of it (`cli.py` deliberately
    lets it travel as itself rather than folding it into the one-line
    refiner refusal). `refine_direction` calls this for every asset in a
    direction before any refining starts, and instead of letting the first
    broken sidecar stop every other asset, it catches this `ValueError` and
    gives that one asset an `ASSET_RECORD_UNREADABLE` receipt (CLAUDE.md:
    "one broken asset never costs the bundle").
    """
    path = store.asset_dir(direction_name, name) / ASSET_FILE
    if not path.exists():
        return None
    try:
        return store.read_model(path, AssetRecord).target
    except (ValidationError, UnicodeDecodeError) as error:
        raise ValueError(
            f"asset {name!r} in direction {direction_name!r} has a broken "
            f"{path}: {error}"
        ) from error


def _asset_kind(store: Store, direction_name: str, name: str) -> AssetKind | None:
    """The asset's own kind, or `None` for a bare folder with no `asset.json`.

    Reads the same sidecar `_target_of` reads, for the same reason they are
    two small readers rather than one big one: this is only ever called for a
    name that survived `refine_direction`'s own `_target_of` pass (which
    already turns a broken sidecar into an `ASSET_RECORD_UNREADABLE` receipt),
    so `asset.json` is already known to parse by the time this reads it.
    """
    path = store.asset_dir(direction_name, name) / ASSET_FILE
    if not path.exists():
        return None
    return store.read_model(path, AssetRecord).kind


def _asset_is_rebuilt(store: Store, direction_name: str, name: str) -> bool:
    """Whether the asset carries a grid-rebuild record (a picked prop rebuilt on its grid).

    Reads the same sidecar `_asset_kind` reads, and like it is only called for
    a name whose `asset.json` is already known to parse. A bare folder with no
    `asset.json` is not rebuilt.
    """
    path = store.asset_dir(direction_name, name) / ASSET_FILE
    if not path.exists():
        return False
    return store.read_model(path, AssetRecord).grid_rebuild is not None


def _asset_is_pixel_cut(store: Store, direction_name: str, name: str) -> bool:
    """Whether the asset carries a pixel-cut record (a prop cut from a pixel-art picture).

    Like `_asset_is_rebuilt`, only called for a name whose `asset.json` is already known
    to parse; a bare folder with no `asset.json` is not one.
    """
    path = store.asset_dir(direction_name, name) / ASSET_FILE
    if not path.exists():
        return False
    return store.read_model(path, AssetRecord).pixel_cut is not None


def _recipe_surface(store: Store, direction_name: str, name: str) -> str | None:
    """The surface a recipe tile belongs to, or None when the asset is not a recipe tile.

    Like `_asset_is_rebuilt`, only called for a name whose `asset.json` is
    already known to parse; a bare folder with no `asset.json` is not one.
    """
    path = store.asset_dir(direction_name, name) / ASSET_FILE
    if not path.exists():
        return None
    recipe = store.read_model(path, AssetRecord).recipe
    return None if recipe is None else recipe.surface


def _asset_is_recipe(store: Store, direction_name: str, name: str) -> bool:
    """Whether the asset is a recipe tile (its `asset.json` carries a recipe record)."""
    return _recipe_surface(store, direction_name, name) is not None


#: What to do instead, for each kind of old item. Used by the refine refusals and by
#: `old_pieces_sentence`, which `decompose.promote_cutouts` raises.
_TILE_WAY_FORWARD = (
    "type the surface again as a recipe surface (name:kind; box x1,y1 x2,y2; shape ...) "
    "to get new tiles"
)
_PROP_WAY_FORWARD = "pick and redraw the object again to get a new piece"


def old_item_kind(record: AssetRecord) -> AssetKind | None:
    """The kind (tile or prop) of an asset that comes from a road that is gone, else None.

    An old tile is a tile with no recipe (a bordered block's, or a plain
    patch's). An old prop is a prop with neither a grid rebuild nor a pixel cut
    (cut on the props sheet, or kept before props were rebuilt on their own pixel
    grid). A scene, a recipe tile, a rebuilt prop and a pixel-cut prop are not old.
    """
    if record.kind is AssetKind.TILE and record.recipe is None:
        return AssetKind.TILE
    if record.kind is AssetKind.PROP and record.grid_rebuild is None and record.pixel_cut is None:
        return AssetKind.PROP
    return None


def _old_item_kind(store: Store, direction_name: str, name: str) -> AssetKind | None:
    """`old_item_kind` of a stored asset; None for a bare folder with no `asset.json`.

    Like `_asset_is_recipe`, only called for a name whose `asset.json` is
    already known to parse.
    """
    path = store.asset_dir(direction_name, name) / ASSET_FILE
    if not path.exists():
        return None
    return old_item_kind(store.read_model(path, AssetRecord))


def old_item_sentence(name: str, kind: AssetKind) -> str:
    """Why an old tile or prop cannot be refined, and the way forward."""
    if kind is AssetKind.TILE:
        return (
            f"{name} is a tile from a bordered or plain-patch surface, and that road is "
            f"gone, so it cannot be refined; {_TILE_WAY_FORWARD}"
        )
    return (
        f"{name} is a prop from the old props sheet or from before props were rebuilt on "
        f"their own pixel grid, and that road is gone, so it cannot be refined; "
        f"{_PROP_WAY_FORWARD}"
    )


def old_pieces_sentence(names: Sequence[str]) -> str:
    """Why a cut-up run holding old pieces cannot be kept, and the way forward."""
    return (
        f"this run still holds old piece(s) {', '.join(names)} from a road that is gone "
        f"(the old props sheet, a bordered or plain-patch surface, or props cut before "
        f"the pixel-grid rebuild), so it cannot be kept; for a prop, {_PROP_WAY_FORWARD}; "
        f"for a tile, {_TILE_WAY_FORWARD}; or drop the old pieces and keep the rest"
    )


def _refined_old_item(store: Store, direction_name: str, name: str) -> RefineReport | None:
    """An old item's receipt when it was already refined, else None.

    Such a tile or prop still exports as it is, so refining a direction leaves
    its picture and receipt alone. Refined means a `refined.png` beside a
    `refine.json` that records no refusal.
    """
    folder = store.asset_dir(direction_name, name)
    if not (folder / REFINED_FILE).exists() or not (folder / REPORT_FILE).exists():
        return None
    try:
        report = store.read_model(folder / REPORT_FILE, RefineReport)
    except ValidationError:
        return None
    return None if report.error_code else report


def _recipe_tile_refused(name: str, surface: str) -> RecipeTileRefused:
    """The refusal for a setting or re-pick aimed at a recipe tile of `surface`."""
    return RecipeTileRefused(
        f"{name} is a recipe tile; it is not refined, so a setting or a re-pick "
        "cannot apply. Type the surface again to redraw it, or, once it has been "
        f"exported, edit its master file, <direction>/assets/{surface}.aseprite, in Aseprite"
    )


def _keep_rebuilt(store: Store, direction_name: str, name: str) -> RefineReport:
    """Write a rebuilt prop's `refined.png` and `refine.json` without any refiner call.

    See `_keep_as_it_is`; a rebuilt prop is named `REBUILD_REFINER` on route `REBUILD_ROUTE`.
    """
    return _keep_as_it_is(store, direction_name, name, REBUILD_REFINER, REBUILD_ROUTE)


def _keep_pixel_cut(store: Store, direction_name: str, name: str) -> RefineReport:
    """Write a pixel-cut prop's `refined.png` and `refine.json` without any refiner call.

    See `_keep_as_it_is`; it is named `PIXEL_CUT_REFINER` on route `PIXEL_CUT_ROUTE`.
    """
    return _keep_as_it_is(store, direction_name, name, PIXEL_CUT_REFINER, PIXEL_CUT_ROUTE)


def _keep_recipe(store: Store, direction_name: str, name: str) -> RefineReport:
    """Write a recipe tile's `refined.png` and `refine.json` without any refiner call.

    See `_keep_as_it_is`; a recipe tile is named `RECIPE_REFINER` on route `RECIPE_ROUTE`.
    """
    return _keep_as_it_is(store, direction_name, name, RECIPE_REFINER, RECIPE_ROUTE)


def _keep_as_it_is(
    store: Store, direction_name: str, name: str, refiner_name: str, route: str
) -> RefineReport:
    """Write a piece's `refined.png` and `refine.json` without any refiner call.

    The picture is already one pixel per art pixel, so PixelRefiner resampling
    it would undo how it was made. Export still needs both files, so `source.png`
    is copied byte for byte and the report is read off that picture. The
    palette lists the opaque colours as `#rrggbb`, commonest first, like the
    other reports; `confidence` is 1.0, the same as a forced-size piece.
    A `source.png` that cannot be read raises `ValueError` before any file is
    written.
    """
    folder = store.asset_dir(direction_name, name)
    # Read the picture before anything is written, so a corrupt one leaves
    # the folder as it was and raises a `ValueError` naming the asset.
    try:
        with Image.open(folder / SOURCE_FILE) as picture:
            rgba = picture.convert("RGBA")
            width, height = picture.size
    except OSError as error:
        raise ValueError(f"asset {name!r}: {SOURCE_FILE} cannot be read: {error}") from error
    shutil.copyfile(folder / SOURCE_FILE, folder / REFINED_FILE)
    counts = Counter(pixel[:3] for pixel in rgba.get_flattened_data() if pixel[3] > 0)
    report = RefineReport(
        asset=name,
        route=route,
        confidence=1.0,
        refiner=refiner_name,
        palette=[f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in counts.most_common()],
        width=width,
        height=height,
        color_count=len(counts),
    )
    store.write_model(folder / REPORT_FILE, report)
    return report


def resolve_settings(*layers: RefinerSettings | None) -> RefinerSettings:
    """Fold settings layers left to right, later layers winning per key.

    `preset` and `scale` are whole values, so a later layer that sets one
    replaces it. `quick` and `advanced` are dictionaries, so a later layer
    overrides only the keys it actually mentions — which is the same rule
    PixelRefiner itself uses between preset, quick and advanced.
    """
    merged = RefinerSettings()
    for layer in layers:
        if layer is None:
            continue
        if layer.preset is not None:
            merged.preset = layer.preset
        if layer.scale is not None:
            merged.scale = layer.scale
        merged.quick.update(layer.quick)
        merged.advanced.update(layer.advanced)
    return merged


def set_refiner_defaults(
    store: Store, direction: Direction, given: RefinerSettings
) -> Direction:
    """Layer `given` onto a direction's stored defaults and save the result.

    Layering rather than rebuilding is what makes "just change the scale" a
    one-flag edit: a rebuilt block would drop whatever the direction already
    held in the fields the caller did not mention.

    The merged block is checked before it is written, and refused settings
    raise `RefinerError` with nothing saved. Settings the refiner will not
    accept must never become a direction's default, because they would then
    refuse every later run — including runs that pass no settings at all —
    and the only cure would be editing `direction.json` by hand. The check
    belongs here rather than in a face (locked decision 7): the web page and
    an MCP agent will set defaults too, and each one writing its own guard is
    how they would end up disagreeing about what a direction may hold.
    """
    merged = resolve_settings(direction.refiner_defaults, given)
    try:
        validate_settings(merged)
    except ValueError as error:
        raise RefinerError(
            str(error), exit_code=EXIT_USAGE, code="INVALID_SETTINGS"
        ) from error
    direction.refiner_defaults = merged
    return store.save_direction(direction)


def _is_count(value: object) -> bool:
    """A real family/tone count -- an `int`, but never a `bool`.

    `isinstance(True, int)` is `True` in Python (`bool` subclasses `int`), so
    a bare `isinstance(value, int)` would let a stray boolean through as if
    it were a family or tone count. Nothing in the real CLI's contract sends
    a bool here, but the guard costs one line and this codebase's other
    numeric checks (`validate_target`) already exclude `bool` the same way.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _own_palette_count(settings: RefinerSettings) -> int | None:
    """The colour count a two-step own-palette recipe should fit, or `None`.

    `reductionMode`'s vocabulary mixes real colour counts ("24") with named
    palettes, `auto` and `none` (`QUICK_KEYS["reductionMode"]`,
    `refiner/base.py`); only a count is something the two-step recipe below
    understands as "fit N colours on the source" -- a named palette or
    `auto` has nothing for a fresh fit to reproduce. `reductionMode` is a
    quick value, which the contract says is always a string, so this only
    ever sees a string or nothing at all; `str.isdigit` alone is what tells
    "24" apart from "gb_legacy" or "auto".
    """
    value = settings.quick.get("reductionMode")
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _own_palette_target_count(kind: AssetKind | None, settings: RefinerSettings) -> int | None:
    """Whether an asset of this kind should use the two-step own-palette recipe, and at what count.

    Shared by `refine_asset` and `refine_direction`'s `_refine_alone` helper,
    so an asset refined either way asks the identical question and never
    drifts into running a different recipe for the same asset kind and
    resolved settings (ticket #70 review round 1). Only the kept scene has
    "its own picture" for a fresh fit to describe. A bare folder (no
    `asset.json`, `kind` is `None`) keeps whatever single-call path its
    caller already uses. `_own_palette_count` is the other half of the
    question -- whether the resolved settings even name a colour count.
    """
    if kind is not AssetKind.SCENE:
        return None
    return _own_palette_count(settings)


def _refine_with_own_palette(
    store: Store,
    direction: Direction,
    refiner: Refiner,
    name: str,
    settings: RefinerSettings,
    target: tuple[int, int] | None,
    count: int,
    *,
    raise_on_failure: bool = False,
) -> RefineReport:
    """The M4.1 spike's own two-step recipe, fitting the kept scene's own palette.

    Fit a palette on the scene's untouched source, cohesion and clean-up off,
    then refine it with cohesion on and that palette pinned. Measured on the
    local checkout on 2026-09-25 (M4.1 acceptance run, spec amendment
    "Palettes" 2026-09-25, confirmed by the CEO on 2026-09-25, on locked
    decision 8): a single `reductionMode` call fit *after* cohesion has
    already run does not reproduce the spike's own bed, but this two-step
    order does. Fitting the palette first means the colours describe the art
    itself, not whatever cohesion's own colour grouping already flattened it
    to.

    No `background` flag is sent on either call: a scene's `source.png` is the
    generator's raw, un-cropped frame, still carrying whatever surround the
    image service drew around it, and the refiner's own background removal is
    what clears it. (`background: "keep"` turned both of its background passes
    off, which left `the-den`'s corner opaque instead of transparent.)

    Called only when the resolved settings ask for a colour count
    (`_own_palette_count`/`_own_palette_target_count`); a named palette,
    `auto` or `none` has no fit to reuse "own" from, so those keep a plain
    batch call instead. `settings` here is the fully resolved settings --
    `HOUSE_BASELINE` layered under the direction's own defaults and any
    override -- before any size-changing setting is dropped for the target
    (`_settings_for_target` runs only in the final call), and it is what
    `requested` on a palette-pass refusal's receipt names (rather than
    whatever the final pinned call would have asked for), since the refiner
    never got far enough on that asset to report anything of its own; a
    success's `requested`, by contrast, is `_report`'s ordinary rule and
    reflects the pinned call that actually ran. The palette pin itself is
    the two keys `reduceColorMode: "fixed"` and `fixedPalette: [...]`.

    `raise_on_failure` is `refine_asset`'s own need, not `refine_direction`'s,
    and applies to *both* calls this recipe makes: asked to refine one named
    asset, `refine_asset` raises for every other kind of refiner refusal
    (`refiner.refine`'s own contract), so a refusal reached through either
    step of this recipe -- the palette pass, or the final pinned call it
    feeds into `_run_batch_and_record` -- must reach its caller the same
    way: a `RefinerError` naming the asset, since neither failure shape
    (`fit_palette`'s `RefineFailure`, nor a batch item's) names anything but
    the file. `refine_direction`, refining a whole direction (CLAUDE.md: "one
    broken asset never costs the bundle"), takes the default, `False`: a
    receipt for the asset on either step, so the rest of the direction keeps
    going.
    """
    folder = store.asset_dir(direction.name, name)
    plain_quick = {key: value for key, value in settings.quick.items() if key != "reductionMode"}
    palette_settings = settings.model_copy(
        update={
            "quick": plain_quick,
            "advanced": {**settings.advanced, "cohesion": False, "cleanup": False},
        }
    )
    palette = refiner.fit_palette(folder / SOURCE_FILE, count, palette_settings)
    if isinstance(palette, RefineFailure):
        if raise_on_failure:
            raise RefinerError(
                f"asset {name!r}: {palette.message}",
                exit_code=EXIT_ENGINE_FAILURE,
                code=palette.code,
            )
        # Same shape a refiner's own batch refusal gets below: the asset gets
        # a receipt naming the problem, and loses any stale refined.png an
        # earlier run left -- the refiner never even reached the final
        # refine this time.
        (folder / REFINED_FILE).unlink(missing_ok=True)
        report = RefineReport(
            asset=name,
            refiner=refiner.name,
            error_code=palette.code,
            error_message=palette.message,
            requested=settings,
        )
        store.write_model(folder / REPORT_FILE, report)
        return report
    final_settings = settings.model_copy(
        update={
            "quick": dict(plain_quick),
            "advanced": {
                **settings.advanced,
                "reduceColorMode": "fixed",
                "fixedPalette": palette,
            },
        }
    )
    return _run_batch_and_record(
        store, direction, refiner, name, final_settings, target,
        raise_on_failure=raise_on_failure,
    )


def _cohesion_record(effective_options: Mapping[str, object]) -> CohesionRecord:
    """Read what the refiner says it actually did with cohesion and clean-up.

    A key missing from `effective_options` entirely means it did not run --
    the honest reading whether the refiner is the stand-in or a real
    PixelRefiner build from before M4.1 that has never heard of these keys.
    Built from `effective_options`, never from what was requested: a
    per-piece override can turn cohesion off even when the direction's
    default is on, so "requested" and "ran" can disagree.

    Read with `is True` rather than `bool(...)`: `effective_options` values
    travel through JSON, and `bool("false")` is `True` in Python because a
    non-empty string is truthy — a refiner reply that ever spelled the flag
    as the *string* `"false"` would otherwise be read as cohesion having
    run. Only the real Python `True` counts as on.
    """
    families = effective_options.get("cohesionFamilies")
    tones = effective_options.get("cohesionTones")
    return CohesionRecord(
        cohesion=effective_options.get("cohesion") is True,
        cohesion_families=families if _is_count(families) else None,
        cohesion_tones=tones if _is_count(tones) else None,
        cleanup=effective_options.get("cleanup") is True,
    )


def _report(
    name: str,
    requested: RefinerSettings,
    result: RefineResult,
    tool: str,
    *,
    candidate_id: str | None = None,
) -> RefineReport:
    """Turn a refiner result into the `refine.json` model.

    `result.output.path` is the refined file: the batch contract on `Refiner`
    guarantees it is the request's own `output_path`.
    """
    return RefineReport(
        asset=name,
        route=result.route,
        confidence=result.confidence,
        refiner=tool,
        classification=result.classification,
        warnings=list(result.warnings),
        grid_candidates=list(result.grid_candidates),
        candidates=list(result.candidates),
        candidate_id=candidate_id,
        requested=requested,
        effective_options=dict(result.effective_options),
        palette=list(result.output.palette),
        width=result.output.width,
        height=result.output.height,
        color_count=result.output.color_count,
        cohesion=_cohesion_record(result.effective_options),
    )


def _record_success(
    store: Store,
    folder: Path,
    refiner: Refiner,
    report: RefineReport,
    target: tuple[int, int] | None,
    *,
    dropped_size_setting: bool,
) -> RefineReport:
    """Write a successful refine's receipt, unless the picture is the wrong size.

    A refiner that honours a forced size (`Refiner.honours_forced_size`) must
    return exactly the target. Anything else is not a success: the
    `refined.png` the call wrote is deleted and a `WRONG_SIZE` failure receipt
    is written instead, naming the asset, the size that came back and the
    target. `SIZE_SETTING_DROPPED` is added to a receipt that stands.
    """
    if target is not None and refiner.honours_forced_size and (
        (report.width, report.height) != target
    ):
        (folder / REFINED_FILE).unlink(missing_ok=True)
        report = RefineReport(
            asset=report.asset,
            refiner=refiner.name,
            error_code=WRONG_SIZE,
            error_message=(
                f"asset {report.asset!r}: the refiner returned a picture "
                f"{report.width} x {report.height} but the target is "
                f"{target[0]} x {target[1]}, so it was not kept"
            ),
            requested=report.requested,
            warnings=[SIZE_SETTING_DROPPED] if dropped_size_setting else [],
        )
    elif dropped_size_setting:
        report.warnings.append(SIZE_SETTING_DROPPED)
    store.write_model(folder / REPORT_FILE, report)
    return report


def refine_asset(
    store: Store,
    direction: Direction,
    name: str,
    *,
    refiner: Refiner,
    override: RefinerSettings | None = None,
    candidate_id: str | None = None,
) -> RefineReport:
    """Refine one asset and write its `refine.json`.

    `candidate_id` asks for one of the alternatives a previous run reported
    when it was unsure of the pixel grid (spec §3). It is per-call rather than
    a settings layer: it names one result of one image, so it is not something
    a direction can hold as a default, and `refine_direction` has no way to
    pass it because the CLI's batch verb has no per-input equivalent.

    A missing `source.png` is refused rather than raised: the caller asked for
    one named asset, and a missing input is exactly the kind of thing a
    `refine.json` should say, not crash on.

    The asset's target travels on the request so the refiner is told how big
    the piece must come out; it is checked here in the core before the source
    check can write a receipt, so every face refuses the same bad size. Unlike
    `refine_direction`, an unusable target here is raised rather than turned
    into a receipt: the caller named exactly one asset, so `ValueError` is the
    plain answer to a plain request — but it must still name the asset, since
    `validate_target`'s own message only ever names the target.

    The kept scene whose resolved settings name a colour count runs the
    two-step own-palette recipe (`_refine_with_own_palette`) instead of the
    single call below — the same recipe `refine_direction` gives it by default,
    so refining it by name and refining a whole direction never disagree about
    what it gets. An override that replaces `reductionMode` with a named
    palette, `auto` or `none` turns this off for that call, the same as it
    would for `refine_direction` (`_own_palette_target_count` reads the fully
    resolved settings, override included), and the plain single call below
    still runs. A bare folder (no `asset.json`) is never eligible either way.
    (A `candidate_id` on the scene never gets this far: it is refused above.)

    The asset's target is validated before its settings are resolved, so a
    sidecar that will not parse still raises `_target_of`'s own
    asset-named `ValueError` here, before anything else reads the same
    broken file; there is no pre-filtering here the way `refine_direction`
    has, so this call orders the steps itself.

    An old tile (a bordered block's, or a plain patch's) or an old prop (cut on
    the props sheet, or kept before props were rebuilt on their own pixel grid)
    cannot be refined, refined before or not: its road is gone, so this raises
    a `ValueError` naming the asset and saying what to do instead
    (`old_item_sentence`).

    With a target, saved settings that would change the forced size (`scale`,
    `makeSquare`, `keepAspectRatio`, `outlineStyle`) are not sent and the
    receipt carries `SIZE_SETTING_DROPPED` (`_settings_for_target`). A refiner
    that honours a forced size and returns another is not a success: the
    receipt is a `WRONG_SIZE` failure, returned like `INPUT_NOT_FOUND`, and no
    `refined.png` is left (`_record_success`). A `candidate_id` re-pick on the
    kept scene is refused up front with a `ValueError`, before any refiner call
    or write, because the refiner lets the candidate win over the forced size
    and the result would always come back the wrong size.

    A recipe tile, like a rebuilt prop, is kept as drawn and never sent to the
    refiner: by name it is copied through, and a setting or a candidate re-pick
    aimed at one is refused.
    """
    folder = store.asset_dir(direction.name, name)
    target = _target_of(store, direction.name, name)
    if target is not None:
        try:
            validate_target(target, honours_forced_size=refiner.honours_forced_size)
        except ValueError as error:
            raise ValueError(f"asset {name!r}: {error}") from error
    old_kind = _old_item_kind(store, direction.name, name)
    if old_kind is not None:
        raise ValueError(old_item_sentence(name, old_kind))
    # A rebuilt prop is never sent to the refiner, so a setting or a candidate
    # re-pick aimed at it by name could only be ignored. Refuse it before
    # anything is written. An empty override is not a setting: every face
    # builds a `RefinerSettings` even when the person gave none.
    surface = _recipe_surface(store, direction.name, name)
    if surface is not None and (
        candidate_id is not None
        or (override is not None and override != RefinerSettings())
    ):
        raise _recipe_tile_refused(name, surface)
    if _asset_is_rebuilt(store, direction.name, name) and (
        candidate_id is not None
        or (override is not None and override != RefinerSettings())
    ):
        raise RebuiltPieceRefused(name)
    if _asset_is_pixel_cut(store, direction.name, name) and (
        candidate_id is not None
        or (override is not None and override != RefinerSettings())
    ):
        raise ValueError(
            f"{name} was cut straight from the picture's own pixels and is never sent "
            "to the refiner, so a refine setting or a candidate re-pick cannot apply to it"
        )
    if candidate_id is not None and _asset_kind(store, direction.name, name) is AssetKind.SCENE:
        raise ValueError(
            f"A grid candidate cannot be used on the scene {name!r}: the refiner "
            "would ignore the scene's size and the result would come back the wrong "
            "size. Set the scene's pitch instead."
        )
    settings = resolve_settings(HOUSE_BASELINE, direction.refiner_defaults, override)
    source = folder / SOURCE_FILE
    if not source.exists():
        # Last run's refined.png would otherwise sit next to a report saying
        # this asset was refused, and export skips on the file being absent,
        # not on the report's verdict — so the bundle would ship old pixels
        # under a manifest describing settings that never produced them.
        (folder / REFINED_FILE).unlink(missing_ok=True)
        report = RefineReport(
            asset=name,
            refiner=refiner.name,
            error_code="INPUT_NOT_FOUND",
            error_message=f"{source} does not exist",
            requested=settings,
        )
        store.write_model(folder / REPORT_FILE, report)
        return report
    if _asset_is_rebuilt(store, direction.name, name):
        return _keep_rebuilt(store, direction.name, name)
    if _asset_is_pixel_cut(store, direction.name, name):
        return _keep_pixel_cut(store, direction.name, name)
    if _asset_is_recipe(store, direction.name, name):
        return _keep_recipe(store, direction.name, name)
    kind = _asset_kind(store, direction.name, name)
    own_palette_count = _own_palette_target_count(kind, settings)
    if own_palette_count is not None:
        # The scene whose resolved settings name a colour count runs the same
        # two-step recipe `refine_direction`'s `_refine_alone` helper uses
        # (M4.2, ticket #70 review round 1) -- a single `reductionMode` call
        # here would be the exact recipe M4.1 measured as failing (91%
        # different from the spike). `raise_on_failure=True` is what keeps this
        # call's contract: asked to refine one named asset, a refusal is
        # raised, naming the asset, the same way `refiner.refine` raising
        # below already would be.
        return _refine_with_own_palette(
            store, direction, refiner, name, settings, target, own_palette_count,
            raise_on_failure=True,
        )
    sent, dropped = _settings_for_target(settings, target)
    result = refiner.refine(
        RefineRequest(
            input_path=source,
            output_path=folder / REFINED_FILE,
            settings=sent,
            candidate_id=candidate_id,
            target=target,
        )
    )
    report = _report(name, sent, result, refiner.name, candidate_id=candidate_id)
    return _record_success(
        store, folder, refiner, report, target, dropped_size_setting=dropped
    )


def _run_batch_and_record(
    store: Store,
    direction: Direction,
    refiner: Refiner,
    name: str,
    settings: RefinerSettings,
    target: tuple[int, int] | None,
    *,
    raise_on_failure: bool = False,
) -> RefineReport:
    """Refine one asset as a one-request batch call and write its receipt.

    Shared by the pinned final call of the two-step own-palette recipe and the
    plain single call of `_refine_alone`, so a refused asset's receipt, and a
    succeeded one's, are written exactly the same way whichever it came from
    (CLAUDE.md: "one broken asset never costs the bundle").

    `raise_on_failure` exists for exactly one caller:
    `_refine_with_own_palette`'s final call, on `refine_asset`'s behalf
    (ticket #70 review round 2). Asked to refine one named asset,
    `refine_asset` raises `RefinerError` on a refiner refusal rather than
    writing a receipt -- the same contract its plain single-`refine` call
    already has -- so its two-step recipe's final call must raise too,
    naming the asset, instead of falling into the receipt branch below.
    Nothing is written for the refused name when this raises -- no receipt,
    and `Refiner.batch`'s own contract already guarantees a refused item's
    `output_path` is left untouched -- which is `refine_asset`'s existing
    behaviour for a plain refusal too: the raise happens before anything about
    that asset is written, so a stale `refine.json`/`refined.png` pair from an
    earlier successful run is left matching each other, not replaced by a
    mismatched new one.
    """
    folder = store.asset_dir(direction.name, name)
    settings, dropped = _settings_for_target(settings, target)
    request = RefineRequest(
        input_path=folder / SOURCE_FILE,
        output_path=folder / REFINED_FILE,
        settings=settings,
        target=target,
    )
    # `Refiner.batch` guarantees the request's own output_path holds the
    # refined file when it returns, and that the list holds its one item.
    [item] = refiner.batch([request])
    if isinstance(item, RefineFailure):
        if raise_on_failure:
            raise RefinerError(
                f"asset {name!r}: {item.message}",
                exit_code=EXIT_ENGINE_FAILURE,
                code=item.code,
            )
        # A refused asset gets a report too, and loses any refined.png
        # an earlier run left: the folder has to say what happened, and
        # a stale PNG beside a refusal is the one arrangement that
        # reads as success. The code goes in error_code, not warnings:
        # warnings holds warning codes — the refiner's and PixelMagic's
        # own — and an error code is in neither. `route` is left empty
        # on purpose: the refiner took no route, and claiming
        # "preserve" would read as "copied through".
        (folder / REFINED_FILE).unlink(missing_ok=True)
        report = RefineReport(
            asset=name,
            refiner=refiner.name,
            error_code=item.code,
            error_message=item.message,
            requested=settings,
        )
    else:
        report = _report(name, settings, item, refiner.name)
        return _record_success(
            store, folder, refiner, report, target, dropped_size_setting=dropped
        )
    store.write_model(folder / REPORT_FILE, report)
    return report


def _refine_alone(
    store: Store,
    direction: Direction,
    refiner: Refiner,
    name: str,
    target: tuple[int, int] | None,
) -> RefineReport:
    """Refine one asset inside `refine_direction`'s default call.

    Used for the kept scene and a bare folder with no `asset.json` at all.
    Resolves this one name's own settings and runs the two-step own-palette
    recipe when its kind and those settings call for one
    (`_own_palette_target_count`), or the plain one-request batch otherwise (a
    bare folder, or a resolved `reductionMode` that is not a colour count).
    """
    settings = resolve_settings(HOUSE_BASELINE, direction.refiner_defaults)
    kind = _asset_kind(store, direction.name, name)
    count = _own_palette_target_count(kind, settings)
    if count is not None:
        return _refine_with_own_palette(store, direction, refiner, name, settings, target, count)
    return _run_batch_and_record(store, direction, refiner, name, settings, target)


def refine_direction(
    store: Store,
    direction: Direction,
    *,
    refiner: Refiner,
) -> list[RefineReport]:
    """Refine a direction's assets, each one on its own.

    Only the kept scene (and a bare folder with no `asset.json`) is sent to the
    refiner; every other asset is kept as drawn or left as it is, below. Each
    asset gets its own call, on purpose (spec "Palettes", 2026-09-24
    amendment): no two of them ever share a batch call. One asset's own
    settings go through `refine_asset`'s `override`, not through here.

    When the resolved settings ask for a colour count (`reductionMode` a
    digit string, `HOUSE_BASELINE`'s own "24"), the scene's palette is fit on
    its own untouched source before cohesion or clean-up ever runs, and the
    refine that follows is pinned to exactly that palette
    (`_refine_with_own_palette` -- the M4.2 two-step recipe that reproduced
    the M4.1 spike's own numbers). A named palette, `auto` or `none` has
    nothing for a fresh fit to reuse "own" from, so those keep a plain batch
    instead.

    A tile that is not a recipe tile (a bordered block's, or a plain patch's)
    and a prop that is not a rebuilt one (cut on the props sheet, or kept before
    props were rebuilt on their own pixel grid) are never refined: their road
    is gone. One that was already refined is left exactly as it is, picture and
    receipt, and still exports; one that never was gets an `OLD_ITEM_REFUSED`
    receipt saying what to do instead (`old_item_sentence`).

    Two more things stop one asset without stopping the direction (CLAUDE.md:
    "one broken asset never costs the bundle"), and both are pulled out of the
    per-asset loops before either runs: an asset whose own stored target falls
    outside what `validate_target` accepts gets a `TARGET_NOT_USABLE` failure
    receipt naming the asset and the target, and an asset whose `asset.json`
    will not parse gets an `ASSET_RECORD_UNREADABLE` receipt naming the asset
    and the file. Both are the same shape a refiner's own refusal gets.
    Everything else in `names` still refines.

    A refused *input* -- a bad target, a broken sidecar, or the refiner
    turning down one picture's own content, including either step of the
    scene's own-palette recipe -- never stops this call: it costs that one
    asset a receipt and the rest keep going, per the paragraph above.
    `_refine_with_own_palette` takes `raise_on_failure=False` here (its
    default), unlike `refine_asset`'s own use of it, for exactly that reason.
    A `RefinerError` is a different kind of failure and is never caught here:
    it means the call to the refiner itself went wrong -- the binary is
    missing, or the reply could not be read -- not that it looked at a picture
    and said no. When that happens the whole run stops and the exception
    travels out to the caller, but nothing already written is undone: every
    asset refined in an earlier call keeps its new `refine.json` and
    `refined.png`, each still describing exactly the call that produced it, so
    a caller that retries only has the assets the failure left to redo.

    An asset with a target is held to it. Saved size-changing settings are left
    out of its call with a `SIZE_SETTING_DROPPED` warning, and a result at
    another size from a refiner that honours a forced size gets its own
    `WRONG_SIZE` receipt and no `refined.png`; the rest of the direction
    carries on.

    A recipe tile, like a rebuilt prop, is kept as drawn: its picture is copied
    to `refined.png` and it is never sent to the refiner.
    """
    names = store.list_assets(direction.name)
    # Two things set an asset aside here, before the per-asset loops run,
    # so none of them ever sees it: an unreadable sidecar
    # and an unusable target both land in the same `failures` mapping, keyed
    # by name, holding the receipt's code and message — one mapping and one
    # receipt-writing loop below, rather than two of each for two failure
    # kinds that end up looking the same on disk.
    failures: dict[str, tuple[str, str]] = {}
    targets: dict[str, tuple[int, int] | None] = {}
    for name in names:
        try:
            targets[name] = _target_of(store, direction.name, name)
        except ValueError as error:
            failures[name] = (ASSET_RECORD_UNREADABLE, str(error))
    for name, target in targets.items():
        if target is None:
            continue
        try:
            validate_target(target, honours_forced_size=refiner.honours_forced_size)
        except ValueError as error:
            failures[name] = (TARGET_NOT_USABLE, f"asset {name!r}: {error}")
    # An old tile or prop is set aside before anything else reads it. One
    # already refined is left as it is; one never refined gets a receipt
    # through the same `failures` loop as the other PixelMagic refusals.
    kept_old: dict[str, RefineReport] = {}
    for name in names:
        if name in failures:
            continue
        old_kind = _old_item_kind(store, direction.name, name)
        if old_kind is None:
            continue
        refined_before = _refined_old_item(store, direction.name, name)
        if refined_before is None:
            failures[name] = (OLD_ITEM_REFUSED, old_item_sentence(name, old_kind))
        else:
            kept_old[name] = refined_before.model_copy(update={"route": OLD_ITEM_ROUTE})
    usable = [name for name in names if name not in failures and name not in kept_old]
    # A pixel-cut prop and a recipe tile are kept the same way as a rebuilt prop, so all
    # go in one list. None ever reaches the refiner: `refine_asset` writes its files (and its
    # own `INPUT_NOT_FOUND` receipt when `source.png` is missing).
    kept_as_drawn = [
        name
        for name in usable
        if _asset_is_rebuilt(store, direction.name, name)
        or _asset_is_pixel_cut(store, direction.name, name)
        or _asset_is_recipe(store, direction.name, name)
    ]
    solo = [name for name in usable if name not in kept_as_drawn]
    reports: dict[str, RefineReport] = dict(kept_old)
    for name, (code, message) in failures.items():
        # Same shape as a refiner's own refusal below: the receipt names the
        # problem, and any stale `refined.png` from a run made before this
        # asset went bad is removed rather than left to look current.
        folder = store.asset_dir(direction.name, name)
        (folder / REFINED_FILE).unlink(missing_ok=True)
        report = RefineReport(
            asset=name,
            refiner=refiner.name,
            error_code=code,
            error_message=message,
            requested=resolve_settings(HOUSE_BASELINE, direction.refiner_defaults),
        )
        store.write_model(folder / REPORT_FILE, report)
        reports[name] = report

    for name in solo:
        reports[name] = _refine_alone(store, direction, refiner, name, targets[name])

    for name in kept_as_drawn:
        try:
            reports[name] = refine_asset(store, direction, name, refiner=refiner)
        except ValueError as error:
            # The only ValueError a rebuilt prop or a recipe tile can raise here
            # is its picture not opening (its target was validated above). Same shape as the
            # failures loop: a receipt, no stale `refined.png`, and the rest
            # of the direction carries on.
            folder = store.asset_dir(direction.name, name)
            (folder / REFINED_FILE).unlink(missing_ok=True)
            reports[name] = RefineReport(
                asset=name,
                refiner=refiner.name,
                error_code=PICTURE_UNREADABLE,
                error_message=str(error),
                requested=resolve_settings(HOUSE_BASELINE, direction.refiner_defaults),
            )
            store.write_model(folder / REPORT_FILE, reports[name])

    # Every name is in `reports`: the failure loop covered the assets this call
    # refuses to refine, `kept_old` the old tiles and props left as they are,
    # `solo` and the loop after it the rest of `names`.
    return [reports[name] for name in names]


def load_report(store: Store, direction_name: str, name: str) -> RefineReport:
    """Read one asset's `refine.json` back off disk."""
    return store.read_model(
        store.asset_dir(direction_name, name) / REPORT_FILE, RefineReport
    )


@dataclass(frozen=True)
class RefineTally:
    """What a refine run did, counted once for every face.

    `refined` counts only what went through the refiner; rebuilt props, pixel-cut
    props, recipe tiles and old items left as they are are counted on their own, and
    `refused` names every asset that carries an error code (PixelMagic's own
    refusals and the refiner's), so no face can report a refused asset as done.
    """

    refined: int
    rebuilt: int
    pixel_cut: int
    recipe: int
    old_items: int
    refused: tuple[str, ...]


def tally_reports(reports: Sequence[RefineReport]) -> RefineTally:
    """Count a run's receipts; an old item left as it is is never a refusal."""
    skipped = (REBUILD_ROUTE, PIXEL_CUT_ROUTE, RECIPE_ROUTE, OLD_ITEM_ROUTE)
    return RefineTally(
        refined=sum(
            1 for r in reports if r.route not in skipped and not r.error_code
        ),
        rebuilt=sum(1 for r in reports if r.route == REBUILD_ROUTE),
        pixel_cut=sum(1 for r in reports if r.route == PIXEL_CUT_ROUTE),
        recipe=sum(1 for r in reports if r.route == RECIPE_ROUTE),
        old_items=sum(1 for r in reports if r.route == OLD_ITEM_ROUTE),
        refused=tuple(
            r.asset for r in reports if r.error_code and r.route != OLD_ITEM_ROUTE
        ),
    )


def refused_suffix(tally: RefineTally) -> str:
    """The summary line's ending: empty, or "; N refused: a, b"."""
    if not tally.refused:
        return ""
    return f"; {len(tally.refused)} refused: {', '.join(tally.refused)}"


def needs_attention(report: RefineReport) -> bool:
    """True when the refiner was unsure of the pixel grid.

    `LOW_GRID_CONFIDENCE` means the refiner guessed the grid and is not sure;
    the person should check the scene's pitch and set it if it is wrong.
    """
    return LOW_GRID_CONFIDENCE in report.warnings


def grid_needs_attention(report: RefineReport, record: AssetRecord | None) -> bool:
    """`needs_attention`, but only for an asset whose pitch a person can still set.

    An old tile or prop, a pixel-cut prop, and a rebuilt prop with a receipt from before it was
    rebuilt, can do nothing about an uncertain grid, so it is not flagged.
    `record` is the asset's own sidecar, or None when it will not read.
    """
    if record and (old_item_kind(record) or record.grid_rebuild or record.pixel_cut):
        return False
    return needs_attention(report)


def stored_grid_needs_attention(store: Store, direction_name: str, report: RefineReport) -> bool:
    """`grid_needs_attention` for a receipt, reading the asset's own sidecar itself."""
    try:
        record = store.read_model(
            store.asset_dir(direction_name, report.asset) / ASSET_FILE, AssetRecord
        )
    except (OSError, ValueError):
        record = None
    return grid_needs_attention(report, record)


def require_asset(store: Store, direction_name: str, asset: str) -> str:
    """Return `asset` if this direction has one by that name, or refuse it.

    A name that was never an asset is a mistake to be told about, not a
    receipt to file: `refine_asset` would answer it with an `INPUT_NOT_FOUND`
    report, which is the right answer for a `source.png` that went missing and
    the wrong one for a typo. All three faces need this, so it lives here — the
    same argument `set_refiner_defaults` makes for its own check.
    """
    name = check_name(asset, "asset")
    known = store.list_assets(direction_name)
    if name not in known:
        folder = store.asset_dir(direction_name, name)
        if folder.is_dir():
            # The asset exists; the picture inside it was deleted by hand.
            raise ValueError(
                f"asset {name!r} exists but its picture is missing: "
                f"{folder / SOURCE_FILE} was not found"
            )
        raise ValueError(
            f"{direction_name} has no asset called {name!r}; "
            f"known: {', '.join(known) or 'none yet'}"
        )
    return name
