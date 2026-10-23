"""Every model and enum PixelMagic persists.

This module is the project's vocabulary: stages, providers and the CLI all
speak in these types, and each one is a JSON sidecar on disk. Keeping them in
one file means a reader can learn the whole data model in one sitting.

Every timestamp is an `AwareDatetime`, never a bare `datetime`. Ids and
folder names are built from these moments, and a naive one — a `datetime.now()`
somewhere without a timezone — would be written down as if it were UTC and
silently misdate a record. Rejecting it at the door is cheaper than finding it
later in a sidecar.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field


class PromptKind(StrEnum):
    """Which prompt template a generation uses."""

    SCENE = "scene"
    PROP = "prop"
    # Kept only so an old request.json ("kind": "parts-sheet") still loads; the
    # props sheet it drew is gone, and so is its template.
    PARTS_SHEET = "parts-sheet"
    # Kept only so a pre-#90 request.json ("kind": "tileset-sheet") still
    # loads; no code after this ticket ever writes or asks for it again —
    # its template file and TEMPLATE_FILES entry are both dropped.
    TILESET_SHEET = "tileset-sheet"
    # Kept only so an old request.json ("kind": "patch-sheet") still loads;
    # nothing writes or asks for it any more, and its template is gone.
    PATCH_SHEET = "patch-sheet"
    # Kept only so an old request.json ("kind": "surface-block") still loads;
    # the bordered surface it drew is gone, and so is its template.
    SURFACE_BLOCK = "surface-block"
    GROUP_REDRAW = "group-redraw"


class GenerateKind(StrEnum):
    """The two kinds of picture a person or an agent asks for directly.

    `PromptKind` also holds the two sheet templates, but those describe a
    scene that already exists and are reached through `decompose`. Offering
    them here would let someone ask for a sheet of a scene nobody has drawn,
    so every face's "draw me a picture" takes this narrower set.
    """

    SCENE = "scene"
    PROP = "prop"


class SheetKind(StrEnum):
    """The two sheet types a decomposition asks for."""

    PROPS = "props"
    TILES = "tiles"


class AssetKind(StrEnum):
    """What a finished asset is, as recorded in the export manifest."""

    SCENE = "scene"
    PROP = "prop"
    TILE = "tile"


class BackgroundPolicy(StrEnum):
    """What the image behind the subject should be.

    Transparent is always asked for first; the key colours are the fallback
    for services that cannot produce alpha (spec locked decision 6).
    """

    TRANSPARENT = "transparent"
    MAGENTA = "magenta"
    LIME = "lime"


#: RGB of each key colour. TRANSPARENT deliberately has no entry.
KEY_COLOURS: dict[BackgroundPolicy, tuple[int, int, int]] = {
    BackgroundPolicy.MAGENTA: (255, 0, 255),
    BackgroundPolicy.LIME: (0, 255, 0),
}


#: The art-pixel size of one grid cell, in a direction that has not said
#: otherwise. Tiles are this square and a prop's footprint is measured in it
#: (spec locked decision 3). The bounds are the spec's: below 4 no tile can
#: hold a picture, above 128 it is not a tile any more.
DEFAULT_CELL_SIZE = 16
MIN_CELL_SIZE = 4
MAX_CELL_SIZE = 128

#: The pitch — source pixels per art pixel — a person or an agent may set by
#: hand (spec §5). Below 0.5 the "art" would have more pixels than the
#: picture; above 64 a 1024-pixel sheet would be sixteen art pixels wide.
MIN_PITCH = 0.5
MAX_PITCH = 64.0


class PitchSource(StrEnum):
    """Where a pitch came from: a measurement, a person, or a fallback.

    A measurement is the refiner's reading, or, on a piece rebuilt on its own
    pixel grid, the pitch measured from that grid. `UNSURE` is a reading the
    refiner flagged with its own `LOW_GRID_CONFIDENCE` warning: the number is
    kept but a person should check it. `FALLBACK` means there was no usable
    reading (none at all, or one outside `MIN_PITCH`..`MAX_PITCH`) and a pitch
    of 1.0 was used instead.
    """

    MEASURED = "measured"
    SET = "set"
    UNSURE = "unsure"
    FALLBACK = "fallback"


class GridRebuild(BaseModel):
    """The record of a piece rebuilt on its own pixel grid.

    The finder read the grid off the redraw reply and the piece was saved at
    native size, one pixel per cell. Refine and export recognise a rebuilt
    piece by this record alone. `pitch_x` and `pitch_y` are in reply pixels,
    not original-picture pixels (`Cutout.pitch` holds those).
    """

    #: The one square size the piece was sampled at, in reply pixels (older records may
    #: hold two different readings, one per axis).
    pitch_x: float
    pitch_y: float
    #: Where the grid starts, in the piece's own frame.
    phase_x: float
    phase_y: float
    #: The lower of the two axis scores, taken at that size.
    confidence: float
    #: Size of the rebuilt picture, `(w, h)`, one pixel per cell.
    native_size: tuple[int, int]


class CleanupRecord(BaseModel):
    """The sidecar of one picture's clean-up (`cleanups/<generation>/cleanup.json`).

    Holds what a face tells the person: the size the finder found and how sure
    it was, the grid actually used, and the settings the result was made with.
    """

    generation_id: str
    created: AwareDatetime
    #: Cache key (sha256 of picture, job, settings and code version) and the
    #: clean-up maths version it was made with.
    cache_key: str
    code_version: int
    #: The settings asked for; None means "not given".
    size: float | None
    colours: int | None
    #: The .aseprite whose colours the result must use (its name only, not its
    #: path) and those colours as hex, sorted; None when no palette was asked for.
    palette_file: str | None = None
    palette: list[str] | None = None
    #: What the finder read off the picture, whatever size was typed.
    found_pitch_x: float
    found_pitch_y: float
    found_score_x: float
    found_score_y: float
    #: The one square pixel size the result was cut at: the typed size, else the smaller
    #: of the two axis readings.
    pixel_size: float
    #: True when either axis scored below the finder's bar, the two readings disagree
    #: (`cleanup.READINGS_DISAGREE`), or either reading is within `cleanup.NEAR_SEARCH_TOP`
    #: of the top of the finder's range (`rebuild.PITCH_MAX`); about the found size,
    #: whatever size was typed.
    unsure: bool
    #: The grid the result was cut on, and whether its size was found or typed.
    grid: GridRebuild
    size_source: Literal["found", "typed"]
    #: Colours in the result.
    colour_count: int
    #: The result's distinct opaque colours as lowercase `rrggbb`, most used first.
    shown_colours: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Pitch(BaseModel):
    """One sheet's pitch — source pixels per art pixel — and where it came from.

    Recorded with its source (spec locked decision 1) so that a later reader
    can tell a number the refiner measured from one somebody typed in.
    """

    value: float
    source: PitchSource


class TileRole(StrEnum):
    """Where a tile sits in its 3 × 3 block: the centre, an edge, or a corner.

    Values are the words the manifest uses; an engine reads them, never the
    screen (spec locked decisions 4 and 8). Set by M3.3; `None` until then.
    """

    CENTRE = "centre"
    EDGE_N = "edge-n"
    EDGE_E = "edge-e"
    EDGE_S = "edge-s"
    EDGE_W = "edge-w"
    CORNER_NW = "corner-nw"
    CORNER_NE = "corner-ne"
    CORNER_SW = "corner-sw"
    CORNER_SE = "corner-se"


class ReviewStatus(StrEnum):
    """Where an image sits in the approval loop."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Reviewer(StrEnum):
    """Who made an approval decision."""

    HUMAN = "human"
    AUTO = "auto"
    AGENT = "agent"


class RefinerSettings(BaseModel):
    """Settings handed to PixelRefiner, in its own layered vocabulary.

    `preset` and `scale` are whole-value choices; `quick` and `advanced` are
    key/value layers that merge, so a later layer overrides only the keys it
    actually sets (see `stages.refine.resolve_settings`).
    """

    preset: str | None = None
    quick: dict[str, str] = Field(default_factory=dict)
    advanced: dict[str, object] = Field(default_factory=dict)
    scale: int | None = None


class Project(BaseModel):
    """A game. Holds one or more art directions."""

    name: str
    created: AwareDatetime
    default_direction: str | None = None
    #: Which image service draws for this game: a service name, never a secret.
    #: None means nobody chose, and nothing draws until someone does (`services.chosen_service`).
    service: str | None = None


class SceneChoice(BaseModel):
    """The kept scene a person chose for Refine and Export, by its generation id."""

    generation: str


class Direction(BaseModel):
    """One art direction inside a project: the style and its defaults."""

    name: str
    created: AwareDatetime
    style: str
    donts: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    service: str | None = None
    model: str | None = None
    refiner_defaults: RefinerSettings = Field(default_factory=RefinerSettings)
    approved_prompts: list[str] = Field(default_factory=list)
    #: The art-pixel size of one grid cell (spec locked decision 3). Tiles are
    #: refined to exactly this square; a prop's footprint is counted in it.
    #: Changing it later is allowed and means re-refining the tiles.
    cell_size: int = Field(default=DEFAULT_CELL_SIZE, ge=MIN_CELL_SIZE, le=MAX_CELL_SIZE)
    #: Original-picture pixels per art pixel for this direction's props (#147),
    #: set from one piece's wanted height. None means no scale: props are drawn
    #: and checked as they were before there was one.
    prop_scale: float | None = Field(default=None, ge=MIN_PITCH, le=MAX_PITCH)


class GenerationRequest(BaseModel):
    """Everything sent to an image service, including the expanded prompt.

    `references` holds file NAMES inside the direction's `references/` folder —
    never absolute paths and never paths into `generations/`. `stages.generate`
    resolves them against `store.references_dir(direction)` and hands the real
    paths to the service as `reference_paths`; a service never builds a path of
    its own and never needs to know the store. Anything that
    wants to condition a request on an image must promote that image to a
    reference first (`stages.generate.promote`), so a request can always be
    replayed from the direction folder alone.

    A picture brought in from disk (`stages.generate.import_picture`) was never
    asked of a service, so `prompt`, `service` and `background` are None and
    `imported_from` holds the original file's name (not its path). Nothing
    fakes a value for them: the record says what happened.
    """

    id: str
    created: AwareDatetime
    subject: str
    kind: PromptKind
    prompt: str | None = None
    service: str | None = None
    model: str | None = None
    size: tuple[int, int]
    background: BackgroundPolicy | None = None
    references: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    imported_from: str | None = None
    #: The grid a group redraw asked for (`PromptKind.GROUP_REDRAW` only);
    #: None on every other request, and on any request written before #106.
    group: GridSpec | None = None


class GenerationResult(BaseModel):
    """Provenance for one image a service actually produced."""

    id: str
    created: AwareDatetime
    service: str | None = None
    model: str | None = None
    session_id: str | None = None
    response_id: str | None = None
    elapsed_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class Review(BaseModel):
    """The approval hook's verdict on one image."""

    status: ReviewStatus
    reviewer: Reviewer
    feedback: str | None = None
    decided: AwareDatetime | None = None


class GenerationRecord(BaseModel):
    """The three sidecars of one generation plus where its PNG landed."""

    request: GenerationRequest
    result: GenerationResult
    review: Review
    image_path: Path


class Point(BaseModel):
    """One click on the original picture, in its own pixels."""

    x: int
    y: int
    #: 1 = this pixel belongs to the object; 0 = it does not.
    label: Literal[0, 1]


class Suggestion(BaseModel):
    """One object outline the segmenter proposed, after duplicates were removed."""

    #: 1-based, largest first, stable: saved with the sidecar and never re-sorted.
    number: int
    #: `(x0, y0, x1, y1)` on the original picture; right and bottom exclusive.
    box: tuple[int, int, int, int]
    #: Opaque pixels in the mask.
    area: int
    #: File name of the box-cropped mask inside `suggestions/`.
    mask: str
    #: The segmenter's own confidence, when it gives one.
    score: float | None = None


class SuggestionSet(BaseModel):
    """`suggestions/suggestions.json`: everything the segmenter proposed for one picture."""

    segmenter: str
    #: Id of the approved generation the suggestions were made from.
    source_generation: str
    #: `(width, height)` of the original picture.
    picture_size: tuple[int, int]
    #: How many masks the segmenter returned before duplicates were removed.
    raw_count: int
    #: How many suggestions each contact sheet shows.
    page_size: int
    created: AwareDatetime
    suggestions: list[Suggestion] = Field(default_factory=list)


class PickMethod(StrEnum):
    """How a person or an agent chose an object."""

    SUGGESTION = "suggestion"
    POINTS = "points"
    BOX = "box"
    LASSO = "lasso"


class Pick(BaseModel):
    """One object chosen for redrawing, with the outline it was chosen by."""

    #: 1-based and never reused, even after a pick is removed.
    number: int
    name: str | None = None
    method: PickMethod
    #: The suggestion number this pick took, for `PickMethod.SUGGESTION`.
    suggestion: int | None = None
    #: The clicks that made the mask, for `PickMethod.POINTS`.
    points: list[Point] = Field(default_factory=list)
    #: `(x0, y0, x1, y1)` drawn by hand, for `PickMethod.BOX`.
    box: tuple[int, int, int, int] | None = None
    #: The lasso's corners, `(x, y)` on the original, for `PickMethod.LASSO`.
    loop: list[tuple[int, int]] | None = None
    #: Clicks made after a lasso: 1 = add a piece, 0 = remove one.
    fixes: list[Point] = Field(default_factory=list)
    #: File name of the box-cropped mask inside `picks/`.
    mask: str
    #: `(x0, y0, x1, y1)` of the mask on the original picture.
    picture_box: tuple[int, int, int, int]
    #: Opaque pixels in the mask.
    area: int
    created: AwareDatetime


class PickUndo(BaseModel):
    """What one change to a pick's outline overwrote, so Undo can put it back."""

    #: The pick the change was to.
    number: int
    #: The pick as it was before the change; None when the change made the pick.
    before: Pick | None = None
    #: File name, inside `picks/undo/`, of a copy of the old mask; None for a new pick.
    mask_copy: str | None = None


class PickSet(BaseModel):
    """`picks/picks.json`: every object chosen so far."""

    picks: list[Pick] = Field(default_factory=list)
    #: Changes Undo can take back, oldest first. Kept in the same file so it is
    #: saved in the same atomic write as the change it records.
    undo: list[PickUndo] = Field(default_factory=list)
    #: The highest pick number ever handed out. Kept beside the list, not
    #: derived from it, so a number stays retired after its pick is removed:
    #: a later redraw must never overwrite the wrong piece.
    last_number: int = 0


class GridSpec(BaseModel):
    """The grid one group redraw asks for."""

    cols: int
    rows: int
    #: How many cells are used; the rest of the grid stays empty.
    count: int


class GroupCell(BaseModel):
    """Which pick sits in which cell of a group's grid."""

    #: The pick's number.
    pick: int
    #: `(row, col)`, both 0-based.
    cell: tuple[int, int]


class GroupAttempt(BaseModel):
    """One call to the image service for a group."""

    #: The generation the call made; None when the call failed before one was kept.
    generation: str | None = None
    #: The reference background this attempt used (transparent or a key colour).
    background: BackgroundPolicy
    ok: bool
    error: str | None = None
    seconds: float


class CellReceipt(BaseModel):
    """What became of one pick after its cell was cut out of the reply."""

    pick: int
    #: `(width, height)` of the piece cut from the cell; None when nothing was cut.
    piece_size: tuple[int, int] | None = None
    #: The piece's share of its cell's area, 0 to 1; None when nothing was cut.
    share_in_cell: float | None = None
    #: A sentence naming the problem, or None when the cell was fine.
    warning: str | None = None


class Group(BaseModel):
    """`groups/group-NN/group.json`: one redraw of several picks at once."""

    #: "group-NN".
    id: str
    cols: int
    rows: int
    cells: list[GroupCell] = Field(default_factory=list)
    #: The key colour (or transparent) the reference picture was drawn on.
    reference_key: BackgroundPolicy
    attempts: list[GroupAttempt] = Field(default_factory=list)
    #: The generation of the successful call; None until one succeeds.
    generation: str | None = None
    receipts: list[CellReceipt] = Field(default_factory=list)


class PixelCut(BaseModel):
    """The record of a piece cut straight from a pixel-art picture's own pixels.

    The piece is the native picture's own pixels inside the pick's outline, never
    redrawn or resampled. Refine and export recognise it by this record alone, as they
    do a rebuilt piece by `GridRebuild`.
    """

    #: The one square size the picture was read at, in picture pixels per art pixel.
    pixel_size: float
    #: Whether that size was typed or found (`PixelGrid.size_source`).
    size_source: Literal["found", "typed"]
    #: True when the size was found and the finder was not sure of it.
    unsure: bool
    #: Size of the piece, `(w, h)`, one pixel per cell.
    native_size: tuple[int, int]


class CutoutOrigin(BaseModel):
    """Where an object-by-object cutout came from."""

    #: The pick's number.
    pick: int
    #: The group's id ("group-NN"). None on a piece cut straight from a pixel-art
    #: picture (`Cutout.pixel_cut`), which has no group.
    group: str | None = None
    #: `(row, col)` of the cell the piece was cut from. None, like `group`, on a pixel cut.
    cell: tuple[int, int] | None = None
    #: `(x0, y0, x1, y1)` of the object on the original picture.
    picture_box: tuple[int, int, int, int]
    #: Original pixels per piece pixel. The piece keeps its own shape and its
    #: longer side comes from the object's box on the original.
    scale: float
    segmenter: str
    #: Every pick that went into a merged piece, including the ones its own
    #: members carried. Nothing writes it any more (merging pieces is gone);
    #: it is kept so a sidecar written by an earlier merge still reads. Empty
    #: on any other piece. `pick` stays the first member's.
    merged_picks: list[int] = Field(default_factory=list)


class RecipeRecord(BaseModel):
    """How a recipe tile was made, kept on its cutout, asset and manifest entry.

    A recipe tile is drawn from a few colours read off a small box on a
    picture, not cut from it. This record is what lets the same tiles be drawn
    again and lets export say where they came from. A record made before the
    surface was drawn by hand has a `point` and a `segmenter` and no `shapes`
    or `box`; a new one has `shapes` and `box` and neither of the others.
    """

    #: The recipe: "plaster", "stone" or "wood" (`stages.recipes.RECIPE_KINDS`).
    kind: str
    #: Always "block" (8 blocks of 2 x 2 tiles); kept so a bundle says how it was made.
    layout: str
    #: The tile's corner within its block: "nw", "ne", "sw" or "se".
    place: str | None = None
    #: The surface's name: `Surface.name`, the slug that groups its tiles
    #: (for example `wood-floor`), not the words as typed.
    surface: str
    #: The click `(x, y)` on the picture that outlined the surface. Only on
    #: records made before surfaces were drawn by hand; None on a new one.
    point: tuple[int, int] | None = None
    #: The name of the segmenter that outlined it. Only on records made before
    #: surfaces were drawn by hand; None on a new one (no finder is used).
    segmenter: str | None = None
    #: The rough shapes drawn over the surface, each a list of `(x, y)` corners
    #: in the picture's own pixels. None on an older record.
    shapes: list[list[tuple[int, int]]] | None = None
    #: The colour box `(x0, y0, x1, y1)` on the picture, right and bottom
    #: exclusive, the colours were read from. None on an older record.
    box: tuple[int, int, int, int] | None = None
    #: The recipe's colours as `#rrggbb`, dark to light.
    shades: list[str] = Field(default_factory=list)
    #: One number per shade, same order: the fraction (0-1, four decimals) of
    #: the box's pixels that shade's cluster held.
    shares: list[float] = Field(default_factory=list)


class Cutout(BaseModel):
    """One piece sliced out of a sheet."""

    name: str
    ai_name: str | None = None
    sheet: str
    kind: SheetKind
    bbox: tuple[int, int, int, int]
    cell: tuple[int, int] | None = None
    #: The sheet's pitch when this piece was cut — source pixels per art
    #: pixel. `None` on a tile: a surface's sheet is never measured (spec §2).
    #: A recipe tile has none either: it was drawn, not cut from a sheet.
    pitch: float | None = None
    #: Where that pitch came from — the refiner's reading (`MEASURED`) or a
    #: person (`SET`, written by M3.4). It travels with the piece so that
    #: export never has to search a decomposition file to say it.
    pitch_source: PitchSource | None = None
    #: The size the refiner is told to produce, in art pixels, `(w, h)`
    #: (spec locked decision 2). Props: box ÷ pitch. Tiles: the cell size,
    #: square. `None` only on a sidecar written before M3.
    target: tuple[int, int] | None = None
    #: How many grid cells the piece covers, `(cols, rows)`. Props only;
    #: filled by M3.2.
    footprint: tuple[int, int] | None = None
    #: Where the piece stands on its bottom cell, in art pixels from its own
    #: top-left. Props only; filled by M3.2.
    anchor: tuple[int, int] | None = None
    #: Which part of a 3 × 3 block this tile is. Old bordered tiles only; nothing
    #: fills it now (#179), and it is kept so old sidecars load.
    role: TileRole | None = None
    #: Where this tile sits in its repeating patch, (row, column), both
    #: 0-2. Old plain-patch tiles only; nothing fills it now (#179), and it is
    #: kept so old sidecars load. A recipe tile gets `variant` instead.
    position: tuple[int, int] | None = None
    #: Which of a recipe surface's 8 drawn variants (1-8) this tile is. A
    #: recipe tile only; never together with `role` or `position`. None on
    #: every other piece and on any sidecar written before recipes.
    variant: int | None = None
    #: How a recipe tile was made (kind, shades, touch-up time). None on every
    #: other piece and on any sidecar written before recipes.
    recipe: RecipeRecord | None = None
    #: Where this piece came from when it was cut out object by object
    #: rather than off a sheet. None on the sheet road and on any sidecar
    #: written before #106.
    origin: CutoutOrigin | None = None
    #: Set when the piece was rebuilt on its own pixel grid. Refine and export
    #: recognise a rebuilt piece by this record alone. None otherwise and on
    #: any sidecar written before it existed.
    grid_rebuild: GridRebuild | None = None
    #: Set when the piece was cut straight from a pixel-art picture's own pixels.
    #: Refine and export recognise it by this record alone. None otherwise.
    pixel_cut: PixelCut | None = None


class PixelGrid(BaseModel):
    """The pixel grid of a picture that is already pixel art, for cutting a prop straight from it.

    Its presence on a `Decomposition` marks the pixel-cut road: the object finder is
    asked on an enlarged copy of the native picture (`finder.png`) and no image service
    is paid. Sizes and phases are in the picture's own pixels.
    """

    #: The one square size used, in picture pixels per art pixel: the typed size, else the
    #: smaller of the two axis readings.
    pixel_size: float
    #: Where the grid starts on each axis, in the picture's own frame.
    phase_x: float
    phase_y: float
    #: Whether `pixel_size` was typed or found (the words `CleanupRecord.size_source` uses).
    size_source: Literal["found", "typed"]
    #: What the finder read on each axis and how well it scored, whatever size was typed.
    found_pitch_x: float
    found_pitch_y: float
    found_score_x: float
    found_score_y: float
    #: True when the found readings are doubtful (`cleanup.readings_unsure`).
    unsure: bool
    #: The picture's size and the native picture's size (one pixel per cell), `(w, h)`.
    picture_size: tuple[int, int]
    native_size: tuple[int, int]
    #: How many times `finder.png` enlarges the native picture (nearest-neighbour).
    finder_scale: int


class Decomposition(BaseModel):
    """One approved image broken into sheets and then into cutouts."""

    id: str
    created: AwareDatetime
    source_generation: str
    sheets: dict[SheetKind, str] = Field(default_factory=dict)
    #: One entry per typed surface's own sheet: its `name` (the slug of what
    #: was typed) to the generation id that drew it. `sheets` above stays
    #: props-only — #90 can produce any number of tile sheets, not one.
    tile_sheets: dict[str, str] = Field(default_factory=dict)
    sheet_prompts: dict[str, str] = Field(default_factory=dict)
    sheet_services: dict[str, str] = Field(default_factory=dict)
    cutouts: list[Cutout] = Field(default_factory=list)
    #: One pitch per measured sheet (spec locked decision 1). Only
    #: `SheetKind.PROPS` is ever a key: a surface's pieces take their
    #: size from the cell size instead, so its sheet is never measured.
    pitches: dict[SheetKind, Pitch] = Field(default_factory=dict)
    #: Anything the operator should know about how this run was cut up. Empty
    #: on a clean run. A change of method belongs here rather than only on
    #: screen, where it would be unexplained a week later. An older run may
    #: hold a note about a bordered surface that was not drawn as a 3 × 3 block.
    warnings: list[str] = Field(default_factory=list)
    #: The name of the segmenter that made this decomposition's suggestions
    #: ("stub", "cli"). None on the sheet road and before #106.
    segmenter: str | None = None
    #: The picture's pixel grid when this decomposition is cut straight from a pixel-art
    #: picture's own pixels; None on every other road.
    pixel_grid: PixelGrid | None = None


class AssetRecord(BaseModel):
    """A cutout promoted to an asset, with the provenance export needs.

    The six grid fields are copied from the `Cutout` at promotion so that
    refine and export read one sidecar (spec §1). A kept scene gets `pitch`,
    `pitch_source` and `target` straight from `promote_generation` and never
    has a footprint, an anchor or a role.
    """

    name: str
    kind: AssetKind
    source_generation: str
    source_prompt: str | None = None
    service: str | None = None
    cell: tuple[int, int] | None = None
    pitch: float | None = None
    pitch_source: PitchSource | None = None
    target: tuple[int, int] | None = None
    footprint: tuple[int, int] | None = None
    anchor: tuple[int, int] | None = None
    role: TileRole | None = None
    #: Where this tile sits in its repeating patch, (row, column), both
    #: 0-2. Old plain-patch tiles only; nothing fills it now (#179), and it is
    #: kept so old sidecars load. A recipe tile gets `variant` instead.
    position: tuple[int, int] | None = None
    #: Which of a recipe surface's 8 drawn variants (1-8) this tile is. A
    #: recipe tile only; never together with `role` or `position`. None on
    #: every other piece and on any sidecar written before recipes.
    variant: int | None = None
    #: How a recipe tile was made (kind, shades, touch-up time). None on every
    #: other piece and on any sidecar written before recipes.
    recipe: RecipeRecord | None = None
    #: Copied from the cutout's `origin` at keep; None on the sheet road.
    segmenter: str | None = None
    #: Copied from the cutout at keep. Refine and export recognise a rebuilt
    #: piece by this record alone. None on any other piece.
    grid_rebuild: GridRebuild | None = None
    #: Copied from the cutout at keep. Refine and export recognise a piece cut straight
    #: from a pixel-art picture by this record alone. None on any other piece.
    pixel_cut: PixelCut | None = None
    #: The id of the decomposition this piece was kept from; export reads where
    #: the piece sat on the original picture from that folder. None on a kept
    #: scene and on anything kept before this field existed.
    decomposition: str | None = None


class GridOffset(BaseModel):
    """Where a grid starts, in input pixels.

    Fractional on purpose, like `GridCandidate.cell_w`: PixelRefiner reports a
    fractional origin whenever the sprite was upscaled by a non-integer factor,
    and rounding it here would throw away the grid the refiner actually found.
    """

    x: float = 0
    y: float = 0


class GridCandidate(BaseModel):
    """One grid PixelRefiner considered, named the way the CLI names it.

    `cell_w` and `cell_h` are floats on purpose: a sprite blown up by a
    non-integer factor has a fractional cell size, and rounding it here would
    throw away the grid the refiner actually found.
    """

    out_w: int
    out_h: int
    cell_w: float
    cell_h: float
    offset: GridOffset = Field(default_factory=GridOffset)
    method: str = ""
    total_score: float
    confidence: float


class RefineCandidate(BaseModel):
    """One result PixelRefiner could produce, named the way it names it.

    Not the same list as `GridCandidate`: those are the grids the refiner
    scored, and they carry no id. These carry the ids `refine --candidate-id`
    accepts, which is the only way to ask for a different one — so this is
    what a face offers when the refiner says it is unsure.
    """

    id: str
    kind: str = ""
    recommended: bool = False
    processing_mode: str = ""
    out_w: int = 0
    out_h: int = 0


class CohesionRecord(BaseModel):
    """Whether PixelRefiner's cohesion and clean-up stages actually ran on one piece.

    Built off the refiner's own `effectiveOptions`, never off what was asked
    for (spec "Defaults and the record"): a per-piece override can turn
    cohesion off even when the direction's own default is on, so "requested"
    and "ran" can disagree. Every field defaults to "did not run" -- the
    truthful answer for a sidecar written before this ticket, for a
    `--refiner stub` bundle, and for a real PixelRefiner build from before
    M4.1, none of which ever mention these keys.
    """

    cohesion: bool = False
    cohesion_families: int | None = None
    cohesion_tones: int | None = None
    cleanup: bool = False
    #: Kept only so old `refine.json` files load. A tile block's seam score with
    #: cohesion on and with it off, written by the cohesion-off comparison
    #: (ticket #81) that tile blocks no longer get (#179). Nothing writes
    #: them now; they are `None` on every new receipt.
    centre_seam_with_cohesion: float | None = None
    centre_seam_without_cohesion: float | None = None


class RefineReport(BaseModel):
    """What the refiner did to one asset, saved as `refine.json`."""

    asset: str
    #: The route the refiner took: refine, convert, preserve, rebuild, recipe or stub. Empty when it
    #: never ran — a refused asset must not claim it was preserved, because
    #: `ManifestEntry.route` is fed from here and a reader could not then tell
    #: "copied through unchanged" from "the refiner would not touch it".
    route: str = ""
    confidence: float = 0.0
    #: Which refiner produced this — "stub", "pixel-refiner",
    #: "pixelmagic-grid-rebuild" or "pixelmagic-recipe". A bundle built
    #: by the stand-in has to be distinguishable from a real one.
    refiner: str = ""
    classification: str | None = None
    #: Holds codes from `refiner.base.WARNING_CODES` (the refiner's own) and
    #: `refiner.base.PIXELMAGIC_WARNING_CODES` (added by PixelMagic after the
    #: refiner ran — `SEAM_VISIBLE` on a centre tile), and nothing else.
    warnings: list[str] = Field(default_factory=list)
    #: Set instead of a warning when the refiner refused this asset outright;
    #: an error code is not a member of the warning vocabulary.
    error_code: str | None = None
    #: The refiner's own words for a refusal, including its hint. Empty on
    #: success, so the operator sees why — not just that something failed.
    error_message: str = ""
    grid_candidates: list[GridCandidate] = Field(default_factory=list)
    #: The alternatives the refiner would accept as `--candidate-id`, so a
    #: face can offer them without running `analyze` again.
    candidates: list[RefineCandidate] = Field(default_factory=list)
    #: Which of them was asked for, when one was. `None` means the refiner
    #: chose for itself, which is not the same as having been asked for its
    #: own recommendation.
    candidate_id: str | None = None
    requested: RefinerSettings = Field(default_factory=RefinerSettings)
    effective_options: dict[str, object] = Field(default_factory=dict)
    palette: list[str] = Field(default_factory=list)
    width: int = 0
    height: int = 0
    color_count: int = 0
    #: Whether cohesion and clean-up actually ran, read off the refiner's own
    #: report (`_cohesion_record`, `stages/refine.py`). Defaults to "did not
    #: run" so a `refine.json` written before this ticket still loads.
    cohesion: CohesionRecord = Field(default_factory=CohesionRecord)


class ManifestEntry(BaseModel):
    """One asset's row in the export manifest."""

    name: str
    kind: AssetKind
    width: int
    height: int
    palette: list[str] = Field(default_factory=list)
    #: Read from an old manifest only. A new manifest does not write `grid`,
    #: `role` or `position` (see `export.MANIFEST_DROPPED_FIELDS`): only the
    #: removed roads filled them (#179).
    grid: tuple[int, int] | None = None
    source_prompt: str | None = None
    service: str | None = None
    refiner: RefinerSettings = Field(default_factory=RefinerSettings)
    #: Which tool refined it ("pixel-refiner" when the real one ran, "stub"
    #: when the stand-in did) and which route it took. A
    #: game — or a person — must be able to tell a real pixel-art bundle from
    #: one the stand-in copied through untouched.
    refiner_tool: str = ""
    route: str = ""
    #: The warning codes from this asset's `refine.json`, verbatim and
    #: possibly empty: the refiner's own (`refiner.base.WARNING_CODES`) and
    #: PixelMagic's (`PIXELMAGIC_WARNING_CODES`). A warning is not a refusal —
    #: the asset is in the bundle and its pixels are real — but a game or a
    #: person reading the manifest a month later has no other way to learn that
    #: the refiner was unsure about this one. `RefineReport.error_code` is the
    #: separate, fatal case.
    warnings: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    #: How the piece sits on the grid (spec locked decision 8), copied from
    #: its `AssetRecord` by export. `width` and `height` above stay the
    #: refined image's real size; `target` is what was asked for, and the two
    #: differ only in a stand-in bundle (locked decision 9). `footprint` and
    #: `anchor` are props only, so every one is optional — and an M2 manifest
    #: without any of them still loads.
    pitch: float | None = None
    pitch_source: PitchSource | None = None
    target: tuple[int, int] | None = None
    footprint: tuple[int, int] | None = None
    anchor: tuple[int, int] | None = None
    role: TileRole | None = None
    position: tuple[int, int] | None = None
    #: A recipe tile's variant (1-8) and its record, copied from its
    #: `AssetRecord` by export. None on every other entry and on any older
    #: manifest.
    variant: int | None = None
    recipe: RecipeRecord | None = None
    #: Which segmenter cut this piece out, copied from its `AssetRecord`.
    #: Never dropped; None on a recipe tile and on the sheet road.
    segmenter: str | None = None
    #: Copied from this asset's `refine.json` (`RefineReport.cohesion`), never
    #: from its `AssetRecord`: this is a property of *this refine*, not of
    #: the asset's own provenance. Defaults to "did not run" so an M3
    #: manifest without it still loads.
    cohesion: CohesionRecord = Field(default_factory=CohesionRecord)


class Manifest(BaseModel):
    """The export bundle's index — the file a game reads."""

    project: str
    direction: str
    exported: AwareDatetime
    entries: list[ManifestEntry] = Field(default_factory=list)
    #: The direction's cell size, and the kept scene's pitch keyed `"scene"`.
    #: An old manifest may also carry `"props"` (the props sheet's pitch, no
    #: longer written). `None` and empty on a manifest written before M3,
    #: which must still load.
    cell_size: int | None = None
    pitches: dict[str, Pitch] = Field(default_factory=dict)
