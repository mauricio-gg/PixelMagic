"""Cut a prop straight from a picture that is already pixel art.

The second road to a prop, next to suggest, pick, redraw and rebuild: no image service
and no PixelRefiner. `start_pixel_cut` reads the picture's pixel grid and opens a
decomposition that carries it; picks on it are then asked of the object finder on an
enlarged copy of the native picture (`stages.pick` routes them through `stages.pixelgrid`),
and `cut_pieces` turns the picks into props made of the picture's own pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from PIL import Image

from pixelmagic.model import (
    Cutout,
    CutoutOrigin,
    Decomposition,
    Direction,
    GenerationRecord,
    Pick,
    PickSet,
    PitchSource,
    PixelCut,
    PixelGrid,
    ReviewStatus,
    SheetKind,
)
from pixelmagic.naming import unique_name
from pixelmagic.stages import pixelgrid
from pixelmagic.stages import rebuild as rb
from pixelmagic.stages.decompose import (
    NotApproved,
    _all_cutout_names,
    layout_cutout,
    load_decomposition,
    save_decomposition,
)
from pixelmagic.stages.pick import load_picks, save_picks
from pixelmagic.stages.redraw import live_pieces
from pixelmagic.store import Store, claim_id

#: The native picture, in the decomposition's folder (the finder copy is `pixelgrid.FINDER_PNG`).
NATIVE_PNG = "native.png"

#: How the sentence for an unsure found size begins; `unsure_note` finds it by this.
UNSURE_PHRASE = "Not sure how big each pixel is in this picture."

#: Said when the size was found, not typed, and the finder was not sure of it.
UNSURE_SENTENCE = (
    UNSURE_PHRASE + " We used {size}. If the result looks "
    "wrong, start again and type the size; type 1 when the picture is already one pixel "
    "per art pixel."
)


def start_pixel_cut(
    store: Store,
    direction: Direction,
    source: GenerationRecord,
    *,
    size: float | None = None,
    now: datetime | None = None,
) -> Decomposition:
    """Open a decomposition that cuts props from the picture's own pixels.

    Refuses a picture nobody approved (`NotApproved`) and a typed ``size`` outside 1 to
    the picture's shorter side (`ValueError`), both before anything is written. Writes
    `native.png` and `finder.png` into the new decomposition folder and then
    `decomposition.json` with the picture's `pixel_grid`, so a run cut short is never
    listed. A found size the finder was unsure of adds one sentence to ``warnings``.
    """
    if source.review.status is not ReviewStatus.APPROVED:
        raise NotApproved(
            f"only an approved picture can be cut up; this one is {source.review.status}"
        )
    pixels = np.asarray(Image.open(source.image_path).convert("RGBA"), dtype=np.uint8)
    grid = pixelgrid.read_grid(pixels, size)

    moment = now or datetime.now(UTC)
    decomposition_id = claim_id(store.decompositions_dir(direction.name), now=moment)
    folder = store.decomposition_dir(direction.name, decomposition_id)
    native = pixelgrid.native_picture(pixels, grid)
    Image.fromarray(native, "RGBA").save(folder / NATIVE_PNG)
    Image.fromarray(pixelgrid.finder_copy(native), "RGBA").save(folder / pixelgrid.FINDER_PNG)

    warnings = []
    if grid.size_source == "found" and grid.unsure:
        whole = grid.pixel_size == int(grid.pixel_size)
        shown = f"{grid.pixel_size:.0f}" if whole else f"{grid.pixel_size:.1f}"
        warnings.append(UNSURE_SENTENCE.format(size=shown))
    decomposition = Decomposition(
        id=decomposition_id,
        created=moment,
        source_generation=source.request.id,
        warnings=warnings,
        pixel_grid=grid,
    )
    return save_decomposition(store, direction.name, decomposition)


def unsure_note(decomposition: Decomposition) -> str | None:
    """The run's own sentence about an unsure found size, as `start_pixel_cut` stored it.

    None when the run has none (a typed size, or a found one the finder was sure of).
    """
    return next((w for w in decomposition.warnings if w.startswith(UNSURE_PHRASE)), None)


def _native_mask(
    mask: np.ndarray, picture_box: tuple[int, int, int, int], grid: PixelGrid
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A pick's outline on the picture, as a mask of whole cells on the native grid.

    ``mask`` covers ``picture_box`` on the picture. A cell is in when at least
    ``rb.CELL_IN_MASK`` of its picture span is inside the outline. Returns the cell mask
    cropped to the cells that are in, and their box ``(x0, y0, x1, y1)`` on the native
    picture. Raises ``ValueError`` when no cell is in, which can only happen for a mask
    that did not come from ``pixelgrid.mask_to_picture`` (that already refuses an empty one).
    """
    xs, ys = pixelgrid.cell_spans(grid)
    left, top, right, bottom = picture_box
    width, height = grid.picture_size
    full = np.zeros((height, width), bool)
    full[top:bottom, left:right] = mask
    inside = np.zeros((len(ys), len(xs)), bool)
    for j, (y0, y1) in enumerate(ys):
        for i, (x0, x1) in enumerate(xs):
            inside[j, i] = full[y0:y1, x0:x1].mean() >= rb.CELL_IN_MASK
    if not inside.any():
        raise ValueError("no whole pixel of the picture is inside that outline")
    rows, cols = np.nonzero(inside)
    box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    return inside[box[1] : box[3], box[0] : box[2]], box


def _pitch_source(grid: PixelGrid) -> PitchSource:
    """Where the pixel size came from: typed, found and sure, or found and unsure."""
    if grid.size_source == "typed":
        return PitchSource.SET
    return PitchSource.UNSURE if grid.unsure else PitchSource.MEASURED


def piece_names(
    store: Store,
    direction: Direction,
    decomposition: Decomposition,
    chosen: Sequence[Pick],
    everything: PickSet | None = None,
) -> dict[int, str]:
    """The name `cut_pieces` gives each of `chosen`, by pick number, in the order given.

    A pick that already has a piece keeps that piece's name; any other is named by its own
    name, else `prop-<number>`, made unique against every asset and cutout of the direction
    and the names of the run's other picks. The page calls this to label a row with the name
    the cut will give, so the label and the cut cannot disagree.
    """
    if everything is None:
        everything = load_picks(store, direction.name, decomposition.id)
    taken = set(store.list_assets(direction.name)) | _all_cutout_names(store, direction.name)
    reserved = {pick.name for pick in everything.picks if pick.name}
    names: dict[int, str] = {}
    for pick in chosen:
        existing = live_pieces(decomposition, pick)
        if existing:
            name = existing[0].name
        else:
            wanted = pick.name or f"prop-{pick.number}"
            name = unique_name(wanted, taken | (reserved - {pick.name}))
        names[pick.number] = name
        taken.add(name)
        reserved.add(name)
    return names


def cut_pieces(
    store: Store,
    direction: Direction,
    decomposition_id: str,
    *,
    only: Sequence[int] | None = None,
) -> list[Cutout]:
    """Cut each pick (or each listed pick number) into a prop from the picture's own pixels.

    The piece is `native.png`'s own RGBA where the pick's outline holds a cell
    (`_native_mask`) and fully transparent elsewhere, cropped to the outline's box on the
    native grid: never resampled, redrawn or recoloured, and nothing is removed for other
    picks. It is saved as `cutouts/<name>/source.png` with `cutout.json`, with its pitch
    the grid's pixel size, its target its native size, `pixel_cut` filled and its
    footprint and anchor from `decompose.layout_cutout`. A pick that already has a piece
    here keeps its name and the piece is replaced; any other pick is named by its own name,
    else `prop-<number>`, made unique against every asset and cutout of the direction,
    and the settled name is saved on the pick.

    Raises ``ValueError``, before anything is written, for a decomposition that is not a
    pixel cut, no picks, an unknown pick number, or a pick with no whole cell inside it.
    Returns the pieces cut, in pick order.
    """
    decomposition = load_decomposition(store, direction.name, decomposition_id)
    grid = decomposition.pixel_grid
    if grid is None:
        raise ValueError("this run was not started as a pixel cut, so there is nothing to cut")
    everything = load_picks(store, direction.name, decomposition_id)
    if not everything.picks:
        raise ValueError("nothing is picked yet; pick the objects to cut first")
    if only is None:
        chosen = list(everything.picks)
    else:
        known = {pick.number: pick for pick in everything.picks}
        missing = [number for number in only if number not in known]
        if missing:
            raise ValueError(
                f"no pick numbered {', '.join(map(str, missing))}; "
                f"picks are {', '.join(map(str, known))}"
            )
        chosen = [known[number] for number in dict.fromkeys(only)]

    if decomposition.segmenter is None:
        raise ValueError("no pick on this run recorded which object finder made it")
    folder = store.decomposition_dir(direction.name, decomposition_id)
    native = np.asarray(Image.open(folder / NATIVE_PNG).convert("RGBA"), dtype=np.uint8)
    # Cut every piece in memory first, so a pick with nothing inside it leaves nothing behind.
    cut: list[tuple[Pick, np.ndarray, tuple[int, int, int, int]]] = []
    for pick in chosen:
        with Image.open(store.picks_dir(direction.name, decomposition_id) / pick.mask) as image:
            outline = np.asarray(image) > 0
        try:
            cells, box = _native_mask(outline, pick.picture_box, grid)
        except ValueError as error:
            raise ValueError(f"pick {pick.number}: {error}") from error
        piece = native[box[1] : box[3], box[0] : box[2]].copy()
        piece[~cells] = 0
        cut.append((pick, piece, box))

    names = piece_names(store, direction, decomposition, [pick for pick, _, _ in cut], everything)
    pieces = []
    renamed = False
    for pick, image, box in cut:
        name = names[pick.number]
        if pick.name != name:
            pick.name = name
            renamed = True

        native_size = (image.shape[1], image.shape[0])
        cutout_folder = store.cutout_dir(direction.name, decomposition_id, name)
        cutout_folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, "RGBA").save(cutout_folder / "source.png", format="PNG")
        cutout = layout_cutout(
            Cutout(
                name=name, ai_name=name, sheet=decomposition.source_generation,
                kind=SheetKind.PROPS, bbox=box, pitch=grid.pixel_size,
                pitch_source=_pitch_source(grid), target=native_size,
                pixel_cut=PixelCut(
                    pixel_size=grid.pixel_size, size_source=grid.size_source,
                    unsure=grid.unsure, native_size=native_size,
                ),
                origin=CutoutOrigin(
                    pick=pick.number, picture_box=pick.picture_box,
                    scale=grid.pixel_size, segmenter=decomposition.segmenter,
                ),
            ),
            direction.cell_size,
        )
        store.write_model(cutout_folder / "cutout.json", cutout)
        index = next((i for i, c in enumerate(decomposition.cutouts) if c.name == name), None)
        if index is None:
            decomposition.cutouts.append(cutout)
        else:
            decomposition.cutouts[index] = cutout
        pieces.append(cutout)
    if renamed:
        save_picks(store, direction.name, decomposition_id, everything)
    save_decomposition(store, direction.name, decomposition)
    return pieces
