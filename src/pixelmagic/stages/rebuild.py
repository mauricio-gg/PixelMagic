"""Rebuild one cut-back piece on its own pixel grid.

A chunky redraw is a picture painted on a regular grid of big "art pixels"
(cells). This module reads that grid off the piece and saves the piece at
native size, one pixel per cell, never resized. It imports no PixelRefiner and
runs no subprocess.

The grid finder reads each axis on its own (the rebuild then uses one square size
for both, see ``square_axes``):

1. Between every two neighbouring columns (rows) measure how big the colour
   step is, log-compressed so many weak steps count against a few strong
   silhouette edges.
2. A picture painted on a grid of pitch ``p`` and phase ``f`` piles those steps
   up at ``f + k * p``. For every candidate pitch, lay the steps round a circle
   of one turn per pitch and add them as vectors. Steps on one grid line all
   point the same way and add up; steps anywhere cancel.
3. The score is the length of that sum divided by the total step strength:
   1 means every step is on a grid line, about 0 means they are anywhere. The
   angle of the sum is the phase.
4. Half, a third (and so on) of the true pitch scores almost as well as the
   truth, so the best pitch moves to every whole multiple of itself that still
   fits in the range (2x, 3x, 4x ...) when that scores within
   ``MULTIPLES_SHARE`` of it. A true grid scores near zero at double, so
   doubling alone can never win.
5. Near the top of the range the scan picks up noise from the picture's own big
   shapes, which can score higher than a real smaller grid. A pitch at or above
   ``TOP_BAND_START`` that scores under ``WEAK_TOP_BAR`` is set aside and the
   axis is picked again from the scan below the band.

6. Pixels are square, so the two readings are not used as they are. The smaller of
   the two gives ONE size for both axes, and each axis's phase and score are taken
   again at that size. Two different sizes would stretch the piece and blur its
   colours. The smaller one, because a reading can land on a multiple of the true
   size (the higher-scoring one was about double on small, thin props, which came out
   at half size); a size that is too small keeps the detail, one that is too big
   throws it away.

A grid is found when both axes score at least ``FOUND_BAR``. The finder's
numbers were set on opaque crops on a flat dark board with a 6 px margin, so a
tight transparent piece is padded and laid on one flat dark colour first.
Everything here is deterministic: fixed order, nothing random.

Colour. Once the grid is known, each object cell gets a colour in three moves:

1. Cells are sorted into colour groups, one per material family (wood, quilt,
   metal). The darkest cells are the outline group. Families that are close
   merge. A small family that is far from every main group is an accent (a
   flame, a brass tap) and survives as a group of its own.
2. A group owns each cell. The cell's colour is the median of its own original
   pixels that belong to the owner's material, so a cell on the edge between
   wood and metal is not muddied by the other material's pixels. That is the
   idea of pixel-art-fixer (Retro-Diffusion, github.com/Retro-Diffusion/pixel-art-fixer,
   MIT licence, Astropulse LLC): decide the owner first, then average only that
   material's pixels. No code from it is used.
3. Each non-accent group is cut into a short run of shades by lightness; every
   cell takes its run's mean colour. Accents keep their own colour. A piece
   never has more than ``MAX_COLOURS`` colours: shades per group are lowered
   until it fits.
4. The silhouette is closed: every object cell touching the outside is forced
   into the outline group and takes one of that group's own dark colours.
5. A single stray cell in the middle of a surface is changed to the shade its
   neighbours agree on. The 64-colour cap is checked after both steps.

Distances are measured in OKLab, a colour space where equal numeric steps look
like equal steps to an eye, so "close" means close to the eye and not just
close in red, green and blue.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from pixelmagic.model import GridRebuild


class RebuiltPieceRefused(ValueError):
    """A refine setting or a candidate re-pick was asked of a rebuilt piece.

    A rebuilt piece is saved at native size, one pixel per art pixel, and is
    never sent to the refiner, so a refine setting or a candidate re-pick has
    nothing to apply to. A `ValueError`, so a face that catches that family
    still catches it. The message names the piece and the way forward, so a
    face can print it as it is.
    """

    def __init__(self, name: str) -> None:
        """Name the piece in the core's sentence."""
        super().__init__(
            f"{name} was rebuilt on its own pixel grid and is never sent "
            f"to the refiner, so a refine setting or a candidate re-pick "
            f"cannot apply to it; redraw it by name instead"
        )
        self.name = name


#: Pixels of flat dark colour put round the piece, and how far the edge weight
#: reaches beyond the object.
MARGIN = 6
#: The one flat dark colour the piece is laid on before the finder looks.
FILL_COLOUR = (24, 24, 24)
#: A grid is found when both axes score at least this.
FOUND_BAR = 0.030
#: Candidate pitches, in reply pixels, and the step between them.
PITCH_MIN = 2.0
PITCH_MAX = 14.0
PITCH_STEP = 0.0125
#: A whole multiple of the best pitch wins when it scores this share of it.
MULTIPLES_SHARE = 0.90
#: A chosen pitch from here up is in the top band of the range, where noise from the
#: picture's big shapes is strongest (#253).
TOP_BAND_START = PITCH_MAX - 1.0
#: A top-band pitch scoring under this is set aside as noise. Set on #253: the
#: highest score noise reached in the top band on 40 Codex frames was 0.077, while
#: real grids at 13 to 14 scored 0.13 or more.
WEAK_TOP_BAR = 0.10
#: The found pitch and phase are rounded to this.
PITCH_ROUNDING = 0.025
#: A cell is the object's when at least this share of its whole area is opaque
#: (the part outside the picture counts as transparent).
CELL_IN_MASK = 0.5
#: Alpha above this counts as opaque when reading the object's mask.
OPAQUE_ALPHA = 127
#: Colour families the cells are first split into, before close ones merge.
K_START = 6
#: Two families closer than this (in the colour feature space) are one.
MERGE_DIST = 0.07
#: A family with at least this share of the non-outline cells is a main group.
MAIN_SHARE = 0.04
#: A smaller family is an accent only with at least this many cells ...
ACCENT_MIN_CELLS = 3
#: ... and at least this far from every main group.
ACCENT_DIST = 0.10
#: At most this many accents per piece.
MAX_ACCENTS = 4
#: How much lightness counts next to the two colour axes when finding families.
L_WEIGHT = 0.3
#: The outline group is the cells darker than this percentile of the piece's
#: lightness, but never lighter than ``OUTLINE_L_CAP`` (OKLab L).
OUTLINE_P = 15
OUTLINE_L_CAP = 0.38
#: Most shades one group is reduced to.
SHADES_PER_GROUP = 6
#: A piece never has more colours than this.
MAX_COLOURS = 64
#: A cell is a stray when its own colour (before shades were cut) is further
#: than this, in OKLab, from every shade of its group. First value: the scale
#: of ``MERGE_DIST``, which is the same sort of "these are one material" line
#: (that one is on the colour feature space, where lightness is scaled down, so
#: this is the stricter line for lightness). 134.9 settles it on real pieces.
STRAY_DISTANCE = 0.07
#: A colour is a real shade of a group only when at least this many of its cells
#: hold it. A one-cell run is the stray itself, not a shade to measure against.
MIN_SHADE_CELLS = 2
#: A stray is changed when at least this many of its four neighbours, all in its
#: group, agree on one shade.
STRAY_AGREE = 3
#: The clean-up runs at most this many passes.
STRAY_PASSES = 2


@dataclass(frozen=True)
class RebuiltPiece:
    """A piece rebuilt at native size, with the grid it was read from."""

    image: Image.Image
    grid: GridRebuild


@dataclass(frozen=True)
class GridRefused:
    """No grid found: the score reached on each axis, both below the bar or one of them."""

    confidence_x: float
    confidence_y: float


@dataclass(frozen=True)
class PieceTooThin:
    """The grid was found (both scores at the bar) but no cell was solid enough to keep."""

    confidence_x: float
    confidence_y: float


@dataclass(frozen=True)
class _Axis:
    """One axis's finding: pitch, phase and score.

    The finder's phase is in the padded frame; ``square_axes`` returns the picture's own.
    """

    pitch: float
    phase: float
    score: float


def _grow(mask: np.ndarray, steps: int) -> np.ndarray:
    """The mask grown by ``steps`` pixels in the four directions (a diamond)."""
    return ndimage.binary_dilation(
        mask, structure=ndimage.generate_binary_structure(2, 1), iterations=steps
    )


def _edge_profiles(rgb: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colour-step strength at every column boundary (x) and row boundary (y).

    A step is the log of one plus the summed colour difference between two
    neighbours, counted only where both are inside the weight.
    """
    f = rgb.astype(float)
    dx = np.log1p(np.abs(np.diff(f, axis=1)).sum(axis=2)) * np.minimum(
        weight[:, 1:], weight[:, :-1]
    )
    dy = np.log1p(np.abs(np.diff(f, axis=0)).sum(axis=2)) * np.minimum(weight[1:], weight[:-1])
    return dx.sum(axis=0), dy.sum(axis=1)


def _turn_sum(e: np.ndarray, pitch: float) -> complex:
    """The steps added as vectors round one turn per ``pitch``, over their total.

    Boundary ``i`` sits at coordinate ``i + 1``, between pixels ``i`` and ``i + 1``.
    """
    t = np.arange(len(e)) + 1.0
    return complex((e * np.exp(-2j * np.pi * t / pitch)).sum() / e.sum())


def _scan(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every candidate pitch and its score."""
    pitches = np.arange(PITCH_MIN, PITCH_MAX + 1e-9, PITCH_STEP)
    scores = np.array([abs(_turn_sum(e, p)) for p in pitches])
    return pitches, scores


def _prefer_multiples(pitches: np.ndarray, scores: np.ndarray, top: float = PITCH_MAX) -> int:
    """Index of the best pitch, moved up to a whole multiple of it while that scores nearly as well.

    Every multiple ``k * pitch`` that is at most ``top`` is tried, smallest k first. The
    comparison allows a hair of rounding so that 2.8 x 5 counts as 14.
    """
    best = int(scores.argmax())
    moved = True
    while moved:
        moved = False
        for k in range(2, int(top / pitches[best] + 1e-9) + 1):
            target = pitches[best] * k
            lo = max(0, int((target * 0.97 - PITCH_MIN) / PITCH_STEP))
            hi = min(len(scores), int((target * 1.03 - PITCH_MIN) / PITCH_STEP) + 1)
            j = lo + int(scores[lo:hi].argmax())
            if scores[j] >= MULTIPLES_SHARE * scores[best]:
                best = j
                moved = True
                break
    return best


def _is_fraction_of(pitch: float, whole: float) -> bool:
    """True when ``pitch`` is within 3% of ``whole / k`` for some whole k of 2 or more."""
    return any(
        abs(pitch - whole / k) <= 0.03 * whole / k for k in range(2, int(whole / PITCH_MIN) + 1)
    )


def _peak(scores: np.ndarray, i: int) -> float:
    """Offset, in scan steps, of the parabola through the peak and its two neighbours."""
    if 0 < i < len(scores) - 1:
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        curve = a - 2 * b + c
        if curve != 0:
            return 0.5 * (a - c) / curve
    return 0.0


def _find_axis(e: np.ndarray) -> _Axis:
    """Pitch, phase and score of one axis from its edge profile."""
    if e.sum() <= 0:
        return _Axis(PITCH_MIN, 0.0, 0.0)
    pitches, scores = _scan(e)
    i = _prefer_multiples(pitches, scores)
    if pitches[i] >= TOP_BAND_START and scores[i] < WEAK_TOP_BAR:
        # A weak reading in the top band is noise: look again below the band. (A real but
        # weak grid at 13 to 14 is set aside too: accepted, real ones scored 0.13 or more, #253.)
        below = int(np.searchsorted(pitches, TOP_BAND_START))
        again = _prefer_multiples(pitches[:below], scores[:below], TOP_BAND_START)
        # With nothing real below, the re-pick lands on a whole fraction of the noise;
        # keep the top reading then, so the clean-up still warns about it.
        if not _is_fraction_of(pitches[again], pitches[i]):
            i = again
    pitch = round((pitches[i] + _peak(scores, i) * PITCH_STEP) / PITCH_ROUNDING) * PITCH_ROUNDING
    z = _turn_sum(e, pitch)
    phase = (-np.angle(z) / (2 * math.pi) * pitch) % pitch
    phase = (round(phase / PITCH_ROUNDING) * PITCH_ROUNDING) % pitch
    return _Axis(pitch, phase, float(scores[i]))


def _cell_bounds(size: int, pitch: float, phase: float) -> list[tuple[float, float]]:
    """Cell edges along one axis that overlap ``[0, size)``, in the piece's frame."""
    k = math.floor(-phase / pitch)
    bounds = []
    while phase + k * pitch < size:
        bounds.append((phase + k * pitch, phase + (k + 1) * pitch))
        k += 1
    return [(a, b) for a, b in bounds if b > 0]


def _span(a: float, b: float) -> tuple[int, int]:
    """The whole pixels a cell covers, at least one wide, not yet clipped to the picture."""
    lo = round(a)
    return lo, max(lo + 1, round(b))


def _sample_cells(
    pixels: np.ndarray, grid_x: _Axis, grid_y: _Axis
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """One RGBA pixel per cell on the found grid, transparent where not the object's.

    Also returns, for every object cell ``(row, column)``, its opaque original
    pixels (n x 3), which the colouring step needs to pick the owner's pixels.

    A cell is the object's when at least ``CELL_IN_MASK`` of the cell's whole
    area is opaque; any part of the cell outside the picture counts as
    transparent, so a sliver of a cell at a tight crop's edge is not an object
    cell. Its colour is the median of its opaque original pixels. Every object
    cell is kept: the cut-back has already decided what the object is.
    """
    h, w = pixels.shape[:2]
    xs = _cell_bounds(w, grid_x.pitch, grid_x.phase)
    ys = _cell_bounds(h, grid_y.pitch, grid_y.phase)
    cells = np.zeros((len(ys), len(xs), 4), np.uint8)
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for j, (ya, yb) in enumerate(ys):
        y0, y1 = _span(ya, yb)
        for i, (xa, xb) in enumerate(xs):
            x0, x1 = _span(xa, xb)
            block = pixels[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)].reshape(-1, 4)
            opaque = block[:, 3] > OPAQUE_ALPHA
            if opaque.sum() >= CELL_IN_MASK * (y1 - y0) * (x1 - x0):
                cells[j, i, :3] = np.median(block[opaque, :3], axis=0).astype(np.uint8)
                cells[j, i, 3] = 255
                blocks[(j, i)] = block[opaque, :3]
    return cells, blocks


def _oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0 to 255, shape (..., 3)) to OKLab (L, a, b).

    Undo the sRGB curve to get linear light, mix it into cone responses, take
    the cube root (the eye's compression), and mix again into lightness and two
    colour axes (a: green to red, b: blue to yellow).
    """
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    to_cones = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    to_lab = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return np.cbrt(c @ to_cones.T) @ to_lab.T


def _feature(lab: np.ndarray) -> np.ndarray:
    """The space families are found in: the two colour axes, lightness scaled down.

    Lightness is weak here so one material in light and in shadow stays one family.
    """
    return np.stack([lab[..., 1], lab[..., 2], L_WEIGHT * lab[..., 0]], axis=-1)


def _kmeans(points: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means from the given centres; returns the centres and each point's index."""
    for _ in range(100):
        assign = ((points[:, None] - centres[None]) ** 2).sum(2).argmin(1)
        new = np.array(
            [
                points[assign == c].mean(0) if (assign == c).any() else centres[c]
                for c in range(len(centres))
            ]
        )
        if np.allclose(new, centres):
            break
        centres = new
    return centres, ((points[:, None] - centres[None]) ** 2).sum(2).argmin(1)


def _families(feat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Split the non-outline cells into families; returns each cell's group and the accents.

    Seeds are picked farthest-point from the cell nearest the median, so the
    same cells give the same families. Families closer than ``MERGE_DIST`` merge
    (the lower index absorbs). A family is main with ``MAIN_SHARE`` of the
    cells; a smaller one is an accent when it has ``ACCENT_MIN_CELLS`` cells and
    is ``ACCENT_DIST`` from every main group, at most ``MAX_ACCENTS``, biggest
    first. Cells of dropped families go to the nearest kept one. The result is
    an index into the kept families (mains first, then accents) and the
    accents' indices.
    """
    first = int(((feat - np.median(feat, 0)) ** 2).sum(1).argmin())
    seeds = [first]
    for _ in range(K_START - 1):
        seeds.append(int(np.min([((feat - feat[s]) ** 2).sum(1) for s in seeds], axis=0).argmax()))
    centres, assign = _kmeans(feat, feat[seeds].copy())
    alive = list(range(len(centres)))
    merged = True
    while merged:
        merged = False
        for i in alive:
            for j in alive:
                if j > i and np.linalg.norm(centres[i] - centres[j]) < MERGE_DIST:
                    n_i, n_j = (assign == i).sum(), (assign == j).sum()
                    centres[i] = (centres[i] * n_i + centres[j] * n_j) / max(1, n_i + n_j)
                    assign[assign == j] = i
                    alive.remove(j)
                    merged = True
                    break
            if merged:
                break
    counts = {c: int((assign == c).sum()) for c in alive}
    main = [c for c in alive if counts[c] >= MAIN_SHARE * len(feat)]
    if not main:
        main = [max(alive, key=lambda c: (counts[c], -c))]
    accents: list[int] = []
    for c in sorted((c for c in alive if c not in main), key=lambda c: (-counts[c], c)):
        far = min(np.linalg.norm(centres[c] - centres[m]) for m in main)
        if counts[c] >= ACCENT_MIN_CELLS and far >= ACCENT_DIST and len(accents) < MAX_ACCENTS:
            accents.append(c)
    keep = main + accents
    owner = ((feat[:, None] - centres[keep][None]) ** 2).sum(2).argmin(1)
    return owner, list(range(len(main), len(keep)))


def _runs(lightness: np.ndarray, k: int) -> np.ndarray:
    """Cut values into at most ``k`` runs by lightness; returns each value's run index.

    A one-dimensional k-means seeded from quantiles of the sorted values, so
    the same values always give the same runs.
    """
    k = max(1, min(k, len(np.unique(np.round(lightness, 3)))))
    ordered = np.sort(lightness)
    centres = np.array([ordered[int((i + 0.5) * len(ordered) / k)] for i in range(k)])
    for _ in range(100):
        a = np.abs(lightness[:, None] - centres[None]).argmin(1)
        new = np.array(
            [lightness[a == c].mean() if (a == c).any() else centres[c] for c in range(k)]
        )
        if np.allclose(new, centres):
            break
        centres = new
    return np.abs(lightness[:, None] - centres[None]).argmin(1)


def _reduce(
    colours: np.ndarray,
    group: np.ndarray,
    accent_groups: set[int],
    shades: int,
    accent_shades: int | None = None,
) -> np.ndarray:
    """Every non-accent group cut into at most ``shades`` runs; each cell takes its run's mean.

    Accent cells are returned as they are unless ``accent_shades`` is given, in
    which case each accent group is cut into at most that many runs too (the
    last resort when the cap cannot otherwise hold). Cells with a negative group
    (the ring cells) take their colour from the outline's run.
    """
    out = colours.astype(float)
    lightness = _oklab(colours)[:, 0]
    for g in sorted(set(group.tolist()) - {-1}):
        if g in accent_groups and accent_shades is None:
            continue
        members = np.flatnonzero(group == g)
        run = _runs(lightness[members], accent_shades if g in accent_groups else shades)
        for r in sorted(set(run.tolist())):
            sel = members[run == r]
            out[sel] = colours[sel].mean(0)
    return np.round(out).astype(np.uint8)


def _colour_cells(cells: np.ndarray, blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Give every object cell its group's colour: see the module docstring.

    ``cells`` holds each cell's plain median colour (used to sort cells into
    groups); ``blocks`` holds each cell's own opaque pixels. Returns RGBA cells
    of the same shape with the colours replaced.
    """
    coords = sorted(blocks)
    if not coords:
        return cells
    plain = np.array([cells[j, i, :3] for j, i in coords])
    lab = _oklab(plain)
    feat = _feature(lab)
    light = lab[:, 0]

    # Group 0 is the outline: the darkest cells.
    outline_l = min(np.percentile(light, OUTLINE_P), OUTLINE_L_CAP)
    dark = light <= outline_l
    group = np.zeros(len(coords), int)
    accent_groups: set[int] = set()
    if (~dark).any():
        owner, accents = _families(feat[~dark])
        group[~dark] = 1 + owner
        accent_groups = {1 + a for a in accents}
    centres = {g: feat[group == g].mean(0) for g in sorted(set(group.tolist())) if g}

    # The cell's colour: median of its own pixels that belong to the owner's material.
    own = np.empty((len(coords), 3))
    for n, key in enumerate(coords):
        px = blocks[key]
        px_lab = _oklab(px)
        if group[n] == 0:
            belongs = px_lab[:, 0] <= outline_l
        else:
            ids = sorted(centres)
            dist = np.stack([((_feature(px_lab) - centres[g]) ** 2).sum(1) for g in ids], axis=1)
            belongs = (px_lab[:, 0] > outline_l) & (np.array(ids)[dist.argmin(1)] == group[n])
        own[n] = np.median(px[belongs] if belongs.any() else px, axis=0)
    own = own.astype(np.uint8)

    # Cells touching the outside are forced into the outline group. They are left
    # out of the shades below, so the ring adds no colour of its own.
    ring = _ring(cells, coords)
    forced = ring & (group != 0)
    before = group.copy()
    group[forced] = -1
    protected = {0} | accent_groups

    # Fewer shades per group until the finished piece fits under the cap.
    # Accents keep one colour per cell while that fits; only when every other
    # group is already down to one shade and the piece is still over the cap are
    # the accent groups cut into shades too, so the cap always holds.
    attempts = [(s, None) for s in range(SHADES_PER_GROUP, 0, -1)]
    attempts += [(1, s) for s in range(SHADES_PER_GROUP, 0, -1)]
    for shades, accent_shades in attempts:
        final = _reduce(own, group, accent_groups, shades, accent_shades)
        _force_ring(final, own, before, group, forced)
        group[forced] = 0
        _remove_strays(final, own, group, coords, protected)
        if len(np.unique(final, axis=0)) <= MAX_COLOURS:
            break
        group[forced] = -1
    out = cells.copy()
    for n, (j, i) in enumerate(coords):
        out[j, i, :3] = final[n]
    return out


def _ring(cells: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    """For each object cell in ``coords``, whether a 4-neighbour is outside the object.

    The edge of the array counts as outside.
    """
    solid = np.pad(cells[:, :, 3] > 0, 1)
    outside = ~solid
    touching = outside[:-2, 1:-1] | outside[2:, 1:-1] | outside[1:-1, :-2] | outside[1:-1, 2:]
    return np.array([touching[j, i] for j, i in coords])


def _force_ring(
    final: np.ndarray, own: np.ndarray, before: np.ndarray, group: np.ndarray, forced: np.ndarray
) -> None:
    """Give each forced ring cell the outline colour closest to the outline's old median.

    The median is of the cells the outline group owned before the ring was
    forced (``before``). The colour is one of the outline's own reduced colours
    (``group`` is 0 for those cells here). If the group owned none, it is the
    darkest colour already in the piece; if every cell is forced, the darkest
    of the forced cells' own colours. Changes ``final`` in place.
    """
    if not forced.any():
        return
    owned = before == 0
    run = np.unique(final[group == 0], axis=0)
    if owned.any() and len(run):
        target = _oklab(np.median(own[owned], axis=0))
        lab = _oklab(run)
        final[forced] = run[((lab - target) ** 2).sum(1).argmin()]
        return
    pool = final[group > 0] if (group > 0).any() else own[forced]
    final[forced] = pool[_oklab(pool)[:, 0].argmin()]


def _remove_strays(
    final: np.ndarray,
    own: np.ndarray,
    group: np.ndarray,
    coords: list[tuple[int, int]],
    protected: set[int],
) -> None:
    """Change single stray cells to the shade their neighbours agree on, in place.

    Measured on each cell's own colour from before the shades were cut. After
    the cut every cell already is one of its group's shades, so "far from the
    nearest shade" could never be true there. A stray is a cell whose own colour
    is over ``STRAY_DISTANCE`` from every shade of its group while at least
    ``STRAY_AGREE`` of its four neighbours, all in its group, hold one shade.
    A shade is a colour held by ``MIN_SHADE_CELLS`` cells, so a stray that was
    given a run of its own does not count as a shade of its own. Outline, ring
    and accent cells are never touched. Each pass decides from the picture as
    it stood at the start of the pass, so cell order cannot matter.
    """
    where = {key: n for n, key in enumerate(coords)}
    own_lab = _oklab(own)
    for _ in range(STRAY_PASSES):
        counts: dict[tuple[int, tuple[int, ...]], int] = {}
        for n in range(len(coords)):
            key = (int(group[n]), tuple(int(v) for v in final[n]))
            counts[key] = counts.get(key, 0) + 1
        changes = {}
        for n, (j, i) in enumerate(coords):
            g = int(group[n])
            if g in protected:
                continue
            shades = [
                np.array(c) for (gg, c), k in counts.items() if gg == g and k >= MIN_SHADE_CELLS
            ]
            if not shades:
                continue
            gap = np.linalg.norm(_oklab(np.array(shades)) - own_lab[n], axis=1).min()
            if gap <= STRAY_DISTANCE:
                continue
            seen: dict[tuple[int, ...], int] = {}
            for step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                m = where.get((j + step[0], i + step[1]))
                if m is not None and group[m] == g:
                    colour = tuple(int(v) for v in final[m])
                    seen[colour] = seen.get(colour, 0) + 1
            agreed = sorted(c for c, k in seen.items() if k >= STRAY_AGREE)
            if agreed and agreed[0] != tuple(int(v) for v in final[n]):
                changes[n] = agreed[0]
        if not changes:
            break
        for n, colour in changes.items():
            final[n] = colour


def padded_profiles(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge profiles of an RGBA picture, padded by ``MARGIN`` on the flat dark colour.

    The picture is laid on ``FILL_COLOUR`` first, the input ``FOUND_BAR`` was set
    on. Phases found from these profiles are in the padded frame: subtract
    ``MARGIN`` (modulo the pitch) to get the picture's own. Shared by
    ``rebuild_piece`` and the clean-up job so both read a grid the same way.
    """
    h, w = pixels.shape[:2]
    padded = np.empty((h + 2 * MARGIN, w + 2 * MARGIN, 3), float)
    padded[:] = FILL_COLOUR
    alpha = pixels[:, :, 3:4] / 255.0
    padded[MARGIN : MARGIN + h, MARGIN : MARGIN + w] = pixels[:, :, :3] * alpha + np.array(
        FILL_COLOUR
    ) * (1 - alpha)
    mask = np.zeros(padded.shape[:2], bool)
    mask[MARGIN : MARGIN + h, MARGIN : MARGIN + w] = pixels[:, :, 3] > OPAQUE_ALPHA
    return _edge_profiles(padded, _grow(mask, MARGIN).astype(float))


def _phase_at(e: np.ndarray, pitch: float) -> float:
    """Phase (padded frame) of the steps for a given pitch, rounded as ``_find_axis`` rounds it."""
    z = _turn_sum(e, pitch)
    phase = (-np.angle(z) / (2 * math.pi) * pitch) % pitch
    return (round(phase / PITCH_ROUNDING) * PITCH_ROUNDING) % pitch


def square_axes(
    ex: np.ndarray,
    ey: np.ndarray,
    found_x: _Axis,
    found_y: _Axis,
    size: float | None = None,
) -> tuple[_Axis, _Axis]:
    """Both axes at ONE square pixel size, in the picture's own frame.

    Pixels are square, so one size serves both axes: ``size`` when given (only
    clean-up passes one), else the smaller of the two found pitches, because a
    reading can land on a multiple of the true size and a size too small keeps
    the detail while one too big loses it (measured on 98 Codex pieces: the
    higher-scoring reading made a lamp, a globe and candles half size). Each axis's
    phase and score are then taken again at that size from its padded edge profile
    (``ex``, ``ey``), and the phase is moved back by ``MARGIN`` into the picture's own
    frame. Shared by ``rebuild_piece`` and the clean-up job so the two never disagree.
    """
    pitch = float(size) if size is not None else min(found_x.pitch, found_y.pitch)

    def at_pitch(e: np.ndarray) -> _Axis:
        return _Axis(pitch, (_phase_at(e, pitch) - MARGIN) % pitch, abs(_turn_sum(e, pitch)))

    return at_pitch(ex), at_pitch(ey)


def rebuild_piece(piece: Image.Image) -> RebuiltPiece | GridRefused | PieceTooThin:
    """Find the piece's own pixel grid and rebuild it at one pixel per cell.

    ``piece`` is a cut-back piece: RGBA, its own alpha the object's outline.
    Pixels are square: the cells are sampled at ONE size on both axes
    (``square_axes``: the smaller of the two readings), so a piece whose two
    readings differ keeps its proportions. The grid records that size as both
    pitches and, as ``confidence``, the lower of the two scores taken at it.
    Returns the native-size piece and the grid it came from, or a
    ``GridRefused`` carrying the score found on each axis when either found
    reading is below ``FOUND_BAR`` (the piece is not painted on a grid, so
    nothing is rebuilt), or a ``PieceTooThin`` with the same found scores when
    the grid was found but no cell of the object was solid enough to keep.
    """
    pixels = np.asarray(piece.convert("RGBA"), dtype=np.uint8)
    ex, ey = padded_profiles(pixels)
    axis_x, axis_y = _find_axis(ex), _find_axis(ey)
    if min(axis_x.score, axis_y.score) < FOUND_BAR:
        return GridRefused(axis_x.score, axis_y.score)

    own_x, own_y = square_axes(ex, ey, axis_x, axis_y)
    cells, blocks = _sample_cells(pixels, own_x, own_y)
    cells = _colour_cells(cells, blocks)

    # Crop to the object's cells.
    rows = np.flatnonzero(cells[:, :, 3].any(axis=1))
    cols = np.flatnonzero(cells[:, :, 3].any(axis=0))
    if len(rows) == 0:
        return PieceTooThin(axis_x.score, axis_y.score)
    cells = cells[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    native = (cells.shape[1], cells.shape[0])
    grid = GridRebuild(
        pitch_x=own_x.pitch,
        pitch_y=own_y.pitch,
        phase_x=own_x.phase,
        phase_y=own_y.phase,
        confidence=min(own_x.score, own_y.score),
        native_size=native,
    )
    return RebuiltPiece(Image.fromarray(cells, "RGBA"), grid)
