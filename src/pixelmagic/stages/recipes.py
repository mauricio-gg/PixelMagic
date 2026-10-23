"""Recipe tiles: a surface drawn from a few colours instead of cut from a picture.

It holds `read_shades` (the colours of a box on the picture), `make_tiles` (the surface's
tiles from those colours and the surface's name), `check_seams` (the net that
stops a recipe whose tiles do not join) and `sample_layout` (the order of the
sample floor in the `.aseprite`), plus the constants and the three exceptions
they share. Every kind is drawn as 8 blocks of 2 x 2 tiles.

It is pure image work. It imports numpy and the standard library and nothing from this package, so
it never makes a segmenter call, and `decompose` makes none for a recipe surface either.
A tile is a numpy array of shape (16, 16, 4), dtype uint8, RGBA, fully opaque.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

#: The recipes that exist. The one list: a refusal for a missing or unknown kind
#: prints it, so a new kind shows up in the sentence without editing it.
RECIPE_KINDS = ("plaster", "stone", "wood")

#: Every recipe tile is this many pixels square.
RECIPE_CELL_SIZE = 16
#: How many different units (tiles, or blocks) a recipe draws.
VARIANTS = 8
#: A colour box with fewer pixels than this is refused as too small (32 x 32).
MIN_BOX_PIXELS = 1024
#: A body shade (any shade that is not the kind's line shade) holding less than this
#: share of the box is not a shade.
MIN_SHADE_SHARE = 0.03
#: The same bar for a kind's line shade (wood's joints): lines are thin by nature, and
#: a real wooden floor's plank seams measured 2.92% of its outline (measured 2026-10-07).
MIN_LINE_SHADE_SHARE = 0.01
#: How far plaster's shades are pulled towards its base shade (the one holding the most
#: pixels) to keep it calm.
PLASTER_CALM = 0.5
#: Width in pixels of the edge band of a stone or plaster block that is the same in all 8.
STONE_BAND = 3
#: The sample floor in the `.aseprite` is this many tiles wide and tall (5 x 3 blocks).
SAMPLE_COLUMNS = 10
SAMPLE_ROWS = 6
#: A block is this many pixels square.
BLOCK_SIZE = 2 * RECIPE_CELL_SIZE
#: A block's four tiles, in order: north-west, north-east, south-west, south-east.
PLACES = ("nw", "ne", "sw", "se")
#: Tiles in one block (2 x 2).
BLOCK_TILES = 4


class RecipeRefused(ValueError):
    """A refusal a person can act on.

    Box too small or outside the picture, shapes covering nothing, too few
    colours, wrong cell size, name in use.
    """


class RecipeTileRefused(ValueError):
    """An edit that would break a surface's set of tiles.

    A refine setting or re-pick, or a rename.
    """


class RecipeSeamBroken(RuntimeError):
    """The seam check failed. A bug in a recipe, not something a person did.

    Attributes (all keyword arguments, all stored on the exception):
    - `surface`: the surface's name.
    - `kind`: the recipe kind.
    - `rule`: which rule failed: "block-join", "tile-join", or "duplicate" for two
      equal blocks.
    - `axis`: "beside" or "above"; None for "duplicate".
    - `worst`: the worst join's rate (0-1); None for "duplicate".
    - `busiest_inside`: the busiest inside line's rate (0-1); None for
      "duplicate".
    - `units`: the 1-based variant numbers involved, as a tuple: the worst pair
      (a, b) for "block-join", the one block (n,) for "tile-join",
      the two equal ones (a, b) for "duplicate".
    """

    def __init__(
        self,
        message: str,
        *,
        surface: str,
        kind: str,
        rule: str,
        axis: str | None = None,
        worst: float | None = None,
        busiest_inside: float | None = None,
        units: tuple[int, ...] = (),
    ) -> None:
        """Keep the message and the numbers the face and the tests read."""
        super().__init__(message)
        self.surface = surface
        self.kind = kind
        self.rule = rule
        self.axis = axis
        self.worst = worst
        self.busiest_inside = busiest_inside
        self.units = units


class Shade(NamedTuple):
    """One colour of a surface and how much of the box it covered.

    `rgb` is a whole-number (r, g, b) tuple; `share` is the fraction of the
    box's pixels that ended in this shade's cluster (0 to 1, four decimals).
    """

    rgb: tuple[int, int, int]
    share: float


@dataclass(frozen=True)
class _Kind:
    """What the module knows about one recipe kind: its definition, not a plug-in.

    `colours` is k for the k-means and `draw`
    makes the kind's tiles from its shades and the surface's name. `line_shade`
    is the index, in the dark-to-light shades, of the shade the recipe draws its
    lines with, which only needs MIN_LINE_SHADE_SHARE of the outline; None means
    the kind has no line shade. Wood's is 0 (the darkest, the plank joints) and
    stone's is 4 (the lightest, the grout). `line_name` is what a refusal calls
    that line. `calm` is how far the other shades are pulled towards the shade holding the most
    pixels, which stays where it is (plaster).
    """

    colours: int
    draw: Callable[[Sequence[Shade], str], list[np.ndarray]]
    line_shade: int | None = None
    line_name: str = ""
    calm: float = 0.0


#: How many steps the k-means may take before it stops.
_KMEANS_STEPS = 40
#: The lines inside a block that `check_seams` compares (k against k + 1): k = 0 and 30
#: touch its edges and 14 to 16 the tile join.
_BLOCK_INSIDE_LINES = (*range(1, 14), *range(17, 30))


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Luminance of each row of an (n, 3) array: 0.299 R + 0.587 G + 0.114 B."""
    return rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587 + rgb[:, 2] * 0.114


def _dark_to_light(rgb: np.ndarray) -> np.ndarray:
    """Indexes that sort (n, 3) colours by luminance, ties by R, G, B (stable)."""
    return np.lexsort((rgb[:, 2], rgb[:, 1], rgb[:, 0], _luminance(rgb)))


def _assign(pixels: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Nearest centre for each pixel by squared RGB distance; a tie goes to the lower."""
    gap = pixels[:, None, :] - centres[None, :, :]
    return np.argmin((gap * gap).sum(axis=2), axis=1)


def _kmeans(pixels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The pinned k-means of the spec's "Colours": centres and cluster counts.

    Starts centre i at sorted position floor(n (i + 0.5) / k), moves centres to
    their means (an empty cluster stays), stops after 40 steps or when nothing
    moves. Returns the centres rounded to whole numbers, dark to light, and
    how many pixels each holds, in the same order (the final membership).
    """
    n = len(pixels)
    ordered = pixels[_dark_to_light(pixels)]
    centres = ordered[[n * (2 * i + 1) // (2 * k) for i in range(k)]].astype(float)
    for _ in range(_KMEANS_STEPS):
        owner = _assign(pixels, centres)
        moved = centres.copy()
        for i in range(k):
            members = pixels[owner == i]
            if len(members):
                moved[i] = members.mean(axis=0)
        done = np.array_equal(moved, centres)
        centres = moved
        if done:
            break
    counts = np.bincount(_assign(pixels, centres), minlength=k)
    whole = np.floor(centres + 0.5).astype(np.int64)
    order = _dark_to_light(whole)
    return whole[order], counts[order]


def read_shades(kind: str, picture: np.ndarray, inside: np.ndarray, *, surface: str) -> list[Shade]:
    """Read a surface's colours from the pixels inside its colour box.

    `picture` is the scene as an (h, w, 3 or 4) uint8 array and `inside` a
    boolean (h, w) array on the same pixels, True on the box; nothing outside
    `inside` is read. Returns the kind's shades dark to light, each with its
    share of the box's pixels. Plaster's other colours are pulled towards its biggest
    one by PLASTER_CALM, its shares are not changed. Raises RecipeRefused for an unknown kind, a box
    under MIN_BOX_PIXELS, fewer different colours than the kind needs, or
    a line shade under MIN_LINE_SHADE_SHARE, or any other shade under
    MIN_SHADE_SHARE (an empty cluster counts). `surface` is only there for the
    sentences.
    """
    definition = _kind(kind, surface)
    k = definition.colours
    pixels = picture[inside][:, :3].astype(np.int64)
    n = len(pixels)
    advice = "Draw the box on a plain patch of the surface."
    if n < MIN_BOX_PIXELS:
        raise RecipeRefused(
            f"The box of {surface!r} holds {n} pixels and a recipe needs at "
            f"least {MIN_BOX_PIXELS} (for example 32 x 32). {advice}"
        )
    different = len(np.unique(pixels, axis=0))
    if different < k:
        raise RecipeRefused(
            f"The box of {surface!r} has {different} different colours and the "
            f"{kind} recipe needs at least {k}. {advice}"
        )
    centres, counts = _kmeans(pixels, k)
    if definition.calm:
        # Pulled towards the biggest shade, not the mean: a small dark cluster (outline
        # pixels, a shadow at the edge) must not drag the wall's own colour towards tan.
        biggest = centres[counts.argmax()]
        centres = np.floor(centres + definition.calm * (biggest - centres) + 0.5)
        centres = centres.astype(np.int64)
    shares = counts / n
    line = definition.line_shade
    if line is not None and shares[line] < MIN_LINE_SHADE_SHARE:
        raise RecipeRefused(
            f"The line shade of {surface!r} (the {kind} recipe's "
            f"{definition.line_name} colour) covers "
            f"only {shares[line]:.1%} of the box, under the "
            f"{MIN_LINE_SHADE_SHARE:.0%} a line shade needs. The box may miss the "
            f"surface's lines or hold a flat patch of it. {advice}"
        )
    body = [share for i, share in enumerate(shares) if i != line]
    if min(body) < MIN_SHADE_SHARE:
        raise RecipeRefused(
            f"One colour of {surface!r} covers only {min(body):.1%} of the box, "
            f"under the {MIN_SHADE_SHARE:.0%} a shade needs, so a stray object is "
            f"probably inside it. {advice}"
        )
    return [
        Shade((int(c[0]), int(c[1]), int(c[2])), round(float(count) / n, 4))
        for c, count in zip(centres, counts, strict=True)
    ]


def _seed(name: str, part: str) -> random.Random:
    """The generator for `part` ("fixed", "layout" or a variant number) of a name.

    The seed is the first 8 bytes of sha256 of "name:part", big-endian; never
    `hash()`, which changes every process.
    """
    digest = hashlib.sha256(f"{name}:{part}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _cut_block(block: np.ndarray) -> list[np.ndarray]:
    """A (32, 32, 3) block as 4 opaque RGBA tiles in PLACES order (nw, ne, sw, se)."""
    rgba = np.full((BLOCK_SIZE, BLOCK_SIZE, 4), 255, dtype=np.uint8)
    rgba[..., :3] = block
    half = RECIPE_CELL_SIZE
    return [rgba[y : y + half, x : x + half].copy() for y in (0, half) for x in (0, half)]


#: Wood's rows of planks in a block: the joint line row, then the first and last row of
#: the plank under it. Planks are 10 px wide (the last one 9), 3 to a block.
_WOOD_ROWS = ((0, 1, 10), (11, 12, 21), (22, 23, 31))
#: The shortest a plank may be, in pixels. A plank that runs across a block's left or
#: right edge goes on in the next block, so each end piece is at least half of it and
#: the two pieces together always reach it, whichever blocks sit side by side.
WOOD_PLANK_MIN = 12


#: A row of planks has an end joint in a block with this chance, and none otherwise, so most
#: planks run on into the next block.
_WOOD_JOINT_CHANCE = 0.5
#: The tile join's two columns never hold a joint, so a joint cannot make that line busier
#: than the lines inside the block.
_WOOD_JOIN_COLUMNS = (RECIPE_CELL_SIZE - 1, RECIPE_CELL_SIZE)


def _wood_joint(rng: random.Random) -> int | None:
    """The end-joint column of one row of planks in one block, or None for no joint.

    A joint leaves a piece of at least WOOD_PLANK_MIN // 2 columns on each side, so the
    piece plus the one the next block adds is at least WOOD_PLANK_MIN.
    """
    if rng.random() >= _WOOD_JOINT_CHANCE:
        return None
    end = WOOD_PLANK_MIN // 2
    columns = [c for c in range(end, BLOCK_SIZE - end) if c not in _WOOD_JOIN_COLUMNS]
    return rng.choice(columns)


def _wood_blocks(shades: Sequence[Shade], name: str) -> list[np.ndarray]:
    """The 32 wood tiles: 8 blocks of three rows of long planks, shades[0] the joints.

    The two plank shades next to each other that together cover most of the outline are
    the floor's tone. Each row takes one of the two, from the "fixed" generator, the same
    in all 8 blocks, so the rows barely differ and the planks at a block's left and right
    edge are the same shade: one block's right edge joins any block's left edge exactly.
    A row has no joint or one end joint in a block (see WOOD_PLANK_MIN), so a plank runs on
    into the next block. Grain strokes and tone patches of a neighbouring shade, drawn per
    block, only land on plank pixels away from the edge columns, so the joint columns,
    the joint rows and the edge columns stay as they are.
    """
    joint = shades[0].rgb
    planks = [s.rgb for s in shades[1:]]
    tone = max(range(len(planks) - 1), key=lambda i: shades[i + 1].share + shades[i + 2].share)
    fixed = _seed(name, "fixed")
    row_shades = [tone + fixed.randrange(2) for _ in _WOOD_ROWS]
    blocks = []
    for variant in range(1, VARIANTS + 1):
        rng = _seed(name, str(variant))
        block = np.zeros((BLOCK_SIZE, BLOCK_SIZE, 3), dtype=np.uint8)
        for (joint_row, first, last), shade in zip(_WOOD_ROWS, row_shades, strict=True):
            block[joint_row] = joint
            block[first : last + 1] = planks[shade]
            column = _wood_joint(rng)
            if column is not None:
                block[first : last + 1, column] = joint
            near = planks[rng.choice([i for i in (shade - 1, shade + 1) if 0 <= i < len(planks)])]
            for _ in range(rng.randint(8, 14)):
                # A stroke of grain or a small patch of tone, kept to plank pixels
                # and off the two edge columns.
                high = rng.randint(1, 3)
                row = rng.randint(first, last - high + 1)
                start = rng.randint(1, BLOCK_SIZE - 2)
                wide = rng.randint(3, 6) if high == 1 else rng.randint(2, 4)
                area = block[row : row + high, start : min(start + wide, BLOCK_SIZE - 1)]
                area[(area == planks[shade]).all(axis=-1)] = near
        blocks.append(block)
    return [tile for block in blocks for tile in _cut_block(block)]


#: A plaster block's own tile join may differ in at most this share of its pixels, so no
#: grout line runs along it and the check's "no busier than inside" holds for any name.
_JOIN_QUIET = 0.4
#: How many times a block is drawn again (with the next numbers from its generator)
#: when its tile join is busier than that.
_JOIN_ATTEMPTS = 20


def _tile_join_quiet(block: np.ndarray) -> bool:
    """True when the two lines where a block's four tiles meet are both quiet."""
    half = RECIPE_CELL_SIZE
    beside = (block[:, half - 1] != block[:, half]).any(axis=-1).mean()
    above = (block[half - 1] != block[half]).any(axis=-1).mean()
    return max(beside, above) <= _JOIN_QUIET


#: The two rows of slabs of a stone block. Row "a" is rows 9 to 23, row "b" runs across the
#: block's edge (25 to 31, then 0 to 7 of the next block); rows 8 and 24 between them are
#: grout, and none is on the tile join (15 | 16) or on a block edge.
_ROW_A = tuple(range(9, 24))
_ROW_B = (*range(25, BLOCK_SIZE), *range(8))
#: The four slabs of a block, as (rows, columns), each 15 px across. Every pixel they do
#: not cover is grout: columns 8 and 24 in row a, columns 1 and 17 in row b (9 px after
#: row a's, or 7 before, about half a slab, so the slabs are staggered like laid
#: flagstones), none on column 15 or 16. The first slab lies wholly
#: inside the block and may take either tone; the others run across a block edge, so they
#: are always the main tone and look the same in all 8 blocks (see `_stone_blocks`).
_SLABS = (
    (_ROW_A, tuple(range(9, 24))),
    (_ROW_A, (*range(25, BLOCK_SIZE), *range(8))),
    (_ROW_B, tuple(range(2, 17))),
    (_ROW_B, (*range(18, BLOCK_SIZE), 0)),
)
#: The first and last pixel row or column the specks of a stone block may touch: it keeps
#: them out of the STONE_BAND along every edge. They also never touch rows or columns 15
#: and 16, so a speck never makes the block's own tile join busier than it is already.
_SPECK_FIRST, _SPECK_LAST = STONE_BAND, BLOCK_SIZE - 1 - STONE_BAND
_SPECK_SKIP = (RECIPE_CELL_SIZE - 1, RECIPE_CELL_SIZE)


def _slab_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The layout of a stone block, the same in all 8 blocks.

    Returns the slab number of every pixel as a (32, 32) int array (-1 for grout: everything
    no slab covers, and the corner pixel of every slab, which rounds it), the rim (face
    pixels next to grout, wrapping round the block) and the face pixels a speck may land on
    (not rim).
    """
    slab = np.full((BLOCK_SIZE, BLOCK_SIZE), -1, dtype=int)
    for number, (rows, columns) in enumerate(_SLABS):
        slab[np.ix_(rows, columns)] = number
        for y in (rows[0], rows[-1]):
            slab[y, [columns[0], columns[-1]]] = -1
    grout = slab < 0
    near = np.zeros_like(grout)
    for axis in (0, 1):
        for step in (1, -1):
            near |= np.roll(grout, step, axis=axis)
    rim = near & ~grout
    line = np.arange(BLOCK_SIZE)
    away = (line >= _SPECK_FIRST) & (line <= _SPECK_LAST) & ~np.isin(line, _SPECK_SKIP)
    return slab, rim, away[:, None] & away[None, :] & ~grout & ~rim


def _stone_blocks(shades: Sequence[Shade], name: str) -> list[np.ndarray]:
    """The 32 stone tiles: 8 blocks of four staggered, rounded slabs, shades[4] the grout.

    A slab is 15 px across with its corner pixel rounded off and 1 px of grout between
    slabs; the grout grid is the same in every block. The two rows of slabs are offset
    by about half a slab, and no grout line lies on the tile join or a block edge. A slab
    is one of the two neighbouring body shades that together cover most of the outline
    (a shade of few pixels, the dark outline at the mask's edge, is not a stone), and
    has a 1 px rim next to the grout in the next darker body shade (a slab whose face is
    the darkest shade has no rim). Three slabs run across
    a block edge; they are always the main tone, so they are the same in all 8 blocks.
    The fourth slab takes main or other from the block's own generator, so the other-tone
    slabs land irregularly on a laid floor. The blocks also differ by small specks of the
    slab's other tone, kept off the STONE_BAND and off rows and columns 15 and 16. So the
    STONE_BAND pixels along every edge are identical in all 8 blocks and any block sits
    next to any other. A block's own side-by-side tile join differs in 15 of 32 pixels (the
    rim beside grout in row b), a tie with the busiest line drawn inside a block, so it
    passes `check_seams` with no margin.
    """
    body = np.array([s.rgb for s in shades[:4]], dtype=np.uint8)
    first = max(range(3), key=lambda i: shades[i].share + shades[i + 1].share)
    main, other = (
        (first, first + 1) if shades[first].share > shades[first + 1].share else (first + 1, first)
    )
    slab, rim, speckable = _slab_frame()
    grout = np.array(shades[4].rgb, dtype=np.uint8)
    blocks = []
    for variant in range(1, VARIANTS + 1):
        rng = _seed(name, str(variant))
        tones = np.array([rng.choice((main, other)), main, main, main])
        face = tones[np.maximum(slab, 0)]
        specks = face.copy()
        for _ in range(rng.randint(14, 24)):
            y, x = rng.randrange(BLOCK_SIZE), rng.randrange(BLOCK_SIZE)
            wide = rng.choice((1, 2))
            for xx in range(x, min(x + wide, BLOCK_SIZE)):
                if speckable[y, xx]:
                    # The slab's other tone: the main shade on an other slab and the reverse.
                    specks[y, xx] = main + other - face[y, xx]
        block = body[np.where(rim, np.maximum(face - 1, 0), specks)]
        block[slab < 0] = grout
        blocks.append(block)
    return [tile for block in blocks for tile in _cut_block(block)]


#: Value-noise lattice of a plaster block: 4 x 4 points, 8 px apart, wrapping round.
_LATTICE = 4
_LATTICE_STEP = BLOCK_SIZE // _LATTICE
#: The share of the raw noise that is the base shade, the calm wall; the one shade next to
#: it that is closest in colour, and differs from the base, takes the rest as occasional
#: faint patches (the closest differing shade anywhere when neither neighbour does).
PLASTER_BASE_SHARE = 0.86
#: A block's own noise takes over from the shared noise over this many pixels beyond the
#: STONE_BAND.
_PLASTER_FADE = 2
#: How much of its swing the shared (edge) noise keeps: it only just leaves the base
#: shade's range, so the band holds a few faint patches and no motif the eye can find.
_SHARED_STRENGTH = 0.8


def _plaster_blocks(shades: Sequence[Shade], name: str) -> list[np.ndarray]:
    """The 32 plaster tiles: 8 blocks of large soft patches on a mostly one-shade wall.

    Smooth value noise on a wrap-around lattice (4 x 4 points, 8 px apart). One noise,
    from the "fixed" generator, is the same in all 8 blocks; each block adds its own,
    from its own generator, and the two are mixed: the shared noise alone in the
    STONE_BAND pixels along every edge, the block's own from _PLASTER_FADE pixels
    further in. The shared noise keeps only _SHARED_STRENGTH of its swing around the
    middle of the base shade's range, so the band holds a few faint patches and no
    motif the eye can pick out. The noise is cut into the shades by thresholds that are
    the same for all 8 blocks (quantiles of their pooled values): the base shade, the
    one with the largest share, takes PLASTER_BASE_SHARE of them. The patches are only
    one shade: of the two right next to the base in the dark-to-light order, the one
    closest to it in colour whose colour differs from the base's (a calmed neighbour can
    be rounded onto the base and would leave a flat wall), so they stay faint (a far shade,
    often outline and shadow pixels, would be camouflage blobs on a calm wall). When
    neither neighbour differs, the closest shade anywhere that does. The edge bands are
    identical in all 8 blocks, which is what lets any block sit next to any other; the
    shared noise wraps round, so its patches cross the edges instead of leaving a
    patch-free grid every 32 px. A block
    whose own tile join is busier than the rest is drawn again (up to _JOIN_ATTEMPTS
    rounds), the cut-offs and every block's colours being worked out again each round.
    The band is not flat, so the finished set is run through `check_seams`; if it fails
    the seam check, the shared band is drawn again (up to _JOIN_ATTEMPTS times).
    """
    colours = [s.rgb for s in shades]
    base = max(range(len(shades)), key=lambda i: shades[i].share)
    steps = np.arange(BLOCK_SIZE) / _LATTICE_STEP
    cell = steps.astype(int)
    frac = steps - cell
    frac = frac * frac * (3 - 2 * frac)
    towards = np.minimum(np.arange(BLOCK_SIZE), np.arange(BLOCK_SIZE)[::-1])
    away = np.clip((towards - (STONE_BAND - 1)) / _PLASTER_FADE, 0, 1)
    away = away * away * (3 - 2 * away)
    weight = np.multiply.outer(away, away)
    rngs = [_seed(name, str(variant)) for variant in range(1, VARIANTS + 1)]

    def draw_noise(rng: random.Random) -> np.ndarray:
        """One block's smooth noise, from the next numbers of its own generator."""
        lattice = np.array([[rng.random() for _ in range(_LATTICE)] for _ in range(_LATTICE)])
        top = lattice[cell][:, cell] * (1 - frac) + lattice[cell][:, (cell + 1) % _LATTICE] * frac
        bottom = (
            lattice[(cell + 1) % _LATTICE][:, cell] * (1 - frac)
            + lattice[(cell + 1) % _LATTICE][:, (cell + 1) % _LATTICE] * frac
        )
        return top * (1 - frac)[:, None] + bottom * frac[:, None]

    def distance(i: int) -> int:
        """Squared colour distance of shade `i` from the base shade."""
        return sum((a - b) ** 2 for a, b in zip(colours[i], colours[base], strict=True))

    neighbours = [i for i in (base - 1, base + 1) if 0 <= i < len(shades)]
    # A calmed shade one level off the base is rounded onto it, and a patch of the base's own
    # colour would leave every block flat and identical: take a shade that differs.
    other = min(
        [i for i in neighbours if distance(i) > 0]
        or [i for i in range(len(shades)) if distance(i) > 0]
        or neighbours,
        key=distance,
    )
    portions = [
        PLASTER_BASE_SHARE if i == base else (1 - PLASTER_BASE_SHARE if i == other else 0.0)
        for i in range(len(shades))
    ]
    own = [draw_noise(rng) for rng in rngs]
    bounds = np.quantile(np.concatenate([n.ravel() for n in own]), np.cumsum(portions))
    bounds = np.concatenate([[0.0], bounds])
    middle = (bounds[base] + bounds[base + 1]) / 2
    fixed = _seed(name, "fixed")
    palette = np.array(colours, dtype=np.uint8)
    # Cut only between the two shades that are drawn (the base and its patch shade): a shade
    # with no portion must get no pixel, which a cut at the pooled minimum would give it.
    first = min(base, other)
    last = max(base, other)
    tiles: list[np.ndarray] = []
    for _ in range(_JOIN_ATTEMPTS):
        shared = middle + (draw_noise(fixed) - middle) * _SHARED_STRENGTH
        for _ in range(_JOIN_ATTEMPTS):
            noise = [shared + (n - shared) * weight for n in own]
            cuts = np.quantile(
                np.concatenate([n.ravel() for n in noise]), np.cumsum(portions)[first:last]
            )
            blocks = [palette[first + np.searchsorted(cuts, n)] for n in noise]
            busy = [i for i, block in enumerate(blocks) if not _tile_join_quiet(block)]
            if not busy:
                break
            # The cut-offs are pooled over all 8 blocks, so redrawing a busy block moves
            # them: every block is coloured and judged again after each round.
            for i in busy:
                own[i] = draw_noise(rngs[i])
        tiles = [tile for block in blocks for tile in _cut_block(block)]
        try:
            check_seams(tiles, kind="plaster", surface=name)
        except RecipeSeamBroken:
            continue  # the set fails the seam check: draw the shared band again
        break
    return tiles


#: The recipes that exist: one entry each, so a new kind is one more entry here and
#: in RECIPE_KINDS.
_KINDS: dict[str, _Kind] = {
    "wood": _Kind(colours=5, draw=_wood_blocks, line_shade=0, line_name="joint"),
    "stone": _Kind(colours=5, draw=_stone_blocks, line_shade=4, line_name="grout"),
    "plaster": _Kind(colours=4, draw=_plaster_blocks, calm=PLASTER_CALM),
}


def _kind(kind: str, surface: str) -> _Kind:
    """The definition of `kind`, or RecipeRefused when there is none to use."""
    if kind not in RECIPE_KINDS:
        raise RecipeRefused(
            f"{surface!r} asks for the recipe kind {kind!r}; the kinds are "
            f"{', '.join(RECIPE_KINDS)}."
        )
    return _KINDS[kind]


def make_tiles(kind: str, shades: Sequence[Shade], name: str) -> list[np.ndarray]:
    """The surface's tiles, each (16, 16, 4) uint8 and opaque.

    32: block 1's four in PLACES order, then block 2's, and so on.

    The same name and the same shades with their shares give the same bytes.
    """
    return _KINDS[kind].draw(shades, name)


def _differ_share(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Share of pixel pairs that differ, over the last-but-one axis.

    `left` and `right` are (..., n, 4) arrays of pixels; two pixels differ when
    their four RGBA values are not all equal.
    """
    return (left != right).any(axis=-1).mean(axis=-1)


def check_seams(tiles: Sequence[np.ndarray], *, kind: str, surface: str) -> None:
    """Raise RecipeSeamBroken unless the 8 blocks can sit in any order.

    The 32 tiles are stitched back into 8 blocks of 32 x 32. On each axis the worst
    join over all 64 ordered pairs (a's last column or row against b's first) and
    the worst join between a block's own tiles (column 15 against 16, row 15
    against 16) may not be busier than the busiest line inside a block
    (k to k + 1; k = 1 to 13 and 17 to 29), and no two blocks may be identical.
    Exact RGBA equality, no tolerance. Wood's above axis can never fail: its joint
    rows make the busiest inside line 1.0.
    """
    stack = np.stack(list(tiles))
    stack = np.stack(
        [
            np.concatenate([np.hstack([group[0], group[1]]), np.hstack([group[2], group[3]])])
            for group in stack.reshape(-1, BLOCK_TILES, *stack.shape[1:])
        ]
    )
    for a in range(len(stack)):
        for b in range(a + 1, len(stack)):
            if np.array_equal(stack[a], stack[b]):
                raise RecipeSeamBroken(
                    f"{surface!r} ({kind}): variants {a + 1} and {b + 1} are identical.",
                    surface=surface,
                    kind=kind,
                    rule="duplicate",
                    units=(a + 1, b + 1),
                )
    # Beside: a's last column against b's column 0. Above: a's last row against b's row 0.
    # Swapping the two pixel axes turns the "above" case into the "beside" one.
    middle = stack.shape[2] // 2
    for axis, view in (("beside", stack), ("above", stack.swapaxes(1, 2))):
        rates = _differ_share(view[:, None, :, -1, :], view[None, :, :, 0, :])
        a, b = np.unravel_index(np.argmax(rates), rates.shape)
        worst = float(rates[a, b])
        busiest = max(
            float(_differ_share(view[:, :, k, :], view[:, :, k + 1, :]).max())
            for k in _BLOCK_INSIDE_LINES
        )
        if worst > busiest:
            raise RecipeSeamBroken(
                f"{surface!r} ({kind}): the {axis} join of variants {a + 1} and {b + 1} "
                f"differs in {worst:.3f} of its pixels, busier than the busiest line "
                f"inside a block ({busiest:.3f}).",
                surface=surface,
                kind=kind,
                rule="block-join",
                axis=axis,
                worst=worst,
                busiest_inside=busiest,
                units=(int(a) + 1, int(b) + 1),
            )
        joins = _differ_share(view[:, :, middle - 1, :], view[:, :, middle, :])
        n = int(np.argmax(joins))
        if float(joins[n]) > busiest:
            raise RecipeSeamBroken(
                f"{surface!r} ({kind}): the {axis} tile join inside block {n + 1} "
                f"differs in {float(joins[n]):.3f} of its pixels, busier than the "
                f"busiest line inside a block ({busiest:.3f}).",
                surface=surface,
                kind=kind,
                rule="tile-join",
                axis=axis,
                worst=float(joins[n]),
                busiest_inside=busiest,
                units=(n + 1,),
            )


def sample_layout(name: str) -> list[int]:
    """The order of the sample floor: 15 independent uniform picks of 1 to 8 in reading order.

    One per block of the 5 x 3 blocks, from the "layout" generator of the name, so the
    same name always lays out the same sample.
    """
    rng = _seed(name, "layout")
    return [rng.randrange(VARIANTS) + 1 for _ in range(SAMPLE_COLUMNS * SAMPLE_ROWS // BLOCK_TILES)]


def room_layout(name: str, blocks: int) -> list[int]:
    """One independent uniform pick of 1 to 8 per block of a room's tile map, in reading order.

    Seeded from its own "room" generator of the name, so the same name always
    paints the same room and `sample_layout`'s order is not disturbed.
    """
    rng = _seed(name, "room")
    return [rng.randrange(VARIANTS) + 1 for _ in range(blocks)]
