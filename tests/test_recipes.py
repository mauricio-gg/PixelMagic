"""Tests for `stages/recipes.py`: tiles are reproducible and the seam check can fail.

Also holds `floor_and_block_scene`, the synthetic picture other test files import
for tests that need a real outline (a floor beside a gap and a blue block).
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
from typing import NamedTuple

import numpy as np
import pytest

from pixelmagic.stages.recipes import (
    STONE_BAND,
    WOOD_PLANK_MIN,
    RecipeRefused,
    RecipeSeamBroken,
    Shade,
    check_seams,
    make_tiles,
    read_shades,
    sample_layout,
)

#: Five browns, dark to light, with the share of the floor each one covers. The shares
#: are near equal so the pinned k-means start (centres at 10%, 30%, ... of the
#: sorted pixels) lands one centre in each brown.
FLOOR_COLOURS = [
    ((60, 38, 24), 0.16),
    ((92, 60, 36), 0.20),
    ((126, 86, 52), 0.22),
    ((160, 114, 70), 0.20),
    ((196, 150, 98), 0.22),
]
BLOCK_COLOUR = (40, 90, 200)


class Scene(NamedTuple):
    """A synthetic scene and where its two things are.

    `picture` is (h, w, 4) uint8 RGBA: transparent except the floor and the block.
    `floor` and `block` are boolean (h, w) arrays, true on each one's pixels.
    """

    picture: np.ndarray
    floor: np.ndarray
    block: np.ndarray


def floor_and_block_scene(width: int = 160, height: int = 96) -> Scene:
    """A wood-like floor of five browns and, past a transparent gap, a blue block.

    The floor fills the left 100 columns and the top `height - 8` rows; the block is
    a 40 x 40 square near the right edge. The gap between them is fully transparent,
    so a stub segmenter labels them as two separate blobs. Fixed seed: same scene
    every call.
    """
    rng = random.Random(7)
    picture = np.zeros((height, width, 4), dtype=np.uint8)
    floor = np.zeros((height, width), dtype=bool)
    block = np.zeros((height, width), dtype=bool)
    colours = [c for c, _ in FLOOR_COLOURS]
    weights = [w for _, w in FLOOR_COLOURS]
    for y in range(height - 8):
        for x in range(100):
            picture[y, x, :3] = rng.choices(colours, weights)[0]
    picture[: height - 8, :100, 3] = 255
    floor[: height - 8, :100] = True
    picture[20:60, 110:150] = (*BLOCK_COLOUR, 255)
    block[20:60, 110:150] = True
    return Scene(picture, floor, block)


#: The shades (with shares) the tests make wood from, dark to light.
WOOD_SHADES = [Shade(rgb, share) for rgb, share in FLOOR_COLOURS]
#: A warm grey stone with a light grout (the lightest, last), and the spike's plaster
#: (#deb890 to #fde8c1), both with the shares an outline would have given them.
STONE_SHADES = [
    Shade(rgb, share)
    for rgb, share in [
        ((92, 84, 76), 0.20),
        ((118, 108, 98), 0.25),
        ((142, 131, 119), 0.20),
        ((166, 154, 140), 0.20),
        ((214, 208, 196), 0.15),
    ]
]
PLASTER_SHADES = [
    Shade(rgb, share)
    for rgb, share in [
        ((222, 184, 144), 0.30),
        ((234, 196, 155), 0.35),
        ((244, 214, 172), 0.20),
        ((253, 232, 193), 0.15),
    ]
]
SHADES = {"wood": WOOD_SHADES, "stone": STONE_SHADES, "plaster": PLASTER_SHADES}

_HASH_TILES = """
import hashlib
from pixelmagic.stages.recipes import Shade, make_tiles
shades = [Shade(tuple(rgb), share) for rgb, share in {colours!r}]
tiles = make_tiles({kind!r}, shades, "oak floor")
print(hashlib.sha256(b"".join(t.tobytes() for t in tiles)).hexdigest())
"""


def test_same_name_gives_same_tiles_in_every_process():
    """Each kind made in two processes with different hash seeds is byte-identical."""
    for kind, shades in SHADES.items():
        code = _HASH_TILES.format(colours=[(s.rgb, s.share) for s in shades], kind=kind)
        digests = set()
        for hash_seed in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            run = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
            )
            digests.add(run.stdout.strip())
        assert len(digests) == 1, kind
        here = hashlib.sha256(
            b"".join(t.tobytes() for t in make_tiles(kind, shades, "oak floor"))
        ).hexdigest()
        assert digests == {here}, kind


def _blocks(tiles):
    """The 8 blocks (32 x 32 x 4) of a block kind's 32 tiles, in PLACES order."""
    return [
        np.concatenate([np.hstack(tiles[i : i + 2]), np.hstack(tiles[i + 2 : i + 4])])
        for i in range(0, len(tiles), 4)
    ]


def _tiles(blocks):
    """The 32 tiles of 8 blocks, the inverse of `_blocks`."""
    return [b[y : y + 16, x : x + 16].copy() for b in blocks for y in (0, 16) for x in (0, 16)]


def test_every_kind_passes_the_seam_check():
    """Every kind passes the seam check for many names.

    Catches a recipe whose block edges do not continue into any other block, or whose
    tile join is busier than the inside of a block, for some name.
    """
    for kind, shades in SHADES.items():
        for number in range(30):
            name = f"{kind} floor {number}"
            check_seams(make_tiles(kind, shades, name), kind=kind, surface=name)


def test_plaster_name_whose_first_draw_has_a_busy_tile_join_is_drawn_again():
    """ "plaster surface 223" passes the seam check with the test's plaster shades.

    Catches an unlucky plaster name crashing decompose with RecipeSeamBroken(tile-join):
    its first draw of a block had a tile join busier than the inside of the block.
    """
    name = "plaster surface 223"
    check_seams(make_tiles("plaster", PLASTER_SHADES, name), kind="plaster", surface=name)


def test_plaster_is_a_calm_wall_with_soft_patches_not_speckle():
    """Plaster's base shade covers over 82% of the 8 blocks and few neighbours differ.

    Catches plaster going back to visible camouflage blobs or an even, busy speckle (the
    patches covering a quarter of the wall as before #164, or the shades spread by their
    outline shares: the base shade near a third, over 30% of neighbouring pixel pairs
    differing).
    """
    for number in range(20):
        blocks = np.stack(_blocks(make_tiles("plaster", PLASTER_SHADES, f"plaster {number}")))
        pixels = blocks[..., :3]
        values, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
        assert counts.max() / counts.sum() > 0.82, number
        beside = (pixels[:, :, 1:] != pixels[:, :, :-1]).any(axis=-1).mean()
        below = (pixels[:, 1:] != pixels[:, :-1]).any(axis=-1).mean()
        assert (beside + below) / 2 < 0.08, number


def test_plaster_patches_cross_the_block_edges_so_no_plain_grid_shows_every_32_px():
    """Plaster's 3-pixel edge band is not one flat colour: it holds patches of other shades.

    Catches the plain cross grid every 32 px coming back (every edge-band pixel the base
    shade, which is how a plaster floor looked before #161): over 20 names, more than 2%
    of the band's pixels are not the block's base shade.
    """
    shares = []
    for number in range(20):
        pixels = np.stack(_blocks(make_tiles("plaster", PLASTER_SHADES, f"plaster {number}")))
        pixels = pixels[..., :3]
        values, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
        edge = np.minimum(np.arange(32), 31 - np.arange(32)) < STONE_BAND
        band = edge[:, None] | edge[None, :]
        shares.append((pixels[:, band] != values[counts.argmax()]).any(axis=-1).mean())
    assert np.mean(shares) > 0.02


def test_plaster_with_a_neighbour_the_same_colour_as_the_base_still_passes_the_seam_check():
    """A calm wall whose neighbour shade equals the base colour is not drawn flat.

    Catches a near-flat wall crashing decompose with a `duplicate` seam failure: `read_shades`
    rounds a shade one level off the base onto it, and patches of that colour left every
    block flat and identical.
    """
    shades = [
        Shade(rgb, share)
        for rgb, share in [
            ((240, 207, 167), 0.29),
            ((240, 207, 167), 0.32),
            ((241, 207, 168), 0.14),
            ((241, 208, 168), 0.24),
        ]
    ]
    for number in range(20):
        name = f"plaster {number}"
        check_seams(make_tiles("plaster", shades, name), kind="plaster", surface=name)


def test_plaster_draws_only_the_base_shade_and_the_shades_next_to_it():
    """With the base shade second from the lightest or the lightest, no pixel is a farther shade.

    Catches one pixel per surface coming out as the darkest shade (a shade with no part in
    the patches was cut at the pooled minimum): the den's real shape, a cream base at
    index 2 or 3, which `PLASTER_SHADES` (base at index 1) never exercised.
    """
    for base in (2, 3):
        shares = [0.05, 0.15, 0.15, 0.15]
        shares[base] = 0.65
        shades = [Shade(s.rgb, share) for s, share in zip(PLASTER_SHADES, shares, strict=True)]
        near = {shades[i].rgb for i in (base - 1, base, base + 1) if i < len(shades)}
        for number in range(20):
            blocks = np.stack(_blocks(make_tiles("plaster", shades, f"plaster {number}")))
            drawn = {
                tuple(int(v) for v in p) for p in np.unique(blocks.reshape(-1, 4)[:, :3], axis=0)
            }
            assert drawn <= near, (base, number)


def test_plaster_keeps_the_walls_own_colour_when_a_dark_minority_is_in_the_outline():
    """The plaster base shade is the outline's most common colour, not dragged darker.

    Catches the den wall coming out tan: a small dark cluster (outline pixels, a shadow
    at the edge) pulled every calmed shade, the wall's own colour included, towards the
    mean. Read from hand-built pixels: a cream wall with 14% dark brown.
    """
    wall = [
        ((120, 80, 50), 0.14),
        ((200, 160, 110), 0.26),
        ((235, 195, 140), 0.24),
        ((250, 210, 150), 0.36),
    ]
    pixels = np.concatenate([np.tile(c, (round(s * 10_000), 1)) for c, s in wall])
    picture = pixels.astype(np.uint8).reshape(1, -1, 3)
    inside = np.ones(picture.shape[:2], dtype=bool)
    shades = read_shades("plaster", picture, inside, surface="bedroom wall")
    assert max(shades, key=lambda s: s.share).rgb == (250, 210, 150)


def test_block_kinds_refuse_a_broken_block_edge_a_broken_tile_join_and_twin_blocks():
    """The block rules: the block edge and the tile join cannot excuse themselves.

    Catches a block edge broken in every block (column 31 inverted) let through,
    a tile join broken in every block (column 16 inverted) let through, and two
    equal blocks let through, for every kind.
    """
    for kind in SHADES:
        blocks = _blocks(make_tiles(kind, SHADES[kind], "oak floor"))

        edge = [b.copy() for b in blocks]
        for block in edge:
            block[:, 31, :3] = 255 - block[:, 31, :3]
        with pytest.raises(RecipeSeamBroken) as edge_error:
            check_seams(_tiles(edge), kind=kind, surface="oak floor")
        assert (edge_error.value.rule, edge_error.value.axis) == ("block-join", "beside")
        assert edge_error.value.worst > edge_error.value.busiest_inside

        join = [b.copy() for b in blocks]
        for block in join:
            block[:, 16, :3] = 255 - block[:, 16, :3]
        with pytest.raises(RecipeSeamBroken) as join_error:
            check_seams(_tiles(join), kind=kind, surface="oak floor")
        assert (join_error.value.rule, join_error.value.axis) == ("tile-join", "beside")
        assert join_error.value.units == (1,)

        twins = list(blocks)
        twins[4] = blocks[1].copy()
        with pytest.raises(RecipeSeamBroken) as twin:
            check_seams(_tiles(twins), kind=kind, surface="oak floor")
        assert (twin.value.rule, twin.value.units) == ("duplicate", (2, 5))


def _closed_runs(is_joint):
    """Lengths of the stretches between two joints in a row; the two open ends are left out."""
    joints = np.flatnonzero(is_joint)
    return [int(gap) - 1 for gap in np.diff(joints) if gap > 1]


def test_no_wood_plank_is_shorter_than_the_minimum_even_across_a_block_edge():
    """Catches a stub plank: one running from a block into the next, whichever two blocks
    sit side by side, or one inside a block, shorter than WOOD_PLANK_MIN."""
    joint = WOOD_SHADES[0].rgb
    for number in range(20):
        blocks = _blocks(make_tiles("wood", WOOD_SHADES, f"oak floor {number}"))
        for left in blocks:
            for right in blocks:
                for row in (5, 16, 27):  # a row of planks in each of a block's three bands
                    strip = np.hstack([left, right])[row, :, :3]
                    runs = _closed_runs((strip == joint).all(axis=-1))
                    assert min(runs, default=WOOD_PLANK_MIN) >= WOOD_PLANK_MIN, (number, row, runs)


def test_wood_planks_are_long_and_the_rows_barely_differ_in_shade():
    """Catches wood going back to brickwork or stripes.

    On the 5 x 3 sample floor of 20 names, the planks closed by two joints have a median
    length over 24 px, and the three rows' edge shades are the same plank shade or two
    next to each other in the sorted shades, not light against dark.
    """
    joint = WOOD_SHADES[0].rgb
    planks = [s.rgb for s in WOOD_SHADES[1:]]
    lengths = []
    for number in range(20):
        name = f"oak floor {number}"
        blocks = _blocks(make_tiles("wood", WOOD_SHADES, name))
        layout = sample_layout(name)
        for line in range(3):
            floor_row = np.hstack([blocks[n - 1] for n in layout[line * 5 : line * 5 + 5]])
            for row in (5, 16, 27):
                lengths += _closed_runs((floor_row[row, :, :3] == joint).all(axis=-1))
        used = [planks.index(tuple(blocks[0][row, 0, :3])) for row in (5, 16, 27)]
        assert max(used) - min(used) <= 1, (number, used)
    assert np.median(lengths) > 24


def test_stone_is_big_staggered_slabs_in_the_read_shades_with_thin_grout():
    """Catches stone going back to a mosaic of small pebbles, an even grid with no stagger,
    a face painted with a shade of few pixels (the dark outline at the mask's edge) or a
    colour that is not a read shade, and grout turning thick: across 20 names every row of
    a block has long unbroken runs between grout, the colours are the two main shades, the
    next darker shade (the rims) and the grout, the vertical grout of the two rows of slabs
    never shares a column, and grout stays a thin share of the pixels."""
    shares = (0.06, 0.10, 0.28, 0.43, 0.13)
    shades = [Shade(s.rgb, share) for s, share in zip(STONE_SHADES, shares, strict=True)]
    allowed = {shades[1].rgb, shades[2].rgb, shades[3].rgb, shades[4].rgb}
    runs, grout_share = [], []
    for number in range(20):
        name = f"stone floor {number}"
        for block in (b[..., :3] for b in _blocks(make_tiles("stone", shades, name))):
            colours = {tuple(c) for c in block.reshape(-1, 3)}
            assert colours <= allowed, (number, colours)
            is_grout = (block == shades[4].rgb).all(axis=-1)
            grout_share.append(is_grout.mean())
            # Row 12 and row 28 are in different rows of slabs, half a slab apart.
            assert not (is_grout[12] & is_grout[28]).any(), number
            for row in is_grout:
                # Runs of face pixels along the row, the block wrapped round its edge.
                cut = np.flatnonzero(row)
                runs.extend(np.diff(np.append(cut, cut[0] + len(row))) - 1)
    assert np.median([r for r in runs if r]) >= 13
    assert max(grout_share) < 0.2


def test_a_thin_joint_shade_passes_and_a_thin_plank_shade_is_refused():
    """The joint shade needs 1%, every other wood shade 3%.

    Catches a joint-shade bar set too high (a floor with thin plank seams
    refused for no reason), a thin joint shade let through or given the
    stray-object sentence, and a thin plank shade let through. Read from
    hand-built pixels: each of the five browns of FLOOR_COLOURS repeated to the
    share given.
    """

    def shares_of(*shares: float) -> list[float]:
        """The shares `read_shades` returns for a wood outline built with these shares."""
        browns = [colour for colour, _ in FLOOR_COLOURS]
        pixels = np.concatenate(
            [np.tile(c, (round(s * 10_000), 1)) for c, s in zip(browns, shares, strict=True)]
        )
        picture = pixels.astype(np.uint8).reshape(1, -1, 3)
        inside = np.ones(picture.shape[:2], dtype=bool)
        return [s.share for s in read_shades("wood", picture, inside, surface="oak floor")]

    # A clean floor whose plank seams are 2% (under the body bar, over the line bar).
    assert shares_of(0.02, 0.30, 0.20, 0.20, 0.28) == [0.02, 0.30, 0.20, 0.20, 0.28]
    # Seams under 1%: refused with the line-shade sentence, never the stray-object one.
    with pytest.raises(RecipeRefused, match="line shade.*0.5%.*under the 1%") as thin_line:
        shares_of(0.005, 0.30, 0.20, 0.20, 0.295)
    assert "stray" not in str(thin_line.value)
    # A plank shade at 2% is still refused with the body sentence.
    with pytest.raises(RecipeRefused, match="2.0%.*under the 3%.*stray object"):
        shares_of(0.30, 0.02, 0.20, 0.20, 0.28)
