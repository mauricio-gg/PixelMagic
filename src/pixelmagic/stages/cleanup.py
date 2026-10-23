"""Job 1, clean up a picture: any picture to clean pixel art at native size.

Any generation can be cleaned, approved or not: clean-up is free and changes nothing
else.

A hand-drawn or AI-made picture, such as an enlarged pixel-art sprite, comes
back as one pixel per art pixel, saved as ``cleaned.png`` and ``cleaned.aseprite``
under ``directions/<direction>/cleanups/<generation-id>/`` with a
``cleanup.json`` sidecar. One result per picture: a run with other settings
replaces the last one. A clean-up result is not an asset; refine, export, the
journey and the room never see it.

The steps:

1. The finder in ``stages/rebuild.py``, unchanged, reads a pixel size and a score
   on each axis. It always runs, so both readings are always reported, even when the
   person typed a size. Pixel art pixels are square, so the result uses ONE size for
   both axes: the typed one, else the smaller of the two readings (a reading can land on a
   multiple of the true size, and too small keeps detail where too big loses it; the
   rule is ``rebuild.square_axes``, shared with the rebuild). The phase of
   each axis is then worked out at that one size. The result is
   flagged unsure when either axis scored below ``rebuild.FOUND_BAR``, the two
   readings disagree by more than ``READINGS_DISAGREE``, or either reading is within
   ``NEAR_SEARCH_TOP`` of the top of the finder's range; it is still made, with the
   found size unchanged, so the person can see the size and correct it.
2. Cells: every grid cell with at least half its length inside the picture is
   one output pixel (so no extra row or column).
3. A cell is opaque when at least ``rebuild.CELL_IN_MASK`` of its in-picture
   pixels are; the result's alpha is 0 or 255.
4. A cell's colour is the per-channel median of the opaque pixels in its centre
   (``CENTRE_TRIM`` of the pitch cut off every side), so blur at cell edges and
   bleed from neighbours do not vote.
5. Colours are merged in OKLab: leader clustering (heaviest colour first, a
   colour within ``LEADER_DIST`` of a heavier one joins it, the distance widened
   until at most ``WARD_START`` groups remain), then Ward merging
   (the pair whose merge adds least squared error, weighted by cell count) down
   to the asked count, or to ``MAX_COLOURS`` when none was asked. A merged group
   is shown by its most common member's exact colour, never an average.
6. With a palette file (an .aseprite) the colour reduction is replaced: every
   opaque cell takes the nearest colour of the palette in OKLab. The palette is
   the set of distinct opaque colours actually drawn in the sprite (every frame
   as it looks, visible layers), not Aseprite's palette table, which is often
   just the default 32 colours.

Ideas borrowed (their code is not copied): pixel-snapper (MIT) for one output
pixel per grid cell, the colour vote per cell, and the palette idea; PixelRefiner
for sampling the centre of a cell and reducing colours in OKLab.

A repeat run with the same picture and settings is a cache hit: the stored
result is returned and nothing is recomputed or written.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from pixelmagic.model import CleanupRecord, Direction, GridRebuild
from pixelmagic.stages import export
from pixelmagic.stages import rebuild as rb
from pixelmagic.stages.generate import load_generation
from pixelmagic.store import Store

#: Bump when the clean-up maths changes, so stored results made by the old maths
#: are not reused.
CLEANUP_VERSION = 6
#: The found size is not sure when the larger axis reading is more than this times the
#: smaller one. Set on spike #240: the soft AI-made board figures read 8.85 x 10.3 (1.16,
#: an AI-made character sprite) and 10.98 x 12.08 (1.10) and count as not sure; real
#: enlargements read 4 x 4, 5 x 5, 7 x 7 and 5 x 4.98 (1.004) and do not.
READINGS_DISAGREE = 1.05
#: The found size is not sure when either axis reads within this much of the top of the
#: finder's search (``rebuild.PITCH_MAX``). The finder cannot see sizes above that, so a
#: reading at the top of its range is often the wrong peak. Set on spike #242: three Codex
#: frames read 13.9 to 14.0 for a true size of about 8.6, and #240's noisy 7x enlargement
#: of a character sprite read 13.75 x 13.6 for a true 7; every right reading on those
#: pictures was 10.3 or below.
NEAR_SEARCH_TOP = 0.5
#: Name of the job in the cache key.
JOB_NAME = "cleanup"
#: Share of the pitch cut off every side of a cell before its colour is read.
CENTRE_TRIM = 0.25
#: Starting distance of the leader step: two colours closer than this in OKLab (Euclidean)
#: are one. Widened when too many groups remain (see ``WARD_START``).
LEADER_DIST = 0.015
#: Ward merging starts from at most this many groups (keeps it fast on noisy input).
WARD_START = 300

CLEANED_PNG = "cleaned.png"
CLEANED_ASEPRITE = "cleaned.aseprite"
CLEANUP_FILE = "cleanup.json"


@dataclass(frozen=True)
class CleanupResult:
    """What a face gets back: the record, the files, and whether nothing was recomputed."""

    record: CleanupRecord
    png_path: Path
    #: None when no .aseprite was written (see the record's notes).
    aseprite_path: Path | None
    cache_hit: bool


def cache_key(
    picture: bytes,
    size: float | None,
    colours: int | None,
    palette: list[str] | None = None,
) -> str:
    """sha256 over the picture's bytes, the job name, the settings and the code version.

    The settings go in as canonical JSON (sorted keys) so a setting added later
    (a palette) changes the key without any change to this function's shape.
    """
    settings = json.dumps(
        {"size": size, "colours": colours, "palette": palette}, sort_keys=True
    )
    digest = hashlib.sha256()
    for part in (picture, JOB_NAME.encode(), settings.encode(), str(CLEANUP_VERSION).encode()):
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()


def clean_up(
    store: Store,
    direction: Direction,
    generation_id: str,
    *,
    size: float | None = None,
    colours: int | None = None,
    palette_file: Path | None = None,
    reuse_last_palette: bool = False,
    now: datetime | None = None,
) -> CleanupResult:
    """Clean up one picture of the direction; reuse the stored result if nothing changed.

    ``size`` is a typed pixel size (source pixels per art pixel, used on both axes;
    without one the smaller axis reading is used on both);
    ``colours`` a colour count; ``palette_file`` an existing .aseprite
    whose drawn colours the result must use (not together with ``colours``).
    ``reuse_last_palette`` takes the colours (and the file's name, for the record) from
    this picture's last stored result instead of a file, so a page that cannot refill a
    file input can still repeat a palette run; it is refused when that result has no
    palette, or together with ``palette_file`` or ``colours``.
    ``ValueError`` is raised before anything is written, including for a picture with no
    opaque pixel (nothing to clean up).
    """
    record = load_generation(store, direction.name, generation_id)
    picture = record.image_path.read_bytes()
    pixels = np.asarray(Image.open(record.image_path).convert("RGBA"), dtype=np.uint8)
    shorter = min(pixels.shape[:2])
    if not (pixels[..., 3] > rb.OPAQUE_ALPHA).any():
        raise ValueError(
            "This picture has no solid pixels, so there is nothing to clean up."
        )
    if size is not None and not 1 <= size <= shorter:
        raise ValueError(
            f"The pixel size must be between 1 and {shorter} (the picture's shorter side)."
        )
    if colours is not None and (isinstance(colours, bool) or colours < 1):
        raise ValueError("The colour count must be a whole number, at least 1.")

    palette = None
    palette_name = None if palette_file is None else palette_file.name
    if reuse_last_palette and (palette_file is not None or colours is not None):
        raise ValueError("Reuse the last palette on its own, not with a file or a colour count.")
    if palette_file is not None:
        if colours is not None:
            raise ValueError("Give either a colour count or a palette file, not both.")
        palette = read_palette(palette_file)
    palette_hex = (
        None if palette is None else sorted(f"{r:02x}{g:02x}{b:02x}" for r, g, b in palette)
    )
    if reuse_last_palette:
        last = stored_cleanup(store, direction.name, generation_id)
        if last is None or not last.record.palette:
            raise ValueError("There is no earlier palette for this picture to reuse.")
        palette_hex, palette_name = last.record.palette, last.record.palette_file
        palette = np.array(
            [[int(h[i : i + 2], 16) for i in (0, 2, 4)] for h in palette_hex], np.uint8
        )

    folder = store.cleanup_dir(direction.name, generation_id)
    key = cache_key(picture, size, colours, palette_hex)
    stored = _stored_result(store, folder, key)
    if stored is not None:
        return stored

    ex, ey = rb.padded_profiles(pixels)
    found_x, found_y = rb._find_axis(ex), rb._find_axis(ey)
    # One square size for both axes, the rule shared with the rebuild.
    axis_x, axis_y = rb.square_axes(ex, ey, found_x, found_y, size)
    pixel_size = axis_x.pitch
    pitch_x, pitch_y = axis_x.pitch, axis_y.pitch
    phase_x, phase_y = axis_x.phase, axis_y.phase
    score_x, score_y = axis_x.score, axis_y.score

    rgb, solid = _sample_cells(pixels, pitch_x, phase_x, pitch_y, phase_y)
    if solid.any():
        if palette is not None:
            rgb = snap_to_palette(rgb, solid, palette)
        else:
            rgb = _reduce_colours(rgb, solid, colours)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.where(solid[..., None], rgb, 0)
    out[..., 3] = np.where(solid, 255, 0)

    folder.mkdir(parents=True, exist_ok=True)
    png_path = folder / CLEANED_PNG
    aseprite_path = folder / CLEANED_ASEPRITE
    aseprite_path.unlink(missing_ok=True)  # an older result's file must not outlive it
    # Drop the old sidecar before the new PNG lands: if the run stops between the two, the
    # old key must not sit beside a picture made with other settings and read as a cache hit.
    (folder / CLEANUP_FILE).unlink(missing_ok=True)
    Image.fromarray(out, "RGBA").save(png_path)
    notes: list[str] = []
    wrote_aseprite = _write_aseprite(png_path, aseprite_path, direction.cell_size, notes)

    sidecar = CleanupRecord(
        generation_id=generation_id,
        created=now or datetime.now(UTC),
        cache_key=key,
        code_version=CLEANUP_VERSION,
        size=size,
        colours=colours,
        palette_file=palette_name,
        palette=palette_hex,
        found_pitch_x=found_x.pitch,
        found_pitch_y=found_y.pitch,
        found_score_x=found_x.score,
        found_score_y=found_y.score,
        pixel_size=pixel_size,
        unsure=readings_unsure(found_x, found_y),
        grid=GridRebuild(
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            phase_x=phase_x,
            phase_y=phase_y,
            confidence=float(min(score_x, score_y)),
            native_size=(out.shape[1], out.shape[0]),
        ),
        size_source="found" if size is None else "typed",
        colour_count=int(len(np.unique(out[solid][:, :3], axis=0))) if solid.any() else 0,
        shown_colours=_shown_colours(out),
        notes=notes,
    )
    store.write_model(folder / CLEANUP_FILE, sidecar)
    return CleanupResult(sidecar, png_path, aseprite_path if wrote_aseprite else None, False)


def read_palette(palette_file: Path) -> np.ndarray:
    """The distinct opaque colours drawn in an .aseprite, as an (n, 3) uint8 array.

    Every frame is flattened as it looks and read as RGB (see ``read_frames.lua``);
    only solid pixels (alpha above ``rb.OPAQUE_ALPHA``) count. ``ValueError`` when Aseprite
    is missing, cannot read the file, or the file draws no opaque colour.
    """
    binary = export.find_aseprite()
    if binary is None:
        raise ValueError("Aseprite is needed to read the palette file, and it was not found.")
    try:
        with tempfile.TemporaryDirectory(prefix="pixelmagic-palette-") as scratch:
            frames = export.AsepriteRunner(binary).read_frames(palette_file, Path(scratch))
            found: list[np.ndarray] = []
            for number in range(1, frames + 1):
                with Image.open(Path(scratch) / f"frame-{number}.png") as frame:
                    pixels = np.asarray(frame.convert("RGBA")).reshape(-1, 4)
                found.append(np.unique(pixels[pixels[:, 3] > rb.OPAQUE_ALPHA][:, :3], axis=0))
    except (RuntimeError, OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"Aseprite could not read the palette file {palette_file.name}: {error}"
        ) from error
    colours = np.unique(np.concatenate(found), axis=0) if found else np.empty((0, 3), np.uint8)
    if len(colours) == 0:
        raise ValueError(f"The palette file {palette_file.name} has no opaque colour in it.")
    return colours.astype(np.uint8)


def snap_to_palette(rgb: np.ndarray, solid: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Every solid cell takes the nearest palette colour in OKLab (pixel-snapper's palette idea)."""
    out = rgb.copy()
    uniq, inverse = np.unique(rgb[solid], axis=0, return_inverse=True)
    distance = np.linalg.norm(rb._oklab(uniq)[:, None] - rb._oklab(palette)[None], axis=2)
    out[solid] = palette[distance.argmin(1)][inverse.reshape(-1)]
    return out


def stored_cleanup(store: Store, direction_name: str, generation_id: str) -> CleanupResult | None:
    """The last stored result of a picture, or None when there is none to use.

    None also when the sidecar will not read or it was made by another ``CLEANUP_VERSION``
    (then it is just "no result yet").

    Runs nothing and never opens the picture, so it does not check that the result still
    matches the picture or any setting; ``cache_hit`` is True on what it returns.
    """
    result = _stored_result(store, store.cleanup_dir(direction_name, generation_id), None)
    if result is not None and result.record.code_version != CLEANUP_VERSION:
        return None
    return result


def _stored_result(store: Store, folder: Path, key: str | None) -> CleanupResult | None:
    """The stored result when its sidecar reads, its PNG exists and (given a ``key``) it matches."""
    sidecar_path = folder / CLEANUP_FILE
    png_path = folder / CLEANED_PNG
    if not sidecar_path.exists() or not png_path.exists():
        return None
    try:
        stored = store.read_model(sidecar_path, CleanupRecord)
    except ValueError:
        return None  # an unreadable sidecar is just a miss; the run rewrites it
    if key is not None and stored.cache_key != key:
        return None
    aseprite_path = folder / CLEANED_ASEPRITE
    return CleanupResult(stored, png_path, aseprite_path if aseprite_path.exists() else None, True)


def readings_unsure(found_x: rb._Axis, found_y: rb._Axis) -> bool:
    """True when a score is low, the readings disagree or one sits at the range top."""
    if min(found_x.score, found_y.score) < rb.FOUND_BAR:
        return True
    if max(found_x.pitch, found_y.pitch) >= rb.PITCH_MAX - NEAR_SEARCH_TOP:
        return True
    return max(found_x.pitch, found_y.pitch) > READINGS_DISAGREE * min(found_x.pitch, found_y.pitch)


def _shown_colours(out: np.ndarray) -> list[str]:
    """Distinct opaque colours of the result as ``rrggbb``, most used first, ties by hex."""
    opaque = out[out[..., 3] > 0][:, :3]
    if len(opaque) == 0:
        return []
    colours, counts = np.unique(opaque, axis=0, return_counts=True)
    hexes = [f"{r:02x}{g:02x}{b:02x}" for r, g, b in colours]
    return [h for h, _ in sorted(zip(hexes, counts, strict=True), key=lambda p: (-p[1], p[0]))]


def _write_aseprite(png_path: Path, out_path: Path, cell: int, notes: list[str]) -> bool:
    """Write the .aseprite beside the PNG; on any trouble add a note and keep the PNG."""
    binary = export.find_aseprite()
    if binary is None:
        notes.append("No .aseprite was written because Aseprite was not found.")
        return False
    try:
        export.AsepriteRunner(binary).build(png_path, out_path, cell=cell)
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        out_path.unlink(missing_ok=True)
        notes.append(f"No .aseprite was written: Aseprite failed ({error}).")
        return False
    return True


def _cell_edges(size: int, pitch: float, phase: float) -> list[tuple[float, float]]:
    """Cells with at least half their length inside ``[0, size)``."""
    return [
        (a, b)
        for a, b in rb._cell_bounds(size, pitch, phase)
        if (min(b, size) - max(a, 0.0)) >= 0.5 * (b - a)
    ]


def _centre(a: float, b: float, pitch: float, size: int) -> tuple[int, int]:
    """Whole pixels of the cell's centre, at least one wide, clipped to the picture."""
    q = CENTRE_TRIM * pitch
    lo = round(a + q)
    hi = max(lo + 1, round(b - q))
    lo, hi = max(lo, 0), min(hi, size)
    return (lo, max(hi, lo + 1)) if lo < size else (size - 1, size)


def _sample_cells(pixels, pitch_x, phase_x, pitch_y, phase_y) -> tuple[np.ndarray, np.ndarray]:
    """Per cell (rows x cols): median colour of the centre's opaque pixels, and an opaque flag."""
    h, w = pixels.shape[:2]
    xs = _cell_edges(w, pitch_x, phase_x)
    ys = _cell_edges(h, pitch_y, phase_y)
    rgb = np.zeros((len(ys), len(xs), 3), np.uint8)
    solid = np.zeros((len(ys), len(xs)), bool)
    for j, (ya, yb) in enumerate(ys):
        y0, y1 = max(rb._span(ya, yb)[0], 0), min(rb._span(ya, yb)[1], h)
        cy0, cy1 = _centre(ya, yb, pitch_y, h)
        for i, (xa, xb) in enumerate(xs):
            x0, x1 = max(rb._span(xa, xb)[0], 0), min(rb._span(xa, xb)[1], w)
            block = pixels[y0:y1, x0:x1].reshape(-1, 4)
            if (block[:, 3] > rb.OPAQUE_ALPHA).mean() < rb.CELL_IN_MASK:
                continue
            cx0, cx1 = _centre(xa, xb, pitch_x, w)
            mid = pixels[cy0:cy1, cx0:cx1].reshape(-1, 4)
            mid = mid[mid[:, 3] > rb.OPAQUE_ALPHA]
            if len(mid) == 0:  # centre fully see-through: fall back to the whole cell
                mid = block[block[:, 3] > rb.OPAQUE_ALPHA]
            rgb[j, i] = np.median(mid[:, :3], axis=0).astype(np.uint8)
            solid[j, i] = True
    return rgb, solid


def _leader(lab: np.ndarray, counts: np.ndarray, dist: float) -> np.ndarray:
    """Group id per unique colour: heaviest first, joining an earlier centre within ``dist``."""
    order = np.lexsort((np.arange(len(counts)), -counts))
    group = np.full(len(counts), -1)
    centres: list[int] = []
    for u in order:
        if centres:
            d = np.linalg.norm(lab[centres] - lab[u], axis=1)
            k = int(d.argmin())
            if d[k] <= dist:
                group[u] = k
                continue
        group[u] = len(centres)
        centres.append(int(u))
    return group


def _ward(lab: np.ndarray, counts: np.ndarray, group: np.ndarray, target: int) -> np.ndarray:
    """Merge groups, cheapest first (count-weighted squared OKLab), until ``target`` remain."""
    ids = sorted(set(group.tolist()))
    n = np.array([counts[group == g].sum() for g in ids], float)
    mean = np.array(
        [(lab[group == g] * counts[group == g, None]).sum(0) / n[k] for k, g in enumerate(ids)]
    )
    alive = np.ones(len(ids), bool)
    parent = list(range(len(ids)))
    while alive.sum() > target:
        idx = np.flatnonzero(alive)
        d2 = ((mean[idx, None] - mean[None, idx]) ** 2).sum(2)
        cost = d2 * (n[idx, None] * n[None, idx]) / (n[idx, None] + n[None, idx])
        np.fill_diagonal(cost, np.inf)
        a, b = np.unravel_index(cost.argmin(), cost.shape)
        i, j = idx[a], idx[b]
        mean[i] = (mean[i] * n[i] + mean[j] * n[j]) / (n[i] + n[j])
        n[i] += n[j]
        alive[j] = False
        parent[j] = i
    mapping = {}
    for k, g in enumerate(ids):
        while parent[k] != k:
            k = parent[k]
        mapping[g] = k
    return np.array([mapping[g] for g in group])


def _reduce_colours(rgb: np.ndarray, solid: np.ndarray, n_colours: int | None) -> np.ndarray:
    """Reduce the solid cells' colours (see the module docstring); returns a new ``rgb``."""
    cells = rgb[solid]
    uniq, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    lab = rb._oklab(uniq)
    target = n_colours or rb.MAX_COLOURS
    dist = LEADER_DIST
    group = _leader(lab, counts, dist)
    # Make Ward affordable: widen the leader step until small enough. This departs from the
    # spike prototype, which capped only when a count was given: without a cap a noisy
    # picture (thousands of groups) took 13 to 87 seconds on the default path.
    while len(set(group.tolist())) > WARD_START:
        dist *= 1.5
        group = _leader(lab, counts, dist)
    if len(set(group.tolist())) > target:
        group = _ward(lab, counts, group, target)
    # Each group shows the exact colour of its most common member.
    shown = np.zeros_like(uniq)
    for g in set(group.tolist()):
        members = np.flatnonzero(group == g)
        shown[members] = uniq[members[np.lexsort((-members, counts[members]))[-1]]]
    out = rgb.copy()
    out[solid] = shown[inverse]
    return out
