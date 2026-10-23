"""An image service that draws synthetic pictures, offline and deterministic.

This is what makes the whole pipeline testable and what makes
`pixelmagic --service fake …` work on a plane. The shapes it paints are
chosen to be exactly what the next stage needs to chew on: a group redraw comes
back as a small chunky board in each used cell.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from pixelmagic.model import (
    KEY_COLOURS,
    BackgroundPolicy,
    GenerationRequest,
    GenerationResult,
    PromptKind,
)
from pixelmagic.providers.base import ImageCapabilities

#: Distinct, non-key colours for the shapes the fake paints.
_SHAPE_COLOURS = [
    (40, 90, 200),
    (200, 80, 40),
    (60, 160, 70),
    (170, 60, 170),
    (210, 170, 50),
]

DEFAULT_DESCRIPTIONS = ["Wooden Chair", "Round Rug", "Small Lamp", "Stone Floor"]


def _canvas(request: GenerationRequest) -> Image.Image:
    """A blank image in the requested background."""
    width, height = request.size
    if request.background is BackgroundPolicy.TRANSPARENT:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    red, green, blue = KEY_COLOURS[request.background]
    return Image.new("RGBA", (width, height), (red, green, blue, 255))


def _chunky_object(
    image: Image.Image, cell: tuple[int, int, int, int], index: int
) -> None:
    """Paint one chunky-board stand-in into `cell` (left, top, width, height).

    The shape is built from equal square blocks on one coarse grid, as a
    real chunky redraw is: block size 4 to 12 reply pixels (it varies with
    `index`, shrunk when the cell is too small for six blocks across), a
    one-block darker outline, an interior of a base fill with a second fill in
    a dotted pattern on every other block and a third as one horizontal band,
    so colour steps run in both directions at the block spacing, and
    transparent outside. The shape fills the middle half of the cell, as the
    old solid rectangle did, with corner blocks cut off by an amount that also
    varies with `index`. Deterministic: only `index` and the cell decide it.
    """
    left, top, cell_w, cell_h = cell
    inner_w, inner_h = cell_w // 2, cell_h // 2
    block = min(4 + (index * 3) % 9, inner_w // 6, inner_h // 6)
    block = max(block, 1)
    cols, rows = inner_w // block, inner_h // block
    # Centre the block grid in the cell.
    origin_x = left + (cell_w - cols * block) // 2
    origin_y = top + (cell_h - rows * block) // 2
    base = _SHAPE_COLOURS[index % len(_SHAPE_COLOURS)]
    dots = _SHAPE_COLOURS[(index + 1) % len(_SHAPE_COLOURS)]
    band = _SHAPE_COLOURS[(index + 2) % len(_SHAPE_COLOURS)]
    outline = tuple(channel // 3 for channel in base)
    cut = index % 3  # corner blocks removed along each side of a corner
    draw = ImageDraw.Draw(image)
    for by in range(rows):
        for bx in range(cols):
            # Cut the four corners, a staircase of `cut` blocks.
            if min(bx, cols - 1 - bx) + min(by, rows - 1 - by) < cut:
                continue
            edge = (
                bx in (0, cols - 1)
                or by in (0, rows - 1)
                or min(bx, cols - 1 - bx) + min(by, rows - 1 - by) == cut
            )
            if edge:
                colour = outline
            elif by == rows // 2:
                colour = band
            elif bx % 2 == 0 and by % 2 == 0:
                colour = dots
            else:
                colour = base
            x, y = origin_x + bx * block, origin_y + by * block
            draw.rectangle([x, y, x + block - 1, y + block - 1], fill=(*colour, 255))


def _paint(request: GenerationRequest) -> Image.Image:
    """Paint the shape layout that matches the request's prompt kind."""
    image = _canvas(request)
    draw = ImageDraw.Draw(image)
    width, height = request.size

    if request.kind is PromptKind.GROUP_REDRAW and request.group is not None:
        # A small chunky stand-in in each used cell: see `_chunky_object`.
        grid = request.group
        cell_w, cell_h = width // grid.cols, height // grid.rows
        for index in range(grid.count):
            row, col = divmod(index, grid.cols)
            _chunky_object(
                image,
                (col * cell_w, row * cell_h, cell_w, cell_h),
                index,
            )
        return image

    # Scene and prop: one centred block, inset so nothing touches the border.
    inset_x, inset_y = width // 4, height // 4
    draw.rectangle(
        [inset_x, inset_y, width - inset_x - 1, height - inset_y - 1],
        fill=(*_SHAPE_COLOURS[0], 255),
    )
    return image


class FakeImageService:
    """Offline `ImageService` used by every test and by `--service fake`."""

    name = "fake"

    def __init__(
        self,
        *,
        painter: Callable[[GenerationRequest], Image.Image] | None = None,
        descriptions: Sequence[str] | None = None,
        capabilities: ImageCapabilities | None = None,
    ) -> None:
        """Optionally override the painter, the labels, or the capabilities.

        Tests that need a specific picture pass their own `painter`; tests that
        need specific cutout names pass `descriptions`, which are handed out in
        order and then start again from the beginning.

        An empty `descriptions` list means the same as no list at all: the
        built-in `DEFAULT_DESCRIPTIONS`. There is no "name nothing" mode —
        `describe` picks a label by remainder, so an empty list would divide
        by zero on the first cutout.
        """
        self._painter = painter or _paint
        self._descriptions = list(descriptions or DEFAULT_DESCRIPTIONS)
        self._description_index = 0
        #: One entry per `generate` call, in order. The fake ignores references
        #: — it always paints the same shapes — but recording them is the only
        #: way an offline test can prove that `stages.generate` drops them for
        #: a service whose capabilities say it cannot take them.
        self.reference_paths_seen: list[list[Path]] = []
        self._capabilities = capabilities or ImageCapabilities(
            transparent_background=True,
            reference_images=True,
            edit=False,
            vision=True,
            sizes=[(512, 512), (1024, 1024)],
        )

    def capabilities(self) -> ImageCapabilities:
        """Describe what this service supports."""
        return self._capabilities

    def generate(
        self,
        request: GenerationRequest,
        out_path: Path,
        *,
        reference_paths: Sequence[Path] = (),
    ) -> GenerationResult:
        """Paint the request and save it, returning provenance."""
        self.reference_paths_seen.append(list(reference_paths))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._painter(request).save(out_path, format="PNG")
        return GenerationResult(
            id=request.id,
            created=datetime.now(UTC),
            service=self.name,
            model=request.model,
            elapsed_seconds=0.0,
        )

    def describe(self, image_path: Path | None, question: str) -> str:
        """Hand out the next canned label, cycling round when they run out."""
        label = self._descriptions[self._description_index % len(self._descriptions)]
        self._description_index += 1
        return label
