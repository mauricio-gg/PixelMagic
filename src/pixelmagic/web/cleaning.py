"""The clean-up screen: any picture turned into true pixel art at native size.

The page only reads a form and shows what the core kept. Every rule is the core's
(`stages/cleanup.py`): what a usable size is, what counts as unsure, what a
palette file must hold, and when a run is a repeat. A run is a POST that redirects
back to the page (so a refresh never repeats it); the page itself runs nothing and
shows the last stored result.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, Response

from pixelmagic.model import CleanupRecord
from pixelmagic.stages.cleanup import CLEANED_PNG, clean_up, stored_cleanup
from pixelmagic.stages.generate import load_generation
from pixelmagic.store import check_run_id
from pixelmagic.web.deps import Workbench, page, require_direction, see_other, workbench
from pixelmagic.web.images import send_image
from pixelmagic.web.timeline import frame_context

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]

#: The three ways to choose colours; the form posts exactly one of them.
PALETTE_CHOICES = ("auto", "file", "count")
#: The colour count the box shows before anyone has chosen a number.
DEFAULT_COUNT = 16


def _base(direction: str, generation: str) -> str:
    """The URL of one picture's clean-up screen."""
    return f"/directions/{direction}/generations/{generation}/cleanup"


def number(value: float) -> str:
    """A size for a person: whole when it is, else one decimal (4.0 is "4", 10.31 is "10.3")."""
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _read_size(size: str, found_size: str) -> float | None:
    """The typed size, or None for an empty box or one left holding the prefilled found size.

    The page fills the box with the size the last run used. A found size comes back
    in the hidden `found_size` field; a box still equal to it was not typed, so the
    repeat counts as "no size" (a cache hit, recorded as found). A person who types
    the very same number cannot be told apart from that, and gets the same result.
    """
    text = size.strip()
    if not text or text == found_size:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError("The pixel size must be a number, such as 4 or 8.5.") from None


def _read_count(count: str) -> int:
    """The colour count box as a whole number."""
    try:
        return int(count.strip())
    except ValueError:
        raise ValueError("The number of colours must be a whole number, such as 16.") from None


def _exact(value: float) -> str:
    """A float in its shortest form that reads back as the same number (4.0 is "4")."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _prefill(record: CleanupRecord) -> tuple[str, str]:
    """The size box's value after a run, and the part of it that is only the found size.

    A typed size is filled back exactly as typed (the float's shortest round-trip
    form, with a pointless ".0" dropped), so a repeat is the same run. A found size is
    the one size used, filled too in the same exact form (and returned a second time so a
    repeat can tell it was not typed). It is not rounded for the box: a found 4.975 shown
    as "5" would make a person typing 5 look like an unchanged repeat.
    """
    if record.size_source == "typed":
        return _exact(record.size) if record.size is not None else "", ""
    return _exact(record.pixel_size), _exact(record.pixel_size)


def _palette_view(record: CleanupRecord | None) -> tuple[str, int]:
    """Which palette option to show selected, and the number for the count box."""
    if record is not None and record.colours is not None:
        return "count", record.colours
    if record is not None and record.palette_file is not None:
        return "file", DEFAULT_COUNT
    return "auto", DEFAULT_COUNT


@router.get("/directions/{direction}/generations/{generation}/cleanup")
def show(request: Request, direction: str, generation: str, bench: Bench) -> Response:
    """The clean-up screen: the settings, and the last result if there is one. Runs nothing."""
    found = require_direction(bench, direction)
    generation = check_run_id(generation, "generation id")
    picture = load_generation(bench.store, found.name, generation)
    result = stored_cleanup(bench.store, found.name, generation)
    record = result.record if result else None
    size_box, found_size = _prefill(record) if record else ("", "")
    palette, count = _palette_view(record)
    base = _base(found.name, generation)
    return page(
        request,
        "cleanup.html",
        direction=found,
        **frame_context(
            bench, found.name, generation, None, picture.request.subject, timeline=False
        ),
        picture=picture,
        generation_url=f"/directions/{found.name}/generations/{generation}",
        action=base,
        record=record,
        image_url=f"{base}/cleaned.png?v={record.cache_key[:12]}" if record else None,
        saved_in=(
            f"{bench.store.cleanup_dir(found.name, generation).relative_to(bench.store.root)}/"
            if record
            else None
        ),
        has_aseprite=bool(result and result.aseprite_path),
        instant=request.query_params.get("instant") == "1",
        size_box=size_box,
        found_size=found_size,
        size_text=number(record.pixel_size) if record else None,
        palette=palette,
        count=count,
    )


@router.post("/directions/{direction}/generations/{generation}/cleanup")
def run(
    direction: str,
    generation: str,
    bench: Bench,
    size: Annotated[str, Form()] = "",
    found_size: Annotated[str, Form()] = "",
    palette: Annotated[str, Form()] = "auto",
    count: Annotated[str, Form()] = "",
    palette_file: Annotated[UploadFile | None, File()] = None,
) -> Response:
    """Clean the picture up with the form's settings and go back to the page.

    The size box is the one setting that needs care: see `_read_size`. The palette
    file is copied under its own file name (the name only: a client can send a path
    there) into a folder that is deleted afterwards, worked or not, because the core
    reads it in place and keeps only its name. A repeat run (the core's cache hit)
    comes back with `?instant=1`, a fixed code and never text.
    """
    found = require_direction(bench, direction)
    generation = check_run_id(generation, "generation id")
    if palette not in PALETTE_CHOICES:
        raise ValueError("Choose where the colours come from: pick for me, a file or a number.")
    typed_size = _read_size(size, found_size)
    colours = _read_count(count) if palette == "count" else None
    with tempfile.TemporaryDirectory() as scratch:
        chosen = None
        if palette == "file":
            name = Path(getattr(palette_file, "filename", None) or "").name
            if name not in ("", ".", ".."):
                chosen = Path(scratch) / name
                chosen.write_bytes(palette_file.file.read())
            else:
                # A file input cannot be refilled: a repeat sends none, so use the last colours.
                last = stored_cleanup(bench.store, found.name, generation)
                if not (last and last.record.palette):
                    raise ValueError("Choose an Aseprite file to take the colours from.")
        result = clean_up(
            bench.store,
            found,
            generation,
            size=typed_size,
            colours=colours,
            palette_file=chosen,
            reuse_last_palette=palette == "file" and chosen is None,
        )
    return see_other(_base(found.name, generation) + ("?instant=1" if result.cache_hit else ""))


@router.get("/directions/{direction}/generations/{generation}/cleanup/cleaned.png")
def cleaned_image(direction: str, generation: str, bench: Bench) -> FileResponse:
    """The cleaned picture, at native size (the page scales it up crisp)."""
    folder = bench.store.cleanup_dir(
        require_direction(bench, direction).name, check_run_id(generation, "generation id")
    )
    return send_image(folder / CLEANED_PNG, bench.store.root)
