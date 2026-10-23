"""Serving the pictures inside the project folder, and nothing else.

`FileResponse` has no path check of its own — it stats whatever path it is
handed — and a percent-encoded `..` survives routing and arrives in a path
parameter intact. So every route here builds its path through `Store` out of
segments `check_name` has already refused to let through, and `send_image` is
the second net: it resolves symlinks and `..` before comparing, so a symlink
planted inside the project folder cannot be used to read outside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from pixelmagic.naming import check_name
from pixelmagic.stages.refine import REFINED_FILE, SOURCE_FILE
from pixelmagic.store import check_run_id
from pixelmagic.web.deps import Workbench, workbench

router = APIRouter()

Bench = Annotated[Workbench, Depends(workbench)]


def send_image(path: Path, root: Path) -> FileResponse:
    """Serve one PNG, refusing any path that is not inside the project folder.

    No `filename=` argument: `FileResponse` only sets `content-disposition`
    when it is given one, and an `<img>` needs the picture rendered inline
    rather than downloaded.
    """
    base = Path(root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"{path.name} is not inside this project")
    if not resolved.is_file():
        raise FileNotFoundError(f"there is no picture at {path.name}")
    return FileResponse(resolved, media_type="image/png")


@router.get("/directions/{direction}/generations/{generation}/image.png")
def generation_image(direction: str, generation: str, bench: Bench) -> FileResponse:
    """The picture one generation produced."""
    folder = bench.store.generation_dir(
        check_name(direction, "direction"), check_run_id(generation, "generation id")
    )
    return send_image(folder / "image.png", bench.store.root)


@router.get(
    "/directions/{direction}/decompositions/{decomposition}/cutouts/{cutout}/image.png"
)
def cutout_image(
    direction: str, decomposition: str, cutout: str, bench: Bench
) -> FileResponse:
    """One redrawn piece, as the picking screen's rows show it, before it is kept as an asset."""
    folder = bench.store.cutout_dir(
        check_name(direction, "direction"),
        check_run_id(decomposition, "cut-up run id"),
        check_name(cutout, "piece"),
    )
    return send_image(folder / SOURCE_FILE, bench.store.root)


@router.get("/directions/{direction}/assets/{asset}/source.png")
def asset_source(direction: str, asset: str, bench: Bench) -> FileResponse:
    """An asset as it went into the refiner."""
    folder = bench.store.asset_dir(
        check_name(direction, "direction"), check_name(asset, "asset")
    )
    return send_image(folder / SOURCE_FILE, bench.store.root)


@router.get("/directions/{direction}/assets/{asset}/refined.png")
def asset_refined(direction: str, asset: str, bench: Bench) -> FileResponse:
    """An asset as it came out of the refiner."""
    folder = bench.store.asset_dir(
        check_name(direction, "direction"), check_name(asset, "asset")
    )
    return send_image(folder / REFINED_FILE, bench.store.root)
