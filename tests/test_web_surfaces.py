import json
import re

from PIL import Image

from pixelmagic.model import Reviewer
from pixelmagic.stages.generate import import_picture
from pixelmagic.store import Store
from test_recipes import floor_and_block_scene

#: A wood floor typed the recipe way, on the shared synthetic scene's floor.
FLOOR_SURFACE = "wood floor:wood; box 10,10 50,50; shape 0,0 100,0 100,88 0,88"


def approved_floor(project_root):
    """Import the shared synthetic scene (a floor beside a block), which is approved already."""
    store = Store(project_root)
    path = project_root / "den.png"
    Image.fromarray(floor_and_block_scene().picture, "RGBA").save(path)
    imported = import_picture(
        store, store.load_direction("cosy-den"), path, reviewer=Reviewer.HUMAN
    )
    return store, f"/directions/cosy-den/generations/{imported.request.id}/surfaces"


def geometry_of(response):
    """The JSON the page hands its script: the drawn surfaces and the one being edited."""
    found = re.search(r'id="surfaces-geometry">(.*?)</script>', response.text, re.S)
    return json.loads(found.group(1))


def test_the_colours_route_answers_with_colours_or_the_cores_sentence(client, project_root):
    """Catches: the colour readout failing silently (or with a traceback page) when the box
    is off the picture or mistyped, so a person is left guessing why nothing shows."""
    _, surfaces = approved_floor(project_root)
    good = client.post(f"{surfaces}/colours", data={"kind": "wood", "box": "50,50 10,10"})
    assert good.status_code == 200
    body = good.json()
    assert len(body["shades"]) == len(body["shares"]) == 5
    assert all(re.fullmatch(r"#[0-9a-f]{6}", shade) for shade in body["shades"])

    off = client.post(f"{surfaces}/colours", data={"kind": "wood", "box": "130,10 170,50"})
    assert off.status_code == 400
    assert "wholly inside" in off.json()["error"]

    one_corner = client.post(f"{surfaces}/colours", data={"kind": "wood", "box": "10,10"})
    assert one_corner.status_code == 400
    assert "two corners" in one_corner.json()["error"]


def test_the_draw_route_makes_assets_or_answers_with_a_sentence_and_no_traceback(
    client, project_root
):
    """Catches: a refused surface (a 2-corner shape, a name another piece uses) reaching the
    page as a traceback or an HTML error page the script cannot show, or half-drawn tiles."""
    store, surfaces = approved_floor(project_root)

    two_corners = client.post(
        f"{surfaces}/draw",
        data={"surface": "wood floor:wood; box 10,10 50,50; shape 0,0 100,0"},
    )
    assert two_corners.status_code == 400
    assert "at least 3 corners" in two_corners.json()["error"]

    taken = store.asset_dir("cosy-den", "wood-floor")
    taken.mkdir(parents=True)
    (taken / "source.png").write_bytes(b"another piece")
    clash = client.post(f"{surfaces}/draw", data={"surface": FLOOR_SURFACE})
    assert clash.status_code == 400
    assert "already used by another piece" in clash.json()["error"]
    assert "Traceback" not in clash.text
    assert not any(store.decompositions_dir("cosy-den").iterdir())
    assert store.list_assets("cosy-den") == ["wood-floor"]

    (taken / "source.png").unlink()
    taken.rmdir()
    drawn = client.post(f"{surfaces}/draw", data={"surface": FLOOR_SURFACE})
    assert drawn.status_code == 200
    assert drawn.json() == {"next": f"{surfaces}?drawn=wood-floor"}
    assert len(store.list_assets("cosy-den")) == 32


def test_a_drawn_surface_opens_for_editing_with_its_geometry_and_can_be_removed(
    client, project_root
):
    """Catches: Edit opening the editor empty or with another shape than the one drawn, a
    name nobody drew being accepted, Remove leaving the surface's assets behind,
    and a repeated Remove landing on an error page that a reload would re-post."""
    store, surfaces = approved_floor(project_root)
    client.post(f"{surfaces}/draw", data={"surface": FLOOR_SURFACE})

    page = client.get(f"{surfaces}?edit=wood-floor&drawn=wood-floor")
    assert page.status_code == 200
    assert geometry_of(page)["editing"] == {
        "name": "wood-floor", "text": "wood floor", "kind": "wood",
        "box": [10, 10, 50, 50], "shapes": [[[0, 0], [100, 0], [100, 88], [0, 88]]],
    }

    unknown = client.get(f"{surfaces}?edit=marble")
    assert unknown.status_code == 404
    assert "Traceback" not in unknown.text

    removed = client.post(f"{surfaces}/wood-floor/remove", follow_redirects=False)
    assert removed.status_code == 303
    assert removed.headers["location"] == surfaces
    assert store.list_assets("cosy-den") == []
    assert geometry_of(client.get(surfaces))["drawn_count"] == 0

    # A second remove (a stale page) goes back to the screen, not to an error page
    # at the POST-only address that a reload would send again.
    again = client.post(f"{surfaces}/wood-floor/remove", follow_redirects=False)
    assert (again.status_code, again.headers["location"]) == (303, surfaces)
