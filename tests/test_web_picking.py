"""The picking page, driven through the real routes with the stand-in segmenter."""

from fastapi.testclient import TestClient

from pixelmagic.model import Reviewer
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.stages.generate import import_picture
from pixelmagic.stages.pick import load_picks
from pixelmagic.store import Store
from pixelmagic.web.app import create_app

PICK = "/directions/cosy-den/decompositions/{id}"


def suggested_cut(client, project_root, object_picture):
    """Import the eight-object picture and ask for suggestions; return its cut id."""
    store = Store(project_root)
    record = import_picture(
        store, store.load_direction("cosy-den"), object_picture, reviewer=Reviewer.HUMAN
    )
    response = client.post(f"/directions/cosy-den/generations/{record.request.id}/suggest")
    assert response.status_code == 200, response.text
    return store, store.list_decompositions("cosy-den")[0]


def test_a_picture_with_no_suggestions_can_still_be_picked_by_clicking(
    project_root, object_picture
):
    """Catches: the picking page dead-ending ("no page 1") when the segmenter found
    nothing, which is exactly when a person must click the object themselves."""

    class FindsNothing(StubSegmenter):
        def suggest(self, image_path):
            return []

    application = create_app(
        project_root, service="fake", refiner=StubRefiner(), segmenter=FindsNothing()
    )
    with TestClient(application) as client:
        store, cut_id = suggested_cut(client, project_root, object_picture)
        assert client.get(PICK.format(id=cut_id) + "/pick").status_code == 200

        client.post(PICK.format(id=cut_id) + "/pick/click", data={"x": 20, "y": 20})

    picks = load_picks(store, "cosy-den", cut_id).picks
    assert [pick.number for pick in picks] == [1]


def test_a_suggestion_pick_and_a_shift_click_are_saved_and_the_pick_keeps_a_box(
    client, project_root, object_picture
):
    """Catches: the page's POST routes not reaching the core, or an Add-a-spot click
    replacing a picked suggestion's outline with a one-point guess."""
    store, cut_id = suggested_cut(client, project_root, object_picture)
    base = PICK.format(id=cut_id)
    assert client.get(base + "/pick").status_code == 200

    client.post(base + "/pick/suggestion/1")
    before = load_picks(store, "cosy-den", cut_id).picks[0]
    client.post(base + "/picks/1/points", data={"x": 20, "y": 20, "label": 1})

    (after,) = load_picks(store, "cosy-den", cut_id).picks
    assert len(after.points) == len(before.points) + 1
    assert after.box is not None  # the pick still prompts with a box; growth is the core's test


def test_a_malformed_lasso_line_is_a_sentence_and_picks_nothing(
    client, project_root, object_picture
):
    """Catches: a hand-posted or broken line (bad JSON, wrong shape, words for
    numbers) reaching the person as a server error or as the JSON parser's own
    message, or making a pick from garbage."""
    store, cut_id = suggested_cut(client, project_root, object_picture)
    base = PICK.format(id=cut_id)
    for bad in ["not json", "[[1, 2, 3]]", '[["a", "b"]]', "[[NaN, 1]]", "5", "[[true, false]]"]:
        response = client.post(base + "/pick/lasso", data={"loop": bad})
        assert response.status_code == 400, bad
        assert "not a list of [x, y] points" in response.text, bad
        assert "Traceback" not in response.text
    assert load_picks(store, "cosy-den", cut_id).picks == []


def test_undo_takes_back_the_last_pick_and_with_nothing_left_is_a_sentence(
    client, project_root, object_picture
):
    """Catches: the Undo button not reaching the core, or an empty undo list being a 500
    page instead of a sentence the person can read."""
    store, cut_id = suggested_cut(client, project_root, object_picture)
    base = PICK.format(id=cut_id)
    client.post(base + "/pick/click", data={"x": 20, "y": 20})
    assert load_picks(store, "cosy-den", cut_id).picks

    taken_back = client.post(base + "/pick/undo", follow_redirects=False)
    assert taken_back.status_code == 303
    assert load_picks(store, "cosy-den", cut_id).picks == []

    nothing = client.post(base + "/pick/undo")
    assert nothing.status_code == 400
    assert "There is nothing to undo." in nothing.text
    assert "Traceback" not in nothing.text
