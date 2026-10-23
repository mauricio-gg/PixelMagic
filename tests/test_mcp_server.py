"""Tests for the MCP server's shape and the tools that only look.

These exercise `create_server` through an in-memory `Client`, the same way an
agent would talk to it — no subprocess, no network. `StubRefiner` stands in
for PixelRefiner, and the `fake` image service is the only one a test ever
names, matching the web page's own rule.
"""

import pytest
from mcp import Client
from mcp.types import ImageContent, TextContent
from PIL import Image

from pixelmagic import registry
from pixelmagic.mcp_server import PREVIEW_LIMIT, create_server
from pixelmagic.model import AssetRecord, Manifest, PitchSource, RefineReport, ReviewStatus
from pixelmagic.refiner.stub import StubRefiner
from pixelmagic.segmenter.stub import StubSegmenter
from pixelmagic.stages.decompose import load_decomposition
from pixelmagic.stages.generate import load_generation
from pixelmagic.stages.layout import default_anchor
from pixelmagic.stages.refine import ASSET_RECORD_UNREADABLE, REPORT_FILE, WRONG_SIZE, load_report
from pixelmagic.store import Store
from test_recipes import floor_and_block_scene

#: A wood floor typed the recipe way, on the shared synthetic scene's floor.
FLOOR_SURFACE = "wood floor:wood; box 10,10 50,50; shape 0,0 100,0 100,88 0,88"


class RenamedStubRefiner(StubRefiner):
    """A stand-in with a name of its own.

    `CandidateRefiner` in `conftest.py` subclasses `StubRefiner` too, but
    keeps its `name`, which makes it useless for proving the "not real pixel
    art" warning is keyed off `StubRefiner.name` and not off any refiner that
    merely behaves like the stand-in (guardian M2.2 #3).
    """

    name = "renamed-stub"


@pytest.fixture(autouse=True)
def no_aseprite(monkeypatch):
    """Keep this module's exports off any Aseprite the machine happens to have.

    Every test here that calls `export_bundle` goes through the same
    `stages.export.export` the CLI and the web page use, and that function
    reaches for a real Aseprite install when none is stubbed in. These tests
    are about the bundle's PNGs and manifest, not about driving a GUI app in
    batch mode, so `find_aseprite` is patched the same way
    `tests/test_web_refine.py`'s `no_aseprite` fixture (and `tests/test_export.py`,
    `tests/test_cli.py` and `tests/test_end_to_end.py`) already do.
    """
    monkeypatch.setattr("pixelmagic.stages.export.find_aseprite", lambda: None)


def said(result) -> str:
    """Everything a tool answered in words."""
    return "\n".join(
        block.text for block in result.content if isinstance(block, TextContent)
    )


def pictures(result) -> list[ImageContent]:
    """Every picture a tool answered with."""
    return [block for block in result.content if isinstance(block, ImageContent)]


@pytest.mark.anyio
async def test_cut_into_pieces_with_surfaces_names_tiles(agent, project_root):
    generation_id = await bring_in_floor(agent, project_root)
    result = await agent.call_tool(
        "cut_into_pieces", {"generation_id": generation_id, "surfaces": [FLOOR_SURFACE]},
    )
    assert result.is_error is False, said(result)
    assert "wood-floor-v1-nw" in said(result)


@pytest.mark.anyio
async def test_a_bordered_surface_is_refused_with_the_recipe_way_forward(agent, tmp_path):
    """Catches: the removed `:bordered` marker still being accepted by the tool, or the agent
    being told only "Error executing tool" instead of how to type a surface now."""
    generation_id = await bring_in_floor(agent, tmp_path)
    result = await agent.call_tool(
        "cut_into_pieces", {"generation_id": generation_id, "surfaces": ["red rug:bordered"]},
    )
    assert result.is_error is True
    assert "bordered surfaces are gone" in said(result)


@pytest.mark.anyio
async def test_the_server_offers_the_tools_that_only_look(agent):
    names = {tool.name for tool in (await agent.list_tools()).tools}
    assert {"project_overview", "list_assets", "show_asset"} <= names


@pytest.mark.anyio
async def test_every_tool_says_what_it_is_for(agent):
    for tool in (await agent.list_tools()).tools:
        assert tool.description, tool.name


@pytest.mark.anyio
async def test_the_overview_names_the_project_and_its_directions(agent):
    result = await agent.call_tool("project_overview", {})
    assert result.is_error is False
    body = said(result)
    assert "DemoGame" in body
    assert "cosy-den" in body
    assert "fake" in body


@pytest.mark.anyio
async def test_the_overview_says_which_services_are_not_built_yet(agent, monkeypatch):
    """Same rule as the web page's: announce a name, never build a real service."""
    monkeypatch.setitem(registry.PLANNED_SERVICES, "someday", "M9.9")
    body = said(await agent.call_tool("project_overview", {}))
    assert "someday" in body
    assert "M9.9" in body
    assert "not built yet" in body


@pytest.mark.anyio
async def test_the_overview_says_nothing_about_planned_services_when_there_are_none(agent):
    """Which is the state this plan actually runs in, all three having merged."""
    body = said(await agent.call_tool("project_overview", {}))
    assert "not built yet" not in body


@pytest.mark.anyio
async def test_a_direction_nobody_has_is_a_sentence_and_not_a_crash(agent):
    result = await agent.call_tool("list_assets", {"direction": "nope"})
    assert result.is_error is True
    # An unexpected exception would come back as the bare "Error executing
    # tool list_assets"; this is the proof every tool wraps its failures.
    assert "nope" in said(result)


@pytest.mark.anyio
async def test_a_name_that_could_leave_the_project_folder_is_refused(agent):
    result = await agent.call_tool("list_assets", {"direction": "../escape"})
    assert result.is_error is True
    assert "not a name PixelMagic uses" in said(result)


@pytest.mark.anyio
async def test_an_empty_direction_has_no_assets_to_list(agent):
    result = await agent.call_tool("list_assets", {})
    assert result.is_error is False
    assert "no assets" in said(result).lower()


@pytest.mark.anyio
async def test_showing_an_asset_nobody_has_is_a_sentence_and_not_a_crash(agent):
    result = await agent.call_tool("show_asset", {"asset": "nope"})
    assert result.is_error is True
    assert "nope" in said(result)


@pytest.mark.anyio
async def test_an_unknown_tool_name_is_reported_and_not_raised(agent):
    """The SDK answers with is_error rather than raising, so tests assert on it."""
    result = await agent.call_tool("polish_my_shoes", {})
    assert result.is_error is True


async def bring_in_floor(agent, project_root):
    """Bring in the shared synthetic scene (a floor beside a block); hand back its id."""
    path = project_root / "den.png"
    Image.fromarray(floor_and_block_scene().picture, "RGBA").save(path)
    result = await agent.call_tool("bring_in_picture", {"path": str(path)})
    assert result.is_error is False, said(result)
    return Store(project_root).list_generations("cosy-den")[0]


async def drawn(agent, project_root, subject="a cosy den"):
    """Draw one picture and hand back the store and the record."""
    result = await agent.call_tool("draw", {"subject": subject, "size": 64})
    assert result.is_error is False, said(result)
    store = Store(project_root)
    generation_id = store.list_generations("cosy-den")[0]
    return store, load_generation(store, "cosy-den", generation_id), result


@pytest.mark.anyio
async def test_drawing_answers_with_words_and_the_picture(agent, project_root):
    """GATE for mixed content: the SDK's docs say a mixed list is unsupported
    and the 2.1.1 runtime accepts it. If this fails, see the deviations
    register — every preview tool drops to returning the picture alone and the
    sentence moves into `project_overview`'s vocabulary."""
    store, record, result = await drawn(agent, project_root)
    assert record.request.id in said(result)
    shown = pictures(result)
    assert len(shown) == 1
    assert shown[0].mime_type == "image/png"
    assert shown[0].data.startswith("iVBORw0KGgo")


@pytest.mark.anyio
async def test_drawing_with_no_service_chosen_is_a_tool_error_with_the_sentence(project_root):
    """Catches: the agent's draw tool silently using the stand-in, or failing with the
    SDK's withheld "Error executing tool" instead of the sentence it can act on."""
    from mcp import Client

    from pixelmagic.mcp_server import create_server
    from pixelmagic.services import NO_SERVICE_SENTENCE

    server = create_server(project_root, refiner=StubRefiner(), segmenter=StubSegmenter())
    async with Client(server) as unchosen:
        result = await unchosen.call_tool("draw", {"subject": "a cosy den", "size": 64})
    assert result.is_error is True
    assert NO_SERVICE_SENTENCE in said(result)
    assert Store(project_root).list_generations("cosy-den") == []


@pytest.mark.anyio
async def test_rejecting_with_no_service_chosen_leaves_the_picture_pending(agent, project_root):
    """Catches: the rejection being saved before the follow-up is refused, which strands
    the picture as rejected with no next attempt and blocks rejecting it again."""
    from pixelmagic.services import NO_SERVICE_SENTENCE

    store, record, _ = await drawn(agent, project_root)
    server = create_server(project_root, refiner=StubRefiner(), segmenter=StubSegmenter())
    async with Client(server) as unchosen:
        result = await unchosen.call_tool(
            "reject", {"generation_id": record.request.id, "feedback": "too dark"}
        )
    assert result.is_error is True
    assert NO_SERVICE_SENTENCE in said(result)
    assert store.list_generations("cosy-den") == [record.request.id]
    still = load_generation(store, "cosy-den", record.request.id)
    assert still.review.status is ReviewStatus.PENDING


@pytest.mark.anyio
async def test_drawing_leaves_the_picture_waiting_for_the_agents_verdict(agent, project_root):
    store, record, _ = await drawn(agent, project_root)
    assert record.review.status is ReviewStatus.PENDING
    assert record.review.reviewer.value == "agent"


@pytest.mark.anyio
async def test_drawing_a_kind_that_is_not_offered_is_refused(agent):
    result = await agent.call_tool(
        "draw", {"subject": "a sheet", "kind": "parts-sheet", "size": 64}
    )
    assert result.is_error is True


@pytest.mark.anyio
@pytest.mark.parametrize("size", [0, 99999])
async def test_drawing_an_out_of_range_size_is_a_sentence_not_a_crash(
    agent, project_root, size
):
    """Guardian M2.2 #1: `size` had a floor but no ceiling, so an agent could
    ask for a picture large enough that Pillow could not even open it three
    steps later — a bare exception that slips past `readable_errors`
    entirely. The ceiling is now on the field, so a bad size never reaches
    `generate` and no generation folder is left behind."""
    result = await agent.call_tool("draw", {"subject": "a cosy den", "size": size})
    assert result.is_error is True
    assert Store(project_root).list_generations("cosy-den") == []


@pytest.mark.anyio
async def test_approving_records_the_yes_and_keeps_the_reference(agent, project_root):
    store, record, _ = await drawn(agent, project_root)
    result = await agent.call_tool("approve", {"generation_id": record.request.id})
    assert result.is_error is False
    after = load_generation(store, "cosy-den", record.request.id)
    assert after.review.status is ReviewStatus.APPROVED
    assert after.review.reviewer.value == "agent"
    assert store.load_direction("cosy-den").references == [f"{record.request.id}.png"]


@pytest.mark.anyio
async def test_approving_can_keep_the_whole_picture_as_an_asset(agent, project_root):
    store, record, _ = await drawn(agent, project_root)
    await agent.call_tool(
        "approve", {"generation_id": record.request.id, "keep_as_asset": True}
    )
    assert store.list_assets("cosy-den") == ["cosy-den"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("code", "says", "must_not_say"),
    [
        (ASSET_RECORD_UNREADABLE, "PixelMagic did not send it to the refiner", "refiner refused"),
        (
            WRONG_SIZE,
            "the refiner returned the wrong size, so PixelMagic did not keep it",
            "refused",
        ),
    ],
)
async def test_the_asset_list_says_who_refused_an_asset(
    agent, project_root, code, says, must_not_say
):
    """A receipt whose code is PixelMagic's own must not be reported as the
    refiner saying no (and the other way round), or an agent chases the wrong fault."""
    store, record, _ = await drawn(agent, project_root)
    await agent.call_tool(
        "approve", {"generation_id": record.request.id, "keep_as_asset": True}
    )
    receipt = RefineReport(asset="cosy-den", error_code=code, error_message="why it failed")
    store.write_model(store.asset_dir("cosy-den", "cosy-den") / REPORT_FILE, receipt)
    listing = said(await agent.call_tool("list_assets", {}))
    assert says in listing
    assert must_not_say not in listing
    assert "why it failed" in listing


@pytest.mark.anyio
async def test_approving_can_decline_to_keep_a_reference(agent, project_root):
    store, record, _ = await drawn(agent, project_root)
    await agent.call_tool(
        "approve", {"generation_id": record.request.id, "keep_as_reference": False}
    )
    assert store.load_direction("cosy-den").references == []


@pytest.mark.anyio
async def test_rejecting_writes_the_reason_and_draws_the_next_attempt(agent, project_root):
    store, record, _ = await drawn(agent, project_root)
    result = await agent.call_tool(
        "reject", {"generation_id": record.request.id, "feedback": "too dark"}
    )
    assert result.is_error is False
    first = load_generation(store, "cosy-den", record.request.id)
    assert first.review.status is ReviewStatus.REJECTED
    assert first.review.feedback == "too dark"

    ids = store.list_generations("cosy-den")
    assert len(ids) == 2
    child = load_generation(
        store, "cosy-den", next(one for one in ids if one != record.request.id)
    )
    assert child.request.parent_id == record.request.id
    assert "too dark" in child.request.prompt
    assert child.review.status is ReviewStatus.PENDING
    assert child.request.id in said(result)
    assert len(pictures(result)) == 1


@pytest.mark.anyio
async def test_a_second_verdict_on_the_same_picture_is_refused(agent, project_root):
    """The agent's equivalent of a stale tab: a verdict is given once.

    `record_decision` raises `ValueError`, `readable_errors` turns it into a
    `ToolError`, and the caller is told which picture and which verdict rather
    than quietly overwriting the first one.
    """
    _, record, _ = await drawn(agent, project_root)
    first = await agent.call_tool("approve", {"generation_id": record.request.id})
    assert first.is_error is False
    again = await agent.call_tool("approve", {"generation_id": record.request.id})
    assert again.is_error is True
    assert "already" in said(again)


@pytest.mark.anyio
async def test_approving_a_picture_nobody_drew_is_a_sentence(agent):
    result = await agent.call_tool("approve", {"generation_id": "20260101-000000"})
    assert result.is_error is True
    assert "20260101-000000" in said(result)


@pytest.mark.anyio
async def test_a_refiner_failure_while_keeping_a_picture_as_an_asset_leaves_it_retryable(
    project_root,
):
    """Mirrors `test_web_generate.py`'s test of the same name. `approve` runs
    `prepare_promotion` — which calls `refiner.analyze` — before
    `record_decision`, precisely so a refiner failure here leaves the
    generation pending rather than approved with nothing to show for it. A
    `record_decision` written first would refuse a retry as "already
    approved". A retry with a working refiner must then succeed and create
    the asset."""
    from pixelmagic.refiner.base import EXIT_ENGINE_FAILURE, RefinerError

    class FlakyAnalyzeRefiner(StubRefiner):
        """A stub whose `analyze` refuses once, then works like the real thing."""

        def __init__(self):
            self.analyze_calls = 0

        def analyze(self, input_path, settings=None):
            self.analyze_calls += 1
            if self.analyze_calls == 1:
                raise RefinerError(
                    "pixel-refiner crashed", exit_code=EXIT_ENGINE_FAILURE, code="ENGINE_FAILURE"
                )
            return super().analyze(input_path, settings)

    server = create_server(project_root, service="fake", refiner=FlakyAnalyzeRefiner())
    async with Client(server) as flaky:
        store, record, _ = await drawn(flaky, project_root)

        failed = await flaky.call_tool(
            "approve", {"generation_id": record.request.id, "keep_as_asset": True}
        )
        assert failed.is_error is True
        after_failure = load_generation(store, "cosy-den", record.request.id)
        assert after_failure.review.status is ReviewStatus.PENDING
        assert store.list_assets("cosy-den") == []

        retried = await flaky.call_tool(
            "approve", {"generation_id": record.request.id, "keep_as_asset": True}
        )
        assert retried.is_error is False, said(retried)
        after_retry = load_generation(store, "cosy-den", record.request.id)
        assert after_retry.review.status is ReviewStatus.APPROVED
        assert store.list_assets("cosy-den") == ["cosy-den"]


@pytest.mark.anyio
async def test_a_service_that_is_not_built_yet_names_its_ticket(project_root, monkeypatch):
    """Never `service="codex"` — see the Global Constraints. This one draws."""
    monkeypatch.setitem(registry.PLANNED_SERVICES, "someday", "M9.9")
    server = create_server(project_root, service="someday", refiner=StubRefiner())
    async with Client(server) as unbuilt:
        result = await unbuilt.call_tool("draw", {"subject": "a cosy den", "size": 64})
    assert result.is_error is True
    assert "not built yet" in said(result)
    assert "M9.9" in said(result)


async def cut_up(agent, project_root, object_picture):
    """Bring in a picture of separate objects, pick three and redraw them into pieces.

    Hands back the store, the cut-up run's id and the run. The pieces are rebuilt props
    named from the fake service's labels; nothing is kept yet.
    """
    store = Store(project_root)
    await agent.call_tool("bring_in_picture", {"path": str(object_picture)})
    generation_id = store.list_generations("cosy-den")[0]
    await agent.call_tool("suggest_objects", {"generation_id": generation_id})
    cut_id = store.list_decompositions("cosy-den")[0]
    await agent.call_tool("pick_objects", {"cut_id": cut_id, "numbers": [1, 2, 3]})
    redrawn = await agent.call_tool("redraw_objects", {"cut_id": cut_id})
    assert redrawn.is_error is False, said(redrawn)
    return store, cut_id, load_decomposition(store, "cosy-den", cut_id)


@pytest.mark.anyio
async def test_the_server_offers_the_whole_pipeline_as_tools(agent):
    names = {tool.name for tool in (await agent.list_tools()).tools}
    assert names == {
        "project_overview", "list_assets", "show_asset", "new_direction", "draw", "approve",
        "reject", "cut_into_pieces", "rename_piece", "drop_piece",
        "keep_pieces", "refine", "set_scene_pitch", "set_anchor",
        "export_bundle", "bring_in_picture", "suggest_objects", "show_suggestions",
        "pick_objects", "fix_pick", "unpick", "show_picks", "redraw_objects", "keep_scene",
        "rename_scene",
        "keep_reference", "start_pixel_cut", "cut_pixel_pieces",
    }


@pytest.mark.anyio
async def test_cutting_up_names_the_pieces_and_shows_them(agent, project_root):
    generation_id = await bring_in_floor(agent, project_root)
    # A recipe surface makes 32 tiles, which is what takes the cut-up past `PREVIEW_LIMIT`.
    result = await agent.call_tool(
        "cut_into_pieces", {"generation_id": generation_id, "surfaces": [FLOOR_SURFACE]},
    )
    assert result.is_error is False, said(result)
    store = Store(project_root)
    decomposition = load_decomposition(
        store, "cosy-den", store.list_decompositions("cosy-den")[0]
    )
    assert len(decomposition.cutouts) >= 3
    for cutout in decomposition.cutouts:
        assert cutout.name in said(result)
    # Guardian M2.2 #4: the cut-up has to actually produce more pieces
    # than `PREVIEW_LIMIT` here, or the two assertions below would also pass
    # with the cap deleted and prove nothing. Checking that first, then
    # pinning the shown count to the limit and requiring the "more not
    # shown" line, is what would have caught deviation 14's cap quietly
    # disappearing.
    assert len(decomposition.cutouts) > PREVIEW_LIMIT
    assert len(pictures(result)) == PREVIEW_LIMIT
    assert "more not shown" in said(result)


@pytest.mark.anyio
async def test_cutting_up_promotes_nothing_on_its_own(agent, project_root, object_picture):
    store, _, _ = await cut_up(agent, project_root, object_picture)
    assert store.list_assets("cosy-den") == []


@pytest.mark.anyio
async def test_a_piece_can_be_renamed_and_dropped(agent, project_root, object_picture):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    pieces = [c.name for c in load_decomposition(store, "cosy-den", cut_id).cutouts]

    assert (await agent.call_tool(
        "rename_piece",
        {"cut_id": cut_id, "piece": pieces[0], "name": "Grandad's Armchair"},
    )).is_error is False
    assert (await agent.call_tool(
        "drop_piece", {"cut_id": cut_id, "piece": pieces[-1]}
    )).is_error is False

    left = [c.name for c in load_decomposition(store, "cosy-den", cut_id).cutouts]
    assert "grandads-armchair" in left
    assert pieces[-1] not in left


@pytest.mark.anyio
async def test_a_sidecar_that_will_not_read_back_says_so_not_a_bad_request(
    agent, project_root, object_picture
):
    """Guardian M2.2 #5: pydantic's `ValidationError` is a `ValueError`, so it
    used to be caught by the same handler as a bad argument and told the
    agent "that was a bad request" — which an agent would retry forever with
    different arguments, never learning the file itself is broken."""
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    path = store.decomposition_dir("cosy-den", cut_id) / "decomposition.json"
    path.write_text("{}", encoding="utf-8")
    result = await agent.call_tool(
        "rename_piece", {"cut_id": cut_id, "piece": "sofa", "name": "settee"}
    )
    assert result.is_error is True
    assert "will not read back" in said(result)


@pytest.mark.anyio
async def test_keeping_the_pieces_makes_them_assets(agent, project_root, object_picture):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    names = [c.name for c in load_decomposition(store, "cosy-den", cut_id).cutouts]
    result = await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    assert result.is_error is False
    assert store.list_assets("cosy-den") == sorted(names)


@pytest.mark.anyio
async def test_refining_the_direction_writes_a_receipt_for_every_asset(
    agent, project_root, object_picture
):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    result = await agent.call_tool(
        "refine", {"preset": "crisp-sprite", "quick": {"dithering": "off"}}
    )
    assert result.is_error is False
    assert store.load_direction("cosy-den").refiner_defaults.preset == "crisp-sprite"
    for name in store.list_assets("cosy-den"):
        # A rebuilt prop is never sent to the refiner, but it still gets its receipt.
        assert load_report(store, "cosy-den", name).route == "rebuild"


@pytest.mark.anyio
async def test_refining_with_the_stand_in_warns_it_is_not_real_pixel_art(
    agent, project_root, object_picture
):
    """Guardian M2.2 #3: the plan's own deviation 10 lists where this warning
    has to be said, and named the agent as the face most likely to report the
    pixel art finished without it."""
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    result = await agent.call_tool("refine", {})
    assert result.is_error is False
    assert "not pixel art" in said(result)
    assert "--refiner cli" in said(result)


@pytest.mark.anyio
async def test_a_renamed_refiner_gets_no_stand_in_warning(project_root, object_picture):
    """The warning is keyed off `StubRefiner.name`, not off "any refiner that
    behaves like the stand-in" — a real, differently-named refiner must never
    be told it is copying pictures through unchanged."""
    server = create_server(
        project_root, service="fake", refiner=RenamedStubRefiner(), segmenter=StubSegmenter()
    )
    async with Client(server) as renamed:
        store, cut_id, _ = await cut_up(renamed, project_root, object_picture)
        await renamed.call_tool("keep_pieces", {"cut_id": cut_id})
        result = await renamed.call_tool("refine", {})
        assert result.is_error is False
        assert "not pixel art" not in said(result)


@pytest.mark.anyio
async def test_refining_one_asset_leaves_the_directions_defaults_alone(agent, project_root):
    store, record, _ = await drawn(agent, project_root)
    await agent.call_tool(
        "approve", {"generation_id": record.request.id, "keep_as_asset": True}
    )
    name = store.list_assets("cosy-den")[0]
    await agent.call_tool("refine", {"asset": name, "preset": "retro-game"})
    assert store.load_direction("cosy-den").refiner_defaults.preset is None
    assert load_report(store, "cosy-den", name).requested.preset == "retro-game"


@pytest.mark.anyio
async def test_settings_the_refiner_refuses_are_a_sentence(agent, project_root, object_picture):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    result = await agent.call_tool("refine", {"quick": {"background": "pick"}})
    assert result.is_error is True
    assert "backgroundColor" in said(result)


@pytest.mark.anyio
async def test_refining_with_a_negative_scale_is_a_sentence_and_leaves_defaults_alone(
    agent, project_root
):
    """Guardian M2.2 #2: `scale` had no floor, so a negative value with no
    `asset` named would have been written into `direction.json` as the art
    direction's permanent default — silently breaking every future run once a
    real refiner is in use. The field's own `ge=1` refuses it before
    `set_refiner_defaults` is ever called."""
    store = Store(project_root)
    before = store.load_direction("cosy-den").refiner_defaults
    result = await agent.call_tool("refine", {"scale": -4})
    assert result.is_error is True
    assert store.load_direction("cosy-den").refiner_defaults == before


@pytest.mark.anyio
async def test_refining_an_asset_that_does_not_exist_names_the_ones_that_do(
    agent, project_root, object_picture
):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    result = await agent.call_tool("refine", {"asset": "sofa"})
    assert result.is_error is True
    assert "no asset called" in said(result)


@pytest.mark.anyio
async def test_exporting_writes_a_bundle_and_says_what_went_in(agent, project_root, object_picture):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    await agent.call_tool("refine", {})
    result = await agent.call_tool("export_bundle", {"scales": [1, 2]})
    assert result.is_error is False

    bundles = list((project_root / "exports").iterdir())
    assert len(bundles) == 1
    assert bundles[0].name in said(result)
    manifest = Manifest.model_validate_json(
        (bundles[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert {entry.name for entry in manifest.entries} == set(store.list_assets("cosy-den"))


@pytest.mark.anyio
async def test_exporting_with_the_stand_in_warns_it_is_not_real_pixel_art(
    agent, project_root, object_picture
):
    """Guardian M2.2 #3: `export_bundle` said nothing about the stand-in at
    all before this — the one tool of the two that had no warning."""
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    await agent.call_tool("refine", {})
    result = await agent.call_tool("export_bundle", {})
    assert result.is_error is False
    assert "not pixel art" in said(result)


@pytest.mark.anyio
async def test_exporting_a_direction_with_nothing_refined_says_so(
    agent, project_root, object_picture
):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    result = await agent.call_tool("export_bundle", {})
    assert result.is_error is False
    assert "0 asset" in said(result)
    assert "refined" in said(result)


@pytest.mark.anyio
async def test_cutting_up_a_rejected_picture_is_a_sentence_naming_the_status(
    agent, project_root
):
    """Controller ruling 2: the "only an approved picture can be cut up" check
    now lives in `stages.decompose.decompose` itself, so this MCP tool gets it
    for free rather than having to invent its own copy — the same way the web
    page's route no longer needs one."""
    store, record, _ = await drawn(agent, project_root)
    rejected = await agent.call_tool(
        "reject", {"generation_id": record.request.id, "feedback": "too dark"}
    )
    assert rejected.is_error is False
    result = await agent.call_tool(
        "cut_into_pieces",
        {"generation_id": record.request.id, "surfaces": [FLOOR_SURFACE]},
    )
    assert result.is_error is True
    assert "rejected" in said(result)


# --- M3.6: sitting on the grid ---------------------------------------------


@pytest.mark.anyio
async def test_an_agent_can_start_a_direction_with_a_cell_size(agent, project_root):
    """Mutation proof (a): a tool that dropped `cell_size` on the floor would
    leave the new direction at the model's own default of 16, not 32."""
    result = await agent.call_tool(
        "new_direction",
        {"style": "Cold blue caves.", "name": "cold-cave", "donts": ["no gradients"],
         "cell_size": 32},
    )
    assert result.is_error is False, said(result)
    assert "cold-cave" in said(result)
    made = Store(project_root).load_direction("cold-cave")
    assert made.cell_size == 32
    assert made.donts == ["no gradients"]
    # The project already had a default (`project_root`'s "cosy-den"); a
    # second direction leaves it alone.
    assert Store(project_root).load_project().default_direction == "cosy-den"


@pytest.mark.anyio
@pytest.mark.parametrize("cell_size", [3, 129])
async def test_a_cell_size_outside_the_bound_is_refused_before_anything_is_written(
    agent, project_root, cell_size
):
    """`Field(ge=MIN_CELL_SIZE, le=MAX_CELL_SIZE)` on the argument: the SDK
    refuses it, as it refuses `draw`'s size, and no folder is made.

    `Direction` carries the same bound (`model.py`'s own
    `Field(ge=MIN_CELL_SIZE, le=MAX_CELL_SIZE)`), so deleting the tool's bound
    would still leave this refused — just later, and by a `ValidationError`
    that `readable_errors` reports as "will not read back". That is a
    different failure from the one this test is about, so it is ruled out by
    name.
    """
    result = await agent.call_tool(
        "new_direction", {"style": "Cold blue caves.", "name": "cold-cave", "cell_size": cell_size}
    )
    assert result.is_error is True
    assert "will not read back" not in said(result)
    assert Store(project_root).list_directions() == ["cosy-den"]


def a_prop(store):
    """The name and record of one asset that has an anchor — a prop."""
    for name in store.list_assets("cosy-den"):
        record = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
        if record.anchor is not None:
            return name, record
    raise AssertionError("the fake pipeline made no prop with an anchor")


@pytest.mark.anyio
async def test_setting_the_scene_pitch_changes_the_target(agent, project_root):
    """Mutation proof (e): a tool body that never called the core would leave
    the asset exactly as `approve` first wrote it."""
    store, record, _ = await drawn(agent, project_root)
    await agent.call_tool(
        "approve", {"generation_id": record.request.id, "keep_as_asset": True}
    )
    name = store.list_assets("cosy-den")[0]
    before = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert before.target == (64, 64)
    result = await agent.call_tool("set_scene_pitch", {"asset": name, "value": 2.0})
    assert result.is_error is False, said(result)
    assert "2.0" in said(result) and "set" in said(result)
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.pitch == 2.0
    assert after.pitch_source is PitchSource.SET
    assert after.target == (32, 32)


@pytest.mark.anyio
async def test_setting_the_scene_pitch_on_a_prop_is_a_sentence(agent, project_root, object_picture):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    name, _ = a_prop(store)
    result = await agent.call_tool("set_scene_pitch", {"asset": name, "value": 2.0})
    assert result.is_error is True
    assert "only a scene's pitch" in said(result)


@pytest.mark.anyio
async def test_setting_the_anchor_rewrites_the_asset(agent, project_root, object_picture):
    """`x` and `y` are chosen distinct on purpose. Mutation proof (d): a tool
    body that swapped them when calling the core would fail this on `(3, 9)`
    coming back as `(9, 3)`."""
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    name, before = a_prop(store)
    assert before.anchor == default_anchor(before.target)
    result = await agent.call_tool("set_anchor", {"asset": name, "x": 3, "y": 9})
    assert result.is_error is False, said(result)
    assert "(3, 9)" in said(result)
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.anchor == (3, 9)


@pytest.mark.anyio
async def test_an_anchor_outside_the_target_is_a_sentence_and_changes_nothing(
    agent, project_root, object_picture
):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    name, before = a_prop(store)
    result = await agent.call_tool("set_anchor", {"asset": name, "x": 99, "y": 99})
    assert result.is_error is True
    assert "99" in said(result)
    after = store.read_model(store.asset_dir("cosy-den", name) / "asset.json", AssetRecord)
    assert after.anchor == before.anchor


@pytest.mark.anyio
async def test_anchoring_an_asset_nobody_has_names_the_ones_that_do(
    agent, project_root, object_picture
):
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    known = store.list_assets("cosy-den")[0]
    result = await agent.call_tool("set_anchor", {"asset": "sofa", "x": 1, "y": 1})
    assert result.is_error is True
    assert "no asset called" in said(result)
    assert known in said(result)


@pytest.mark.anyio
async def test_every_name_argument_refuses_a_path_that_could_leave_the_project(
    agent, project_root, object_picture
):
    """Guardian M2.2 #6: only one argument of one tool was ever proven to
    refuse a path escape, which is nothing stopping the next tool from
    arriving without the same guard. This walks every string argument that
    becomes part of a file path — `direction` on every tool that takes one,
    `generation_id` on the tools that load a generation, `cut_id` and `piece`
    on the tools that load or name a cutout, and `asset` on the two tools
    that load one — and checks each is refused on its own, with every other
    argument left valid.

    `rename_piece`'s free-text `name` is deliberately not tested here: that
    value goes through `naming.typed_slug` inside `rename_cutout`, which squeezes
    free text into a safe name (refusing only an empty or too long one) rather
    than refusing it outright, so it never says "not a name PixelMagic uses".
    `new_direction`'s `name` is free text that goes through `naming.typed_slug`,
    like `rename_piece`'s, so it is not here either.
    """
    store, cut_id, _ = await cut_up(agent, project_root, object_picture)
    piece = load_decomposition(store, "cosy-den", cut_id).cutouts[0].name
    escape = "../escape"

    cases: list[tuple[str, dict]] = [
        # direction, on every tool that takes one
        ("list_assets", {"direction": escape}),
        ("show_asset", {"asset": "sofa", "direction": escape}),
        ("draw", {"subject": "a cosy den", "direction": escape}),
        ("approve", {"generation_id": "z", "direction": escape}),
        ("reject", {"generation_id": "z", "feedback": "too dark", "direction": escape}),
        ("cut_into_pieces", {"generation_id": "z", "direction": escape}),
        ("rename_piece", {"cut_id": cut_id, "piece": piece, "name": "y", "direction": escape}),
        ("drop_piece", {"cut_id": cut_id, "piece": piece, "direction": escape}),
        ("keep_pieces", {"cut_id": cut_id, "direction": escape}),
        ("refine", {"direction": escape}),
        ("export_bundle", {"direction": escape}),
        ("set_scene_pitch", {"asset": "sofa", "value": 2.0, "direction": escape}),
        ("set_anchor", {"asset": "sofa", "x": 1, "y": 1, "direction": escape}),
        # generation_id
        ("approve", {"generation_id": escape}),
        ("reject", {"generation_id": escape, "feedback": "too dark"}),
        ("cut_into_pieces", {"generation_id": escape}),
        # cut_id
        ("rename_piece", {"cut_id": escape, "piece": "x", "name": "y"}),
        ("drop_piece", {"cut_id": escape, "piece": "x"}),
        ("keep_pieces", {"cut_id": escape}),
        # piece names, against the real cut-up run
        ("rename_piece", {"cut_id": cut_id, "piece": escape, "name": "y"}),
        ("drop_piece", {"cut_id": cut_id, "piece": escape}),
        # asset
        ("show_asset", {"asset": escape}),
        ("refine", {"asset": escape}),
        ("set_scene_pitch", {"asset": escape, "value": 2.0}),
        ("set_anchor", {"asset": escape, "x": 1, "y": 1}),
    ]
    for tool, arguments in cases:
        result = await agent.call_tool(tool, arguments)
        assert result.is_error is True, f"{tool}({arguments}) should have been refused"
        # A run id is worded as an id; every other argument as a name.
        is_run_id = escape in (arguments.get("generation_id"), arguments.get("cut_id"))
        phrase = "not an id PixelMagic uses" if is_run_id else "not a name PixelMagic uses"
        assert phrase in said(result), f"{tool}({arguments}): {said(result)}"


@pytest.mark.anyio
async def test_an_agent_can_go_from_a_prompt_to_a_bundle(agent, project_root, object_picture):
    """The whole face, called once through, entirely offline.

    The same promise `test_end_to_end.py` makes for the command line and
    `test_web_end_to_end.py` makes for the page.
    """
    store = Store(project_root)

    assert "cosy-den" in said(await agent.call_tool("project_overview", {}))

    drawn_result = await agent.call_tool(
        "draw", {"subject": "a cosy den", "size": 120}
    )
    first = store.list_generations("cosy-den")[0]
    assert len(pictures(drawn_result)) == 1

    await agent.call_tool(
        "reject", {"generation_id": first, "feedback": "too dark"}
    )
    second = next(one for one in store.list_generations("cosy-den") if one != first)
    await agent.call_tool("approve", {"generation_id": second})

    cut_id = (await cut_up(agent, project_root, object_picture))[1]
    pieces = [c.name for c in load_decomposition(store, "cosy-den", cut_id).cutouts]
    await agent.call_tool(
        "rename_piece", {"cut_id": cut_id, "piece": pieces[0], "name": "Ice Shelf"}
    )
    await agent.call_tool("drop_piece", {"cut_id": cut_id, "piece": pieces[-1]})

    await agent.call_tool("keep_pieces", {"cut_id": cut_id})
    assert "ice-shelf" in store.list_assets("cosy-den")

    await agent.call_tool("refine", {"preset": "crisp-sprite"})
    assert "ice-shelf" in said(await agent.call_tool("list_assets", {}))
    assert len(pictures(await agent.call_tool("show_asset", {"asset": "ice-shelf"}))) == 2

    anchor_result = await agent.call_tool(
        "set_anchor", {"asset": "ice-shelf", "x": 3, "y": 9}
    )
    assert anchor_result.is_error is False, said(anchor_result)

    result = await agent.call_tool("export_bundle", {"scales": [1, 2]})
    assert result.is_error is False
    bundle = next(iter((project_root / "exports").iterdir()))
    manifest = Manifest.model_validate_json(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    assert {entry.name for entry in manifest.entries} == set(store.list_assets("cosy-den"))
    for entry in manifest.entries:
        assert entry.refiner_tool == "pixelmagic-grid-rebuild"
        if entry.name == "ice-shelf":
            assert entry.anchor == (3, 9)
            assert entry.pitch is not None


@pytest.mark.anyio
async def test_bringing_in_a_picture_says_where_it_came_from_and_shows_it(
    agent, project_root, tmp_path
):
    """`preview` used to read the drawn-only fields and would crash on an import."""
    source = tmp_path / "my-den.png"
    Image.new("RGB", (16, 16), "red").save(source)
    result = await agent.call_tool("bring_in_picture", {"path": str(source)})
    assert result.is_error is False, said(result)
    assert "imported from my-den.png" in said(result)
    assert len(pictures(result)) == 1


@pytest.mark.anyio
async def test_bringing_in_a_text_file_is_a_sentence_naming_the_file(
    agent, project_root, tmp_path
):
    source = tmp_path / "notes.png"
    source.write_text("not a picture")
    result = await agent.call_tool("bring_in_picture", {"path": str(source)})
    assert result.is_error is True
    assert "notes.png" in said(result)


@pytest.mark.anyio
async def test_an_agent_can_suggest_pick_redraw_and_keep_objects(
    project_root, object_picture
):
    """The picking walk through the server, offline: a fake service, the stub
    refiner and the stub segmenter.

    Catches: a tool raising a bare exception the agent sees only as "Error
    executing tool X", or a wiring slip (argument names, the segmenter or
    service not reaching the core) that only the server can show. Also the
    "nothing picked" refusals arriving as words that name the way forward.
    """
    server = create_server(
        project_root, service="fake", refiner=StubRefiner(), segmenter=StubSegmenter()
    )
    async with Client(server) as agent:
        store = Store(project_root)
        brought = await agent.call_tool("bring_in_picture", {"path": str(object_picture)})
        assert brought.is_error is False, said(brought)
        generation_id = store.list_generations("cosy-den")[0]

        # Props by the sheet are refused with the way forward.
        refused = await agent.call_tool("cut_into_pieces", {"generation_id": generation_id})
        assert refused.is_error is True
        assert "suggest_objects" in said(refused)

        suggested = await agent.call_tool("suggest_objects", {"generation_id": generation_id})
        assert suggested.is_error is False, said(suggested)
        assert len(pictures(suggested)) == 1
        cut_id = store.list_decompositions("cosy-den")[0]

        page = await agent.call_tool("show_suggestions", {"cut_id": cut_id})
        assert page.is_error is False, said(page)
        assert "1: box" in said(page)
        assert len(pictures(page)) == 1
        past = await agent.call_tool("show_suggestions", {"cut_id": cut_id, "page": 9})
        assert past.is_error is True and "no page 9" in said(past)

        # Nothing picked yet: redraw says so in words.
        nothing = await agent.call_tool("redraw_objects", {"cut_id": cut_id})
        assert nothing.is_error is True
        assert "nothing is picked" in said(nothing)
        assert "nothing picked yet" in said(await agent.call_tool("show_picks", {"cut_id": cut_id}))

        picked = await agent.call_tool("pick_objects", {"cut_id": cut_id, "numbers": [1, 2]})
        assert picked.is_error is False, said(picked)
        assert "suggestion 1 is now pick 1" in said(picked)
        shown = await agent.call_tool("show_picks", {"cut_id": cut_id})
        assert "(named at redraw)" in said(shown) and len(pictures(shown)) == 1
        gone = await agent.call_tool("unpick", {"cut_id": cut_id, "pick": 2})
        assert "not used again" in said(gone)

        redrawn = await agent.call_tool("redraw_objects", {"cut_id": cut_id})
        assert redrawn.is_error is False, said(redrawn)
        assert len(pictures(redrawn)) == 1
        assert "with fake" in said(redrawn)  # who drew is named, so who is paid is seen

        kept = await agent.call_tool("keep_pieces", {"cut_id": cut_id})
        assert kept.is_error is False, said(kept)
        (asset,) = store.list_assets("cosy-den")
        record = store.read_model(store.asset_dir("cosy-den", asset) / "asset.json", AssetRecord)
        assert record.segmenter == "stub"


@pytest.mark.anyio
async def test_fixing_a_redrawn_pick_says_how_to_redraw_it(project_root, object_picture):
    """Catches: a fix that silently leaves the agent believing the piece it
    already redrew now matches the new outline (the core leaves the piece
    alone on purpose), and a wrong-shaped click never reaching the core.
    """
    server = create_server(
        project_root, service="fake", refiner=StubRefiner(), segmenter=StubSegmenter()
    )
    async with Client(server) as agent:
        store = Store(project_root)
        await agent.call_tool("bring_in_picture", {"path": str(object_picture)})
        generation_id = store.list_generations("cosy-den")[0]
        await agent.call_tool("suggest_objects", {"generation_id": generation_id})
        cut_id = store.list_decompositions("cosy-den")[0]
        await agent.call_tool(
            "pick_objects", {"cut_id": cut_id, "numbers": [1], "name": "lamp"}
        )
        # An object no suggestion covers is picked by a click (the small
        # object at 300,180 is suggestion 7; here it is reached by point).
        by_point = await agent.call_tool(
            "pick_objects",
            {"cut_id": cut_id, "points": [{"x": 310, "y": 190, "this": True}]},
        )
        assert by_point.is_error is False, said(by_point)
        assert "picked 1 point as pick 2, outline box (300, 180, 340, 210)" in said(by_point)

        before = await agent.call_tool(
            "fix_pick", {"cut_id": cut_id, "pick": 1, "points": [{"x": 30, "y": 25, "this": True}]}
        )
        assert before.is_error is False, said(before)
        assert "unchanged" not in said(before)

        await agent.call_tool("redraw_objects", {"cut_id": cut_id})
        after = await agent.call_tool(
            "fix_pick", {"cut_id": cut_id, "pick": 1, "points": [{"x": 30, "y": 25, "this": True}]}
        )
        assert after.is_error is False, said(after)
        assert "unchanged" in said(after)
        assert "only=['lamp']" in said(after)


@pytest.mark.anyio
async def test_picking_with_a_picture_id_says_it_is_a_picture_id_and_suggest_objects_is_next(
    agent, project_root, object_picture
):
    """Catches: an MCP picking tool answering a picture id with a bare missing-file message."""
    store = Store(project_root)
    await agent.call_tool("bring_in_picture", {"path": str(object_picture)})
    generation_id = store.list_generations("cosy-den")[0]
    for tool, arguments in (
        ("pick_objects", {"numbers": [1]}),
        ("show_picks", {}),
    ):
        result = await agent.call_tool(tool, {"cut_id": generation_id, **arguments})
        assert result.is_error is True
        assert "picture id" in said(result) and "suggest_objects" in said(result)


@pytest.mark.anyio
async def test_an_agent_can_cut_props_from_a_pixel_art_picture_to_a_bundle(
    project_root, object_picture
):
    """The pixel-cut road through the server, offline: no image service is set.

    Catches: the tools not reaching the core (argument names, the segmenter), a pixel-cut
    prop leaving the road as something other than a pixel-cut entry in the manifest, the
    refine reply passing it off as refined, a fix-up on a cut piece sending the agent to
    redraw_objects (which refuses a pixel cut), and a bad typed size reaching the agent without
    the core's sentence.
    """
    server = create_server(project_root, refiner=StubRefiner(), segmenter=StubSegmenter())
    async with Client(server) as agent:
        store = Store(project_root)
        await agent.call_tool("bring_in_picture", {"path": str(object_picture)})
        generation_id = store.list_generations("cosy-den")[0]

        too_big = await agent.call_tool(
            "start_pixel_cut", {"generation_id": generation_id, "size": 5000}
        )
        assert too_big.is_error is True
        assert "shorter side" in said(too_big)
        assert store.list_decompositions("cosy-den") == []

        opened = await agent.call_tool(
            "start_pixel_cut", {"generation_id": generation_id, "size": 1}
        )
        assert opened.is_error is False, said(opened)
        cut_id = store.list_decompositions("cosy-den")[0]
        picked = await agent.call_tool(
            "pick_objects", {"cut_id": cut_id, "box": [8, 8, 52, 42], "name": "lamp"}
        )
        assert picked.is_error is False, said(picked)
        cut = await agent.call_tool("cut_pixel_pieces", {"cut_id": cut_id})
        assert cut.is_error is False, said(cut)
        assert "lamp" in said(cut) and len(pictures(cut)) == 1

        fixed = await agent.call_tool(
            "fix_pick", {"cut_id": cut_id, "pick": 1, "box": [8, 8, 52, 42]}
        )
        assert "cut_pixel_pieces" in said(fixed) and "redraw_objects" not in said(fixed)

        kept = await agent.call_tool("keep_pieces", {"cut_id": cut_id})
        assert kept.is_error is False, said(kept)
        refined = await agent.call_tool("refine", {})
        assert "1 pixel-cut prop(s) were not sent to the refiner" in said(refined)
        assert (await agent.call_tool("export_bundle", {})).is_error is False

        (bundle,) = (project_root / "exports").iterdir()
        manifest = Manifest.model_validate_json(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        (entry,) = [e for e in manifest.entries if e.name == "lamp"]
        assert (entry.refiner_tool, entry.route) == ("pixelmagic-pixel-cut", "pixel-cut")
