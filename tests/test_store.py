from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pixelmagic.model import Direction, RefinerSettings
from pixelmagic.naming import UnusableName
from pixelmagic.store import AlreadyExists, Store, claim_id, new_id

NOW = datetime(2026, 9, 3, 19, 12, 33, tzinfo=UTC)


def test_project_round_trips(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    loaded = store.load_project()
    assert loaded.name == "DemoGame"
    assert loaded.created == NOW
    assert (tmp_path / "pixelmagic.json").exists()


def test_direction_round_trips_with_nested_refiner_settings(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    store.create_direction(
        Direction(
            name="cosy-den",
            created=NOW,
            style="Warm SNES interiors, soft lamp light, chunky outlines.",
            donts=["no gradients"],
            refiner_defaults=RefinerSettings(preset="crisp-sprite", quick={"dithering": "off"}),
        )
    )
    loaded = store.load_direction("cosy-den")
    assert loaded.style.startswith("Warm SNES interiors")
    assert loaded.donts == ["no gradients"]
    assert loaded.refiner_defaults.preset == "crisp-sprite"
    assert loaded.refiner_defaults.quick == {"dithering": "off"}
    assert store.list_directions() == ["cosy-den"]


def test_creating_a_direction_makes_its_folders(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    store.create_direction(Direction(name="cosy-den", created=NOW, style="Warm."))
    assert store.references_dir("cosy-den").is_dir()
    assert store.generations_dir("cosy-den").is_dir()
    assert store.decompositions_dir("cosy-den").is_dir()
    assert store.assets_dir("cosy-den").is_dir()


def test_loading_a_missing_direction_raises(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    with pytest.raises(FileNotFoundError):
        store.load_direction("nope")


def test_list_assets_only_counts_folders_with_a_source_image(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    store.create_direction(Direction(name="cosy-den", created=NOW, style="Warm."))
    for name, has_source in (("wooden-chair", True), ("round-rug", True), ("empty", False)):
        folder = store.asset_dir("cosy-den", name)
        folder.mkdir(parents=True, exist_ok=True)
        if has_source:
            (folder / "source.png").write_bytes(b"")
    assert store.list_assets("cosy-den") == ["round-rug", "wooden-chair"]


def test_list_assets_is_empty_before_anything_is_promoted(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    assert store.list_assets("cosy-den") == []


def test_new_id_is_a_utc_timestamp_slug(tmp_path):
    assert new_id(tmp_path, now=NOW) == "20260903-191233"


def test_new_id_uniquifies_within_a_parent(tmp_path):
    (tmp_path / "20260903-191233").mkdir()
    assert new_id(tmp_path, now=NOW) == "20260903-191233-2"
    (tmp_path / "20260903-191233-2").mkdir()
    assert new_id(tmp_path, now=NOW) == "20260903-191233-3"


def test_a_second_claim_in_the_same_second_gets_its_own_existing_folder(tmp_path):
    """Catches: two runs in one second taking the same id and writing into one folder."""
    first = claim_id(tmp_path / "decompositions", now=NOW)
    second = claim_id(tmp_path / "decompositions", now=NOW)
    assert (first, second) == ("20260903-191233", "20260903-191233-2")
    assert (tmp_path / "decompositions" / second).is_dir()


def test_a_sidecar_is_written_whole_and_leaves_no_scratch_file(tmp_path):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    store.create_direction(Direction(name="cosy-den", created=NOW, style="Warm."))
    # Written twice: the second write replaces the first, and neither leaves a
    # half-written file or a leftover *.tmp beside the sidecar.
    store.save_direction(Direction(name="cosy-den", created=NOW, style="Colder."))
    folder = store.direction_dir("cosy-den")
    assert list(folder.glob("*.tmp")) == []
    assert store.load_direction("cosy-den").style == "Colder."


def test_a_failed_write_leaves_the_old_sidecar_and_no_scratch_file(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    store.create_direction(Direction(name="cosy-den", created=NOW, style="Warm."))

    def full_disk(self, *args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr("pathlib.Path.write_text", full_disk)
    with pytest.raises(OSError, match="no space"):
        store.save_direction(Direction(name="cosy-den", created=NOW, style="Colder."))
    monkeypatch.undo()
    assert list(store.direction_dir("cosy-den").glob("*.tmp")) == []
    assert store.load_direction("cosy-den").style == "Warm."


def test_a_naive_timestamp_is_refused(tmp_path):
    """A `datetime` with no timezone would be recorded as if it were UTC."""
    store = Store(tmp_path)
    with pytest.raises(ValidationError):
        store.create_project("DemoGame", now=datetime(2026, 9, 3, 19, 12, 33))
    with pytest.raises(ValidationError):
        Direction(name="cosy-den", created=datetime(2026, 9, 3, 19, 12, 33), style="Warm.")


def new_store(tmp_path, *, direction="cosy-den"):
    """A project with one direction and a default pointer, ready to poke at."""
    store = Store(tmp_path)
    store.create_project("DemoGame", now=NOW)
    store.create_direction(
        Direction(name=direction, created=NOW, style="Warm SNES interiors.")
    )
    project = store.load_project()
    project.default_direction = direction
    store.save_project(project)
    return store


def test_creating_a_project_twice_is_refused_and_keeps_the_first(tmp_path):
    store = new_store(tmp_path)
    with pytest.raises(AlreadyExists, match="already exists"):
        store.create_project("SomethingElse", now=NOW)
    assert store.load_project().name == "DemoGame"
    assert store.load_project().default_direction == "cosy-den"


def test_creating_a_direction_twice_is_refused_and_keeps_the_first(tmp_path):
    store = new_store(tmp_path)
    with pytest.raises(AlreadyExists, match="cosy-den"):
        store.create_direction(
            Direction(name="cosy-den", created=NOW, style="Something else entirely.")
        )
    assert store.load_direction("cosy-den").style == "Warm SNES interiors."


def test_a_refused_direction_does_not_disturb_the_one_that_is_there(tmp_path):
    """The guard runs before any mkdir, so a refusal cannot touch the folder."""
    store = new_store(tmp_path)
    (store.references_dir("cosy-den") / "keep.png").write_bytes(b"pretend png")
    with pytest.raises(AlreadyExists):
        store.create_direction(Direction(name="cosy-den", created=NOW, style="a"))
    assert (store.references_dir("cosy-den") / "keep.png").exists()


def test_already_exists_is_a_file_exists_error():
    """A face that catches the OSError family catches this without an import."""
    assert issubclass(AlreadyExists, FileExistsError)


def test_renaming_a_direction_moves_the_folder_and_its_contents(tmp_path):
    store = new_store(tmp_path)
    (store.references_dir("cosy-den") / "keep.png").write_bytes(b"pretend png")
    renamed = store.rename_direction("cosy-den", "warm-den")
    assert renamed.name == "warm-den"
    assert store.list_directions() == ["warm-den"]
    # Off disk and not the value that was returned: the sidecar's own `name`
    # field is the one thing a folder move does not rewrite by itself.
    assert store.load_direction("warm-den").name == "warm-den"
    assert store.load_direction("warm-den").style == "Warm SNES interiors."
    assert (store.references_dir("warm-den") / "keep.png").exists()
    assert not store.direction_dir("cosy-den").exists()


def test_renaming_the_default_direction_repoints_the_project(tmp_path):
    store = new_store(tmp_path)
    store.rename_direction("cosy-den", "warm-den")
    assert store.load_project().default_direction == "warm-den"


def test_renaming_another_direction_leaves_the_default_pointer_alone(tmp_path):
    store = new_store(tmp_path)
    store.create_direction(Direction(name="cold-cave", created=NOW, style="Cold."))
    store.rename_direction("cold-cave", "icy-cave")
    assert store.load_project().default_direction == "cosy-den"


def test_renaming_onto_a_name_that_exists_is_refused(tmp_path):
    store = new_store(tmp_path)
    store.create_direction(Direction(name="cold-cave", created=NOW, style="Cold."))
    with pytest.raises(AlreadyExists, match="cold-cave"):
        store.rename_direction("cosy-den", "cold-cave")
    assert store.load_direction("cosy-den").style == "Warm SNES interiors."


def test_renaming_a_direction_that_does_not_exist_says_so(tmp_path):
    store = new_store(tmp_path)
    with pytest.raises(FileNotFoundError, match="nope"):
        store.rename_direction("nope", "whatever")


@pytest.mark.parametrize(
    "bad", ["", "Warm Den", "../escape", "a/b", "warm den", "日本語"]
)
def test_renaming_to_a_name_the_store_could_not_find_again_is_refused(tmp_path, bad):
    store = new_store(tmp_path)
    with pytest.raises(ValueError):
        store.rename_direction("cosy-den", bad)
    assert store.list_directions() == ["cosy-den"]


def test_renaming_from_a_name_the_store_could_not_find_again_is_refused(tmp_path):
    """`old` becomes a path too, so it is refused by name, not by absence."""
    store = new_store(tmp_path)
    with pytest.raises(ValueError):
        store.rename_direction("../escape", "warm-den")


def test_a_direction_json_that_will_not_parse_leaves_everything_where_it_was(tmp_path):
    """The other half of "must not half-finish": a sidecar that will not read.

    The sidecar is read and validated *before* the folder moves, so a broken
    one is a refusal and not a direction in two places. Moving first would
    leave the folder under the new name with `direction.json` still saying the
    old one, and anything still holding the old `Direction` would then
    `save_direction` the old folder back into existence.
    """
    store = new_store(tmp_path)
    store.direction_file("cosy-den").write_text("{not json at all")
    with pytest.raises(ValueError):
        store.rename_direction("cosy-den", "warm-den")
    assert store.list_directions() == ["cosy-den"]
    assert not store.direction_dir("warm-den").exists()


def test_renaming_works_in_a_folder_with_no_project_sidecar(tmp_path):
    """The rename must not half-finish on a folder tree nobody made properly."""
    store = Store(tmp_path)
    store.create_direction(
        Direction(name="cosy-den", created=NOW, style="Warm SNES interiors.")
    )
    assert store.rename_direction("cosy-den", "warm-den").name == "warm-den"
    assert store.list_directions() == ["warm-den"]


def test_listing_generations_and_decompositions_puts_the_newest_first(tmp_path):
    store = new_store(tmp_path)
    for stamp in ("20260903-100000", "20260903-110000"):
        folder = store.generation_dir("cosy-den", stamp)
        folder.mkdir(parents=True)
        (folder / "request.json").write_text("{}")
        other = store.decomposition_dir("cosy-den", stamp)
        other.mkdir(parents=True)
        (other / "decomposition.json").write_text("{}")
    assert store.list_generations("cosy-den") == ["20260903-110000", "20260903-100000"]
    assert store.list_decompositions("cosy-den") == ["20260903-110000", "20260903-100000"]


def test_listing_a_direction_with_nothing_in_it_yet_is_empty(tmp_path):
    store = new_store(tmp_path)
    assert store.list_generations("cosy-den") == []
    assert store.list_decompositions("cosy-den") == []


def test_a_half_written_generation_folder_is_not_listed(tmp_path):
    """The marker file is what makes a folder a record, not the folder itself."""
    store = new_store(tmp_path)
    store.generation_dir("cosy-den", "20260903-100000").mkdir(parents=True)
    assert store.list_generations("cosy-den") == []


@pytest.mark.parametrize("method", ["generation_dir", "decomposition_dir"])
def test_a_mistyped_run_id_is_refused_with_the_shape_of_an_id(tmp_path, method):
    """A timestamp typed with an underscore is told what an id looks like."""
    with pytest.raises(UnusableName) as refusal:
        getattr(Store(tmp_path), method)("cosy-den", "20260903_191233")
    assert "20260903-191233" in str(refusal.value)
    assert "lowercase words" not in str(refusal.value)
