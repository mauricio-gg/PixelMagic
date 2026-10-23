import pytest

from pixelmagic.naming import (
    FALLBACK_DIRECTION_NAME,
    FALLBACK_NAME,
    UnusableName,
    check_name,
    describe_cutout,
    kebab_case,
    parse_surface,
    slug,
    suggest_direction_name,
    unique_name,
)
from pixelmagic.providers.base import ImageCapabilities
from pixelmagic.providers.fake import FakeImageService
from pixelmagic.stages.recipes import RECIPE_KINDS


class ExplodingService(FakeImageService):
    def describe(self, image_path, question):
        raise RuntimeError("no vision here")


def test_kebab_case_lowercases_and_joins_with_hyphens():
    assert kebab_case("A Wooden  Chair") == "a-wooden-chair"


def test_kebab_case_strips_punctuation_and_quotes():
    assert kebab_case('"Grandma\'s Rug!"') == "grandmas-rug"


def test_kebab_case_keeps_digits():
    assert kebab_case("Tile 3 Variant") == "tile-3-variant"


def test_kebab_case_trims_to_four_words():
    assert kebab_case("one two three four five") == "one-two-three-four"


def test_kebab_case_falls_back_on_empty_input():
    assert kebab_case("   ") == FALLBACK_NAME
    assert kebab_case("!!!", fallback="tile") == "tile"


def test_kebab_case_caps_a_single_unbroken_token_at_sixty_characters():
    huge = "a" * 300
    result = kebab_case(huge, max_words=1)
    assert result == "a" * 60
    assert len(result) == 60


def test_kebab_case_length_cap_does_not_leave_a_dangling_hyphen():
    # 14 "a"s + the joining hyphen is exactly 15 characters, so a max_length
    # of 15 cuts right on the hyphen; the trailing hyphen must be dropped.
    result = kebab_case("aaaaaaaaaaaaaa bbbb", max_words=2, max_length=15)
    assert result == "aaaaaaaaaaaaaa"
    assert not result.endswith("-")


def test_unique_name_passes_through_when_free():
    assert unique_name("chair", set()) == "chair"


def test_unique_name_suffixes_collisions():
    assert unique_name("chair", {"chair"}) == "chair-2"
    assert unique_name("chair", {"chair", "chair-2"}) == "chair-3"


def test_describe_cutout_returns_the_service_label(tmp_path):
    service = FakeImageService(descriptions=["Wooden Chair"])
    assert describe_cutout(service, tmp_path / "a.png") == "Wooden Chair"


def test_describe_cutout_falls_back_when_the_service_has_no_vision(tmp_path):
    service = FakeImageService(capabilities=ImageCapabilities(vision=False))
    assert describe_cutout(service, tmp_path / "a.png") == FALLBACK_NAME


def test_describe_cutout_falls_back_when_the_service_raises(tmp_path):
    assert describe_cutout(ExplodingService(), tmp_path / "a.png") == FALLBACK_NAME


def test_suggest_direction_name_kebab_cases_the_answer():
    service = FakeImageService(descriptions=["Cosy Lamplit Den"])
    assert suggest_direction_name(service, "Warm SNES interiors.") == "cosy-lamplit-den"


def test_suggest_direction_name_falls_back_when_the_service_raises():
    assert (
        suggest_direction_name(ExplodingService(), "Warm SNES interiors.")
        == FALLBACK_DIRECTION_NAME
    )


def test_check_name_lets_our_own_names_through():
    assert check_name("cosy-den", "direction") == "cosy-den"
    assert check_name("20260903-191233", "generation") == "20260903-191233"
    assert check_name("20260903-191233-2", "generation") == "20260903-191233-2"


@pytest.mark.parametrize(
    "bad",
    # The last two are why this uses `fullmatch`: `re.match` with a `$` accepts
    # a trailing newline, and `directions/cosy-den\n` is a second direction
    # nobody can see.
    ["..", "../etc", "a/b", "a\\b", "a.b", "", "-leading", "with space", "café",
     "cosy-den\n", "cosy-den\t"],
)
def test_check_name_refuses_anything_that_could_leave_the_project_folder(bad):
    with pytest.raises(ValueError, match="direction"):
        check_name(bad, "direction")


def test_an_unusable_name_is_a_value_error_a_face_can_catch():
    """A face that only knows `ValueError` still catches this; one that wants
    to tell "this is not one of our names" apart from a corrupt sidecar's
    `ValidationError` (also a `ValueError`) can catch `UnusableName` alone.
    """
    assert issubclass(UnusableName, ValueError)
    with pytest.raises(UnusableName):
        check_name("Warm Den", "direction")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Café Noir", "cafe-noir"),
        ("Über Chair", "uber-chair"),
        ("Piñata", "pinata"),
        ("Ångström Lamp", "angstrom-lamp"),
        ("naïve façade", "naive-facade"),
        # NFKD cannot split these: the stroke or the ligature is part of the
        # letter, not a mark sat on it, so without the pre-map `Straße` folds
        # to `strae` and `Ørsted` to `rsted` — the same letter loss again.
        ("Straße", "strasse"),
        # U+1E9E has no NFKD decomposition either, and it is the one entry in
        # the table that was missing its uppercase partner — so this is the
        # exact bug the table exists to prevent, still reachable in caps.
        ("STRAẞE", "strasse"),
        ("Ørsted Lamp", "orsted-lamp"),
        ("Łódź Tile", "lodz-tile"),
        ("Æther", "aether"),
        ("Þor", "thor"),
    ],
)
def test_accents_fold_down_instead_of_being_dropped(text, expected):
    """`Café` lost its e before this; a letter gone is worse than an accent gone."""
    assert kebab_case(text) == expected


@pytest.mark.parametrize("text", ["日本語", "", "   ", "!!!", "…"])
def test_slug_says_when_nothing_usable_is_left(text):
    assert slug(text) is None


def test_kebab_case_still_falls_back_for_a_services_answer():
    """Naming a cutout is a convenience and must never stop a decomposition."""
    assert kebab_case("日本語") == FALLBACK_NAME
    assert kebab_case("日本語", fallback="thing") == "thing"


def test_slug_keeps_everything_kebab_case_keeps():
    assert slug("A Wooden Chair, Please") == "a-wooden-chair-please"
    assert slug("A Wooden Chair, Please", max_words=3) == "a-wooden-chair"
    assert slug("x" * 200) == "x" * 60


def test_a_recipe_surface_reads_its_shapes_and_box_and_the_old_point_form_is_refused():
    surface = parse_surface(
        "plaster wall:plaster ; box 212,285 180,240; shape 140,400 200,230 330,110 520,70;"
        " shape 1100,440 1100,330 -5,285"
    )
    assert (surface.text, surface.name, surface.kind) == ("plaster wall", "plaster-wall", "plaster")
    assert surface.box == (180, 240, 212, 285)
    assert surface.shapes == (
        ((140, 400), (200, 230), (330, 110), (520, 70)),
        ((1100, 440), (1100, 330), (-5, 285)),
    )
    assert parse_surface("wood floor:wood; box 1,1 40,40; shape 1,1 5,1 5,5;;") == parse_surface(
        "wood floor:wood; box 1,1 40,40; shape 1,1 5,1 5,5"
    )

    for typed in (
        "wood floor:wood@412,630",
        "wood floor:wood",
        "wood floor",
        "wood floor:wood; shape 1,1 5,1 5,5",
        "wood floor:wood; box 1,1 40,40",
        "wood floor:wood; box 1,1 40,40; box 2,2 50,50; shape 1,1 5,1 5,5",
        "wood floor:wood; box 1,1 40,40; shape 1,1 5,1",
        "wood floor:wood; box 1,1 40; shape 1,1 5,1 5,5",
        "wood floor:wood; box 1,1 40,40; blob 1,1 5,1 5,5",
        "wood floor:oak; box 1,1 40,40; shape 1,1 5,1 5,5",
    ):
        with pytest.raises(UnusableName):
            parse_surface(typed)
    with pytest.raises(UnusableName, match="the point after @ is gone"):
        parse_surface("wood floor:wood@412,630; box 1,1 40,40; shape 1,1 5,1 5,5")


def test_a_surface_name_over_four_words_is_refused_not_shortened():
    """Catches: a typed surface name losing words (`old stone castle hall floor` filed
    as `old-stone-castle-hall`); folding case and accents must still be allowed."""
    with pytest.raises(UnusableName, match="old-stone-castle-hall"):
        parse_surface(
            "Old Stone Castle Hall Floor:stone; box 1,1 40,40; shape 1,1 5,1 5,5"
        )
    assert parse_surface(
        "Café Noir:stone; box 1,1 40,40; shape 1,1 5,1 5,5"
    ).name == "cafe-noir"
    # Folding can lengthen text; that is not shortening and must not be refused.
    assert parse_surface(
        "Oﬃce:stone; box 1,1 40,40; shape 1,1 5,1 5,5"
    ).name == "office"


@pytest.mark.parametrize("typed", ["red rug:bordered", "red rug:bordered "])
def test_a_bordered_surface_is_refused_with_the_recipe_way_forward(typed):
    """Catches: the removed `:bordered` marker still being accepted, or (with a trailing
    space) slipping through as a plain surface named `red-rug-bordered`; and a refusal that
    does not tell the person how to type a surface now."""
    with pytest.raises(UnusableName) as refusal:
        parse_surface(typed)
    sentence = str(refusal.value)
    assert "bordered surfaces are gone" in sentence
    assert "box" in sentence and "shape" in sentence
    assert all(kind in sentence for kind in RECIPE_KINDS)


def test_the_word_bordered_without_a_colon_is_not_the_marker():
    """A surface can be *about* a border without being the removed marker — only the
    exact trailing `:bordered` is refused as bordered, so the plain words are
    refused like any other kindless surface."""
    with pytest.raises(UnusableName, match="needs a recipe kind"):
        parse_surface("cobbled bordered path")


def test_an_empty_surface_is_refused():
    with pytest.raises(UnusableName):
        parse_surface("")
    with pytest.raises(UnusableName):
        parse_surface("   ")


def test_a_scene_name_drops_filler_and_keeps_at_most_three_words():
    from pixelmagic.naming import scene_name

    assert scene_name("a cozy den with a stove and a bed") == "cozy-den-stove"


def test_a_scene_name_from_only_filler_or_another_script_still_exists():
    from pixelmagic.naming import scene_name

    assert scene_name("the of and") == "the-of-and"
    assert scene_name("温かい部屋") == "scene"


def test_an_imported_files_stem_keeps_its_words():
    from pixelmagic.naming import scene_name

    assert scene_name("My Den_v2") == "my-den-v2"
    assert scene_name("image") == "image"
