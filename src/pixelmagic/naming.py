"""Turning free text into names that are safe to use as folder names.

Everything a service says back — a cutout label, an art-direction name — is
free-form prose. It becomes a folder name, a PNG filename and a manifest key,
so it has to be squeezed into lowercase kebab-case first, and a service that
cannot answer must never be able to stop the pipeline.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from pixelmagic.providers.base import ImageService, load_direction_name_question
from pixelmagic.stages.recipes import RECIPE_KINDS

#: What a service is asked about a single cut-out piece.
CUTOUT_QUESTION = (
    "Name this single game object in two to four plain English words. "
    "Reply with the name only — no punctuation, no explanation."
)

#: Used when a service cannot name a cutout. `unique_name` will number these.
FALLBACK_NAME = "item"

#: Used when a service cannot invent an art-direction name.
FALLBACK_DIRECTION_NAME = "new-direction"

_UNSAFE = re.compile(r"[^a-z0-9\s-]")
_SEPARATORS = re.compile(r"[\s-]+")

#: What one of our own names may look like: lowercase alphanumeric words
#: joined by single hyphens. Direction, asset and cutout names are kebab-case;
#: generation and decomposition ids are timestamp slugs with an optional `-2`
#: suffix, which this matches too. None of them can hold a dot, a slash, a
#: backslash or a space, which is what makes a value that matches this safe to
#: hand to `Store` — the store does not sanitise a direction name (see
#: `Store.direction_dir`; run ids are checked in `generation_dir` and
#: `decomposition_dir`), so that check has to happen where a name arrives
#: from outside the program.
#:
#: This is the *only* definition of a usable name a face reads out of a URL or
#: a tool argument. `Store.rename_direction` calls `check_name` rather than
#: spelling a second one, because two whitelists that disagree is how a face
#: lets a name through that the store then refuses with `UnusableName`.
#: `Store.create_direction` still takes whatever its caller hands it — see
#: deviation 4.
#:
#: Anchored with `\A`/`\Z` rather than left to `fullmatch`, because the pattern
#: itself is exported: an unanchored `SAFE_NAME.match("a/b")` is a match on
#: `"a"`, so a face reaching for the pattern instead of the function would get
#: a guard that accepts a path separator. `\Z` and not `$`, so the anchors and
#: `fullmatch` agree about a trailing newline.
SAFE_NAME = re.compile(r"\A[a-z0-9]+(-[a-z0-9]+)*\Z")


class UnusableName(ValueError):
    """A value that is not one of PixelMagic's own names.

    A `ValueError`, so a face that catches that family at its boundary still
    catches this. A class of its own, so a caller can catch *this* rather
    than a bare `ValueError`: a pydantic `ValidationError` is a `ValueError`
    too, and a corrupt sidecar must stay a traceback rather than being read
    as "the name was bad" — the same argument `registry.RegistryError` makes
    for `cli.load_service`.
    """


#: Latin letters that `NFKD` cannot take apart, and what they are worth in
#: ASCII. A stroke or a ligature is part of the letter rather than a mark sat
#: on top of one, so decomposition leaves them whole and the encode step then
#: drops them entirely — `Straße` would become `strae` and `Ørsted` would
#: become `rsted`, which is the same "a letter gone is worse than an accent
#: gone" failure this function exists to fix. These are the ones that turn up
#: in European product and place names — deliberately the common
#: Western-European set. A letter outside it (Turkish dotless `ı`, say) falls
#: through to the honest refusal `slug` gives rather than being silently
#: dropped.
_LATIN_EQUIVALENTS = str.maketrans(
    {
        "ß": "ss", "ẞ": "SS", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L",
        "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
        "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH",
        "đ": "d", "Đ": "D", "ħ": "h", "Ħ": "H",
    }
)


def _fold_to_ascii(text: str) -> str:
    """Rewrite accented Latin letters as their plain ASCII equivalents.

    Two steps, because `NFKD` alone is not enough. It splits `é` into `e` and a
    combining accent, and the accent has no ASCII form so it is dropped —
    leaving the letter. Without that the whole character was dropped instead,
    so `Café` became `caf`: a name with a letter missing, which is worse than
    one with an accent missing.

    But `NFKD` cannot split a letter whose mark is *part of* it: `ß`, `ø`, `ł`,
    `æ`, `þ` and their kin come through whole and are then dropped by the
    encode, which is the same failure again. `_LATIN_EQUIVALENTS` is applied
    first so those become the letters they stand for.

    Nothing outside the Latin alphabet has a *letter* form in ASCII, so a name
    in another script comes back empty here rather than mangled, and `slug`
    reports that honestly. (Compatibility decomposition also unpacks a few
    symbols and full-width forms into their ASCII spellings — `™` to `TM`,
    `２` to `2` — which is harmless and not what this function is for.)
    """
    folded = unicodedata.normalize("NFKD", text.translate(_LATIN_EQUIVALENTS))
    return folded.encode("ascii", "ignore").decode("ascii")


#: The most words and characters a name PixelMagic keeps has. `slug` and
#: `kebab_case` cut to these; `typed_slug` refuses what would be cut and says them.
MAX_NAME_WORDS = 4
MAX_NAME_LENGTH = 60


def slug(
    text: str, *, max_words: int = MAX_NAME_WORDS, max_length: int = MAX_NAME_LENGTH
) -> str | None:
    """Squeeze free text into a safe folder name, or None if nothing is left.

    Lowercase, accents folded to plain letters, anything else dropped, the
    first few words joined with hyphens and the result cut to `max_length`.

    `max_words` caps the word count, not the length — a single long, unbroken
    token (a run-on label, a service hallucinating a sentence with no spaces)
    would otherwise become a folder name long enough to blow past the
    filesystem's name limit and crash mid-loop. The joined slug is then cut to
    `max_length` characters wherever that lands — mid-word if need be — and
    only a hyphen left dangling by the cut is trimmed off the end.

    Returning None rather than a fallback is the whole point of this function
    existing beside `kebab_case`: a caller that got a name from a *person* can
    then say "that name has nothing in it I can use" instead of silently
    creating something called `item`.
    """
    cleaned = _UNSAFE.sub("", _fold_to_ascii(text).lower())
    words = [word for word in _SEPARATORS.split(cleaned) if word][:max_words]
    if not words:
        return None
    return "-".join(words)[:max_length].rstrip("-") or None


def typed_slug(text: str, what: str) -> str | None:
    """`slug` for a name a person typed: refused when it would be shortened.

    `slug` keeps four words and 60 characters, which is right for a name
    PixelMagic makes up but silently loses part of what a person chose.
    Folding case, accents and punctuation is not shortening and is allowed.
    Returns None, as `slug` does, when nothing usable is left. `UnusableName`
    when the cap would cut words or characters; the sentence says the limit
    and offers the shortened form to type. `what` names the thing being
    named ("pick name", "surface name").
    """
    # Truly uncapped: folding can lengthen text ("ﬃ" becomes "ffi").
    uncapped = slug(text, max_words=sys.maxsize, max_length=sys.maxsize)
    capped = slug(text)
    if uncapped != capped:
        raise UnusableName(
            f"{what} {text.strip()!r} is too long: at most {MAX_NAME_WORDS} words, "
            f"{MAX_NAME_LENGTH} characters; "
            f"try {capped!r}"
        )
    return capped


def kebab_case(
    text: str,
    *,
    max_words: int = MAX_NAME_WORDS,
    max_length: int = MAX_NAME_LENGTH,
    fallback: str = FALLBACK_NAME
) -> str:
    """`slug`, with a fallback for text that squeezed down to nothing.

    This is what a *service's* answer goes through. Naming a cutout is a
    convenience, not a requirement — `describe_cutout` promises a service that
    cannot answer never stops a decomposition — so an unusable label becomes
    `item` and `unique_name` numbers the rest. Where a *person* typed the name,
    call `slug` and tell them.
    """
    return slug(text, max_words=max_words, max_length=max_length) or fallback


#: Little words a scene's name does without ("a cozy den with a stove" is
#: `cozy-den-stove`).
SCENE_NAME_FILLER = frozenset(
    "a an the of with and or in on at for to from by into onto over under near".split()
)

#: How many words, and how many characters, a scene's name may keep.
SCENE_NAME_WORDS = 3
SCENE_NAME_LENGTH = 40

#: The name of a scene whose text has no usable letters (another script, say).
FALLBACK_SCENE_NAME = "scene"


def scene_name(text: str) -> str:
    """A short, safe name for a whole picture, from its subject.

    The subject is what the person typed, or an imported file's stem.

    Folds the text with `slug`, drops the filler words, keeps the first
    `SCENE_NAME_WORDS` of what is left and cuts to `SCENE_NAME_LENGTH`. Text
    that was only filler gives the plain slug of the text; text with nothing
    usable in it gives `scene`. An underscore counts as a space, since file
    names use it between words. The caller makes it unique with `unique_name`.
    """
    text = text.replace("_", " ")
    folded = slug(text, max_words=len(text) + 1, max_length=len(text) + 1)
    if folded is None:
        return FALLBACK_SCENE_NAME
    words = [word for word in folded.split("-") if word not in SCENE_NAME_FILLER]
    chosen = "-".join(words[:SCENE_NAME_WORDS])[:SCENE_NAME_LENGTH].rstrip("-")
    if chosen:
        return chosen
    plain = slug(text, max_words=SCENE_NAME_WORDS, max_length=SCENE_NAME_LENGTH)
    return plain or FALLBACK_SCENE_NAME


def unique_name(base: str, taken: Collection[str]) -> str:
    """Return `base`, or `base-2`, `base-3` … until it is not already `taken`."""
    if base not in taken:
        return base
    counter = 2
    while f"{base}-{counter}" in taken:
        counter += 1
    return f"{base}-{counter}"


def describe_cutout(service: ImageService, image_path: Path) -> str:
    """Ask the service what one cut-out piece is.

    Naming is a convenience, not a requirement — a service without vision or
    one that errors gets the fallback rather than failing the decomposition.
    """
    if not service.capabilities().vision:
        return FALLBACK_NAME
    try:
        answer = service.describe(image_path, CUTOUT_QUESTION).strip()
    except Exception:  # any provider failure degrades to the fallback
        return FALLBACK_NAME
    return answer or FALLBACK_NAME


def suggest_direction_name(service: ImageService, style: str) -> str:
    """Ask the service to invent a kebab-case name for an art direction.

    This is the fallback for `pixelmagic direction new` with no `--name`
    (spec §1, "Direction naming").
    """
    question = load_direction_name_question().format(style=style.strip())
    try:
        answer = service.describe(None, question)
    except Exception:  # naming must never block creating a direction
        return FALLBACK_DIRECTION_NAME
    return kebab_case(answer, fallback=FALLBACK_DIRECTION_NAME)


def check_name(value: str, what: str) -> str:
    """Return `value` if it is one of our own names, or refuse it by name.

    The web page takes these out of URLs and the MCP server takes them out of
    tool arguments. Both are places a name arrives from outside and then
    becomes part of a file path, so both need this one guard rather than a
    guard each. `Store.rename_direction` calls it too, so there is one rule
    and not two. Its other callers are the faces landing next — see the
    deviations register — which is why it is unit-tested directly here rather
    than only through whatever calls it first.

    `fullmatch` and not `match`: `re.match` is not anchored at the end, and a
    dollar sign would match before a trailing newline, so a value ending in one
    would be accepted and become a direction folder whose name ends in a
    newline — a different direction from the one without it, and one a browser
    can ask for as `%0A`. The pattern carries the absolute-start and
    absolute-end anchors as well, so a caller who reaches for `SAFE_NAME`
    rather than this function gets the same answer either way. (Spelled out
    rather than shown, because a backslash in a docstring is `D301`.)
    """
    if not SAFE_NAME.fullmatch(value):
        raise UnusableName(
            f"{what} {value!r} is not a name PixelMagic uses — "
            f"lowercase words joined by single hyphens"
        )
    return value


@dataclass(frozen=True)
class Surface:
    """One surface a person typed for decompose to turn into tiles.

    Exists only for the length of one `decompose()` call — never written to
    a sidecar, which is why this lives beside `parse_surface`, its one real
    producer, rather than in `model.py` (that module's own docstring:
    "every model and enum PixelMagic *persists*").

    `text` is the typed words, unslugged, exactly as they go into the
    surface's own generation subject (spec 2026-09-25 amendment (a)/(c)).
    `name` is `text` squeezed through `slug` — the base every one of the
    surface's own tile names is built from (`wood-floor-v1` ... for a recipe
    surface).
    `kind` is the recipe kind; `shapes` are the rough polygons drawn over the
    surface (each a tuple of `(x, y)` corners, in the picture's own pixels)
    and `box` is the small patch `(x0, y0, x1, y1)`, right and bottom
    exclusive, its colours are read from. A surface with no kind is never
    produced by `parse_surface`; a hand-built one is refused by `decompose`.
    """

    text: str
    name: str
    kind: str | None = None
    shapes: tuple[tuple[tuple[int, int], ...], ...] = ()
    box: tuple[int, int, int, int] | None = None


#: The exact trailing marker of the removed bordered surface. Checked as a
#: literal suffix, never a substring search, so a surface whose own words
#: happen to contain "bordered" without the colon is not caught by it.
_BORDERED_MARKER = ":bordered"

#: The tail of a recipe surface after its colon: lowercase letters, then
#: optionally `@` and whatever was typed after it (the old point form, refused).
_KIND_TAIL = re.compile(r"[a-z]+(@.*)?")

#: One corner: two whole numbers, optionally negative, so a corner past the
#: picture's edge is allowed (a shape is clipped to the picture).
_CORNER = re.compile(r"(-?\d+),(-?\d+)")

#: The recipe example every typing sentence points at.
RECIPE_EXAMPLE = (
    '"plaster wall:plaster; box 180,240 212,285; '
    'shape 140,400 200,230 330,110 520,70; shape 1100,440 1100,330 1170,285"'
)

#: The same surface string with letters for every number, for a hint printed beside a real id,
#: where plausible-looking coordinates would be taken as real. Kinds are `RECIPE_KINDS`.
RECIPE_PLACEHOLDER = (
    '"NAME:KIND; box X0,Y0 X1,Y1; shape X0,Y0 X1,Y1 X2,Y2" '
    f"(KIND is {' or '.join(RECIPE_KINDS)}; box and shape corners are the picture's own "
    "pixels; a shape needs three or more corners)"
)

#: What the example means, said once for every sentence that shows it.
RECIPE_EXAMPLE_HELP = (
    f"for example {RECIPE_EXAMPLE}: 'box' is two opposite corners of a small patch of "
    f"plain surface (its colours), each 'shape' is the corners of a rough outline "
    f"drawn over the surface (at least 3, and at least one shape)"
)


def kindless_surface_sentence(text: str) -> str:
    """The refusal for a typed surface with no recipe kind.

    One sentence for every caller: `parse_surface` raises it for a typed
    entry and `decompose` raises it for a hand-built `Surface`, so a person
    is told the same way forward wherever the plain surface is caught. The
    kinds in it are `RECIPE_KINDS`, never retyped here.
    """
    kinds = ", ".join(RECIPE_KINDS)
    return (
        f"surface {text!r} needs a recipe kind, a box and shapes on the picture, "
        f"{RECIPE_EXAMPLE_HELP} (kinds: {kinds})"
    )


def _corners(parts: Sequence[str], text: str, word: str) -> list[tuple[int, int]]:
    """Read `x,y` corners, each refused with its own sentence when it is not two numbers."""
    corners = []
    for part in parts:
        match = _CORNER.fullmatch(part)
        if match is None:
            raise UnusableName(
                f"surface {text!r}: {part!r} in a {word} is not a corner; write x,y with "
                f"two whole numbers, {RECIPE_EXAMPLE_HELP}"
            )
        corners.append((int(match.group(1)), int(match.group(2))))
    return corners


def _box_of(corners: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Two opposite corners, in either order, as `(x0, y0, x1, y1)`."""
    (xa, ya), (xb, yb) = corners
    return (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))


def parse_box(raw: str) -> tuple[int, int, int, int]:
    """Read a colour box typed on its own, as `x0,y0 x1,y1` (either corner first).

    The same corner reading and ordering `parse_surface` gives a surface's
    `box`, for a face that asks about a box before any surface exists (the
    page's live colour readout). Refused as `UnusableName` when it is not
    exactly two `x,y` corners.
    """
    parts = raw.split()
    if len(parts) != 2:
        raise UnusableName(
            f"a box is two corners written x,y x,y, but {raw!r} is not; "
            f"{RECIPE_EXAMPLE_HELP}"
        )
    return _box_of(_corners(parts, "the colour box", "box"))


def parse_surface(raw: str) -> Surface:
    """Turn one `--surface`/MCP `surfaces` entry into a `Surface`.

    A recipe surface is one string of segments split by `;`:
    `"plaster wall:plaster; box 180,240 212,285; shape 140,400 200,230 330,110"`.
    The first is `words:kind`, then exactly one `box` (two opposite corners, in
    either order, stored as `(x0, y0, x1, y1)` with right and bottom exclusive)
    and one or more `shape`s (at least three corners each). Anything else is
    refused with `UnusableName`, including a plain `"wood floor"`, the old
    `name:kind@x,y` point form and the old `name:bordered` marker, which are
    gone. The kinds in each sentence are `RECIPE_KINDS`, never retyped here.

    The `:bordered` marker is checked against `raw.strip()`, not `raw` itself:
    a trailing space after it is an easy typo and must not change the refusal.
    The kind part is split off at the last `:` of the first segment, so the
    words never hold one. `name` is `text` through `typed_slug` — the same rule
    `rename_cutout` uses for a person-typed name — so an entry with nothing
    usable in it (`""`, `"   "`, punctuation only) is refused by name rather
    than silently becoming `item`, and one longer than four words or 60
    characters is refused rather than shortened.
    """
    kinds = ", ".join(RECIPE_KINDS)
    stripped = raw.strip()
    if stripped.endswith(_BORDERED_MARKER):
        raise UnusableName(
            f"surface {stripped!r}: bordered surfaces are gone; a surface is now a kind, a "
            f"box and shapes drawn on the picture, {RECIPE_EXAMPLE_HELP} (kinds: {kinds})"
        )
    first, *others = (segment.strip() for segment in stripped.split(";"))
    # A trailing `;` or an empty segment (`;;`) is a typo, not a refusal.
    others = [segment for segment in others if segment]
    words, colon, tail = first.rpartition(":")
    if not colon:
        words, tail = first, ""
    text = words.strip()
    name = typed_slug(text, "surface name")
    if name is None:
        raise UnusableName(f"surface {raw!r} has nothing in it that can be a name")
    tail = tail.strip()
    if not colon or not _KIND_TAIL.fullmatch(tail):
        raise UnusableName(kindless_surface_sentence(first))
    kind, at, _ = tail.partition("@")
    if at:
        raise UnusableName(
            f"surface {text!r}: the point after @ is gone; a surface is now a box and "
            f"shapes drawn on the picture, {RECIPE_EXAMPLE_HELP}"
        )
    if kind not in RECIPE_KINDS:
        raise UnusableName(
            f"surface {text!r} has no recipe kind {kind!r}; the kinds are {kinds}"
        )
    box: tuple[int, int, int, int] | None = None
    shapes: list[tuple[tuple[int, int], ...]] = []
    for segment in others:
        word, _, body = segment.partition(" ")
        parts = body.split()
        if word == "box":
            if box is not None:
                raise UnusableName(
                    f"surface {text!r} has more than one box; it needs exactly one, "
                    f"{RECIPE_EXAMPLE_HELP}"
                )
            if len(parts) != 2:
                raise UnusableName(
                    f"surface {text!r}: a box is two corners, {RECIPE_EXAMPLE_HELP}"
                )
            box = _box_of(_corners(parts, text, "box"))
        elif word == "shape":
            if len(parts) < 3:
                raise UnusableName(
                    f"surface {text!r}: a shape needs at least 3 corners, {RECIPE_EXAMPLE_HELP}"
                )
            shapes.append(tuple(_corners(parts, text, "shape")))
        else:
            raise UnusableName(
                f"surface {text!r}: {segment!r} is not a box or a shape, {RECIPE_EXAMPLE_HELP}"
            )
    if box is None:
        raise UnusableName(f"surface {text!r} has no box, {RECIPE_EXAMPLE_HELP}")
    if not shapes:
        raise UnusableName(f"surface {text!r} has no shape, {RECIPE_EXAMPLE_HELP}")
    return Surface(
        text=text,
        name=name,
        kind=kind,
        shapes=tuple(shapes),
        box=box,
    )


def parse_surfaces(raw: Sequence[str]) -> list[Surface]:
    """`parse_surface` over every typed surface, in the order they were given."""
    return [parse_surface(entry) for entry in raw]
