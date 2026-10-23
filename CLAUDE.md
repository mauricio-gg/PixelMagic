# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PixelMagic turns a text prompt into game-ready pixel art. An image service draws concept art (or a person brings a picture), a person or an agent approves it, the tool finds the objects on it and the person picks the ones to keep, the image service redraws them and each is rebuilt on its own pixel grid, floors and walls are drawn from the picture's own colours, PixelRefiner, when turned on with `--refiner cli`, turns the scene into true pixel art (by default a stand-in copies it through), and the result is exported as PNGs, a `manifest.json` and Aseprite files.

The workspace-level rules in `../CLAUDE.md` apply (that parent workspace is in the private working setup, not the public copy).

## Repo layout

```
src/pixelmagic/
  model.py      every Pydantic model and enum — the project's vocabulary
  store.py      the folder tree: paths, JSON sidecars, ids
  approval.py   the one approval hook, answered differently by each face
  naming.py     free text to safe kebab-case names
  directions.py starting an art direction: the one rule all three faces share
  keys.py       where an image service's API key comes from (environment, then
                the keychain) and saving or removing one; the only place a key is read
  services.py   what the Services page and every face need: display names, a status
                per service, the free test-key call, the per-game choice and which
                service draws (the flag, else the saved choice, else none; `require_service`
                is the check every face makes before it draws)
  providers/    image services (base.py = the protocol, fake.py = the offline one)
  stages/       generate, decompose, layout (target, footprint and anchor sums),
                slice (keying, boxes, blob labels), refine, export, recipes.py
                (recipe tiles: the colours of a box, the outline filled from
                drawn shapes, and the drawing of a surface's 32 tiles;
                numpy only, no segmenter call), and the
                object-by-object road for props: suggest.py (find and number
                objects), pick.py (choose them: number, point, box), lasso.py
                (choose one made of several parts), groups.py (the maths around
                one redraw call), redraw.py (the groups, the service, the pieces),
                rebuild.py (finds each redrawn piece's own pixel grid and
                rebuilds the piece on it); room.py (plans a kept scene's room
                file: scale, canvas, each prop's place, each surface's map,
                the structure layer's pixels; compose_room pastes the plan's
                layers as pictures; no Aseprite in it); room_preview.py (builds
                the room picture the export screen shows); cleanup.py (Job 1,
                cleans a picture up to pixel art at native size);
                pixelgrid.py (reads a pixel-art picture's pixel size with
                clean-up's rule, or a typed size; makes the native picture and
                the 2x finder copy; maps outlines back as whole cells);
                pixelcut.py (`start_pixel_cut`, `cut_pieces`: the second road
                to a prop, straight from the picture's own pixels)
  refiner/      the PixelRefiner seam (base.py protocol, stub.py stand-in, cli_adapter.py)
  segmenter/    the core's side of the object-finder seam (base.py = the contract
                and protocol, stub.py the offline one, cli_adapter.py the subprocess)
  prompts/      the prompt templates, as markdown
  aseprite/     the Lua scripts that write and read .aseprite files
                (single_sprite.lua for one piece, tileset_sprite.lua for a
                surface's tiles, read_tileset.lua for reading a master back,
                room_sprite.lua for a scene's room, read_frames.lua for the
                flat frames of a sprite)
  journey.py    works out which of a scene's seven steps (Start, Approve, Keep
                scene, Pick props, Surfaces, Refine, Export) is done, current, next
                or waiting, by reading the project folder; an unreadable
                decomposition, asset, refine record or manifest counts as not
                there, but the scene's own generation must read
  cli.py        the command line
  web/          the web page: app.py, one module per screen (picking.py, surfaces.py, cleaning.py = the clean-up screen,
                pixelcutting.py = the pixel-cut size step and picking screen,
                services.py = the Services page, ...), templates/
                timeline.py turns those steps into links for the page; static/
                holds the one stylesheet (pixelmagic.css), pixelmagic.js, pixelcut.js and
                pixelcut_pick.js (the pixel-cut screens' scripts), the three
                background SVGs (grain.svg, mosaic.svg, mosaic-tile.svg), and Inter
                and Silkscreen bundled with their OFL licences so the page works offline
  mcp_server.py the MCP server: one tool per pipeline step
segmenter/      the real object finder (Segment Anything, MLX): its own folder
                and environment, a separate command, never imported; its README
                and vendor/NOTICE say what is vendored
tests/          synthetic-image unit tests; nothing here touches the network
                `test_end_to_end.py` is the README walkthrough's props path in library form
docs/superpowers/specs/   the design spec (private working repo only)
docs/superpowers/plans/   the implementation plans (private working repo only)
LICENSE         MIT
THIRD_PARTY.md  the licences of everything PixelMagic uses or downloads
```

## Commands

```bash
uv sync                      # install everything
uv run pixelmagic --help     # the command line
uv run pytest                # the whole test suite
uv run ruff check            # lint
uv run pixelmagic --root <folder> web    # the web page, localhost only; --service <name> before the command overrides the game's saved choice; --refiner goes after web (e.g. `web --refiner cli` for the real PixelRefiner)
uv run pixelmagic --root <folder> mcp    # the MCP server, on stdio
(cd segmenter && uv sync && uv run pixelmagic-segmenter fetch-model)
                             # one-time: the real object finder and its 184 MB model
```

The whole pipeline runs offline, with `--service fake` and `--segmenter stub`; the
stand-in refiner is the default, so no refiner flag is needed. The README's "Five
minutes with PixelMagic" is that path command by command: `import` a picture of
separate objects (the fake service's own picture is a single block, which
`BACKGROUND_SHARE` drops, so it cannot be cut), `keep-scene <id>` (with the stand-in
it prints a pitch note), `suggest <id> --segmenter stub`, `pick <decomposition-id> 1
2 3`, `redraw <decomposition-id>` (with `--service fake`), `decompose <id> --surface
"wood floor:wood; box ...; shape ..."` for a floor, `refine`, `export`. The fake
service answers a group with a small chunky stand-in board, so the rebuild finds a
grid on it. `--segmenter stub` finds each blob of opaque pixels and needs no setup;
the default, `cli`, is the real finder and needs the one-time setup above (Apple
silicon only). `--refiner cli` turns on the real PixelRefiner: it runs the command
in `PIXELMAGIC_REFINER_BIN`, else `pixel-refiner` on the PATH, and a machine without
it gets a `RefinerError` that names the way out. The stand-in copies the scene
through and reads no pitch of its own. Do not create a project inside this repo —
pass `--root` a scratch folder.

## What must not break silently

- **Keys live in the keychain or the environment, and the choice of service is a name.** The key lookup is `keys.py` and nowhere else: the environment variable wins over a key saved in the operating system's keychain (through `keyring`), and nothing is ever written to the project folder. A key is never logged, never put in an exception message, never returned to a page (the Services page's field is never filled in; a redirect after a post, whether it saved or tested a key, carries a short fixed code, never text, so reloading never repeats the call) and goes only to its own service. The "test key" call is the free models-list request, which draws and charges nothing. Which service draws is decided in one place, `services.chosen_service`, read per request or tool call by all three faces: the `--service` flag wins over the game's saved choice in `pixelmagic.json`; with neither there is no service and nothing draws. Every face makes its drawing step call `services.require_service`, which refuses with `services.NO_SERVICE_SENTENCE` (a usage error on the command line, a `ToolError` in MCP, a 400 page on the web, a hand-posted form included); that covers generate, retry and reject, redraw, and creating a direction with no name (naming is a paid call). The saved choice holds a name only and is checked strictly against `services.CHOOSABLE` (`codex`, `gemini`, `openai`): anything else (a shared or hand-edited project, or `fake` from an older one) counts as no choice and logs a warning, so a file in a project can never supply a command, path, address or key (`PIXELMAGIC_CODEX_BIN` and `CODEX_HOME` come from the process environment only). Because that choice can spend a person's own key, every button that draws names the service that will be paid ("Draw with Gemini", "Redraw with Gemini"), every MCP tool that draws names it in its result, and every command that draws prints "Drawing with <name>." on standard error before it does; no web route draws without a button press. With no service the web page's Draw and Redraw controls become a "Choose an image service first" link to `/services` and no "Drawing with" line is printed. The offline stand-in is reached only through `--service fake`; it cannot be saved as a game's choice and is not on the Services page. While Codex is chosen, `services.codex_status` keeps the last `codex login status` answer for 30 seconds so a page load never waits on Codex, and the Services page's Check login always runs a fresh one. Tests never touch the real keychain or a real key: the autouse `isolated_keys` fixture in `tests/conftest.py` gives each test an in-memory keychain and clears the key variables.
- **Run ids are checked where they become paths.** A generation id or a decomposition id is checked for a safe name in `Store.generation_dir` and `Store.decomposition_dir`, so every face is covered and an id such as `../..` is refused, never followed. Every face checks a run id with `store.check_run_id` (it accepts exactly what `check_name` does), whose refusal says what an id looks like (`20260903-191233`, sometimes with a `-2` suffix) instead of the name sentence.
- **Every stage writes its JSON sidecar.** Generate writes `request.json`, `result.json` and `review.json`; decompose writes `cutout.json`; refine writes `refine.json`; export writes `manifest.json`. That record is what makes any step re-runnable, and it is the only memory the tool has. `Store.write_model` writes to a scratch file and renames it into place, so an interrupted write can never leave half a sidecar.
- **The three faces share one core.** The command line, the web page and the MCP server are thin: they build a service, an approval hook, a refiner and a segmenter, then call the same stage functions. Logic never goes in a face. Picking objects is the clearest case: the command line has `suggest`, `pick`, `picks`, `unpick`, `redraw`, `pixel-cut` and `cut`; the MCP server has `suggest_objects`, `show_suggestions`, `pick_objects`, `fix_pick`, `unpick`, `show_picks`, `redraw_objects`, `start_pixel_cut` and `cut_pixel_pieces`; the web page has a picking screen with a floating tool tray: Pick, Add a spot, Remove a spot and Lasso, plus Undo and Bigger/Smaller. With a mouse, hover outlines the object under the pointer, scroll steps to bigger or smaller ones, a click picks, and Add and Remove fix a spot on the selected pick. Holding a key switches the tray's tool only while it is held (`PixelMagic.holdKeys` in `pixelmagic.js`, shared with the pixel-cut screen): Cmd (Ctrl on Windows and Linux) is Add, Option (Alt) is Remove, Shift is Lasso. A click means the tool when the press began and never the event's modifier keys; losing window focus returns the tool; a held tool is never remembered; the tray buttons show their key (⌘ ⌥ ⇧ or Ctrl Alt Shift) for a fine pointer. On touch, a tap previews and a second tap on the same object picks it, Bigger and Smaller replace scroll, and Add and Remove are modes; the hints follow the input device. Picks are listed as rows with their own warnings and problems underneath (`redraw.notes_for`). Redraw lands back on this screen (`redrawn=1`), which then offers "Continue to Surfaces" and "Redraw again". Number tags on the unpicked objects are drawn only when there are at most `suggest.PAGE_SIZE` (40) suggestions. The lasso and Undo exist on the page only: not on the command line, not as MCP tools. Undo's rule lives in the core (`stages/pick.py`), not in the script. What a click means, the 12 px guard and every refusal are the core's. The web page also has the pixel-cut screen (`web/pixelcutting.py`, see the pixel-cut rule below), which shares this core. The picture page offers both roads (an approved picture's page has an "Already pixel art?" section that leads to the pixel cut, and once a run exists a "Continue the pixel cut" link; `journey.props_runs` gives it the newest run of each road), and the timeline's Pick props step links the newest run of either road (`journey.py`). On the page every tray (picking, surfaces, pixel-cut picking, Refine's Before and After) is the app's yellow (`--accent`) with dark text, and the selected tool is a white pill (`DESIGN.md`, "Floating tool tray").
- **Transparent first, key colour second.** Always ask the image service for a transparent background. Only fall back to magenta or lime when the service cannot do it or the picture comes back with no transparency at all.
- **PixelRefiner is a subprocess, opt-in, and only the kept scene goes to it.** Only with `--refiner cli` (the default is the stand-in, `registry.DEFAULT_REFINER`) we call the command in `PIXELMAGIC_REFINER_BIN`, else `pixel-refiner` on the PATH, and read its JSON. We never reimplement any of its image maths in Python. A prop comes only from the object road, rebuilt on its own pixel grid, or from the pixel cut, made of the picture's own pixels; a recipe tile is drawn at its final size; none of the three is sent to the refiner. The one scoped exception to "no image maths in Python" is recovering the grid of a chunky redraw (the finder, the grouping of cells into colours, the rebuild), which is PixelMagic's own Python in `stages/rebuild.py` under the 2026-10-06 amendment to locked decision 2. It imports no PixelRefiner and runs no subprocess; everything else here stays PixelRefiner's, through the subprocess.
- **A batch takes one picture and leaves the file where the caller asked for it.** `Refiner.batch` takes exactly one picture, and both refiners refuse any other count before anything runs or is written. When the call returns, that request's own `output_path` holds its refined image; the contract is written on the protocol in `refiner/base.py`.
- **A bundle says how it was made.** The stand-in refiner, which a bare run uses,
  produces a scene that is not pixel art. `manifest.json` records `refiner_tool` and
  `route` on every entry precisely so a real bundle can be told from a copied-through
  one; never remove either field. At the top it records `cell_size` and `pitches` (one
  `scene` entry, when the scene has a pitch); on every entry it records `pitch`,
  `pitch_source`, `target`, `footprint`, `anchor`, `variant`, `recipe`, `segmenter`,
  `cohesion` and `warnings`, so an engine places a piece on the grid without ever
  looking at a screen. A new manifest does not write `role`, `position` or `grid` on
  an entry, or `pitches.props`; the models keep those fields so manifests and sidecars
  made earlier still load. `variant` and `recipe` are written on every entry but are
  null except on a recipe tile (`recipe` is its `RecipeRecord`). Every exported
  `.aseprite` carries the direction's `cell_size` as its own grid.
  `segmenter` is the name of the object finder that cut the piece out, copied
  from its asset; it is null on the scene and on a recipe tile, and is never
  dropped.

  A prop rebuilt on its own pixel grid says so with two values:
  `refiner_tool` is `pixelmagic-grid-rebuild` (`stages/refine.REBUILD_REFINER`)
  and `route` is `rebuild` (`stages/refine.REBUILD_ROUTE`). Its `pitch`,
  `pitch_source`, `target`, `footprint` and `anchor` are the piece's own. A
  bundle that mixes a refined or stand-in scene, rebuilt props, pixel-cut props and recipe tiles
  tells them apart by those two fields.

  A recipe tile says so the same way: `refiner_tool` is `pixelmagic-recipe`
  (`stages/refine.RECIPE_REFINER`) and `route` is `recipe`
  (`stages/refine.RECIPE_ROUTE`), whether the refiner was real or the stand-in.
  Its `target` is the cell size, and it has no `pitch`, `footprint` or `anchor`.

  A pixel-cut prop says so the same way: `refiner_tool` is `pixelmagic-pixel-cut`
  (`stages/refine.PIXEL_CUT_REFINER`) and `route` is `pixel-cut`
  (`stages/refine.PIXEL_CUT_ROUTE`). It is never refined or sent to PixelRefiner,
  and a refine setting or a candidate re-pick aimed at it is refused.
- **A project made before the old roads were removed never gives a traceback.**
  Tiles and props from the props sheet, from `:bordered` surfaces, from plain
  patches, or cut before the pixel-grid rebuild are "old items": a tile with no `recipe`, or a prop with neither `grid_rebuild` nor `pixel_cut`. One that was
  already refined exports as it is (its `footprint` and `anchor` copied across
  when it has them); one that never was refused refine with an `OLD_ITEM_REFUSED`
  receipt (`stages/refine.old_item_sentence` says what to do instead), is
  skipped by export with its "has not been refined yet" note, and stops `keep`
  on a cut-up run that still holds it (`OldPieceRefused`). `refine_asset` by name
  raises the same `old_item_sentence` for any old item, refined before or not. Reading and deleting
  such a run still work, and an old `refine.json` that carries `SEAM_VISIBLE`
  still loads and exports.
- **A tile comes from a recipe surface the person typed, and carries a variant.**
  The person names the scene's surfaces: `--surface
  "wood floor:wood; box 180,240 212,285; shape 140,400 200,230 330,110 520,70"` on
  the command line, the same strings as `surfaces` on the MCP `cut_into_pieces`
  tool; the web page's surfaces screen draws the shapes and the box and sends the
  same string (see below). With no surface typed, decompose makes
  nothing: it is refused (`NoPicks`), and each face words its own way
  forward. Every tile's `ai_name` is the typed text.

  **A recipe surface** is typed as one string, parsed only by `naming.parse_surface`:
  `plaster wall:plaster; box 180,240 212,285; shape 140,400 200,230 330,110 520,70;
  shape 1100,440 1100,330 1170,285`. `kind` is one of `RECIPE_KINDS` (`plaster`,
  `stone`, `wood`). The **box** is two opposite corners (either order, right and
  bottom exclusive) of a small plain patch of the surface; the colours are read from
  the box and nowhere else. Each **shape** is at least 3 corners of a rough outline
  drawn over the surface, and there is one or more; together they say where the
  surface goes. All corners are in the picture's own pixels. The web page's surfaces screen
  (`web/surfaces.py`, `templates/surfaces.html`) lets a person draw one or more shapes
  and one colour box on the picture, name the surface and choose its kind; it
  composes this same string and `parse_surface` parses it, so the typed form
  remains on the command line and the MCP server. `parse_surface` refuses, as
  `UnusableName`, a plain name with no kind, an unknown kind, a missing or doubled box,
  no shape, a shape with fewer than 3 corners, a corner that is not two whole numbers,
  a name over `MAX_NAME_WORDS` (4) words or `MAX_NAME_LENGTH` (60) characters (refused,
  never shortened),
  and the two removed forms, `name:bordered` and `name:kind@x,y`, each with a sentence
  giving the way forward; the kinds in each sentence are `RECIPE_KINDS`, never retyped.
  `decompose` refuses, as `RecipeRefused` and before a decomposition folder is made or
  anything is drawn, a recipe surface in a direction whose cell size is not
  `RECIPE_CELL_SIZE` (16), with a box not wholly inside the picture, or with shapes that
  cover no pixel of it. A refusal that comes before any tile is written removes the
  decomposition folder `decompose` made, so it leaves nothing behind; a refusal after an
  earlier surface's tiles were written keeps those tiles and writes the sidecar.
  `decompose` takes no segmenter and no image service, and the
  command line's `decompose` has no `--segmenter`.

  The drawing is `stages/recipes.py`, pure numpy image work of PixelMagic's own (the
  2026-10-07 amendment to locked decision 2); no image service and no PixelRefiner is
  involved, and the same surface name and the same colours with their shares give the
  same bytes (seeds come from sha256 of the name, never `hash()`). The steps, in order:

  - **The outline.** No object finder is involved. `decompose` fills each shape as a
    polygon on a picture-sized mask (clipped to the picture), joins the shapes, crops
    the result to the box they cover and saves it as
    `decompositions/<id>/surfaces/<name>/outline.png` (`Store.surface_dir`). The tiles'
    `bbox` comes from it, and the room reads it unchanged.
  - **The colours** are read by `read_shades` from the picture's own pixels inside the
    box and nowhere else: a pinned k-means, 5 colours for wood and stone and 4 for
    plaster, sorted dark to light, each with its share of the box's pixels
    (`RecipeRecord.shades`, `.shares`). Refused as `RecipeRefused` when the box has
    fewer than `MIN_BOX_PIXELS` (1024, a 32 x 32 patch) pixels, fewer different colours
    than the kind needs, a line shade under `MIN_LINE_SHADE_SHARE` (1%), or any other
    shade under `MIN_SHADE_SHARE` (3%). A kind's line shade is wood's darkest (the
    joints) and stone's lightest (the grout); plaster has none. Plaster's other shades
    are pulled halfway (`PLASTER_CALM`) towards its biggest shade, not towards the mean,
    and the shares are not changed.
  - **The tiles.** Every kind is drawn as `VARIANTS` (8) blocks of 2 x 2 tiles, so 32
    tiles of `RECIPE_CELL_SIZE` square, opaque, named `<surface>-vN-nw`, `-ne`, `-sw`,
    `-se` (`PLACES`). Wood is three rows of long planks per block. Stone is four rounded
    slabs per block, in two rows offset by about half a slab, each with a thin rim in the
    next darker shade from the box (none when the slab is already the darkest) and thin
    grout in the lightest shade; most slabs are the main tone and at most one slab per
    block is the second tone. Plaster is a mostly smooth wall with a few faint patches in
    one shade: of the two next to the wall's own in the dark-to-light order, the nearer in
    colour. Each cutout carries `variant` (1 to 8) and a `RecipeRecord` (`kind`, `layout`
    always `"block"`, `place`, `surface`, `shapes`, `box`, `shades`, `shares`; `point` and
    `segmenter` only on an older record), copied to the asset and the manifest entry. Its
    `target` is the cell size, square; it gets no footprint or anchor (`layout_cutout`,
    which gives a prop both, is never called on it).
  - **The seam check.** `check_seams` runs on the finished set before any tile is
    written. The blocks must be able to sit in any order: no two identical, and on each
    axis the worst join between any two blocks and the worst join between a block's
    own tiles may be no busier than the busiest line inside a block, by exact RGBA
    equality. A failure is `RecipeSeamBroken` (`rule` is `block-join`, `tile-join` or
    `duplicate`), a bug in a recipe and not something the person did, so it is not
    caught by the command line and has its own page in `web/app.py`. Plaster
    draws its shared edge band again (`_JOIN_ATTEMPTS`, 20) when its own check fails.
  - **Names.** `_reserve_recipe_names` checks the bare surface name (export writes
    `<surface>.aseprite` under it) and the 32 tile names. A name already used by
    something that is not this same surface is refused (`RecipeRefused`); one used
    only by earlier recipe tiles and assets of this surface drawn from the same picture
    (`source_generation`) is a re-run and is allowed. The same surface name on another
    picture is refused as `RecipeRefused`, on every face, since it would take that
    picture's tiles and assets over. A recipe surface never moves to `<name>-2`: the person types a different name.
  - **Refine skips them.** A recipe tile is not refined: no palette fit, no
    refiner call. `refine.py`'s `_keep_recipe` copies `source.png` to `refined.png` and
    writes `refine.json` with `refiner_tool` `pixelmagic-recipe` (`RECIPE_REFINER`) and
    `route` `recipe` (`RECIPE_ROUTE`). `refine_direction` skips them and the report counts
    them; a setting or a `candidate_id` aimed at one (`refine_asset`) is refused as
    `RecipeTileRefused`, the same error that
    refuses renaming one (it would break the set). A stub
    refiner changes nothing here: a stand-in bundle's recipe tiles say
    `pixelmagic-recipe` too.
  - **Export.** `_write_tilesets` writes one `<surface>.aseprite` per recipe surface
    (needs Aseprite): a tileset of the 32 tiles in order (block 1's `nw`, `ne`, `sw`,
    `se`, then block 2's, ...) and a sample floor of `SAMPLE_COLUMNS` x `SAMPLE_ROWS`
    (10 x 6) tiles laid from them by `sample_layout`. Tiles are found by `recipe.surface`,
    `variant` and `place` on the manifest entries, never by name; a surface that does not
    hold exactly 32 tiles gets a note and no file, and a file name already taken by a
    prop's own `.aseprite` is never overwritten. A recipe tile has no `.aseprite` of its
    own, and every one of a surface's entries lists the surface's file under
    `files["aseprite"]`. The file is built from the bundle's native-size pictures; export always writes
    native size, whatever `scales` asks for (`stages/export.export`).
  - **Touch-ups go through the master file.** The first export that writes a surface's
    file also saves a copy as its **master**, `<direction>/assets/<surface>.aseprite`
    (`Store.surface_master`; `list_assets` does not see it). The person edits the master in
    Aseprite. On every export, if the master exists, the surface's bundle PNGs (every
    scale) and its manifest palettes are taken from the master's tileset by
    `_take_from_master`, and the master is copied into the bundle as the surface's file.
    The master must have a tilemap layer, be readable, and hold exactly 32 tiles of
    `cell_size` square; if not, export uses the recipe's own tiles, adds a note naming
    the file and what is wrong, and leaves the master as it is. Typing the surface again
    redraws the recipe tiles but keeps the master, and the decomposition warns that it
    did; deleting the master starts fresh. There is no `touchup` command and no
    hand-touched flag on a tile: the master, not the tile folders, holds the edits, and
    export never changes the asset folders.
- **On the web page a surface is drawn and kept in one step.** The surfaces screen
  (`/directions/{d}/generations/{g}/surfaces`, one per picture) shows the colours read
  from the box before anything is drawn (`decompose.read_box_shades`, which shares
  `decompose`'s checks and draws nothing), then draws the surface with
  `decompose.draw_surface`: it runs `decompose` for that one surface, keeps its tiles as
  assets (`promote_cutouts`), and then drops the same surface's tiles from older
  decompositions of the picture. `draw_surface(..., replaces=)` decides the name in the
  core: a new surface (no `replaces`) under a name already drawn on that picture is
  refused as `AlreadyExists`, "That name is taken. Pick another."; a `replaces` that
  names nothing drawn there is a `FileNotFoundError`. Edit posts `replaces`, the edited
  surface's slug: redrawing under the same name replaces its tiles, and under a new name
  the old surface is taken off afterwards through `remove_surface`, which keeps its
  master file. The command line's re-run behaviour is unchanged apart from the
  cross-picture refusal. The colours route reads the typed `name`, so its sentences
  name the surface (empty, they say "this surface"). Remove (`decompose.remove_surface`) takes the surface's tiles,
  outline and assets off the picture but never its master file, which holds the
  person's touch-ups. Every refusal comes back as the core's own sentence. The page has
  no decomposition review page, no rename or drop of pieces and no "Keep these pieces"
  step; the command line and MCP `decompose`, `keep` and `cut_into_pieces` are unchanged.
  Picking's "Continue to Surfaces" is a POST
  (`.../decompositions/{id}/pick/continue`) that keeps the redrawn props as assets and
  opens the surfaces screen; the timeline's Surfaces step links to that screen, and the
  direction page has no "Cut-up runs" list.
- **A kept scene gets a room file, a starting point and nothing more: props, surfaces and a soft structure layer.** Every
  exported asset of kind `scene` gets `<scene-name>-room.aseprite`
  (`export._write_rooms`, after `_write_tilesets`, and gated only on Aseprite
  being available: export always writes native size, and the room is built from the
  bundle's native PNGs), listed as `files["room"]` on
  that scene's manifest entry only once the file exists. The planning is
  `stages/room.plan_room`, with no Aseprite in it; `AsepriteRunner.build_room`
  and `aseprite/room_sprite.lua` draw it. A name already taken in the bundle is
  not overwritten and gets a note; a failed Aseprite run deletes the half file
  and adds a note, and never costs the bundle its PNGs or manifest. With no
  Aseprite there is the one existing warning and nothing more; with no kept
  scene there is no room. Only pieces with the scene's `source_generation`
  belong; pieces from other pictures are ignored. A prop is placed from
  `AssetRecord.decomposition` (filled by `promote_cutouts`; keeping a
  decomposition again fills it for older assets) and that decomposition's
  `cutouts/<name>/cutout.json` `origin`. A recipe surface needs its whole tile
  set in the bundle and `surfaces/<surface>/outline.png` (the filled shapes), positioned by the
  `bbox` on the surface's first tile's `cutout.json`. Anything from the same
  picture that cannot be placed (an old prop or tile, an asset kept before
  `decomposition` existed, a missing or unreadable record, a surface whose
  outline covers no cell) is left out with a "left out of the room: ..." note
  (`room._leave_out`), never guessed. A scene with nothing to put in a room is
  quiet: when no piece from its picture is a prop with a place or a recipe
  surface, there is no room and no notes at all, so the "left out" notes appear
  only alongside a room that has such candidates. "No room written" is said
  only when there were candidates but none could be placed (or, with surfaces only,
  the scene has no pitch).

  The scale `S` (original pixels per room pixel) is the median of the placed
  props' own scales, else the scene's `pitch`; with neither there is no room
  and a note. A prop's scale is its box on the picture (`origin.picture_box`;
  `origin` is needed for the box, not for `origin.scale`) over the native image
  actually placed, longer side over longer side. The canvas is the scene's
  `source.png` size over `S`, rounded. Each prop is one image layer named the
  asset, using its bundle native PNG. Its box centre and box bottom over `S` are
  rounded to whole room pixels first, then x is that centre minus half the
  image's width (whole-number division) and y is that bottom minus its height; props go back to front by box bottom (ties by name), all
  above the tile layers. Each surface is one tilemap layer named the surface
  slug, its tileset in the same order and with the same tile data as the
  per-surface file. A cell (the direction's `cell_size`, from 0,0) is painted
  when the original-picture pixel at its centre is inside the outline, where
  holes the outline fully encloses count as inside (`binary_fill_holes` in
  `room._plan_surface`), so a floor runs under a bed or rug the shapes left
  out; only the painting uses the filled mask, never the saved
  `outline.png` or the colour reading. Surfaces may overlap, and no rule is added
  for it: each surface paints every cell its own shapes cover, so where a floor and
  a wall overlap both layers hold those cells and the layer order below decides what
  shows. A person erases the unwanted cells in Aseprite. Each
  2 x 2 block of the map takes one block of the surface's eight, by an
  independent uniform pick seeded from the name (`recipes.room_layout`, its own
  seed so `sample_layout` is unchanged), so a block always stays whole. The
  surface with the most painted cells is the bottom layer. The room uses the
  bundle's own PNGs, so touch-ups in a surface's master file show in it.
  Between the tiles and the props sits one image layer named `structure`
  (`room._plan_structure`): the kept scene's own `source.png` shrunk to the room
  canvas by the room's own scale `S`, with an area-averaging (BOX) resize, so the
  frame, beams, door arch and stove pipe that no prop or surface holds are in the
  room. A shrunk pixel is made transparent when an opaque pixel of a placed prop
  lies over it or it falls inside a painted surface cell, so nothing is drawn
  twice, and also when its alpha is below `STRUCTURE_MIN_ALPHA` (128), which drops
  the faint glow halo outside the frame. The person types nothing for it and there is
  no palette snapping, so it is accepted as soft: a starting point to polish in
  Aseprite, not finished pixel art. It is written whenever a room is (same
  rule); the picture goes through a scratch file and that file never
  enters the bundle. Edits
  in a room file never flow back, and every separate asset file is unchanged.
  The export screen's room picture is `stages/room_preview.py`: it pastes the plan's
  layers in the room file's order with `room.compose_room` and no Aseprite. Before export
  it builds from a throwaway scale-1 bundle in a temp folder (`export(into=...)`, whose
  only caller in `src/` is the preview), which never lands in `exports/`. With `into`,
  export does not look for Aseprite, so no `.aseprite` is written and no master is read:
  the preview shows the recipe's drawn tiles, not master touch-ups. After export it reads
  the real bundle, so touch-ups show. Before export, the page's "In the bundle" list is
  that scratch export's own manifest entries (one row per asset and per recipe surface);
  after export, "Files written" is the real manifest's files. Either way the page holds no
  rules of its own.
- **Clean-up turns a picture into clean pixel art at native size, and has a web face only.** `stages/cleanup.clean_up` has no command line or MCP face, by CEO ruling of 2026-10-20; do not assume one. The web face is `web/cleaning.py` and `templates/cleanup.html`: the page `/directions/{d}/generations/{g}/cleanup`, reached by a "Clean up" button on every picture page. GET runs nothing and shows `stored_cleanup`'s last result. POST runs `clean_up` and redirects back to the page, with `?instant=1` (a fixed code, never text) when the core says it was a cache hit. The size box is refilled with the size the last run used; a found size (always one number) also goes in the hidden `found_size`, both in the float's shortest round-trip form (never the one-decimal text of the page's sentence), and a box still equal to it counts as no size typed, so a repeat is a cache hit while typing a rounded version of the found size is a real run (a typed size is refilled the same way, never rounded). A palette upload is copied under its own base name (a client can send a path) into a temporary folder that is deleted afterwards, worked or not; a file name of `.` or `..` counts as no file. "From a file" posted with no upload (a browser cannot refill a file input) while the stored result has a palette calls `clean_up(reuse_last_palette=True)`, and with no stored palette is refused. The page shows one of three lines, and decides none of them itself: for a typed size "Cleaned at the size you typed."; for a found size that is `unsure`, an orange "Not sure how big each pixel is in this picture. We used {size}. If the result looks too blocky or too smudged, type a different number and clean up again."; else "Each pixel is {size} picture pixels wide." (the size is a whole number when it is one, else one decimal). `cleaned.png` is served at `.../cleanup/cleaned.png`. The input is any generation, approved or not (clean-up is free and changes nothing else). The output is `directions/<d>/cleanups/<generation-id>/` (`Store.cleanup_dir`) holding `cleaned.png`, `cleaned.aseprite` and the `cleanup.json` sidecar (`CleanupRecord`). The `.aseprite` needs Aseprite; without it, or if Aseprite fails, a note goes in the sidecar and the PNG stays. There is one result per picture and a run with other settings replaces it (the old `.aseprite` is deleted first). A clean-up result is not an asset: refine, export, the journey and the room never see it. The grid finder of `stages/rebuild.py` always runs, so both axis readings and their scores (`found_pitch_x/y`, `found_score_x/y`) are always recorded. The finder moves from a fraction of the size up to a whole multiple of it (any that fits up to `rebuild.PITCH_MAX`) when that multiple scores at least `rebuild.MULTIPLES_SHARE` (0.90) as well; a reading at or above `rebuild.TOP_BAND_START` (13) scoring under `rebuild.WEAK_TOP_BAR` (0.10) is set aside as noise and the axis is picked again from below 13, unless that pick is only a whole fraction of the top reading, which is then kept so clean-up still flags it (#253). Pixels are square, so the result uses ONE size on both axes, recorded as `pixel_size`: the typed size, else the smaller of the two axis readings (a reading can land on a multiple of the true size, and a size too small keeps detail where one too big loses it; the rule is `rebuild.square_axes`, shared with the rebuild); each axis's phase and the grid confidence are then taken at that one size, so the result keeps the original's proportions. `unsure` is about the found readings, whatever size was typed: true when either axis scored below `rebuild.FOUND_BAR`, the larger reading is more than `READINGS_DISAGREE` (1.05) times the smaller (set on spike #240: soft AI-made figures 8.85 x 10.3 and 10.98 x 12.08 are not sure; exact enlargements 4x4, 5x5, 7x7 and 5 x 4.98 are), or either reading is within `NEAR_SEARCH_TOP` (0.5) of the finder's top size `rebuild.PITCH_MAX` (14), which it cannot see past (set on spike #242: three Codex frames read 13.9 to 14 for a true size of about 8.6; a true enlargement read that high is flagged too). The result is still made when unsure. The record also carries `shown_colours`, the result's distinct opaque colours as lowercase `rrggbb`, most used first, ties by hex. `cleanup.stored_cleanup(store, direction, generation_id)` reads the last stored result without running or opening the picture (None when there is none, the sidecar will not read, or the record's `code_version` is not `CLEANUP_VERSION`; `cache_hit` is True), so it can serve a stale result by design. A typed `size` replaces the found one and is refused outside 1 to the picture's shorter side. `colours` is a count of at least 1 (the default is `rebuild.MAX_COLOURS`); `palette_file` is an .aseprite; giving both is refused. `reuse_last_palette=True` takes the palette colours and the file's name from the picture's last stored result (same cache key as the original palette run, so an unchanged repeat is a cache hit and a changed size recomputes with the same colours); it is refused with a `ValueError` when there is no stored palette or when given with `palette_file` or `colours`. The palette is the distinct opaque colours actually drawn in every frame of that file as it looks (`read_frames.lua`, through `export.AsepriteRunner.read_frames`), not its palette table, which is often just Aseprite's default 32, so a palette needs Aseprite and is refused without it. Every refusal (`ValueError`) comes before anything is written. A picture with no solid pixel (none with alpha above `rebuild.OPAQUE_ALPHA`) is refused (#245). The step cache key (`cleanup.cache_key`) is a sha256 over the picture's bytes, the job name, the settings (size, colours, palette colours) and `CLEANUP_VERSION`, and is stored in the sidecar. A matching key with `cleaned.png` present returns the stored result and writes nothing (an unreadable sidecar is a miss). Bump `CLEANUP_VERSION` whenever the clean-up maths changes, or results made by the old maths are served as new. Reading a palette file still happens before the cache is checked.
- **An imported picture is recorded as imported, not drawn.**
  `stages.generate.import_picture` writes a generation whose `request.json`
  carries `imported_from` (the file's name) and no prompt, service or
  background, and never a made-up one; `result.json` has no service. A reader
  of those fields must cope with None (`retry_generation` refuses an import,
  `promote` skips a missing prompt, a kept import's asset and manifest entry
  say null). The picture is copied in as `image.png`, so the original file is
  never needed again.
  Keeping an approved picture afterwards as the scene is
  `stages/decompose.keep_scene`, reached by the command line's `keep-scene`, the
  MCP `keep_scene` tool and the picture page's "Keep as scene" button
  (`picture.html`; the web page's approve no longer has a keep-as-asset choice,
  keeping the scene is its own step; the MCP `approve` and the command line keep
  theirs). A whole picture kept as an asset is named by `naming.scene_name`
  (short: filler words dropped, at most 3 words, from the picture's subject: what the person typed, or the imported
  file's stem), clashes numbered by `unique_name`; `decompose.scene_asset_name`
  is that rule, used by `file_promotion`, so every face gets it and the web page
  can say the name before the person keeps the picture. It
  refuses a generation that is not approved (`NotApproved`) and one already
  kept as the scene (`AlreadyExists`), both before the refiner is called.

  A person can type the scene's name instead (`keep-scene --name`, the MCP
  `keep_scene` `name`, the page's name box, prefilled with the suggestion).
  `decompose.check_scene_name` is the one check, run before the refiner is
  called, after trimming surrounding spaces: `UnusableName` for a name `naming.check_name`
  refuses or that is not in `kebab_case` form (at most four words, 60 characters; the
  sentence offers that form), `AlreadyExists` for a name any asset, any cut-out, kept
  or not, or any recipe surface already has. A typed name is refused, never numbered.
  `decompose.rename_scene` (`rename-scene`, the MCP `rename_scene`, the
  picture page's Rename) renames a kept scene at any time, with the same check
  (renaming to its own name is refused), refusing a prop or tile first:
  it moves `assets/<old>` to `assets/<new>` and rewrites the `name` in
  `asset.json` and the `asset` in `refine.json`; nothing else names a scene (the
  rest use the generation id). An export bundle already on disk keeps the old
  name, as with `Store.rename_direction`; the next export writes the new name,
  in its manifest entry and its `<new>-room.aseprite`.
  Keeping one as a style reference is `stages/generate.keep_reference`, reached by
  the command line's `keep-reference`, the MCP `keep_reference` tool and the web
  page's `keep-reference` route. It reuses `promote` for the picture only (no prompt), refuses a
  generation that is not approved (`NotApproved`, defined in `stages/generate.py` and
  re-exported by `stages/decompose`) before writing anything, and returns False,
  writing nothing, when the picture already is a reference.
- **The scene is refined at a size PixelMagic chose; a prop and a recipe tile are
  not sent a size at all.** Keeping the scene (`keep-scene`, `generate --as-asset`)
  measures its pitch (source pixels per art pixel) with the refiner's `analyze`
  (`decompose.prepare_promotion`; the reading is real only under
  `--refiner cli`), and the scene's `target` in art pixels is its size over that
  pitch. The pitch and target go onto `asset.json`, refine puts the
  target on the `RefineRequest`, and the CLI adapter turns it into
  `forcePixelsW/H`. Drop the target anywhere along that chain and the scene goes
  back to being refined at its own guessed scale. The stand-in ignores the target
  on purpose (locked decision 9): a stand-in bundle carries targets its
  images do not match, and `manifest.json`'s `refiner_tool` is how it says so. The
  sums live in `stages/layout.py`; do not repeat them in a face or a stage. A person
  can overrule the scene's pitch with one number from any face (`set-scene-pitch`,
  the `set_scene_pitch` tool, the refine screen). `set_scene_pitch` refuses a pitch whose
  target would be over `MAX_FORCE_PIXELS` before writing anything (a person sets a
  pitch for the real refiner). The pitch is filed
  with a `PitchSource`: `measured` (the refiner's reading), `set` (a person's number),
  `unsure` (the refiner flagged its own reading with `LOW_GRID_CONFIDENCE`; the value is
  kept) or `fallback` (no reading, or one outside `MIN_PITCH`..`MAX_PITCH`, 0.5 to 64;
  1.0 is used; a refiner that does not honour a forced size, the stand-in, measures
  nothing, so its reading is also filed as `fallback`, without a log line).
  PixelMagic sets no confidence threshold of its own. For `unsure` and
  `fallback`, `decompose.pitch_note` gives a sentence asking the person to check the
  pitch and set it; the command line and MCP keep paths print it after their success
  line, and the refine page and the manifest's `pitch_source` carry the label. A rebuilt prop's size is what its
  grid gives, one pixel per art pixel, saved as `source.png` and never resized; its
  `target` is that native size, and its `footprint` and default `anchor` come from
  it (`decompose.layout_cutout`). A recipe tile's target is the
  cell size.
- **One bad asset record or target costs one asset.** In `refine_direction`,
  an asset whose target is unusable gets its own `TARGET_NOT_USABLE` receipt,
  and one whose `asset.json` will not parse gets its own
  `ASSET_RECORD_UNREADABLE` receipt. The rest of the direction still refines.
  A single named-asset refine (`refine_asset`) raises instead, naming the
  asset, since there is no direction left to keep going for. Those codes are
  PixelMagic's own refusals (`PIXELMAGIC_REFUSAL_CODES` in
  `stages/refine.py`): the refiner never saw the asset, so no face may say
  the refiner refused it. `PICTURE_UNREADABLE` is the third of them: a rebuilt
  prop's own `source.png` would not open. A rebuilt prop never reaches the
  refiner, so that is PixelMagic's own finding too; `refine_direction` writes
  the receipt for that one asset and keeps going, and `refine_asset` raises.
  `OLD_ITEM_REFUSED` is the fourth (see the old-items rule above). `WRONG_SIZE`
  is not among them: the refiner did see that asset (see the size-check rule below),
  so it is a per-asset receipt in `refine_direction` like the others, and
  `refine_asset` reports it the same way.
- **A forced size is at most 1024 per side** (`MAX_FORCE_PIXELS` in
  `refiner/base.py`). The real PixelRefiner silently clamps larger
  `forcePixelsW/H` values, so a bigger target is refused rather than sent, at
  refine time, when the refiner in use honours a forced size
  (`Refiner.honours_forced_size`; `refine.validate_target`'s `honours_forced_size`).
  The stand-in ignores the target, so it refines a scene over 1024 at its own
  size and only `MAX_IMAGE_SIZE` bounds it (#295). `set_scene_pitch` has no
  refiner and keeps the 1024 limit. A
  batch sends the size only as `forcePixelsW/H`, never together with
  `--grid force`, because the CLI rejects the two together (spec §3
  amendment, 2026-09-23). With a target, `scale`, `makeSquare`, `keepAspectRatio` and
  `outlineStyle` are left out of the call (each moves the output size on the real
  CLI) and the receipt carries a `SIZE_SETTING_DROPPED` warning (in
  `PIXELMAGIC_WARNING_CODES`, worded for a person through `ACTIONABLE_WARNINGS`):
  dropped with a warning, not refused, because they are settings a person saved. A
  `scale` aimed at the scene from MCP or the command line is still dropped with
  `SIZE_SETTING_DROPPED`; the web page no longer offers it.
- **A scene refined at a forced size is checked at that size.** When
  `Refiner.honours_forced_size` is True (the real refiner; the stand-in says False,
  locked decision 9), a result at any size other than the target becomes a `WRONG_SIZE`
  failure receipt and its `refined.png` is deleted. `WRONG_SIZE` is not in
  `PIXELMAGIC_REFUSAL_CODES`, since the refiner did see the asset and answered. Every
  face words it with `refine.wrong_size_sentence`, which says the refiner returned
  the wrong size so PixelMagic did not keep it. A grid candidate re-pick aimed at the
  scene is refused by `refine_asset` with a `ValueError` ("A grid candidate cannot be
  used on the scene ...: ... Set the scene's pitch instead."), before the refiner runs
  or anything is written, because the real refiner lets a candidate win over the forced
  size. The web Refine page offers the scene no candidate buttons and no "Scale the
  result up by" field; the MCP `refine` tool passes the refusal on as a `ToolError`; the
  command line has no candidate option.
- **Project = a game. Direction = one art direction inside it.** Assets, generations and exports all hang off a direction.
- **Timestamps are UTC** and ids are timestamp slugs like `20260903-191233`. Every timestamp field is an `AwareDatetime`, so a naive `datetime` is refused rather than filed as if it were UTC.
- **One broken asset never costs the bundle.** `export` skips an asset whose sidecar will not parse or whose `refined.png` will not open, adds a warning naming it and the problem, and carries on. The bundle folder only ever holds files `manifest.json` lists.
- **The object finder is a subprocess in its own folder and environment.**
  `segmenter/` holds the real one (Segment Anything 2.1 small, MLX build); the
  core runs the `pixelmagic-segmenter` command and reads its JSON
  (`segmenter/cli_adapter.py`), exactly as it does PixelRefiner. Nothing in
  `src/pixelmagic` imports `sam_helper`, `sam2` or `mlx`. The MLX port is
  vendored unchanged in `segmenter/vendor/sam2` with its `NOTICE` (source,
  commit, and that the licence is stated but not shipped). The contract (verbs,
  JSON, exit codes) is the docstring of `src/pixelmagic/segmenter/base.py`.
  Rules about what to do with outlines never go in the helper. A machine
  without the helper finds out on first use as a `SegmenterError`, not when
  the name `cli` is chosen (`registry.SEGMENTERS`).
- **Props are cut out object by object, by one of two roads: suggest, pick, redraw in groups of five,
  cut back to one piece per object; or, on a picture that is already pixel art, the pixel cut (below).** The first road:
  `stages/suggest.py` asks the segmenter for
  every outline, drops duplicates (`DUPLICATE_IOU`), numbers the rest largest
  first and pages them; `stages/pick.py` saves a choice the moment it is made
  (number, point, box; `picks.json` and `picks/<n>.png`; a number is never
  reused, `PickSet.last_number`, and Undo never lowers it). Every change to a
  pick's outline (new pick, fix, lasso pick, lasso fix, unpick) also records an
  undo entry (`PickSet.undo`, `PickUndo`, mask copies in `picks/undo/`, at most
  `UNDO_LIMIT = 20`; `record_undo`, `undo_last`, `clear_undo`); a rename records
  nothing and Undo never reverts a name, and a redraw empties the list; `stages/redraw.py` splits picks (`groups.group_picks`) into groups of
  at most `groups.GROUP_SIZE = 5`, draws each group on one reference picture with each
  object cut through its own outline, asks the service once per group, and cuts
  the reply back with `groups.cut_back`, whose own rule is one blob (joined
  parts included) per cell. A picked object holds exactly its own outline.

  **A size-true reference** is drawn when the direction has a `prop_scale` (#147;
  `Direction.prop_scale`, set by `redraw.set_prop_scale`, core only with no face yet:
  it refuses a height under 1, an unknown pick, or a scale outside the pitch bounds).
  Each object's wanted size is its box over the scale (`layout.target_for`);
  `groups.build_sized_reference` draws every object pixelated at that size, in
  solid blocks `BLOCK_WIDTH` wide, on the key-colour canvas (`SIZED_CANVAS`). An
  object whose wanted longer side is under `SMALL_LIMIT` is drawn at
  `SMALL_COMPENSATION` of its size (`groups.drawn_size`). When the biggest object
  does not fit its cell, the group's block width drops until it does, and a group
  that would not fit at 1 is refused with a `ValueError`; `redraw_picks` makes that check
  first (`groups.fitting_block_width`), before any paid call (the naming call included),
  before the undo list is cleared and before anything is written. The prompt's `{sizing}`
  lines (`groups.sizing_rules`, fixed wording and computed numbers) list each
  object's drawn size and ask for it back; with no scale `{sizing}` is
  `generate.GROUP_SIZING`, today's line, and the reference is `build_reference`'s.
  Only this reference is resampled, never the reply, the cut-back or the rebuild.

  One step follows the cut-back: each piece is rebuilt on its own pixel grid by
  `rebuild.rebuild_piece` (`stages/rebuild.py`; `redraw._land_group` calls it).
  The finder reads the grid off the piece and the piece is saved at native size
  as `GridRebuild` records it (`Cutout.grid_rebuild`, copied to
  `AssetRecord.grid_rebuild` when kept). Pixels are square: the piece is sampled
  at ONE size on both axes, the smaller of the two axis readings, with each axis's
  phase taken at that size (`rebuild.square_axes`, the one rule shared with
  clean-up); `GridRebuild.pitch_x` and `pitch_y` both hold that one size. A grid is
  still found only when both axes' own readings score at least `FOUND_BAR`, and a weak reading at the top of the range
  (`TOP_BAND_START`, `WEAK_TOP_BAR`) is set aside as noise; the piece is padded by `MARGIN` pixels and laid
  on one flat dark colour (`FILL_COLOUR`) first, the input the bar was set on.
  A piece never has more than `MAX_COLOURS` colours, with at most
  `SHADES_PER_GROUP` shades per colour group, and `STRAY_DISTANCE` decides
  which single cell counts as a stray. Refine and export recognise a rebuilt
  piece by `grid_rebuild` alone. A piece the rebuild refuses is not a piece:
  the object stays picked. The numbers live in `rebuild.py` under those names
  and nowhere else.
- **`BACKGROUND_SHARE = 0.2` (`stages/suggest.py`).** A suggestion covering more
  than a fifth of the picture is the room, a wall or the floor: it is dropped
  before duplicate removal and before numbering. On the den that drops only the
  whole-room mask.
- **The 12 px guard lives only in `stages/pick.py`.** `POINT_GUARD_PIXELS = 12`
  and `dedupe_points`: a click within 12 px of an earlier click with the same
  label is ignored (repeated near-identical points make Segment Anything jump
  to another object, the shift-click bug the M5 spike met); the other label at
  the same spot is kept. `pick_points`, `pick_box`, `fix_pick` and `add_clicks`
  all reach it through `_ask_segmenter`. `lasso.fix_lasso` does not call
  `dedupe_points` but uses the same constant against earlier fixes of the same
  kind since the last fix of the other kind, so a piece removed and then clicked
  again still comes back. Never write a second 12.
- **The lasso rule lives in the core (`stages/lasso.py`).** The segmenter is
  asked for the piece under many points inside the line; a piece with more than
  `EDGE_OUT` (15%) outside the line is background unless it covers the loop's
  middle, pieces mostly covered by background are dropped too, and the object is
  what is left, clipped to the line. A fix-up click (`fix_lasso`) adds or
  removes one piece whole and is not clipped. The page's script only records the
  drag.
- **The pixel cut is the second road to a prop, for a picture that is already pixel art.**
  `stages/pixelcut.start_pixel_cut` opens a decomposition carrying `pixel_grid`; picks are
  made on it with the usual pick functions, and `pixelcut.cut_pieces` turns them into
  props. No image service and no PixelRefiner are involved.
  - The faces. Command line: `pixel-cut <generation-id> [--size N]` starts it and prints
    the run id, the pixel size and each core warning as a `note:` line (so
    `UNSURE_SENTENCE` reaches the person); `cut <run-id> [--only N ...] [--keep/--no-keep]`
    calls `cut_pieces` and keeps by default, like `redraw`. Box picks and fix-up clicks
    use the existing `pick` command; on a pixel cut, `pick --fix` points to `cut --only <n>`
    and `picks` says unnamed picks are "named when cut". MCP: `start_pixel_cut` and
    `cut_pixel_pieces` (keeping is left to `keep_pieces`); `fix_pick` and `pick_objects`
    point to `cut_pixel_pieces` on a pixel cut, never to redraw, and `show_picks` says unnamed
    picks are "named when cut". The import hints on the command line and MCP also offer the pixel cut (the
    command line prints an "already pixel art? pixelmagic pixel-cut <id>" line; the MCP import
    reply mentions `start_pixel_cut`). Web: the picture page's "Already pixel art?" section links to
    `/directions/{d}/generations/{g}/pixel-cut`, the size step (GET starts nothing; POST starts the
    run with `start_pixel_cut`, "Find it for me" with an empty size or a typed one) and redirects to
    the run's picking address `/directions/{d}/decompositions/{id}/pick`, which `picking.picking`
    hands to `pixelcutting.pick_screen` for a run carrying `pixel_grid`; the same address with
    `?cut=1` shows the props cut when at least one pick has a piece, else the picking screen. On that screen a drag-box picks (`pick/box`, the core's
    `pick_box`); Add and Remove (or holding Cmd or Ctrl for Add, Option or Alt for Remove; Shift does nothing here) fix a spot through the picking page's own
    `picks/{n}/points` route; Undo, rename and remove are the picking page's routes. A single click
    does not pick. There are no numbered suggestions, hover outlines or lasso on this road. Picks
    are labelled with the name the cut will give them (`pixelcut.piece_names`: prop-1, prop-2...,
    made unique, so a clash can give prop-1-2); a pick can be renamed until it is cut, after which
    rename is hidden and the core refuses it. "Cut N props"
    (`cut`, the core's `cut_pieces`) is free; "Continue to Surfaces" is the picking page's
    `pick/continue`. "Start again with another size" goes back to the size step and a new run is
    made, leaving the old one on disk. Core changes this needed: `decompose.is_props_run` counts a
    run carrying `pixel_grid`; `render_tint` takes the picture size from the pixel grid
    (`pick._picture_size`, which the other pick functions already used); `journey.props_runs`
    gives the newest run of each road to the picture page's two links; `pixelcut.unsure_note`
    finds a run's stored unsure-size sentence by `UNSURE_PHRASE`. `pick.rename_pick` already
    refused a pick with a piece; only its sentence changed. The refine summaries on the command line and MCP name pixel-cut props (`RefineTally.pixel_cut`).
    `decompose.missing_run_sentence` takes an optional `making` wording (default "Suggesting
    objects on the picture"; the command line and MCP pass "Starting a pixel cut on the picture") so "that is a
    picture id, not a run id" can name the pixel-cut step.
  - The pixel size is clean-up's finder rule (`stages/pixelgrid.read_grid`), or a typed
    size, where 1 means the picture is already native. A typed size outside 1 to the
    picture's shorter side is refused before anything is written. A found size the finder
    was unsure of adds a warning to the run (`pixelcut.UNSURE_SENTENCE`).
  - The object finder always runs on a nearest-neighbour copy of the native picture
    enlarged `pixelgrid.FINDER_SCALE` (2) times (`finder.png`), through
    `pick._ask_segmenter`, the one place every pick function goes through. The outline comes
    back as whole cells (`rebuild.CELL_IN_MASK`).
  - A piece is `native.png`'s own pixels inside the outline, never resampled or recoloured.
    Its `pitch_source` is `set` for a typed size, `measured` for a found sure one, and
    `unsure` otherwise. A kept pixel-cut prop records no `service` (null on its asset and
    its manifest entry), like a recipe tile: no image service drew it.
  - `redraw_picks` refuses a pixel-cut run.
  - Not on this road, on any face: numbered suggestions and the lasso. Not done by it: removing
    another pick's pixels from a piece, recovering hidden parts, filling the hole.
- **A prop's pitch is its own.** A redrawn piece is on the group reply's own
  pixels. Its pitch is its object's box on the original picture over its native
  size, and `Cutout.origin.scale` is that same number
  (`layout.scale_picked_box` sizes a new piece). There is no props pitch to set
  and no one reading shared by the props; `pitches` in a manifest holds the
  scene's only. `RebuiltPieceRefused` is what a refine setting or a candidate
  re-pick aimed at a rebuilt prop gets (see below).
- **A rebuilt piece drawn at the wrong scale is flagged, never changed.**
  `redraw.check_scale` runs at the end of every redraw. A rebuilt
  piece's ratio is the longer side of its native size over the longer side of
  its object's box on the original (`origin.picture_box`); every rebuilt piece
  of the decomposition counts, not only this call's. With no `prop_scale`, with at
  least three rebuilt pieces, one more than `SCALE_TOLERANCE` (15%) of the median
  away from the median of the ratios gets one warning naming it and the factor
  (`SCALE_PHRASE`). Under three it stays quiet. With a `prop_scale`, each piece is
  instead compared with its own wanted size (`layout.target_for` of its box, longer
  side), with no minimum count, and one more than `SCALE_TOLERANCE` off it gets a
  warning with `TARGET_PHRASE`. Every earlier scale warning, found by `SCALE_PHRASE`
  or `TARGET_PHRASE` in its text, is cleared first on every run, even a quiet one,
  since a later redraw moves the median. No piece or file is touched. Known gap:
  dropping or unpicking a piece leaves the scale notes as they are until the next
  redraw.
- **`decompose` never makes props.** It makes tiles for the surfaces typed, and
  raises `NoPicks` when no surface is typed. It has no choice of kind. A face
  words its own way forward to `suggest`; the refusal is the core's.
- **A group redraw is auto-approved, and a failed group costs only its own
  objects.** `redraw._call_group` uses `AutoApprove`: a group is not a picture a
  person reviews one at a time. A group that fails both attempts (`ATTEMPTS`)
  leaves its objects picked with no piece and a warning each, so `redraw` with no
  `--only` picks exactly those up again. An object whose grid is not found costs
  only that object the same way: it stays picked with no piece, a warning names it
  (with the weaker axis's score against `FOUND_BAR`), and a plain `redraw` tries
  it again. A piece whose grid was found but whose cells are all too thin to keep
  (`PieceTooThin`) gets its own true sentence, not the "no grid" one. A failed
  `--only` retry never costs a piece the object already has; the older piece
  and its files stay. A new failure note for an object (no grid, too thin, group
  not redrawn, cell left empty) replaces that object's earlier failure notes of
  all four kinds (`redraw._clear_failure_notes`), and a note that is not a failure
  note, such as a cut-back share note, is left alone. The group's reference stays on an opaque
  magenta background (`groups.GROUP_REFERENCE_KEY`, CEO ruling on #110); the
  reply is asked for transparent first, key colour second, as everywhere.
- **Refine and Export follow a scene the person chooses, by generation id.**
  `journey.followed_scene` is the one answer (the Refine and Export pages, their
  room preview and their timeline use it). The choice is stored per direction in
  `directions/<direction>/scene-choice.json` (`SceneChoice`, `Store.scene_choice_file`),
  holding the scene picture's generation id, not the asset name, so renaming
  the scene cannot make it stale. `journey.choose_scene` records it and refuses
  (`FileNotFoundError`) a generation that is not a kept scene of the direction.
  With no usable choice the default is the newest kept scene that has pieces (a
  decomposition of it holds a rebuilt or pixel-cut prop, or a recipe tile, the same test that
  marks Pick props and Surfaces done), else the newest kept scene, else with
  none kept the newest scene picture. A stale choice (scene deleted) or an
  unreadable sidecar silently falls to that default and never raises.
  `journey.kept_scenes` lists the kept scenes for the chooser. Pages about the
  whole direction (`latest_scene`) and the per-picture pages are not affected.
- **The web page is local-only.** It binds to loopback and refuses any other
  host. It shows one person's whole project folder and has no login; remote and
  hosted use are out of scope for v1. A POST, PUT, PATCH or DELETE that
  `Sec-Fetch-Site`, `Origin` or `Referer` says came from another site is refused
  with a 403, and a Host that is not a loopback name with a 400
  (`web.app.LocalOnlyGuard`); a request with none of those headers is allowed.
  The page takes its look
  from `DESIGN.md` and `JOURNEY.md` at the repo root. Silkscreen is for the
  wordmark only. Nothing loads from another host. Pixel-art previews (the Refine
  page's scene once refined, Before and After both sized from the refined
  picture; the Export room preview; the cleaned picture on the clean-up page) are
  sized by `PixelMagic.wholeZoom` in `static/pixelmagic.js` to the largest
  whole-number zoom that fits, and left fitted to the panel when even 1x does not
  fit; screens that show only the raw drawn picture (picking, surfaces, the
  picture page) are not zoomed.
- **The page's routes, and where step states come from.** `/` is the game's home,
  showing its art directions as cards. A folder with no `pixelmagic.json` gets a
  404 page that names `project new`. The new-direction form is at
  `/new-direction`. The state of each step on the timeline comes from
  `journey.py` in the core, never from a template.
- **No page ever shows a traceback.** Every failure the core raises —
  `ApprovalExhausted`, `RefinerError`, a `ValueError` for settings the refiner
  will not take, a `FileNotFoundError` for something that is not there, an
  `AlreadyExists` for a name in use — has a handler in `web/app.py` that turns
  it into a sentence and a status code. That includes `SegmenterError` (the
  helper is missing, its model is not fetched, or it failed), which gets its
  own handler ahead of the `RuntimeError` one: a 400 "The object finder could
  not do that", never a traceback. The catch-all deliberately does not
  repeat an unplanned exception's message: that message is an internal detail,
  not an explanation.
- **The settings screen reads the real refiner, never the stand-in.**
  `StubRefiner.options()` answers out of the contract's own constants, so a
  screen built on it would offer keys PixelRefiner does not have and hide every
  palette it does. `create_app` takes the options source separately from the
  refiner that does the work. With the stand-in doing the work the options source
  is None: the screen starts nothing and says the real PixelRefiner is not turned
  on and that `--refiner cli` turns it on. With `--refiner cli` the work refiner
  is the options source, and when it cannot be reached the screen says so rather
  than falling back.
- **The MCP server prints nothing to standard output.** On stdio, standard
  output is the protocol. Log to standard error.
- **An MCP tool reports a failure as a `ToolError`.** The SDK withholds the
  message of any other exception — the agent is told "Error executing tool X"
  and nothing it can act on — so every tool body runs inside
  `mcp_server.readable_errors()`.
- A rule every face needs lives in the core, not in `cli.py` (locked decision
  7). The already-exists guards, the safe-name check and the service lookup
  are all in the core for exactly this reason; adding the next one to `cli.py`
  is how the three faces start disagreeing.
- **A name a person types is refused when too long, never shortened.** A direction,
  piece, pick or surface name goes through `naming.typed_slug` (a scene name through
  `decompose.check_scene_name`); over `MAX_NAME_WORDS` words or `MAX_NAME_LENGTH`
  characters it is `UnusableName`, and the sentence offers the shortened form to type.
  Names PixelMagic makes up still shorten.
- A service must never raise anything but `providers.base.ServiceError`. The
  generate stage removes the folder it made when one does, and it can only do
  that for a failure it expects.
- `capabilities().transparent_background` is the only thing that decides
  whether a request asks for transparency (locked decision 6). A service that
  lies here breaks the key-colour fallback for everybody.
- **Only the kept scene is refined, with the house-style cohesion settings and
  its own 24-colour palette, fitted before cohesion touches the picture —
  whether refined through `refine_direction` or by name through `refine_asset`.**
  This applies under `--refiner cli`; by default the stand-in copies the scene through.
  `refine_direction` refines the scene on its own (and a bare
  folder with no `asset.json`, with the plain single call); `refine_asset`, asked
  for the scene by name (a candidate re-pick aimed at the scene is refused before
  anything runs), runs the identical recipe, so the two ways of refining the scene never
  disagree. When the resolved settings ask for a colour count, that refine is two
  calls — `Refiner.fit_palette` fits the palette on the untouched source with
  cohesion and clean-up off, then the real refine pins that palette
  (`reduceColorMode: "fixed"`) with cohesion on (`stages/refine.py`,
  `_refine_with_own_palette` — the M4.1 spike's own recipe, the only one that
  reproduced its numbers; a single `reductionMode` call after cohesion does
  not). A named palette, `auto` or `none` has nothing for a fresh fit to
  reuse "own" from, so those keep the plain single call, and so does a call
  that also names a `candidate_id`: the recipe's final call is a `batch`,
  which has no per-input candidate flag and would silently drop the re-pick.
  Since the scene refuses a candidate, that last case can now only be a bare folder
  with no `asset.json`, the one place a candidate is still accepted.

  When the refiner was unsure of the grid (`LOW_GRID_CONFIDENCE`), the rule for
  whether to tell the person lives in the core, in
  `refine.grid_needs_attention` (and `stored_grid_needs_attention`, which reads the
  asset's sidecar itself): it flags only an asset whose pitch a person can still set,
  never an old item or a rebuilt prop. The web Refine page ("Set the scene pitch under
  More options") and the MCP `refine` note (pointing to `set_scene_pitch`) both use it.

  No `background` flag is sent on either call for the scene: its `source.png` is
  still the generator's raw, uncropped frame, and the refiner's own background
  removal is what clears it (`background: "keep"` there left a den export with an
  opaque surround).

  The defaults themselves (`reductionMode: "24"`, `cohesion`,
  `cohesionFamilies`, `cohesionTones`, `cleanup`) live in
  `stages/refine.HOUSE_BASELINE`, layered underneath a direction's own
  `refiner_defaults` at refine time — never inside `set_refiner_defaults`'s
  persisted merge, so an existing direction's `direction.json` is never
  rewritten just because it was refined; a new direction stores a deep copy
  of the baseline outright. The `analyze` call that measures the scene's pitch
  always sends `cohesion` and `cleanup` off (`decompose.PITCH_ANALYZE_SETTINGS`,
  locked decision 19): a pitch reading measures the original drawing, not
  PixelRefiner's own redraw of it. `refine.json` and every `manifest.json`
  entry carry a `cohesion` record of what actually ran, read off the
  refiner's own `effectiveOptions` (with `is True`, not `bool(...)`, so a
  stray string never reads as "ran") — never off what was asked for, since a
  per-piece override can disagree with the direction's default — and a
  stand-in bundle's record always says neither stage ran, since the
  stand-in changes no pixels and its own `fit_palette` just reads the
  picture's existing colours.

  A rebuilt prop is outside all of this. It is not refined: no palette fit, no
  cohesion, no clean-up, no refiner call of any kind; `refine.py`'s
  `_keep_rebuilt` copies its `source.png` to `refined.png` and writes
  `refine.json` with route `rebuild`, and its colours come from the rebuild
  (never more than `rebuild.MAX_COLOURS`, not the 24-colour palette). Its
  `cohesion` record says neither stage ran. `refine_direction` skips rebuilt
  props without refusing and the report counts them; `refine_asset` by name
  refuses a rebuilt prop only when given a non-empty setting or a candidate
  re-pick, with `RebuiltPieceRefused`, because neither could apply; plain
  `refine_asset` by name keeps it through `_keep_rebuilt`. "Refined N" counts neither rebuilt nor
  refused assets; `refine.tally_reports` is the one place that counting lives (`RefineTally` has a `pixel_cut` count beside `rebuilt`, `recipe` and `old_items`) (the
  command line and the MCP tool both use it), and the command line exits 1 on any refusal. The stand-in does not refine a rebuilt prop either, so an
  offline run's rebuilt props say `pixelmagic-grid-rebuild`, not `stub`: they are
  real rebuilds of the fake service's stand-in board.

## Pointers

- The design spec: `docs/superpowers/specs/2026-09-03-pixelmagic-design.md`. Its "Locked decisions" section is a contract — change it only through a dated amendment agreed with the CEO. `docs/superpowers/` (the design spec and plans) and `.design-foundations/` (the design research and mocks) exist only in the project's private working repository and are not in the public copy; every mention of them in this file points there.
- The PixelRefiner command-line contract (commands, settings layers, result
  fields, exit codes, and what `--grid force` / `forcePixelsW/H` really mean)
  is written out at the top of `src/pixelmagic/refiner/base.py`. That
  docstring is the source of truth for the adapter.
- The name a face was given — `--service codex`, `--refiner cli` — is turned
  into an object by `registry.py`, which raises plain `ValueError`s. `cli.py`'s
  `load_service` and `load_refiner` are two-line wrappers that turn those into
  a usage error; the web page and the MCP server translate them their own way.
  A new service is registered by adding one entry to `registry.SERVICES` and
  deleting the name from `registry.PLANNED_SERVICES`, and it is then reachable
  from all three faces at once.
- A verdict does not have to arrive inside the call that drew the picture.
  `approval.PendingApproval` records that the question is open and
  `stages.generate.record_decision` closes it later, which is how a browser
  and an agent approve without a background thread.
- The Codex contract — the command, the events, where the pictures land — is
  the module docstring of `providers/codex.py`, read off the binary on
  2026-09-04. The headless driver the spec originally pointed at no longer
  exists.
- The object finder's contract (the two verbs, their JSON, exit codes, what
  the helper must not do) is the docstring of
  `src/pixelmagic/segmenter/base.py`; setup is in `segmenter/README.md`. The
  plan for cutting props object by object is
  `docs/superpowers/plans/2026-09-30-object-cutting.md` (private working repo;
  built in M5, tickets #110 to #115); where it and this file disagree, this
  file is what was built.
- `Refiner.fit_palette` (`refiner/base.py`) is the seam's one addition for
  the scene's own-palette recipe: a palette fit on one untouched picture, no
  forced size, no cohesion. It reads a batch reply's top-level
  `sharedPalette`, never a single image's own `output.palette` — the two can
  list the same colours in a different order.

## Working rules

- Never commit to `main`. Work on a branch named `feature/…`, `fix/…`, `refactor/…` or `docs/…`.
- Every pull request is reviewed by `code-guardian` before it merges — no exceptions for small changes.
- Commit messages explain *why*, not *what*.
- Write docstrings on every module, class and function. Add comments wherever the logic needs explaining.
- No dead code. Delete it rather than commenting it out.
- Legibility beats cleverness.
