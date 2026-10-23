# PixelMagic

PixelMagic turns a text prompt into game-ready pixel art.

You describe an art direction once. After that you get a scene (a picture an image service draws, or one of your own), look at it, and say yes or no. When you say yes, you pick the objects you want out of it, PixelMagic has each one redrawn and rebuilt as real pixel art, draws floors and walls from the picture's own colours, turns the scene itself into pixel art, and hands you a folder of PNGs, a `manifest.json` your game can read, and Aseprite files so you can still touch anything up by hand.

## Install

You need two things:

- **Python 3.12 or newer**
- **[uv](https://docs.astral.sh/uv/)** — the package manager this project uses

```bash
uv sync
uv run pixelmagic --help
```

That is enough to run the whole tour in "Five minutes with PixelMagic" below, offline. Everything else is optional, and each piece unlocks one thing:

- **An image service** — unlocks drawing: scenes from a prompt, and redrawing the objects you pick. Without one, you can still bring your own picture. You only need the one you choose.
  - **Codex** needs no key and no extra package. Install the `codex` command and log in; check with `codex exec "say ok"`.
  - **Gemini**: `uv sync --extra gemini`, then a key (`GEMINI_API_KEY`, or paste it on the Services page).
  - **OpenAI Images**: `uv sync --extra openai`, then a key (`OPENAI_API_KEY`, or paste it on the Services page).

  Choosing one, and the keys, are covered in "Choosing who draws, and adding a key" below.
- **The real object finder** — unlocks finding the objects on a picture by itself (Segment Anything 2.1, an MLX port, so Apple silicon only). Set it up once: `(cd segmenter && uv sync && uv run pixelmagic-segmenter fetch-model)`. That downloads a 184 MB model. Without it, `--segmenter stub` finds each separate blob of opaque pixels instead, and `suggest` stops with a message until you use the stub or set the finder up. More in "Picking the objects yourself".
- **Aseprite** — unlocks the `.aseprite` files: each item's file, a surface's tileset (and the master file you touch up), the room file, and clean-up's `cleaned.aseprite`. Aseprite is paid software and is not open source; PixelMagic never downloads or bundles it, it only finds one you already have. PNGs, `manifest.json` and everything else work without it. If it is not in the usual place, set `PIXELMAGIC_ASEPRITE` to its full path.
- **PixelRefiner** — unlocks true scene refining. PixelRefiner is a separate open-source project (MIT), not part of this repo. True scene refining needs it; the pixel cut (command line, MCP and web page), recipe tiles, clean-up (web page only) and export all work without it. Without it the scene is copied through unchanged by a stand-in (the PNG you get for it is the concept art, not pixel art). To turn it on, pass `--refiner cli` (to `refine`, `keep-scene`, `generate`, `web` or `mcp`); PixelMagic then runs the `pixel-refiner` command found on your PATH, or the command in `PIXELMAGIC_REFINER_BIN`. To get it: clone <https://github.com/mauricio-gg/PixelRefiner> (a fork of HappyOnigiri/PixelRefiner), install Node 24 or newer and `pnpm`, then from the clone run `pnpm install` and `pnpm --filter pixel-refiner-mcp run build` (the build output is not in the repo, so rebuild after pulling changes). Then either put a `pixel-refiner` command on your PATH, or set `PIXELMAGIC_REFINER_BIN` to `node <clone>/packages/mcp/bin/pixel-refiner.mjs`; PixelMagic reads that value the way a shell would, so a command with arguments works.

## Environment variables

None of these are needed to try PixelMagic out — the built-in fake image service (`--service fake`) runs offline. They matter once you use a real image service.

## Choosing who draws, and adding a key

Open the **Services** page (the gear link at the top right of every page of the web page). It lists the three image services: Codex, Gemini, OpenAI Images. Until you choose one, nothing draws and the Draw buttons say "Choose an image service first". Pick the one that should draw for this game; the choice is saved in the project's `pixelmagic.json`, so each game has its own. For Gemini and OpenAI, paste your key into the field and press **Save key**, then **Test key** to check it (the test is a free call that draws nothing). Codex needs no key; **Check login** tells you whether you are signed in.

Keys live in your computer's keychain, never in the project folder, and are never shown again after you save them. Every game on this computer can use them. The environment variables below still work, and a key set in the environment wins over one saved on the page. `--service <name>` before a command still works too, and wins over the game's saved choice for as long as that server or command runs; without it, the command line, the web page and the MCP server all follow the choice saved on the Services page.

| Variable | What it is for |
|---|---|
| `OPENAI_API_KEY` | The OpenAI image service. |
| `GEMINI_API_KEY` | The Google Gemini image service. `GOOGLE_API_KEY` works too; set one, not both. |
| `PIXELMAGIC_ASEPRITE` | Full path to the Aseprite program, if it is not in the usual place. |
| `PIXELMAGIC_REFINER_BIN` | The PixelRefiner command to run with `--refiner cli`, if it is not simply `pixel-refiner` on your PATH. Split like a shell would split it, e.g. `node /path/to/pixel-refiner.mjs`. |
| `PIXELMAGIC_CODEX_BIN` | The `codex` command, if it is not simply `codex`. Split like a shell would split it. |
| `CODEX_HOME` | Where Codex keeps its files, if it is not `~/.codex`. PixelMagic reads it to find the pictures Codex made. |

The Codex image service needs no key. It drives the `codex` command, which has
to be installed and logged in already — check with `codex exec "say ok"`. If
your `codex` lives somewhere unusual, or you want it to use a particular
Codex profile, set `PIXELMAGIC_CODEX_BIN` to the whole command, e.g.
`PIXELMAGIC_CODEX_BIN="/opt/bin/codex --profile art"`.

The Codex service uses your own ChatGPT login and is bound by its terms; OpenAI
recommends an API key for scripted use. Gemini's API terms require users to be 18
or older and are meant for professional or business use. See `THIRD_PARTY.md` for
details.

## Five minutes with PixelMagic

Nothing here needs an internet connection or an API key. PixelMagic has no
default image service, so anything that draws needs one chosen (on the Services
page, or with `--service`); `import`, `keep-scene` and the rest do not draw. Three stand-ins make
that possible: `--service fake` is an image service that draws on your own
machine (a small chunky board when it redraws objects) and `--segmenter stub`
finds objects without the real finder. The scene is copied through by a
stand-in unless you turn PixelRefiner on with `--refiner cli`. Copy each command in turn. Where a command prints an id, use that
id in the next command wherever you see `<...>`.

**1. Make a picture to work on.** Normally you bring your own, or have an image
service draw one (see "Drawing for real"). The fake service's own picture is a
single flat shape with nothing to pick out, so for this tour make a small one
with four separate objects and a wooden floor, `den.png`:

```bash
mkdir -p ~/art
uv run python - <<'EOF'
from pathlib import Path
from PIL import Image

picture = Image.new("RGBA", (480, 320), (0, 0, 0, 0))
for i, box in enumerate([(20, 20, 80, 80), (120, 30, 200, 90), (260, 20, 310, 100), (370, 40, 450, 110)]):
    picture.paste((200 - i * 40, 80 + i * 40, 120 + i * 30, 255), box)
browns = [(92, 58, 34), (120, 78, 46), (146, 98, 58), (170, 120, 72), (196, 144, 90)]
for x in range(480):
    for y in range(200, 320):
        picture.putpixel((x, y), (*browns[(x // 9 + y // 6) % 5], 255))
picture.save(Path("~/art/den.png").expanduser())
EOF
```

**2. Make a project.** A project is one game. It is just a folder.

```bash
uv run pixelmagic --root ~/art/demo project new DemoGame
```

**3. Describe the look once.** This is the art direction — every picture you
ask for afterwards is drawn in this style.

```bash
uv run pixelmagic --root ~/art/demo direction new \
  --name cosy-den \
  --style "Warm SNES interiors, soft lamp light, chunky outlines." \
  --dont "no gradients"
```

Leave `--name` out and the image service invents a name for you (that draws, so it needs a chosen service). Repeat
`--dont` for each thing you never want to see.

**4. Bring in the picture and keep it as the scene.** `import` makes the
picture an approved one without asking an image service for anything, and
prints its id (a stamp like `20261008-164539`). The
scene is the whole picture, kept as one item of its own; it is the one thing
PixelRefiner would handle, if you turned it on. Keeping it reads how big its pixels are.

```bash
uv run pixelmagic --root ~/art/demo --direction cosy-den import ~/art/den.png
uv run pixelmagic --root ~/art/demo --direction cosy-den \
  keep-scene <picture-id>
```

`keep-scene` ends with `kept the whole picture as asset den`. With the stand-in refiner
(the default) it then adds a note that the scene's pixel size was not measured and 1.0 was
used. Before a real refine, keep the scene with `--refiner cli`, or set the pitch with
`set-scene-pitch`.

**5. Pick the objects and have each one redrawn.** PixelMagic finds the
objects on the picture, you choose the ones you want by number, and each chosen
object is redrawn by the image service (up to five at a time) into a separate
piece. `--segmenter stub` treats each separate blob of opaque pixels as one
object.

```bash
uv run pixelmagic --root ~/art/demo --direction cosy-den \
  suggest <picture-id> --segmenter stub
uv run pixelmagic --root ~/art/demo --direction cosy-den pick <decomposition-id> 1 2 3
uv run pixelmagic --root ~/art/demo --service fake --direction cosy-den \
  redraw <decomposition-id>
```

`suggest` prints the decomposition's id and where the numbered contact sheets
are (open them to choose; here it finds 5 blobs and shows 4, because the floor
covers more than a fifth of the picture and counts as background, not an
object). `redraw` keeps the pieces as assets itself. Each redrawn object is
rebuilt on its own pixel grid: PixelMagic finds the grid the picture was
painted on and saves the piece at its real size, one pixel per art pixel. The
image service names each piece (`small-lamp`, `round-rug`, ...). If it cannot
find a grid on an object, that object stays picked and `redraw` says so in a
warning; run `redraw` again and it tries just those. After the last piece,
`redraw` may also warn that a piece was drawn about 2x too big or too small
compared with the other pieces; the piece is kept either way, and redrawing it
by name replaces it.

**6. Draw the floor.** A floor or wall is not redrawn by an image service. You
name it, point at a small plain patch of it, and sketch where it goes;
PixelMagic reads the colours from the patch and draws 32 tiles from them.

```bash
uv run pixelmagic --root ~/art/demo --direction cosy-den \
  decompose <picture-id> \
  --surface "wood floor:wood; box 100,220 150,270; shape 0,200 480,200 480,320 0,320"
```

It prints the 32 tile names (`wood-floor-v1-nw`, `wood-floor-v1-ne`, ... up to
`wood-floor-v8-se`). The numbers are pixels of your picture: the `box` is two
opposite corners of the patch, each `shape` is the corners of a rough outline.

**7. Turn the scene into pixel art.** Pieces you picked and redrew were
already rebuilt at their real size in step 5, and the floor tiles were drawn at
their final size in step 6, so PixelRefiner never touches them. Only the scene
would go to it, and only with `--refiner cli`; otherwise a stand-in copies it through.

```bash
uv run pixelmagic --root ~/art/demo --direction cosy-den refine
```

On this tour it prints `refined 1 asset(s) with stub` and says the three
rebuilt props and the 32 recipe tiles were not sent to the refiner. Nothing is
wrong: they are done. (With the stand-in the scene is only copied through, not made
into pixel art; the export says so.)

**8. Take the bundle away.**

```bash
uv run pixelmagic --root ~/art/demo --direction cosy-den export
```

You get one folder under `exports/` holding a PNG per item at its real size
plus copies at two and four times, a `manifest.json` listing everything and
where it came from, and — if Aseprite is installed — one `.aseprite` per item
you can open and touch up by hand: the scene, each prop, one for the whole
wood floor (a floor made of recipe tiles gets one `.aseprite` for all 32
tiles, not one each), and `den-room.aseprite`, the scene rebuilt in pixel art
with the floor and props where they sit in the picture (see "A room file").
In this tour's `manifest.json` each prop says
`"refiner_tool": "pixelmagic-grid-rebuild"` and `"route": "rebuild"`, each floor
tile says `"pixelmagic-recipe"` and `"recipe"`, and the scene, which here went
through the stand-in, says `"stub"` and `"preserve"`: that is how a rebuilt or
drawn piece is told from a refined one, and from concept art copied through.
The export also prints a note that the scene was copied through (`den: was
refined with warnings: FALLBACK_TO_PRESERVE`), which is what the stand-in does.

### Picking the objects yourself

Step 5 in full. You look at the objects found on a picture, pick the ones you
want by number, and have each redrawn. The commands, in order:

```bash
uv run pixelmagic --root ~/art/demo --direction cosy-den suggest <picture-id> --segmenter stub
# open the contact sheets it lists, note the numbers you want
uv run pixelmagic --root ~/art/demo --direction cosy-den pick <decomposition-id> 1 2 3
uv run pixelmagic --root ~/art/demo --service fake --direction cosy-den \
  redraw <decomposition-id> --no-keep
uv run pixelmagic --root ~/art/demo --direction cosy-den keep <decomposition-id>
```

(Without `--no-keep`, `redraw` keeps the pieces itself and `keep` is not
needed. With it you can look at the pieces first, rename or drop one, and then
`keep` them.)

A redrawn piece keeps its own size and its own pitch (how many pixels of the
picture make one art pixel, worked out from the object's box on the original
picture and the piece's real size). `refine --asset NAME` with a setting is
refused for it (without one it is simply kept), and a plain `refine` skips it
and says how many it skipped. To change a piece, redraw it:
`redraw <decomposition-id> --only <pick-number-or-name>`.

The real object finder (Segment Anything, Apple silicon only) is the default.
Set it up once:

```bash
cd segmenter
uv sync
uv run pixelmagic-segmenter fetch-model   # 184 MB, into segmenter/weights/
```

`--segmenter stub` runs offline with no setup and finds each separate blob of
opaque pixels instead. The finder's command and its JSON are described at the
top of `src/pixelmagic/segmenter/base.py`; `segmenter/README.md` has the setup.
If the contact sheets miss an object, or cut one wrong,
point at it yourself; each of these asks the finder for the outline under your
clicks (X,Y are pixels on the picture, and a box's right and bottom are
exclusive):

```bash
pick <decomposition-id> --point 200,570 --point 240,570   # spots on the object
pick <decomposition-id> --point 200,570 --not 260,600     # and a spot NOT on it
pick <decomposition-id> --box 1210,440,1420,570           # the object lies in this box
pick <decomposition-id> --fix 2 --point 160,560           # redo pick 2's outline
```

Run each after `uv run pixelmagic --root ~/art/demo --direction cosy-den`.
`picks <decomposition-id>` lists what you have and draws every outline,
numbered, on the picture; `unpick <decomposition-id> N` takes one back (its
number is never used again). `--fix` keeps the pick's name and does not touch a
piece already redrawn from it; redraw that piece by name to replace it.

On the web page the same choosing is done on the picture itself (see "The web
page" below), and an agent has the same steps as tools. The redraw works in
groups of five objects per call to the image service; a group that fails leaves
its objects picked, and so does an object whose pixel grid is not found, so
running `redraw` again redraws just those. `manifest.json` records which finder
cut each piece out (`segmenter`).

### Floors and walls from your own picture

A recipe surface reads its colours from the picture, so the picture has to
have some: at least 5 different colours inside a wood or stone floor, 4 inside
a plaster wall. The fake service's flat stand-in picture has one colour, so
`decompose` refuses it ("has 1 different colours and the wood recipe needs at
least 5"); that is why the tour makes its own picture. No object finder is
involved, so nothing here needs it. Add one `--surface` per floor or wall, then
refine and export as in steps 7 and 8:

```bash
uv run pixelmagic --root ~/art/demo --direction cosy-den \
  decompose <picture-id> \
  --surface "wood floor:wood; box 100,220 150,270; shape 0,200 480,200 480,320 0,320" \
  --surface "wall:plaster; box 60,20 100,60; shape 10,10 240,10 240,170 10,170"
```

Use the picture's id from `import`, and a box and shapes that land on the floor
and the wall in your picture. `decompose` prints the 32 tile names per surface.
`refine` says the recipe tiles were not sent to the refiner, which is right;
`export` skips a piece that has not been through `refine`, so run it. The bundle
holds each tile as a PNG at its real size, at two and four times, and, with
Aseprite installed, `wood-floor.aseprite` and `wall.aseprite`.

### Things worth knowing before you start

**Floors and walls become tiles only when you name them.** Type a surface
as one string (`--surface "plaster wall:plaster; box 180,240 212,285; shape
140,400 200,230 330,110 520,70; shape 1100,440 1100,330 1170,285"`, repeat
`--surface` for more surfaces). The first part is the name and the `kind`:
`plaster`, `stone` or `wood`. The `box` is two opposite corners (either order)
of a small patch of plain surface, with nothing else on it; the colours are
read from the box and nowhere else. Each `shape` is the corners of a rough
outline drawn over the surface, at least 3 corners, and you can draw as many
shapes as you need (a floor split by furniture, say). Together the shapes say
where the surface goes in the room. All corners are in the picture's own
pixels. The same string works on the MCP `cut_into_pieces` tool. The web page's
surfaces screen draws the box and shapes on the picture with the mouse and
sends the same string. PixelMagic takes the colours the picture
itself uses inside the box and draws 32 tiles of 16 × 16
pixels from them: 8 blocks of 2 × 2 tiles, named `wood-floor-v1-nw`,
`wood-floor-v1-ne`, `-sw`, `-se`, then `wood-floor-v2-nw` and so on up to
`v8`. The 8 blocks are interchangeable: place them in any order and they join.
Wood is long planks, stone is big square slabs with thin grout, and plaster is
large soft patches on a mostly one-shade wall. Nothing is drawn by the image
service for them, and they are not refined (`refine` keeps them as they
are). The shapes, filled in and joined, are saved next to the cut-up
(`decompositions/<id>/surfaces/<name>/outline.png`) so you can look at what
was covered; the room file uses it to place the surface.

A surface is refused, with a sentence saying why, when the box is too small
(under 1,024 pixels, a 32 × 32 patch) or has too few different colours (5 for
wood and stone, 4 for plaster), when a colour the recipe needs covers too
little of it (something else is probably inside the box), when the box is not
wholly inside the picture, when the shapes cover no pixel of the picture,
when the direction's cell size is not 16, or when the name is already used by
another piece (or by the same surface on another picture). Put the box on a plain patch of the floor or wall. A plain
name with no kind, a missing box, no shape, or a shape with fewer than 3
corners is refused too, and so are the two older forms, `name:bordered` and
`name:kind@x,y`, each with a sentence giving the way forward. With no
`--surface`, `decompose` makes nothing and points you to `suggest`. Where a
floor and a wall overlap, both keep the cells and the room file's layer order
decides what shows (the surface with the most cells is the bottom layer);
erase the unwanted cells in Aseprite.

**Only the scene can go to PixelRefiner, and only if you turn it on.** By
default a stand-in copies the scene through unchanged, so the PNG you get for
it is the concept art, not pixel art. With `--refiner cli` PixelMagic runs
PixelRefiner (a separate project; see "Install" for how to get it) on the scene, with a palette of 24
colours fitted to it alone and the house-style settings. Props you picked and
redrew are rebuilt at their real size and floor tiles are drawn at their final
size, so neither goes there either way. The last paragraphs of the command
reference below say how to tell, and how to switch.

**Aseprite is optional.** If you have it, PixelMagic finds it and writes one
`.aseprite` file per item, and one per recipe surface: a tileset holding its
32 tiles, plus a sample floor of 10 × 6 tiles laid from them so you can see how
they sit together. If you do not, everything else still works and the
export says once that it skipped those files. If it is installed somewhere
unusual, set `PIXELMAGIC_ASEPRITE` to its full path.

**Touching up a floor or wall.** The first export that writes a surface's
`.aseprite` also keeps a copy in the project, the master:
`<root>/directions/<direction>/assets/<surface>.aseprite`. Open the master in
Aseprite, edit the tiles in place, and save it. Every export after that takes
the surface's tile pictures (the PNGs at every size, and the manifest's colour
list) from the master, and the bundle's `<surface>.aseprite` is a copy of it.
Keep the tile count and size: 32 tiles of 16 × 16, on the tilemap layer. If the
master cannot be read or does not match, export falls back to the drawn tiles
and says so, and leaves your file alone. Type the surface again and PixelMagic
draws fresh tiles but keeps the master, and tells you so; delete the master
file to start over.

**A room file.** Every export of a kept scene, with Aseprite installed, also writes `<scene-name>-room.aseprite`: one layer for each floor or wall, painted
over the part of the picture it covers, and one layer for each prop, standing
where it stands in the picture, and one `structure` layer between them: the
scene's own picture shrunk to the room, keeping only what the floors, walls and props
do not already cover (the wooden frame, beams, door arch, stove pipe). It is a little
soft, since it is shrunk and not redrawn as pixel art. Everything is drawn at the pixel
scale of your pieces (the typical scale of the props against their size in the picture).
It is a starting point for you to move things around in Aseprite, nothing more:
changes you make in it never go back into the project, and the separate files
are not touched. Floors and walls use the tiles in the bundle, so touch-ups in a
surface's master file show up in it. A floor runs under a bed or rug
that sits inside its shapes, even if the shapes left that spot out. A piece that cannot be placed
(one kept before its place was recorded, a missing record) is
left out, and the export says which and why; nothing is guessed. A picture with
no placeable props and no floors or walls gets no room file and no notes about
one. The native size is always exported,
whatever `--scale` asks for. To keep a picture you
imported (or any picture you already approved) as the scene afterwards, run
`pixelmagic keep-scene <generation-id>`, call the `keep_scene` MCP tool, or press
"Keep as scene" on the picture's page; it is refused if the picture is not approved
or is already kept as the scene. Type the scene's name with `--name`, the `name`
argument or the name box (otherwise one is made from the picture's subject); a name
that is not lowercase words joined by hyphens, or is already taken, is refused. Rename
a kept scene later with `pixelmagic rename-scene <scene> <new-name>`, the
`rename_scene` tool or "Rename" on the picture's page; an export you already made keeps
the old name and the next one uses the new.

**Style references.** To make an imported picture (or any picture you already
approved) a style reference of its direction, so later pictures are drawn to
match it, run `pixelmagic import <file>`, then
`pixelmagic keep-reference <generation-id>` (or the `keep_reference` MCP tool),
then `pixelmagic generate` as usual: the direction's references go into every
request. It is refused if the picture is not approved; keeping one that already
is a reference changes nothing and says so. On the web page, an approved
picture has a "Keep as reference" button too, or you can tick the box when
you approve it.

**Projects made before this version.** A project folder from an earlier version
still opens. Props that came from the old one-sheet method, and tiles from the
old `:bordered` or plain-patch surfaces, can no longer be made. Ones that were
already refined still export as they are. Ones that never were refined are
refused with a sentence saying to pick and redraw the object again (for a
prop), or to type the surface again as a recipe surface (for a tile).

## Sitting on the grid

A bundle is only useful in a game if the pieces agree with each other and
with the game's own tile grid. Three numbers make that happen — cell size,
pitch and anchor — and every one of them ends up in `manifest.json`, so a
game reads the manifest and never the screen.

- **Cell size** is how many art pixels one tile is on a side — 16 unless you
  say otherwise. It belongs to the art direction: `direction new --cell-size
  32` at the terminal, or `new_direction`'s own `cell_size` for an agent.
  A recipe surface needs 16, and a prop's footprint is counted in cells of
  this size.
- **Pitch** is how many source pixels make one art pixel. The scene has one,
  measured from the whole picture when you keep it, so the scene is refined at
  the right size. If the measurement looks wrong, overrule it with one number:
  `set-scene-pitch <scene-name> 4.2` at the terminal, the `set_scene_pitch`
  tool for an agent, or the refine page. The scene's target size follows;
  refine it again. A prop you picked and redrew has a pitch of its own (its box
  on the picture over its rebuilt size), set by the redraw, so there is nothing
  to set for it.
- **Anchor** is the point a prop stands on, in art pixels of its picture —
  bottom centre unless you say otherwise, and PixelMagic never guesses it from
  what the prop looks like. `anchor lamp 8 31` at the terminal, or the
  `set_anchor` tool, or a click on the prop on the refine page.

In `manifest.json` a prop's entry carries `pitch`, `target` (its size in art
pixels), `footprint` (in cells) and `anchor`; a recipe tile's carries `target`,
a `variant` (1 to 8) and a `recipe` record (the kind, the shapes and box you
drew, the colours it read and how much of the box each covered, and which tile
of its block it is); a kept scene's carries `pitch` and `target` only.
`cell_size` and `pitches` (the scene's) sit at the top of the file, next to
`project`, `direction`, `exported` and `entries`. Each entry also says how it
was made (`refiner_tool`, `route`) and which finder cut a prop out
(`segmenter`).

## The web page

Everything the command line does, with the pictures on screen.

```bash
uv run pixelmagic --root ~/art/my-game --service fake web
```

The page finds objects with the real object finder by default. To pick objects
with no setup, add `--segmenter stub` after `web`; to refine the scene with
PixelRefiner, add `--refiner cli`. Open the address it prints. The page runs on your Mac and
nowhere else — it will refuse to listen to the rest of the network, because it
shows your whole project folder and has no password on it (`--host` and `--port`
change the address, but the page still refuses a host that is not your own
machine).

What the screens do, in the order you use them:

1. **The front page** lists your art directions and lets you make one. An art
   direction is a look: a paragraph describing it, and a list of things to
   avoid. Leave the name empty and the image service invents one. Set the
   cell size there too — how many art pixels one tile is on a side.
2. **A direction's page** shows its pictures, and lets you
   bring in a picture of your own ("Bring in a picture") or rename the
   direction.
3. **Start** asks for one picture, or lets you bring in one of your own, and
   then stops. The page then shows the picture with the steps of the scene
   along the side. If it is right, click Approve — and tick the box to keep it
   as a style reference, so the next picture in this direction is drawn to
   match. If it is wrong, say what should change under "Not quite?" and click Redraw; PixelMagic
   writes down what you said and immediately draws another with your words
   folded into the request. Every attempt is kept, so you can always go back
   and look at what you turned down.
4. **An approved picture's page** first offers **Keep as scene**, which keeps
   the whole picture as the scene (it tells you the name it will be filed
   under). Once kept, **Continue to Pick props** finds the objects on it and
   opens the picking screen, with a small tool tray floating over the picture:
   Pick, Add a spot, Remove a spot and Lasso, plus Undo and Bigger/Smaller. With a
   mouse, move the pointer over the picture and the object under it is
   outlined; a click picks it; scroll to outline a bigger object around the
   same spot (up) or a smaller one (down). On a touch screen, tap an object to
   preview it and tap it again to pick it; Bigger and Smaller replace scrolling,
   and Add a spot and Remove a spot are modes you switch on in the tray. For an object
   made of several parts (a bed with its pillow and blanket), turn on Lasso and
   drag a line round it. On the selected pick, Add a spot adds a spot (or, on a
   lasso pick, the piece under the click) and Remove a spot marks a spot that is
   not part of it (or removes that piece). With a mouse you can hold a key
   instead of switching tools: hold Cmd (Ctrl on Windows and Linux) for Add a
   spot, Option (Alt) for Remove a spot, and Shift for Lasso. The tool changes
   only while the key is down, and the tray shows each key. Undo
   takes back your latest change to a pick's outline (up to 20; a redraw
   clears them, and it never undoes a name). The picks are listed as rows, each
   with its own warnings and problems underneath. Small numbers mark the
   objects not yet picked, unless there are more than 40 of them. Then redraw the
   picks: you land back on the same screen, which shows what went wrong for any
   pick and offers **Continue to Surfaces** or **Redraw again**. Continue keeps the
   redrawn props as assets and opens the surfaces screen.
5. **The surfaces screen** is where floors and walls are drawn, one picture at
   a time (a "Draw floors and walls" link on the approved picture's page opens
   it too). Draw one or more shapes over the surface and one small box on a
   plain patch of it, name the surface and choose its kind (plaster, stone or
   wood). The colours read from the box are shown before anything is drawn.
   Drawing keeps the surface's tiles as assets in the same step. A new surface
   under a name already drawn on that picture is refused ("That name is taken.
   Pick another."); Edit redraws the surface, and if you gave it a new name the
   old one is taken off. Remove takes its tiles
   and assets off the picture but keeps its master file, with your touch-ups.
   Every refusal is the same sentence the command line gives.
6. **Turn the pieces into pixel art** is PixelRefiner's own settings, read off
   PixelRefiner itself rather than a list we copied. Pick a preset, turn the
   knobs, and refine the scene, or one asset with settings of its own. You see
   each asset before and after. Rebuilt props and recipe tiles are listed as
   such: settings do not apply to them. If PixelRefiner says it is not sure how
   big the pixels are, the alternatives it found are listed with a button each.
   Once `refine` has run, click a prop where it stands on the ground to set its anchor. The
   kept scene has no anchor, but you can correct its pitch here — and if
   PixelMagic could not refine it because of that pitch, the screen says so and
   lets you fix it without leaving the page.
7. **Export a bundle** shows the kept scene's room as three pictures (props,
   floor tiles, structure) you can hide and show one by one, and lists what
   the bundle will hold, a floor or wall as one tile set. The preview needs no
   Aseprite and writes nothing into `exports/`; before the first export it shows the
   drawn tiles, since touch-ups in a master file are read through Aseprite.
   One button writes the folder a game reads: the PNGs, a `manifest.json` and
   the Aseprite files. 1x is always written; a switch adds 2x. Afterwards the
   page shows the folder, the files in it and which `.aseprite` to open.

Two things worth knowing. By default the page uses the stand-in, which copies
the scene through untouched and says so in a yellow box; start it with
`--refiner cli` to use the real PixelRefiner, the same one the command line
reaches for. And the settings screen reads what the real PixelRefiner accepts,
so without `--refiner cli` it says the real one is not turned on, and if it is
turned on but not installed it tells you that, instead of offering you settings
that do not exist.

The page is otherwise plain HTML forms; the only script on it is a few small
handlers — one turns a click on a prop (once `refine` has run) into its anchor, and the
picking screen's script does the hovering, scrolling, tapping and lasso drawing and
posts the result, and the surfaces screen's script (`static/surfaces.js`) turns your clicks and drags into
shapes and a colour box in the picture's own pixels, asks the server for the colours a box gives, and posts
the typed surface. The anchor form still works typed by hand with the script
turned off. On the picking screen, only renaming, removing and redrawing picks
work without the script: making a pick (hover, click, tap, scroll, lasso, the hold-down keys)
and Undo need it. To pick without it, use the command line or an agent.

## Driving it from an AI agent

PixelMagic also speaks MCP, so an agent can run the whole pipeline and judge
the pictures itself. Point your MCP client at this command:

```json
{
  "command": "uv",
  "args": ["run", "--directory", "/path/to/PixelMagic",
           "pixelmagic", "--root", "/path/to/your/art", "--service", "fake", "mcp"]
}
```

The server uses the stand-in refiner by default, which copies the scene
through untouched rather than turning it into pixel art — add `"--refiner",
"cli"` to `args` above to use the real PixelRefiner. In the same way, `"--segmenter", "stub"`
swaps the real object finder for the offline one.

The agent gets one tool per step:

| Step | Tools |
|---|---|
| Look at the project | `project_overview`, `list_assets`, `show_asset` |
| Start an art direction | `new_direction` |
| Draw and judge a picture | `draw`, `approve`, `reject` |
| Bring in or keep a picture | `bring_in_picture`, `keep_scene`, `keep_reference` |
| Find, pick and redraw objects | `suggest_objects`, `show_suggestions`, `pick_objects`, `fix_pick`, `unpick`, `show_picks`, `redraw_objects` |
| Draw floors and walls | `cut_into_pieces` (with `surfaces`) |
| Fix the pieces | `rename_piece`, `drop_piece`, `keep_pieces` |
| Refine, adjust, export | `refine`, `set_scene_pitch`, `set_anchor`, `export_bundle` |

Anything that produces a picture hands the picture back, so the agent can
actually look at what it asked for rather than being told where the file is.

Drawing works the same way it does on the web page: one attempt at a time,
waiting for a verdict. The agent looks, then approves or rejects with a reason,
and rejecting draws the next attempt with that reason folded in. Everything is
written to the same folders the command line and the web page use, so you can
start something in one and finish it in another.

If you point it at an image service that has not been built yet, it says so and
names the ticket rather than failing in a way the agent cannot act on.

## Drawing for real

The walkthrough above uses `--service fake`, which draws a flat shape for a
scene and a small chunky stand-in board for a group of objects, and never
touches the network. To draw actual art, swap the service. `--service` is a
global option, so it goes before the command, not after it. Draw the scene,
look at it, then follow steps 4 to 8 with `--service codex` on `redraw`, skipping `import` and using the id `generate` prints in `keep-scene`. The box and shapes in the floor step must fit your own picture:

```bash
uv run pixelmagic --root . --service codex generate "a cosy den" --wait
```

`generate` prints the path of the picture and waits. Open the file. If you
like it, answer `y`. If you do not, answer `n` and say what is wrong — it draws
again with your note attached, up to three tries (`--rounds`). After three
refusals it stops and tells you which attempt was the last one, so you can look
at it and try different words. The last line it prints is the picture's id.
With `--auto-approve` instead of `--wait` it accepts whatever comes back; with
neither, it asks when you are at a terminal and approves automatically when you
are not. Add `--as-asset` to keep the whole picture as the scene straight away.

Three things are worth knowing about Codex specifically:

- It is paid for by your ChatGPT subscription, not per picture.
- It cannot be told what size to draw. Ask for 1024 and you may get 1254; the
  size you asked for is recorded next to the size you got.
- It really does draw on a transparent background, which is what the rest of
  the tool wants.

Each picture takes one to two minutes. A redraw is one call per group of up to
five objects, and the image service is also asked to name every object you
picked (one short question per object, a few seconds each — and it prints
nothing while it does, so don't take the silence for a hang).

Gemini needs its package installed once — `uv sync --extra gemini` — and a key
(`GEMINI_API_KEY`, or saved on the Services page). Three things to expect from it:

- It cannot draw on a transparent background at all. PixelMagic knows that, so
  it asks Gemini for a flat magenta background instead and cuts the magenta out
  afterwards. You do not have to do anything.
- Every Gemini picture carries an invisible Google watermark. PixelRefiner
  takes it off when it turns the scene into pixel art (`--refiner cli`); the
  default stand-in copies the scene through with the watermark still in it.
- At most 14 reference pictures travel with a request; if a direction has more
  than that, the older ones are left out and the result says so.

OpenAI needs its package installed once — `uv sync --extra openai` — and a key
(`OPENAI_API_KEY`, or saved on the Services page). It is the only one of the three billed per picture. It
draws on a transparent background, and if a picture comes back solid anyway the
note is recorded next to it. It cannot look at a picture and tell you what it
is, so pieces redrawn through OpenAI are named `item`, `item-2`, `item-3` and so
on until you rename them (name a pick yourself with `pick ... --name`). At most
16 reference pictures travel with a request; if a direction has more than that,
the older ones are left out and the result says so.

## Command reference

These three go before the command name, and every command takes them:

| Option | What it does | Default |
|---|---|---|
| `--root` | The project folder to work in. | the folder you are in |
| `--service` | Which image service draws. `codex`, `gemini` or `openai`, or `fake` (the offline stand-in, reached only this way; it cannot be saved as a game's choice). Wins over the game's saved choice (the Services page). There is no default: a command that draws with none chosen stops and says so. | the game's saved choice; if there is none, nothing draws |
| `--direction` | Which art direction to use. | the project's first one |

| Command | What it does |
|---|---|
| `project new NAME` | Start a new project (one game). Refuses if there is already a project in that folder. |
| `project list` | List the projects found in `--root` or one level below it. |
| `direction new --style TEXT [--name NAME] [--dont TEXT] [--cell-size N]` | Create an art direction. Without `--name` the image service invents one, so that needs a service chosen. `--dont` can be repeated. `--cell-size` is how many art pixels one tile is on a side, 4 to 128, 16 by default. Refuses a name the project already has. |
| `direction list` | List the art directions in this project. |
| `direction rename OLD NEW` | Rename an art direction, keeping everything inside it. |
| `generate SUBJECT` | Draw one picture and approve it. `--kind scene` or `--kind prop`, `--background transparent\|magenta\|lime`, `--size N` (pixels, square, 1 to 4096), `--rounds N` (how many refusals before it gives up), `--wait` or `--auto-approve`, `--no-keep` to skip saving it as a style reference, `--as-asset` to also keep the whole picture as the scene, `--refiner stub\|cli` (it only reads the kept picture's pixel size). |
| `import PICTURE` | Bring in a PNG, JPEG or WebP of your own as an approved scene, ready to cut up. |
| `keep-scene GENERATION_ID` | Keep an already-approved picture (imported or drawn) as the scene, so it can be refined and exported. `--refiner stub\|cli` (it reads the picture's pixel size). `--name` types the scene's name. Refused if the picture is not approved or already is the scene, or the name is unsafe or taken. |
| `rename-scene SCENE NEW_NAME` | Rename a kept scene. A bundle already exported keeps the old name. |
| `keep-reference GENERATION_ID` | Keep an already-approved picture (imported or drawn) as a style reference of the direction. |
| `suggest PICTURE_ID` | Find the objects on an approved picture and write numbered contact sheets. `--segmenter stub\|cli` (default `cli`, the real finder; it needs the one-time setup above). Prints the decomposition's id. |
| `pick DECOMPOSITION_ID [NUMBER ...]` | Pick objects to be redrawn: by suggestion number, or `--point X,Y` (repeat) and `--not X,Y` for a spot not on the object, or `--box X0,Y0,X1,Y1`. `--fix N` redoes pick N's outline from those. `--name TEXT` names one pick. `--segmenter` as above. |
| `pixel-cut GENERATION_ID` | For a picture that is already pixel art: start cutting props straight from its own pixels, with no image service and no PixelRefiner. `--size N` is picture pixels per art pixel (1 means already one pixel per art pixel); left out, the size is found. Then `pick` objects by `--box` or `--point` (not by number), and `cut DECOMPOSITION_ID` turns the picks into props (`--only N` to cut some, `--no-keep` to look first). |
| `picks DECOMPOSITION_ID` | List the picks and draw each one's outline, numbered, on the picture. |
| `unpick DECOMPOSITION_ID N` | Take pick N back; its number is never used again. |
| `redraw DECOMPOSITION_ID` | Redraw the picked objects, five at a time, into separate pieces, each rebuilt on its own pixel grid at real size. `--only PICK` (number or name, repeatable) redraws those again; with none, every pick with no piece yet. `--no-keep` leaves the pieces as pieces so you can fix them first; then `keep`. An object whose grid is not found stays picked with a warning; run `redraw` again to retry it. |
| `decompose PICTURE_ID --surface "..."` | Draw floors and walls as tiles. `--surface "wood floor:wood; box 180,240 212,285; shape 140,400 200,230 330,110 520,70"` (repeatable; kind `plaster`, `stone` or `wood`, one `box` of two corners on a plain patch, and one or more `shape`s of at least 3 corners, all in the picture's pixels) makes 32 tiles of 16 px, 8 blocks of 2 × 2, from the picture's own colours inside the box, placed where the shapes cover. A plain name with no kind, `name:bordered` and the old `name:kind@x,y` are refused with a sentence; with no `--surface` it is refused and points you to `suggest`. `--no-keep` leaves the pieces as pieces so you can look at them first; then `keep`. |
| `keep DECOMPOSITION_ID` | Turn a cut-up run's pieces into assets, once they are right. The other half of `decompose --no-keep` and `redraw --no-keep`. Refused for a run that still holds pieces from the old roads. |
| `set-scene-pitch NAME VALUE` | Overrule the measured pitch of a kept scene (0.5 to 64); its target size follows. |
| `anchor NAME X Y` | Say where a prop stands, in art pixels of its picture. Only works after the prop is kept. |
| `refine` | Turn the scene into pixel art. Rebuilt props and recipe tiles are skipped and counted in the report; a setting aimed at one is refused. `--refiner stub\|cli` (default `stub`), `--preset NAME`, `--quick JSON`, `--advanced JSON`, `--scale N`. Add `--asset NAME` to apply those settings to one item only, leaving the art direction's own settings alone. The summary line names any refused assets (`refined 0 asset(s) with stub; 2 refused: cosy-den, objects`), and `refine` exits 1 when anything was refused. |
| `export` | Write the bundle. `--scale N`, repeatable; the default is real size, two times and four times. |
| `web` | Open the web page for this project folder. `--host`, `--port`, `--refiner stub\|cli`, `--segmenter stub\|cli`. |
| `mcp` | Serve PixelMagic to an AI agent over MCP (stdio). `--refiner stub\|cli`, `--segmenter stub\|cli`. |

Renaming keeps everything: `pixelmagic direction rename cosy-den warm-den`
moves the whole folder, its pictures, its pieces and its finished assets, and
repoints the project at it if that was the direction you were working in.
Export bundles you have already written keep the old name — a bundle is a
record of a day's work, not a live link.

A direction's name becomes a folder name, so `direction new --name` squeezes
whatever you type down to lowercase letters, numbers and hyphens: `Café Noir`
becomes `cafe-noir`. Accented letters survive that — only the accent is lost.
A name written entirely in another script has nothing left afterwards, so
PixelMagic refuses it and says so rather than making up a name for you.
`direction rename` does not squeeze its argument — it expects the new name to
already be in that shape (lowercase letters, numbers, hyphens) and refuses
anything else.

`generate` takes `--wait` (stop and ask you) or `--auto-approve` (accept
whatever comes back). With neither, it asks when you are at a terminal and
approves automatically when you are not — so a script never hangs waiting for
an answer, and an interactive run never quietly accepts a bad picture.

Anything that goes wrong that PixelMagic already understands — a picture nobody
approved, a name that does not exist, a pixel-art tool that is not installed —
comes out as one sentence and a failure, not a page of Python.

**Settings for the pixel-art step.** `--preset`, `--quick`, `--advanced` and
`--scale` on `refine` are passed straight through to PixelRefiner, the tool
that does the conversion; run `pixel-refiner options` to see what it accepts.
Used on their own they become the art direction's standing settings, and are
checked before anything is written: a value PixelRefiner refuses is reported
and nothing is saved, so a bad setting can never get stuck in the art direction
and break every later run.

By default the scene is copied through by a stand-in, so it is your concept
art unchanged, not pixel art, and `manifest.json` says so on its entry
(`"refiner_tool": "stub"`, `"route": "preserve"`). Do not put one into a game.
To turn the scene into real pixel art, add `--refiner cli`: PixelMagic then runs
PixelRefiner (see "Install" for how to get and build it), found as
`pixel-refiner` on your PATH or as the command in `PIXELMAGIC_REFINER_BIN` (see
**Environment variables** above). If it cannot be run, PixelMagic tells you in
one line and names the two ways out: leave out `--refiner cli`, or set
`PIXELMAGIC_REFINER_BIN`.
A prop you picked and redrew is not copied through either way: it is rebuilt at
its real size and says `"refiner_tool": "pixelmagic-grid-rebuild"`, `"route":
"rebuild"`; a floor tile says `"pixelmagic-recipe"` and `"recipe"`.

Each item in `manifest.json` also carries a `warnings` list, usually empty.
`LOW_GRID_CONFIDENCE` means the refiner had to guess how big a pixel is in the
original, and that piece is worth a look before it goes into a game.

## Licence

PixelMagic is MIT-licensed (see `LICENSE`). The outside tools and models it uses
keep their own licences; `THIRD_PARTY.md` lists them.
