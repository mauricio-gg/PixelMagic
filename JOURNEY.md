# JOURNEY.md: PixelMagic interface

Scope: the four mocked screens (project home, pick-props workspace, surface-drawing workspace, export). Start, Approve and Keep scene are built (ticket #191: `start.html`, `picture.html`, `web/generating.py`) and Refine is built (ticket #194: `refine.html`); they follow the same look but have no mocked spec here. The timeline shows every step and a done one can be reopened.

## Job

When I am an indie game developer who is not a pixel artist, I want to turn a text prompt (or a picture) into game-ready pixel art files, so my engine can use them without me drawing anything.

## Journey (the animated timeline)

The Refine and Export pages (and the timeline on them) follow a scene the person chooses. With no choice they follow the newest kept scene that has pieces (props or surfaces), else the newest kept scene. With two or more kept scenes a quiet chooser sits at the top of the timeline column: a collapsed "Scene: name" line that opens to a list of the kept scenes with a small thumbnail each; the one followed carries a tick and the word "Following". With one kept scene there is no chooser.

Seven steps, always visible inside a scene: a vertical timeline on the left on desktop, a horizontal strip across the top on tablet.
1. Start (prompt or import) 2. Approve 3. Keep scene 4. Pick props 5. Surfaces 6. Refine 7. Export.

Each step is **done** (green marker with a check; "Reopen" on hover and focus; on touch see below), **current** (dark marker with the one yellow ring), **next** (bright neutral ring) or **waiting** (small dot, not reachable yet). One continuous line runs through the markers and is green where work is finished. A step an agent did carries a small agent icon; pressing it opens a popover with what the agent did and an Undo. Steps map to real routes: generation page, keep-scene, pick, decomposition (surfaces), refine, export.

Behaviour that carries the flow:
- **Completing a step** plays the timeline animation (marker squash, check, the line grows, the next marker takes the ring) and then moves to the next page with a page transition. The picture stays steady where the same scene continues.
- **Reopening a finished step** turns its marker yellow, dims the later steps, makes their line dotted, and shows the note "Editing this step. Later results may update." with a way back. Nothing animates backward.
- **Reopen on touch.** There is no hover on touch, so tapping a done step (`(pointer: coarse)`) does not reopen or navigate at once. It opens a small popover next to the step with the step name and one 44 px button, "Reopen this step", plus Close. Pressing the button does what a click does with a mouse (goes to that step's page, or shows the editing state in place); tapping outside or Close dismisses it. With a mouse or keyboard, click and Enter reopen directly.
- **A long job** (the redraw) never blocks the page: the main button gives way to the job strip (label, honest status, slim bar), the panel stays usable, and when it ends the main button becomes the way forward ("Continue to Surfaces") with "Redraw again" beside it.

## IA

- Home (games and directions) -> Scene workspace (timeline + canvas + side panel) -> steps.
- Top bar: wordmark, game and direction breadcrumb, then room at the right end for the job chip, usage chip and account menu. **Not built in v1.** The account menu, the usage/credits chip and the job chip are not built in v1, which is local-only. The layout only leaves room for them (the right end of the top bar); building them is ticket #195.
- Scene workspace layout (desktop): timeline (left, 224 px, no box), canvas (centre, the neutral grey, with the floating tool tray above it and a one-line hint below), side panel (right, 360 px, one surface). At 1180 px and below the timeline becomes a strip across the top; at 900 px and below the page is one column and the main action is pinned to the bottom of the screen.

## Page specs

### 1. Project home
- Purpose: choose a game and direction, or start a new scene.
- Content: heading "Your games"; on the right a large yellow "New scene" and a soft secondary "Bring in a picture" (no yellow); in the empty state the header's New scene is hidden so the centred one is the only yellow. The top "Bring in a picture" and "New scene" buttons stay whenever games exist. The "Start something new" strip (New game, Bring in a picture) shows only in the empty state, under "Make your first scene"; once the folder has a game it is gone. Behind the lead card a blurred, darkened copy of the latest scene glows (home only). The game with the latest scene leads as a large tile (about 58 percent of the first row; with a single game the lead card is instead centred on its own, up to 880 px wide, with the glow behind it) using its real scene art, its name, and a metadata line (scenes, edited when, direction, cell size, asset count). The other games are open rows with a quiet thumbnail (or, with no picture yet, a neutral texture and the words "No picture yet"), the name, and directions as plain metadata. The whole tile or row is the link; "Open game" with an arrow appears on hover and focus. Hovering a tile brightens it and scales its art slightly. Choosing the lead tile morphs its picture into the scene's canvas.
- States: empty (a soft illustration slot, "Make your first scene" and a yellow button), loading (shimmering skeletons, then the games rise in with a 40 ms stagger), 200-character name (two lines with an ellipsis, full name in a tooltip and on the game page), error (a sentence, never a traceback).
- Local vs hosted: v1 is local-only, so the account menu, usage chip and job chip do not exist yet (ticket #195); the mocks show them only through the mock-only "preview the hosted layout" action. One column at 768 px.

### 2. Pick-props workspace (step 4)
- Purpose: choose which objects in the scene become props.
- Canvas: the scene picture on the neutral grey; thin outlines with a translucent fill on found objects; number tags that sit at 70 percent opacity until hovered; picked objects tinted blue with a fully visible number. Number tags show only when the picture has 40 found objects or fewer (`PAGE_SIZE`); on a busier picture they would cover it, so there are none. Only the hovered object shows its name. The canvas never pans or zooms.
- Tool tray (floating above the canvas): one segmented control for the mode (Pick, Add, Remove, Lasso), Undo set apart, and compact Bigger and Smaller (icon-only above 900 px wide, icon plus label at 900 px and below; the accessible name is always "Bigger object" or "Smaller object"). With a mouse, holding a key picks a tool only while it is held: Cmd (Ctrl on Windows and Linux) is Add, Option (Alt) is Remove, Shift is Lasso (the pixel-cut screen has no Lasso). The tray buttons show their key. The hint line under the canvas changes with the mode and follows the input device, not the width: mouse hints under `@media (hover: hover) and (pointer: fine)`, touch hints under `@media (pointer: coarse)`. A hybrid device shows the hints of its primary pointer.
- Pointer: hover outlines the object under the pointer; scroll steps to a bigger or smaller object; click picks. Touch and pen (all 44 px or more): tap previews (outline and a name tag); tap the same object again to pick; the Bigger and Smaller buttons replace scroll; Add and Remove are modes you switch on, where a mouse holds a key instead (see the tray); pen hover behaves like the mouse.
- Keyboard on a busy picture: above "Your picks" the panel has a closed fold, "Found objects", with a count. It holds one numbered button per object not yet picked, with one tab stop. Arrow keys, Home and End move between them and outline each object on the picture, as a hover does; Enter picks it. After a pick the page reloads with the fold open and the focus on the next object. Each button is named "Object 12, 140 by 90 pixels. Enter to pick". It appears only when there are no number tags.
- Undo takes back the newest change to a pick's outline: a new pick goes away, a fixed pick gets its previous outline back, a removed pick comes back under its own number. A redraw empties the list, and with nothing to take back the button is disabled.
- Side panel (one surface): "Your picks" with a count, then open rows (thumbnail, number, name, how it was picked, Remove). A warning or error is a tinted strip under its row with the way forward in a sentence. The note "Props are redrawn in groups of five." sits above the pinned main button "Redraw N props". Rebuilt props show no merge, split, pitch or settings controls. On tablet the picks collapse into a "3 picked" section under the canvas.
- Redraw job: the button gives way to the job strip ("Redrawing 3 props", "Group 1 of 1 · drawing", then "Rebuilding pixel grids · 2 of 3", a slim bar). When done, rows settle in with a stagger, each shows "Redrawn", a retried prop that failed before adds "Worked this time" (the page remembers which rows had a problem when it sent the redraw; nothing is stored on disk for it), and the main button becomes "Continue to Surfaces". Picking another object after that brings "Redraw N props" back.
- States: nothing found ("Nothing was found. Tap an object to pick it."), zero picks (main button disabled, "Pick something to redraw"), 100 picks (the list scrolls, the count chip counts), redraw running (job strip, hosted job chip), failed group (its props stay picked, a warning each).

### 2b. Pixel-cut road (step 4, for a picture that is already pixel art; mock approved 2026-10-23)
- Purpose: cut props straight from a pixel-art picture's own pixels. Free, runs on this computer, no image service.
- Where it starts: the picture page has an "Already pixel art?" section with "Cut props from this pixel art". The paid road ("find the objects") stays beside it. Once a pixel cut has been started, the timeline's Pick props step and the page's "Continue to Pick props" lead back to it.
- Size step: "Find it for me" (PixelMagic reads how big each pixel is) or "I will type it", with a "Already native: 1" shortcut for a picture that is one pixel per art pixel. The yellow button is "Start cutting". The core finds, checks and refuses the size; the page decides nothing.
- Picking screen: the picture on the neutral grey and the floating tray with Box, Add, Remove and Undo. Holding Cmd or Ctrl switches to Add and Option or Alt to Remove while held. Drag a box round a prop to pick it. A single click alone never picks. There are no numbered suggestions, no hover outlines, no scroll to resize and no lasso on this road. A pick is shown as the core's outline (whole cells), not a rectangle. Props start as prop-1, prop-2 and so on and are renamed in the picks list.
- Side panel: the size strip at the top (the core's sentence; an unsure size is the orange one), with "Start again with another size" to change it (there is no size box here). Then "Your picks", with the core's own warnings under their rows. The pinned yellow button is "Cut N props" (free), or disabled "Pick something to cut".
- After the cut: the same address shows the picture with each prop outlined and named, the props cut, and "Continue to Surfaces". A person who comes back later finds the same way on from the picking screen.
- States: no picks yet, an unsure size, a box the core refuses (its sentence), cut done.

### 3. Surface-drawing workspace (step 5)
- Purpose: tell PixelMagic where a floor or wall is and what it is made of, by drawing.
- Tool tray: Draw shape, Colour box, Move corner (one segmented control) and Undo set apart. Draw shape adds corners (one surface may have several shapes; Enter or a double-tap closes a shape; at least 3 corners). Colour box drags one rectangle on a plain patch. Move corner drags a handle (10 px dot, 44 px touch area).
- Canvas: the current shape has a strong blue outline and a light fill; finished shapes drop to about 45 percent; soft dark capsule labels ("Shape 1", the box size) never cover handles. The box shows its size live.
- Side panel (one surface), in order: Name; Kind as one segmented selector (Plaster, Stone, Wood) with a one-line example under it that changes with the choice; "Where it goes" with a compact row per shape ("Shape 1 · 7 corners", Remove) and the colour box line (size and status); "Colours" (the swatches the surface will be drawn with, five for wood and stone, four for plaster); the main button "Draw this surface"; then "Surfaces already drawn" collapsed (thumbnail, name, kind, Edit).
- Colour box rule: under 1024 pixels (any shape, about 32 by 32 as a square) the box edge and caption turn warning-coloured, the status strip says "A little bigger please: at least 1024 pixels, about 32 by 32, so the colours are true.", the swatches dim, and the main button is disabled. This replaces the old mock-only card: the real status line shows it.
- Draw job: the main button gives way to a short job strip (reading the colours, drawing 32 tiles, checking the tiles join); the new surface joins the drawn list and the button becomes "Continue to Refine".
- States: no shapes yet, box too small, name already used (sentence: "That name is taken. Pick another."), drawn surface (swatch strip), refused by the core (the sentence from the core).

### 4. Export screen (step 7)
- Purpose: see the finished room, check what goes in the bundle, export it, and know what to do next.
- Layout: timeline (Export current, Refine done), the assembled room shown big on the neutral canvas with three layer chips floating at the canvas's bottom-left (Props, Floor tiles, Structure: a light outline when on, never yellow; switching one off hides that layer; in the mock the picture is one flat stand-in, so Props and Floor tiles grey out their areas and Structure only changes its chip), and one side panel.
- Side panel, in order: "In the bundle" with the count and "All ready to export", open asset rows (thumbnail, name typed by the person, kind and size, a ready tick), "Scale" (1x shown as always included because the native size is always written; "Also include 2x" as a switch, with a note that the room and tile set files are built from 1x), then the pinned yellow "Export bundle". There is no "Open in Aseprite" card before the export.
- Export moment: the button turns in place into "Bundle exported" with a check and a soft green glow, the panel's top shows the folder path, the **Open in Aseprite guidance** (`<scene-name>-room.aseprite`, `<surface>.aseprite`, and the honest line that the room is a starting point and edits made in Aseprite stay there) and the list of files written, and step 7 completes on the timeline. "Export again" stays available.
- States: nothing refined yet (button disabled, a sentence saying which step to reopen), an asset left out (a sentence beside it saying why, the same wording as the refine screen), export running (the button shows a bar; the hosted job chip), exported (above), no Aseprite installed (the one existing warning, no room file), scale 1 not chosen (not possible here, 1x is always on; the room and tileset notes still apply to the core).
- Tablet: the room preview comes first, then the panel with the pinned button.

### 5. Services (settings, proposed 2026-10-09)
- Purpose: choose which image service draws for this game, and give a service the key it needs.
- Where it lives: a quiet "Services" link (gear icon, no yellow) at the right of the top bar on every page, and a nudge on the Start page. The page's crumbs are the game, then "Services".
- Layout: one centred column (760 px) on one rounded surface. Three open rows separated by faint dividers: Codex, Gemini, OpenAI Images (the offline stand-in is not a choice; it is reached only with `--service fake`). No boxes inside boxes.
- No service chosen: there is no default. Every Draw or Redraw control on the other pages becomes a "Choose an image service first" link to this page, in the same button style, and the new-direction form requires a name (naming is a paid call).
- A row: a radio (the whole 44 px label is the target) with the service name and one honest line (what pays for it), then its status, then its actions. The chosen row has the bright light outline, a filled radio and the words "Drawing here" (never yellow). The choice is saved per game; keys belong to the computer. The page says both.
- Status is always a sentence with an icon, never colour alone. Codex: "Signed in with your ChatGPT login", no key, a "Check login" button; if not signed in, it says to run `codex login` in a terminal. Gemini and OpenAI: "No key yet", "Key saved in your keychain" or "Key found in the environment variable GEMINI_API_KEY / OPENAI_API_KEY". When the key comes from the environment the row says it wins over any saved key and how to change it; there is no Remove, since PixelMagic cannot unset a variable.
- Keys: a write-only password field and "Save key" (yellow only on the chosen service's row and only once something is pasted; otherwise soft). Saved to the OS keychain, never shown again: the row shows only "Key saved in your keychain", with Test key, Replace key and Remove. Remove asks once inline ("Remove for good" or "Keep it"). Nothing about a key is ever written to the project folder. Saving a key for a service that is not chosen offers "Draw with <service>".
- Test: "Test key" makes a free call that draws nothing. Result in a plain sentence under the row, announced politely: "Gemini accepted the key. Nothing was drawn and nothing was charged." or "Gemini did not accept this key. Check that you copied all of it, with no spaces, or paste a new one." A failed test turns the row to the rejected state and shows the paste field.
- Privacy line, always visible under the list: the key goes only to its own service, lives in the keychain, never in the project folder or git, and is never shown again.
- Start page nudge: when the chosen service is not ready, the panel foot shows a warning strip ("Gemini needs a key. Add it in Services.", "Codex is not signed in. Fix it in Services.", "Gemini did not accept its key. Fix it in Services.") with an "Open Services" button, and "Draw it" is disabled and described by the strip. When ready there is no strip.
- States: first run (Codex chosen and signed in, other keys missing), Gemini key from the keychain, OpenAI key from the environment, Codex not signed in, a failed key test, testing (a "Asking Gemini. This costs nothing." line), key removed.
- Keyboard: radios move with arrow keys; Enter in the key field saves; focus returns to the control that replaced the one pressed. Tablet: one column; at 640 px and below the key field stacks over a full-width Save.
- Words: Services, Drawing here, Paste your Gemini key, Save key, Test key, Test again, Check login, Replace key, Remove, Remove for good, Keep it, Open Services.

## Decisions recorded 2026-10-08 (CEO feedback on the first mocks)

- Type: anything a person types (game, direction, scene, surface and prop names) is set in Inter, semibold where it is a heading.
- Workspace: the picture fills the workspace height; the side panel scrolls on its own with its main button pinned. No dead grey space under the picture.
- One yellow thing per area: the main action is the only yellow in the work area and panel; the current step is the only yellow in the timeline. The selected tool or choice uses a bright light outline plus a mark that does not rely on colour.
- Help line: one short line under the canvas showing only the hints for the current input.
- (a) On tablet the agent badge stays a tappable control, so "what it did" and Undo stay reachable.
- (b) When tap areas overlap, the thing being dragged or selected wins; otherwise the smaller target wins.

## Decisions recorded 2026-10-08 (second round: CEO ruling "beautiful, with flow")

- No squares or boxy chrome: rounded shapes, tone and space instead of borders, no hard shadows, no boxes inside boxes.
- The step rail is an animated timeline; completing a step and moving between pages are animated and reversible only by reopening.
- The pixel font (Silkscreen) appears only in the PixelMagic wordmark. Every label, button and heading is Inter.
- Motion respects reduced-motion: instant state changes and 80 ms crossfades, progress stated in words.
- The Aseprite guidance appears after the export, not before.
- The kind of surface is one segmented selector with a one-line example, not three boxes.
- The mocks include mock-only demo controls (a small pill in the top bar). They are not part of the product.

## Decisions recorded 2026-10-23 (CEO, pixel-cut mock approved)

- The pixel-cut road is built as mocked: the picture-page entry, the size step, the Box / Add / Remove / Undo tray, "Cut N props" (free) and "Continue to Surfaces".
- The floating tool tray is the app's yellow, like the main action, with dark text and the selected tool as a white pill with dark text. One style everywhere: the picking, surfaces and refine trays change to match.
- Props start as prop-1, prop-2 and so on and are renamed for free in the picks list.
- An unsure pixel size is changed by going back to the size step; the picking screen has no size box.
- No typed-box (x, y, width, height) fold for keyboard users for now.
- A pick is shown as the core's outline (whole cells), and warnings are the core's own sentences.

## Words used on these pages (for the mock)

Steps: Start, Approve, Keep scene, Pick props, Surfaces, Refine, Export. Buttons: New scene, Bring in a picture, Redraw 3 props, Continue to Surfaces, Draw this surface, Continue to Refine, Export bundle, Export again, Undo. Modes: Pick, Add, Remove, Lasso, Draw shape, Colour box, Move corner.
