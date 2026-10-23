# DESIGN.md: PixelMagic interface

**Status:** locked, second revision (CEO ruling, 2026-10-08: "beautiful and with flow"). Hex values contrast-checked (WCAG relative luminance).
**Pins:** dark theme; one yellow accent (#FFC833) with one yellow thing per area; a neutral canvas that never tints the art; Inter for all interface text; the pixel font (Silkscreen) on the PixelMagic wordmark and nowhere else; desktop plus tablet, with a touch equivalent for every pointer action.
**Direction:** a soft, dark creative studio around a crisp, untouched artwork canvas. The pixel character belongs to the art and the logo, not the chrome. The direction came from the 2026-10-08 Codex revamp brief.

## What changed in this revision (and why)

- **No squares, no boxy chrome.** Rounded shapes everywhere except the pixel art itself. The 2 px borders and the hard offset shadow (`--shadow-chunk`) are gone.
- **Tone and space instead of borders.** Regions are separated by a slightly different surface and by gaps. Borders survive only on inputs, focus rings, canvas geometry, and error/warning strips.
- **No boxes inside boxes.** The side panel is the one surface. Inside it: open rows, headings and faint dividers. Only the primary action, editable inputs and genuinely selectable controls are enclosed.
- **The step rail is now an animated timeline** (vertical on desktop, horizontal on tablet).
- **Motion is a system** (tokens below), and every state change has a cause.
- **Silkscreen is the wordmark only.** Buttons, steps, headings, tool labels, chips: Inter.

## Identity

- **Wordmark:** "PixelMagic" in Silkscreen Bold 700, text colour, plus one small yellow sparkle (a plus-shaped 5-pixel star) after the final "c". Never smaller than 16 px tall. This is the only pixel type in the product.
**Not built in v1.** The account menu, the usage/credits chip and the job chip are not built in v1, which is local-only. The layout only leaves room for them at the right end of the top bar; building them is ticket #195.
- **Accent rule:** yellow marks the frame, never the canvas. One yellow thing per area: the current timeline marker (timeline), the main action (panel), the focus ring. Hover is never yellow. Selected tools and choices use a light inset surface plus a bright light outline, so selection does not depend on colour.
- **Canvas rule:** the stage surround is `--canvas` (#4A4E54), a neutral grey, flat. No tint, no gradient, no vignette and no texture on it. The soft glow (see the look section) sits around the stage, outside the canvas edge, and never on the canvas or under the picture. The picture inside keeps square corners (4 px at most). There is no black box and no extra frame around the art: the picture sits straight on the grey.

## The look: mosaic, glow, grain (CEO choice, 2026-10-08)

A cozy, soft dark studio, made of three quiet layers. All of it sits around the art, never under it.

| Layer | Where | Strength |
|---|---|---|
| Pixel mosaic | The page background on every screen: 16 px cells in three near-identical greys over `--base` (+0, +3, +5 grey levels, weights 5:3:2), tiled at 256 px | Spread of about 5 levels, about 2 percent luminance. It reads as texture, not as a pattern. Half of the first trial (10 levels, 4 percent), which was too busy |
| Soft glow | Work screens only: a radial grey (`rgb(74 78 84)`, 85 percent in the middle, 42 percent at half radius, 0 at the edge) that extends 130 px above and below and 150 px beside the stage, around it and outside the canvas edge | It lifts the surround around the picture so the canvas does not sit in a hole. It never lies on the canvas or under the picture: the canvas itself stays flat |
| Grain | Panels only (side panel, its footer, Mock-only menu, game tiles, the empty-state strip): fine grey noise at 4.5 percent alpha | Panels feel like material. Never on the page, the canvas or the art |
| Scene glow | Home only, behind the big latest-game card: a copy of that game's latest scene, blurred 72 px, darkened to 50 percent brightness, saturation 135 percent, 80 percent opacity | Ties the home to the game it leads with. It never appears on a work screen (the art there must not be competing with a coloured haze). With no games there is no card and no glow |

The "no picture yet" tile placeholder uses the same mosaic idea at 8 px cells (+0, +3, +5 over #25282C).

Because the textures lift the floor under small text a little, two quiet text colours step up (below): `--text-faint` to #8E939B and the number inside a pick badge to #8CC6FF (the lighter `--info`). Both were measured on the finished textures and pass AA with room to spare.

## Fonts (SIL Open Font License 1.1, Google Fonts, free to ship)

- **Wordmark only:** Silkscreen Bold 700. Fallback `ui-monospace, "SF Mono", monospace`.
- **Everything else:** Inter (variable 400 to 700). Fallback `system-ui, sans-serif`. Sentence case always.

### Type scale

| Use | Size / line height | Weight |
|---|---|---|
| Page title | 36 / 44 | 650 |
| Panel title | 20 / 28 | 650 |
| Section | 16 / 24 | 650 |
| Body | 15 / 22 | 400 |
| Control label | 14 / 20 | 600 |
| Metadata and help | 13 / 18 | 450 |
| Small status | 12 / 16 | 600 |

Large headings get -0.02em letter-spacing. Names a person types (game, scene, prop, surface) stay more prominent (semibold, body size or larger) than the kind and status labels beside them. 12 px is the floor; body-copy minimum is 13 px (help text only).

## Tokens

```css
:root {
  /* surfaces, dark to light: tone and space group things, borders are rare */
  --base: #17181A;       /* page */
  --embedded: #1D1F22;   /* inset wells: segmented-control track, inputs, code */
  --panel: #23262A;      /* side panel, floating tray, popovers */
  --hover: #2C3035;      /* hover and selected rows, secondary buttons */
  --canvas: #4A4E54;     /* neutral canvas surround: never tinted */
  --divider: rgb(255 255 255 / 7%);
  --edge: rgb(255 255 255 / 12%);

  /* text */
  --text: #E8EAEC;  --text-muted: #A3A8AE;
  --text-faint: #8E939B;     /* waiting steps and separators (non-interactive); lifted from #858A92 so it clears AA on the textured page */
  --text-disabled: #6B7078;  /* disabled controls only */

  /* accent */
  --accent: #FFC833;  --accent-hover: #FFD964;  --accent-pressed: #E6AE14;  --on-accent: #17181A;

  /* functional */
  --error: #FF7A7A;  --success: #5AD18A;  --warning: #FF9A3D;  --info: #6CB4FF;
  --control-edge: #8B9097;   /* inputs and nothing else */
  --key-fill: #DDDFE2;  --key-edge: #959AA2;   /* a keycap's face and border (tray key hints): reads on both the yellow tray and the white selected pill */

  /* space, 4 px base: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 */
  /* shape */
  --r-input: 10px;   /* inputs, small controls, tags, segmented selections */
  --r-btn: 14px;     /* buttons, row hovers, canvas labels */
  --r-panel: 20px;   /* side panels, game tiles, floating trays */
  --r-stage: 12px;   /* the grey surround of the canvas */
  --r-art: 4px;      /* the pixel picture itself: square-ish */
  /* pills (999px) only for short status chips and counts */

  /* depth: only what really floats */
  --shadow-float: 0 12px 36px rgb(0 0 0 / 28%);   /* tray, popovers, menus */
  --shadow-lift:  0 4px 14px rgb(0 0 0 / 18%);    /* hovered buttons, lead tile, name tag */
  --inset-top:    inset 0 1px rgb(255 255 255 / 5%);
  /* things inside a raised panel cast no shadow */

  /* motion */
  --ease-out:      cubic-bezier(.22, 1, .36, 1);    /* layout moves, the timeline, enters */
  --ease-standard: cubic-bezier(.2, .8, .2, 1);     /* fades, hovers, page transitions */
  --ease-spring:   cubic-bezier(.34, 1.56, .64, 1); /* small returns, pulses, chip insertion only */
  --d-press: 80ms;  --d-hover: 140ms;  --d-lift: 180ms;  --d-tile: 220ms;
  --d-enter: 260ms; --d-progress: 320ms; --d-line: 360ms;
  --touch: 44px;
}
```

Registered property for animated progress: `@property --progress { syntax: '<number>'; inherits: true; initial-value: 0; }`. The job bar and the timeline use it so a value change animates over `--d-progress`.

## Contrast report (measured, WCAG 2.x relative luminance)

| Pair | Ratio | Result |
|---|---|---|
| text on base / panel / hover / canvas | 14.7 / 12.6 / 11.0 / 6.9 | AAA |
| text-muted (#A3A8AE) on base / panel / hover | 7.4 / 6.3 / 5.5 | AA |
| text-faint (#8E939B) on the lightest mosaic cell (#1C1D1F) | 5.6 | AA (waiting step labels, separators; was #858A92) |
| pick badge number (#8CC6FF) on its tinted badge | 5.2 (worst pixel, grain included) | AA (was `--info` #6CB4FF) |
| text-muted on the textured panel, hover and tile, worst pixel | 5.2 to 5.6 | AA |
| Measured on the final render, every text element, worst background pixel under it | lowest 5.2 | AA everywhere (white text on the scene art and the game card's dark cover is judged on that art, 5.8 at the lowest) |
| accent on base / panel | 11.5 / 9.8 | pass (text and non-text) |
| on-accent (#17181A) on accent | 11.5 | AAA |
| control edge (#8B9097) on embedded | 5.1 | 3:1 non-text pass |
| error / success / warning / info on panel | 6.0 / 7.9 / 7.2 / 7.0 | AA |
| selected tool: white outline (70%) on embedded | 8.7 | 3:1 non-text pass |
| "Bundle exported" green text on its tinted button | 5.4 | AA |

Rule: text that sits on the canvas goes on a dark chip (`--base` at 78 to 90 percent), never directly on the grey.

## Components

### Buttons
Rounded 14 px, 44 px minimum height and width, label 14/20 semibold Inter. Secondary: `--hover` fill, no border. Quiet: transparent, muted text, fill on hover. Primary: the yellow, one per area, 52 px tall and 16 px label for the large one. Hover: background change over `--d-hover` and `translateY(-1px)` with `--shadow-lift` over `--d-lift`. Press: `scale(.975)` over `--d-press`, back over `--d-lift` with the spring easing. Disabled: `--embedded` fill, `--text-disabled`, no lift. Loading: the label stays, a slim bar shows the job (see "Long job").

### Floating tool tray
One rounded 20 px surface above the canvas, in the app's yellow (`--accent`, the same fill as the primary action, no grain) with `--shadow-float`. Its text and icons are dark (`--on-accent`, semibold). Inside, a segmented control for the modes (a darker 14 percent track, 16 px radius, 44 px segments). The selected segment is a white pill with dark text (and a dark dot as its non-colour mark); hover is the track's darker tone; a disabled segment keeps the dark text at 55 percent, with no fill. A segment that a held key can switch to shows that key after its label (⌘ ⌥ ⇧ on a Mac, Ctrl Alt Shift elsewhere) drawn as a small keycap: `--key-fill` face, 1 px `--key-edge` border with a 2 px bottom edge, 4 px corners, dark 12 px medium text, 5 px side padding; only for a fine pointer and above 480 px wide, never on touch, and hidden from screen readers (the key is in the button's tooltip). Undo and the other quiet buttons are dark text on the yellow, a darker tone on hover, 55 percent when disabled, and sit after a divider (a 1 px dark line at 28 percent). Bigger and Smaller (pick screen) are icon-only above 900 px wide (the tray must stay inside its surface) and icon plus label at 900 px and below; the accessible name is always "Bigger object" or "Smaller object". The same tray is used on every screen that has one (picking, pixel-cut picking, surfaces, refine's Before and After). A segmented control outside a tray (the size step, clean-up's choice) keeps the dark track and light selected pill. The tray is yellow like the main action, so on those screens the area's yellow things are the tray and the one primary button.

### Open rows
Used in the side panels (picks, geometry, assets, drawn surfaces). Thumbnail 48 px (10 px radius), name semibold 15/22, metadata 13/18 muted, action on the right as a quiet button. 12 px vertical padding. A faint divider starts after the thumbnail. Hover: `--hover` background, dividers around it fade. Warnings and errors sit under the row as a tinted strip with a 3 px coloured side mark (warning, error, success), never as a nested card.

### Side panel
One surface, 20 px radius, 24 px inside padding (16 px on tablet). The scrolling body is separated from the pinned footer by a soft fade, not a line. The footer holds the one primary action, with its note above it.

### Timeline (see Motion for the animation)
Left column, 224 px, no box. One 2 px track runs through the centres of the 32 px markers (base `rgb(255 255 255 / 12%)`, finished part green). Markers 64 px apart (76 px around the current step), labels 14 px to the right.

| State | Marker | Label |
|---|---|---|
| Done | green ring, tinted fill, white check | 14/20 text; "Reopen" appears on hover and focus and is in the accessible name. On touch (`(pointer: coarse)`) a tap opens a small popover (`--panel`, `--shadow-float`, 280 px max) with a 44 px "Reopen this step" button and Close; the tap alone does not reopen |
| Current | dark fill, 3 px yellow ring, soft yellow halo (the only yellow here) | on a raised 14 px surface, with "You are here" |
| Next | bright neutral ring | normal text |
| Waiting | small hollow dot | `--text-faint` |
| Reopened | yellow ring on the reopened marker; later steps dimmed, their line dotted | an inline note "Editing this step. Later results may update." |

Agent-done steps carry an icon button (44 px target) that opens a popover with what the agent did and Undo. At 1180 px and below the timeline is a 64 px horizontal strip: 28 px markers on one line, done steps are checks without labels, the current step expands to a capsule with its name, and the agent badge sits on its marker.

### Canvas overlays
Object outlines are thin (2 to 2.5 px) with a translucent fill: hover is a white outline and 10 percent fill; picked is `--info` with a 20 percent fill. Number tags are 26 px dark capsules at 70 percent opacity until hover, and exist only when the picture has 40 found objects or fewer (`PAGE_SIZE`); a busier picture gets the "Found objects" fold in the panel instead, a grid of 44 px number buttons that the keyboard steps through, each outlined on the picture and carrying the yellow focus ring; picked tags are fully visible and `--info`. Only the hovered object shows its name. Shape corners are 10 px handles with a 44 px transparent touch area; the current shape is a strong `--info` outline with a light fill, finished shapes drop to 45 percent. Labels are soft dark capsules placed so they never cover handles.

### Forms
Inputs are the one place with a visible border: 1.5 px `--control-edge`, 10 px radius, 44 px tall, `--embedded` fill. Error: `--error` border plus a sentence. The 2x switch is a 44 x 26 pill.

## Motion

Everything animates for a reason: cause, effect, or orientation. Nothing is decorative, nothing loops except an honest indeterminate cycle.

| Moment | Spec |
|---|---|
| Hover, button | background `--d-hover`, lift `-1px` over `--d-lift` |
| Hover, tile or row | surface brightens; art scales to 1.015 over `--d-tile` |
| Hover, canvas target | outline settles in 100 ms; name tag rises 4 px and fades in over 140 ms |
| Press | `scale(.975)` over 80 ms, back over `--d-lift` with the spring |
| Pick an object | outline brightens 80 ms; fill 0 to 20 percent over `--d-lift`; number pulses 1 to 1.16 to 1 over 220 ms; panel row enters (opacity, 8 px up) over `--d-enter`. The canvas never pans or zooms |
| Complete a step | marker squashes to .88 (90 ms); check draws (180 ms); the line grows to the next marker (360 ms); next marker takes the ring and its label expands (240 ms). `--ease-out`, steps overlap, about 520 ms in all |
| Between pages | View Transitions API: old page fades and rises out 8 px over 160 ms; new page rises in from 10 px over 260 ms; the canvas morphs and stays stable (named `canvas`); the panel crossfades over 180 ms; the current marker slides to its new place (named `marker`); the pinned action keeps its place (named `action`). The home game tile's picture morphs into the canvas |
| Reopen a step | reopened marker takes the yellow ring; later steps dim and their line goes dotted; green line returns to neutral over 240 ms. Nothing animates backward step by step |
| Long job | the label stays ("Redrawing 3 props"); under it the honest status ("Group 1 of 1 · drawing", "Rebuilding pixel grids · 2 of 3"); a 3 px bar below. Determinate where the count is known (change animates over 320 ms, never faked); otherwise a subdued travelling highlight on a 1.2 s cycle. The panel stays usable. Finished rows settle with a 60 ms stagger |
| Export success | the button turns in place into "Bundle exported" with a check (280 ms) and a soft green glow (520 ms); the folder path, the Aseprite guidance and the files written reveal over 240 ms with a 40 ms stagger; step 7 completes on the timeline. No confetti |
| Home | games rise in with a 40 ms stagger; skeletons shimmer while loading |

**Reduced motion** (`prefers-reduced-motion: reduce`): no transforms, pulses, line growth, stagger or continuous motion. State changes are instant or an 80 ms crossfade. Progress stays explicit in the text, the indeterminate highlight becomes a static bar, and page transitions become a crossfade. Where the View Transitions API is missing (or the mock is opened from disk), the pages fall back to a plain fade.

## States (every interactive element)

Default; hover (surface or lift, never yellow); pressed (scale .975); focus (3 px yellow ring, 2 px offset); selected (light inset fill and outline, plus a mark that is not colour); disabled (muted text, no lift); loading (label stays, slim bar); error (error edge plus a sentence); success (a green check and a sentence).

## Layout

Desktop (above 1180 px): top bar 64 px; timeline 224 px; canvas fills the rest and is centred with 24 px gutters; side panel 360 px. Small laptop and tablet landscape (901 to 1180 px): timeline becomes a horizontal strip across the top; canvas and a 320 px panel below it. Tablet portrait (900 px and below): one column: timeline strip, tool tray, canvas, one-line hint, then the panel with its primary action pinned to the bottom of the screen. The home page is one column at 768 px. Frame: the 1288 px pages are centred; the wide work pages (pick, surfaces, refine, export) keep the same left edge, so the timeline and the wordmark in the top bar stay at the same x on every page, and only the right side grows, up to 1680 px of width. At 1180 px and below nothing changes.

**Hints follow the input device, not the width.** The canvas hint line shows mouse hints under `@media (hover: hover) and (pointer: fine)` and touch hints under `@media (pointer: coarse)`. A laptop window narrowed to 700 px still shows mouse hints; a tablet in landscape still shows touch hints.

## Mock-only controls

A small "Mock only" pill in the top bar opens a menu with the page links and the demo actions (complete a step, pick an object, run the redraw job, shrink the colour box, run the export, preview the hosted layout, show loading and empty). It never appears in the product. The mocks need to be served over http (for example `python3 -m http.server` in the folder that holds them; the design mocks are kept in the private working repository) for the page transitions to play; opened from disk they use the fade fallback.
