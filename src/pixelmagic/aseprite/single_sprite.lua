-- Build an editable .aseprite from one PNG (batch mode only).
--
-- Invoked as:
--   aseprite -b --script-param job=<path/to/job.json> --script-param cell=<n> \
--            --script single_sprite.lua
--
-- The job JSON is {"source_image": "<abs path>", "output": "<abs path>"}.
-- `cell` is the direction's cell size in art pixels; the sprite's grid is set
-- to cell x cell so the file shows the cells the engine lays (spec M3, §6).
-- Headless constraints honoured here: no dialogs, and nothing relies on a
-- pre-existing app.activeSprite — the sprite is created explicitly.
-- Requires Aseprite >= 1.3 for the built-in `json` global.
-- Errors use error(), never os.exit (unsupported in Aseprite 1.3 batch mode).

local function fail(message)
  error("single_sprite.lua: " .. message)
end

if not app.params or not app.params.job then
  fail("missing --script-param job=<job.json>")
end
if not app.params.cell then
  fail("missing --script-param cell=<n>")
end

-- Script params arrive as strings; a grid needs a whole number of pixels.
local cell = tonumber(app.params.cell)
if not cell or cell < 1 or cell ~= math.floor(cell) then
  fail("cell must be a whole number of pixels, got " .. tostring(app.params.cell))
end

local file = io.open(app.params.job, "r")
if not file then
  fail("cannot open job file: " .. tostring(app.params.job))
end
local job = json.decode(file:read("a"))
file:close()

local source = Image{ fromFile = job.source_image }
if not source then
  fail("cannot load source image: " .. tostring(job.source_image))
end

local sprite = Sprite(source.width, source.height, ColorMode.RGB)
local layer = sprite.layers[1]
-- A brand-new sprite already owns a cel on frame 1, and newCel errors if one
-- exists, so the empty cel is removed before the real one goes in.
local existing = layer:cel(1)
if existing then
  sprite:deleteCel(existing)
end
sprite:newCel(layer, 1, source, Point(0, 0))

-- The grid is saved in the file, so View -> Grid in Aseprite shows the cells.
sprite.gridBounds = Rectangle(0, 0, cell, cell)

sprite:saveAs(job.output)
print("single_sprite.lua: wrote " .. job.output)
