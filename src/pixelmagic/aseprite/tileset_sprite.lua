-- Build an editable tileset .aseprite from a surface's tile PNGs (batch mode only).
--
-- Invoked as:
--   aseprite -b --script-param job=<path/to/job.json> --script-param cell=<n> \
--            --script tileset_sprite.lua
--
-- The job JSON is {"tiles": [<abs png path>, ...] (tileset order, index 1 up),
-- "tile_data": [<string>, ...] (one per tile), "layout": [<tile index>, ...]
-- (the sample floor in reading order), "columns": <n>, "rows": <n>,
-- "output": "<abs path>"}. `cell` is the tile size in art pixels.
-- The result is one sprite of columns*cell x rows*cell with one tilemap layer
-- named "sample floor"; tile index 0 stays Aseprite's own empty tile.
-- Errors use error(), never os.exit (unsupported in Aseprite 1.3 batch mode).
-- Requires Aseprite >= 1.3.

if not app.params or not app.params.job then
  error("tileset_sprite.lua: missing --script-param job=<job.json>")
end
local cell = tonumber(app.params.cell)
if not cell or cell < 1 or cell ~= math.floor(cell) then
  error("tileset_sprite.lua: cell must be a whole number of pixels, got " .. tostring(app.params.cell))
end

local file = io.open(app.params.job, "r")
if not file then
  error("tileset_sprite.lua: cannot open job file: " .. tostring(app.params.job))
end
local job = json.decode(file:read("a"))
file:close()

if #job.layout ~= job.columns * job.rows then
  error("tileset_sprite.lua: layout has " .. #job.layout .. " cells, expected " .. job.columns * job.rows)
end
if #job.tile_data ~= #job.tiles then
  error("tileset_sprite.lua: tile_data has " .. #job.tile_data .. " entries for " .. #job.tiles .. " tiles")
end

local sprite = Sprite(job.columns * cell, job.rows * cell, ColorMode.RGB)
-- A plain layer given a tileset stays an image layer, so the tilemap layer
-- is made by the command, and the default first layer is dropped.
app.sprite = sprite
local first = sprite.layers[1]
app.command.NewLayer{ name = "sample floor", tilemap = true, gridBounds = Rectangle(0, 0, cell, cell) }
sprite:deleteLayer(first)
local layer = app.layer
local tileset = layer.tileset

for i, path in ipairs(job.tiles) do
  local image = Image{ fromFile = path }
  if not image then
    error("tileset_sprite.lua: cannot load tile image: " .. tostring(path))
  end
  local tile = sprite:newTile(tileset, #tileset)
  tile.image = image
  tile.data = job.tile_data[i]
end

local map = Image(job.columns, job.rows, ColorMode.TILEMAP)
for n, index in ipairs(job.layout) do
  if index < 1 or index > #job.tiles then
    error("tileset_sprite.lua: layout names tile " .. tostring(index) .. " but there are " .. #job.tiles)
  end
  map:putPixel((n - 1) % job.columns, (n - 1) // job.columns, app.pixelColor.tile(index, 0))
end
local existing = layer:cel(1)
if existing then
  sprite:deleteCel(existing)
end
sprite:newCel(layer, 1, map, Point(0, 0))

sprite:saveAs(job.output)
print("tileset_sprite.lua: wrote " .. job.output)
