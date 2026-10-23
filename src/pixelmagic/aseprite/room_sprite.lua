-- Build a layered room .aseprite from a plan (batch mode only).
--
-- Invoked as:
--   aseprite -b --script-param job=<path/to/job.json> --script-param cell=<n> \
--            --script room_sprite.lua
--
-- The job JSON is {"width": <n>, "height": <n>, "output": "<abs path>",
-- "surfaces": [{"name", "tiles": [<abs png>, ...] (tileset order, index 1 up),
-- "tile_data": [<string>, ...], "columns": <n>, "rows": <n>,
-- "cells": [<tile index, 0 = empty>, ...] (reading order)}, ...] (bottom to top),
-- "structure": <abs png the size of the sprite>,
-- "props": [{"name", "image": <abs png>, "x": <n>, "y": <n>}, ...] (back to front)}.
-- `cell` is the tile size in art pixels. The result is one sprite of
-- width x height with one tilemap layer per surface, then one image layer named
-- "structure" at 0,0, then one image layer per prop above them,
-- each layer named as given.
-- Errors use error(), never os.exit (unsupported in Aseprite 1.3 batch mode).
-- Requires Aseprite >= 1.3.

if not app.params or not app.params.job then
  error("room_sprite.lua: missing --script-param job=<job.json>")
end
local cell = tonumber(app.params.cell)
if not cell or cell < 1 or cell ~= math.floor(cell) then
  error("room_sprite.lua: cell must be a whole number of pixels, got " .. tostring(app.params.cell))
end

local file = io.open(app.params.job, "r")
if not file then
  error("room_sprite.lua: cannot open job file: " .. tostring(app.params.job))
end
local job = json.decode(file:read("a"))
file:close()

local function load(path)
  local image = Image{ fromFile = path }
  if not image then
    error("room_sprite.lua: cannot load image: " .. tostring(path))
  end
  return image
end

local sprite = Sprite(job.width, job.height, ColorMode.RGB)
app.sprite = sprite
local first = sprite.layers[1]

for _, surface in ipairs(job.surfaces) do
  if #surface.cells ~= surface.columns * surface.rows then
    error("room_sprite.lua: " .. surface.name .. " has " .. #surface.cells .. " cells, expected "
      .. surface.columns * surface.rows)
  end
  if #surface.tile_data ~= #surface.tiles then
    error("room_sprite.lua: " .. surface.name .. " has " .. #surface.tile_data .. " tile_data entries for "
      .. #surface.tiles .. " tiles")
  end
  -- A plain layer given a tileset stays an image layer, so the tilemap layer
  -- is made by the command.
  app.command.NewLayer{ name = surface.name, tilemap = true, gridBounds = Rectangle(0, 0, cell, cell) }
  local layer = app.layer
  local tileset = layer.tileset
  for i, path in ipairs(surface.tiles) do
    local tile = sprite:newTile(tileset, #tileset)
    tile.image = load(path)
    tile.data = surface.tile_data[i]
  end
  local map = Image(surface.columns, surface.rows, ColorMode.TILEMAP)
  for n, index in ipairs(surface.cells) do
    if index < 0 or index > #surface.tiles then
      error("room_sprite.lua: " .. surface.name .. " names tile " .. tostring(index)
        .. " but there are " .. #surface.tiles)
    end
    map:putPixel((n - 1) % surface.columns, (n - 1) // surface.columns, app.pixelColor.tile(index, 0))
  end
  local existing = layer:cel(1)
  if existing then
    sprite:deleteCel(existing)
  end
  sprite:newCel(layer, 1, map, Point(0, 0))
end

do
  local layer = sprite:newLayer()
  layer.name = "structure"
  sprite:newCel(layer, 1, load(job.structure), Point(0, 0))
end

for _, prop in ipairs(job.props) do
  local layer = sprite:newLayer()
  layer.name = prop.name
  sprite:newCel(layer, 1, load(prop.image), Point(prop.x, prop.y))
end

sprite:deleteLayer(first)
sprite:saveAs(job.output)
print("room_sprite.lua: wrote " .. job.output)
