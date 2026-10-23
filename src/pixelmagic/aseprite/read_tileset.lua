-- Read a surface's tiles back out of a tileset .aseprite (batch mode only).
--
-- Invoked as:
--   aseprite -b --script-param job=<path/to/job.json> --script read_tileset.lua
--
-- The job JSON is {"file": "<abs path of the .aseprite>", "out_dir": "<abs folder>"}.
-- The first tilemap layer's tileset is read. Each tile at index 1 up is written as
-- <out_dir>/tile-<index>.png (index 0 is Aseprite's own empty tile, never written or
-- counted), and <out_dir>/result.json says {"has_tilemap": bool, "count": <tiles>,
-- "sizes": [[w, h], ...]} in index order.
-- Errors use error(), never os.exit (unsupported in Aseprite 1.3 batch mode).
-- Requires Aseprite >= 1.3.

if not app.params or not app.params.job then
  error("read_tileset.lua: missing --script-param job=<job.json>")
end

local handle = io.open(app.params.job, "r")
if not handle then
  error("read_tileset.lua: cannot open job file: " .. tostring(app.params.job))
end
local job = json.decode(handle:read("a"))
handle:close()

local sprite = app.open(job.file)
if not sprite then
  error("read_tileset.lua: cannot open " .. tostring(job.file))
end

local tileset = nil
for _, layer in ipairs(sprite.layers) do
  if layer.isTilemap then
    tileset = layer.tileset
    break
  end
end

local result = { has_tilemap = tileset ~= nil, count = 0, sizes = {} }
if tileset then
  result.count = #tileset - 1
  for index = 1, result.count do
    local image = tileset:getTile(index)
    image:saveAs(job.out_dir .. "/tile-" .. index .. ".png")
    result.sizes[index] = { image.width, image.height }
  end
end

local out = io.open(job.out_dir .. "/result.json", "w")
out:write(json.encode(result))
out:close()
print("read_tileset.lua: read " .. result.count .. " tiles from " .. job.file)
