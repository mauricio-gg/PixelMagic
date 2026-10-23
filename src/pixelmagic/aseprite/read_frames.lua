-- Read a sprite's frames back out as flat RGB pictures (batch mode only).
--
-- Invoked as:
--   aseprite -b --script-param job=<path/to/job.json> --script read_frames.lua
--
-- The job JSON is {"file": "<abs path of the .aseprite>", "out_dir": "<abs folder>"}.
-- Every frame is flattened as it looks (visible layers only; an indexed sprite
-- comes out in RGB) and written as <out_dir>/frame-<n>.png, then
-- <out_dir>/result.json says {"frames": <count>}.
-- Errors use error(), never os.exit (unsupported in Aseprite 1.3 batch mode).
-- Requires Aseprite >= 1.3.

if not app.params or not app.params.job then
  error("read_frames.lua: missing --script-param job=<job.json>")
end

local handle = io.open(app.params.job, "r")
if not handle then
  error("read_frames.lua: cannot open job file: " .. tostring(app.params.job))
end
local job = json.decode(handle:read("a"))
handle:close()

local sprite = app.open(job.file)
if not sprite then
  error("read_frames.lua: cannot open " .. tostring(job.file))
end

for index = 1, #sprite.frames do
  local image = Image(sprite.width, sprite.height, ColorMode.RGB)
  image:drawSprite(sprite, index)
  image:saveAs(job.out_dir .. "/frame-" .. index .. ".png")
end

local out = io.open(job.out_dir .. "/result.json", "w")
out:write(json.encode({ frames = #sprite.frames }))
out:close()
print("read_frames.lua: read " .. #sprite.frames .. " frames from " .. job.file)
