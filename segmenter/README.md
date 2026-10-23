# pixelmagic-segmenter

The Segment Anything helper PixelMagic shells out to (SAM 2.1 small, MLX build; Apple silicon only).
It has its own environment and is never imported by PixelMagic.

```
cd segmenter
uv sync                                   # builds segmenter/.venv
uv run pixelmagic-segmenter fetch-model   # one-time, 184 MB, into segmenter/weights/ (hash-checked)
```

Then `pixelmagic suggest ...` finds it by itself at `segmenter/.venv/bin/pixelmagic-segmenter`; without it,
use `--segmenter stub`. The verbs and their JSON are documented in `src/pixelmagic/segmenter/base.py`.
`vendor/sam2` is someone else's code, copied unchanged; see `vendor/NOTICE`.
