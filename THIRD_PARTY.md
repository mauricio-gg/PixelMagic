# Third-party licences

PixelMagic itself is MIT licensed (see `LICENSE`, copyright 2026 The
PixelMagic authors). This file lists every outside tool, library, font and model it
uses, under what licence, whether commercial use is allowed, and how it gets
onto a machine. Licences were read from installed package metadata and the
licence files inside the packages (versions as locked in `uv.lock` on
2026-10-22), not from memory. "Things to check" items 2 and 5 (the model weights
and the image services) were read online on 2026-10-23, with the pages named
there. Anything that could not be read is marked **unverified**. This is a
working list, not legal advice.

## Things to check

1. **The object finder's vendored SAM 2 code: settled.** `segmenter/vendor/sam2`
   is a copy of an MLX port whose README says Apache-2.0 but whose repository
   ships no LICENSE file. We ship the standard Apache License 2.0 text as
   `segmenter/vendor/LICENSE` (copied from the unmodified licence file of an
   installed Apache-2.0 package), and `segmenter/vendor/NOTICE` credits Meta
   (SAM 2) and the port (`eisneim/sam2.1_mlx`, with its commit). The copied files
   themselves are unchanged and carry no licence header, as in the port.
2. **The model weights: settled, Apache-2.0.** `fetch-model` downloads
   `sam2.1_hiera_small.safetensors` from the Hugging Face repo `eisneim/sam2.1_mlx`
   (https://huggingface.co/eisneim/sam2.1_mlx; hash pinned in
   `segmenter/src/sam_helper/model.py`). Its card metadata says
   `license: apache-2.0` and its README says "Apache 2.0 (same as the original
   SAM2)"; the repo has no LICENSE file. The file is a format conversion of Meta's
   SAM 2.1 checkpoint. Meta's README (https://github.com/facebookresearch/sam2)
   says "The SAM 2 model checkpoints ... are licensed under Apache 2.0", and Meta's
   own `facebook/sam2.1-hiera-small` is tagged apache-2.0. Commercial use is
   allowed, including on a server. PixelMagic redistributes nothing: each person
   downloads the file themselves, no weights file is committed, `segmenter/weights/`
   is git-ignored and `git ls-files segmenter` lists none. Pages read 2026-10-23.
   The only remaining risk is trusting that the conversion is faithful to Meta's
   checkpoint; neither page adds terms. An optional later step is to convert
   Meta's own checkpoint ourselves.
3. **Aseprite is paid, closed software.** The licence (EULA) on this machine
   lets you install and use it on your computer, and forbids copying it to
   others, renting it out, or reverse engineering it. It says nothing about files
   you make with it. A hosted version that runs Aseprite on our server is not
   covered by that text (unverified: ask Igara Studio). Each self-hosting person
   needs their own licence. Aseprite stays optional.
4. **Copyleft pieces (what each asks of us).**
   - `certifi` (MPL-2.0, Gemini extra only) and `tqdm` (MPL-2.0 AND MIT, segmenter
     only): file-level copyleft. Use them unchanged and nothing is required. If we
     modify their files and ship them, those files stay MPL and their source must
     be offered.
   - `libgfortran` and `libgcc` inside numpy and scipy wheels (GPL-3.0-or-later
     with the GCC exception): the exception allows use in any program, so no
     obligation on PixelMagic. `libquadmath` inside the scipy wheel is
     LGPL-2.1-or-later: nothing to do unless we redistribute
     the binaries ourselves (a Docker image counts), then include the licence text.
   - `Pillow` uses MIT-CMU, a plain permissive licence. `numpy` is BSD with some
     MIT/Zlib/0BSD/CC0 parts. No issue.
   Nothing here is GPL on its own, so nothing forces PixelMagic itself to be GPL,
   as long as we do not edit these files and only link to them.
5. **Image services: the terms bind each person, and PixelMagic ships no key.**
   Everyone uses their own key or account. Pages read 2026-10-23.
   - **Gemini API** (terms https://ai.google.dev/gemini-api/terms, last updated
     2026-04-28; Prohibited Use Policy
     https://policies.google.com/terms/generative-ai/use-policy): "Google won't
     claim ownership over that content", and commercial use is not restricted.
     The API is "for developers building with Google AI models for professional or
     business purposes, not for consumer use", and users "must be 18 years of age
     or older". Every image carries an invisible SynthID watermark
     (https://ai.google.dev/gemini-api/docs/image-generation). In Switzerland, the
     EEA and the UK even unpaid quota gets paid-tier data handling. The rule to
     use "only Paid Services when making API Clients available to users in the
     EEA, Switzerland, or the UK" is about offering an app to others, not about a
     person running PixelMagic on their own key.
   - **OpenAI API** (Services Agreement
     https://openai.com/policies/services-agreement/, effective 2026-01-01; read
     through the Wayback Machine because openai.com refused direct fetches): the
     customer "owns all Output" and OpenAI assigns its rights to it; commercial
     use is fine; API content is not used for training unless you opt in; keys
     must not be traded or shared.
   - **Codex through a ChatGPT plan** (Europe Terms of Use
     https://openai.com/policies/eu-terms-of-use/, updated 2026-01-16, which apply
     to Swiss residents; Codex auth docs https://developers.openai.com/codex/auth):
     the person owns the output, but the consumer terms forbid "Automatically or
     programmatically extracting data or Output", and OpenAI says "API keys are
     still the recommended default for automation." PixelMagic drives the Codex
     command from a program, so this is a grey area and at the person's own risk.
     No rule was found against using your own ChatGPT login in a local tool.
   - None of these terms require a tool to label AI output. Both providers forbid
     users passing AI output off as made by a person; that obligation is the
     user's.
6. **PixelRefiner is a separate MIT-licensed project and not part of this repo.**
   It lives at https://github.com/mauricio-gg/PixelRefiner (a fork of
   HappyOnigiri/PixelRefiner). It is optional and off by default. Props, floors and the offline path never use it; the scene is
   copied through by the stand-in unless the real one is turned on. Its licence is its own and is
   not covered here.
7. **Packages that only install on other systems were not on this machine, so
   their licences are unverified:** `colorama`, `pywin32`, `pywin32-ctypes`
   (Windows), `jeepney`, `secretstorage` (Linux), `httpx2-jsfetch` (browser/WASM
   only). They are pulled in by `keyring`, `pytest`, `click` and `httpx2` on those
   systems. Mac users never install them. Check them before documenting Windows or
   Linux support.
8. **Not on a PyPI licence field:** `Jinja2`, `markdown-it-py`, `mdurl`, `jaraco.classes`
   and `safetensors` show their licence only as a classifier (BSD, MIT, MIT,
   MIT, Apache) with no SPDX text; classifiers are still the project's own claim.

Everything else in the tables below is MIT, BSD, Apache-2.0, ISC, PSF or
similar and is fine for commercial use, with the usual condition that you keep
the licence notice when you redistribute the code.

## How things get installed

- **required**: comes with `uv sync`.
- **optional extra**: `uv sync --extra gemini` or `--extra openai`.
- **segmenter**: installed by `(cd segmenter && uv sync)`, a separate environment.
- **separate install**: the person installs it themselves.
- **bundled in the repo**: the file is in the git repository.

## PixelMagic's own pieces

| Item | Used for | Licence | Commercial use | Installed |
|---|---|---|---|---|
| PixelMagic | the tool itself | MIT | yes, keep the notice | this repo |
| `grain.svg`, `mosaic.svg`, `mosaic-tile.svg`, `favicon.svg` (`src/pixelmagic/web/static`) | page backgrounds and icon | our own: plain SVG shapes and a noise filter, no outside source (git history shows they arrived in our own interface commits #190 and #201) | yes | bundled in the repo |
| `pixelmagic.css`, the page `.js` files, prompts, Lua scripts | the web page, prompts, Aseprite scripts | our own | yes | bundled in the repo |

## Fonts (bundled in the repo)

| Item | Used for | Licence | Commercial use | Installed |
|---|---|---|---|---|
| Inter (`InterVariable.woff2`, copyright The Inter Project Authors) | all interface text | SIL Open Font License 1.1 (text in `Inter-OFL.txt`) | yes, keep the licence file with the font. The font may not be sold on its own. A modified copy may not use the name "Inter". | bundled in the repo |
| Silkscreen Bold (`Silkscreen-Bold.ttf`, copyright The Silkscreen Project Authors) | the wordmark only | SIL Open Font License 1.1 (text in `Silkscreen-OFL.txt`) | yes, same conditions | bundled in the repo |

## Python libraries that ship with PixelMagic (required)

Direct dependencies from `pyproject.toml` first, then what they pull in (read from `uv.lock`).

| Item | Used for | Licence | Commercial use | Installed |
|---|---|---|---|---|
| pydantic, pydantic-core | all data models | MIT | yes | required |
| Pillow | reading and writing pictures | MIT-CMU | yes, keep the notice | required |
| numpy | image maths | BSD-3-Clause and parts MIT, Zlib, 0BSD, CC0; the wheel includes libgfortran/libgcc (GPL-3 with GCC exception) | yes (see "Things to check" 4) | required |
| scipy | image maths (labelling, filling) | BSD-3-Clause; the wheel includes libgfortran/libgcc (GPL-3 with GCC exception) and libquadmath (LGPL-2.1+) | yes (see "Things to check" 4) | required |
| typer | command line | MIT | yes | required |
| fastapi | the web page | MIT | yes | required |
| jinja2 | web page templates | BSD (classifier) | yes | required |
| python-multipart | reading HTML forms | Apache-2.0 | yes, keep the notice | required |
| uvicorn | runs the web page | BSD-3-Clause | yes | required |
| mcp, mcp-types | the MCP server | MIT | yes | required |
| keyring | keeping API keys in the OS keychain | MIT | yes | required |
| annotated-doc | typer/fastapi helper | MIT | yes | required |
| annotated-types | pydantic helper | MIT | yes | required |
| anyio | async plumbing | MIT | yes | required |
| attrs | helper (jsonschema) | MIT | yes | required |
| cffi | C bindings for cryptography | MIT-0 | yes | required |
| click | typer's base | BSD-3-Clause | yes | required |
| cryptography | via mcp (token handling) | Apache-2.0 OR BSD-3-Clause | yes | required |
| h11 | HTTP | MIT | yes | required |
| httpcore2, httpx2 | HTTP client (used by mcp) | BSD-3-Clause | yes | required |
| idna | domain names | BSD-3-Clause | yes | required |
| jaraco.classes, jaraco.context, jaraco.functools, more-itertools | keyring helpers | MIT | yes | required |
| jsonschema, jsonschema-specifications, referencing, rpds-py | via mcp, JSON checking | MIT | yes | required |
| markdown-it-py, mdurl | via rich | MIT (classifier) | yes | required |
| MarkupSafe | escaping in templates | BSD-3-Clause | yes | required |
| opentelemetry-api | via mcp, tracing hooks | Apache-2.0 | yes, keep the notice | required |
| pycparser | via cffi | BSD-3-Clause | yes | required |
| PyJWT | via mcp, tokens | MIT | yes | required |
| Pygments | via rich, colouring output | BSD-2-Clause | yes | required |
| rich | via typer, terminal output | MIT | yes | required |
| shellingham | via typer, finds the shell | ISC | yes | required |
| sse-starlette | via mcp, server events | BSD-3-Clause | yes | required |
| starlette | under fastapi | BSD-3-Clause | yes | required |
| truststore | via mcp, system certificates | MIT | yes | required |
| typing-extensions | typing helpers | PSF-2.0 | yes | required |
| typing-inspection | via pydantic | MIT | yes | required |
| colorama, pywin32, pywin32-ctypes, jeepney, secretstorage, httpx2-jsfetch | Windows, Linux or browser only | unverified (not installed on this Mac) | unverified | required, on those systems only |

## Optional extras

`uv sync --extra gemini` (the Google image service):

| Item | Used for | Licence | Commercial use | Installed |
|---|---|---|---|---|
| google-genai | the Gemini API client | Apache-2.0 | yes, keep the notice | optional extra |
| google-auth | via google-genai, sign-in | Apache-2.0 | yes | optional extra |
| certifi | via requests, certificate list | MPL-2.0 | yes (file-level copyleft, see "Things to check" 4) | optional extra |
| requests, httpx, httpcore | HTTP | Apache-2.0, BSD-3-Clause, BSD-3-Clause | yes | optional extra |
| charset-normalizer, urllib3 | via requests | MIT | yes | optional extra |
| websockets | live connections | BSD-3-Clause | yes | optional extra |
| pyasn1, pyasn1-modules | via google-auth | BSD-2-Clause, BSD (classifier) | yes | optional extra |
| tenacity | retries | Apache-2.0 | yes | optional extra |
| distro | via google-genai, OS name | Apache-2.0 | yes | optional extra |
| sniffio | async helper | MIT OR Apache-2.0 | yes | optional extra |

`uv sync --extra openai` (the OpenAI image service):

| Item | Used for | Licence | Commercial use | Installed |
|---|---|---|---|---|
| openai | the OpenAI API client | Apache-2.0 | yes, keep the notice | optional extra |
| jiter | via openai, fast JSON | MIT | yes | optional extra |
| sniffio | async helper | MIT OR Apache-2.0 | yes | optional extra |

## The object finder (`segmenter/`, its own environment)

| Item | Used for | Licence | Commercial use | Installed |
|---|---|---|---|---|
| `segmenter/src/sam_helper` | our command that runs the finder | MIT (part of PixelMagic) | yes | bundled in the repo |
| `segmenter/vendor/sam2` (MLX port of Meta's SAM 2.1, source github.com/eisneim/sam2.1_mlx, commit 036f2d0) | the Segment Anything model code | Apache-2.0; licence text in `segmenter/vendor/LICENSE`, credits in `segmenter/vendor/NOTICE` (see "Things to check" 1) | yes, keep the licence and NOTICE | bundled in the repo |
| SAM 2.1 small weights (`sam2.1_hiera_small.safetensors`, Hugging Face `eisneim/sam2.1_mlx`, 184 MB) | the finder's trained model | Apache-2.0 (card metadata and README; a conversion of Meta's Apache-2.0 checkpoint) | yes (see "Things to check" 2) | separate install: `pixelmagic-segmenter fetch-model`, git-ignored, never in the repo |
| mlx, mlx-metal | runs the model on Apple chips | MIT | yes | segmenter |
| numpy, Pillow | see above | BSD / MIT-CMU | yes | segmenter |
| safetensors | reads the weights file | Apache-2.0 (classifier) | yes | segmenter |
| huggingface-hub | downloads the weights | Apache-2.0 | yes | segmenter |
| hf-xet, filelock, fsspec, PyYAML, tqdm, packaging | helpers of huggingface-hub | Apache-2.0, MIT, BSD-3-Clause, MIT, MPL-2.0 AND MIT, Apache-2.0 OR BSD-2-Clause | yes (tqdm: see "Things to check" 4) | segmenter |
| `segmenter/src/cv2` | our own stand-in for the few OpenCV calls inside the vendored code (resize only); it is not OpenCV, and OpenCV is no longer installed | MIT (part of PixelMagic) | yes | bundled in the repo |
| click, h11, httpcore2, httpx2, idna, anyio, truststore, typing-extensions | helpers | BSD-3-Clause, MIT, BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, MIT, MIT, PSF-2.0 | yes | segmenter |

## Separate tools

| Item | Used for | Licence | Commercial use | Installed |
|---|---|---|---|---|
| Python 3.12+ | runs everything | PSF licence (not read on this machine) | yes (unverified here) | separate install |
| uv | installs the Python packages | MIT or Apache-2.0 (not read on this machine) | yes (unverified here) | separate install |
| Node 24+ | only to run PixelRefiner | MIT (not read on this machine) | yes (unverified here) | separate install, optional |
| Aseprite | writes the `.aseprite` files in the export (optional) | closed source, paid EULA from Igara Studio S.A. | the person needs their own licence; the EULA says nothing on what you do with files you make; use on a server is unverified (see "Things to check" 3) | separate install, optional |
| Codex CLI (`codex`) | one of the three image services | not read on this machine (OpenAI's terms apply, see "Things to check" 5) | yours, but scripted use is a grey area under the consumer terms | separate install, optional |
| PixelRefiner | turns the scene into pixel art | separate MIT-licensed project (https://github.com/mauricio-gg/PixelRefiner), not part of this repo; its own licence applies and is not covered here | not applicable | separate install, optional, off by default |

## Online services (the person's own key or account)

| Item | Used for | Terms | Commercial use of the pictures | Installed |
|---|---|---|---|---|
| Gemini API (Google) | draws pictures | Google's Gemini API terms; every picture carries an invisible SynthID watermark; users must be 18 or older | yours, Google claims no ownership | their own key; client via optional extra |
| OpenAI Images API | draws pictures | OpenAI's Services Agreement | yours (you own all Output) | their own key; client via optional extra |
| Codex (ChatGPT sign-in) | draws pictures | OpenAI's consumer terms for the plan; they forbid scripted extraction of Output | yours, but scripted use is a grey area (see "Things to check" 5) | separate install |

## Development tools (not shipped)

pytest (MIT), ruff (MIT), anyio and httpx2 (as above), pluggy and iniconfig (MIT).
They are in the `dev` group only and are never installed for a user.
