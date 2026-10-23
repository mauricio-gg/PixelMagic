"""Draw with the Codex CLI's built-in image tool.

Codex is billed against the ChatGPT subscription rather than per image, which
is why the spec puts it first (locked decision 1). It is not an HTTP API: it is
an agent, and the whole job here is giving it one instruction it cannot
misread and then finding the file it made.

CONTRACT — read off `codex-cli 0.152.0` on 2026-09-04 by running it three
times from a scratch directory and keeping the replies. Nothing below is
guessed. The spec points at a headless driver that no longer exists, and
nothing else in the workspace runs `codex exec`, so this docstring is the only
surviving source.

The command:

    codex exec --skip-git-repo-check -s workspace-write -C <workspace>
        --json -o <workspace>/last.txt [-m MODEL] [-i REF]... <brief>

  --skip-git-repo-check   a scratch workspace is not a git repo and is
                          otherwise refused
  -s workspace-write      lets it write inside -C, and nowhere else
  -C <workspace>          the working root; the brief only ever names paths
                          inside it
  --json                  JSONL events on stdout
  -o FILE                 the agent's last message, written to a file
  -m MODEL                optional; `request.model` when the direction sets one
  -i FILE                 attaches one image to the prompt; repeat per image
  <brief>                 the positional prompt, last

`describe` runs the same binary in a lighter mode — no writing, no session —
because naming a cutout or inventing a direction name touches nothing:

    codex exec --skip-git-repo-check --ephemeral -s read-only -C <workspace>
        --json -o <workspace>/last.txt <question> [-i IMAGE]

  --ephemeral   the session leaves nothing behind once it answers
  -s read-only  no filesystem write, ever, on this call

Events on stdout, one JSON object per line:

    {"type":"thread.started","thread_id":"01a06ce6-…"}   always first
    {"type":"turn.started"}
    {"type":"item.started"  ,"item":{…}}                 command_execution
    {"type":"item.completed","item":{"type":"agent_message","text":"DONE"}}
    {"type":"turn.completed","usage":{…}}                always last

`thread_id` is the session id, and it is also the folder the images land in.
The image tool itself emits NO event of its own — only `agent_message` and
`command_execution` items ever appear — so there is nothing on stdout that
says a picture was made. The only proof is the file.

Where the picture lands:

    $CODEX_HOME/generated_images/<thread_id>/exec-<uuid>.png

$CODEX_HOME defaults to ~/.codex. That folder is authoritative and the
workspace copy is only the fallback, because on 2026-09-05 the first real
`run` showed why: the image tool prints no file path, so Codex guessed where
its picture was with a shell wildcard over the whole temp directory, copied
the newest .png it found — one of this repo's own test files — to
`<workspace>/out/image.png` and exited 0. A copy that is present is therefore
no evidence that it is the right file, while the thread's own folder is the
one place that certainly holds this run's picture.

Transparency works. A probe on 2026-09-04 asked for a red circle on a fully
transparent background and got back a 1254x1254 RGBA PNG whose alpha ran the
full 0..255 and whose four corners were all 0. That is why
`capabilities().transparent_background` is True — and it is the only reason,
because `stages.generate.choose_background` acts on it (locked decision 6).

Sizes are not controllable. The same probe asked for "a very small image" and
got 1254x1254: `codex exec` has no size flag and the built-in tool takes no
destination or dimension argument. `capabilities().sizes` is therefore empty
and a request whose size the picture does not match is recorded as a warning
on the result rather than treated as a failure.

stderr is not a signal. A run that exited 0 in the probe still printed three
`rmcp::transport::worker` errors from an unrelated MCP server in the user's
config. Only the exit code and the file decide.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from pixelmagic.model import GenerationRequest, GenerationResult
from pixelmagic.providers.base import ImageCapabilities, ServiceError

#: Overrides the command, split the way a shell would split it. A person who
#: keeps codex somewhere unusual, or who wants a config profile, sets this.
COMMAND_ENV = "PIXELMAGIC_CODEX_BIN"

#: The command, as argv tokens. A list because the override may add flags.
DEFAULT_COMMAND: tuple[str, ...] = ("codex",)

#: `$CODEX_HOME`, which decides where `generated_images/` lives.
CODEX_HOME_ENV = "CODEX_HOME"

#: A scene-sized picture took about a minute in the probe. Ten minutes leaves
#: room for a slow queue without hanging a generate or redraw call all afternoon.
DEFAULT_TIMEOUT = 600

#: What the brief asks Codex to copy the picture to, inside the workspace.
OUTPUT_NAME = "image.png"

#: The name of the service, and the code prefix on an exit-code failure.
SERVICE_NAME = "codex"

#: Added to the brief only when there are references. A sentence about images
#: that are not attached invites the model to invent some.
REFERENCE_NOTE = (
    "The {count} image(s) attached to this message are references for style "
    "and subject, in the order given. Pass them to the image tool as "
    "references. Do not copy them and do not open them with any other tool.\n"
)

#: The one-job brief. Everything about it is defensive: Codex is an agent with
#: a shell, and left to itself it will read files, write notes and explain its
#: reasoning — all of which costs time and none of which we want.
BRIEF = """\
You are running one job with nobody watching. Do exactly what is listed below \
and nothing else.

1. Call the built-in image_gen tool exactly once, with the prompt between the \
markers at the end of this message, used verbatim. Do not rewrite it, do not \
shorten it, do not add to it.
{reference_note}\
2. The image tool saves its picture under $CODEX_HOME/generated_images/ \
(usually ~/.codex/generated_images/). Copy that exact file — the one the tool \
has just made — to {out_path} with `cp`, and create the folder first if it \
does not exist. Do not search any other folder, do not use a wildcard over \
the temp directory, and if you cannot find the file the tool made, reply \
FAILED and say so.
3. Reply with the single word DONE.

Do not read any other file. Do not write any other file. Do not run any other \
command. Do not ask a question — there is nobody to answer it. If the image \
tool fails, reply with the single word FAILED followed by one sentence saying \
why, and stop.

--- PROMPT START ---
{prompt}
--- PROMPT END ---
"""

#: What `describe` asks, wrapped so a one-word answer comes back clean.
DESCRIBE_BRIEF = """\
Answer the question below in one short line and nothing else. Do not run any \
command, do not read or write any file, and do not explain yourself.

{question}
"""


def default_command() -> list[str]:
    """The command to run: the environment's, or plain `codex`.

    Split the way a shell would split it, so an override can carry flags —
    `codex --profile art` is two tokens, not one impossible executable name.
    """
    override = os.environ.get(COMMAND_ENV)
    if override:
        return shlex.split(override)
    return list(DEFAULT_COMMAND)


def default_codex_home() -> Path:
    """Where Codex keeps its state, and so where generated images land.

    Named `default_` because `CodexImageService.__init__` takes a `codex_home`
    argument, and a module function with the same name as a parameter is a
    function the constructor cannot reach.
    """
    override = os.environ.get(CODEX_HOME_ENV)
    return Path(override) if override else Path.home() / ".codex"


def build_brief(prompt: str, out_path: Path, *, reference_count: int) -> str:
    """Wrap one art prompt in the one-job instruction Codex is given.

    The prompt travels between markers and is never edited: `request.json`
    records the prompt PixelMagic composed, and a service that quietly reworded
    it would make that record a lie.
    """
    note = REFERENCE_NOTE.format(count=reference_count) if reference_count else ""
    return BRIEF.format(prompt=prompt.strip(), out_path=out_path, reference_note=note)


def build_generate_args(
    brief: str,
    workspace: Path,
    *,
    model: str | None,
    reference_paths: Sequence[Path],
) -> list[str]:
    """The arguments after the command tokens, for one drawing run.

    Paths are absolute: `-C` moves the process's working root, so a relative
    one would be read against a directory this code never chose.
    """
    args = [
        "exec",
        "--skip-git-repo-check",
        "-s",
        "workspace-write",
        "-C",
        str(workspace),
        "--json",
        "-o",
        str(workspace / "last.txt"),
    ]
    if model:
        args += ["-m", model]
    # The brief goes here — before every `-i`, and the ordering is load-bearing.
    # `-i` is spelled `--image <FILE>...`: it is multi-value and keeps eating
    # arguments until the next flag, so a positional placed after one is read
    # as another image path. Codex then falls back to reading its prompt from
    # stdin, finds nothing there, and draws nothing. (`_run` starts Codex with
    # stdin closed, so the prompt only ever arrives here, as the positional.)
    # Verified against codex-cli 0.152.0; see the register at the top of the
    # plan for the probes. Bound as the single positional PROMPT, the brief is safe even if
    # a service invents a line inside it that starts with a dash.
    args.append(brief)
    for path in reference_paths:
        args += ["-i", str(Path(path).resolve())]
    return args


def thread_id(stdout: str) -> str | None:
    """The session id out of the first `thread.started` event, if there is one.

    Returns None rather than raising: a run that never printed one still
    produced whatever it produced, and the id is provenance, not a result.
    """
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started":
            found = event.get("thread_id")
            return str(found) if found else None
    return None


def _newest_generated_image(home: Path, session: str | None) -> Path | None:
    """The most recent PNG in this session's generated-images folder.

    Codex names them `exec-<uuid>.png` in no useful order, so the newest by
    modification time is the only way to pick. The brief asks for exactly one
    image, so there is normally only one.
    """
    if not session:
        return None
    folder = home / "generated_images" / session
    if not folder.is_dir():
        return None
    images = [path for path in folder.glob("*.png") if path.is_file()]
    if not images:
        return None
    return max(images, key=lambda path: path.stat().st_mtime)


def _save_as_png(source: Path, out_path: Path) -> tuple[int, int]:
    """Copy `source` to `out_path` as a PNG, and return the size it really is.

    Reading it through Pillow rather than copying the bytes is deliberate: it
    proves the file is a picture here, at the boundary, instead of four stages
    later when the slicer opens it. A truncated download and a text file that
    happens to end in .png both fail on this line.
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.load()
            size = image.size
            image.save(out_path, format="PNG")
    # `UnidentifiedImageError` is itself an `OSError` (Pillow 12), so naming it
    # separately would be redundant; `DecompressionBombError` is not — it
    # derives from `Exception` — so it is named here or it escapes untyped.
    except (OSError, Image.DecompressionBombError) as error:
        raise ServiceError(
            f"codex left {source}, but that is not a picture this build can "
            f"read ({error})",
            service=SERVICE_NAME,
            code="UNREADABLE_IMAGE",
        ) from error
    return size


def _save_first_readable(
    sources: Sequence[Path], out_path: Path
) -> tuple[tuple[int, int], Path]:
    """Save the first of `sources` that really is a picture, in the order given.

    A candidate that exists is not proof: Codex can leave a file that is not a
    picture where a picture should be (see the module docstring), so each one
    is opened in turn and the first Pillow can read wins. When none can be
    read, the failure raised is the first one's, because `sources` is in
    priority order and that is the file the caller most wanted. `sources` is
    never empty — `generate` raises `NO_IMAGE` before calling here.
    """
    first_failure: ServiceError | None = None
    for source in sources:
        try:
            return _save_as_png(source, out_path), source
        except ServiceError as error:
            if first_failure is None:
                first_failure = error
    raise first_failure


class CodexImageService:
    """An `ImageService` backed by the `codex` command's built-in image tool."""

    name = SERVICE_NAME

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        timeout: int = DEFAULT_TIMEOUT,
        codex_home: Path | None = None,
    ) -> None:
        """Bind the command, the function that runs it, and where Codex lives.

        `runner` is injected for the same reason `PixelRefinerCli` injects one:
        every test in this repo runs offline, and a service that reached for
        `subprocess.run` directly could only be tested by starting Codex.
        """
        self._command = list(command) if command else default_command()
        self._label = shlex.join(self._command)
        self._runner = runner
        self._timeout = timeout
        self._home = Path(codex_home) if codex_home else default_codex_home()

    def capabilities(self) -> ImageCapabilities:
        """Describe what this service supports.

        Transparency is True because a probe on 2026-09-04 produced a real
        alpha channel; `sizes` is empty because nothing in `codex exec` sets
        the pixel size. Both are read by the stages, not by a person, so an
        optimistic answer here would become a wrong decision there.
        """
        return ImageCapabilities(
            transparent_background=True,
            reference_images=True,
            edit=False,
            vision=True,
            sizes=[],
        )

    def _run(self, args: Sequence[str], workspace: Path) -> tuple[str, str]:
        """Run one `codex exec` and return its stdout and its last message.

        Every way this can go wrong becomes a `ServiceError`, because that is
        the whole failure contract: a stage that caught anything else could not
        clean up after it.
        """
        command = [*self._command, *args]
        try:
            # stdin is the null device (#149): in an agent's shell it stays
            # open and Codex waits on it until the timeout. The brief is the
            # positional PROMPT, so Codex never needs stdin.
            completed = self._runner(
                command, capture_output=True, text=True, check=False,
                stdin=subprocess.DEVNULL, timeout=self._timeout,
            )
        except FileNotFoundError as error:
            raise ServiceError(
                f"{self._label} is not installed — install the Codex CLI and "
                f"log in, or set {COMMAND_ENV}",
                service=self.name,
                code="NOT_INSTALLED",
            ) from error
        except subprocess.TimeoutExpired as error:
            partial = error.stderr or b""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            tail = partial.strip()[-400:]
            raise ServiceError(
                f"{self._label} exec did not finish within {self._timeout} "
                "seconds. Likely causes: Codex is waiting on a login or "
                "approval prompt, or the image service is slow or down."
                + (f" Last of its stderr: {tail}" if tail else ""),
                service=self.name,
                code="TIMEOUT",
            ) from error
        except OSError as error:
            raise ServiceError(
                f"{self._label} could not be started: {error}",
                service=self.name,
                code="NOT_STARTED",
            ) from error

        last_message = ""
        message_file = workspace / "last.txt"
        if message_file.exists():
            last_message = message_file.read_text(encoding="utf-8", errors="replace").strip()

        if completed.returncode != 0:
            # stderr is noisy even on a clean run, so it is only quoted when
            # the exit code already says something went wrong.
            detail = (last_message or completed.stderr or completed.stdout).strip()
            raise ServiceError(
                f"{self._label} exited {completed.returncode}: {detail[:400]}",
                service=self.name,
                code=f"EXIT_{completed.returncode}",
            )
        return completed.stdout, last_message

    def generate(
        self,
        request: GenerationRequest,
        out_path: Path,
        *,
        reference_paths: Sequence[Path] = (),
    ) -> GenerationResult:
        """Draw `request` with the built-in image tool and save it at `out_path`.

        The workspace is a temporary directory Codex is confined to, so a run
        that misreads the brief and starts writing notes writes them somewhere
        that is deleted a moment later.
        """
        started = time.monotonic()
        warnings: list[str] = []
        with tempfile.TemporaryDirectory(prefix="pixelmagic-codex-") as scratch:
            workspace = Path(scratch)
            wanted = workspace / "out" / OUTPUT_NAME
            brief = build_brief(
                request.prompt, wanted, reference_count=len(reference_paths)
            )
            args = build_generate_args(
                brief, workspace, model=request.model, reference_paths=reference_paths
            )
            stdout, last_message = self._run(args, workspace)
            session = thread_id(stdout)

            # The thread's own folder comes first: it is the only place that
            # is certainly this run's picture. The workspace copy is Codex's
            # own `cp`, and an agent guessing at a filename can copy anything.
            drawn = _newest_generated_image(self._home, session)
            copied = wanted if wanted.is_file() else None
            candidates = [path for path in (drawn, copied) if path is not None]
            if not candidates:
                raise ServiceError(
                    f"codex produced no picture: nothing at "
                    f"{Path('out') / OUTPUT_NAME} in its workspace and nothing "
                    f"in {self._home / 'generated_images' / (session or '<no session>')}. "
                    f"Its last words were: {last_message or '(none)'}",
                    service=self.name,
                    code="NO_IMAGE",
                )
            (width, height), used = _save_first_readable(candidates, out_path)
            if used is copied:
                # Said of the file that was actually taken rather than of the
                # one that was missing, so the warning stays true when the
                # drawn file exists but turns out to be unreadable.
                warnings.append(
                    "codex's own generated_images folder held no picture this "
                    "build could read, so the copy codex made by hand in its "
                    "workspace was used instead"
                )

        if (width, height) != tuple(request.size):
            # Not a failure: `codex exec` has no size flag, so this is the
            # normal case and the operator needs to know rather than be stopped.
            warnings.append(
                f"asked for {request.size[0]}x{request.size[1]} and got "
                f"{width}x{height}; codex does not take a size"
            )
        return GenerationResult(
            id=request.id,
            created=datetime.now(UTC),
            service=self.name,
            model=request.model,
            session_id=session,
            elapsed_seconds=round(time.monotonic() - started, 3),
            warnings=warnings,
        )

    def describe(self, image_path: Path | None, question: str) -> str:
        """Answer a short question, optionally about one picture.

        Used to name cutouts and to invent an art-direction name. This runs
        `--ephemeral` and read-only: naming is a side errand and must not leave
        a session behind or be able to touch anything.
        """
        with tempfile.TemporaryDirectory(prefix="pixelmagic-codex-ask-") as scratch:
            workspace = Path(scratch)
            args = [
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "-s",
                "read-only",
                "-C",
                str(workspace),
                "--json",
                "-o",
                str(workspace / "last.txt"),
            ]
            # Same ordering rule as `build_generate_args`, and it bites harder
            # here: `describe` always attaches the cutout, so a question placed
            # after the `-i` would be eaten on *every* call and every piece
            # would come back named `item` — silently, because
            # `naming.describe_cutout` catches everything and falls back.
            args.append(DESCRIBE_BRIEF.format(question=question.strip()))
            if image_path is not None:
                args += ["-i", str(Path(image_path).resolve())]
            _, last_message = self._run(args, workspace)
        return last_message
