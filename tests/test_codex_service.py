import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import BackgroundPolicy, GenerationRequest, PromptKind
from pixelmagic.providers.base import ServiceError
from pixelmagic.providers.codex import (
    CODEX_HOME_ENV,
    COMMAND_ENV,
    DEFAULT_COMMAND,
    OUTPUT_NAME,
    CodexImageService,
    build_brief,
    build_generate_args,
    default_command,
    thread_id,
)

NOW = "2026-09-04T14:49:55+00:00"
THREAD = "01a06ce6-2789-7b42-b490-1189401fd019"


@pytest.fixture(autouse=True)
def isolated_codex_home(request, monkeypatch, tmp_path):
    """Point every offline test at a throwaway `CODEX_HOME`.

    `test_a_size_codex_ignored_is_recorded_as_a_warning` used to pass for the
    wrong reason: it never gave the service a `codex_home`, so
    `default_codex_home()` fell through to the developer's real `~/.codex`,
    where an earlier live probe had left a picture at the exact thread id this
    file reuses (`THREAD`) and of the exact size the test's fake workspace
    writer produces. The assertion passed by reading a file the test never
    wrote. Every test that is not marked `live` now runs against an empty
    `tmp_path / "codex-home"` instead, so a test can only see a picture this
    file put there. `default_codex_home()` still reads `CODEX_HOME` — that is
    what this fixture relies on.
    """
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setenv(CODEX_HOME_ENV, str(tmp_path / "codex-home"))


#: A trimmed copy of a real `codex exec --json` run (codex-cli 0.152.0,
#: 2026-09-04). Every line is spelled the way the CLI spells it.
EVENTS = "\n".join(
    [
        json.dumps({"type": "thread.started", "thread_id": THREAD}),
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {"type": "item.completed",
             "item": {"id": "item_0", "type": "agent_message", "text": "DONE"}}
        ),
        json.dumps({"type": "turn.completed", "usage": {"output_tokens": 829}}),
    ]
)


def request(size=(1024, 1024), model=None):
    return GenerationRequest(
        id="20260904-144955",
        created=NOW,
        subject="a wooden chair",
        kind=PromptKind.PROP,
        prompt="Draw a wooden chair.",
        service="codex",
        model=model,
        size=size,
        background=BackgroundPolicy.TRANSPARENT,
    )


def write_png(path: Path, size=(1024, 1024)) -> Path:
    """A tiny valid RGBA PNG, which is all the pickup code has to read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (10, 20, 200, 255)
    Image.fromarray(pixels, "RGBA").save(path)
    return path


class FakeCodex:
    """Stands in for `codex exec`: records the argv, then acts like the real one.

    `workspace_writer` is handed the `-C` directory the way Codex is: it is
    the only thing that decides whether an image turns up and where.
    """

    def __init__(self, workspace_writer=None, *, returncode=0, stdout=EVENTS, stderr="",
                 last_message="DONE"):
        self.commands: list[list[str]] = []
        self._writer = workspace_writer
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._last_message = last_message

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        workspace = Path(command[command.index("-C") + 1])
        if self._last_message is not None:
            Path(command[command.index("-o") + 1]).write_text(self._last_message)
        if self._writer is not None:
            self._writer(workspace)
        return SimpleNamespace(
            returncode=self._returncode, stdout=self._stdout, stderr=self._stderr
        )


def in_workspace(workspace: Path) -> None:
    """Codex did as it was told: the picture is where the brief asked for it."""
    write_png(workspace / "out" / OUTPUT_NAME)


def service(runner, *, codex_home=None, timeout=600):
    return CodexImageService(
        ["codex"], runner=runner, timeout=timeout, codex_home=codex_home
    )


def test_the_brief_carries_the_prompt_verbatim_between_markers(tmp_path):
    brief = build_brief("Draw a wooden chair.", tmp_path / "out" / OUTPUT_NAME,
                        reference_count=0)
    assert "Draw a wooden chair." in brief
    assert "--- PROMPT START ---" in brief and "--- PROMPT END ---" in brief
    assert str(tmp_path / "out" / OUTPUT_NAME) in brief
    # With no references there is nothing to say about them, and a sentence
    # about images that are not there invites the model to invent some.
    assert "reference" not in brief.lower()


def test_the_brief_explains_the_references_when_there_are_some(tmp_path):
    brief = build_brief("Draw a chair.", tmp_path / "out" / OUTPUT_NAME,
                        reference_count=2)
    assert "reference" in brief.lower()


def test_the_argv_is_the_one_that_was_proved_against_the_binary(tmp_path):
    args = build_generate_args("BRIEF", tmp_path, model=None, reference_paths=())
    assert args[0] == "exec"
    # Without this a scratch workspace is refused for not being a git repo.
    assert "--skip-git-repo-check" in args
    assert args[args.index("-s") + 1] == "workspace-write"
    assert args[args.index("-C") + 1] == str(tmp_path)
    assert "--json" in args
    assert args[args.index("-o") + 1] == str(tmp_path / "last.txt")
    # The brief is the positional prompt. With no reference attached it is
    # also the last token; with one it is not, because every `-i` follows it —
    # see `test_the_brief_is_never_placed_after_an_image_flag`.
    assert args[-1] == "BRIEF"


def test_each_reference_gets_its_own_image_flag(tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    args = build_generate_args(
        "BRIEF", tmp_path, model="gpt-5", reference_paths=[first, second]
    )
    images = [args[index + 1] for index, part in enumerate(args) if part == "-i"]
    assert images == [str(first), str(second)]
    assert args[args.index("-m") + 1] == "gpt-5"


def test_the_brief_is_never_placed_after_an_image_flag(tmp_path):
    """`-i` is `--image <FILE>...` and eats arguments until the next flag.

    A brief that follows an `-i` is swallowed as one more image path; Codex
    then reads its prompt from stdin, finds none, and draws nothing. This
    asserts the binary's parsing rule rather than the shape of a list, which
    is the difference between a test that would have caught it and one that
    would not. It matters from the *second* generation in a direction onward,
    because the first one has no references to attach.
    """
    reference = tmp_path / "a.png"
    args = build_generate_args(
        "BRIEF", tmp_path, model=None, reference_paths=[reference]
    )
    assert args.index("BRIEF") < args.index("-i")
    assert args[-2:] == ["-i", str(reference.resolve())]
    # With nothing attached there is no -i at all and the brief is simply last.
    assert build_generate_args(
        "BRIEF", tmp_path, model=None, reference_paths=[]
    )[-1] == "BRIEF"


def test_the_thread_id_is_read_off_the_first_event():
    assert thread_id(EVENTS) == THREAD
    assert thread_id("not json at all") is None


def test_a_picture_in_the_workspace_is_picked_up(tmp_path):
    """The fallback: nothing in generated_images, so Codex's own copy is used.

    It is a fallback and not the normal path because the copy is made by an
    agent guessing at a filename, so the result has to say it was used.
    """
    runner = FakeCodex(in_workspace)
    out = tmp_path / "image.png"
    result = service(runner, codex_home=tmp_path / "codex-home").generate(
        request(), out
    )
    assert out.exists()
    assert Image.open(out).mode == "RGBA"
    assert result.session_id == THREAD
    assert result.service == "codex"
    assert result.elapsed_seconds >= 0
    assert any("copy" in warning for warning in result.warnings)


def test_a_picture_only_in_the_codex_home_is_still_found(tmp_path):
    """The normal path: the tool's own output folder, so nothing to report."""
    home = tmp_path / "codex-home"
    write_png(home / "generated_images" / THREAD / "exec-1015a8d5.png")
    runner = FakeCodex()  # writes nothing into the workspace
    out = tmp_path / "image.png"
    result = service(runner, codex_home=home).generate(request(), out)
    assert out.exists()
    assert result.warnings == []


def test_a_wrong_file_in_the_workspace_does_not_hide_the_real_picture(tmp_path):
    """The 2026-09-05 failure: Codex copied a file that was not its picture.

    The image tool printed no path, so Codex guessed with a shell wildcard
    over the whole temp tree and copied the newest .png it found — one of our
    own test files, with exit 0. The picture it really made was in its own
    output folder all along, so that folder has to be read first.
    """
    home = tmp_path / "codex-home"
    write_png(home / "generated_images" / THREAD / "exec-1015a8d5.png", size=(64, 64))

    def copies_the_wrong_file(workspace: Path) -> None:
        (workspace / "out").mkdir(parents=True, exist_ok=True)
        (workspace / "out" / OUTPUT_NAME).write_bytes(b"pretend png")

    out = tmp_path / "image.png"
    result = service(FakeCodex(copies_the_wrong_file), codex_home=home).generate(
        request(size=(64, 64)), out
    )
    with Image.open(out) as image:
        assert image.mode == "RGBA"
        assert image.size == (64, 64)
    assert result.session_id == THREAD


def test_the_thread_folder_wins_over_the_workspace_copy(tmp_path):
    """Both files are readable pictures, so only the order decides.

    A wrong copy that happens to be a real picture is the quiet version of the
    same failure: nothing would ever have said the bundle held the wrong art.
    """
    home = tmp_path / "codex-home"
    write_png(home / "generated_images" / THREAD / "exec-1015a8d5.png", size=(32, 32))

    def copies_another_picture(workspace: Path) -> None:
        write_png(workspace / "out" / OUTPUT_NAME, size=(128, 128))

    out = tmp_path / "image.png"
    service(FakeCodex(copies_another_picture), codex_home=home).generate(
        request(size=(32, 32)), out
    )
    with Image.open(out) as image:
        assert image.size == (32, 32)


def test_the_newest_file_in_the_codex_home_wins(tmp_path):
    home = tmp_path / "codex-home"
    folder = home / "generated_images" / THREAD
    old = write_png(folder / "exec-old.png", size=(8, 8))
    new = write_png(folder / "exec-new.png", size=(16, 16))
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    out = tmp_path / "image.png"
    service(FakeCodex(), codex_home=home).generate(request(), out)
    assert Image.open(out).size == (16, 16)


def test_no_picture_anywhere_is_a_service_error_naming_both_places(tmp_path):
    home = tmp_path / "codex-home"
    with pytest.raises(ServiceError) as caught:
        service(FakeCodex(last_message="FAILED"), codex_home=home).generate(
            request(), tmp_path / "image.png"
        )
    assert caught.value.service == "codex"
    assert "out/image.png" in str(caught.value)
    assert "generated_images" in str(caught.value)
    # The agent's own last word is the best clue there is about why.
    assert "FAILED" in str(caught.value)


def test_a_non_zero_exit_is_a_service_error_with_the_stderr(tmp_path):
    # `last_message=None` means no last.txt is written, which is what a run
    # that failed before Codex ever spoke actually looks like — and it is the
    # only case where stderr is the best detail there is.
    runner = FakeCodex(returncode=1, stdout="", stderr="not logged in",
                       last_message=None)
    with pytest.raises(ServiceError) as caught:
        service(runner).generate(request(), tmp_path / "image.png")
    assert "not logged in" in str(caught.value)
    assert caught.value.code == "EXIT_1"


def test_a_timeout_is_a_service_error_not_a_subprocess_error(tmp_path):
    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 600)

    with pytest.raises(ServiceError, match="codex exec did not finish within 600") as raised:
        service(slow).generate(request(), tmp_path / "image.png")
    assert raised.value.code == "TIMEOUT"


def test_codex_is_started_with_stdin_closed(tmp_path):
    # catches: dropping stdin=DEVNULL brings back the #149 hang, where Codex
    # waits on an open stdin in an agent's shell until the 600 s timeout.
    seen = {}

    def runner(command, **kwargs):
        seen.update(kwargs)
        return FakeCodex(in_workspace)(command, **kwargs)

    service(runner).generate(request(), tmp_path / "image.png")
    assert seen["stdin"] == subprocess.DEVNULL


def test_a_missing_codex_binary_says_so(tmp_path):
    def missing(command, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "codex")

    with pytest.raises(ServiceError, match="not installed"):
        service(missing).generate(request(), tmp_path / "image.png")


def test_a_file_that_is_not_an_image_is_a_service_error(tmp_path):
    """The only candidate that exists is rubbish, so there is nowhere to fall."""

    def writes_rubbish(workspace: Path) -> None:
        (workspace / "out").mkdir(parents=True)
        (workspace / "out" / OUTPUT_NAME).write_bytes(b"not a png")

    with pytest.raises(ServiceError, match="not a picture") as caught:
        service(
            FakeCodex(writes_rubbish), codex_home=tmp_path / "codex-home"
        ).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "UNREADABLE_IMAGE"
    assert OUTPUT_NAME in str(caught.value)


def test_a_decompression_bomb_is_a_service_error(tmp_path, monkeypatch):
    """`DecompressionBombError` is an `Exception`, not an `OSError`.

    `_save_as_png` opened `source` inside a `try` that only caught `OSError`,
    so a candidate Pillow judges to be a decompression bomb escaped `generate`
    untyped instead of becoming a `ServiceError` — the one thing CLAUDE.md says
    a service must never do. Lowering `Image.MAX_IMAGE_PIXELS` well below a
    small real picture reproduces the bomb check without needing an actually
    huge file.
    """
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    def writes_a_normal_picture(workspace: Path) -> None:
        write_png(workspace / "out" / OUTPUT_NAME, size=(64, 64))

    with pytest.raises(ServiceError) as caught:
        service(
            FakeCodex(writes_a_normal_picture), codex_home=tmp_path / "codex-home"
        ).generate(request(), tmp_path / "image.png")
    assert caught.value.code == "UNREADABLE_IMAGE"


def test_a_size_codex_ignored_is_recorded_as_a_warning(tmp_path):
    def wrong_size(workspace: Path) -> None:
        write_png(workspace / "out" / OUTPUT_NAME, size=(1254, 1254))

    result = service(FakeCodex(wrong_size)).generate(
        request(size=(1024, 1024)), tmp_path / "image.png"
    )
    assert any("1024x1024" in w and "1254x1254" in w for w in result.warnings)


def test_transparency_is_advertised_because_the_built_in_tool_produces_it(tmp_path):
    capabilities = CodexImageService(["codex"], runner=FakeCodex()).capabilities()
    assert capabilities.transparent_background is True
    assert capabilities.reference_images is True
    assert capabilities.vision is True
    # Nothing in `codex exec` sets the pixel size, so promising sizes would be
    # a promise `choose_background`'s neighbours would act on.
    assert capabilities.sizes == []


def test_the_command_can_be_swapped_from_the_environment(monkeypatch):
    monkeypatch.delenv(COMMAND_ENV, raising=False)
    assert default_command() == list(DEFAULT_COMMAND)
    monkeypatch.setenv(COMMAND_ENV, "/opt/bin/codex --profile art")
    assert default_command() == ["/opt/bin/codex", "--profile", "art"]


def test_describing_a_picture_asks_read_only_and_leaves_no_session(tmp_path):
    runner = FakeCodex()
    answer = service(runner).describe(tmp_path / "cutout.png", "Name this.")
    args = runner.commands[0]
    assert "--ephemeral" in args
    assert args[args.index("-s") + 1] == "read-only"
    assert "workspace-write" not in args
    # The question is the positional and the picture follows it, for the same
    # reason `build_generate_args` puts the brief first: `-i` is multi-value.
    question = next(part for part in args if "Name this." in part)
    assert args.index(question) < args.index("-i")
    assert args[-2:] == ["-i", str((tmp_path / "cutout.png").resolve())]
    assert answer == "DONE"


def test_a_question_with_no_picture_attaches_nothing(tmp_path):
    """`suggest_direction_name` asks about no image at all."""
    runner = FakeCodex()
    answer = service(runner).describe(None, "Invent a name.")
    args = runner.commands[0]
    assert "-i" not in args
    assert "Invent a name." in args[-1]
    assert answer == "DONE"


#: Set this to run the one test in this file that actually starts Codex.
LIVE_ENV = "PIXELMAGIC_LIVE_CODEX"


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get(LIVE_ENV),
    reason=f"set {LIVE_ENV}=1 to draw one real picture with Codex",
)
def test_codex_really_draws_a_transparent_picture(tmp_path):
    """Opt-in, and the only test here that spends anything.

    It is the gate on three things no fake can check: that `codex exec` still
    reaches the built-in image tool, that a transparent background really
    comes back with an alpha channel, and that the file turns up where the
    contract says it does. It draws one small picture.
    """
    out = tmp_path / "image.png"
    service = CodexImageService(timeout=900)
    result = service.generate(
        GenerationRequest(
            id="live", created=NOW, subject="a red circle", kind=PromptKind.PROP,
            prompt=(
                "A single flat red circle, nothing else in the picture. The "
                "background must be fully transparent — save a PNG with a real "
                "alpha channel, not a white or checkerboard background."
            ),
            service="codex", size=(1024, 1024),
            background=BackgroundPolicy.TRANSPARENT,
        ),
        out,
    )
    assert out.exists()
    assert result.session_id
    with Image.open(out) as image:
        assert image.mode == "RGBA"
        # A real alpha channel, not an opaque one that merely has the field.
        assert image.getchannel("A").getextrema()[0] == 0


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get(LIVE_ENV),
    reason=f"set {LIVE_ENV}=1 to draw one real picture with Codex",
)
def test_codex_really_draws_with_a_reference_attached(tmp_path):
    """Opt-in, and the shape the pipeline actually sends.

    After the first approved generation every request in a direction carries
    at least one reference, because `promote` adds one each time. So a live
    test that only ever sends a prompt proves the *first* call and nothing
    after it — which is exactly how an argument-order bug in `-i` could sit
    unnoticed. This sends a prompt and a reference together, the way `generate`
    does, and asserts the brief still arrived by requiring the drawing to obey
    it.
    """
    reference = tmp_path / "swatch.png"
    Image.new("RGBA", (64, 64), (0, 0, 255, 255)).save(reference)
    out = tmp_path / "image.png"
    CodexImageService(timeout=900).generate(
        GenerationRequest(
            id="live-ref", created=NOW, subject="a red circle", kind=PromptKind.PROP,
            prompt=(
                "A single flat red circle, nothing else in the picture. Ignore "
                "the attached swatch except as a note on saturation. The "
                "background must be fully transparent — save a PNG with a real "
                "alpha channel."
            ),
            service="codex", size=(1024, 1024),
            background=BackgroundPolicy.TRANSPARENT,
        ),
        out,
        reference_paths=[reference],
    )
    assert out.exists()
    with Image.open(out) as image:
        assert image.mode == "RGBA"
        assert image.getchannel("A").getextrema()[0] == 0


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get(LIVE_ENV),
    reason=f"set {LIVE_ENV}=1 to ask Codex one real question",
)
def test_codex_really_sees_an_attached_image(tmp_path):
    """Opt-in: the gate on `-i` actually reaching the model's eyes.

    `codex exec --help` documents the flag; nothing proved the model can see
    what it attaches. The picture is a solid magenta square and the question
    asks only for its colour, so a model that cannot see it cannot guess.
    """
    picture = tmp_path / "swatch.png"
    Image.new("RGBA", (64, 64), (255, 0, 255, 255)).save(picture)
    answer = CodexImageService(timeout=300).describe(
        picture, "What colour is this square? Reply with one word."
    )
    assert "magenta" in answer.lower() or "pink" in answer.lower()
