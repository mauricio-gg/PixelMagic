"""The folder tree and its JSON sidecars.

PixelMagic has no database (spec locked decision 8). This module is the only
place that knows where things live, so a stage never builds a path by hand and
re-running a stage is just overwriting a file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from pixelmagic.model import Direction, Project
from pixelmagic.naming import SAFE_NAME, UnusableName, check_name, kebab_case

ModelT = TypeVar("ModelT", bound=BaseModel)

PROJECT_FILE = "pixelmagic.json"
DIRECTION_FILE = "direction.json"
SCENE_CHOICE_FILE = "scene-choice.json"
ID_FORMAT = "%Y%m%d-%H%M%S"


def check_run_id(value: str, what: str) -> str:
    """Return `value` if it is safe to use as a run id, or refuse it.

    The accept rule is `naming.SAFE_NAME`'s, the same as `check_name`, so what
    opens is unchanged. Only the sentence differs: an id is a date and time
    (`ID_FORMAT`, as `new_id` mints it), so the refusal says that rather than
    describing a kebab-case name.
    """
    if not SAFE_NAME.fullmatch(value):
        raise UnusableName(
            f"{what} {value!r} is not an id PixelMagic uses — an id is a date and "
            f"time like 20260903-191233, sometimes followed by -2 or -3"
        )
    return value


class AlreadyExists(FileExistsError):
    """Creating this would overwrite something already on disk.

    A `FileExistsError` on purpose: a face that catches the OSError family at
    its boundary catches this too, without importing anything from the store.
    """


def new_id(parent: Path, *, now: datetime | None = None) -> str:
    """Mint a timestamp-slug id that is free inside `parent`.

    Two generations within the same second would otherwise collide, so a
    `-2`, `-3` … suffix is appended until the folder name is unused.
    """
    stamp = (now or datetime.now(UTC)).strftime(ID_FORMAT)
    if not (parent / stamp).exists():
        return stamp
    counter = 2
    while (parent / f"{stamp}-{counter}").exists():
        counter += 1
    return f"{stamp}-{counter}"


def claim_id(parent: Path, *, now: datetime | None = None) -> str:
    """Mint an id like `new_id` and create its folder in one step.

    `new_id` only checks that the folder is absent, so two runs in the same
    second could both pass the check and write into one folder. Creating the
    folder without `exist_ok` is atomic: the loser gets `FileExistsError` and
    moves to the next free id (`-2`, `-3` ...).
    """
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = new_id(parent, now=now)
        try:
            (parent / candidate).mkdir()
        except FileExistsError:
            continue
        return candidate


class Store:
    """Reads and writes one project folder."""

    def __init__(self, root: Path) -> None:
        """Bind to the folder that holds (or will hold) `pixelmagic.json`."""
        self.root = Path(root)

    # --- paths -------------------------------------------------------------

    def project_file(self) -> Path:
        """The project sidecar."""
        return self.root / PROJECT_FILE

    def direction_dir(self, direction: str) -> Path:
        """The folder for one art direction.

        `direction` is already a safe folder name when it gets here: names are
        squeezed into kebab-case at the boundary where free text arrives — the
        CLI's `--name`, or a service's answer — by `naming.kebab_case`. The
        store does not sanitise a direction, so that there is exactly one place
        a name can change shape and a folder always matches the name in the
        sidecar. (Run ids are different: they arrive from outside as they are,
        so `generation_dir` and `decomposition_dir` check them.)
        """
        return self.root / "directions" / direction

    def direction_file(self, direction: str) -> Path:
        """The direction sidecar."""
        return self.direction_dir(direction) / DIRECTION_FILE

    def scene_choice_file(self, direction: str) -> Path:
        """The sidecar naming the scene Refine and Export follow."""
        return self.direction_dir(direction) / SCENE_CHOICE_FILE

    def references_dir(self, direction: str) -> Path:
        """Promoted reference images for a direction."""
        return self.direction_dir(direction) / "references"

    def generations_dir(self, direction: str) -> Path:
        """All generations for a direction."""
        return self.direction_dir(direction) / "generations"

    def generation_dir(self, direction: str, generation_id: str) -> Path:
        """One generation's folder.

        `generation_id` is checked with `SAFE_NAME` (`UnusableName`, worded as
        an id, when unsafe): the command line hands an id over as typed, and
        this is where it becomes a path.
        """
        check_run_id(generation_id, "generation id")
        return self.generations_dir(direction) / generation_id

    def cleanup_dir(self, direction: str, generation_id: str) -> Path:
        """One picture's clean-up folder (`generation_id` is checked like `generation_dir`'s)."""
        check_run_id(generation_id, "generation id")
        return self.direction_dir(direction) / "cleanups" / generation_id

    def decompositions_dir(self, direction: str) -> Path:
        """All decompositions for a direction."""
        return self.direction_dir(direction) / "decompositions"

    def decomposition_dir(self, direction: str, decomposition_id: str) -> Path:
        """One decomposition's folder.

        `decomposition_id` is checked the same way as in `generation_dir`.
        """
        check_run_id(decomposition_id, "cut-up run id")
        return self.decompositions_dir(direction) / decomposition_id

    def surface_dir(self, direction: str, decomposition_id: str, surface_name: str) -> Path:
        """Where one recipe surface's outline lives, under its decomposition.

        Keyed by the surface's own name (`Surface.name`). It holds only the
        outline a person can look at.
        """
        return self.decomposition_dir(direction, decomposition_id) / "surfaces" / surface_name

    def cutout_dir(self, direction: str, decomposition_id: str, asset: str) -> Path:
        """Where one cutout of a decomposition lives."""
        return self.decomposition_dir(direction, decomposition_id) / "cutouts" / asset

    def suggestions_dir(self, direction: str, decomposition_id: str) -> Path:
        """Where a decomposition's suggested object masks and contact sheets live."""
        return self.decomposition_dir(direction, decomposition_id) / "suggestions"

    def picks_dir(self, direction: str, decomposition_id: str) -> Path:
        """Where a decomposition's picks and their masks live."""
        return self.decomposition_dir(direction, decomposition_id) / "picks"

    def groups_dir(self, direction: str, decomposition_id: str) -> Path:
        """Where a decomposition's group redraws live, one folder each."""
        return self.decomposition_dir(direction, decomposition_id) / "groups"

    def group_dir(self, direction: str, decomposition_id: str, group_id: str) -> Path:
        """Where one group redraw of a decomposition lives."""
        return self.groups_dir(direction, decomposition_id) / group_id

    def assets_dir(self, direction: str) -> Path:
        """All assets promoted from cutouts."""
        return self.direction_dir(direction) / "assets"

    def asset_dir(self, direction: str, name: str) -> Path:
        """One asset's folder.

        Like `direction_dir`, `name` is expected to be safe already: cutout
        labels pass through `naming.kebab_case` and `naming.unique_name` when
        they are promoted, not here.
        """
        return self.assets_dir(direction) / name

    def surface_master(self, direction: str, surface: str) -> Path:
        """A recipe surface's master `.aseprite`, beside its tile folders.

        `list_assets` only lists folders holding a `source.png`, so this file
        is invisible to it.
        """
        return self.assets_dir(direction) / f"{surface}.aseprite"

    def export_dir(self, stamp: str) -> Path:
        """One export bundle, named `<direction>-<stamp>` by the caller."""
        return self.root / "exports" / stamp

    # --- json --------------------------------------------------------------

    def read_model(self, path: Path, model_type: type[ModelT]) -> ModelT:
        """Load one sidecar into its model."""
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))

    def write_model(
        self, path: Path, model: BaseModel, *, exclude: Mapping[str, Any] | None = None
    ) -> Path:
        """Write one model as a pretty JSON sidecar, creating parents.

        `exclude` is Pydantic's own exclusion spec: fields (nested ones too) the
        model still has but the file should not carry.

        The text goes to a temporary file beside the target and is then
        renamed over it. `os.replace` is atomic within one filesystem, so a
        crash or a full disk halfway through leaves either the old sidecar or
        the new one — never half of either. The sidecars are the only memory
        this tool has (locked decision 8: no database), and a truncated one
        loses a whole stage's record.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            model.model_dump(mode="json", exclude=exclude), indent=2, ensure_ascii=False
        )
        # The pid keeps two processes writing the same sidecar off each
        # other's scratch file; the rename below is what makes either one safe.
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(text + "\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            # A successful replace already moved it; this only bites on a
            # failed write, which must not leave litter behind.
            temporary.unlink(missing_ok=True)
        return path

    # --- project and direction ---------------------------------------------

    def create_project(self, name: str, *, now: datetime | None = None) -> Project:
        """Create the project folder and its sidecar.

        Refuses to overwrite an existing `pixelmagic.json`: rewriting it would
        throw away the project's name and its default direction while every
        direction folder underneath stayed exactly where it was. The guard
        lives here and not in a face because all three faces create projects,
        and a guard each is how they end up disagreeing — the same argument
        `stages.refine.set_refiner_defaults` makes for its own check.
        """
        if self.project_file().exists():
            raise AlreadyExists(f"a project already exists in {self.root}")
        project = Project(name=name, created=now or datetime.now(UTC))
        self.write_model(self.project_file(), project)
        return project

    def load_project(self) -> Project:
        """Load the project sidecar."""
        return self.read_model(self.project_file(), Project)

    def save_project(self, project: Project) -> Project:
        """Overwrite the project sidecar."""
        self.write_model(self.project_file(), project)
        return project

    def create_direction(self, direction: Direction) -> Direction:
        """Create a direction's folders and sidecar.

        Refuses an existing direction for the same reason `create_project`
        does: rewriting `direction.json` would drop the style paragraph, the
        don'ts, the references and the refiner defaults it already holds.
        """
        if self.direction_file(direction.name).exists():
            raise AlreadyExists(
                f"direction {direction.name!r} already exists in {self.root}"
            )
        for folder in (
            self.references_dir(direction.name),
            self.generations_dir(direction.name),
            self.decompositions_dir(direction.name),
            self.assets_dir(direction.name),
        ):
            folder.mkdir(parents=True, exist_ok=True)
        self.write_model(self.direction_file(direction.name), direction)
        return direction

    def load_direction(self, name: str) -> Direction:
        """Load a direction sidecar; raises FileNotFoundError when absent."""
        path = self.direction_file(name)
        if not path.exists():
            raise FileNotFoundError(f"no direction named {name!r} under {self.root}")
        return self.read_model(path, Direction)

    def save_direction(self, direction: Direction) -> Direction:
        """Overwrite a direction sidecar."""
        self.write_model(self.direction_file(direction.name), direction)
        return direction

    def rename_direction(self, old: str, new: str) -> Direction:
        """Move a direction's folder and rewrite its sidecar under the new name.

        Spec §1 promises renaming and M1 deferred it (plan deviation 2)
        because no face could reach it. `new` must already be a safe folder
        name: the store does not sanitise directions (see `direction_dir`), so a face
        squeezes free text through `naming.kebab_case` first, and a name that
        would not survive that trip — including one with nothing ASCII left in
        it at all — is refused here rather than becoming a folder nobody can
        find again.

        The shape of a usable name is `naming.check_name`'s and not a second
        opinion spelled here: two whitelists that disagree is how a face lets
        `Warm-Den` through and the store then refuses it. `kebab_case` stays as
        the *word count and length* rule on top of that one shape. `old` goes
        through the same check, because it too becomes a path here.

        Nothing moves until the sidecar has been read and validated. A
        `direction.json` that will not parse would otherwise raise after the
        folder had already moved, leaving the direction under the new name with
        a sidecar naming the old one.

        The project's `default_direction` is the only thing outside the folder
        that names a direction, so it follows the rename. References,
        generations, decompositions and assets all live inside the folder and
        travel with it. An export bundle already on disk keeps the old name in
        its folder name and its manifest: a bundle is a record of what was
        exported on a day, not a live link, and rewriting it would falsify it.
        """
        # Both arguments, not just the new one: `old` becomes a path in
        # `direction_file(old)` two lines down, so a caller that passes
        # `"../escape"` should be refused by name rather than told the
        # direction does not exist.
        check_name(old, "direction")
        check_name(new, "direction")
        if new != kebab_case(new, fallback=""):
            raise UnusableName(
                f"{new!r} is not a usable direction name — "
                f"lowercase words joined by hyphens, at most four"
            )
        if not self.direction_file(old).exists():
            raise FileNotFoundError(f"no direction named {old!r} under {self.root}")
        if self.direction_dir(new).exists():
            raise AlreadyExists(f"direction {new!r} already exists in {self.root}")

        # Everything that can be *checked* has been, above — but the sidecar
        # write and the project repoint below still run after the rename, and
        # a failure in either leaves the folder under the new name with a
        # sidecar naming the old one. Nothing outside the sidecars records
        # which name is current, so that window cannot be closed here.
        direction = self.read_model(self.direction_file(old), Direction)
        direction.name = new
        self.direction_dir(old).rename(self.direction_dir(new))
        self.write_model(self.direction_file(new), direction)

        # Checked rather than assumed: the folder has already moved by now, so
        # a project sidecar that is not there would leave the rename half done
        # and raise about the wrong thing.
        if self.project_file().exists():
            project = self.load_project()
            if project.default_direction == old:
                project.default_direction = new
                self.save_project(project)
        return direction

    def list_generations(self, direction: str) -> list[str]:
        """Generation ids in this direction, newest first.

        Ids are timestamp slugs, so sorting them backwards sorts them by time
        backwards, and a face showing "what have I drawn lately" wants the
        last picture at the top.
        """
        return self._ids(self.generations_dir(direction), "request.json")

    def list_decompositions(self, direction: str) -> list[str]:
        """Decomposition ids in this direction, newest first."""
        return self._ids(self.decompositions_dir(direction), "decomposition.json")

    def _ids(self, root: Path, marker: str) -> list[str]:
        """Folder names under `root` that hold `marker`, newest first.

        The marker is what makes a folder a record: `generate` creates the
        folder before it writes anything into it, so a run interrupted between
        the two would otherwise show up in a list as a generation with no
        request.
        """
        if not root.is_dir():
            return []
        return sorted(
            (child.name for child in root.iterdir() if (child / marker).exists()),
            reverse=True,
        )

    def list_directions(self) -> list[str]:
        """Direction names, alphabetically."""
        root = self.root / "directions"
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if (p / DIRECTION_FILE).exists())

    def list_assets(self, direction: str) -> list[str]:
        """Names of every asset with a `source.png`, alphabetically.

        This lives on the store rather than in a stage because three stages
        need it — generate and decompose to avoid re-using a name, refine and
        export to know what to work on — and none of them should have to
        import another stage to ask a question about folders.
        """
        root = self.assets_dir(direction)
        if not root.is_dir():
            return []
        return sorted(folder.name for folder in root.iterdir() if (folder / "source.png").exists())
