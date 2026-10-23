"""What settings the refiner accepts, for the screen that offers them.

Ticket M2.1 binds this to the real tool: `StubRefiner.options()` answers out of
the contract's own constants — every preset, no palettes at all, and the quick
keys filed under `advanced` — so a settings screen built from it would offer
keys the real tool does not have and hide the palettes it does. When the real
tool cannot be reached, the screen says so; it never falls back to the
stand-in's answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pixelmagic.refiner.base import RefinerError, RefinerOptions


class OptionsSource(Protocol):
    """Anything that can say what settings the refiner accepts."""

    def options(self) -> RefinerOptions:
        """List the presets, palettes, quick keys and advanced keys."""
        ...


@dataclass
class OptionsView:
    """The settings vocabulary, or the reason there is none to show."""

    options: RefinerOptions | None
    problem: str = ""
    #: True when the real PixelRefiner is not the refiner doing the work, so
    #: nothing was asked of it. The page says how to turn it on.
    turned_off: bool = False


def read_options(source: OptionsSource) -> OptionsView:
    """Ask the refiner what it accepts, or say plainly why we could not.

    `RefinerError` is what the adapter raises for a missing binary, a timeout
    or a reply it cannot read; `OSError` covers a command that cannot be
    spawned at all. Neither is a reason to show a wrong list, so both become
    one sentence and no list.
    """
    try:
        return OptionsView(options=source.options())
    except (RefinerError, OSError) as error:
        return OptionsView(options=None, problem=str(error))
