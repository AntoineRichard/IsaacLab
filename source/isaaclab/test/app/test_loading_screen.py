# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the responsive console loading screen."""

import zlib
from io import StringIO
from unittest.mock import Mock

import pytest
from rich.cells import cell_len
from rich.console import Console, RenderableType

from isaaclab.app import anims, loading_screen

_SUMMARY_WIDTH = 50
"""The width the summary box is locked to; anything beyond it on a box row is the greeting."""

_COLUMN_GAP = 6
"""Columns between the summary box and the greeting.

Restated here rather than imported so the layout the screen promises is pinned by the test
instead of read back out of the code it is checking.
"""

_ALT_SCREEN_ON = "\x1b[?1049h"
_ALT_SCREEN_OFF = "\x1b[?1049l"


def _render(display: RenderableType, width: int) -> str:
    stream = StringIO()
    Console(file=stream, width=width, color_system=None).print(display)
    return stream.getvalue()


def _pin_greetings(monkeypatch) -> None:
    """Replace the random pick with one solid block per slot.

    Layout tests must not depend on which art was drawn. Real greetings carry their shape in
    colour, and these render without it, so a greeting whose cells are all background collapses
    to spaces that the console then strips -- making the measured width depend on the draw.
    """
    pinned = tuple(
        anims.Animation(f"pinned-{cols}", cols, rows, ("\n".join(["█" * cols] * rows),)) for cols, rows in anims.SLOTS
    )
    monkeypatch.setattr(anims, "choose", lambda *a, **k: pinned)


@pytest.mark.parametrize(
    ("terminal_width", "display_width", "greeting_cols"),
    [
        (4, 4, None),
        (5, 5, None),
        (79, 79, None),
        (80, 80, 24),
        (119, 80, 24),
        (120, 120, 64),
        (140, 120, 64),
    ],
)
def test_display_reflows_at_responsive_boundaries(
    monkeypatch, terminal_width: int, display_width: int, greeting_cols: int | None
):
    _pin_greetings(monkeypatch)
    screen = loading_screen.LoadingScreen(2, enabled=True)
    screen.summary("Run", {"Task": "Cartpole", "Description": "A long description that wraps cleanly."})
    screen.stage("Loading task")

    output = _render(screen._display, terminal_width)
    lines = output.splitlines()

    assert max(map(cell_len, lines)) <= display_width
    assert cell_len(lines[-1]) == display_width
    # measures how many columns the greeting took beside the box. Picking the wide art for the
    # narrow slot is the failure this guards, and it shows up here as the wrong count.
    box_row = next(line for line in lines if line.startswith("╭"))
    if greeting_cols is None:
        assert cell_len(box_row) == min(_SUMMARY_WIDTH, display_width)
    else:
        assert cell_len(box_row) - _SUMMARY_WIDTH - _COLUMN_GAP == greeting_cols
    if terminal_width >= 79:
        assert "Loading task" in lines[-1]
        assert "cleanly." in output


def test_live_updates_refresh_after_releasing_state_lock():
    screen = loading_screen.LoadingScreen(2, enabled=True)
    lock_states: list[bool] = []

    class _RefreshProbe:
        def refresh(self) -> None:
            lock_states.append(screen._render_lock._is_owned())

    probe = _RefreshProbe()
    screen._live = probe

    screen.summary("Run", {"Task": "Cartpole"})
    screen.stage("Loading task")
    screen.set_activity("Loading assets")

    assert lock_states == [False, False, False]


class _OpenStringIO(StringIO):
    def close(self) -> None:
        pass


def _prepare_redirect(screen: loading_screen.LoadingScreen, monkeypatch: pytest.MonkeyPatch) -> _OpenStringIO:
    output = _OpenStringIO()
    monkeypatch.setenv("TERM", "xterm-256color")

    def redirect() -> None:
        screen._console = output
        screen._saved_fds = (-1, -1)

    def restore() -> str:
        screen._saved_fds = None
        return "hidden diagnostic\n"

    monkeypatch.setattr(screen, "_redirect", redirect)
    monkeypatch.setattr(screen, "_restore", restore)
    return output


def test_live_lifecycle_restores_normal_screen(monkeypatch: pytest.MonkeyPatch):
    screen = loading_screen.LoadingScreen(2, enabled=True)
    output = _prepare_redirect(screen, monkeypatch)

    with screen:
        screen.summary("Run", {"Task": "Cartpole"})
        screen.stage("Loading task")
        assert screen._rich_console is not None
        assert screen._rich_console.color_system == "truecolor"
        screen.close()

    rendered = output.getvalue()
    normal_screen = rendered.index(_ALT_SCREEN_OFF)

    assert _ALT_SCREEN_ON in rendered
    assert "Loading task" in rendered[:normal_screen]
    assert "%" in rendered[:normal_screen]
    assert rendered.rindex("Run") > normal_screen
    assert "hidden diagnostic" not in rendered
    assert "Ready in" in rendered


def test_shutdown_restores_output_when_live_rendering_fails(monkeypatch: pytest.MonkeyPatch):
    screen = loading_screen.LoadingScreen(1, enabled=True)
    screen._live = Mock()
    screen._live.stop.side_effect = RuntimeError("render failed")
    screen._saved_fds = (-1, -1)
    screen._console = Mock()
    restore = Mock(return_value="")
    monkeypatch.setattr(screen, "_restore", restore)

    with pytest.raises(RuntimeError, match="render failed"):
        screen.close()

    restore.assert_called_once_with()


def test_logos_are_empty_when_the_greeting_is_disabled():
    assert loading_screen.LoadingScreen(1, enabled=False, logo=False)._logos == ()


def test_logos_offer_one_greeting_per_slot():
    screen = loading_screen.LoadingScreen(1, enabled=False)
    assert len(screen._logos) == len(anims.SLOTS)


def test_logos_advance_with_time(monkeypatch):
    # pin an animated greeting: the pick is random, and a run that happened to draw two still
    # ones would pass or fail by chance
    animated = next(n for n in anims.available() if len(anims.load(n).frames) > 1)
    monkeypatch.setenv("ISAACLAB_LOADING_ANIM", animated)
    screen = loading_screen.LoadingScreen(1, enabled=False)
    screen._started = 0.0
    monkeypatch.setattr(loading_screen.time, "monotonic", lambda: 0.0)
    first = screen._logos
    monkeypatch.setattr(loading_screen.time, "monotonic", lambda: 5.0)
    assert screen._logos != first


def test_environment_variable_pins_the_greeting(monkeypatch):
    monkeypatch.setenv("ISAACLAB_LOADING_ANIM", "hello-small")
    picked = {loading_screen.LoadingScreen(1, enabled=False)._animations[0].name for _ in range(8)}
    assert picked == {"hello-small"}


def test_environment_variable_can_disable_the_greeting(monkeypatch):
    monkeypatch.setenv("ISAACLAB_LOADING_ANIM", "none")
    assert loading_screen.LoadingScreen(1, enabled=False)._logos == ()


def test_an_unknown_name_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv("ISAACLAB_LOADING_ANIM", "no-such-greeting")
    # a decorative feature must never be able to stop a run from starting
    assert loading_screen.LoadingScreen(1, enabled=False)._logos


def test_unreadable_greetings_fall_back_to_the_builtin_logos(monkeypatch):
    monkeypatch.setattr(anims, "choose", lambda *a, **k: (_ for _ in ()).throw(LookupError("boom")))
    assert loading_screen.LoadingScreen(1, enabled=False)._logos == (loading_screen.LOGO, loading_screen.LOGO_WIDE)


@pytest.mark.parametrize("error", [zlib.error("invalid block type"), EOFError("ended before end-of-stream")])
def test_a_damaged_container_costs_the_greeting_not_the_launch(monkeypatch, error):
    # what a half-written or truncated container raises out of gzip. Neither descends from
    # OSError or ValueError, so listing those two let a bad file abort startup outright
    monkeypatch.setattr(anims, "choose", lambda *a, **k: (_ for _ in ()).throw(error))
    assert loading_screen.LoadingScreen(1, enabled=False)._logos == (loading_screen.LOGO, loading_screen.LOGO_WIDE)
