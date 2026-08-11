# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the greeting viewer. These do not launch the simulator."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from isaaclab.app import anims  # noqa: E402
from tools import anim_view  # noqa: E402

pytestmark = pytest.mark.unit


def plain(text: str) -> str:
    """Text with its colour escapes removed, for measuring."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_resolve_finds_a_shipped_greeting():
    name = anims.available()[0]
    assert anim_view.resolve(name).name == name


def test_resolve_reads_a_container_from_a_path(tmp_path):
    original = anims.Animation("scratch", 2, 1, ("ab",))
    path = tmp_path / "scratch.anim"
    path.write_bytes(anims.pack(original))
    assert anim_view.resolve(str(path)) == original


def test_resolve_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        anim_view.resolve("no-such-greeting")


def test_beside_summary_keeps_the_box_square():
    animation = anims.load(anims.available()[0])
    laid_out = anim_view.beside_summary(animation, animation.frames[0])
    # every box line must be the same width, or the border visibly steps in and out
    widths = {len(plain(line)[:50].rstrip()) for line in laid_out.splitlines() if plain(line).startswith(("╭", "│", "╰"))}
    assert widths == {50}


def test_beside_summary_covers_every_frame_row():
    animation = anims.load(anims.available()[0])
    laid_out = anim_view.beside_summary(animation, animation.frames[0])
    assert len(laid_out.splitlines()) >= animation.rows


def test_summarise_reports_every_shipped_greeting(capsys):
    assert anim_view.summarise() == 0
    printed = capsys.readouterr().out
    for name in anims.available():
        assert name in printed


def test_a_still_greeting_prints_instead_of_playing(capsys, tmp_path):
    path = tmp_path / "still.anim"
    path.write_bytes(anims.pack(anims.Animation("still", 2, 1, ("ab",))))
    # a one-frame greeting must not enter the playback loop, which never returns on its own
    assert anim_view.main([str(path)]) == 0
    assert "ab" in capsys.readouterr().out


def test_frame_index_wraps():
    name = anims.available()[0]
    animation = anims.load(name)
    assert anim_view.main([name, "--frame", str(len(animation.frames) + 1)]) == 0
