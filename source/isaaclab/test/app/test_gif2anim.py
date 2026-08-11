# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the greeting authoring tool. These do not launch the simulator."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL", reason="gif2anim is a build-time tool and needs Pillow")

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools import gif2anim  # noqa: E402

pytestmark = pytest.mark.unit


def inked(row: str) -> list[bool]:
    """Which cells of a rendered row carry ink.

    Measured by colour, not by glyph: a uniform block encodes as a space with a background
    colour, which fills the cell. Counting non-space characters would read it as empty.
    """
    import re

    cells = []
    for match in re.finditer(r"((?:\x1b\[[0-9;]*m)*)(.)", row):
        style, glyph = match.group(1), match.group(2)
        if "38;2;" in style or "48;2;" in style:
            cells.append(True)
        elif glyph == " " and "\x1b[0m" in style:
            cells.append(False)
        elif cells:
            cells.append(cells[-1])  # style unchanged, so ink state carries over
    return cells


def ink_width(row: str) -> int:
    """Number of cells from the first inked one to the last."""
    cells = inked(row)
    if not any(cells):
        return 0
    return len(cells) - cells[::-1].index(True) - cells.index(True)


@pytest.fixture
def png(tmp_path):
    """A factory for small opaque test images."""
    from PIL import Image

    def make(width: int, height: int, colour=(118, 185, 0, 255)) -> Path:
        path = tmp_path / f"{width}x{height}.png"
        Image.new("RGBA", (width, height), colour).save(path)
        return path

    return make


def test_every_loading_screen_slot_has_a_name_on_the_command_line():
    from isaaclab.app import anims

    # SIZES takes its geometry from SLOTS, so what can still go wrong is a slot being added
    # to the runtime with no way to ask the converter for it
    assert sorted(gif2anim.SIZES.values()) == sorted(anims.SLOTS)


@pytest.mark.parametrize("size", sorted(gif2anim.SIZES))
def test_frames_have_the_declared_size(png, size):
    cols, rows = gif2anim.SIZES[size]
    animation = gif2anim.convert(png(128, 64), size, name="probe")
    assert (animation.cols, animation.rows) == (cols, rows)
    for frame in animation.frames:
        assert len(frame.splitlines()) == rows


def test_a_still_image_yields_one_frame(png):
    assert len(gif2anim.convert(png(48, 24), "small", name="still").frames) == 1


def test_a_fully_transparent_source_paints_no_background(png):
    animation = gif2anim.convert(png(48, 24, (0, 0, 0, 0)), "small", name="empty")
    # a background would draw an opaque box around art that should composite over the terminal
    assert "48;2;" not in "".join(animation.frames)


def test_an_opaque_source_uses_two_colours_per_cell(png):
    from PIL import Image

    path = png(64, 32)
    image = Image.open(path).convert("RGBA")
    for x in range(image.width):  # a vertical split, so cells straddle two colours
        for y in range(image.height):
            image.putpixel((x, y), (240, 176, 42, 255) if x < image.width // 2 else (26, 26, 32, 255))
    image.save(path)
    animation = gif2anim.convert(path, "wide", name="split")
    assert "48;2;" in animation.frames[0]


def test_a_gif_keeps_its_frames(tmp_path):
    from PIL import Image

    path = tmp_path / "spin.gif"
    frames = [Image.new("RGBA", (48, 24), (c, 100, 0, 255)) for c in (0, 80, 160, 240)]
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0)
    assert len(gif2anim.convert(path, "small", name="spin").frames) == 4


def test_frame_count_can_be_resampled(tmp_path):
    from PIL import Image

    path = tmp_path / "spin.gif"
    frames = [Image.new("RGBA", (48, 24), (c, 100, 0, 255)) for c in range(0, 240, 20)]
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0)
    assert len(gif2anim.convert(path, "small", name="spin", frames=4).frames) == 4


def test_from_module_packs_a_procedural_sprite():
    animation = gif2anim.from_module("tools.sprites.probe", "wide", name="probe")
    cols, rows = gif2anim.SIZES["wide"]
    assert (animation.cols, animation.rows) == (cols, rows)
    assert len(animation.frames) > 1


def test_from_module_rejects_a_module_outside_the_sprite_package():
    with pytest.raises(ValueError, match="tools.sprites"):
        gif2anim.from_module("os.path", "wide", name="nope")


def test_padding_keeps_the_source_aspect(png):
    # a square mark is as wide as it is tall; the slot is rows*2 units tall in display terms.
    # Landing near rows*2 rather than at cols is also what says it was padded, not stretched
    cols, rows = gif2anim.SIZES["wide"]
    animation = gif2anim.convert(png(64, 64), "wide", name="square")
    widest = max(ink_width(row) for row in animation.frames[0].splitlines())
    assert abs(widest - rows * 2) <= 2, f"square source drew {widest} cells wide, expected about {rows * 2}"


def test_a_matching_aspect_fills_the_grid(png):
    cols, rows = gif2anim.SIZES["wide"]
    # the grid displays cols across and rows*2 down, so a source of that shape should fill it
    animation = gif2anim.convert(png(cols * 8, rows * 2 * 8), "wide", name="fits")
    assert max(ink_width(row) for row in animation.frames[0].splitlines()) >= cols - 1


def test_stretch_fills_the_grid_regardless_of_aspect(png):
    cols, _ = gif2anim.SIZES["wide"]
    animation = gif2anim.convert(png(64, 64), "wide", name="square", fit=False)
    assert max(ink_width(row) for row in animation.frames[0].splitlines()) >= cols - 1


def test_repeated_colours_are_suppressed(png):
    animation = gif2anim.convert(png(128, 64), "wide", name="flat")
    frame = animation.frames[0]
    # a flat image needs one colour set-up, not one per cell
    assert frame.count("38;2;") < gif2anim.SIZES["wide"][0]
