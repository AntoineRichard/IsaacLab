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


@pytest.fixture
def png(tmp_path):
    """A factory for small opaque test images."""
    from PIL import Image

    def make(width: int, height: int, colour=(118, 185, 0, 255)) -> Path:
        path = tmp_path / f"{width}x{height}.png"
        Image.new("RGBA", (width, height), colour).save(path)
        return path

    return make


def test_sizes_match_the_loading_screen_slots():
    from isaaclab.app import anims

    assert set(gif2anim.SIZES.values()) == set(anims.SLOTS)


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


def test_repeated_colours_are_suppressed(png):
    animation = gif2anim.convert(png(128, 64), "wide", name="flat")
    frame = animation.frames[0]
    # a flat image needs one colour set-up, not one per cell
    assert frame.count("38;2;") < gif2anim.SIZES["wide"][0]
