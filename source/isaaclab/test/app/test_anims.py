# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the loading screen animation container. These do not launch the simulator."""

import gzip
import json
import random

import pytest

from isaaclab.app import anims

pytestmark = pytest.mark.unit

SLOTS = {(24, 12), (64, 12)}
"""The two greeting sizes the loading screen has room for."""


def test_round_trip_preserves_frames():
    original = anims.Animation("demo", 4, 2, ("ab\ncd", "ef\ngh"))
    assert anims.unpack(anims.pack(original)) == original


def test_round_trip_preserves_colour_escapes():
    # frames are pre-rendered terminal output; the container must not touch the escapes
    frame = "\x1b[38;2;1;2;3m\x1b[48;2;4;5;6m▀\x1b[0m"
    assert anims.unpack(anims.pack(anims.Animation("colour", 1, 1, (frame,)))).frames == (frame,)


def test_round_trip_preserves_a_single_frame():
    original = anims.Animation("static", 2, 1, ("ab",))
    assert anims.unpack(anims.pack(original)).frames == ("ab",)


def test_pack_is_deterministic():
    # gzip stamps a timestamp unless told not to, which would make every regeneration a diff
    animation = anims.Animation("demo", 2, 1, ("ab", "cd"))
    assert anims.pack(animation) == anims.pack(animation)


def test_pack_compresses():
    repetitive = anims.Animation("big", 8, 4, ("x" * 4000,) * 8)
    assert len(anims.pack(repetitive)) < 4000


def test_unpack_rejects_a_future_version():
    header, _, body = gzip.decompress(anims.pack(anims.Animation("demo", 1, 1, ("x",)))).partition(b"\n")
    meta = json.loads(header)
    meta["version"] = anims.VERSION + 1
    broken = gzip.compress(json.dumps(meta).encode() + b"\n" + body)
    with pytest.raises(ValueError, match="version"):
        anims.unpack(broken)


def test_unpack_rejects_a_frame_count_that_disagrees_with_the_header():
    header, _, body = gzip.decompress(anims.pack(anims.Animation("demo", 1, 1, ("x", "y")))).partition(b"\n")
    meta = json.loads(header)
    meta["frames"] = 99
    with pytest.raises(ValueError, match="frames"):
        anims.unpack(gzip.compress(json.dumps(meta).encode() + b"\n" + body))


def test_every_shipped_animation_loads_and_matches_its_header():
    names = anims.available()
    assert names, "no animations are shipped"
    for name in names:
        animation = anims.load(name)
        assert animation.frames, name
        for frame in animation.frames:
            assert len(frame.splitlines()) == animation.rows, name


def test_shipped_animations_use_a_known_slot_size():
    names = anims.available()
    # without this the loop body never runs when nothing is shipped, and the test passes
    # while checking nothing at all
    assert names, "no animations are shipped"
    for name in names:
        animation = anims.load(name)
        assert (animation.cols, animation.rows) in SLOTS, name


def test_choose_returns_one_animation_per_slot():
    small, wide = anims.choose(random.Random(0))
    assert (small.cols, small.rows) == (24, 12)
    assert (wide.cols, wide.rows) == (64, 12)


def test_choose_is_deterministic_for_a_seeded_generator():
    assert anims.choose(random.Random(7)) == anims.choose(random.Random(7))


def test_choose_varies_across_seeds():
    picks = {anims.choose(random.Random(seed))[1].name for seed in range(20)}
    assert len(picks) > 1, "every seed picked the same wide greeting"


def test_load_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        anims.load("no-such-greeting")


def test_load_is_cached():
    name = anims.available()[0]
    assert anims.load(name) is anims.load(name)
