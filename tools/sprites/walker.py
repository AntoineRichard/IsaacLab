# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A quadruped robot trotting in place.

The chassis holds still while the ground scrolls beneath it at exactly the speed its planted
feet travel, so the gait reads as walking rather than sliding. The head is the raised front of
one continuous mass, not a separate box on a neck -- that keeps the silhouette whole and
spends the length on the body, where it reads.

Each slot gets its own :class:`Layout`. The two are laid out separately rather than scaled:
pixel art does not survive resampling, and the wide slot is not simply a bigger version of the
narrow one -- it has room for detail the narrow one has to drop.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from .canvas import Canvas

CREAM = (232, 228, 214)
CREAM_SHADE = (176, 172, 158)
SHELL_HI = (255, 253, 244)
DARK = (48, 48, 56)
DARKER = (26, 26, 32)
AMBER = (240, 176, 42)
AMBER_HI = (255, 214, 110)
GREY = (158, 160, 165)
FAR_THIGH = (38, 39, 45)
FAR_SHIN = (60, 62, 68)
FAR_JOINT = (104, 71, 14)
CYAN = (77, 217, 232)
EYE_CORE = (220, 255, 255)
GROUND = (56, 56, 62)
GRASS = (96, 158, 68)

OFFSETS = {"ff": 0.0, "rn": 0.0, "fn": 0.5, "rf": 0.5}
"""Trot: diagonal pairs move together, so from the side each near leg opposes its neighbour."""

DUTY = 0.5
"""Share of the cycle a foot spends planted."""

FRAMES = 24
"""Frames in one gait cycle."""


class Layout(NamedTuple):
    """Geometry for one slot, in subpixels."""

    ground_y: int
    head_x0: int
    head_x1: int
    head_y0: int
    body_x1: int
    body_y0: int
    body_y1: int
    face_inset: int
    eye_x0: int
    eye_x1: int
    eye_y: int
    eye_w: int
    band_x0: int
    band_x1: int
    front_hip: int
    rear_hip: int
    hip_y: int
    thigh: float
    stride: float
    step_h: float
    bob: int
    ground_period: int
    detail: bool


LAYOUTS = {
    # 64 x 12 cells -> 128 x 24 subpixels. Room for the vent band, status lights and grass.
    (64, 12): Layout(
        ground_y=21,
        head_x0=6,
        head_x1=34,
        head_y0=4,
        body_x1=104,
        body_y0=7,
        body_y1=17,
        face_inset=3,
        eye_x0=12,
        eye_x1=24,
        eye_y=8,
        eye_w=5,
        band_x0=44,
        band_x1=72,
        front_hip=46,
        rear_hip=90,
        hip_y=17,
        thigh=5.4,
        stride=8.0,
        step_h=3.0,
        bob=2,
        ground_period=4,
        detail=True,
    ),
    # 24 x 12 cells -> 48 x 24 subpixels. Half the width: the band and lights are dropped and
    # only the silhouette, the eyes and the gait survive.
    (24, 12): Layout(
        ground_y=21,
        head_x0=2,
        head_x1=16,
        head_y0=5,
        body_x1=40,
        body_y0=8,
        body_y1=17,
        face_inset=2,
        eye_x0=5,
        eye_x1=10,
        eye_y=9,
        eye_w=3,
        band_x0=18,
        band_x1=30,
        front_hip=19,
        rear_hip=34,
        hip_y=17,
        thigh=4.4,
        stride=4.0,
        step_h=2.2,
        bob=1,
        ground_period=2,
        detail=False,
    ),
}


def foot(phase: float, leg: str, layout: Layout) -> tuple[float, float]:
    """Foot offset from its hip and height above the ground, in subpixels.

    The robot faces left, so a planted foot travels right -- with the ground, against the
    direction of travel.
    """
    local = (phase + OFFSETS[leg]) % 1.0
    if local < DUTY:
        return -layout.stride / 2 + layout.stride * (local / DUTY), 0.0
    swung = (local - DUTY) / (1 - DUTY)
    return layout.stride / 2 - layout.stride * swung, layout.step_h * math.sin(math.pi * swung)


def knee(hx: float, hy: float, fx: float, fy: float, thigh: float, forward: bool) -> tuple[float, float]:
    """Two-link inverse kinematics: the knee between a hip and a foot."""
    dx, dy = fx - hx, fy - hy
    span = math.hypot(dx, dy) or 1e-9
    reach = min(span, thigh * 2)
    along = reach / 2
    across = math.sqrt(max(thigh**2 - along**2, 0.0))
    ux, uy = dx / span, dy / span
    s = 1.0 if forward else -1.0
    return hx + ux * along - s * uy * across, hy + uy * along + s * ux * across


def _leg(c: Canvas, hip_x: int, hip_y: float, phase: float, tag: str, layout: Layout, near: bool) -> None:
    """One leg, dimmed on the far side so the pair reads as depth."""
    thigh_col = DARK if near else FAR_THIGH
    shin_col = GREY if near else FAR_SHIN
    joint_col = AMBER if near else FAR_JOINT
    off, lift = foot(phase, tag, layout)
    fx, fy = hip_x + off, layout.ground_y - lift
    kx, ky = knee(hip_x, hip_y, fx, fy, layout.thigh, tag[0] == "f")
    thick = 1.5 if layout.detail else 1.1
    c.limb(hip_x, hip_y, kx, ky, thick, thigh_col)
    c.limb(kx, ky, fx, fy - 1, thick * 0.8, shin_col)
    if layout.detail:
        c.disc(hip_x, hip_y, 1.5, joint_col)
        c.disc(hip_x, hip_y, 0.5, DARKER)
        c.disc(kx, ky, 1.1, joint_col)
    c.rect(fx - 2, fy - 1, fx + 2, fy, joint_col)


def frame(phase: float, cols: int, rows: int) -> str:
    """Draw one frame of the cycle at the given slot size."""
    layout = LAYOUTS[(cols, rows)]
    phase %= 1.0
    c = Canvas(cols, rows)
    bob = round(-layout.bob * abs(math.sin(2 * math.pi * phase)))

    # ground and grass, scrolling right at exactly planted-foot speed. The dash period must
    # divide the per-cycle scroll or the pattern jumps when the loop wraps.
    shift = phase * layout.stride / DUTY
    for x in range(c.width):
        world = int(x - shift)
        if world % layout.ground_period < layout.ground_period // 2:
            c.rect(x, layout.ground_y + 2, x, layout.ground_y + 2, GROUND)

    for tag in ("ff", "rf"):  # far legs first, so the near pair draws over them
        _leg(c, layout.front_hip if tag[0] == "f" else layout.rear_hip, layout.hip_y + bob, phase, tag, layout, False)

    hy0, by0, by1 = layout.head_y0 + bob, layout.body_y0 + bob, layout.body_y1 + bob

    # tail hook, wagging a beat behind the body
    wag = round(1.4 * math.sin(4 * math.pi * phase)) if layout.detail else 0
    c.rect(layout.body_x1 - 1, by0 + wag, layout.body_x1 + 6, by0 + 1 + wag, AMBER)

    # chassis: one cream mass whose front stands taller as the head
    c.rect(layout.head_x0, hy0, layout.head_x1, by1, CREAM)
    c.rect(layout.head_x1, by0, layout.body_x1, by1, CREAM)
    c.rect(layout.head_x0, by1 - 1, layout.body_x1, by1, CREAM_SHADE)
    c.rect(layout.head_x0, hy0, layout.head_x1, hy0, SHELL_HI)
    c.rect(layout.head_x1, by0, layout.body_x1, by0, SHELL_HI)
    # dark inset face panel
    c.rect(layout.head_x0 + layout.face_inset, hy0 + 2, layout.head_x1 - layout.face_inset, by1 - 3, DARKER)

    if layout.detail:
        c.rect(layout.band_x0, by1 - 5, layout.band_x1, by1 - 2, AMBER)
        c.rect(layout.band_x0, by1 - 5, layout.band_x1, by1 - 5, AMBER_HI)
        for i in range(3):
            c.rect(layout.band_x0 + 4 + i * 5, by1 - 4, layout.band_x0 + 5 + i * 5, by1 - 2, DARKER)
        c.rect(layout.band_x1 + 6, by1 - 4, layout.body_x1 - 4, by1 - 2, DARK)
        for i in range(3):
            c.rect(layout.band_x1 + 8 + i * 7, by0 + 2, layout.band_x1 + 10 + i * 7, by0 + 3, CYAN)

    for tag in ("fn", "rn"):  # near legs over the chassis
        _leg(c, layout.front_hip if tag[0] == "f" else layout.rear_hip, layout.hip_y + bob, phase, tag, layout, True)

    # Eyes follow only half the bob, so the head moves around a steady gaze. A pixel is the
    # smallest move available, which is why the body bounces two.
    eye_bob = round(bob * 0.5)
    lids = 0 if 0.89 <= phase < 0.93 else (1 if 0.86 <= phase < 0.96 else 4)
    for ex in (layout.eye_x0, layout.eye_x1):
        if lids:
            c.rect(ex, layout.eye_y + eye_bob, ex + layout.eye_w, layout.eye_y + eye_bob + lids - 1, CYAN)
            if lids >= 4:
                c.rect(ex + 1, layout.eye_y + eye_bob, ex + 1, layout.eye_y + eye_bob, EYE_CORE)

    # antenna, whipping a beat behind the head
    whip = round(1.4 * math.sin(4 * math.pi * (phase - 0.15)))
    ax = layout.head_x1 - 6
    c.limb(ax, hy0, ax + whip, hy0 - 3, 0.5, DARK)
    c.rect(ax + whip - 1, hy0 - 4, ax + whip + 1, hy0 - 4, CYAN)

    # grass in front of the legs, so a boot passing behind a tuft reads as depth
    for x in range(c.width):
        if int(x - shift) % (layout.ground_period * 3) == 0:
            c.rect(x, layout.ground_y, x, layout.ground_y + 1, GRASS)
    return c.render()


def frames(cols: int, rows: int) -> list[str]:
    """Render the whole gait cycle at the given slot size."""
    return [frame(i / FRAMES, cols, rows) for i in range(FRAMES)]
