# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A robot arm waving hello, over a greeting line.

Both joints move. The shoulder sways a little and the elbow swings a lot, with the shoulder
trailing behind -- a chain driven from its base, where each link starts after the one before
it. Driving both from the same phase would make the arm swing as one rigid stick.

The elbow also carries a constant bend, so the arm is never straight. A straight arm at this
size reads as a pole; a bent one reads as an arm.

The base sits left of centre and waves into the space on the right, rather than being centred
and waving symmetrically about itself.

Every angle is a sine of the phase, so the cycle closes on its own -- the arm returns to where
it started with the same velocity, and there is no seam at the wrap.

One subpixel is half a cell wide but a whole cell tall, so a horizontal swing needs twice the
travel of a vertical one to look like the same angle. :data:`REACH_X` and :data:`REACH_Y`
carry that factor rather than a trigonometric radius, which would come out visibly squashed.
"""

from __future__ import annotations

import math

from .canvas import Canvas
from .isaac import blank, row

CREAM = (232, 228, 214)
SHELL_HI = (255, 253, 244)
DARK = (48, 48, 56)
DARKER = (26, 26, 32)
JOINT = (118, 185, 0)
"""NVIDIA green for the actuators, tying the arm to the rest of the greetings."""

CYAN = (77, 217, 232)
CAPTION_COLOUR = (150, 154, 160)

CAPTION = "Welcome to Isaac Lab!"
"""Set under the arm. Twenty-one columns, so it fits the narrow slot with room to spare."""

BASE_Y = 19
SHOULDER = (17, 15)
"""Pedestal top, left of centre so the arm waves into the open space rather than over itself."""

UPPER = (11.0, 5.5)
FORE = (11.0, 5.5)
"""Upper arm and forearm reach, across and up. Twice as far across as up, because a subpixel
displays twice as tall as it is wide and a trigonometric radius would come out squashed."""

SHOULDER_SWING = 0.24
"""Peak shoulder sway in radians. Small: the shoulder carries the arm, it does not wave."""

SWING = 0.58
"""Peak elbow swing in radians, about 33 degrees either side of its rest angle."""

BEND = 0.42
"""Constant elbow bend, so the arm is never straight."""

SHOULDER_LAG = 0.14
"""How far the shoulder trails the elbow, as a fraction of the cycle."""

WRIST_LAG = 0.07
"""How far the wrist trails the arm, as a fraction of the cycle.

Without it the wave is perfectly symmetric -- the hand retraces its own path, so the outward
and return strokes render identically and half the frames are duplicates. A lag makes the
return differ from the outward swing, and is truer to how a hand moves.
"""

FRAMES = 24
"""Frames in one wave."""


def _angles(phase: float) -> tuple[float, float]:
    """Upper arm and forearm angles from vertical at *phase*, in radians."""
    upper = SHOULDER_SWING * math.sin(2 * math.pi * (phase - SHOULDER_LAG))
    fore = upper + BEND + SWING * math.sin(2 * math.pi * phase)
    return upper, fore


def _joints(phase: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Elbow and hand positions at *phase*."""
    upper, fore = _angles(phase)
    elbow = (SHOULDER[0] + UPPER[0] * math.sin(upper), SHOULDER[1] - UPPER[1] * math.cos(upper))
    hand = (elbow[0] + FORE[0] * math.sin(fore), elbow[1] - FORE[1] * math.cos(fore))
    return elbow, hand


def frame(phase: float, cols: int, rows: int) -> str:
    """Draw one frame of the wave at the given slot size."""
    phase %= 1.0  # so frame(1.0) is bit-identical to frame(0.0) and the loop closes
    art_rows = rows - 2
    canvas = Canvas(cols, art_rows)
    (elbow_x, elbow_y), (hand_x, hand_y) = _joints(phase)

    # pedestal, then the fixed upper arm, then the swinging forearm over it
    canvas.rect(SHOULDER[0] - 8, BASE_Y - 2, SHOULDER[0] + 8, BASE_Y, DARK)
    canvas.rect(SHOULDER[0] - 8, BASE_Y - 2, SHOULDER[0] + 8, BASE_Y - 2, DARKER)
    canvas.rect(SHOULDER[0] - 3, SHOULDER[1], SHOULDER[0] + 3, BASE_Y - 2, CREAM)
    canvas.limb(*SHOULDER, elbow_x, elbow_y, 2.0, CREAM)
    canvas.rect(SHOULDER[0] - 2, SHOULDER[1] - 1, SHOULDER[0] + 2, SHOULDER[1] - 1, SHELL_HI)
    canvas.disc(*SHOULDER, 1.6, JOINT)
    canvas.disc(*SHOULDER, 0.5, DARKER)

    canvas.limb(elbow_x, elbow_y, hand_x, hand_y, 1.6, CREAM)
    canvas.disc(elbow_x, elbow_y, 1.6, JOINT)
    canvas.disc(elbow_x, elbow_y, 0.5, DARKER)

    # the hand: a palm with two fingers splayed along the wrist's direction, which trails the
    # forearm rather than matching it
    wrist = _angles(phase - WRIST_LAG)[1]
    ux, uy = math.sin(wrist), -math.cos(wrist)
    canvas.disc(hand_x, hand_y, 1.8, CYAN)
    for side in (-1.0, 1.0):
        tip_x = hand_x + (ux * 1.2 - side * uy * 1.6) * 2
        tip_y = hand_y + (uy * 1.2 + side * ux * 1.6)
        canvas.limb(hand_x, hand_y, tip_x, tip_y, 0.8, CYAN)

    lines = canvas.render().splitlines()
    lines.append(blank(cols))
    lines.append(row(CAPTION, cols, max((cols - len(CAPTION)) // 2, 0), lambda g, i: CAPTION_COLOUR))
    return "\n".join(lines)


def frames(cols: int, rows: int) -> list[str]:
    """Render the whole wave at the given slot size."""
    return [frame(i / FRAMES, cols, rows) for i in range(FRAMES)]
