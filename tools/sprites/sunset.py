# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The NVIDIA eye mark with a sunset gradient travelling across it.

The shape is the same one :mod:`.nvidia` draws; only the colour moves. A pixel's colour is a
function of its column and the phase, so the whole cycle is one sweep of the palette from left
to right.

The palette is treated as a **ring**: the last colour interpolates back to the first. Without
that the sweep would jump from deep purple to orange as the cycle wrapped, which is exactly the
kind of seam that makes a short loop look broken.
"""

from __future__ import annotations

from .canvas import Canvas
from .nvidia import SLOGAN_STACKED, _centre, mask

PALETTE = (
    (238, 175, 97),
    (251, 144, 98),
    (238, 93, 108),
    (206, 73, 147),
    (106, 13, 131),
)
"""Sunset ramp, light to dark. Interpolated cyclically, so the end meets the beginning."""

SPREAD = 1.4
"""Palette turns visible across the mark at once. Above one, more than a full ramp shows."""

FRAMES = 24
"""Frames in one sweep."""


def ramp(t: float) -> tuple[int, int, int]:
    """Colour at position *t* of the palette ring, wrapping at 1.0.

    Args:
        t: Position along the ring. Only the fractional part matters.

    Returns:
        The interpolated colour.
    """
    t = t % 1.0 * len(PALETTE)
    first = PALETTE[int(t) % len(PALETTE)]
    second = PALETTE[(int(t) + 1) % len(PALETTE)]
    blend = t - int(t)
    return tuple(round(a + (b - a) * blend) for a, b in zip(first, second))


def frame(phase: float, cols: int, rows: int) -> str:
    """Draw the mark at *phase* of the sweep."""
    text_rows = len(SLOGAN_STACKED) + 1
    canvas = Canvas(cols, rows - text_rows)
    for y, row in enumerate(mask(canvas.width, canvas.height)):
        for x, ink in enumerate(row):
            if ink:
                canvas.set(x, y, ramp(x / canvas.width * SPREAD - phase))
    lines = canvas.render().splitlines()
    lines.append("\x1b[0m" + " " * cols)
    # the slogan takes the colour arriving at its own column, so the sweep reads as one motion
    # across the whole greeting rather than the mark animating above static text
    lines += [_centre(part, cols, ramp(-phase + 0.5)) for part in SLOGAN_STACKED]
    return "\n".join(lines)


def frames(cols: int, rows: int) -> list[str]:
    """Render the whole sweep at the given slot size."""
    return [frame(i / FRAMES, cols, rows) for i in range(FRAMES)]
