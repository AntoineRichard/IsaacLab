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
from .isaac import FACE_GLYPHS, row
from .nvidia import GAP, IL3, MARK_CELLS, SLOGAN, SLOGAN_STACKED, _centre, mask

PALETTE = (
    (238, 175, 97),
    (251, 144, 98),
    (238, 93, 108),
    (206, 73, 147),
    (106, 13, 131),
)
"""Sunset ramp, light to dark. Interpolated cyclically, so the end meets the beginning."""

STEPS = 24
"""Distinct colours the ramp is quantised to.

A continuous ramp gives almost every cell its own colour, which leaves gzip nothing to repeat
and triples the stored size. At this many steps the banding is invisible against the block
glyphs but the frames compress like flat colour.
"""

SPREAD = 1.4
"""Palette turns visible across the mark at once. Above one, more than a full ramp shows."""

EDGE_SHADE = 0.34
"""How much of the lit colour a lockup edge keeps, so shading travels with the light."""

FRAMES = 24
"""Frames in one sweep."""


def ramp(t: float) -> tuple[int, int, int]:
    """Colour at position *t* of the palette ring, wrapping at 1.0.

    Args:
        t: Position along the ring. Only the fractional part matters.

    Returns:
        The interpolated colour.
    """
    t = round(t % 1.0 * STEPS) / STEPS * len(PALETTE)
    first = PALETTE[int(t) % len(PALETTE)]
    second = PALETTE[(int(t) + 1) % len(PALETTE)]
    blend = t - int(t)
    return tuple(round(a + (b - a) * blend) for a, b in zip(first, second))


def frame(phase: float, cols: int, rows: int) -> str:
    """Draw the mark at *phase* of the sweep.

    Every piece takes the colour arriving at its own column of the whole greeting, not of the
    piece it sits in, so the ramp crosses the mark, the lockup and the slogan as one motion.
    """
    phase %= 1.0  # so frame(1.0) is bit-identical to frame(0.0) and the loop closes

    def lit(cell: float) -> tuple[int, int, int]:
        """Colour arriving at *cell*, measured in cells from the greeting's left edge."""
        return ramp(cell / cols * SPREAD - phase)

    if cols < MARK_CELLS + GAP + max(len(part) for part in IL3):
        # narrow: the mark alone, over the slogan split across two lines
        text_rows = len(SLOGAN_STACKED) + 1
        canvas = Canvas(cols, rows - text_rows)
        for y, line in enumerate(mask(canvas.width, canvas.height)):
            for x, ink in enumerate(line):
                if ink:
                    canvas.set(x, y, lit(x / 2))
        lines = canvas.render().splitlines()
        lines.append("\x1b[0m" + " " * cols)
        lines += [_centre(part, cols, lit(cols / 2)) for part in SLOGAN_STACKED]
        return "\n".join(lines)

    art_rows = rows - 2
    lockup_width = max(len(part) for part in IL3)
    block = MARK_CELLS + GAP + lockup_width
    left = max((cols - block) // 2, 0)
    top = (art_rows - len(IL3)) // 2

    canvas = Canvas(MARK_CELLS, art_rows)
    for y, line in enumerate(mask(canvas.width, canvas.height)):
        for x, ink in enumerate(line):
            if ink:
                canvas.set(x, y, lit(left + x / 2))

    def letter(glyph: str, index: int) -> tuple[int, int, int]:
        colour = lit(left + MARK_CELLS + GAP + index)
        return colour if glyph in FACE_GLYPHS else tuple(round(c * EDGE_SHADE) for c in colour)

    lines = []
    for index, line in enumerate(canvas.render().splitlines()):
        part = IL3[index - top] if 0 <= index - top < len(IL3) else ""
        lines.append(
            "\x1b[0m"
            + " " * left
            + line
            + "\x1b[0m"
            + " " * GAP
            + row(part, lockup_width, 0, letter)
            + "\x1b[0m"
            + " " * max(cols - left - block, 0)
        )
    lines.append("\x1b[0m" + " " * cols)
    pad = max((cols - len(SLOGAN)) // 2, 0)
    lines.append(row(SLOGAN, cols, pad, lambda g, i: lit(pad + i)))
    lines += ["\x1b[0m" + " " * cols] * (rows - len(lines))
    return "\n".join(lines)


def frames(cols: int, rows: int) -> list[str]:
    """Render the whole sweep at the given slot size."""
    return [frame(i / FRAMES, cols, rows) for i in range(FRAMES)]
