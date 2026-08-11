# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drawing surface and encoder shared by the sprites and the image converter.

A greeting is a grid of quadrant subpixels: two across and two down per character cell, so a
``cols`` x ``rows`` greeting is drawn on a ``cols * 2`` x ``rows * 2`` pixel buffer.

A subpixel is half a cell wide but a whole cell tall, so it **displays twice as tall as it is
wide**. Horizontal and vertical distances are not interchangeable, and anything meant to look
round or square has to be drawn twice as wide as it is tall. :data:`ASPECT` is that factor and
:meth:`Canvas.disc` applies it.

See ``tools/ascii_rendering.md`` for the encoder's reasoning.
"""

from __future__ import annotations

import functools
import math

ASPECT = 2.0
"""Subpixel height over width. Multiply horizontal radii by this to keep circles round."""

QUAD = {
    0b0000: " ",
    0b0001: "▘",
    0b0010: "▝",
    0b0011: "▀",
    0b0100: "▖",
    0b0101: "▌",
    0b0110: "▞",
    0b0111: "▛",
    0b1000: "▗",
    0b1001: "▚",
    0b1010: "▐",
    0b1011: "▜",
    0b1100: "▄",
    0b1101: "▙",
    0b1110: "▟",
    0b1111: "█",
}
"""Quadrant blocks indexed by a 2x2 occupancy mask, bit 0 top-left through bit 3 bottom-right."""

RGB = tuple[int, int, int]


def _mean(colours: list[RGB]) -> RGB:
    """Average of a non-empty list of colours."""
    n = len(colours)
    return (sum(c[0] for c in colours) // n, sum(c[1] for c in colours) // n, sum(c[2] for c in colours) // n)


def _error(quad: tuple, mask: int, fg: RGB, bg: RGB) -> int:
    """Squared RGB error of reproducing *quad* with *mask* split into *fg* and *bg*."""
    return sum(sum((a - b) ** 2 for a, b in zip(px, fg if mask >> i & 1 else bg)) for i, px in enumerate(quad))


@functools.cache
def encode(quad: tuple) -> tuple[str, RGB | None, RGB | None]:
    """Choose the glyph and colours that best reproduce one 2x2 block.

    All sixteen masks are tried and the least-error split wins, so a block holding at most two
    distinct colours reproduces exactly. A single colour per cell would average the two, which
    is what erases one-pixel detail.

    Returns:
        The glyph, its foreground colour, and its background colour. The background is None
        where the block is partly transparent, so the greeting composites over the terminal
        rather than painting a box around itself.
    """
    if all(p is None for p in quad):
        return " ", None, None
    if any(p is None for p in quad):
        mask = sum(1 << i for i, p in enumerate(quad) if p is not None)
        return QUAD[mask], _mean([p for p in quad if p is not None]), None
    best = None
    for mask in range(16):
        front = [p for i, p in enumerate(quad) if mask >> i & 1]
        back = [p for i, p in enumerate(quad) if not mask >> i & 1]
        fg = _mean(front) if front else _mean(back)
        bg = _mean(back) if back else _mean(front)
        score = (_error(quad, mask, fg, bg), len(front), mask)
        if best is None or score < best[0]:
            best = (score, mask, fg, bg)
    _, mask, fg, bg = best
    return QUAD[mask], fg, bg


def render(pixels: list[list[RGB | None]], cols: int, rows: int) -> str:
    """Pack a subpixel grid into styled quadrant rows.

    A colour is emitted only when it differs from the previous cell's; repeating it for every
    cell roughly doubles the byte count, which matters when the frames ship in the wheel.

    Args:
        pixels: ``rows * 2`` rows of ``cols * 2`` colours, None where transparent.
        cols: Output width in cells.
        rows: Output height in cells.

    Returns:
        The frame, without a trailing newline.
    """
    lines = []
    for row in range(rows):
        line, current = "", None
        for col in range(cols):
            quad = tuple(pixels[row * 2 + dy][col * 2 + dx] for dy in (0, 1) for dx in (0, 1))
            glyph, fg, bg = encode(quad)
            if (fg, bg) != current:
                line += "\x1b[0m"
                if fg is not None:
                    line += f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m"
                if bg is not None:
                    line += f"\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m"
                current = (fg, bg)
            line += glyph
        lines.append(line + "\x1b[0m")
    return "\n".join(lines)


class Canvas:
    """A subpixel buffer for one frame. Later writes win, so draw back to front."""

    def __init__(self, cols: int, rows: int) -> None:
        """Create a buffer for a *cols* x *rows* greeting."""
        self.cols, self.rows = cols, rows
        self.width, self.height = cols * 2, rows * 2
        self.px: list[list[RGB | None]] = [[None] * self.width for _ in range(self.height)]

    def set(self, x: int, y: int, rgb: RGB) -> None:
        """Write one subpixel, ignoring anything outside the buffer."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.px[y][x] = rgb

    def rect(self, x0: float, y0: float, x1: float, y1: float, rgb: RGB) -> None:
        """A filled axis-aligned rectangle, inclusive of both ends."""
        for y in range(int(round(y0)), int(round(y1)) + 1):
            for x in range(int(round(x0)), int(round(x1)) + 1):
                self.set(x, y, rgb)

    def disc(self, cx: float, cy: float, r: float, rgb: RGB) -> None:
        """A dot of radius *r* that displays round, widened by :data:`ASPECT`."""
        rx, ry = max(r * ASPECT, 0.5), max(r, 0.5)
        for y in range(math.floor(cy - ry), math.ceil(cy + ry) + 1):
            for x in range(math.floor(cx - rx), math.ceil(cx + rx) + 1):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                    self.set(x, y, rgb)

    def limb(self, x0: float, y0: float, x1: float, y1: float, half: float, rgb: RGB) -> None:
        """A straight segment *half* subpixels thick either side of its centreline."""
        steps = max(int(math.hypot(x1 - x0, y1 - y0)) * 4, 1)
        for s in range(steps + 1):
            t = s / steps
            cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            self.rect(cx - half, cy - half / ASPECT, cx + half, cy + half / ASPECT, rgb)

    def render(self) -> str:
        """Encode the buffer as quadrant rows."""
        return render(self.px, self.cols, self.rows)
