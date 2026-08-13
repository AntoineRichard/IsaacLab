# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-color quadrant encoder: turns a subpixel color grid into terminal rows.

A terminal cell carries one glyph, one foreground color and one background color, so it can
show two colors divided by whatever shape the glyph draws. Quadrant block characters split a
cell into 2x2, which gives four times the spatial resolution of one character per cell while
keeping two independent colors.

The grid is a grid of quadrant subpixels: two across and two down per character cell, so a
``cols`` x ``rows`` greeting is drawn on a ``cols * 2`` x ``rows * 2`` pixel buffer.

A subpixel is half a cell wide and half a cell tall, so it keeps the cell's own proportions:
about twice as tall as it is wide. A caller projecting geometry onto this grid needs the same
aspect correction it would use for whole cells, at twice the resolution in each axis.
"""

from __future__ import annotations

import functools

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


def _mean(colors: list[RGB]) -> RGB:
    """Average of a non-empty list of colors."""
    n = len(colors)
    return (sum(c[0] for c in colors) // n, sum(c[1] for c in colors) // n, sum(c[2] for c in colors) // n)


def _error(quad: tuple, mask: int, fg: RGB, bg: RGB) -> int:
    """Squared RGB error of reproducing *quad* with *mask* split into *fg* and *bg*."""
    return sum(sum((a - b) ** 2 for a, b in zip(px, fg if mask >> i & 1 else bg)) for i, px in enumerate(quad))


@functools.cache
def encode(quad: tuple) -> tuple[str, RGB | None, RGB | None]:
    """Choose the glyph and colors that best reproduce one 2x2 block.

    All sixteen masks are tried and the least-error split wins, so a block holding at most two
    distinct colors reproduces exactly. A single color per cell would average the two, which
    is what erases one-pixel detail.

    Returns:
        The glyph, its foreground color, and its background color. The background is None
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

    A color is emitted only when it differs from the previous cell's; repeating it for every
    cell grows the raw frame by about half, which matters when the frames ship in the wheel.

    Args:
        pixels: ``rows * 2`` rows of ``cols * 2`` colors, None where transparent.
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
