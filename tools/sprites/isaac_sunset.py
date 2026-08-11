# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Isaac Lab wordmark with the sunset ramp sweeping across it.

Same letterforms and same structure as :mod:`.isaac`: the solid blocks are the letter faces
and the box-drawing characters are their edges. The sweep colours the faces, and the edges take
a darkened version of *the same* colour rather than a fixed shade -- so the shading travels
with the light instead of sitting still underneath it.

The caption sweeps too, half a turn behind, so the greeting reads as one motion rather than an
animated wordmark above static text.
"""

from __future__ import annotations

from .isaac import CAPTION, FACE_GLYPHS, WORDMARK, blank, block_left, layout, row
from .sunset import ramp

SPREAD = 1.2
"""Palette turns visible across the wordmark at once. Above one, more than a full ramp shows."""

EDGE_SHADE = 0.34
"""How much of the face colour an edge keeps. Low enough to read as shadow at every hue."""

FRAMES = 24
"""Frames in one sweep."""


def frame(phase: float, cols: int, rows: int) -> str:
    """Draw the wordmark at *phase* of the sweep."""
    phase %= 1.0  # so frame(1.0) is bit-identical to frame(0.0) and the loop closes
    body, top = layout(cols, rows)
    width = max(len(line) for line in body)
    lines = [blank(cols)] * top

    for offset, line in enumerate(body[: rows - top]):
        if not line:
            lines.append(blank(cols))
            continue
        if offset < len(WORDMARK) and len(body) > 1:

            def colour_of(glyph: str, index: int) -> tuple[int, int, int]:
                lit = ramp(index / width * SPREAD - phase)
                return lit if glyph in FACE_GLYPHS else tuple(round(c * EDGE_SHADE) for c in lit)

        else:

            def colour_of(glyph: str, index: int) -> tuple[int, int, int]:
                return ramp(index / max(len(CAPTION), 1) * SPREAD - phase + 0.5)

        start = (
            block_left(list(WORDMARK), cols)
            if offset < len(WORDMARK) and len(body) > 1
            else max((cols - len(line)) // 2, 0)
        )
        lines.append(row(line, cols, start, colour_of))

    lines += [blank(cols)] * (rows - len(lines))
    return "\n".join(lines[:rows])


def frames(cols: int, rows: int) -> list[str]:
    """Render the whole sweep at the given slot size."""
    return [frame(i / FRAMES, cols, rows) for i in range(FRAMES)]
