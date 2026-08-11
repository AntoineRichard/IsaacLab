# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Isaac Lab greeting face, tinted by role, with its caption below.

Antenna arms take NVIDIA green, the eyes and antenna tip cyan, the panel cream. The caption
sits under the face, where every other greeting puts its text.

Like the wordmark, the art is left as characters and tinted in place. Redrawing five rows of
line art into a pixel grid would only soften edges the terminal font already renders crisply.
"""

from __future__ import annotations

from .isaac import blank, block_left, row

FACE = (
    r"   \   /",
    r" .-------.",
    r" | o   o |",
    r" |   _   |",
    r" '-------'",
)
"""The face, without its caption."""

CAPTION = "Welcome to Isaac Lab!"
"""Set below the face, matching the caption position of the other greetings."""

SHELL = (232, 228, 214)
"""Cream, for the panel the face is drawn on."""

EYE = (77, 217, 232)
"""Cyan, for the eyes and the antenna tip -- the same cyan the arm's hand uses."""

ANTENNA = (118, 185, 0)
"""NVIDIA green, for the antenna arms."""

CAPTION_COLOUR = (150, 154, 160)
"""Caption grey, quieter than the face."""

EYE_GLYPHS = "o"
ANTENNA_GLYPHS = "\\/"
"""Glyph roles. Everything else drawn is panel."""


def _colour(glyph: str, _index: int) -> tuple[int, int, int]:
    """Colour for one glyph of the face, chosen by what it depicts."""
    if glyph in EYE_GLYPHS:
        return EYE
    if glyph in ANTENNA_GLYPHS:
        return ANTENNA
    return SHELL


def frames(cols: int, rows: int) -> list[str]:
    """Render the face at the given slot size.

    Returns a single frame: this greeting is still.
    """
    body = [*FACE, "", CAPTION]
    top = max((rows - len(body)) // 2, 0)
    left = block_left(list(FACE), cols)
    lines = [blank(cols)] * top
    for offset, line in enumerate(body[: rows - top]):
        if not line:
            lines.append(blank(cols))
            continue
        face = offset < len(FACE)
        colour_of = _colour if face else (lambda g, i: CAPTION_COLOUR)
        # the face shares one origin so the antenna stays over the centre of the head
        start = left if face else max((cols - len(line)) // 2, 0)
        lines.append(row(line, cols, start, colour_of))
    lines += [blank(cols)] * (rows - len(lines))
    return ["\n".join(lines[:rows])]
