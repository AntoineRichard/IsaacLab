# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Isaac Lab wordmark, tinted.

The letters are already block characters, so they are tinted rather than redrawn: colouring
existing glyphs keeps their edges exactly as crisp as the terminal font can render, where
resampling them into a pixel grid and re-encoding would only soften them. Nothing here touches
:mod:`~isaaclab.app.anims`' canvas.

Colour follows the letterforms rather than position. The wordmark is drawn with solid blocks
for the letter faces and box-drawing lines for their edges, so tinting those two groups
differently makes the edges read as shading and the letters gain depth. A gradient across the
columns ignores that structure and just looks like a colour wash laid over the top.
"""

from __future__ import annotations

WORDMARK = (
    "██╗███████╗ █████╗  █████╗  ██████╗   ██╗      █████╗ ██████╗",
    "██║██╔════╝██╔══██╗██╔══██╗██╔════╝   ██║     ██╔══██╗██╔══██╗",
    "██║███████╗███████║███████║██║        ██║     ███████║██████╔╝",
    "██║╚════██║██╔══██║██╔══██║██║        ██║     ██╔══██║██╔══██╗",
    "██║███████║██║  ██║██║  ██║╚██████╗   ███████╗██║  ██║██████╔╝",
    "╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚══════╝╚═╝  ╚═╝╚═════╝",
)
"""Six rows of block letters, as shipped in :mod:`~isaaclab.app.loading_screen`."""

CAPTION = "The way it's meant to be trained"
"""Set under the wordmark, matching the narrow NVIDIA greeting."""

FACE = (118, 185, 0)
"""NVIDIA green, for the solid blocks that make up the letter faces."""

EDGE = (52, 84, 0)
"""A deep green for the box-drawing edges, which read as the letters' shading."""

FACE_GLYPHS = "█▓▒░"
"""Glyphs treated as letter face; everything else drawn is an edge."""

CAPTION_COLOUR = (150, 154, 160)
"""Caption grey, quieter than the wordmark."""


def row(line: str, cols: int, left: int, colour_of) -> str:
    """One output row: *line* placed at *left*, padded to *cols*, coloured per glyph.

    Shared with the animated variant, which passes a *colour_of* that also depends on phase.

    Args:
        line: Text to place.
        cols: Row width in cells.
        left: Column to start at.
        colour_of: Called with ``(glyph, index)`` and returning the colour for that glyph.

    Returns:
        The row, exactly *cols* cells wide. A colour is emitted only when it changes, so a
        tinted row costs far less than an escape sequence per character.
    """
    out, current = ["\x1b[0m" + " " * left], None
    for index, glyph in enumerate(line[: cols - left]):
        if glyph == " ":
            if current is not None:
                out.append("\x1b[0m")
                current = None
            out.append(" ")
            continue
        colour = colour_of(glyph, index)
        if colour != current:
            out.append(f"\x1b[38;2;{colour[0]};{colour[1]};{colour[2]}m")
            current = colour
        out.append(glyph)
    out.append("\x1b[0m" + " " * max(cols - left - len(line), 0))
    return "".join(out)


def block_left(body: list[str], cols: int) -> int:
    """Column to start every row of *body* at, so the block is centred as one object.

    Centring each row on its own length looks equivalent and is not: rows of different length
    get different offsets, which shifts them relative to each other and breaks any alignment
    the art was drawn with. The original greeting's antenna sits exactly over the centre of
    its face; centring per row moved it a column off.
    """
    return max((cols - max((len(line) for line in body), default=0)) // 2, 0)


def blank(cols: int) -> str:
    """An empty row of *cols* cells."""
    return "\x1b[0m" + " " * cols


def layout(cols: int, rows: int) -> tuple[list[str], int]:
    """The rows to draw and the padding above them, for either slot.

    The wordmark is 62 columns wide, so a narrower slot gets the caption alone rather than a
    wordmark cut off mid-letter.
    """
    fits = cols >= max(len(line) for line in WORDMARK)
    body = [*WORDMARK, "", CAPTION] if fits else [CAPTION]
    return body, max((rows - len(body)) // 2, 0)


def frames(cols: int, rows: int) -> list[str]:
    """Render the wordmark at the given slot size.

    Returns a single frame: this greeting is still. The wordmark is 62 columns wide, so it
    only fits the wide slot; a narrower one gets the caption alone rather than a wordmark cut
    off mid-letter.
    """
    body, top = layout(cols, rows)
    left = block_left(WORDMARK if len(body) > 1 else body, cols)
    lines = [blank(cols)] * top
    for offset, line in enumerate(body[: rows - top]):
        if not line:
            lines.append(blank(cols))
            continue
        wordmark = offset < len(WORDMARK) and len(body) > 1
        colour_of = (lambda g, i: FACE if g in FACE_GLYPHS else EDGE) if wordmark else (lambda g, i: CAPTION_COLOUR)
        # the wordmark shares one origin so its rows stay aligned; the caption centres alone
        start = left if wordmark else max((cols - len(line)) // 2, 0)
        lines.append(row(line, cols, start, colour_of))
    lines += [blank(cols)] * (rows - len(lines))
    return ["\n".join(lines[:rows])]
