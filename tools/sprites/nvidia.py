# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The NVIDIA eye mark, with a slogan where there is room for one.

The mark is read from the logo already vendored in the repository, cropped to the eye and
fitted to the slot. The wordmark below it is dropped: at twelve rows it renders as a smear,
and the slogan says more in the same space.

The slogan is emitted as **literal characters**, not drawn into the pixel grid. Text rendered
as quadrant blocks at this size is unreadable, whereas the terminal's own font is exactly as
legible as the terminal can be. A frame is just lines of terminal output, so the two mix
freely: art on the upper rows, type on the lower ones.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .canvas import Canvas
from .isaac import EDGE, FACE, FACE_GLYPHS, row

LOGO = Path(__file__).resolve().parents[2] / "docs" / "source" / "_static" / "NVIDIA-logo-black.png"
"""The NVIDIA logo vendored in the repository; the eye mark is cropped out of it."""

GREEN = (118, 185, 0)
"""NVIDIA green."""

GAP = 3
"""Cells between the mark and the lockup."""


def _letter(glyph: str, _index: int) -> tuple[int, int, int]:
    """Face green for the solid blocks, a darker shade for the box-drawing edges."""
    return FACE if glyph in FACE_GLYPHS else EDGE


SLOGAN = "The way it's meant to be trained"
"""Shown with the mark: beside it where there is room, stacked under it where there is not."""

SLOGAN_STACKED = ("The way it's meant", "to be trained")
"""The slogan split for a narrow slot, where one line would not fit."""

IL3 = (
    "██╗██╗     ██████╗",
    "██║██║     ╚════██╗",
    "██║██║      █████╔╝",
    "██║██║      ╚═══██╗",
    "██║███████╗██████╔╝",
    "╚═╝╚══════╝╚═════╝",
)
"""Isaac Lab 3, short. Same block letters as the wordmark, so the two read as one family."""

SLOGAN_COLOUR = (150, 154, 160)
"""Slogan grey. Quieter than the mark, which should stay the thing the eye lands on."""


def _eye() -> Image.Image:
    """The eye mark alone, cropped off the top of the shipped logo.

    The logo stacks the mark over the wordmark with a blank band between them; the band is
    found rather than hard-coded, so a future asset with different padding still crops right.
    """
    image = Image.open(LOGO).convert("RGBA")
    alpha = image.getchannel("A")
    rows = [max(alpha.crop((0, y, image.width, y + 1)).getextrema()) for y in range(image.height)]
    band, start = None, None
    for y, value in enumerate(rows):
        if value < 16 and start is None:
            start = y
        elif value >= 16 and start is not None:
            if start > 0 and y - start > image.height * 0.02:
                band = start
                break
            start = None
    return image.crop(image.crop((0, 0, image.width, band or image.height)).getbbox())


def mask(width: int, height: int) -> list[list[bool]]:
    """The eye mark as a coverage grid *width* x *height* subpixels, centred and proportional.

    Shared with the animated variants, which colour the same shape rather than re-deriving it.
    The alpha threshold is deliberately low: the eye is built from thin strokes, and a strict
    threshold drops the pixels where a stroke only partly covers its cell, which breaks the
    shape into fragments rather than thinning it.
    """
    art = _eye()
    # a subpixel displays twice as tall as it is wide, so a mark of aspect w:h wants
    # 2 * h * w / h subpixels across
    drawn_h = height
    drawn_w = round(2 * drawn_h * art.width / art.height)
    if drawn_w > width:
        drawn_w = width
        drawn_h = round(drawn_w * art.height / (2 * art.width))
    resized = art.resize((max(drawn_w, 1), max(drawn_h, 1)), Image.LANCZOS)
    left, top = (width - drawn_w) // 2, (height - drawn_h) // 2
    grid = [[False] * width for _ in range(height)]
    for y in range(drawn_h):
        for x in range(drawn_w):
            *_, alpha = resized.getpixel((x, y))
            if alpha >= 60:
                grid[top + y][left + x] = True
    return grid


def _draw(canvas: Canvas, span: int) -> None:
    """Paint the mark in green across the leftmost *span* subpixels of *canvas*.

    Shape, fit and alpha threshold all come from :func:`mask`.
    """
    for y, line in enumerate(mask(span, canvas.height)):
        for x, ink in enumerate(line):
            if ink:
                canvas.set(x, y, GREEN)


def _centre(text: str, cols: int, colour: tuple[int, int, int]) -> str:
    """One row of *cols* cells holding *text*, centred and coloured."""
    pad = max((cols - len(text)) // 2, 0)
    body = text[:cols]
    return (
        "\x1b[0m"
        + " " * pad
        + f"\x1b[38;2;{colour[0]};{colour[1]};{colour[2]}m"
        + body
        + "\x1b[0m"
        + " " * max(cols - pad - len(body), 0)
    )


def frames(cols: int, rows: int) -> list[str]:
    """Render the mark at the given slot size.

    Returns a single frame: this greeting is still.

    Wide enough, and the mark is paired with the :data:`IL3` lockup and captioned underneath.
    Too narrow for the pair, and the mark stands alone over the slogan split across two lines.
    Composing each frame from pieces of known width keeps every line exactly *cols* cells wide,
    which the loading screen needs if the greeting is not to wrap.
    """
    paired = cols >= ((rows - 2) * 2) + GAP + max(len(part) for part in IL3)

    if paired:
        # the mark takes the upper rows, the lockup sits beside it, the slogan spans the whole
        # width below. Three pieces of known width, so every line lands at exactly *cols*
        art_rows = rows - 2
        mark_cols = art_rows * 2
        canvas = Canvas(mark_cols, art_rows)
        _draw(canvas, canvas.width)
        art = canvas.render().splitlines()
        lockup_width = max(len(part) for part in IL3)
        block = mark_cols + GAP + lockup_width
        left = max((cols - block) // 2, 0)
        top = (art_rows - len(IL3)) // 2

        lines = []
        for index, line in enumerate(art):
            part = IL3[index - top] if 0 <= index - top < len(IL3) else ""
            lines.append(
                "\x1b[0m"
                + " " * left
                + line
                + "\x1b[0m"
                + " " * GAP
                + row(part, lockup_width, 0, _letter)
                + "\x1b[0m"
                + " " * max(cols - left - block, 0)
            )
        lines.append("\x1b[0m" + " " * cols)
        lines.append(_centre(SLOGAN, cols, SLOGAN_COLOUR))
        lines += ["\x1b[0m" + " " * cols] * (rows - len(lines))
        return ["\n".join(lines)]

    # stacked: the mark takes the rows the slogan does not need
    text_rows = len(SLOGAN_STACKED) + 1
    canvas = Canvas(cols, rows - text_rows)
    _draw(canvas, canvas.width)
    lines = canvas.render().splitlines()
    lines.append("\x1b[0m" + " " * cols)
    lines += [_centre(part, cols, SLOGAN_COLOUR) for part in SLOGAN_STACKED]
    return ["\n".join(lines)]
