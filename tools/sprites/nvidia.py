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

LOGO = Path(__file__).resolve().parents[2] / "docs" / "source" / "_static" / "NVIDIA-logo-black.png"
"""The NVIDIA logo vendored in the repository; the eye mark is cropped out of it."""

GREEN = (118, 185, 0)
"""NVIDIA green."""

SLOGAN = "The way it's meant to be trained"
"""Shown with the mark: beside it where there is room, stacked under it where there is not."""

SLOGAN_STACKED = ("The way it's meant", "to be trained")
"""The slogan split for a narrow slot, where one line would not fit."""

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
    for y, row in enumerate(mask(span, canvas.height)):
        for x, ink in enumerate(row):
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

    The slogan sits beside the mark where a line of it fits, and stacked underneath where it
    does not. Composing each frame from pieces of known width keeps every line exactly *cols*
    cells wide, which the loading screen needs if the greeting is not to wrap.
    """
    gap = 2
    beside = cols >= (rows * 2) + gap + len(SLOGAN)

    if beside:
        art_cols = rows * 2
        canvas = Canvas(art_cols, rows)
        _draw(canvas, canvas.width)
        art = canvas.render().splitlines()
        middle = rows // 2
        lines = []
        for row, line in enumerate(art):
            tail = SLOGAN if row == middle else ""
            lines.append(
                line + _centre(tail, cols - art_cols, SLOGAN_COLOUR)
                if tail
                else line + "\x1b[0m" + " " * (cols - art_cols)
            )
        return ["\n".join(lines)]

    # stacked: the mark takes the rows the slogan does not need
    text_rows = len(SLOGAN_STACKED) + 1
    canvas = Canvas(cols, rows - text_rows)
    _draw(canvas, canvas.width)
    lines = canvas.render().splitlines()
    lines.append("\x1b[0m" + " " * cols)
    lines += [_centre(part, cols, SLOGAN_COLOUR) for part in SLOGAN_STACKED]
    return ["\n".join(lines)]
