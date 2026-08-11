# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn an image, a GIF, or a procedural sprite into a loading screen greeting.

This is a build-time tool. It reads images, so it needs Pillow -- which is exactly why it
lives here and not in ``isaaclab.app``: the simulator startup path must not import an image
library to draw a greeting. The output is a container of pre-rendered terminal frames that
the loading screen decompresses and prints.

Two input paths:

* **An image or GIF.** Resampled to the target grid and encoded. Use this for logos and
  anything drawn outside this repository.
* **A procedural sprite** under ``tools.sprites``, via ``--from-module``. Those modules draw
  straight onto the target grid, so routing them through an image would resample art that was
  authored at exactly the right size -- the one thing pixel art does not survive.

Encoding is the two-colour quadrant scheme described in ``tools/ascii_rendering.md``: each
cell carries a foreground and a background, and the glyph is chosen to split them with the
least error. See that document before changing anything here.

    uv run python tools/gif2anim.py logo.png --size wide --name nvidia-wide --out DIR
    uv run python tools/gif2anim.py --from-module tools.sprites.walker --size small \\
        --name walker-small --out DIR
"""

from __future__ import annotations

import argparse
import functools
import importlib
import sys
from pathlib import Path

from PIL import Image, ImageSequence

_ROOT = Path(__file__).resolve().parents[1]
# the repository root as well as the package: --from-module imports tools.sprites.*, which is
# not importable when this script is run by path rather than as a module
sys.path[:0] = [str(_ROOT), str(_ROOT / "source" / "isaaclab")]

from isaaclab.app.anims import Animation, pack  # noqa: E402

SIZES = {"small": (24, 12), "wide": (64, 12)}
"""Target sizes in character cells, keyed by the loading screen slot they fill."""

SPRITE_PACKAGE = "tools.sprites"
"""Procedural sprites must live here, so ``--from-module`` cannot import arbitrary code."""

MAX_FRAMES = 24
"""Frames kept from a source unless ``--frames`` says otherwise."""

QUAD = {
    0b0000: " ", 0b0001: "▘", 0b0010: "▝", 0b0011: "▀", 0b0100: "▖", 0b0101: "▌",
    0b0110: "▞", 0b0111: "▛", 0b1000: "▗", 0b1001: "▚", 0b1010: "▐", 0b1011: "▜",
    0b1100: "▄", 0b1101: "▙", 0b1110: "▟", 0b1111: "█",
}
"""Quadrant blocks indexed by a 2x2 occupancy mask, bit 0 top-left through bit 3 bottom-right."""


def _mean(colours):
    """Average of a non-empty list of colours."""
    n = len(colours)
    return tuple(sum(c[i] for c in colours) // n for i in range(3))


def _error(quad, mask, fg, bg) -> int:
    """Squared RGB error of reproducing *quad* with *mask* split into *fg* and *bg*."""
    return sum(
        sum((a - b) ** 2 for a, b in zip(px, fg if mask >> i & 1 else bg)) for i, px in enumerate(quad)
    )


@functools.cache
def _encode(quad) -> tuple[str, tuple | None, tuple | None]:
    """Choose the glyph and colours that best reproduce one 2x2 block.

    Returns:
        The glyph, its foreground colour, and its background colour. The background is None
        where the block is partly transparent, so the greeting composites over the terminal
        instead of painting a box around itself.
    """
    if all(p is None for p in quad):
        return " ", None, None
    if any(p is None for p in quad):
        inked = [p for p in quad if p is not None]
        mask = sum(1 << i for i, p in enumerate(quad) if p is not None)
        return QUAD[mask], _mean(inked), None
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


def render(pixels: list[list[tuple | None]], cols: int, rows: int) -> str:
    """Pack a subpixel grid into styled quadrant rows.

    A colour is emitted only when it differs from the previous cell's. Repeating it for every
    cell roughly doubles the byte count and buys nothing, which matters when the frames ship
    in the wheel.

    Args:
        pixels: ``rows * 2`` rows of ``cols * 2`` RGB tuples, None where transparent.
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
            glyph, fg, bg = _encode(quad)
            style = (fg, bg)
            if style != current:
                line += "\x1b[0m"
                if fg is not None:
                    line += f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m"
                if bg is not None:
                    line += f"\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m"
                current = style
            line += glyph
        lines.append(line + "\x1b[0m")
    return "\n".join(lines)


def _sample(image: Image.Image, cols: int, rows: int, alpha_threshold: int) -> list[list[tuple | None]]:
    """Resample one image onto the subpixel grid.

    A quadrant subpixel is half a cell wide but a whole cell tall, so the grid is twice as
    wide as it is tall for a given area. Feeding a square source straight in would stretch it
    vertically; the caller is expected to have shaped the source, and this only fills the grid.
    """
    resized = image.convert("RGBA").resize((cols * 2, rows * 2), Image.LANCZOS)
    grid = []
    for y in range(rows * 2):
        line = []
        for x in range(cols * 2):
            r, g, b, a = resized.getpixel((x, y))
            line.append(None if a < alpha_threshold else (r, g, b))
        grid.append(line)
    return grid


def _resample(items: list, count: int) -> list:
    """Pick *count* evenly spaced items, so a long source becomes a short cycle."""
    if count >= len(items):
        return items
    return [items[round(i * len(items) / count)] for i in range(count)]


def convert(source: Path, size: str, *, name: str, frames: int | None = None, alpha_threshold: int = 128) -> Animation:
    """Build a greeting from an image or GIF.

    Args:
        source: Image file. Multi-frame files contribute one frame each.
        size: A key of :data:`SIZES`.
        name: Name to store in the container.
        frames: Keep this many frames, evenly spaced. Defaults to the source's own count,
            capped at :data:`MAX_FRAMES`.
        alpha_threshold: Alpha below this counts as transparent.

    Returns:
        The greeting.
    """
    cols, rows = SIZES[size]
    with Image.open(source) as image:
        sources = [frame.copy() for frame in ImageSequence.Iterator(image)]
        wanted = image.width / image.height
    target = (cols * 2) / (rows * 2) / 2  # subpixels are 1:2, so the grid's visual aspect halves
    if abs(wanted - target) / target > 0.10:
        print(
            f"warning: {source.name} is {wanted:.2f}:1 but the {size} slot is {target:.2f}:1;"
            " the result will be distorted. Reshape the source rather than the output.",
            file=sys.stderr,
        )
    kept = _resample(sources, frames or min(len(sources), MAX_FRAMES))
    return Animation(name, cols, rows, tuple(render(_sample(f, cols, rows, alpha_threshold), cols, rows) for f in kept))


def from_module(dotted: str, size: str, *, name: str, frames: int | None = None) -> Animation:
    """Build a greeting from a procedural sprite module.

    The module must expose ``frames(cols, rows)`` returning rendered frames at that size.
    Drawing straight onto the target grid avoids resampling art that was authored for it.

    Args:
        dotted: Import path, which must sit inside :data:`SPRITE_PACKAGE`.
        size: A key of :data:`SIZES`.
        name: Name to store in the container.
        frames: Keep this many frames, evenly spaced.

    Returns:
        The greeting.

    Raises:
        ValueError: If the module is outside :data:`SPRITE_PACKAGE`. This is a build-time
            tool, but importing an arbitrary dotted path on request is still worth refusing.
    """
    if dotted != SPRITE_PACKAGE and not dotted.startswith(f"{SPRITE_PACKAGE}."):
        raise ValueError(f"{dotted!r} is outside {SPRITE_PACKAGE}; sprites must live there")
    cols, rows = SIZES[size]
    module = importlib.import_module(dotted)
    rendered = list(module.frames(cols, rows))
    return Animation(name, cols, rows, tuple(_resample(rendered, frames or min(len(rendered), MAX_FRAMES))))


def main(argv: list[str] | None = None) -> int:
    """Convert a source into a greeting and write it out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", type=Path, help="image or GIF to convert")
    parser.add_argument("--from-module", dest="module", help=f"a sprite module under {SPRITE_PACKAGE}")
    parser.add_argument("--size", choices=sorted(SIZES), required=True)
    parser.add_argument("--name", required=True, help="name stored in the container")
    parser.add_argument("--out", type=Path, required=True, help="directory to write <name>.anim into")
    parser.add_argument("--frames", type=int, help="keep this many frames, evenly spaced")
    parser.add_argument("--alpha-threshold", type=int, default=128)
    args = parser.parse_args(argv)

    if bool(args.source) == bool(args.module):
        parser.error("give either a source image or --from-module, not both")
    if args.module:
        animation = from_module(args.module, args.size, name=args.name, frames=args.frames)
    else:
        animation = convert(
            args.source, args.size, name=args.name, frames=args.frames, alpha_threshold=args.alpha_threshold
        )

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.name}.anim"
    destination.write_bytes(pack(animation))
    print(f"{destination}: {len(animation.frames)} frame(s), {animation.cols}x{animation.rows}, {destination.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
