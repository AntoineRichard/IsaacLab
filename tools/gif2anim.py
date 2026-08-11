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
import importlib
import sys
from pathlib import Path

from PIL import Image, ImageSequence

_ROOT = Path(__file__).resolve().parents[1]
# the repository root as well as the package: --from-module imports tools.sprites.*, which is
# not importable when this script is run by path rather than as a module
sys.path[:0] = [str(_ROOT), str(_ROOT / "source" / "isaaclab")]

from isaaclab.app.anims import SLOTS, Animation, pack  # noqa: E402

from tools.sprites.canvas import render  # noqa: E402

SIZES = dict(zip(("small", "wide"), SLOTS))
"""Target sizes in character cells, keyed by the loading screen slot they fill."""

SPRITE_PACKAGE = "tools.sprites"
"""Procedural sprites must live here, so ``--from-module`` cannot import arbitrary code."""

MAX_FRAMES = 24
"""Frames kept from a source unless ``--frames`` says otherwise."""


def _sample(
    image: Image.Image, cols: int, rows: int, alpha_threshold: int, *, fit: bool = True
) -> list[list[tuple | None]]:
    """Resample one image onto the subpixel grid.

    A quadrant subpixel is half a cell wide but a whole cell tall, so it displays twice as
    tall as it is wide. Any sizing has to account for that or the result comes out stretched
    vertically -- the single most common way terminal art goes wrong.

    Args:
        image: Source frame.
        cols: Output width in cells.
        rows: Output height in cells.
        alpha_threshold: Alpha below this counts as transparent.
        fit: Scale to fit and pad the remainder with transparency, keeping the source's
            proportions. When False the source is stretched to fill the grid instead.

    Returns:
        ``rows * 2`` rows of ``cols * 2`` RGB tuples, None where transparent.
    """
    source = image.convert("RGBA")
    width, height = cols * 2, rows * 2
    left = top = 0
    if fit:
        # the grid displays `width` units across and `height * 2` down, so a source of aspect
        # w:h wants `2 * sh * w / h` subpixels across for `sh` down
        drawn_h = height
        drawn_w = round(2 * drawn_h * source.width / source.height)
        if drawn_w > width:
            drawn_w = width
            drawn_h = round(drawn_w * source.height / (2 * source.width))
        left, top = (width - drawn_w) // 2, (height - drawn_h) // 2
        width, height = max(drawn_w, 1), max(drawn_h, 1)

    resized = source.resize((width, height), Image.LANCZOS)
    grid = [[None] * (cols * 2) for _ in range(rows * 2)]
    for y in range(height):
        for x in range(width):
            r, g, b, a = resized.getpixel((x, y))
            if a >= alpha_threshold:
                grid[top + y][left + x] = (r, g, b)
    return grid


def _resample(items: list, count: int) -> list:
    """Pick *count* evenly spaced items, so a long source becomes a short cycle."""
    if count >= len(items):
        return items
    return [items[round(i * len(items) / count)] for i in range(count)]


def convert(
    source: Path,
    size: str,
    *,
    name: str,
    frames: int | None = None,
    alpha_threshold: int = 128,
    fit: bool = True,
) -> Animation:
    """Build a greeting from an image or GIF.

    Args:
        source: Image file. Multi-frame files contribute one frame each.
        size: A key of :data:`SIZES`.
        name: Name to store in the container.
        frames: Keep this many frames, evenly spaced. Defaults to the source's own count,
            capped at :data:`MAX_FRAMES`.
        alpha_threshold: Alpha below this counts as transparent.
        fit: Keep the source's proportions and pad the remainder with transparency. The slots
            are much wider than most logos, so stretching to fill distorts them badly; padding
            costs nothing because the padded columns are transparent and composite away.

    Returns:
        The greeting.
    """
    cols, rows = SIZES[size]
    with Image.open(source) as image:
        sources = [frame.copy() for frame in ImageSequence.Iterator(image)]
    kept = _resample(sources, frames or min(len(sources), MAX_FRAMES))
    return Animation(
        name,
        cols,
        rows,
        tuple(render(_sample(f, cols, rows, alpha_threshold, fit=fit), cols, rows) for f in kept),
    )


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
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="fill the grid instead of padding, distorting the source to match the slot",
    )
    args = parser.parse_args(argv)

    if bool(args.source) == bool(args.module):
        parser.error("give either a source image or --from-module, not both")
    if args.module:
        animation = from_module(args.module, args.size, name=args.name, frames=args.frames)
    else:
        animation = convert(
            args.source,
            args.size,
            name=args.name,
            frames=args.frames,
            alpha_threshold=args.alpha_threshold,
            fit=not args.stretch,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.name}.anim"
    destination.write_bytes(pack(animation))
    print(
        f"{destination}: {len(animation.frames)} frame(s), "
        f"{animation.cols}x{animation.rows}, {destination.stat().st_size} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
