# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Greetings shown beside the run summary while a run starts up.

Each greeting is a sequence of pre-rendered terminal frames -- a single frame for a still
one -- stored in a compressed container beside this module. Rendering happens at authoring
time, so nothing here decodes an image or draws anything: startup only decompresses a few
kilobytes and hands the frames to the loading screen.

The container is a JSON header line followed by the frames, separated by NUL bytes and
gzipped as a whole. NUL cannot occur in terminal output, so frames need no escaping and
their colour sequences survive the round trip untouched.

Build more with ``tools/gif2anim.py``. The sizes are fixed by the space the loading screen
has beside its summary box; see :data:`SLOTS`.
"""

from __future__ import annotations

import functools
import gzip
import json
import random
from importlib import resources
from typing import NamedTuple

VERSION = 1
"""Container format version. :func:`unpack` refuses anything newer than it understands."""

SLOTS = ((24, 12), (64, 12))
"""Greeting sizes in character cells, narrow first.

The widths are what the loading screen leaves beside the summary box at its two display
widths; the height is shared so either size can stand in for the other in a pinch.
"""

_SEPARATOR = b"\x00"
"""Frame separator. Chosen because terminal output can never contain it."""


class Animation(NamedTuple):
    """A greeting: its name, its size in cells, and one or more rendered frames."""

    name: str
    cols: int
    rows: int
    frames: tuple[str, ...]


def pack(animation: Animation) -> bytes:
    """Serialise a greeting into its compressed container form.

    Args:
        animation: The greeting to store.

    Returns:
        The container bytes, ready to write to a ``.anim`` file.
    """
    header = {
        "name": animation.name,
        "cols": animation.cols,
        "rows": animation.rows,
        "frames": len(animation.frames),
        "version": VERSION,
    }
    body = _SEPARATOR.join(frame.encode() for frame in animation.frames)
    # mtime=0 keeps this deterministic: gzip stamps the current time into its header by
    # default, so packing unchanged art would produce different bytes every time and show up
    # as a spurious diff on every regeneration
    return gzip.compress(json.dumps(header, separators=(",", ":")).encode() + b"\n" + body, 9, mtime=0)


def unpack(blob: bytes) -> Animation:
    """Read a greeting back from its container form.

    Args:
        blob: Container bytes, as produced by :func:`pack`.

    Returns:
        The greeting.

    Raises:
        ValueError: If the container is newer than this format version, or if its header
            disagrees with its contents.
    """
    header, _, body = gzip.decompress(blob).partition(b"\n")
    meta = json.loads(header)
    if meta.get("version", 0) > VERSION:
        raise ValueError(f"greeting {meta.get('name')!r} uses container version {meta['version']}, expected {VERSION}")
    frames = tuple(frame.decode() for frame in body.split(_SEPARATOR))
    if len(frames) != meta["frames"]:
        raise ValueError(f"greeting {meta['name']!r} declares {meta['frames']} frames but holds {len(frames)}")
    return Animation(meta["name"], meta["cols"], meta["rows"], frames)


def available() -> tuple[str, ...]:
    """Names of the greetings shipped with Isaac Lab, in a stable order.

    Sorted rather than left in directory order, so a run's choice depends only on its random
    seed and not on the filesystem it happens to be installed on.
    """
    directory = resources.files(__package__) / "anims"
    if not directory.is_dir():
        return ()
    return tuple(sorted(path.name.removesuffix(".anim") for path in directory.iterdir() if path.name.endswith(".anim")))


@functools.cache
def load(name: str) -> Animation:
    """Read a shipped greeting by name.

    Cached, since a run uses at most one greeting per size and the loading screen asks for
    the current frame on every refresh.

    Args:
        name: A name from :func:`available`.

    Returns:
        The greeting.

    Raises:
        KeyError: If no greeting of that name is shipped.
    """
    path = resources.files(__package__) / "anims" / f"{name}.anim"
    if not path.is_file():
        raise KeyError(f"no greeting named {name!r}; available: {', '.join(available()) or 'none'}")
    return unpack(path.read_bytes())


def choose(rng: random.Random | None = None) -> tuple[Animation, ...]:
    """Pick one greeting for each size.

    Args:
        rng: Generator to draw with, so callers can pin the choice. Defaults to the module
            level generator, which gives a different greeting each run.

    Returns:
        One greeting per entry in :data:`SLOTS`, narrowest first, ready to hand to the
        loading screen in the order it expects.

    Raises:
        LookupError: If any size has no greeting to pick from. Returning a partial set would
            leave one display width with nothing to show.
    """
    rng = rng or random.Random()
    by_slot: dict[tuple[int, int], list[Animation]] = {slot: [] for slot in SLOTS}
    for name in available():
        animation = load(name)
        by_slot.get((animation.cols, animation.rows), []).append(animation)
    missing = [slot for slot, found in by_slot.items() if not found]
    if missing:
        raise LookupError(f"no greeting shipped for size(s) {missing}")
    return tuple(rng.choice(by_slot[slot]) for slot in SLOTS)
