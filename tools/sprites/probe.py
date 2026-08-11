# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A trivial sprite: a bar sweeping left to right. Exists to exercise the tooling."""

from __future__ import annotations

FRAMES = 8
"""Frames in the cycle."""


def frames(cols: int, rows: int) -> list[str]:
    """Render the cycle at *cols* x *rows* cells."""
    out = []
    for step in range(FRAMES):
        head = round(step * cols / FRAMES)
        lines = []
        for _ in range(rows):
            line = "".join(
                f"\x1b[38;2;118;185;0m█" if col == head else "\x1b[0m " for col in range(cols)
            )
            lines.append(line + "\x1b[0m")
        out.append("\n".join(lines))
    return out
