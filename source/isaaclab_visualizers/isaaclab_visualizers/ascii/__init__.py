# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ASCII terminal visualizer backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ascii_visualizer_cfg import AsciiVisualizerCfg

if TYPE_CHECKING:
    from .ascii_visualizer import AsciiVisualizer

__all__ = ["AsciiVisualizer", "AsciiVisualizerCfg"]


def __getattr__(name: str):
    if name == "AsciiVisualizer":
        from .ascii_visualizer import AsciiVisualizer

        return AsciiVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
