# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ASCII terminal visualizer."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


@configclass
class AsciiVisualizerCfg(VisualizerCfg):
    """Configuration for the lightweight ASCII terminal visualizer."""

    visualizer_type: str = "ascii"
    """Type identifier for the ASCII visualizer."""

    width: int = 140
    """Maximum frame width in terminal character cells."""

    height: int = 30
    """Maximum frame height in terminal character cells."""

    max_fps: float = 12.0
    """Maximum terminal refresh rate in Hz."""

    env_index: int = 0
    """Environment index to display."""

    view_span: float = 6.0
    """Horizontal width of the orthographic view in meters when :attr:`auto_fit` is disabled."""

    auto_fit: bool = True
    """Automatically frame the selected environment's dynamic geometry."""

    auto_fit_margin: float = 1.4
    """Multiplicative margin around automatically framed geometry."""

    max_faces_per_body: int = 96
    """Maximum triangle count retained for each body's terminal-rendering mesh."""

    body_axis_length: float = 1.0
    """Length of fallback body-orientation axes in meters when USD geometry is unavailable."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.width < 20:
            raise ValueError(f"AsciiVisualizerCfg.width must be at least 20, got {self.width}.")
        if self.height < 8:
            raise ValueError(f"AsciiVisualizerCfg.height must be at least 8, got {self.height}.")
        if self.max_fps <= 0.0:
            raise ValueError(f"AsciiVisualizerCfg.max_fps must be positive, got {self.max_fps}.")
        if self.env_index < 0:
            raise ValueError(f"AsciiVisualizerCfg.env_index must be non-negative, got {self.env_index}.")
        if self.view_span <= 0.0:
            raise ValueError(f"AsciiVisualizerCfg.view_span must be positive, got {self.view_span}.")
        if self.auto_fit_margin <= 1.0:
            raise ValueError(f"AsciiVisualizerCfg.auto_fit_margin must be greater than 1, got {self.auto_fit_margin}.")
        if self.max_faces_per_body < 12:
            raise ValueError(
                f"AsciiVisualizerCfg.max_faces_per_body must be at least 12, got {self.max_faces_per_body}."
            )
        if self.body_axis_length <= 0.0:
            raise ValueError(f"AsciiVisualizerCfg.body_axis_length must be positive, got {self.body_axis_length}.")
