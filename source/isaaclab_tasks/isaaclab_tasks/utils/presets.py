# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for inspecting named physics presets on raw env configs."""

from __future__ import annotations

__all__ = ["has_physics_preset", "MultiBackendRendererCfg"]

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class MultiBackendRendererCfg(PresetCfg):
    default: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    ovrtx_renderer: OVRTXRendererCfg = OVRTXRendererCfg()
    isaacsim_rtx_renderer = default


def has_physics_preset(raw_cfg, preset_name: str) -> bool:
    """Check if a raw (unresolved) env config has a named physics preset.

    Must be called with the result of
    :func:`~isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry`, not
    :func:`~isaaclab_tasks.utils.parse_cfg.parse_env_cfg`, because the latter
    resolves all ``PresetCfg`` wrappers to their default before returning.

    Args:
        raw_cfg: Raw env config from :func:`load_cfg_from_registry`.
        preset_name: Name of the preset to check for (e.g. ``"newton"``).

    Returns:
        ``True`` if ``raw_cfg.sim.physics`` is a ``PresetCfg`` wrapper that
        defines a field named ``preset_name``, ``False`` otherwise.
    """
    if isinstance(raw_cfg, dict):
        return False
    env_cfg = raw_cfg
    # If the top-level cfg is itself a PresetCfg wrapper, unwrap to its
    # default. A PresetCfg wrapper is a dataclass that has a ``default``
    # attribute and whose type does NOT declare ``class_type`` (which is
    # how an env-config dataclass is distinguished from a preset wrapper).
    if (
        hasattr(env_cfg, "__dataclass_fields__")
        and hasattr(env_cfg, "default")
        and not hasattr(type(env_cfg), "class_type")
    ):
        env_cfg = env_cfg.default
    physics = getattr(getattr(env_cfg, "sim", None), "physics", None)
    return physics is not None and hasattr(physics, preset_name)
