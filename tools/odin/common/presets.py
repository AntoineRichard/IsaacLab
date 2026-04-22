# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin-side re-export of the upstream :mod:`isaaclab_tasks.utils.presets`.

Kept as a separate module so that when Odin graduates to its own repo the
upstream dependency surface is visible at a glance: any replacement backend
just needs to provide a ``has_physics_preset`` with the same signature.
"""

from isaaclab_tasks.utils.presets import has_physics_preset

__all__ = ["has_physics_preset"]
