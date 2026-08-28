# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms MicroDuck needs that have no stock Isaac Lab counterpart.

Ported from ``pollen-robotics/microduck_rl``; see section 6 of
``artifacts/microduck/upstream_reference.md`` for the verbatim upstream formula.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def robot_state_is_nan(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str] = (),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate environments whose robot state has stopped being finite.

    Ported from reference section 6 (``robot_state_is_nan``). A solver that diverges leaves NaN or
    infinite values in the state, which then propagate into the observations and poison the policy
    gradient for the whole batch; terminating resets the environment instead. Despite its name the
    check is for non-finiteness, so infinities are caught as well.

    This is not a time-out: configure it with ``time_out=False`` so the value bootstraps as a real
    failure.

    There is no stock counterpart. :func:`isaaclab.envs.mdp.root_height_below_minimum` and
    :func:`~isaaclab.envs.mdp.bad_orientation` catch a fallen robot, not a broken one, and a NaN
    height compares false against every bound rather than tripping them.

    Args:
        env: The environment instance.
        sensor_names: Names of contact sensors whose forces are checked as well. Names the scene
            does not carry are ignored, as upstream does. Defaults to an empty tuple.
        asset_cfg: The articulation to check. Defaults to the entity named ``"robot"``.

    Returns:
        Whether each environment must terminate. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    is_broken = ~torch.isfinite(data.joint_pos.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.joint_vel.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.root_link_pos_w.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.root_link_quat_w.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.root_link_lin_vel_w.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.root_link_ang_vel_w.torch).all(dim=1)

    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        net_forces_w = env.scene.sensors[name].data.net_forces_w
        if net_forces_w is None:
            continue
        is_broken |= ~torch.isfinite(net_forces_w.torch).flatten(start_dim=1).all(dim=1)
    return is_broken
