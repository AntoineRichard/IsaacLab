# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms MicroDuck needs that have no stock Isaac Lab counterpart.

This module also owns the encoder-bias storage the biased observation and the biased action term
share; see :func:`encoder_bias`. Ported from ``pollen-robotics/microduck_rl``, whose upstream
formulas are quoted in sections 2.6 and 5 of ``artifacts/microduck/upstream_reference.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

_ENCODER_BIAS_ATTR = "_microduck_encoder_bias"
"""Attribute the per-environment encoder bias is cached under on the environment."""


def encoder_bias(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The per-environment, per-joint encoder bias [rad] of an articulation.

    The bias models a joint-encoder calibration error: a constant offset between the joint position
    the servo reports and the one the simulation holds. It is a single quantity read by three
    parties, so it cannot live in any one of them:

    * :func:`~isaaclab_tasks.contrib.microduck.mdp.observations.joint_pos_rel_biased` **adds** it,
      because that is what the encoders report to the policy;
    * :class:`~isaaclab_tasks.contrib.microduck.mdp.actions.BiasedJointPositionAction`
      **subtracts** it, because the servo closes its position loop on the same biased reading;
    * :func:`randomize_encoder_bias` samples it.

    Upstream stores the tensor on the articulation itself (``asset.data.encoder_bias``, reference
    section 5); Isaac Lab's articulation data has no such field, so it is cached on the environment
    instead, keyed by asset name. The tensor is allocated on first use and only ever written
    in place, so a reader may resolve it at any time -- including before the startup randomization
    runs, which is when the observation manager first probes the observation dimensions.

    A task that never wires :func:`randomize_encoder_bias` therefore reads an all-zero bias, and
    the biased terms degenerate to their unbiased counterparts.

    Args:
        env: The environment holding the articulation.
        asset_cfg: The articulation the bias belongs to. Only its name is used; the returned tensor
            spans every joint of the articulation, not the selection.

    Returns:
        The encoder bias [rad]. Shape is (num_envs, num_joints).
    """
    store: dict[str, torch.Tensor] | None = getattr(env, _ENCODER_BIAS_ATTR, None)
    if store is None:
        store = {}
        setattr(env, _ENCODER_BIAS_ATTR, store)
    bias = store.get(asset_cfg.name)
    if bias is None:
        asset: Articulation = env.scene[asset_cfg.name]
        bias = torch.zeros(env.num_envs, asset.num_joints, device=env.device)
        store[asset_cfg.name] = bias
    return bias


def randomize_encoder_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize the joint-encoder calibration error of an articulation.

    Draws an independent ``U(bias_range)`` offset per environment and per selected joint. Upstream
    runs this on **startup** (reference section 2.6), which makes the bias a fixed property of each
    simulated robot rather than a per-episode disturbance; running it on reset would let the policy
    average the bias out over an episode instead of learning to tolerate it.

    Args:
        env: The environment holding the articulation.
        env_ids: The environments to resample. Defaults to None, which resamples all of them.
        bias_range: The ``(low, high)`` bounds [rad] of the bias.
        asset_cfg: The articulation and the joints to bias.
    """
    bias = encoder_bias(env, asset_cfg)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    num_joints = bias.shape[1] if isinstance(joint_ids, slice) else len(joint_ids)
    samples = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(*bias_range)
    if isinstance(joint_ids, slice):
        bias[env_ids] = samples
    else:
        bias[env_ids[:, None], torch.as_tensor(joint_ids, device=env.device)] = samples
