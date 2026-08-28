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


def randomize_bam_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] = (0.9, 1.1),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize the friction budget of an articulation's BAM servo groups.

    Draws one ``U(scale_range)`` multiplier per environment and applies it to the whole
    velocity-independent friction budget -- the Coulomb, Stribeck and load-dependent terms
    together -- of every :class:`~isaaclab.actuators.BamActuatorCfg` group on the asset. Upstream
    runs this on **reset** (reference section 2.6, ``randomize_joint_friction``), which makes a
    worn or freshly greased gearbox an episode-scale disturbance rather than a fixed property.

    The stock :func:`~isaaclab.envs.mdp.randomize_actuator_gains` does not cover this model: its
    filter admits implicit, ``IdealPDActuator`` and Newton-native groups only, and its explicit
    branch writes ``stiffness``/``damping``, which BAM never reads.

    Both execution paths are covered, and the group is discovered from the *configuration* rather
    than from the runtime object, because only the configuration is the same on both. With
    ``use_newton_actuators=False`` the mapping entry is a
    :class:`~isaaclab.actuators.BamActuator` and the scale is written through its public
    randomization hook; with ``use_newton_actuators=True`` it is a Newton actuator object holding
    the Warp-side controller, whose ``friction_scale`` carries the same name and meaning and is
    reached through :func:`~isaaclab.actuators.newton.write_group_parameter`.

    The scale is one number per environment, matching the reference implementation and
    :attr:`~isaaclab.actuators.BamActuator.friction_scale`, so ``asset_cfg`` selects the
    articulation only -- its joint selection is not used.

    Args:
        env: The environment holding the articulation.
        env_ids: The environments to resample. Defaults to None, which resamples all of them.
        scale_range: The ``(low, high)`` bounds [-] of the friction-budget multiplier.
        asset_cfg: The articulation whose BAM groups are randomized. Only its name is used.
    """
    from isaaclab.actuators import BamActuator, BamActuatorCfg  # noqa: PLC0415
    from isaaclab.actuators.newton import read_group_parameter, write_group_parameter  # noqa: PLC0415

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for name, actuator_cfg in asset.cfg.actuators.items():
        if not isinstance(actuator_cfg, BamActuatorCfg):
            continue
        scales = torch.empty(len(env_ids), 1, device=env.device).uniform_(*scale_range)
        actuator = asset.actuators[name]
        if isinstance(actuator, BamActuator):
            actuator.set_friction_scale(env_ids, scales)
        else:
            # the native controller stores the scale per driven joint, so the per-environment
            # draw is broadcast across the group's columns
            num_group_joints = read_group_parameter(asset.actuators, name, "controller", "friction_scale").shape[1]
            write_group_parameter(
                asset.actuators,
                name,
                "controller",
                "friction_scale",
                values=scales.expand(len(env_ids), num_group_joints),
                env_ids=env_ids,
            )
