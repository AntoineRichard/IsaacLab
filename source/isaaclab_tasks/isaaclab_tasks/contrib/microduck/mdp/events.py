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
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from collections.abc import Mapping

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

_ENCODER_BIAS_ATTR = "_microduck_encoder_bias"
"""Attribute the per-environment encoder bias is cached under on the environment."""

_GROUND_STATE_JOINT_IDS_ATTR = "_microduck_ground_state_joint_ids"
"""Attribute the resolved sitting-pose joint indices are cached under on the environment."""


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


def _sitting_joint_ids(env: ManagerBasedEnv, asset: Articulation, joint_names: tuple[str, ...]) -> list[int]:
    """Resolve, and cache on the environment, the indices of the joints the sitting pose overrides.

    A reset event runs every episode, so the name resolution is done once per articulation and
    selection rather than per reset. The cache mirrors upstream's own memoized servo-index lookup.
    """
    cache: dict[tuple[int, tuple[str, ...]], list[int]] = env.__dict__.setdefault(_GROUND_STATE_JOINT_IDS_ATTR, {})
    key = (id(asset), joint_names)
    joint_ids = cache.get(key)
    if joint_ids is None:
        joint_ids, _ = asset.find_joints(list(joint_names), preserve_order=True)
        cache[key] = joint_ids
    return joint_ids


def reset_ground_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    face_down_prob: float,
    face_up_prob: float,
    sitting_prob: float,
    standing_prob: float,
    prone_z_range: tuple[float, float],
    sitting_z_range: tuple[float, float],
    standing_z_range: tuple[float, float],
    sitting_joint_pos: Mapping[str, float],
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
    face_up_roll_max: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset an articulation into one of four ground poses: face-down, face-up, sitting or standing.

    Ported from addendum section 3.4 (``set_random_ground_state``). This single event *is* the
    stand-up task's episode distribution: which pose an episode starts from decides which skill it
    exercises, and a curriculum ramps the four probabilities from the easy mix (mostly standing and
    sitting) to the hard one (mostly prone) as the run progresses. There is no stock counterpart --
    :func:`isaaclab.envs.mdp.reset_root_state_uniform` samples one continuous band, not a mixture of
    named keyframes with a matching joint pose.

    The four buckets:

    * **face-down** -- the trunk pitched +90 degrees, belly to the floor, at a random yaw.
    * **face-up** -- pitched -90 degrees, back to the floor, then rolled by up to
      :attr:`face_up_roll_max` about the body's long axis. That partial roll is a reverse
      curriculum: recovering from flat on the back has no reward gradient until the roll completes,
      so some episodes start part-way along it.
    * **sitting** -- upright within :attr:`sitting_tilt_max`, low, with the legs folded into
      :attr:`sitting_joint_pos`. This is the hand-off pose a sit policy leaves the robot in, so it
      is sampled with joint noise rather than exactly: a policy trained on the exact keyframe does
      not survive the real hand-off.
    * **standing** -- the same orientation sampler as sitting but at standing height and with the
      joints left at the stand pose, so the policy also learns to *hold* a stand.

    The probabilities are normalized and need not sum to one. Only the sitting bucket touches the
    joints; the other three keep whatever the joint reset wrote.

    This term overwrites the root height and orientation, so it must run **after** any root reset it
    shares an episode with, and that reset's horizontal spread is what survives. Isaac Lab fires
    reset events in configuration declaration order.

    Args:
        env: The environment holding the articulation.
        env_ids: The environments to reset. Defaults to None, which resets all of them.
        face_down_prob: Relative probability of the face-down bucket.
        face_up_prob: Relative probability of the face-up bucket.
        sitting_prob: Relative probability of the sitting bucket.
        standing_prob: Relative probability of the standing bucket.
        prone_z_range: Trunk height band [m] the two prone buckets spawn in, above the environment
            origin.
        sitting_z_range: Trunk height band [m] the sitting bucket spawns in.
        standing_z_range: Trunk height band [m] the standing bucket spawns in.
        sitting_joint_pos: Joint name to angle [rad] the sitting bucket folds its legs into. Joints
            left out of it stay at the stand pose, which is how the neck and head are handled.
        sitting_joint_noise_std: Standard deviation [rad] of the zero-mean noise added to *every*
            joint selected by :attr:`asset_cfg` in the sitting bucket, on top of the overrides.
            Defaults to 0.0, which writes the keyframe exactly.
        sitting_tilt_max: Bound [rad] on the uniform pitch and roll the sitting and standing buckets
            are tilted by. Defaults to 0.0, which spawns them exactly upright.
        face_up_roll_max: Bound [rad] on the uniform roll about the body long axis applied to the
            face-up bucket. Defaults to 0.0, which spawns it flat on its back.
        asset_cfg: The articulation to reset, and the joints the sitting noise reaches. Selecting
            them by name reproduces upstream's exclusion of passive joints, which are not part of
            any keyframe.

    Raises:
        ValueError: If the four probabilities do not sum to a positive number.
    """
    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    if total <= 0.0:
        raise ValueError(
            "'reset_ground_state' needs at least one bucket with a positive probability. Received"
            f" face_down={face_down_prob}, face_up={face_up_prob}, sitting={sitting_prob},"
            f" standing={standing_prob}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    # Bucket assignment from one uniform draw, over the normalized cumulative probabilities.
    # Face-down needs no mask of its own: it is the first bucket, so it is what the orientation and
    # the height fall back to when none of the other three masks select an environment.
    bucket = torch.rand(num_resets, device=env.device) * total
    is_face_up = (bucket >= face_down_prob) & (bucket < face_down_prob + face_up_prob)
    is_sitting = (bucket >= face_down_prob + face_up_prob) & (bucket < face_down_prob + face_up_prob + sitting_prob)
    is_standing = bucket >= face_down_prob + face_up_prob + sitting_prob

    # orientation. The yaw is shared by all four buckets, so an episode's heading does not correlate
    # with the pose it starts from.
    yaw = (torch.rand(num_resets, device=env.device) * 2.0 - 1.0) * torch.pi
    zeros = torch.zeros_like(yaw)
    half_pi = torch.full_like(yaw, torch.pi / 2.0)
    face_down_quat = quat_from_euler_xyz(zeros, half_pi, yaw)
    face_up_quat = quat_from_euler_xyz(zeros, -half_pi, yaw)
    if face_up_roll_max > 0.0:
        roll_angle = (torch.rand(num_resets, device=env.device) * 2.0 - 1.0) * face_up_roll_max
        long_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(num_resets, 3)
        # right-multiplied, so the roll is about the body's own long axis rather than the world's
        face_up_quat = quat_mul(face_up_quat, quat_from_angle_axis(roll_angle, long_axis))
    if sitting_tilt_max > 0.0:
        tilt = (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * sitting_tilt_max
        upright_quat = quat_from_euler_xyz(tilt[:, 1], tilt[:, 0], yaw)
    else:
        upright_quat = quat_from_euler_xyz(zeros, zeros, yaw)

    quat = torch.where(is_face_up.unsqueeze(-1), face_up_quat, face_down_quat)
    quat = torch.where((is_sitting | is_standing).unsqueeze(-1), upright_quat, quat)

    # height, drawn per bucket
    def _uniform(bounds: tuple[float, float]) -> torch.Tensor:
        return torch.empty(num_resets, device=env.device).uniform_(*bounds)

    height = _uniform(prone_z_range)
    height = torch.where(is_sitting, _uniform(sitting_z_range), height)
    height = torch.where(is_standing, _uniform(standing_z_range), height)

    # the horizontal position is left alone: it is what an earlier root reset spread out
    pose = torch.cat(
        (asset.data.root_link_pos_w.torch[env_ids], asset.data.root_link_quat_w.torch[env_ids]), dim=-1
    ).clone()
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + height
    pose[:, 3:7] = quat
    asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_com_velocity_to_sim_index(
        root_velocity=torch.zeros(num_resets, 6, device=env.device), env_ids=env_ids
    )

    sitting_env_ids = env_ids[is_sitting]
    if len(sitting_env_ids) == 0:
        return
    joint_pos = asset.data.default_joint_pos.torch[sitting_env_ids].clone()
    override_ids = _sitting_joint_ids(env, asset, tuple(sitting_joint_pos))
    angles = torch.tensor(list(sitting_joint_pos.values()), device=env.device)
    joint_pos[:, override_ids] = angles
    if sitting_joint_noise_std > 0.0:
        noise = torch.randn(len(sitting_env_ids), len(joint_pos[0, asset_cfg.joint_ids]), device=env.device)
        joint_pos[:, asset_cfg.joint_ids] += noise * sitting_joint_noise_std
    asset.write_joint_state_to_sim_index(
        position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=sitting_env_ids
    )
