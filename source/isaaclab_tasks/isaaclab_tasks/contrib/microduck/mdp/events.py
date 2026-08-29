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

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from collections.abc import Mapping

    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv

_ENCODER_BIAS_ATTR = "_microduck_encoder_bias"
"""Attribute the per-environment encoder bias is cached under on the environment."""

_GROUND_STATE_JOINT_IDS_ATTR = "_microduck_ground_state_joint_ids"
"""Attribute the resolved sitting-pose joint indices are cached under on the environment."""

_ROULADE_STATE_ATTR = "_microduck_roulade_state"
"""Attribute the forward-roll bookkeeping is cached under on the environment."""


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


def _keyframe_joint_ids(env: ManagerBasedEnv, asset: Articulation, joint_names: tuple[str, ...]) -> list[int]:
    """Resolve, and cache on the environment, the indices of the joints a keyframe overrides.

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
    prone_z_range: tuple[float, float] | None = None,
    sitting_z_range: tuple[float, float] | None = None,
    standing_z_range: tuple[float, float] | None = None,
    sitting_joint_pos: Mapping[str, float] | None = None,
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
            origin. Defaults to None, which is only allowed when both prone buckets are disabled.
        sitting_z_range: Trunk height band [m] the sitting bucket spawns in. Defaults to None, which
            is only allowed when that bucket is disabled.
        standing_z_range: Trunk height band [m] the standing bucket spawns in. Defaults to None,
            which is only allowed when that bucket is disabled.
        sitting_joint_pos: Joint name to angle [rad] the sitting bucket folds its legs into. Joints
            left out of it stay at the stand pose, which is how the neck and head are handled.
            Defaults to None, which is only allowed when that bucket is disabled.
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
        ValueError: If the four probabilities do not sum to a positive number, or if a bucket with a
            positive probability is missing the parameters it spawns from.
    """
    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    if total <= 0.0:
        raise ValueError(
            "'reset_ground_state' needs at least one bucket with a positive probability. Received"
            f" face_down={face_down_prob}, face_up={face_up_prob}, sitting={sitting_prob},"
            f" standing={standing_prob}."
        )
    # A task that uses only some of the four buckets configures only their parameters: the ball-kick
    # task, for instance, is standing-only and has no seated keyframe to name. Leaving a *live*
    # bucket's parameters out is a configuration error rather than a default, so it is caught here
    # rather than sampled from a fallback band the task never chose.
    for probability, missing, names in (
        (face_down_prob + face_up_prob, prone_z_range is None, "prone_z_range"),
        (sitting_prob, sitting_z_range is None or sitting_joint_pos is None, "sitting_z_range/sitting_joint_pos"),
        (standing_prob, standing_z_range is None, "standing_z_range"),
    ):
        if probability > 0.0 and missing:
            raise ValueError(
                f"'reset_ground_state' samples a bucket with probability {probability} whose"
                f" '{names}' is not configured."
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

    # Height, drawn per bucket. An unconfigured band belongs to a bucket with zero probability --
    # the validation above is what guarantees that -- so the branch it fills selects nothing.
    def _uniform(bounds: tuple[float, float] | None) -> torch.Tensor:
        if bounds is None:
            return torch.zeros(num_resets, device=env.device)
        return torch.empty(num_resets, device=env.device).uniform_(*bounds)

    height = _uniform(prone_z_range)
    height = torch.where(is_sitting, _uniform(sitting_z_range), height)
    height = torch.where(is_standing, _uniform(standing_z_range), height)

    # The horizontal position is left alone: it is what an earlier root reset spread out. ``cat``
    # already allocates, so the assignments below cannot reach the articulation's own buffers.
    pose = torch.cat((asset.data.root_link_pos_w.torch[env_ids], asset.data.root_link_quat_w.torch[env_ids]), dim=-1)
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
    override_ids = _keyframe_joint_ids(env, asset, tuple(sitting_joint_pos))
    angles = torch.tensor(list(sitting_joint_pos.values()), device=env.device)
    joint_pos[:, override_ids] = angles
    if sitting_joint_noise_std > 0.0:
        noise = torch.randn(len(sitting_env_ids), len(joint_pos[0, asset_cfg.joint_ids]), device=env.device)
        joint_pos[:, asset_cfg.joint_ids] += noise * sitting_joint_noise_std
    asset.write_joint_state_to_sim_index(
        position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=sitting_env_ids
    )


class RouladeRollState:
    """Per-environment bookkeeping of the forward roll, shared by the roll's rewards and its reset.

    Ported from addendum section 4.3, where it lives as five loose attributes on the environment
    (``_roulade_accum``, ``_roulade_max``, ``_roulade_paid``, ``_roulade_head_latch`` and
    ``_roulade_last_update_step``). The roll is a *stateful* task: nothing in the instantaneous
    physics says how far around the robot has come, so the rotation is integrated by hand and the
    completion gates read the integral rather than a clock. Every consumer therefore has to agree on
    one object, which is why it is stored on the environment rather than inside a term.

    The four buffers mean different things and all four are load-bearing:

    * :attr:`accumulated_angle` is the running integral. It can go **down**, because rocking
      backwards integrates a negative rate, and it is what the head latch and the head-pivot reward
      window are measured against.
    * :attr:`frontier` is the maximum of that integral so far, which is what the completion gates
      read: backward rocking may not un-open a gate the robot has already earned.
    * :attr:`paid` is the part of the frontier :func:`~..rewards.roulade_progress` has already paid
      out, so the reward is an increment rather than a level.
    * :attr:`head_latch` records that the robot went over the flat top of its head, which the
      landing rewards require.

    :attr:`last_update_step` is the step guard: several rewards read the state in one control step
    and the integral must advance exactly once.
    """

    def __init__(self, num_envs: int, device: str) -> None:
        """Allocate the per-environment buffers.

        Args:
            num_envs: Number of environments.
            device: Device the buffers live on.
        """
        self.accumulated_angle = torch.zeros(num_envs, device=device)
        """Supported, sagittal-gated integral of the forward pitch rate [rad]. Shape is (num_envs,)."""
        self.frontier = torch.zeros(num_envs, device=device)
        """Largest :attr:`accumulated_angle` reached this episode [rad]. Shape is (num_envs,)."""
        self.paid = torch.zeros(num_envs, device=device)
        """Part of :attr:`frontier` already paid out as progress reward [rad]. Shape is (num_envs,)."""
        self.head_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        """Whether the roll went over the flat top of the head. Shape is (num_envs,)."""
        self.last_update_step = -1
        """Global step count the integral was last advanced on."""


def roulade_roll_state(env: ManagerBasedEnv) -> RouladeRollState:
    """The environment's forward-roll bookkeeping, allocated on first use.

    Args:
        env: The environment the roll is bookkept for.

    Returns:
        The shared state. The same object on every call for a given environment.
    """
    state: RouladeRollState | None = getattr(env, _ROULADE_STATE_ATTR, None)
    if state is None:
        state = RouladeRollState(env.num_envs, env.device)
        setattr(env, _ROULADE_STATE_ATTR, state)
    return state


def reset_roulade_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    standing_prob: float,
    midroll_prob: float,
    standing_z_range: tuple[float, float],
    midroll_z_range: tuple[float, float],
    midroll_pitch_range: tuple[float, float],
    standing_tilt_max: float = 0.0,
    forward_vel_range: tuple[float, float] = (0.0, 0.0),
    midroll_omega_range: tuple[float, float] = (0.0, 0.0),
    tuck_joint_pos: Mapping[str, float] | None = None,
    tuck_factor_range: tuple[float, float] = (0.3, 1.0),
    joint_noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset an articulation into a standing start or part-way through a forward roll.

    Ported from addendum section 4.5 (``reset_roulade_state``). The two buckets are the roll task's
    whole episode distribution, and the mid-roll one is a **reverse curriculum**: the second half of
    a forward roll -- supine, then sitting up, then standing -- is the face-up recovery problem the
    stand-up task already proves is learnable, so spawning into it gives the policy dense data on
    the part of the trick a standing start only reaches after it has already solved the flip.

    * **standing** -- upright within :attr:`standing_tilt_max`, at a random yaw, at standing height,
      with the joints left wherever the joint reset put them. :attr:`forward_vel_range` is
      upstream's "élan" hook: a forward base velocity that would let the roll be entered out of a
      walk. The roll task ships it at ``(0.0, 0.0)``, so the branch never runs.
    * **mid-roll** -- pitched :attr:`midroll_pitch_range` into the roll (90 degrees is balanced on
      the head, 180 on the back, 340 seated and leaning back), low, with the legs and the neck
      lerped toward :attr:`tuck_joint_pos` by a per-environment factor, and with optional forward
      angular momentum.

    This term also **writes the roll bookkeeping** :func:`roulade_roll_state`, and the two halves of
    that are as load-bearing as the pose (addendum section 4.5):

    * A mid-roll spawn has its rotation accumulator, frontier and paid pointer all pre-set to the
      **spawn pitch**, so a 170-degree spawn is only paid for the remaining 190 degrees and the
      completion gates stay consistent with how far around the robot actually is.
    * A mid-roll spawn is **granted the head latch**, because it never had the chance to earn one;
      requiring it would keep the landing gate shut for the whole bucket. A standing spawn starts at
      zero and must earn the latch by rolling over its head.

    The probabilities are normalized and need not sum to one. Only the mid-roll bucket touches the
    joints.

    This term overwrites the root height, orientation and velocity, so it must run **after** any
    root reset it shares an episode with, and **after** the joint reset whose pose the tuck lerps
    from. Isaac Lab fires reset events in configuration declaration order.

    Args:
        env: The environment holding the articulation.
        env_ids: The environments to reset. Defaults to None, which resets all of them.
        standing_prob: Relative probability of the standing bucket.
        midroll_prob: Relative probability of the mid-roll bucket.
        standing_z_range: Trunk height band [m] the standing bucket spawns in, above the environment
            origin.
        midroll_z_range: Trunk height band [m] the mid-roll bucket spawns in.
        midroll_pitch_range: Bounds [rad] on how far into the roll the mid-roll bucket spawns.
        standing_tilt_max: Bound [rad] on the uniform pitch the standing bucket is tilted by.
            Defaults to 0.0, which spawns it exactly upright. The roll is sampled independently and
            is at least 5 degrees wide, as upstream's is.
        forward_vel_range: Bounds [m/s] on the forward base velocity given to standing spawns.
            Defaults to ``(0.0, 0.0)``, which is a standstill start and skips the branch.
        midroll_omega_range: Bounds [rad/s] on the forward roll rate given to mid-roll spawns.
            Defaults to ``(0.0, 0.0)``, which skips the branch.
        tuck_joint_pos: Joint name to angle [rad] the mid-roll bucket lerps its legs and neck
            toward. Defaults to None, which leaves the joints at the pose the joint reset wrote.
        tuck_factor_range: Bounds [-] on the per-environment lerp factor from the reset pose toward
            :attr:`tuck_joint_pos`. Defaults to ``(0.3, 1.0)``, so the tuck is never fully absent.
        joint_noise_std: Standard deviation [rad] of the zero-mean noise added to every joint
            selected by :attr:`asset_cfg` in the mid-roll bucket, on top of the tuck. Defaults to
            0.0.
        asset_cfg: The articulation to reset, and the joints the tuck noise reaches. Selecting them
            by name reproduces upstream's exclusion of passive joints, which are not part of any
            keyframe.

    Raises:
        ValueError: If the two probabilities do not sum to a positive number.
    """
    total = standing_prob + midroll_prob
    if total <= 0.0:
        raise ValueError(
            "'reset_roulade_state' needs at least one bucket with a positive probability. Received"
            f" standing={standing_prob}, midroll={midroll_prob}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    is_midroll = torch.rand(num_resets, device=env.device) < (midroll_prob / total)

    def _uniform(bounds: tuple[float, float]) -> torch.Tensor:
        return torch.empty(num_resets, device=env.device).uniform_(*bounds)

    # orientation. The yaw is shared by both buckets, so an episode's heading does not correlate
    # with how far into the roll it starts.
    yaw = (torch.rand(num_resets, device=env.device) * 2.0 - 1.0) * torch.pi
    midroll_pitch = _uniform(midroll_pitch_range)
    pitch = (torch.rand(num_resets, device=env.device) * 2.0 - 1.0) * standing_tilt_max
    pitch = torch.where(is_midroll, midroll_pitch, pitch)
    # upstream floors the roll spread at 5 degrees for both buckets, so a mid-roll spawn is never
    # perfectly sagittal either -- the flatness penalty has something to correct from step one
    roll = (torch.rand(num_resets, device=env.device) * 2.0 - 1.0) * max(standing_tilt_max, math.radians(5.0))
    quat = quat_from_euler_xyz(roll, pitch, yaw)

    height = torch.where(is_midroll, _uniform(midroll_z_range), _uniform(standing_z_range))

    # The horizontal position is left alone: it is what an earlier root reset spread out. ``cat``
    # already allocates, so the assignments below cannot reach the articulation's own buffers.
    pose = torch.cat((asset.data.root_link_pos_w.torch[env_ids], asset.data.root_link_quat_w.torch[env_ids]), dim=-1)
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + height
    pose[:, 3:7] = quat
    asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)

    velocity = torch.zeros(num_resets, 6, device=env.device)
    if midroll_omega_range[1] > 0.0:
        # Upstream writes the free joint's ``qvel[4]``, which MuJoCo expresses in the **body**
        # frame, so the spin is about the robot's own pitch axis whatever its spawn yaw. Isaac Lab
        # writes world-frame velocities, so the same body-frame vector is rotated by the spawn
        # orientation here rather than being handed over as-is.
        omega_b = torch.zeros(num_resets, 3, device=env.device)
        omega_b[:, 1] = torch.where(
            is_midroll, _uniform(midroll_omega_range), torch.zeros(num_resets, device=env.device)
        )
        velocity[:, 3:6] = quat_apply(quat, omega_b)
    if forward_vel_range[1] > 0.0:
        # élan hook: a forward base velocity for standing spawns, body x mapped through the spawn
        # yaw. Never runs at the shipped ``(0.0, 0.0)``.
        forward_speed = torch.where(
            ~is_midroll, _uniform(forward_vel_range), torch.zeros(num_resets, device=env.device)
        )
        velocity[:, 0] = forward_speed * torch.cos(yaw)
        velocity[:, 1] = forward_speed * torch.sin(yaw)
    # the link frame rather than the centre of mass, because upstream's ``qvel`` is the free joint's
    # own frame: at 3 rad/s the two differ by several centimetres per second across the head offset
    asset.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)

    midroll_env_ids = env_ids[is_midroll]
    if len(midroll_env_ids) > 0 and (tuck_joint_pos or joint_noise_std > 0.0):
        # lerped from the live pose, which is the one the joint reset wrote a moment ago
        joint_pos = asset.data.joint_pos.torch[midroll_env_ids].clone()
        if tuck_joint_pos:
            tuck_ids = _keyframe_joint_ids(env, asset, tuple(tuck_joint_pos))
            angles = torch.tensor(list(tuck_joint_pos.values()), device=env.device)
            factor = torch.empty(len(midroll_env_ids), 1, device=env.device).uniform_(*tuck_factor_range)
            joint_pos[:, tuck_ids] += factor * (angles - joint_pos[:, tuck_ids])
        if joint_noise_std > 0.0:
            noise = torch.randn(len(midroll_env_ids), len(joint_pos[0, asset_cfg.joint_ids]), device=env.device)
            joint_pos[:, asset_cfg.joint_ids] += noise * joint_noise_std
        asset.write_joint_state_to_sim_index(
            position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=midroll_env_ids
        )

    state = roulade_roll_state(env)
    spawn_angle = torch.where(is_midroll, midroll_pitch, torch.zeros_like(midroll_pitch))
    state.accumulated_angle[env_ids] = spawn_angle
    state.frontier[env_ids] = spawn_angle
    state.paid[env_ids] = spawn_angle
    state.head_latch[env_ids] = is_midroll


def randomize_joint_dry_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Set the dry (Coulomb) friction of selected joints to a value drawn per environment and joint.

    Ported from addendum section 5.6 (``randomize_wheel_friction``, upstream's stock
    ``dr.dof_frictionloss`` at ``operation="abs"``). The roller task drives it on the four passive
    wheel hinges, whose MJCF friction is zero for trainability, and ramps it in by curriculum: a real
    bearing has drag, and a policy that has only ever skated on frictionless wheels overestimates how
    far it coasts.

    The stock :func:`isaaclab.envs.mdp.randomize_joint_parameters` is **not** reusable here. Its
    ``friction_distribution_params`` writes the sampled value into the joint's dry friction *and*
    into its viscous friction, and the wheel bearings have no authored viscous term: at the top of
    upstream's ramp the extra viscous coefficient would brake a wheel spinning at skating speed an
    order of magnitude harder than the dry friction it is meant to model.

    The passive wheels are outside the BAM servo group's ``^(?!passive_).*`` selection, so nothing
    republishes over this write -- unlike the driven joints, whose solver friction the actuator owns
    on both execution paths.

    Args:
        env: The environment holding the articulation.
        env_ids: The environments to resample. Defaults to None, which resamples all of them.
        friction_range: The ``(low, high)`` bounds [N*m] of the dry friction torque.
        asset_cfg: The articulation and the joints to write.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = None if isinstance(asset_cfg.joint_ids, slice) else asset_cfg.joint_ids
    num_joints = asset.num_joints if joint_ids is None else len(joint_ids)
    samples = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(*friction_range)
    asset.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=samples,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )


_BALL_KICK_DIRECTION_ATTR = "_microduck_ball_kick_direction"
"""Attribute the per-environment ball-kick direction is cached under on the environment."""


def ball_kick_direction(env: ManagerBasedEnv) -> torch.Tensor:
    """The world-frame direction "forward" means to the ball-kick rewards, allocated on first use.

    Ported from addendum section 2.3 (``_ball_kick_dir``). It is the robot's heading at the episode
    reset, frozen for the whole episode by :func:`reset_ball_in_front_of_foot`, and both kick rewards
    project the ball's velocity onto it. Freezing it is the point: a live heading would let the
    policy redefine "forward" by turning after the kick and collect the reward for a ball it pushed
    sideways.

    Before the first reset it is ``+x``, which is what upstream's lazy allocation gives too.

    Args:
        env: The environment the direction is bookkept for.

    Returns:
        The shared per-environment unit direction in the horizontal plane. Shape is (num_envs, 2).
        The same tensor on every call for a given environment.
    """
    direction: torch.Tensor | None = getattr(env, _BALL_KICK_DIRECTION_ATTR, None)
    if direction is None:
        direction = torch.zeros(env.num_envs, 2, device=env.device)
        direction[:, 0] = 1.0
        setattr(env, _BALL_KICK_DIRECTION_ATTR, direction)
    return direction


def reset_ball_in_front_of_foot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    offset: tuple[float, float],
    noise_xy: float = 0.0,
    ball_radius: float = 0.035,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Place the ball just in front of one foot, in the robot's own yaw frame, and freeze the kick
    direction for the episode.

    Ported from addendum section 2.3 (``reset_ball_in_front_of_foot``). The offset is applied in the
    frame of the robot's *reset* yaw rather than along the world axes, because the ground-state reset
    spawns the robot at a uniformly random heading; placing the ball at a world-frame offset would
    put it behind the robot on half the episodes.

    The ball is set down at rest, exactly touching the ground, with an identity orientation -- a
    sphere has no meaningful one, and the initial spin is what the kick has to create.

    This term reads the robot's root pose, so it must run **after** every reset event that writes it.
    Isaac Lab fires reset events in configuration declaration order.

    Args:
        env: The environment holding the two assets.
        env_ids: The environments to reset. Defaults to None, which resets all of them.
        offset: Ball centre offset ``(forward, left)`` [m] from the robot root, in the robot's yaw
            frame. Negative ``left`` places it in front of the right foot.
        noise_xy: Half-width [m] of the uniform noise added to both offset components. Defaults to
            0.0, which places the ball exactly at the offset.
        ball_radius: Radius [m] of the ball, which is how high its centre is set above the ground.
            Defaults to 0.035, the radius of :data:`~isaaclab_assets.MICRODUCK_BALL_CFG`.
        asset_cfg: The rigid object to place.
        robot_cfg: The articulation whose root pose the placement is measured from.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[asset_cfg.name]

    # Upstream reads the robot root straight out of ``qpos`` because its own derived buffers lag
    # until the next ``forward()``. Isaac Lab's root writes update the articulation's buffers as they
    # go, so the pose read here is the one the preceding reset events wrote.
    root_pos = robot.data.root_link_pos_w.torch[env_ids]
    quat = robot.data.root_link_quat_w.torch[env_ids]
    # Isaac Lab quaternions are (x, y, z, w) where upstream's are (w, x, y, z), so the four
    # components below are read at shifted indices; the yaw formula itself is upstream's.
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    planar_offset = torch.tensor(offset, device=env.device, dtype=root_pos.dtype).repeat(num_resets, 1)
    if noise_xy > 0.0:
        planar_offset += (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * noise_xy

    pose = torch.zeros(num_resets, 7, device=env.device, dtype=root_pos.dtype)
    pose[:, 0] = root_pos[:, 0] + cos_yaw * planar_offset[:, 0] - sin_yaw * planar_offset[:, 1]
    pose[:, 1] = root_pos[:, 1] + sin_yaw * planar_offset[:, 0] + cos_yaw * planar_offset[:, 1]
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + ball_radius
    pose[:, 6] = 1.0  # (x, y, z, w) identity
    ball.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    ball.write_root_com_velocity_to_sim_index(
        root_velocity=torch.zeros(num_resets, 6, device=env.device, dtype=root_pos.dtype), env_ids=env_ids
    )

    direction = ball_kick_direction(env)
    direction[env_ids, 0] = cos_yaw
    direction[env_ids, 1] = sin_yaw
