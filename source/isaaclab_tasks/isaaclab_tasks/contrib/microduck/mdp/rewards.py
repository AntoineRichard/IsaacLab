# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms MicroDuck needs that have no stock Isaac Lab counterpart.

Every term here is ported from ``pollen-robotics/microduck_rl`` and the mjlab velocity template it
builds on; see sections 5 (mjlab kernels) and 6 (MicroDuck's own kernels) of
``artifacts/microduck/upstream_reference.md`` for the verbatim upstream formulas. Each function
cites the section it comes from and, where a stock ``isaaclab.envs.mdp`` term covers the same
quantity, why that term could not be reused.

Three families of divergence recur and are worth stating once:

* **Penalties folded into tracking exponents.** Upstream scores the vertical velocity and the
  roll/pitch rate *inside* the velocity-tracking exponents rather than as separate additive terms.
  Inside the exponent a penalty saturates together with the tracking error; added on the outside it
  grows without bound. The stock ``track_lin_vel_xy_exp`` / ``track_ang_vel_z_exp`` therefore are
  not substitutes, however similar they look.
* **Link frame, not centre-of-mass frame.** Upstream reads ``root_link_lin_vel_b`` and
  ``root_link_ang_vel_b``; the stock terms read ``root_lin_vel_b`` / ``root_ang_vel_b``, which
  Isaac Lab measures at the body centre of mass.
* **Sites versus bodies.** Upstream measures foot quantities on MJCF *sites* placed at the soles.
  Isaac Lab has no site concept, so the ports measure the ankle *body* frames instead. The
  functional form is unchanged, but height thresholds are offset by the sole-to-ankle distance and
  need re-tuning rather than copying.

Upstream's diagnostic metrics -- the per-term breakdowns it writes into ``env.extras["log"]``, and
its ``mean_action_acc`` metric term (reference section 2.9) -- are intentionally not ported here:
Isaac Lab's reward manager already logs each term's episode sum, and any further logging belongs
with the task's instrumentation pass rather than inside the reward kernels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.string import resolve_matching_names_values

from .events import roulade_roll_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _required_entity_cfg(cfg: RewardTermCfg, key: str, term_name: str) -> SceneEntityCfg:
    """Fetch a scene entity configuration that a stateful term needs at construction time.

    A class-based term is built before it is ever called, so it cannot fall back on a signature
    default: the entity has to be spelled out in :attr:`RewardTermCfg.params`. Reading it directly
    would surface as a bare ``KeyError`` from inside the manager, which says nothing about what to
    fix.

    Args:
        cfg: The term configuration.
        key: Name of the parameter holding the entity configuration.
        term_name: Name of the term, used in the error message.

    Returns:
        The scene entity configuration.

    Raises:
        ValueError: If the parameter is missing or is not a :class:`SceneEntityCfg`.
    """
    entity_cfg = cfg.params.get(key)
    if not isinstance(entity_cfg, SceneEntityCfg):
        raise ValueError(
            f"The reward term '{term_name}' requires a 'SceneEntityCfg' under the '{key}' entry of"
            f" its 'params'. Received: {entity_cfg!r}."
        )
    return entity_cfg


def _required_joint_ids(entity_cfg: SceneEntityCfg, key: str, term_name: str) -> list[int]:
    """Fetch an explicit joint selection from a resolved scene entity configuration.

    An unset :attr:`SceneEntityCfg.joint_names` leaves :attr:`~SceneEntityCfg.joint_ids` as
    ``slice(None)``, which selects every joint in whatever order the backend resolved them. The
    terms that use this helper index a command tensor positionally, so silently selecting all
    joints would mis-pair the columns rather than fail.

    Args:
        entity_cfg: The resolved scene entity configuration.
        key: Name of the parameter holding it, used in the error message.
        term_name: Name of the term, used in the error message.

    Returns:
        The selected joint indices.

    Raises:
        ValueError: If no joints were selected by name.
    """
    if isinstance(entity_cfg.joint_ids, slice):
        raise ValueError(
            f"The reward term '{term_name}' requires '{key}' to select its joints by name, so that"
            " their order is pinned. Set 'joint_names' with 'preserve_order=True'."
        )
    return entity_cfg.joint_ids


def _command_magnitude(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Upstream's scalar command magnitude ``|v_xy| + |w_z|``.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command term.

    Returns:
        The command magnitude. Shape is (num_envs,).
    """
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])


def _root_height_above_ground(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    """Height of the articulation's root link above the ground its environment sits on [m].

    Upstream measures this as ``root_link_pos_w[:, 2] - terrain.env_origins[:, 2]`` and guards it
    against non-finite values, because a diverged solver would otherwise poison every term that
    reads it before the NaN termination gets to fire.

    The stand-up terms that use this are configured upstream with ``body_names=("trunk_base",)``,
    but their kernels read the *root link* and ignore that selection. On MicroDuck the trunk is the
    root, so the two are the same body and the selection is inert; this port drops it rather than
    carry a parameter nothing reads.
    """
    return torch.nan_to_num(asset.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2], nan=0.0)


def _trunk_tilt_squared(quat: torch.Tensor) -> torch.Tensor:
    """Upstream's ``2 * (q_x^2 + q_y^2)`` tilt measure, which is ``1 - cos(tilt)``.

    Zero when the body frame is upright, 1 when it is horizontal and 2 when it is inverted.

    Note:
        Upstream's quaternions are ``(w, x, y, z)`` and Isaac Lab's are ``(x, y, z, w)``, so the two
        components read here are columns 0 and 1 rather than upstream's 1 and 2. Every stand-up term
        that reads an orientation goes through this helper for that reason.

    Args:
        quat: Root link orientation in (x, y, z, w). Shape is (num_envs, 4).

    Returns:
        The squared tilt in ``[0, 2]``. Shape is (num_envs,).
    """
    return 2.0 * (torch.square(quat[:, 0]) + torch.square(quat[:, 1]))


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    """Upstream's ``t^2 (3 - 2t)`` smoothstep on an already-normalized, unclamped ratio."""
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _height_gate(z: torch.Tensor, height_low: float, height_high: float) -> torch.Tensor:
    """Smoothstep gate that opens as the trunk rises from ``height_low`` [m] to ``height_high`` [m]."""
    return _smoothstep((z - height_low) / max(height_high - height_low, 1e-6))


def _tilt_gate(quat: torch.Tensor, tilt_full_deg: float, tilt_zero_deg: float) -> torch.Tensor:
    """Smoothstep gate that closes as the trunk tilts from ``tilt_full_deg`` to ``tilt_zero_deg``."""
    cos_tilt = 1.0 - _trunk_tilt_squared(quat)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
    return _smoothstep((tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6))


def _feet_height_above_ground(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height of the selected foot bodies above the ground [m], shape (num_envs, num_feet).

    Upstream reads a terrain height sensor that ray-casts a small ring under each foot site.
    Isaac Lab has no equivalent sensor on this environment, so the ground is taken to be the plane
    through the environment origin. That is exact on flat terrain and approximate on generated
    terrain, where it holds only as long as the sub-terrain under an environment is level.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_z = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    return foot_pos_z - env.scene.env_origins[:, 2].unsqueeze(1)


"""
Velocity tracking.
"""


def track_linear_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of the linear velocity command, penalizing vertical drift in the same kernel.

    Ported from reference section 5 (``track_linear_velocity``). The stock
    :func:`isaaclab.envs.mdp.track_lin_vel_xy_exp` is **not** reusable: it omits the ``v_z^2`` term
    and reads the centre-of-mass frame instead of the root link frame.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel [m/s].
        command_name: Name of the velocity command term.
        asset_cfg: The articulation to measure. Defaults to the entity named ``"robot"``.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b.torch
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    # the commanded vertical velocity is always zero, so the drift itself is the error
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of the yaw-rate command, penalizing roll and pitch rate in the same kernel.

    Ported from reference section 5 (``track_angular_velocity``). The stock
    :func:`isaaclab.envs.mdp.track_ang_vel_z_exp` is **not** reusable: it omits the ``|w_xy|^2``
    term and reads the centre-of-mass frame instead of the root link frame.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel [rad/s].
        command_name: Name of the velocity command term.
        asset_cfg: The articulation to measure. Defaults to the entity named ``"robot"``.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b.torch
    z_error = torch.square(command[:, 2] - actual[:, 2])
    # the commanded roll and pitch rates are always zero
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)


"""
Posture.
"""


def upright(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward keeping a body's frame aligned with gravity, using a Gaussian kernel.

    Ported from reference section 5 (``upright``). The stock
    :func:`isaaclab.envs.mdp.flat_orientation_l2` measures the same tilt but as an unbounded L2
    *penalty*; upstream uses a bounded Gaussian *reward* with a positive weight, so the two differ
    in shape and in sign and cannot be substituted for one another.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel, in units of the projected gravity direction.
        asset_cfg: The articulation and the single body to measure. With no body selected the
            articulation root is used, as upstream does.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).

    Raises:
        ValueError: If ``asset_cfg`` selects more than one body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_names is None:
        projected_gravity_b = asset.data.projected_gravity_b.torch
    else:
        if len(asset_cfg.body_ids) != 1:
            raise ValueError(
                f"'upright' measures a single body's tilt; 'asset_cfg' selected"
                f" {len(asset_cfg.body_ids)} bodies: {asset_cfg.body_names}."
            )
        body_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
        gravity_dir_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1).unsqueeze(1)
        projected_gravity_b = math_utils.quat_apply_inverse(body_quat_w, gravity_dir_w).squeeze(1)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-xy_squared / std**2)


class pose_mode_switch(ManagerTermBase):
    """Reward holding the stand pose, with a per-joint tolerance that widens once the robot moves.

    Ported from reference section 5 (``variable_posture``, upstream term name ``pose``). There is
    no stock counterpart: :func:`isaaclab.envs.mdp.joint_deviation_l1` measures the same deviation
    but as an L1 penalty with a single global scale, whereas this term is a Gaussian reward whose
    width is a per-joint vector selected by the commanded speed.

    The commanded speed ``|v_xy| + |w_z|`` selects one of three per-joint standard-deviation
    vectors -- standing, walking or running -- and the reward is
    ``exp(-mean((q - q_home)^2 / std^2))`` over the selected joints. MicroDuck configures the
    running vector equal to the walking one, so the running regime is inactive in practice; the
    third vector is kept because upstream's curriculum can separate them.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Resolve the three per-joint standard-deviation vectors.

        Args:
            cfg: The term configuration, whose ``params`` carry the three name-to-value dictionaries
                and the selected joints.
            env: The environment instance.

        Raises:
            ValueError: If ``asset_cfg`` is missing, selects no joints by name, or if one of the
                three standard-deviation dictionaries is missing.
        """
        super().__init__(cfg, env)

        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        asset: Articulation = env.scene[asset_cfg.name]
        joint_names = [asset.joint_names[index] for index in joint_ids]
        self._std = {}
        for key in ("std_standing", "std_walking", "std_running"):
            if key not in cfg.params:
                raise ValueError(
                    f"The reward term '{self.__name__}' requires the '{key}' entry of its 'params'"
                    " to map joint-name patterns to standard deviations."
                )
            # ordered by the selected joints, so the vector lines up with ``joint_ids``
            _, _, values = resolve_matching_names_values(cfg.params[key], joint_names)
            self._std[key] = torch.tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float,
        running_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the posture reward.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            std_standing: Joint-name pattern to standard deviation [m or rad, depending on joint
                type] used below :attr:`walking_threshold`.
            std_walking: The same, used between the two thresholds.
            std_running: The same, used above :attr:`running_threshold`.
            walking_threshold: Command magnitude above which the robot counts as walking.
            running_threshold: Command magnitude above which the robot counts as running.
            asset_cfg: The articulation and the joints to hold at the stand pose.

        Returns:
            The reward in ``(0, 1]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        total_speed = _command_magnitude(env, command_name)

        standing_mask = (total_speed < walking_threshold).float().unsqueeze(1)
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float().unsqueeze(1)
        running_mask = (total_speed >= running_threshold).float().unsqueeze(1)
        std = (
            self._std["std_standing"] * standing_mask
            + self._std["std_walking"] * walking_mask
            + self._std["std_running"] * running_mask
        )

        joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        default_joint_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        error_squared = torch.square(joint_pos - default_joint_pos)
        return torch.exp(-torch.mean(error_squared / std**2, dim=1))


"""
Head and body pose tracking.
"""


def head_pose_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of the commanded head-pose delta, averaged over the neck and head joints.

    Ported from reference section 6 (``head_pose_tracking``). The command is a joint-position delta
    from the stand pose, so the tracked error is ``(q - q_home) - cmd``, and the reward is the mean
    of the per-joint Gaussians. There is no stock counterpart: the closest stock term,
    :func:`isaaclab.envs.mdp.joint_deviation_l1`, tracks the stand pose rather than a command.

    Upstream also carries an optional narrow-Gaussian blend (``fine_std`` / ``fine_weight``) that
    the velocity environment never configures, so it is deliberately not ported.

    Args:
        env: The environment instance.
        command_name: Name of the head-pose command term. Its columns must line up with the joints
            selected by :attr:`asset_cfg`, which upstream orders
            ``(neck_pitch, head_pitch, head_yaw, head_roll)``.
        std: Width of the per-joint Gaussian kernel [rad].
        asset_cfg: The articulation and the head joints to track. Select them by name with
            ``preserve_order=True`` so the columns match the command.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    error = (measured - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]) - command
    return torch.exp(-((error / std) ** 2)).mean(dim=-1)


class head_pose_bias_penalty(ManagerTermBase):
    """Penalize a head-pose error that persists, as opposed to one that averages out.

    Ported from reference section 6 (``head_pose_bias_penalty``). The term holds a per-environment,
    per-joint exponential moving average of the head-pose error with time constant ``tau_s`` and
    returns ``-mean(|ema|)``. It is stateful, so it is a class rather than a function; the state is
    cleared for the environments the reward manager resets.

    The term **negates itself** and is therefore configured with a *positive* weight, matching
    upstream's ``0.0 -> 3.0`` ramp.

    Where upstream clears the average whenever ``episode_length_buf <= 1``, this port clears it from
    :meth:`reset`, which the manager calls with exactly the environments that restarted. The two
    are equivalent and the hook is the Isaac Lab convention.

    An optional **upright gate** suppresses the term while the robot is low or tilted, for the
    recovery tasks whose episodes start on the ground. It multiplies both the error entering the
    average and the returned penalty, so a prone episode accumulates nothing and pays nothing --
    without it the term would tax the head-first phase of a stand-up, which is the motion it needs
    to discover. The velocity task leaves it off.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-environment moving average.

        Args:
            cfg: The term configuration, whose ``params`` carry the tracked joints.
            env: The environment instance.

        Raises:
            ValueError: If ``asset_cfg`` is missing or selects no joints by name.
        """
        super().__init__(cfg, env)

        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        self._error_ema = torch.zeros(env.num_envs, len(joint_ids), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the average for the environments that restarted.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are cleared.
        """
        if env_ids is None:
            self._error_ema[:] = 0.0
        else:
            self._error_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        tau_s: float,
        asset_cfg: SceneEntityCfg,
        gate_height_low: float | None = None,
        gate_height_high: float = 0.11,
        gate_tilt_full_deg: float = 20.0,
        gate_tilt_zero_deg: float = 45.0,
    ) -> torch.Tensor:
        """Advance the moving average and return the penalty.

        Args:
            env: The environment instance.
            command_name: Name of the head-pose command term.
            tau_s: Time constant of the moving average [s].
            asset_cfg: The articulation and the head joints to track. Mandatory: the term sizes its
                state from the selection at construction time, and the command columns are paired
                with the joints positionally.
            gate_height_low: Trunk height [m] below which the upright gate is fully closed. Defaults
                to None, which disables the gate entirely and leaves the other three unread.
            gate_height_high: Trunk height [m] above which the height half of the gate is fully
                open. Defaults to 0.11.
            gate_tilt_full_deg: Trunk tilt [deg] below which the tilt half of the gate is fully
                open. Defaults to 20.0.
            gate_tilt_zero_deg: Trunk tilt [deg] above which the tilt half of the gate is fully
                closed. Defaults to 45.0.

        Returns:
            The penalty in ``(-inf, 0]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        measured = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        error = (measured - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]) - command

        gate = None
        if gate_height_low is not None:
            quat = asset.data.root_link_quat_w.torch
            gate = _height_gate(_root_height_above_ground(env, asset), gate_height_low, gate_height_high)
            gate = gate * _tilt_gate(quat, gate_tilt_full_deg, gate_tilt_zero_deg)
            error = error * gate.unsqueeze(-1)

        alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
        self._error_ema = (1.0 - alpha) * self._error_ema + alpha * error
        penalty = -self._error_ema.abs().mean(dim=-1)
        return penalty if gate is None else penalty * gate


def body_pose_tracking_6d(
    env: ManagerBasedRLEnv,
    command_name: str,
    nominal_height: float,
    xy_std: float,
    z_std: float,
    angle_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Reward tracking of a six-dimensional trunk-pose command, averaged over the six axes.

    Ported from reference section 6 (``body_pose_tracking_6d``). The command is a delta from the
    nominal stand: ``(dx, dy, dz)`` [m] measured from the environment origin, with the vertical
    target at ``nominal_height + dz``, and ``(droll, dpitch, dyaw)`` [rad]. Each axis contributes
    its own Gaussian and the six are averaged, so no single axis can dominate.

    There is no stock counterpart. :func:`isaaclab.envs.mdp.position_command_error` and
    :func:`~isaaclab.envs.mdp.orientation_command_error` measure a pose command as an unbounded
    error norm against a pose command term, not as a per-axis Gaussian around a nominal stand.

    Note:
        Upstream ships a second kernel, ``body_pose_tracking_locomotion``, which differs only in
        measuring ``x``/``y`` against the feet-site centroid and ``yaw`` against the mean foot-site
        yaw, so that those axes stay meaningful while the robot walks away from its spawn. The
        stand-up task selects it but weights those three axes **zero**, so the two kernels agree
        exactly there and this one is used with :attr:`axis_weights` instead. Reviving the
        horizontal axes for a task that does weight them needs the foot *sites*, which Isaac Lab
        has no equivalent of: the ankle bodies this port measures feet with sit 16 mm outboard of
        the sites and, on the left side, with a 180-degree yaw flip.

    Args:
        env: The environment instance.
        command_name: Name of the six-dimensional body-pose command term.
        nominal_height: Trunk height the vertical command is measured from [m].
        xy_std: Width of the Gaussian kernel on the horizontal position [m].
        z_std: Width of the Gaussian kernel on the height [m].
        angle_std: Width of the Gaussian kernel on each Euler angle [rad].
        asset_cfg: The articulation whose root link carries the trunk pose.
        axis_weights: Relative weight of each axis, ordered ``(x, y, z, roll, pitch, yaw)``. The
            reward is their weighted mean. Defaults to equal weights, which is the plain average.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # a non-finite root position would otherwise poison every axis; upstream guards it here and
    # terminates on it separately
    relative_pos = torch.nan_to_num(asset.data.root_link_pos_w.torch - env.scene.env_origins, nan=0.0)
    x_error = relative_pos[:, 0] - command[:, 0]
    y_error = relative_pos[:, 1] - command[:, 1]
    z_error = relative_pos[:, 2] - (nominal_height + command[:, 2])

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w.torch)
    roll_error = roll - command[:, 3]
    pitch_error = pitch - command[:, 4]
    yaw_error = math_utils.wrap_to_pi(yaw - command[:, 5])

    weight_x, weight_y, weight_z, weight_roll, weight_pitch, weight_yaw = axis_weights
    reward = weight_x * torch.exp(-((x_error / xy_std) ** 2))
    reward = reward + weight_y * torch.exp(-((y_error / xy_std) ** 2))
    reward = reward + weight_z * torch.exp(-((z_error / z_std) ** 2))
    reward = reward + weight_roll * torch.exp(-((roll_error / angle_std) ** 2))
    reward = reward + weight_pitch * torch.exp(-((pitch_error / angle_std) ** 2))
    reward = reward + weight_yaw * torch.exp(-((yaw_error / angle_std) ** 2))
    return reward / max(sum(axis_weights), 1e-6)


"""
Gait and feet.
"""


def feet_air_time_windowed(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold_min: float,
    threshold_max: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward feet whose current air time lies strictly inside a window, gated on the command.

    Ported from reference section 5 (``feet_air_time``, upstream term name ``air_time``). The
    reward is the *count* of feet in the window, so it is bounded by the number of feet -- it is
    not an integral over the swing. Neither stock counterpart reproduces that:
    :func:`isaaclab_tasks.core.velocity.mdp.feet_air_time` pays out the completed air time minus a
    threshold on the landing step, and
    :func:`~isaaclab_tasks.core.velocity.mdp.feet_air_time_positive_biped` pays out the clamped
    single-stance mode time; both reward *longer* steps without an upper bound on the swing.

    Air-time semantics carry over directly: Isaac Lab's ``current_air_time`` is reset to zero while
    a body registers a contact force above :attr:`~isaaclab.sensors.ContactSensorCfg.force_threshold`
    and accumulates the elapsed time otherwise, which is what upstream's sensor reports.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the foot bodies to read. Select them by name with
            ``preserve_order=True``; the sensor resolves bodies in prim-label order.
        command_name: Name of the velocity command term.
        threshold_min: Lower edge of the rewarded air-time window [s], exclusive.
        threshold_max: Upper edge of the rewarded air-time window [s], exclusive.
        command_threshold: Command magnitude below which the reward is suppressed.

    Returns:
        The number of feet inside the window, or zero for a standing command. Shape is (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    return reward * (_command_magnitude(env, command_name) > command_threshold).float()


def foot_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize feet that deviate from a target height while moving horizontally.

    Ported from reference section 5 (``feet_clearance``, upstream term name ``foot_clearance``).
    The cost is ``sum(|h - target| * |v_xy|)`` over the feet, gated on the command magnitude:
    weighting by planar speed charges a swinging foot at the wrong height and leaves a planted one
    alone. There is no stock counterpart.

    Args:
        env: The environment instance.
        target_height: Height above the ground the swinging foot should hold [m]. Upstream measures
            it at the sole site; this port measures the ankle body frame, so the value needs
            re-tuning by the sole-to-ankle offset rather than copying.
        command_name: Name of the velocity command term.
        asset_cfg: The articulation and the foot bodies to measure. Select them by name with
            ``preserve_order=True``.
        command_threshold: Command magnitude below which the penalty is suppressed.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = _feet_height_above_ground(env, asset_cfg)
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.abs(foot_height - target_height) * torch.norm(foot_vel_xy, dim=-1), dim=1)
    return cost * (_command_magnitude(env, command_name) > command_threshold).float()


class foot_swing_height(ManagerTermBase):
    """Penalize, on touchdown, how far the peak of the swing was from a target height.

    Ported from reference section 5 (``feet_swing_height``, upstream term name
    ``foot_swing_height``). The term tracks the peak height each foot reaches while airborne,
    charges the *relative* squared error ``(peak / target - 1)^2`` on the step the foot lands, and
    clears that foot's peak. Being stateful, it is a class; the peaks are cleared for the
    environments the reward manager resets. There is no stock counterpart.

    Upstream detects "airborne" through its sensor's ``found`` field. Isaac Lab's equivalent is
    ``current_contact_time``, which is zero exactly while no contact force above the sensor's
    threshold is registered.

    Note:
        :meth:`~isaaclab.sensors.ContactSensor.compute_first_contact` returns a **float** mask of
        ones and zeros, not a boolean one, so its result is thresholded before it is used as a
        condition.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-foot peak heights.

        Args:
            cfg: The term configuration, whose ``params`` carry the tracked feet.
            env: The environment instance.

        Raises:
            ValueError: If ``sensor_cfg`` is missing from the term parameters.
        """
        super().__init__(cfg, env)

        sensor_cfg = _required_entity_cfg(cfg, "sensor_cfg", self.__name__)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        num_feet = len(contact_sensor.data.current_contact_time.torch[0, sensor_cfg.body_ids])
        self._peak_heights = torch.zeros(env.num_envs, num_feet, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the tracked peaks for the environments that restarted.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are cleared.
        """
        if env_ids is None:
            self._peak_heights[:] = 0.0
        else:
            self._peak_heights[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        """Track the swing peaks and charge the landing ones.

        Args:
            env: The environment instance.
            sensor_cfg: The contact sensor and the foot bodies to read.
            asset_cfg: The articulation and the foot bodies to measure. Its body order must match
                :attr:`sensor_cfg`.
            target_height: Peak swing height the foot should reach [m]. Measured at the ankle body
                frame rather than at upstream's sole site.
            command_name: Name of the velocity command term.
            command_threshold: Command magnitude below which the penalty is suppressed.

        Returns:
            The cost in ``[0, inf)``. Shape is (num_envs,).
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_air = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] == 0.0
        foot_height = _feet_height_above_ground(env, asset_cfg)
        self._peak_heights = torch.where(in_air, torch.maximum(self._peak_heights, foot_height), self._peak_heights)

        # the sensor reports the landing mask as ones and zeros in float32, which cannot be used as
        # a ``torch.where`` condition directly
        first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids] > 0.5
        error = self._peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1)
        cost = cost * (_command_magnitude(env, command_name) > command_threshold).float()

        # the peak is charged once; the next swing starts from scratch
        self._peak_heights = torch.where(first_contact, torch.zeros_like(self._peak_heights), self._peak_heights)
        return cost


def foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize the squared planar speed of feet that are on the ground.

    Ported from reference section 5 (``feet_slip``, upstream term name ``foot_slip``). The stock
    :func:`isaaclab_tasks.core.velocity.mdp.feet_slide` measures the same quantity but charges the
    speed *linearly* and applies no command gate, so the two differ in both shape and coverage.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the foot bodies to read.
        command_name: Name of the velocity command term.
        asset_cfg: The articulation and the foot bodies to measure. Its body order must match
            :attr:`sensor_cfg`.
        command_threshold: Command magnitude below which the penalty is suppressed.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = (contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).float()
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)
    return cost * (_command_magnitude(env, command_name) > command_threshold).float()


"""
Trunk regularizers.
"""


def body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the roll and pitch rate of selected bodies, in the world frame.

    Ported from reference section 5 (``body_angular_velocity_penalty``, upstream term name
    ``body_ang_vel``). The stock :func:`isaaclab.envs.mdp.ang_vel_xy_l2` measures the articulation
    root in its own *body* frame; upstream measures a named body in the *world* frame, so the two
    disagree as soon as the trunk is not level.

    The yaw rate is deliberately excluded: it is what the velocity command asks for.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the bodies to measure.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=(1, 2))


def angular_momentum_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the squared magnitude of the articulation's angular momentum about its own centre of mass.

    Ported from reference section 5 (``angular_momentum_penalty``, upstream term name
    ``angular_momentum``). Upstream reads a MuJoCo ``subtreeangmom`` sensor, which Isaac Lab has no
    equivalent of, so the quantity is assembled from the body states instead:

    .. math::

        L = \\sum_i \\left[ R_i I_i R_i^T \\omega_i + m_i (r_i - r_c) \\times (v_i - v_c) \\right]

    where :math:`r_c` and :math:`v_c` are the mass-weighted centre of mass and its velocity. This
    is the same physical quantity the sensor reports, computed rather than measured. There is no
    stock counterpart.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the bodies to sum over. Defaults to every body of the
            entity named ``"robot"``, matching upstream's whole-robot subtree.

    Returns:
        The squared angular momentum [kg^2 m^4 / s^2]. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    mass = asset.data.body_mass.torch[:, asset_cfg.body_ids]
    pos_w = asset.data.body_com_pos_w.torch[:, asset_cfg.body_ids]
    lin_vel_w = asset.data.body_com_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang_vel_w = asset.data.body_com_ang_vel_w.torch[:, asset_cfg.body_ids]
    quat_w = asset.data.body_com_quat_w.torch[:, asset_cfg.body_ids]
    # ``body_inertia`` is stored about the body centre of mass in the body frame, so it has to be
    # rotated into the world frame before it can act on a world-frame angular velocity
    inertia_b = asset.data.body_inertia.torch[:, asset_cfg.body_ids].view(*mass.shape, 3, 3)

    total_mass = mass.sum(dim=1, keepdim=True).unsqueeze(-1)
    com_pos_w = (mass.unsqueeze(-1) * pos_w).sum(dim=1, keepdim=True) / total_mass
    com_lin_vel_w = (mass.unsqueeze(-1) * lin_vel_w).sum(dim=1, keepdim=True) / total_mass

    rot = math_utils.matrix_from_quat(quat_w)
    inertia_w = torch.matmul(rot, torch.matmul(inertia_b, rot.transpose(-2, -1)))
    spin = torch.matmul(inertia_w, ang_vel_w.unsqueeze(-1)).squeeze(-1)
    orbit = mass.unsqueeze(-1) * torch.cross(pos_w - com_pos_w, lin_vel_w - com_lin_vel_w, dim=-1)

    momentum = (spin + orbit).sum(dim=1)
    return torch.sum(torch.square(momentum), dim=-1)


def self_collision_cost(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0
) -> torch.Tensor:
    """Penalize bodies of the robot that are touching the robot itself.

    Ported from reference section 5 (``self_collision_cost``, upstream term name
    ``self_collisions``). Upstream counts the self-contact slots its sensor reports as found. Isaac
    Lab reports filtered contacts as a force matrix, so this port counts the *bodies* carrying a
    self-contact force above :attr:`force_threshold` -- the same "how much of the robot is touching
    itself" signal, quantized per body rather than per contact slot. The stock
    :func:`isaaclab.envs.mdp.undesired_contacts` counts bodies the same way but reads
    ``net_forces_w``, the total contact force including the ground, which for a walking robot is
    never zero.

    The sensor must be configured with
    :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` pointing back at the robot,
    otherwise no force matrix is produced.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the bodies to read.
        force_threshold: Contact-force magnitude above which a body counts as touching [N].
            Defaults to 1.0, the same "a real contact" threshold the stock contact terms use.

    Returns:
        The number of bodies in self-contact. Shape is (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(
            f"The contact sensor '{sensor_cfg.name}' reports no force matrix. Set"
            " 'filter_prim_paths_expr' on its configuration so that self-contacts are resolved."
        )
    forces = force_matrix_w.torch[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.norm(forces, dim=-1) > force_threshold
    return in_contact.any(dim=-1).sum(dim=-1).float()


"""
Standing up: posture, height and orientation terms.
"""


def joint_pose_gaussian(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward holding selected joints at the stand pose, with a single Gaussian tolerance.

    Ported from addendum section 3.3 (``pose_target_match``). Unlike
    :class:`pose_mode_switch`, which selects a per-joint tolerance vector from the commanded speed,
    this is one scalar width applied to every selected joint -- the stand-up task has no walking
    regime to switch into.

    Upstream additionally accepts a ``target_overrides`` mapping that shifts individual targets off
    the stand pose. Every stand-up slot passes ``None``, so it is not ported.

    Args:
        env: The environment instance.
        std: Width of the per-joint Gaussian kernel [m or rad, depending on joint type].
        asset_cfg: The articulation and the joints to hold at the stand pose.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return torch.exp(-((error / std) ** 2)).mean(dim=-1)


def joint_pose_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the mean absolute deviation of selected joints from the stand pose.

    Ported from addendum section 3.3 (``pose_l1_penalty``). It is the constant-gradient companion
    to :func:`joint_pose_gaussian`, which saturates once the pose error grows past its width and
    then stops pulling. The stock :func:`isaaclab.envs.mdp.joint_deviation_l1` sums rather than
    averages, so its scale follows the number of selected joints.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the joints to hold at the stand pose.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return -torch.abs(error).mean(dim=-1)


def root_height_gaussian(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding the trunk at a target height above the ground, using a Gaussian kernel.

    Ported from addendum section 3.3 (``height_target_gaussian``). The stand-up task instantiates it
    twice, at a wide width that pulls all the way up from the sitting keyframe and at a narrow one
    that only has gradient in the last centimetre; the wide layer alone saturates before the robot
    is standing. The stock :func:`isaaclab.envs.mdp.base_height_l2` is an unbounded penalty around
    the same target, so it neither saturates nor stacks.

    Args:
        env: The environment instance.
        target_height: Trunk height to hold [m], measured above the environment origin.
        std: Width of the Gaussian kernel [m].
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-(((_root_height_above_ground(env, asset) - target_height) / std) ** 2))


def root_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the absolute distance of the trunk from a target height above the ground.

    Ported from addendum section 3.3 (``height_l1_penalty``). It is what makes staying seated cost
    something: the Gaussian layers are near zero down there and give the policy nothing to descend,
    whereas this one charges a constant gradient all the way from the sitting height.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        target_height: Trunk height to hold [m], measured above the environment origin.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.abs(_root_height_above_ground(env, asset) - target_height)


def com_upward_velocity(
    env: ManagerBasedRLEnv,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward rising: the trunk's upward speed, paid only while it is below a ceiling.

    Ported from addendum section 3.3 (``com_upward_velocity``). This is the term that pays for the
    *motion* of standing up rather than for arriving: with destination rewards alone, sitting still
    and collecting most of the posture and upright terms is the dominant local optimum. The ceiling
    stops the policy farming the reward by bobbing once it is up. Downward motion is free, not
    penalized -- :func:`trunk_vertical_accel_penalty` is what keeps the rise smooth.

    Upstream also accepts a ``max_vz`` cap on the rewarded speed, which the stand-up task
    deliberately leaves unset after two runs in which capping it suppressed the noisy recovery
    attempts the policy has to make before it can flip; it is not ported.

    Args:
        env: The environment instance.
        max_height: Trunk height [m] above which the reward is switched off.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    below_target = (_root_height_above_ground(env, asset) < max_height).float()
    vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    return torch.clamp(vertical_speed, min=0.0) * below_target


class trunk_vertical_accel_penalty(ManagerTermBase):
    """Penalize the magnitude of the trunk's vertical acceleration.

    Ported from addendum section 3.3 (``trunk_vertical_accel_penalty``). Paired with
    :func:`com_upward_velocity` it selects a smooth, constant-speed rise: a constant upward velocity
    collects the speed reward *and* has zero acceleration, so the two pressures agree only on that
    trajectory. There is no stock counterpart; the closest term,
    :func:`isaaclab.envs.mdp.lin_vel_z_l2`, penalizes the speed itself, which would fight the rise
    rather than smooth it.

    Being a finite difference the term is stateful, so it is a class. The step after a reset is
    charged nothing: the previous velocity then belongs to the previous episode, and differencing
    across the discontinuity would charge every environment a spurious impulse on its first step.

    The term **negates itself** and is therefore configured with a *positive* weight. Upstream
    documents a historical sign bug here, where the doubly negated form paid a *reward* for vertical
    shocks.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the previous-velocity buffer.

        Args:
            cfg: The term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._previous_speed = torch.zeros(env.num_envs, device=env.device)
        self._is_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Mark the environments that restarted, so their next difference is not charged.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are marked.
        """
        if env_ids is None:
            self._is_fresh[:] = True
        else:
            self._is_fresh[env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Difference the vertical velocity and return the penalty.

        Args:
            env: The environment instance.
            asset_cfg: The articulation whose root link carries the trunk.

        Returns:
            The penalty in ``(-inf, 0]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
        acceleration = (vertical_speed - self._previous_speed) / env.step_dt
        acceleration = torch.where(self._is_fresh, torch.zeros_like(acceleration), acceleration)
        self._previous_speed = vertical_speed.clone()
        self._is_fresh[:] = False
        return -torch.abs(acceleration)


def body_ang_vel_at_height(
    env: ManagerBasedRLEnv,
    height_low: float,
    height_high: float,
    tilt_full_deg: float,
    tilt_zero_deg: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the roll and pitch rate of a body, but only once the robot is up and near vertical.

    Ported from addendum section 3.3 (``body_ang_vel_at_height``). It is
    :func:`body_ang_vel_xy_l2` behind a height-times-tilt smoothstep gate, and the gate is the whole
    point: an ungated rotation penalty is an attempt tax on exactly the thrashing a robot has to do
    to flip itself off its back, and upstream measured that taxing it makes "do nothing" win. Above
    the gate it damps the overshoot-and-tip loop at the top of the rise instead.

    Args:
        env: The environment instance.
        height_low: Trunk height [m] below which the gate is fully closed.
        height_high: Trunk height [m] above which the height half of the gate is fully open.
        tilt_full_deg: Trunk tilt [deg] below which the tilt half of the gate is fully open.
        tilt_zero_deg: Trunk tilt [deg] above which the tilt half of the gate is fully closed.
        asset_cfg: The articulation and the bodies to measure.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.square(ang_vel_xy), dim=(1, 2))

    gate = _height_gate(_root_height_above_ground(env, asset), height_low, height_high)
    gate = gate * _tilt_gate(asset.data.root_link_quat_w.torch, tilt_full_deg, tilt_zero_deg)
    return cost * gate


def body_upright_linear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the cosine of the trunk's tilt from vertical.

    Ported from addendum section 3.3 (``body_upright_linear``). Returns +1 upright, 0 horizontal and
    -1 inverted, so unlike the Gaussian :func:`upright` it keeps a strong gradient at large tilt --
    which is where a robot on its back starts. :func:`upright_gaussian_at_height` is its
    near-vertical counterpart, and the stand-up task carries both layers.

    Args:
        env: The environment instance.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[-1, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return 1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch)


def upright_gaussian_at_height(
    env: ManagerBasedRLEnv,
    std: float,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a near-vertical trunk, paid in proportion to how far the robot has risen.

    Ported from addendum section 3.3 (``upright_gaussian_at_height``). The height gate is what
    closes the "crouch low and vertical" exploit that an ungated upright reward opens, and the
    Gaussian has its gradient exactly where :func:`body_upright_linear` runs out of steam.

    Args:
        env: The environment instance.
        std: Width of the Gaussian kernel, in units of ``1 - cos(tilt)``.
        height_low: Trunk height [m] below which the gate is fully closed.
        height_high: Trunk height [m] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright_score = torch.exp(-_trunk_tilt_squared(asset.data.root_link_quat_w.torch) / std**2)
    return upright_score * _height_gate(_root_height_above_ground(env, asset), height_low, height_high)


def standing_composite_score(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the goal state as one multiplicative score over height, uprightness and pose.

    Ported from addendum section 3.3 (``standing_composite_score``). The product of the three
    Gaussians -- rather than their sum, which the separate height, upright and posture terms already
    provide -- is what makes "standing" a single attractor: two out of three earns almost nothing,
    so the policy cannot settle on a tall crouch or an upright sit. The widths are deliberately
    broad, because a tight product scores numerically zero everywhere except at the goal and has no
    gradient to follow there.

    Args:
        env: The environment instance.
        target_height: Trunk height of the goal state [m].
        height_std: Width of the Gaussian kernel on the height [m].
        upright_std: Width of the Gaussian kernel on the tilt, in units of ``1 - cos(tilt)``.
        pose_std: Width of the Gaussian kernel on the joint-position RMS error [rad].
        asset_cfg: The articulation and the joints scored against the stand pose. Its root link
            carries the trunk height and orientation.

    Returns:
        The reward in ``(0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_score = torch.exp(-(((_root_height_above_ground(env, asset) - target_height) / height_std) ** 2))
    upright_score = torch.exp(-_trunk_tilt_squared(asset.data.root_link_quat_w.torch) / upright_std**2)
    pose_error = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    pose_score = torch.exp(-torch.square(pose_error).mean(dim=-1) / pose_std**2)
    return height_score * upright_score * pose_score


class joint_torque_rate_l2(ManagerTermBase):
    """Penalize the squared change in applied joint torque between two control steps.

    Ported from addendum section 3.3 (``joint_torque_rate_l2``). It prices jitter rather than
    effort: the stock :func:`isaaclab.envs.mdp.joint_torques_l2` charges the torque itself, which a
    0.74 kg robot pushing itself off the floor cannot avoid, whereas the *rate* is what a chattering
    policy spends and a smooth one does not.

    Being a finite difference the term is stateful, so it is a class.

    Note:
        Upstream caches its previous torques on the environment with no reset hook, so the first
        step of every episode is charged against the last step of the previous one (addendum section
        7.18). This port clears the state on reset, as its sibling
        :class:`trunk_vertical_accel_penalty` does upstream. The weight is zero until late in the
        schedule, so the two behaviours are indistinguishable in practice.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the previous-torque buffer.

        Args:
            cfg: The term configuration, whose ``params`` carry the driven joints.
            env: The environment instance.

        Raises:
            ValueError: If ``asset_cfg`` is missing or selects no joints by name.
        """
        super().__init__(cfg, env)

        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        self._previous_torque = torch.zeros(env.num_envs, len(joint_ids), device=env.device)
        self._is_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Mark the environments that restarted, so their next difference is not charged.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are marked.
        """
        if env_ids is None:
            self._is_fresh[:] = True
        else:
            self._is_fresh[env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Difference the applied torque and return the cost.

        Args:
            env: The environment instance.
            asset_cfg: The articulation and the driven joints. Mandatory: the term sizes its state
                from the selection at construction time.

        Returns:
            The cost in ``[0, inf)``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
        rate = torque - self._previous_torque
        rate = torch.where(self._is_fresh.unsqueeze(-1), torch.zeros_like(rate), rate)
        self._previous_torque = torque.clone()
        self._is_fresh[:] = False
        return torch.sum(torch.square(rate), dim=1)


"""
Rolling forward (roulade): the rotation accumulator and the terms that read it.
"""

_ROULADE_FORWARD_SIGN = 1.0
"""Sign of the body-frame pitch rate that counts as *forward* rotation.

Face down is a +90 degree pitch in this family's reset convention, so a forward roll is a positive
body-frame ``omega_y`` (addendum section 4.3).
"""

_ROULADE_HEAD_LATCH_WINDOW = (math.radians(20.0), math.radians(170.0))
"""Rotation window [rad] inside which head-ground contact latches the roll as over-the-head.

In a real roulade the head plants at 60 to 120 degrees of body rotation; the window is generous
around that, so contact before the robot has committed and contact once it is already on its back
do not count (addendum section 4.3).
"""

_ROULADE_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
"""The flat top of the head shell, as a unit vector in the head body's own frame.

Upstream measured it as world-up expressed in ``jaw_soft``'s frame with the robot settled at the
stand pose (addendum section 4.3).
"""

_ROULADE_HEAD_TOP_DOWN_MIN = 0.3
"""How far the head top must point at the floor before a head contact counts, as ``dot(axis, -z)``.

Upstream's measured landmarks at a 110 degree trunk pitch: a passive face-plant with the neck at the
stand pose reads +0.6, a full chin tuck reads -0.99. The threshold accepts partial tucks while
staying far away from a face or side-shell contact, which is what stops the policy rolling over its
shoulder instead of over its head.
"""

_ROULADE_SAGITTAL_FULL = 0.5
_ROULADE_SAGITTAL_ZERO = 0.866
"""Bounds on ``|lateral axis z|`` between which rotation credit fades from full to none.

``sin(30 deg)`` and ``sin(60 deg)``. In a clean forward roll the body's lateral axis stays
horizontal for *any* amount of pure pitch and its world-z component stays near zero; it grows toward
one as the roll goes over a shoulder, which upstream's run-5 policy discovered as a lower-energy
cheat. Above the upper bound a side roll does not count as rotation at all.
"""


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z component of a body's lateral (y) axis. Zero when the body is sagittally flat.

    Note:
        Upstream's quaternions are ``(w, x, y, z)`` and Isaac Lab's are ``(x, y, z, w)``, so its
        ``2 (q_y q_z + q_w q_x)`` reads columns 1, 2, 3 and 0 rather than 2, 3, 0 and 1.

    Args:
        quat: Body orientation in (x, y, z, w). Shape is (num_envs, 4).

    Returns:
        The lateral axis' world-z component in ``[-1, 1]``. Shape is (num_envs,).
    """
    return 2.0 * (quat[:, 1] * quat[:, 2] + quat[:, 3] * quat[:, 0])


def _head_top_is_down(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Whether the flat top of the head shell is pointing at the floor.

    Args:
        env: The environment instance.
        asset_cfg: The articulation and the single body carrying the head shells.

    Returns:
        Whether the head top faces down. Shape is (num_envs,).

    Raises:
        ValueError: If ``asset_cfg`` does not select exactly one body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_names is None or len(asset_cfg.body_ids) != 1:
        raise ValueError(
            "The roll's head terms measure the orientation of a single head body; 'asset_cfg' must"
            f" select exactly one by name. Received: {asset_cfg.body_names}."
        )
    head_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    axis_b = torch.tensor(_ROULADE_HEAD_TOP_AXIS, device=head_quat_w.device).expand_as(head_quat_w[..., :3])
    axis_world_z = math_utils.quat_apply(head_quat_w, axis_b).squeeze(1)[:, 2]
    return axis_world_z < -_ROULADE_HEAD_TOP_DOWN_MIN


def _any_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Whether any body a contact sensor senses is touching something.

    Upstream reads its sensors' boolean ``found`` field, which is set for every contact the solver
    keeps. Isaac Lab reports the net contact force per sensing body instead, so "found" is a
    non-zero force -- the same signal, since the shapes carry no collision margin.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor and the bodies to read.

    Returns:
        Whether at least one selected body is in contact. Shape is (num_envs,).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > 0.0).any(dim=-1)


def _update_roulade_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    support_sensor_cfg: SceneEntityCfg,
    head_sensor_cfg: SceneEntityCfg,
) -> None:
    """Advance the rotation accumulator and the head latch by one control step.

    Ported verbatim from addendum section 4.3 (``_update_roulade_accum``). Two gates make the
    integral mean "rotation of a *roulade*" rather than "rotation":

    * **Support gate.** Rotation is integrated only while some part of the robot touches the ground.
      A roulade is a supported motion, and upstream's first run discovered that without this gate
      the optimal policy is a ballistic whip that earns the same rotation sooner.
    * **Sagittal gate.** Rotation is scaled by a smoothstep on how flat the roll is, so going over a
      shoulder earns little and going over the side earns nothing.

    The frontier only moves forward, so rocking backwards neither pays nor un-pays, and the head
    latch is set once head-ground contact happens with the flat top of the head pointing down while
    the accumulated angle is inside :data:`_ROULADE_HEAD_LATCH_WINDOW`.

    The update is guarded by the global step count, so calling it from several terms in one control
    step integrates once.

    Args:
        env: The environment instance.
        asset_cfg: The articulation, whose root link carries the trunk and whose single selected
            body carries the head shells.
        support_sensor_cfg: The contact sensor reporting whether the robot touches the ground.
        head_sensor_cfg: The contact sensor reporting whether the head touches the ground.
    """
    state = roulade_roll_state(env)
    step = int(env.common_step_counter)
    if step == state.last_update_step:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    forward_rate = _ROULADE_FORWARD_SIGN * asset.data.root_link_ang_vel_b.torch[:, 1]
    delta = torch.nan_to_num(forward_rate, nan=0.0) * env.step_dt
    delta = delta * _any_contact(env, support_sensor_cfg).float()
    lateral = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w.torch), nan=1.0).abs()
    sagittal_gate = _smoothstep((_ROULADE_SAGITTAL_ZERO - lateral) / (_ROULADE_SAGITTAL_ZERO - _ROULADE_SAGITTAL_FULL))
    # in place throughout, so the buffers stay the ones allocated on the environment: a rollout runs
    # under ``torch.inference_mode`` and rebinding them there would leave the reset event unable to
    # write them from outside it
    state.accumulated_angle.add_(delta * sagittal_gate)
    torch.maximum(state.frontier, state.accumulated_angle, out=state.frontier)

    latch_lo, latch_hi = _ROULADE_HEAD_LATCH_WINDOW
    in_window = (state.accumulated_angle > latch_lo) & (state.accumulated_angle < latch_hi)
    head_contact = _any_contact(env, head_sensor_cfg)
    state.head_latch.logical_or_(head_contact & in_window & _head_top_is_down(env, asset_cfg))
    state.last_update_step = step


def roulade_completion_gate(
    env: ManagerBasedRLEnv, gate_lo: float, gate_hi: float, require_head: bool = True
) -> torch.Tensor:
    """Smoothstep on the rotation frontier: zero below ``gate_lo``, one above ``gate_hi``.

    Ported from addendum section 4.3 (``_roulade_completion_gate``). This is what replaces a phase
    clock: the landing rewards can only be opened by *having rotated* -- while supported, and in the
    sagittal plane -- so a standing spawn cannot farm them by doing nothing and a ballistic flip
    cannot open them at all.

    It is public because it is the whole landing annuity in one function: a new gated roll reward
    multiplies by this and nothing else, and a test that wants to know whether completion is
    *reachable* reads it directly.

    Args:
        env: The environment instance.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        require_head: Whether the gate additionally requires the over-the-head latch. Defaults to
            True, which is what every shipped landing reward passes -- "went over the head" is a
            requirement of the trick rather than a bonus.

    Returns:
        The gate in ``[0, 1]``. Shape is (num_envs,).
    """
    state = roulade_roll_state(env)
    gate = _smoothstep((state.frontier - gate_lo) / max(gate_hi - gate_lo, 1e-6))
    if require_head:
        gate = gate * state.head_latch.float()
    return gate


def roulade_progress(
    env: ManagerBasedRLEnv,
    target_angle: float,
    max_paid_rate: float,
    support_sensor_cfg: SceneEntityCfg,
    head_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Pay increments of the rotation frontier, up to one full roll, at a capped rate.

    Ported from addendum section 4.4 (``roulade_progress``). It is the one dense task signal during
    the roll, and it is potential-based: a full roll pays ``target_angle`` worth of reward in total
    however it is spread out, camping anywhere pays zero per step, rocking back below the frontier
    pays zero, and spinning past the target is clamped.

    The rate cap **forfeits** rather than defers: the paid pointer still jumps to the frontier, so a
    violent whip collects strictly *less* total progress reward than a controlled roll rather than
    the same amount sooner.

    This is also the term that advances the accumulator every control step -- see
    :func:`_update_roulade_state`. The other roll terms only read the frontier it leaves behind, so
    it has to be **declared first** among them; Isaac Lab evaluates reward terms in declaration
    order.

    Args:
        env: The environment instance.
        target_angle: Rotation [rad] that counts as a complete roll, normally ``2 pi``.
        max_paid_rate: Largest rotation rate [rad/s] that is paid for. Rotation faster than this
            forfeits the excess.
        support_sensor_cfg: The contact sensor reporting whether the robot touches the ground, which
            gates the accumulator.
        head_sensor_cfg: The contact sensor reporting whether the head touches the ground, which
            drives the over-the-head latch.
        asset_cfg: The articulation, whose root link carries the trunk and whose single selected
            body carries the head shells.

    Returns:
        The reward in ``[0, inf)``, normalized so a roll at the cap pays about one per step.
        Shape is (num_envs,).
    """
    _update_roulade_state(env, asset_cfg, support_sensor_cfg, head_sensor_cfg)
    state = roulade_roll_state(env)
    new_paid = torch.clamp(state.frontier, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(state.paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    torch.maximum(state.paid, new_paid, out=state.paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    angle_lo: float,
    angle_hi: float,
    rate_norm: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward pivoting over the head: head-ground contact while rotating forward, mid-roll.

    Ported from addendum section 4.4 (``roulade_head_pivot``). The rate factor is the anti-camping
    guard -- a robot resting face down with its head on the floor is not rotating and earns nothing,
    so the term pays for pivoting *over* the head rather than for touching it. The head-top factor
    is the gradient that teaches the chin tuck: any head contact mid-roll pays 30 %, contact on the
    flat top pays full, which is the same distinction the latch makes but continuous.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor reporting whether the head touches the ground.
        angle_lo: Rotation [rad] below which the term is off.
        angle_hi: Rotation [rad] above which the term is off.
        rate_norm: Forward rotation rate [rad/s] at which the rate factor saturates.
        asset_cfg: The articulation, whose root link carries the trunk and whose single selected
            body carries the head shells.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    state = roulade_roll_state(env)
    contact = _any_contact(env, sensor_cfg).float()
    in_window = ((state.accumulated_angle > angle_lo) & (state.accumulated_angle < angle_hi)).float()
    forward_rate = _ROULADE_FORWARD_SIGN * asset.data.root_link_ang_vel_b.torch[:, 1]
    rate = torch.clamp(torch.nan_to_num(forward_rate, nan=0.0) / rate_norm, 0.0, 1.0)
    head_top = 0.3 + 0.7 * _head_top_is_down(env, asset_cfg).float()
    return contact * in_window * rate * head_top


def roulade_landing_composite(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward standing at the stand pose, but only once the roll has been completed over the head.

    Ported from addendum section 4.4 (``roulade_landing_composite``). It is
    :func:`standing_composite_score` behind the completion gate, and it is the dominant attractor of
    the whole recipe: finishing on the feet and staying there pays every step, while a standing
    spawn that never rolls collects nothing.

    Args:
        env: The environment instance.
        target_height: Trunk height of the goal state [m].
        height_std: Width of the Gaussian kernel on the height [m].
        upright_std: Width of the Gaussian kernel on the tilt, in units of ``1 - cos(tilt)``.
        pose_std: Width of the Gaussian kernel on the joint-position RMS error [rad].
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation and the joints scored against the stand pose. Its root link
            carries the trunk height and orientation.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    score = standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        asset_cfg=asset_cfg,
    )
    return score * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_upright_after_roll(
    env: ManagerBasedRLEnv,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a vertical trunk once the roll is complete, linearly in ``cos(tilt)``.

    Ported from addendum section 4.4 (``roulade_upright_after_roll``). It is the bootstrap layer for
    :func:`roulade_landing_composite`, whose product of Gaussians is numerically zero far from the
    goal: this one has gradient from any orientation. The gate is what makes it safe -- an
    always-on upright reward opposes the flip, which is the failure that killed upstream's earlier
    attempt at this task.

    Args:
        env: The environment instance.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright_score = 1.0 - _trunk_tilt_squared(asset.data.root_link_quat_w.torch)
    return torch.clamp(upright_score, min=0.0) * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_height_after_roll(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward standing height once the roll is complete, with a broad Gaussian.

    Ported from addendum section 4.4 (``roulade_height_after_roll``). The second bootstrap layer
    under the landing composite; see :func:`roulade_upright_after_roll` for why they are gated.

    Args:
        env: The environment instance.
        target_height: Trunk height to reach [m], measured above the environment origin.
        std: Width of the Gaussian kernel [m].
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_score = torch.exp(-(((_root_height_above_ground(env, asset) - target_height) / std) ** 2))
    return height_score * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_landing_sharp(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the last mile of the landing: tight Gaussians on height and tilt, behind the gate.

    Ported from addendum section 4.4 (``roulade_landing_sharp``). This is the stand-up task's
    two-layer lesson applied to the landing: upstream's run-3 policies all parked in the same basin
    a centimetre low and 27 degrees off vertical, where the broad composite already scores about
    0.5 and has nothing left to pull with. At that pose this layer scores about 0.1 and at vertical
    it pays about 1, which is the differential that finishes the rise.

    Args:
        env: The environment instance.
        target_height: Trunk height of the goal state [m].
        height_std: Width of the Gaussian kernel on the height [m].
        upright_std: Width of the Gaussian kernel on the tilt, in units of ``1 - cos(tilt)``.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright_score = torch.exp(-_trunk_tilt_squared(asset.data.root_link_quat_w.torch) / upright_std**2)
    height_score = torch.exp(-(((_root_height_above_ground(env, asset) - target_height) / height_std) ** 2))
    return upright_score * height_score * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_stand_tax(
    env: ManagerBasedRLEnv,
    target_height: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Charge every step spent below standing height once the roll is complete.

    Ported from addendum section 4.4 (``roulade_stand_tax``). The gated landing rewards make
    standing better than lying in a heap, but they leave the heap itself *free*, which is a
    comfortable basin -- the same static-sit trap the stand-up task had to break. Taxing it makes
    the basin net negative. The gate keeps the roll itself untaxed and requires the head latch, so
    an episode that never rolled is never punished into avoidance behaviour.

    The term **negates itself** and is therefore configured with a *positive* weight.

    Args:
        env: The environment instance.
        target_height: Trunk height [m] below which the shortfall is charged.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The penalty in ``(-inf, 0]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    shortfall = torch.clamp(target_height - _root_height_above_ground(env, asset), min=0.0)
    return -shortfall * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_rise_velocity(
    env: ManagerBasedRLEnv,
    max_height: float,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward rising out of the roll, gated to its late phase.

    Ported from addendum section 4.4 (``roulade_rise_velocity``). It is :func:`com_upward_velocity`
    behind a gate that opens around 180 degrees -- on the back -- because from there the rest of a
    roulade *is* the face-up recovery problem, and the stand-up task proved end-state rewards have
    no gradient at all at zero motion there. The gate keeps pre-roll bobbing from earning anything
    and the ceiling keeps hopping from farming it.

    Args:
        env: The environment instance.
        max_height: Trunk height [m] above which the reward is switched off.
        gate_lo: Rotation [rad] below which the gate is shut.
        gate_hi: Rotation [rad] above which the gate is fully open.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The reward in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    below_ceiling = (_root_height_above_ground(env, asset) < max_height).float()
    vertical_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w.torch[:, 2], nan=0.0)
    reward = torch.clamp(vertical_speed, min=0.0) * below_ceiling
    return reward * roulade_completion_gate(env, gate_lo, gate_hi)


def roulade_overspeed_penalty(
    env: ManagerBasedRLEnv, omega_max: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize rotating faster than a threshold, quadratically in the excess.

    Ported from addendum section 4.4 (``roulade_overspeed_penalty``). It complements the paid-rate
    cap in :func:`roulade_progress`: the cap removes the *incentive* to whip, this adds a *cost*, so
    violent is strictly worse than controlled rather than merely not better. Upstream measured the
    natural over-the-top transit of this 10 cm robot at 3.5 to 5.5 rad/s and set the threshold above
    it, so a controlled roll never touches this term.

    Args:
        env: The environment instance.
        omega_max: Forward rotation rate [rad/s] above which the excess is charged.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rate = torch.nan_to_num(asset.data.root_link_ang_vel_b.torch[:, 1], nan=0.0)
    return torch.clamp(rate.abs() - omega_max, min=0.0).pow(2)


def roulade_flatness_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize tipping out of the sagittal plane, quadratically in the lateral axis' world-z.

    Ported from addendum section 4.4 (``roulade_flatness_penalty``). Zero when standing and zero
    through an arbitrarily deep *clean* forward roll, because pure pitch keeps the lateral axis
    horizontal; up to one when tipped fully onto a shoulder. The accumulator's own sagittal gate
    already makes a side roll unprofitable, and this term is the per-step gradient that steers back
    toward the plane.

    Args:
        env: The environment instance.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, 1]``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w.torch), nan=0.0).pow(2)


def roulade_sagittal_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize body-frame roll and yaw rate, leaving the roll axis free.

    Ported from addendum section 4.4 (``roulade_sagittal_penalty``). The pitch rate is the trick, so
    unlike :func:`body_ang_vel_xy_l2` -- which the roll task keeps at a 25 times lighter weight for
    exactly this reason -- this term charges the *other* two axes.

    Args:
        env: The environment instance.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rate = asset.data.root_link_ang_vel_b.torch
    return torch.nan_to_num(rate[:, 0].pow(2) + rate[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize sideways travel, quadratically in the body-frame lateral velocity.

    Ported from addendum section 4.4 (``roulade_lateral_velocity_penalty``). Keeps the roll straight
    where :func:`roulade_flatness_penalty` keeps it upright.

    Args:
        env: The environment instance.
        asset_cfg: The articulation whose root link carries the trunk.

    Returns:
        The cost in ``[0, inf)``. Shape is (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b.torch[:, 1].pow(2), nan=0.0)
