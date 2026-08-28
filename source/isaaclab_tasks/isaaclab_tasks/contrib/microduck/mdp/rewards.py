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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.string import resolve_matching_names_values

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
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_names is None:
        projected_gravity_b = asset.data.projected_gravity_b.torch
    else:
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
    ) -> torch.Tensor:
        """Advance the moving average and return the penalty.

        Args:
            env: The environment instance.
            command_name: Name of the head-pose command term.
            tau_s: Time constant of the moving average [s].
            asset_cfg: The articulation and the head joints to track. Mandatory: the term sizes its
                state from the selection at construction time, and the command columns are paired
                with the joints positionally.

        Returns:
            The penalty in ``(-inf, 0]``. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        measured = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        error = (measured - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]) - command

        alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
        self._error_ema = (1.0 - alpha) * self._error_ema + alpha * error
        return -self._error_ema.abs().mean(dim=-1)


def body_pose_tracking_6d(
    env: ManagerBasedRLEnv,
    command_name: str,
    nominal_height: float,
    xy_std: float,
    z_std: float,
    angle_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of a six-dimensional trunk-pose command, averaged over the six axes.

    Ported from reference section 6 (``body_pose_tracking_6d``). The command is a delta from the
    nominal stand: ``(dx, dy, dz)`` [m] measured from the environment origin, with the vertical
    target at ``nominal_height + dz``, and ``(droll, dpitch, dyaw)`` [rad]. Each axis contributes
    its own Gaussian and the six are averaged, so no single axis can dominate.

    There is no stock counterpart. :func:`isaaclab.envs.mdp.position_command_error` and
    :func:`~isaaclab.envs.mdp.orientation_command_error` measure a pose command as an unbounded
    error norm against a pose command term, not as a per-axis Gaussian around a nominal stand.

    Args:
        env: The environment instance.
        command_name: Name of the six-dimensional body-pose command term.
        nominal_height: Trunk height the vertical command is measured from [m].
        xy_std: Width of the Gaussian kernel on the horizontal position [m].
        z_std: Width of the Gaussian kernel on the height [m].
        angle_std: Width of the Gaussian kernel on each Euler angle [rad].
        asset_cfg: The articulation whose root link carries the trunk pose.

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

    reward = torch.exp(-((x_error / xy_std) ** 2))
    reward = reward + torch.exp(-((y_error / xy_std) ** 2))
    reward = reward + torch.exp(-((z_error / z_std) ** 2))
    reward = reward + torch.exp(-((roll_error / angle_std) ** 2))
    reward = reward + torch.exp(-((pitch_error / angle_std) ** 2))
    reward = reward + torch.exp(-((yaw_error / angle_std) ** 2))
    return reward / 6.0


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
