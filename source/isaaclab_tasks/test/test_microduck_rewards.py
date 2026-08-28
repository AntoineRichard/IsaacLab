# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the MicroDuck reward and termination terms.

Every term reads a handful of articulation, sensor and command tensors and returns a number per
environment, so the whole suite runs against doubles rather than a simulated scene: the states are
crafted, and each expected value is worked out by hand from the upstream formula it is ported from
(``artifacts/microduck/upstream_reference.md`` sections 5 and 6). A live scene would only make the
same arithmetic harder to read and impossible to pin exactly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest
import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg

import isaaclab_tasks.contrib.microduck.mdp as mdp

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


HEAD_JOINT_IDS = [0, 1, 2, 3]
"""Joint slots the doubles below fill with ``(neck_pitch, head_pitch, head_yaw, head_roll)``."""

FOOT_BODY_IDS = [0, 1]
"""Body/sensor slots the doubles below fill with the two feet."""


##
# Doubles
##


class _DummyTensorView:
    """Stands in for a ``ProxyArray``, which exposes its contents under ``.torch``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.torch = tensor


class _DummyData:
    """Articulation or sensor data whose fields are set by keyword."""

    def __init__(self, **fields: torch.Tensor | None) -> None:
        for name, value in fields.items():
            setattr(self, name, None if value is None else _DummyTensorView(value))


class _DummyAsset:
    def __init__(self, **fields: torch.Tensor | None) -> None:
        self.data = _DummyData(**fields)


class _DummySensor:
    """Contact sensor double.

    ``compute_first_contact`` mirrors the real sensor: a body has just landed when it is in contact
    and has been for no longer than the control step.
    """

    def __init__(self, **fields: torch.Tensor | None) -> None:
        self.data = _DummyData(**fields)

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> _DummyTensorView:
        contact_time = self.data.current_contact_time.torch
        return _DummyTensorView((contact_time > 0.0) & (contact_time <= dt + abs_tol))


class _DummyCommandManager:
    def __init__(self, commands: dict[str, torch.Tensor]) -> None:
        self._commands = commands

    def get_command(self, name: str) -> torch.Tensor:
        return self._commands[name]


class _DummyScene:
    def __init__(self, assets: dict, sensors: dict, env_origins: torch.Tensor) -> None:
        self._assets = assets
        self.sensors = sensors
        self.env_origins = env_origins

    def __getitem__(self, name: str):
        return self._assets[name]


class _DummyEnv:
    """Minimal environment double for reward and termination terms."""

    def __init__(
        self,
        num_envs: int = 2,
        device: str = "cpu",
        assets: dict | None = None,
        sensors: dict | None = None,
        commands: dict[str, torch.Tensor] | None = None,
        env_origins: torch.Tensor | None = None,
        step_dt: float = 0.02,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self.scene = _DummyScene(
            assets or {},
            sensors or {},
            torch.zeros(num_envs, 3, device=device) if env_origins is None else env_origins,
        )
        self.command_manager = _DummyCommandManager(commands or {})

    def as_env(self) -> ManagerBasedRLEnv:
        return cast("ManagerBasedRLEnv", self)


def _entity(name: str, joint_ids=None, body_ids=None, joint_names=None, body_names=None) -> SceneEntityCfg:
    """Build an already-resolved scene entity configuration.

    The managers resolve names to indices before a term ever runs, so the doubles skip straight to
    the resolved form.
    """
    cfg = SceneEntityCfg(name)
    if joint_ids is not None:
        cfg.joint_ids = joint_ids
        cfg.joint_names = joint_names
    if body_ids is not None:
        cfg.body_ids = body_ids
        cfg.body_names = body_names
    return cfg


##
# Velocity tracking (reference section 5)
##


def _velocity_env(lin_vel_b: list[float], ang_vel_b: list[float], command: list[float]) -> _DummyEnv:
    robot = _DummyAsset(
        root_link_lin_vel_b=torch.tensor([lin_vel_b]),
        root_link_ang_vel_b=torch.tensor([ang_vel_b]),
    )
    return _DummyEnv(num_envs=1, assets={"robot": robot}, commands={"twist": torch.tensor([command])})


def test_track_linear_velocity_is_one_when_the_command_is_met_exactly():
    """A robot moving at the commanded planar velocity with no vertical drift scores the maximum."""
    env = _velocity_env([0.3, -0.2, 0.0], [0.0, 0.0, 0.0], [0.3, -0.2, 0.0])

    reward = mdp.track_linear_velocity(env.as_env(), std=math.sqrt(0.1), command_name="twist")

    torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_track_linear_velocity_charges_vertical_drift_inside_the_exponent():
    """Upstream folds ``v_z^2`` into the tracking exponent, so a bounce costs tracking reward."""
    std = 0.5
    env = _velocity_env([0.3, -0.2, std], [0.0, 0.0, 0.0], [0.3, -0.2, 0.0])

    reward = mdp.track_linear_velocity(env.as_env(), std=std, command_name="twist")

    # only error is v_z = std, so the exponent is exactly -1
    torch.testing.assert_close(reward, torch.tensor([math.exp(-1.0)]))


def test_track_angular_velocity_charges_roll_and_pitch_rate_inside_the_exponent():
    """Upstream folds ``|w_xy|^2`` into the yaw-tracking exponent."""
    std = 0.5
    env = _velocity_env([0.0, 0.0, 0.0], [std, 0.0, 0.7], [0.0, 0.0, 0.7])

    reward = mdp.track_angular_velocity(env.as_env(), std=std, command_name="twist")

    torch.testing.assert_close(reward, torch.tensor([math.exp(-1.0)]))


def test_track_angular_velocity_is_one_when_the_yaw_rate_is_met_and_the_base_is_steady():
    """Meeting the yaw command with no roll or pitch rate scores the maximum."""
    env = _velocity_env([0.0, 0.0, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, 0.7])

    reward = mdp.track_angular_velocity(env.as_env(), std=math.sqrt(0.5), command_name="twist")

    torch.testing.assert_close(reward, torch.tensor([1.0]))


##
# Upright (reference section 5)
##


def test_upright_is_one_when_level_and_decays_with_tilt():
    """The Gaussian is measured on the selected body's projected gravity, not on an L2 penalty."""
    std = 0.2
    # env 0 level, env 1 pitched so the gravity direction has an x-component of ``std``
    gravity_b = torch.tensor([[0.0, 0.0, -1.0], [std, 0.0, -1.0]])
    robot = _DummyAsset(projected_gravity_b=gravity_b)
    env = _DummyEnv(num_envs=2, assets={"robot": robot})

    reward = mdp.upright(env.as_env(), std=std)

    torch.testing.assert_close(reward, torch.tensor([1.0, math.exp(-1.0)]))


def test_upright_reads_the_selected_body_rather_than_the_root():
    """With a body selected the term projects gravity into that body's frame."""
    std = math.sqrt(0.05)
    # 90 deg roll about x maps world -z onto body +y, so the xy tilt is a full unit
    quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]]])
    robot = _DummyAsset(
        body_link_quat_w=quat,
        GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -9.81]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    env = _DummyEnv(num_envs=1, assets={"robot": robot})

    level = mdp.upright(env.as_env(), std=std, asset_cfg=_entity("robot", body_ids=[0], body_names=["trunk"]))
    rolled = mdp.upright(env.as_env(), std=std, asset_cfg=_entity("robot", body_ids=[1], body_names=["other"]))

    torch.testing.assert_close(level, torch.tensor([1.0]))
    torch.testing.assert_close(rolled, torch.tensor([math.exp(-1.0 / 0.05)]))


##
# Posture (reference section 5, ``variable_posture``)
##


def _pose_mode_switch_term(command: torch.Tensor, joint_pos: torch.Tensor) -> tuple:
    num_envs = command.shape[0]
    robot = _DummyAsset(
        joint_pos=joint_pos,
        default_joint_pos=torch.zeros_like(joint_pos),
    )
    robot.joint_names = ["left_hip_pitch", "left_knee"]
    env = _DummyEnv(num_envs=num_envs, assets={"robot": robot}, commands={"twist": command})
    params = {
        "command_name": "twist",
        "std_standing": {r".*hip_pitch.*": 0.15, r".*knee.*": 0.15},
        "std_walking": {r".*hip_pitch.*": 0.4, r".*knee.*": 0.4},
        "std_running": {r".*hip_pitch.*": 0.4, r".*knee.*": 0.4},
        "walking_threshold": 0.01,
        "running_threshold": 1.5,
        "asset_cfg": _entity("robot", joint_ids=[0, 1], joint_names=["left_hip_pitch", "left_knee"]),
    }
    cfg = RewardTermCfg(func=mdp.pose_mode_switch, weight=1.0, params=params)
    return mdp.pose_mode_switch(cfg, env.as_env()), env, params


def test_pose_mode_switch_is_one_at_the_home_pose():
    """Standing exactly at the stand pose scores the maximum in every regime."""
    command = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    term, env, params = _pose_mode_switch_term(command, torch.zeros(2, 2))

    reward = term(env.as_env(), **params)

    torch.testing.assert_close(reward, torch.tensor([1.0, 1.0]))


def test_pose_mode_switch_tightens_the_standard_deviation_below_the_walking_threshold():
    """A standing environment is held to the tight standing std, a walking one to the loose one."""
    # both environments are 0.15 rad off the stand pose on both joints
    joint_pos = torch.full((2, 2), 0.15)
    command = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    term, env, params = _pose_mode_switch_term(command, joint_pos)

    reward = term(env.as_env(), **params)

    standing = math.exp(-((0.15 / 0.15) ** 2))
    walking = math.exp(-((0.15 / 0.4) ** 2))
    torch.testing.assert_close(reward, torch.tensor([standing, walking]))


def test_pose_mode_switch_counts_yaw_rate_towards_the_speed():
    """Upstream's speed is ``|v_xy| + |w_z|``, so turning on the spot is not standing."""
    joint_pos = torch.full((2, 2), 0.15)
    # the second environment has no linear command at all, only a yaw rate
    command = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    term, env, params = _pose_mode_switch_term(command, joint_pos)

    reward = term(env.as_env(), **params)

    assert reward[0].item() != pytest.approx(reward[1].item())
    torch.testing.assert_close(reward[1], torch.tensor(math.exp(-((0.15 / 0.4) ** 2))))


##
# Head pose (reference section 6)
##


def _head_env(joint_pos: torch.Tensor, command: torch.Tensor) -> _DummyEnv:
    robot = _DummyAsset(joint_pos=joint_pos, default_joint_pos=torch.zeros_like(joint_pos))
    return _DummyEnv(
        num_envs=joint_pos.shape[0],
        assets={"robot": robot},
        commands={"head_pose": command},
    )


def test_head_pose_tracking_is_one_when_the_head_sits_on_the_command():
    """A head that already holds the commanded delta scores the maximum."""
    command = torch.tensor([[0.05, -0.02, 0.07, 0.01]])
    env = _head_env(command.clone(), command)

    reward = mdp.head_pose_tracking(
        env.as_env(),
        command_name="head_pose",
        std=0.5,
        asset_cfg=_entity("robot", joint_ids=HEAD_JOINT_IDS),
    )

    torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_head_pose_tracking_averages_the_per_joint_gaussian():
    """Every joint one standard deviation off gives ``exp(-1)`` on every joint, hence on the mean."""
    std = 0.5
    command = torch.zeros(1, 4)
    env = _head_env(torch.full((1, 4), std), command)

    reward = mdp.head_pose_tracking(
        env.as_env(),
        command_name="head_pose",
        std=std,
        asset_cfg=_entity("robot", joint_ids=HEAD_JOINT_IDS),
    )

    torch.testing.assert_close(reward, torch.tensor([math.exp(-1.0)]))


def test_head_pose_tracking_is_a_mean_not_a_sum():
    """One joint off out of four costs only a quarter of the reward, not all of it."""
    std = 0.5
    joint_pos = torch.tensor([[std, 0.0, 0.0, 0.0]])
    env = _head_env(joint_pos, torch.zeros(1, 4))

    reward = mdp.head_pose_tracking(
        env.as_env(),
        command_name="head_pose",
        std=std,
        asset_cfg=_entity("robot", joint_ids=HEAD_JOINT_IDS),
    )

    torch.testing.assert_close(reward, torch.tensor([(3.0 + math.exp(-1.0)) / 4.0]))


def _head_bias_term(joint_pos: torch.Tensor, command: torch.Tensor, tau_s: float = 1.0, step_dt: float = 0.02):
    robot = _DummyAsset(joint_pos=joint_pos, default_joint_pos=torch.zeros_like(joint_pos))
    env = _DummyEnv(
        num_envs=joint_pos.shape[0],
        assets={"robot": robot},
        commands={"head_pose": command},
        step_dt=step_dt,
    )
    params = {
        "command_name": "head_pose",
        "tau_s": tau_s,
        "asset_cfg": _entity("robot", joint_ids=HEAD_JOINT_IDS),
    }
    cfg = RewardTermCfg(func=mdp.head_pose_bias_penalty, weight=1.0, params=params)
    return mdp.head_pose_bias_penalty(cfg, env.as_env()), env, params


def test_head_pose_bias_penalty_follows_the_analytic_exponential_moving_average():
    """After k steps at a constant error the average is ``err * (1 - (1 - alpha)^k)``."""
    error = 0.1
    tau_s, step_dt, steps = 1.0, 0.02, 7
    term, env, params = _head_bias_term(torch.full((1, 4), error), torch.zeros(1, 4), tau_s, step_dt)

    for _ in range(steps):
        penalty = term(env.as_env(), **params)

    alpha = step_dt / tau_s
    expected = -error * (1.0 - (1.0 - alpha) ** steps)
    torch.testing.assert_close(penalty, torch.tensor([expected]))


def test_head_pose_bias_penalty_is_never_positive():
    """The term negates itself, so it is used with a positive weight."""
    term, env, params = _head_bias_term(torch.tensor([[0.1, -0.2, 0.05, -0.01]]), torch.zeros(1, 4))

    penalty = term(env.as_env(), **params)

    assert torch.all(penalty <= 0.0)


def test_head_pose_bias_penalty_forgets_the_average_on_reset():
    """A new episode must not inherit the bias the previous one accumulated."""
    term, env, params = _head_bias_term(torch.full((2, 4), 0.1), torch.zeros(2, 4))
    for _ in range(20):
        term(env.as_env(), **params)
    settled = term(env.as_env(), **params).clone()

    term.reset(torch.tensor([0]))
    penalty = term(env.as_env(), **params)

    # environment 0 restarts from zero, environment 1 keeps accumulating
    assert penalty[0].item() > settled[0].item()
    assert penalty[1].item() < settled[1].item()
    torch.testing.assert_close(penalty[0], torch.tensor(-0.1 * 0.02))


##
# Body pose (reference section 6)
##


def _body_pose_env(pos_w: torch.Tensor, quat_w: torch.Tensor, command: torch.Tensor) -> _DummyEnv:
    robot = _DummyAsset(root_link_pos_w=pos_w, root_link_quat_w=quat_w)
    return _DummyEnv(
        num_envs=pos_w.shape[0],
        assets={"robot": robot},
        commands={"body_pose": command},
    )


def test_body_pose_tracking_6d_is_one_at_the_commanded_pose():
    """Sitting at the nominal height with a zero command scores the maximum on all six axes."""
    nominal_height = 0.095
    env = _body_pose_env(
        torch.tensor([[0.0, 0.0, nominal_height]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.zeros(1, 6),
    )

    reward = mdp.body_pose_tracking_6d(
        env.as_env(),
        command_name="body_pose",
        nominal_height=nominal_height,
        xy_std=0.05,
        z_std=0.02,
        angle_std=math.radians(15.0),
    )

    torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_body_pose_tracking_6d_averages_the_six_axis_gaussians():
    """One axis one standard deviation off costs a sixth of the reward."""
    nominal_height, xy_std = 0.095, 0.05
    env = _body_pose_env(
        torch.tensor([[xy_std, 0.0, nominal_height]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.zeros(1, 6),
    )

    reward = mdp.body_pose_tracking_6d(
        env.as_env(),
        command_name="body_pose",
        nominal_height=nominal_height,
        xy_std=xy_std,
        z_std=0.02,
        angle_std=math.radians(15.0),
    )

    torch.testing.assert_close(reward, torch.tensor([(5.0 + math.exp(-1.0)) / 6.0]))


def test_body_pose_tracking_6d_measures_position_relative_to_the_environment_origin():
    """The trunk position is scored against the environment origin, not the world origin."""
    nominal_height = 0.095
    env = _body_pose_env(
        torch.tensor([[7.0, -3.0, nominal_height]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.zeros(1, 6),
    )
    env.scene.env_origins = torch.tensor([[7.0, -3.0, 0.0]])

    reward = mdp.body_pose_tracking_6d(
        env.as_env(),
        command_name="body_pose",
        nominal_height=nominal_height,
        xy_std=0.05,
        z_std=0.02,
        angle_std=math.radians(15.0),
    )

    torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_body_pose_tracking_6d_wraps_the_yaw_error():
    """A yaw command just across the +/- pi seam is a small error, not a full turn."""
    nominal_height, angle_std = 0.095, math.radians(15.0)
    yaw = math.pi - 0.5 * angle_std
    quat = torch.tensor([[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]])
    command = torch.zeros(1, 6)
    command[0, 5] = -math.pi + 0.5 * angle_std

    env = _body_pose_env(torch.tensor([[0.0, 0.0, nominal_height]]), quat, command)
    reward = mdp.body_pose_tracking_6d(
        env.as_env(),
        command_name="body_pose",
        nominal_height=nominal_height,
        xy_std=0.05,
        z_std=0.02,
        angle_std=angle_std,
    )

    # the wrapped error is exactly one standard deviation
    torch.testing.assert_close(reward, torch.tensor([(5.0 + math.exp(-1.0)) / 6.0]))


##
# Feet (reference section 5)
##


def _feet_env(
    current_air_time: torch.Tensor | None = None,
    current_contact_time: torch.Tensor | None = None,
    body_link_pos_w: torch.Tensor | None = None,
    body_link_lin_vel_w: torch.Tensor | None = None,
    command: torch.Tensor | None = None,
) -> _DummyEnv:
    num_envs = 1
    for tensor in (current_air_time, current_contact_time, body_link_pos_w, command):
        if tensor is not None:
            num_envs = tensor.shape[0]
            break
    sensor = _DummySensor(current_air_time=current_air_time, current_contact_time=current_contact_time)
    robot = _DummyAsset(body_link_pos_w=body_link_pos_w, body_link_lin_vel_w=body_link_lin_vel_w)
    return _DummyEnv(
        num_envs=num_envs,
        assets={"robot": robot},
        sensors={"contact_forces": sensor},
        commands={"twist": torch.zeros(num_envs, 3) if command is None else command},
    )


def test_feet_air_time_windowed_counts_the_feet_inside_the_window():
    """The reward is the number of feet whose current air time lies strictly inside the window."""
    air_time = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.2, 0.15], [0.4, 0.1]])
    command = torch.tensor([[0.3, 0.0, 0.0]] * 4)
    env = _feet_env(current_air_time=air_time, command=command)

    reward = mdp.feet_air_time_windowed(
        env.as_env(),
        sensor_cfg=_entity("contact_forces", body_ids=FOOT_BODY_IDS),
        command_name="twist",
        threshold_min=0.125,
        threshold_max=0.300,
        command_threshold=0.01,
    )

    # both grounded / one in window / both in window / one above and one below the window
    torch.testing.assert_close(reward, torch.tensor([0.0, 1.0, 2.0, 0.0]))


def test_feet_air_time_windowed_is_gated_on_the_command_magnitude():
    """A robot that is not asked to move earns nothing for lifting its feet."""
    air_time = torch.tensor([[0.2, 0.15], [0.2, 0.15]])
    command = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    env = _feet_env(current_air_time=air_time, command=command)

    reward = mdp.feet_air_time_windowed(
        env.as_env(),
        sensor_cfg=_entity("contact_forces", body_ids=FOOT_BODY_IDS),
        command_name="twist",
        threshold_min=0.125,
        threshold_max=0.300,
        command_threshold=0.01,
    )

    torch.testing.assert_close(reward, torch.tensor([0.0, 2.0]))


def test_foot_clearance_charges_height_error_weighted_by_planar_foot_speed():
    """The cost is ``sum(|h - target| * |v_xy|)``, gated on the command magnitude."""
    target_height = 0.02
    positions = torch.tensor([[[0.0, 0.0, 0.05], [0.0, 0.0, 0.02]]])
    velocities = torch.tensor([[[0.3, 0.4, 0.0], [1.0, 0.0, 0.0]]])
    env = _feet_env(
        body_link_pos_w=positions,
        body_link_lin_vel_w=velocities,
        command=torch.tensor([[0.3, 0.0, 0.0]]),
    )

    cost = mdp.foot_clearance(
        env.as_env(),
        target_height=target_height,
        command_name="twist",
        asset_cfg=_entity("robot", body_ids=FOOT_BODY_IDS),
        command_threshold=0.01,
    )

    # |0.05 - 0.02| * 5.0 for the first foot, zero height error for the second
    torch.testing.assert_close(cost, torch.tensor([0.03 * 0.5]))


def test_foot_clearance_measures_height_above_the_environment_origin():
    """On raised terrain the height is taken from the environment origin, not from z = 0."""
    positions = torch.tensor([[[0.0, 0.0, 1.05], [0.0, 0.0, 1.02]]])
    velocities = torch.tensor([[[0.3, 0.4, 0.0], [1.0, 0.0, 0.0]]])
    env = _feet_env(
        body_link_pos_w=positions,
        body_link_lin_vel_w=velocities,
        command=torch.tensor([[0.3, 0.0, 0.0]]),
    )
    env.scene.env_origins = torch.tensor([[0.0, 0.0, 1.0]])

    cost = mdp.foot_clearance(
        env.as_env(),
        target_height=0.02,
        command_name="twist",
        asset_cfg=_entity("robot", body_ids=FOOT_BODY_IDS),
        command_threshold=0.01,
    )

    torch.testing.assert_close(cost, torch.tensor([0.03 * 0.5]))


def test_foot_slip_charges_squared_planar_speed_of_feet_in_contact():
    """Only feet that are on the ground are charged, and the speed enters squared."""
    contact_time = torch.tensor([[0.1, 0.0]])
    velocities = torch.tensor([[[0.3, 0.4, 0.0], [2.0, 0.0, 0.0]]])
    env = _feet_env(
        current_contact_time=contact_time,
        body_link_lin_vel_w=velocities,
        command=torch.tensor([[0.3, 0.0, 0.0]]),
    )

    cost = mdp.foot_slip(
        env.as_env(),
        sensor_cfg=_entity("contact_forces", body_ids=FOOT_BODY_IDS),
        command_name="twist",
        asset_cfg=_entity("robot", body_ids=FOOT_BODY_IDS),
        command_threshold=0.01,
    )

    # 0.5^2 for the grounded foot; the airborne one is free to move
    torch.testing.assert_close(cost, torch.tensor([0.25]))


def _swing_height_term(target_height: float = 0.02, command: torch.Tensor | None = None):
    sensor = _DummySensor(current_contact_time=torch.zeros(1, 2))
    robot = _DummyAsset(body_link_pos_w=torch.zeros(1, 2, 3))
    env = _DummyEnv(
        num_envs=1,
        assets={"robot": robot},
        sensors={"contact_forces": sensor},
        commands={"twist": torch.tensor([[0.3, 0.0, 0.0]]) if command is None else command},
    )
    params = {
        "sensor_cfg": _entity("contact_forces", body_ids=FOOT_BODY_IDS),
        "asset_cfg": _entity("robot", body_ids=FOOT_BODY_IDS),
        "target_height": target_height,
        "command_name": "twist",
        "command_threshold": 0.01,
    }
    cfg = RewardTermCfg(func=mdp.foot_swing_height, weight=1.0, params=params)
    return mdp.foot_swing_height(cfg, env.as_env()), env, params, sensor, robot


def test_foot_swing_height_charges_the_peak_only_on_the_landing_step():
    """The relative peak-height error is charged once, when the foot touches down."""
    target_height = 0.02
    term, env, params, sensor, robot = _swing_height_term(target_height)

    # two airborne steps, the peak of the first foot reaching twice the target height
    for height in (0.03, 0.04):
        robot.data.body_link_pos_w.torch = torch.tensor([[[0.0, 0.0, height], [0.0, 0.0, 0.0]]])
        cost = term(env.as_env(), **params)
        torch.testing.assert_close(cost, torch.tensor([0.0]))

    # the first foot lands this step
    sensor.data.current_contact_time.torch = torch.tensor([[env.step_dt, 0.0]])
    cost = term(env.as_env(), **params)

    torch.testing.assert_close(cost, torch.tensor([(0.04 / target_height - 1.0) ** 2]))


def test_foot_swing_height_clears_the_peak_after_charging_it():
    """The next swing starts from a clean peak rather than from the previous one."""
    target_height = 0.02
    term, env, params, sensor, robot = _swing_height_term(target_height)

    robot.data.body_link_pos_w.torch = torch.tensor([[[0.0, 0.0, 0.04], [0.0, 0.0, 0.0]]])
    term(env.as_env(), **params)
    sensor.data.current_contact_time.torch = torch.tensor([[env.step_dt, 0.0]])
    term(env.as_env(), **params)

    # the foot stays down: nothing more may be charged
    sensor.data.current_contact_time.torch = torch.tensor([[2.0 * env.step_dt, 0.0]])
    cost = term(env.as_env(), **params)

    torch.testing.assert_close(cost, torch.tensor([0.0]))


def test_foot_swing_height_is_gated_on_the_command_magnitude():
    """A standing robot is not charged for how it puts its feet down."""
    term, env, params, sensor, robot = _swing_height_term(command=torch.zeros(1, 3))

    robot.data.body_link_pos_w.torch = torch.tensor([[[0.0, 0.0, 0.04], [0.0, 0.0, 0.0]]])
    term(env.as_env(), **params)
    sensor.data.current_contact_time.torch = torch.tensor([[env.step_dt, 0.0]])
    cost = term(env.as_env(), **params)

    torch.testing.assert_close(cost, torch.tensor([0.0]))


##
# Trunk penalties (reference section 5)
##


def test_body_ang_vel_xy_l2_ignores_the_yaw_rate():
    """Upstream deliberately leaves the commanded yaw rate out of this penalty."""
    ang_vel = torch.tensor([[[0.3, 0.4, 10.0]]])
    robot = _DummyAsset(body_link_ang_vel_w=ang_vel)
    env = _DummyEnv(num_envs=1, assets={"robot": robot})

    cost = mdp.body_ang_vel_xy_l2(env.as_env(), asset_cfg=_entity("robot", body_ids=[0]))

    torch.testing.assert_close(cost, torch.tensor([0.25]))


def _angular_momentum_env(ang_vel, lin_vel, positions=None, masses=None, inertia=None) -> _DummyEnv:
    num_bodies = len(ang_vel)
    positions = positions or [[0.0, 0.0, 0.0]] * num_bodies
    masses = masses or [1.0] * num_bodies
    inertia = inertia or [[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]] * num_bodies
    robot = _DummyAsset(
        body_mass=torch.tensor([masses]),
        body_inertia=torch.tensor([inertia]),
        body_com_pos_w=torch.tensor([positions]),
        body_com_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * num_bodies]),
        body_com_lin_vel_w=torch.tensor([lin_vel]),
        body_com_ang_vel_w=torch.tensor([ang_vel]),
    )
    return _DummyEnv(num_envs=1, assets={"robot": robot})


def test_angular_momentum_l2_is_zero_for_an_articulation_at_rest():
    """A motionless articulation carries no angular momentum."""
    env = _angular_momentum_env(ang_vel=[[0.0, 0.0, 0.0]], lin_vel=[[0.0, 0.0, 0.0]])

    cost = mdp.angular_momentum_l2(env.as_env())

    torch.testing.assert_close(cost, torch.tensor([0.0]))


def test_angular_momentum_l2_is_the_squared_spin_momentum_of_a_single_body():
    """With one body at the centre of mass the momentum reduces to ``I * w``."""
    env = _angular_momentum_env(ang_vel=[[1.0, 2.0, 3.0]], lin_vel=[[0.0, 0.0, 0.0]])

    cost = mdp.angular_momentum_l2(env.as_env())

    # diag(2, 3, 4) @ (1, 2, 3) = (2, 6, 12)
    torch.testing.assert_close(cost, torch.tensor([2.0**2 + 6.0**2 + 12.0**2]))


def test_angular_momentum_l2_includes_the_orbital_term_about_the_centre_of_mass():
    """Two counter-moving bodies carry momentum even though neither one spins."""
    env = _angular_momentum_env(
        ang_vel=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        lin_vel=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        positions=[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    )

    cost = mdp.angular_momentum_l2(env.as_env())

    # each body contributes r x v = (1, 0, 0) x (0, 1, 0) = (0, 0, 1)
    torch.testing.assert_close(cost, torch.tensor([4.0]))


def test_self_collision_cost_counts_the_bodies_touching_the_robot_itself():
    """One count per body that carries a filtered self-contact force above the threshold."""
    forces = torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]],
        ]
    )
    sensor = _DummySensor(force_matrix_w=forces)
    env = _DummyEnv(num_envs=3, sensors={"self_collision": sensor})

    cost = mdp.self_collision_cost(
        env.as_env(),
        sensor_cfg=_entity("self_collision", body_ids=[0, 1]),
        force_threshold=1.0,
    )

    torch.testing.assert_close(cost, torch.tensor([0.0, 1.0, 2.0]))


def test_self_collision_cost_ignores_forces_below_the_threshold():
    """A brush that does not register as a contact costs nothing."""
    forces = torch.tensor([[[[0.0, 0.0, 0.5]], [[0.0, 0.0, 0.0]]]])
    sensor = _DummySensor(force_matrix_w=forces)
    env = _DummyEnv(num_envs=1, sensors={"self_collision": sensor})

    cost = mdp.self_collision_cost(
        env.as_env(),
        sensor_cfg=_entity("self_collision", body_ids=[0, 1]),
        force_threshold=1.0,
    )

    torch.testing.assert_close(cost, torch.tensor([0.0]))


##
# Termination (reference section 6)
##


def _nan_env(num_envs: int = 4, sensor_forces: torch.Tensor | None = None) -> _DummyEnv:
    robot = _DummyAsset(
        joint_pos=torch.zeros(num_envs, 14),
        joint_vel=torch.zeros(num_envs, 14),
        root_link_pos_w=torch.zeros(num_envs, 3),
        root_link_quat_w=torch.zeros(num_envs, 4),
        root_link_lin_vel_w=torch.zeros(num_envs, 3),
        root_link_ang_vel_w=torch.zeros(num_envs, 3),
    )
    sensors = {}
    if sensor_forces is not None:
        sensors["contact_forces"] = _DummySensor(net_forces_w=sensor_forces)
    return _DummyEnv(num_envs=num_envs, assets={"robot": robot}, sensors=sensors)


def test_robot_state_is_nan_flags_only_the_broken_environments():
    """A single non-finite entry anywhere in the root or joint state condemns that environment."""
    env = _nan_env()
    env.scene["robot"].data.joint_pos.torch[1, 3] = float("nan")
    env.scene["robot"].data.root_link_quat_w.torch[2, 0] = float("inf")

    done = mdp.robot_state_is_nan(env.as_env())

    torch.testing.assert_close(done, torch.tensor([False, True, True, False]))


def test_robot_state_is_nan_checks_the_named_contact_sensors():
    """Upstream also guards the contact forces of the sensors it is handed."""
    forces = torch.zeros(4, 2, 3)
    forces[3, 1, 2] = float("nan")
    env = _nan_env(sensor_forces=forces)

    done = mdp.robot_state_is_nan(env.as_env(), sensor_names=("contact_forces",))

    torch.testing.assert_close(done, torch.tensor([False, False, False, True]))


def test_robot_state_is_nan_ignores_sensors_that_are_not_in_the_scene():
    """Naming a sensor the scene does not carry is not an error."""
    env = _nan_env()

    done = mdp.robot_state_is_nan(env.as_env(), sensor_names=("does_not_exist",))

    torch.testing.assert_close(done, torch.zeros(4, dtype=torch.bool))
