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

    ``compute_first_contact`` mirrors the real sensor: a body has just landed when its contact time
    is inside ``(0, dt + abs_tol)``. Critically it returns that mask as **float32 ones and zeros**,
    not as booleans, exactly as
    ``isaaclab_newton.sensors.contact_sensor.ContactSensor.compute_first_contact`` does through
    ``compute_first_transition_kernel``. A boolean double would hide the fact that the result
    cannot be handed to ``torch.where`` as a condition.
    """

    def __init__(self, **fields: torch.Tensor | None) -> None:
        self.data = _DummyData(**fields)

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> _DummyTensorView:
        contact_time = self.data.current_contact_time.torch
        return _DummyTensorView(((contact_time > 0.0) & (contact_time < dt + abs_tol)).float())


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
        common_step_counter: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        # the forward-roll accumulator integrates once per control step, guarded on this counter
        self.common_step_counter = common_step_counter
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


def test_upright_rejects_a_configuration_with_more_than_one_body():
    """A multi-body selection would squeeze the body axis away and slice bodies, not xy, instead."""
    quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
    robot = _DummyAsset(body_link_quat_w=quat, GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -9.81]]))
    env = _DummyEnv(num_envs=1, assets={"robot": robot})

    with pytest.raises(ValueError, match="single body"):
        mdp.upright(env.as_env(), std=0.2, asset_cfg=_entity("robot", body_ids=[0, 1], body_names=["trunk", "other"]))


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


def test_foot_swing_height_handles_the_float_landing_mask_the_sensor_returns():
    """The contact sensor reports the landing mask as floats, which is not a ``where`` condition.

    Regression: consuming ``compute_first_contact`` directly raised
    ``where expected condition to be a boolean tensor`` on the Newton backend.
    """
    term, env, params, sensor, _ = _swing_height_term()

    # the double must keep matching the real sensor, or this test stops covering anything
    assert sensor.compute_first_contact(env.step_dt).torch.dtype is torch.float32

    sensor.data.current_contact_time.torch = torch.tensor([[env.step_dt, 0.0]])
    cost = term(env.as_env(), **params)

    torch.testing.assert_close(cost, torch.tensor([1.0]))


##
# Term configuration errors
##


def test_pose_mode_switch_rejects_a_configuration_without_a_joint_selection():
    """Selecting every joint would silently pair the standard deviations with the wrong ones."""
    command = torch.zeros(1, 3)
    robot = _DummyAsset(joint_pos=torch.zeros(1, 2), default_joint_pos=torch.zeros(1, 2))
    robot.joint_names = ["left_hip_pitch", "left_knee"]
    env = _DummyEnv(num_envs=1, assets={"robot": robot}, commands={"twist": command})
    params = {
        "command_name": "twist",
        "std_standing": {r".*": 0.15},
        "std_walking": {r".*": 0.4},
        "std_running": {r".*": 0.4},
        "walking_threshold": 0.01,
        "running_threshold": 1.5,
        "asset_cfg": SceneEntityCfg("robot"),
    }
    cfg = RewardTermCfg(func=mdp.pose_mode_switch, weight=1.0, params=params)

    with pytest.raises(ValueError, match="select its joints by name"):
        mdp.pose_mode_switch(cfg, env.as_env())


def test_pose_mode_switch_rejects_a_configuration_without_an_asset():
    """A missing entity configuration names the parameter rather than raising a bare KeyError."""
    env = _DummyEnv(num_envs=1)
    cfg = RewardTermCfg(func=mdp.pose_mode_switch, weight=1.0, params={"command_name": "twist"})

    with pytest.raises(ValueError, match="asset_cfg"):
        mdp.pose_mode_switch(cfg, env.as_env())


def test_head_pose_bias_penalty_rejects_a_configuration_without_a_joint_selection():
    """The moving average is sized from the selection, and the command columns are paired with it."""
    robot = _DummyAsset(joint_pos=torch.zeros(1, 4), default_joint_pos=torch.zeros(1, 4))
    env = _DummyEnv(num_envs=1, assets={"robot": robot}, commands={"head_pose": torch.zeros(1, 4)})
    params = {"command_name": "head_pose", "tau_s": 1.0, "asset_cfg": SceneEntityCfg("robot")}
    cfg = RewardTermCfg(func=mdp.head_pose_bias_penalty, weight=1.0, params=params)

    with pytest.raises(ValueError, match="select its joints by name"):
        mdp.head_pose_bias_penalty(cfg, env.as_env())


def test_foot_swing_height_rejects_a_configuration_without_a_sensor():
    """The peak buffer is sized from the sensor, so it cannot be left out."""
    env = _DummyEnv(num_envs=1)
    cfg = RewardTermCfg(func=mdp.foot_swing_height, weight=1.0, params={"command_name": "twist"})

    with pytest.raises(ValueError, match="sensor_cfg"):
        mdp.foot_swing_height(cfg, env.as_env())


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


##
# Rolling forward (addendum sections 4.3 and 4.4)
##

ROULADE_TARGET_ANGLE = 2.0 * math.pi
"""One full roll [rad], which is what ``roulade_progress`` normalizes its payout against."""

ROULADE_MAX_PAID_RATE = 5.0
"""Largest rotation rate [rad/s] upstream pays for; faster rotation forfeits the excess."""

# xyzw quaternions with the properties the roll's gates key on
_LEVEL_QUAT = [0.0, 0.0, 0.0, 1.0]
_PITCHED_QUAT = [0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)]  # 90 deg about y
_ROLLED_QUAT = [math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)]  # 90 deg about x
_HEAD_DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]  # 180 deg about x, which turns the head top toward the floor


def _roulade_env(
    forward_rate: float = 0.0,
    quat: list[float] | None = None,
    head_quat: list[float] | None = None,
    supported: bool = True,
    head_contact: bool = False,
    height: float = 0.0,
    vertical_speed: float = 0.0,
    lateral_speed: float = 0.0,
    roll_rate: float = 0.0,
    yaw_rate: float = 0.0,
    step: int = 1,
) -> _DummyEnv:
    """One environment posed for the roll terms, with both contact sensors wired.

    The head body is body slot 0 of its own selection, and its orientation is set independently of
    the trunk's so the head-top test can be exercised on its own.
    """
    robot = _DummyAsset(
        root_link_ang_vel_b=torch.tensor([[roll_rate, forward_rate, yaw_rate]]),
        root_link_quat_w=torch.tensor([quat if quat is not None else _LEVEL_QUAT]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, height]]),
        root_link_lin_vel_w=torch.tensor([[0.0, 0.0, vertical_speed]]),
        root_link_lin_vel_b=torch.tensor([[0.0, lateral_speed, 0.0]]),
        body_link_quat_w=torch.tensor([[head_quat if head_quat is not None else _LEVEL_QUAT]]),
        joint_pos=torch.zeros(1, 2),
        default_joint_pos=torch.zeros(1, 2),
    )
    support_force = torch.tensor([[[0.0, 0.0, 5.0 if supported else 0.0]]])
    head_force = torch.tensor([[[0.0, 0.0, 5.0 if head_contact else 0.0]]])
    return _DummyEnv(
        num_envs=1,
        assets={"robot": robot},
        sensors={
            "robot_ground_contact": _DummySensor(net_forces_w=support_force),
            "head_ground_contact": _DummySensor(net_forces_w=head_force),
        },
        common_step_counter=step,
    )


def _roulade_head_cfg() -> SceneEntityCfg:
    """The articulation with its single head body selected, as the two accumulator terms take it."""
    return _entity("robot", body_ids=[0], body_names=["jaw_soft"])


def _advance(env: _DummyEnv, steps: int = 1) -> torch.Tensor:
    """Run ``roulade_progress`` for a number of control steps and return its last payout."""
    reward = torch.zeros(env.num_envs)
    for _ in range(steps):
        reward = mdp.roulade_progress(
            env.as_env(),
            target_angle=ROULADE_TARGET_ANGLE,
            max_paid_rate=ROULADE_MAX_PAID_RATE,
            support_sensor_cfg=_entity("robot_ground_contact"),
            head_sensor_cfg=_entity("head_ground_contact"),
            asset_cfg=_roulade_head_cfg(),
        )
        env.common_step_counter += 1
    return reward


def test_roulade_progress_pays_the_rotation_it_integrates():
    """A supported, sagittal roll pays its increment normalized by one full turn."""
    env = _roulade_env(forward_rate=1.0)

    reward = _advance(env)

    # 1 rad/s for 0.02 s is 0.02 rad of the 2*pi that a full roll pays for
    torch.testing.assert_close(reward, torch.tensor([0.02 / (0.02 * ROULADE_TARGET_ANGLE)]))
    torch.testing.assert_close(mdp.roulade_roll_state(env.as_env()).frontier, torch.tensor([0.02]))


def test_roulade_progress_integrates_once_per_control_step():
    """Several terms read the accumulator in one step, and it may only advance for the first."""
    env = _roulade_env(forward_rate=1.0)

    mdp.roulade_progress(
        env.as_env(),
        target_angle=ROULADE_TARGET_ANGLE,
        max_paid_rate=ROULADE_MAX_PAID_RATE,
        support_sensor_cfg=_entity("robot_ground_contact"),
        head_sensor_cfg=_entity("head_ground_contact"),
        asset_cfg=_roulade_head_cfg(),
    )
    second = mdp.roulade_progress(
        env.as_env(),
        target_angle=ROULADE_TARGET_ANGLE,
        max_paid_rate=ROULADE_MAX_PAID_RATE,
        support_sensor_cfg=_entity("robot_ground_contact"),
        head_sensor_cfg=_entity("head_ground_contact"),
        asset_cfg=_roulade_head_cfg(),
    )

    torch.testing.assert_close(mdp.roulade_roll_state(env.as_env()).frontier, torch.tensor([0.02]))
    # the frontier has not moved and the first call already paid it, so the second pays nothing
    torch.testing.assert_close(second, torch.tensor([0.0]))


def test_roulade_progress_forfeits_rotation_faster_than_the_cap():
    """Upstream's paid pointer jumps to the frontier, so the excess is lost rather than deferred."""
    env = _roulade_env(forward_rate=10.0)

    fast = _advance(env)
    # the robot stops dead, so a deferring implementation would pay the held-back rotation now
    env.scene["robot"].data.root_link_ang_vel_b.torch[:, 1] = 0.0
    after = _advance(env)

    torch.testing.assert_close(fast, torch.tensor([ROULADE_MAX_PAID_RATE / ROULADE_TARGET_ANGLE]))
    torch.testing.assert_close(after, torch.tensor([0.0]))
    # 10 rad/s for 0.02 s: the whole 0.2 rad is on the frontier, only 0.1 of it was ever paid
    torch.testing.assert_close(mdp.roulade_roll_state(env.as_env()).frontier, torch.tensor([0.2]))


def test_roulade_progress_pays_nothing_for_unsupported_rotation():
    """The support gate is what makes a ballistic flip worthless: a roulade never leaves the floor."""
    env = _roulade_env(forward_rate=3.0, supported=False)

    reward = _advance(env)

    torch.testing.assert_close(reward, torch.tensor([0.0]))
    torch.testing.assert_close(mdp.roulade_roll_state(env.as_env()).frontier, torch.tensor([0.0]))


def test_roulade_progress_pays_nothing_for_a_roll_over_the_shoulder():
    """The sagittal gate is zero beyond 60 degrees of lateral tilt, so a side roll is not rotation."""
    env = _roulade_env(forward_rate=3.0, quat=_ROLLED_QUAT)

    reward = _advance(env)

    torch.testing.assert_close(reward, torch.tensor([0.0]))
    torch.testing.assert_close(mdp.roulade_roll_state(env.as_env()).frontier, torch.tensor([0.0]))


def test_roulade_progress_does_not_pay_rocking_back_below_the_frontier():
    """The frontier only moves forward, so winding up and unwinding cannot be farmed."""
    env = _roulade_env(forward_rate=1.0)

    _advance(env)
    env.scene["robot"].data.root_link_ang_vel_b.torch[:, 1] = -1.0
    backwards = _advance(env)
    env.scene["robot"].data.root_link_ang_vel_b.torch[:, 1] = 1.0
    forwards = _advance(env)

    state = mdp.roulade_roll_state(env.as_env())
    torch.testing.assert_close(backwards, torch.tensor([0.0]))
    # back to where the frontier already was, so the recovered rotation is not paid a second time
    torch.testing.assert_close(forwards, torch.tensor([0.0]))
    torch.testing.assert_close(state.frontier, torch.tensor([0.02]))
    torch.testing.assert_close(state.accumulated_angle, torch.tensor([0.02]))


def test_the_head_latch_needs_contact_on_the_flat_top_inside_the_rotation_window():
    """All three conditions are load-bearing: a face-plant or a late touch must not latch."""
    inside_window = int(math.degrees(0.02 * 30.0))  # 30 control steps at 1 rad/s is about 34 degrees

    latched = _roulade_env(forward_rate=1.0, head_contact=True, head_quat=_HEAD_DOWN_QUAT)
    _advance(latched, steps=30)
    face_planted = _roulade_env(forward_rate=1.0, head_contact=True, head_quat=_LEVEL_QUAT)
    _advance(face_planted, steps=30)
    no_contact = _roulade_env(forward_rate=1.0, head_contact=False, head_quat=_HEAD_DOWN_QUAT)
    _advance(no_contact, steps=30)
    too_early = _roulade_env(forward_rate=1.0, head_contact=True, head_quat=_HEAD_DOWN_QUAT)
    _advance(too_early, steps=5)

    assert inside_window > 0
    assert bool(mdp.roulade_roll_state(latched.as_env()).head_latch[0])
    assert not bool(mdp.roulade_roll_state(face_planted.as_env()).head_latch[0])
    assert not bool(mdp.roulade_roll_state(no_contact.as_env()).head_latch[0])
    # 5 steps is 0.1 rad, short of the window's 20 degree lower edge
    assert not bool(mdp.roulade_roll_state(too_early.as_env()).head_latch[0])


def test_the_completion_gated_terms_are_shut_until_the_roll_is_finished_over_the_head():
    """Six rewards hang off one gate, and both of its conditions have to hold."""
    gate_lo, gate_hi = math.radians(260.0), math.radians(330.0)
    env = _roulade_env(quat=_LEVEL_QUAT, height=0.115)
    state = mdp.roulade_roll_state(env.as_env())

    shut = mdp.roulade_upright_after_roll(env.as_env(), gate_lo=gate_lo, gate_hi=gate_hi)
    state.frontier[:] = gate_hi
    without_latch = mdp.roulade_upright_after_roll(env.as_env(), gate_lo=gate_lo, gate_hi=gate_hi)
    state.head_latch[:] = True
    open_gate = mdp.roulade_upright_after_roll(env.as_env(), gate_lo=gate_lo, gate_hi=gate_hi)
    state.frontier[:] = 0.5 * (gate_lo + gate_hi)
    half_way = mdp.roulade_upright_after_roll(env.as_env(), gate_lo=gate_lo, gate_hi=gate_hi)

    torch.testing.assert_close(shut, torch.tensor([0.0]))
    torch.testing.assert_close(without_latch, torch.tensor([0.0]))
    # upright, so the term itself is 1 and what is measured is the gate
    torch.testing.assert_close(open_gate, torch.tensor([1.0]))
    # the smoothstep is symmetric, so its midpoint is exactly a half
    torch.testing.assert_close(half_way, torch.tensor([0.5]))


def test_roulade_head_pivot_pays_for_pivoting_rather_than_for_resting_on_the_head():
    """Contact times the rotation window times the rate, with the tuck worth the last 70 percent."""
    params = {"angle_lo": math.radians(30.0), "angle_hi": math.radians(240.0), "rate_norm": 2.0}
    sensor_cfg = _entity("head_ground_contact")

    def pivot(env: _DummyEnv) -> torch.Tensor:
        return mdp.roulade_head_pivot(env.as_env(), sensor_cfg=sensor_cfg, asset_cfg=_roulade_head_cfg(), **params)

    tucked = _roulade_env(forward_rate=1.0, head_contact=True, head_quat=_HEAD_DOWN_QUAT)
    _advance(tucked, steps=40)
    face_planted = _roulade_env(forward_rate=1.0, head_contact=True, head_quat=_LEVEL_QUAT)
    _advance(face_planted, steps=40)
    resting = _roulade_env(forward_rate=0.0, head_contact=True, head_quat=_HEAD_DOWN_QUAT)
    resting.scene["robot"].data.root_link_ang_vel_b.torch[:, 1] = 0.0
    mdp.roulade_roll_state(resting.as_env()).accumulated_angle[:] = math.radians(90.0)

    # 40 steps at 1 rad/s is 0.8 rad, inside the window; the rate factor is 1/2 of its norm
    torch.testing.assert_close(pivot(tucked), torch.tensor([0.5]))
    torch.testing.assert_close(pivot(face_planted), torch.tensor([0.5 * 0.3]))
    # mid-window, head down, head on the floor -- but not rotating, so nothing is earned
    torch.testing.assert_close(pivot(resting), torch.tensor([0.0]))


def test_roulade_stand_tax_charges_the_shortfall_only_once_the_gate_is_open():
    """Self-negating, so its positive weight prices the post-roll heap rather than paying for it."""
    gate_lo, gate_hi = math.radians(260.0), math.radians(330.0)
    env = _roulade_env(height=0.05)
    state = mdp.roulade_roll_state(env.as_env())

    during_the_roll = mdp.roulade_stand_tax(env.as_env(), target_height=0.115, gate_lo=gate_lo, gate_hi=gate_hi)
    state.frontier[:] = gate_hi
    state.head_latch[:] = True
    after_the_roll = mdp.roulade_stand_tax(env.as_env(), target_height=0.115, gate_lo=gate_lo, gate_hi=gate_hi)

    torch.testing.assert_close(during_the_roll, torch.tensor([0.0]))
    torch.testing.assert_close(after_the_roll, torch.tensor([-0.065]))


def test_roulade_rise_velocity_pays_only_below_its_ceiling():
    """Gated one quadrant earlier than the landing, and switched off once the robot is up."""
    gate_lo, gate_hi = math.radians(180.0), math.radians(260.0)
    rising = _roulade_env(height=0.08, vertical_speed=0.3)
    state = mdp.roulade_roll_state(rising.as_env())
    state.frontier[:] = gate_hi
    state.head_latch[:] = True
    hopping = _roulade_env(height=0.2, vertical_speed=0.3)
    hopping_state = mdp.roulade_roll_state(hopping.as_env())
    hopping_state.frontier[:] = gate_hi
    hopping_state.head_latch[:] = True

    torch.testing.assert_close(
        mdp.roulade_rise_velocity(rising.as_env(), max_height=0.125, gate_lo=gate_lo, gate_hi=gate_hi),
        torch.tensor([0.3]),
    )
    torch.testing.assert_close(
        mdp.roulade_rise_velocity(hopping.as_env(), max_height=0.125, gate_lo=gate_lo, gate_hi=gate_hi),
        torch.tensor([0.0]),
    )


def test_roulade_overspeed_penalty_is_quadratic_in_the_excess_only():
    """A controlled roll never touches it; a whip is charged the square of what it exceeds by."""
    controlled = _roulade_env(forward_rate=5.0)
    whipping = _roulade_env(forward_rate=9.0)
    backwards = _roulade_env(forward_rate=-9.0)

    torch.testing.assert_close(mdp.roulade_overspeed_penalty(controlled.as_env(), omega_max=7.0), torch.tensor([0.0]))
    torch.testing.assert_close(mdp.roulade_overspeed_penalty(whipping.as_env(), omega_max=7.0), torch.tensor([4.0]))
    # the magnitude is charged, so spinning backwards is taxed the same
    torch.testing.assert_close(mdp.roulade_overspeed_penalty(backwards.as_env(), omega_max=7.0), torch.tensor([4.0]))


def test_the_straightness_penalties_leave_a_clean_forward_roll_alone():
    """All three are zero through an arbitrarily deep pure-pitch roll and grow off the plane."""
    upright = _roulade_env(quat=_LEVEL_QUAT)
    deep_roll = _roulade_env(quat=_PITCHED_QUAT, forward_rate=4.0)
    shoulder = _roulade_env(quat=_ROLLED_QUAT, roll_rate=0.5, yaw_rate=0.5, lateral_speed=0.3)

    torch.testing.assert_close(mdp.roulade_flatness_penalty(upright.as_env()), torch.tensor([0.0]))
    # 90 degrees of pure pitch: the lateral axis is still horizontal, so nothing is charged
    torch.testing.assert_close(mdp.roulade_flatness_penalty(deep_roll.as_env()), torch.tensor([0.0]))
    torch.testing.assert_close(mdp.roulade_sagittal_penalty(deep_roll.as_env()), torch.tensor([0.0]))
    torch.testing.assert_close(mdp.roulade_lateral_velocity_penalty(deep_roll.as_env()), torch.tensor([0.0]))
    torch.testing.assert_close(mdp.roulade_flatness_penalty(shoulder.as_env()), torch.tensor([1.0]))
    torch.testing.assert_close(mdp.roulade_sagittal_penalty(shoulder.as_env()), torch.tensor([0.5]))
    torch.testing.assert_close(mdp.roulade_lateral_velocity_penalty(shoulder.as_env()), torch.tensor([0.09]))


def test_the_roll_terms_reject_an_asset_configuration_without_a_single_head_body():
    """The head-top test is the difference between a roulade and a face-plant, so it may not guess."""
    env = _roulade_env(head_contact=True)

    with pytest.raises(ValueError, match="single head body"):
        mdp.roulade_head_pivot(
            env.as_env(),
            sensor_cfg=_entity("head_ground_contact"),
            angle_lo=0.0,
            angle_hi=1.0,
            rate_norm=2.0,
            asset_cfg=_entity("robot"),
        )


##
# Roller skating (addendum section 5.3)
##

WHEEL_JOINT_IDS = [0, 1, 2, 3]
"""Joint slots the roller doubles below fill with the four passive wheels."""

TIRE_SENSOR_IDS = [0, 1, 2, 3]
"""Sensor slots the roller doubles fill with ``[left-front, left-rear, right-front, right-rear]``."""


class _DummyActionTerm:
    """Joint-position action term double: the driven joints and the target it produced."""

    def __init__(self, joint_ids: list[int], processed_actions: torch.Tensor) -> None:
        self._joint_ids = joint_ids
        self.processed_actions = processed_actions


class _DummyActionManager:
    """Action manager double carrying one term and the two action buffers the rate reads."""

    def __init__(self, term: _DummyActionTerm, action: torch.Tensor, prev_action: torch.Tensor) -> None:
        self._term = term
        self.action = action
        self.prev_action = prev_action
        self.active_terms = ["joint_pos"]
        self.action_term_dim = [action.shape[1]]

    def get_term(self, name: str) -> _DummyActionTerm:
        assert name == "joint_pos"
        return self._term


def _roller_sensor(air_time: list[float], contact_time: list[float]) -> _DummySensor:
    """A four-tire contact sensor double, one environment."""
    return _DummySensor(
        current_air_time=torch.tensor([air_time]),
        current_contact_time=torch.tensor([contact_time]),
    )


def _roller_env(
    *,
    command: list[float] | None = None,
    forward_vel: float = 0.0,
    air_time: list[float] | None = None,
    contact_time: list[float] | None = None,
    wheel_vel: list[float] | None = None,
    leg_vel: list[float] | None = None,
    height: float = 0.0,
    projected_gravity_b: list[float] | None = None,
    heading: float = 0.0,
    step_dt: float = 0.02,
) -> _DummyEnv:
    """One environment posed for the roller terms."""
    robot = _DummyAsset(
        root_link_lin_vel_b=torch.tensor([[forward_vel, 0.0, 0.0]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, height]]),
        projected_gravity_b=torch.tensor([projected_gravity_b or [0.0, 0.0, -1.0]]),
        joint_vel=torch.tensor([(wheel_vel or [0.0, 0.0, 0.0, 0.0]) + (leg_vel or [0.0, 0.0])]),
        joint_pos=torch.zeros(1, 6),
        default_joint_pos=torch.zeros(1, 6),
        heading_w=torch.tensor([heading]),
    )
    sensors = {}
    if air_time is not None or contact_time is not None:
        sensors["feet_ground_contact"] = _roller_sensor(air_time or [0.0] * 4, contact_time or [0.0] * 4)
    return _DummyEnv(
        num_envs=1,
        assets={"robot": robot},
        sensors=sensors,
        commands={"twist": torch.tensor([command or [0.0, 0.0, 0.0]])},
        step_dt=step_dt,
    )


def _tire_sensor_cfg() -> SceneEntityCfg:
    return _entity("feet_ground_contact", body_ids=TIRE_SENSOR_IDS, body_names=["tire", "tire_2", "tire_3", "tire_4"])


def test_fold_bodies_into_feet_groups_consecutive_bodies():
    """A foot is a run of consecutive sensor slots, so the left pair must stay ahead of the right."""
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    grouped = mdp.fold_bodies_into_feet(values, 2)

    assert grouped.shape == (1, 2, 2)
    torch.testing.assert_close(grouped, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    # a single-collider foot is the identity apart from the inserted axis
    torch.testing.assert_close(mdp.fold_bodies_into_feet(values, 1).squeeze(2), values)


def test_fold_bodies_into_feet_rejects_a_selection_that_is_not_whole_feet():
    """Three tires cannot be two feet, and silently dropping one would mis-pair left and right."""
    with pytest.raises(ValueError, match="whole number of feet"):
        mdp.fold_bodies_into_feet(torch.zeros(1, 3), 2)


def test_com_height_target_pays_a_flat_reward_inside_the_band_and_the_squared_miss_outside():
    """The band is a tolerance, not a target: anywhere inside it scores the same."""
    inside = _roller_env(height=0.11)
    low = _roller_env(height=0.0935 - 0.1)
    high = _roller_env(height=0.1235 + 0.2)

    band = {"target_height_min": 0.0935, "target_height_max": 0.1235}
    torch.testing.assert_close(mdp.com_height_target(inside.as_env(), **band), torch.tensor([1.0]))
    torch.testing.assert_close(mdp.com_height_target(low.as_env(), **band), torch.tensor([-0.01]))
    torch.testing.assert_close(mdp.com_height_target(high.as_env(), **band), torch.tensor([-0.04]))


def test_feet_flat_penalty_charges_only_the_loaded_blade():
    """The stance blade is asked to lie flat; the swing blade is free to tilt."""
    # the left foot's normal axis is 45 degrees from gravity, the right foot's is aligned with it
    tilted = [math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)]  # 90 deg about x
    level = [0.0, 0.0, 0.0, 1.0]
    robot = _DummyAsset(
        body_link_quat_w=torch.tensor([[tilted, level]]),
        GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -9.81]]),
    )
    asset_cfg = _entity("robot", body_ids=[0, 1], body_names=["ankle_l_v1", "ankle_r_v1"])

    def penalty(contact_time):
        env = _DummyEnv(
            num_envs=1,
            assets={"robot": robot},
            sensors={"feet_ground_contact": _roller_sensor([0.0] * 4, contact_time)},
        )
        return mdp.feet_flat_penalty(
            env.as_env(),
            asset_cfg=asset_cfg,
            sensor_cfg=_tire_sensor_cfg(),
            normal_axis=(0.0, 1.0, 0.0),
            bodies_per_foot=2,
        )

    # a 90 degree roll puts the left blade's +y axis on the world -z, so gravity is *on* its normal
    # and the tilt term is zero there; the level right blade has its normal horizontal, so it is 1
    torch.testing.assert_close(penalty([1.0, 1.0, 1.0, 1.0]), torch.tensor([1.0]))
    # only the left foot loaded: its own tilt of zero is charged and the right foot is ignored
    torch.testing.assert_close(penalty([1.0, 0.0, 0.0, 0.0]), torch.tensor([0.0]))
    # only the right foot loaded
    torch.testing.assert_close(penalty([0.0, 0.0, 0.0, 1.0]), torch.tensor([1.0]))
    # airborne: nothing is charged
    torch.testing.assert_close(penalty([0.0, 0.0, 0.0, 0.0]), torch.tensor([0.0]))


def test_joint_pose_l2_sums_the_squared_deviation_from_the_stand_pose():
    """Upstream sums rather than averages, unlike this package's L1 sibling."""
    robot = _DummyAsset(
        joint_pos=torch.tensor([[0.3, -0.4]]),
        default_joint_pos=torch.tensor([[0.1, -0.1]]),
    )
    env = _DummyEnv(num_envs=1, assets={"robot": robot})

    cost = mdp.joint_pose_l2(env.as_env(), asset_cfg=_entity("robot", joint_ids=[0, 1]))

    torch.testing.assert_close(cost, torch.tensor([0.2**2 + 0.3**2]))


def _action_rate_term(joint_ids, action, prev_action, selected, selected_names):
    """Build a resolved ``joint_action_rate_l2`` against an action-manager double."""
    robot = _DummyAsset(joint_pos=torch.zeros(1, 6), default_joint_pos=torch.zeros(1, 6))
    robot.joint_names = [f"joint_{index}" for index in range(6)]
    env = _DummyEnv(num_envs=1, assets={"robot": robot})
    env.action_manager = _DummyActionManager(_DummyActionTerm(joint_ids, torch.zeros_like(action)), action, prev_action)
    asset_cfg = _entity("robot", joint_ids=selected, joint_names=selected_names)
    cfg = RewardTermCfg(
        func=mdp.joint_action_rate_l2,
        weight=-0.5,
        params={"action_name": "joint_pos", "asset_cfg": asset_cfg},
    )
    return mdp.joint_action_rate_l2(cfg, env.as_env()), env, cfg


def test_joint_action_rate_l2_charges_only_the_columns_of_the_selected_joints():
    """Upstream hard-codes action columns 5..8; the port resolves them from the action term's joints."""
    # the action term drives joints 4, 2, 0 in that order, so joint 2 is column 1
    term, env, cfg = _action_rate_term(
        joint_ids=[4, 2, 0],
        action=torch.tensor([[1.0, 0.5, -2.0]]),
        prev_action=torch.tensor([[1.0, 0.2, 0.0]]),
        selected=[2],
        selected_names=["joint_2"],
    )

    cost = term(env.as_env(), **cfg.params)

    torch.testing.assert_close(cost, torch.tensor([0.3**2]))


def test_joint_action_rate_l2_rejects_a_joint_the_action_term_does_not_drive():
    """A passive wheel has no action column, so selecting one has to fail loudly."""
    with pytest.raises(ValueError, match="does not drive"):
        _action_rate_term(
            joint_ids=[0, 1],
            action=torch.zeros(1, 2),
            prev_action=torch.zeros(1, 2),
            selected=[5],
            selected_names=["joint_5"],
        )


def test_action_over_limit_penalty_charges_the_command_past_the_hard_stop_plus_the_tolerance():
    """It reads the target, not the achieved position, so it survives export as learned behaviour."""
    limits = torch.tensor([[[-0.4, 0.4], [-0.4, 0.4], [-0.4, 0.4]]])
    robot = _DummyAsset(joint_pos_limits=limits)
    env = _DummyEnv(num_envs=1, assets={"robot": robot})
    # inside the tolerance, 0.25 past the upper stop plus tolerance, and 0.1 past the lower one
    target = torch.tensor([[0.6, 0.95, -0.8]])
    env.action_manager = _DummyActionManager(_DummyActionTerm([0, 1, 2], target), torch.zeros(1, 3), torch.zeros(1, 3))

    cost = mdp.action_over_limit_penalty(env.as_env(), action_name="joint_pos", overshoot=0.3)

    torch.testing.assert_close(cost, torch.tensor([0.25 + 0.1]))


def test_wheel_speed_reward_saturates_in_the_mean_wheel_rate_and_scales_with_the_throttle():
    """The sole positive task reward: no wheel rotation, no reward, whatever else the policy does."""
    wheel_cfg = _entity("robot", joint_ids=WHEEL_JOINT_IDS, joint_names=["a", "b", "c", "d"])
    params = {"command_name": "twist", "asset_cfg": wheel_cfg, "vel_scale": 0.3, "wheel_radius": 0.0175}
    omega_scale = 0.3 / 0.0175

    still = _roller_env(command=[0.5, 0.0, 0.0], wheel_vel=[0.0] * 4)
    rolling = _roller_env(command=[0.5, 0.0, 0.0], wheel_vel=[omega_scale] * 4)
    coasting = _roller_env(command=[0.0, 0.0, 0.0], wheel_vel=[omega_scale] * 4)
    backwards = _roller_env(command=[0.5, 0.0, 0.0], wheel_vel=[-omega_scale] * 4)

    torch.testing.assert_close(mdp.wheel_speed_reward(still.as_env(), **params), torch.tensor([0.0]))
    torch.testing.assert_close(mdp.wheel_speed_reward(rolling.as_env(), **params), torch.tensor([0.5 * math.tanh(1.0)]))
    # a zero throttle pays nothing however fast the wheels turn
    torch.testing.assert_close(mdp.wheel_speed_reward(coasting.as_env(), **params), torch.tensor([0.0]))
    # and spinning backwards is not rewarded, because ``bidirectional`` is left off
    torch.testing.assert_close(mdp.wheel_speed_reward(backwards.as_env(), **params), torch.tensor([0.0]))


def test_braking_reward_is_silent_unless_the_throttle_is_negative():
    """It never competes with the wheel-speed reward, which only pays above a zero throttle."""
    params = {"command_name": "twist", "vel_std": 0.3}

    stopped = _roller_env(command=[-0.4, 0.0, 0.0], forward_vel=0.0)
    still_rolling = _roller_env(command=[-0.4, 0.0, 0.0], forward_vel=0.3)
    pushing = _roller_env(command=[0.4, 0.0, 0.0], forward_vel=0.0)

    torch.testing.assert_close(mdp.braking_reward(stopped.as_env(), **params), torch.tensor([0.4]))
    torch.testing.assert_close(
        mdp.braking_reward(still_rolling.as_env(), **params), torch.tensor([0.4 * math.exp(-1.0)])
    )
    torch.testing.assert_close(mdp.braking_reward(pushing.as_env(), **params), torch.tensor([0.0]))


def test_skating_air_time_reward_counts_feet_in_the_window_scaled_by_throttle_and_progress():
    """A fast flutter on the spot earns nothing: the forward-progress gate is what shuts it down."""
    params = {
        "sensor_cfg": _tire_sensor_cfg(),
        "command_name": "twist",
        "threshold_min": 0.15,
        "threshold_max": 0.45,
        "vel_gate_ref": 0.2,
        "bodies_per_foot": 2,
    }
    # the left foot is airborne inside the window; the right foot's rear tire is down, so it is not
    swinging = _roller_env(command=[0.5, 0.0, 0.0], forward_vel=0.2, air_time=[0.3, 0.3, 0.3, 0.0])
    on_the_spot = _roller_env(command=[0.5, 0.0, 0.0], forward_vel=0.0, air_time=[0.3, 0.3, 0.3, 0.0])
    half_speed = _roller_env(command=[0.5, 0.0, 0.0], forward_vel=0.1, air_time=[0.3, 0.3, 0.3, 0.0])
    too_short = _roller_env(command=[0.5, 0.0, 0.0], forward_vel=0.2, air_time=[0.1, 0.1, 0.1, 0.0])

    torch.testing.assert_close(mdp.skating_air_time_reward(swinging.as_env(), **params), torch.tensor([0.5]))
    torch.testing.assert_close(mdp.skating_air_time_reward(on_the_spot.as_env(), **params), torch.tensor([0.0]))
    torch.testing.assert_close(mdp.skating_air_time_reward(half_speed.as_env(), **params), torch.tensor([0.25]))
    torch.testing.assert_close(mdp.skating_air_time_reward(too_short.as_env(), **params), torch.tensor([0.0]))


def test_single_support_reward_pays_one_blade_down_and_charges_two():
    """The core anti-swizzle signal, and its double-support charge is deliberately unspeed-gated."""
    params = {
        "sensor_cfg": _tire_sensor_cfg(),
        "command_name": "twist",
        "vel_gate_ref": 0.2,
        "bodies_per_foot": 2,
    }
    single = _roller_env(command=[0.4, 0.0, 0.0], forward_vel=0.2, contact_time=[0.5, 0.0, 0.0, 0.0])
    double = _roller_env(command=[0.4, 0.0, 0.0], forward_vel=0.2, contact_time=[0.5, 0.5, 0.5, 0.5])
    double_still = _roller_env(command=[0.4, 0.0, 0.0], forward_vel=0.0, contact_time=[0.0, 0.5, 0.5, 0.0])

    torch.testing.assert_close(mdp.single_support_reward(single.as_env(), **params), torch.tensor([0.4]))
    torch.testing.assert_close(mdp.single_support_reward(double.as_env(), **params), torch.tensor([-0.1]))
    # standing still on both blades under a forward command is charged in full
    torch.testing.assert_close(mdp.single_support_reward(double_still.as_env(), **params), torch.tensor([-0.1]))


def test_glide_reward_pays_a_quiet_single_support_coast():
    """Cadence and commitment pull against each other; this is the term that pays for committing."""
    leg_cfg = _entity("robot", joint_ids=[4, 5], joint_names=["hip", "knee"])
    params = {
        "sensor_cfg": _tire_sensor_cfg(),
        "command_name": "twist",
        "asset_cfg": leg_cfg,
        "vel_ref": 0.2,
        "bodies_per_foot": 2,
    }
    gliding = _roller_env(command=[0.4, 0.0, 0.0], forward_vel=0.2, contact_time=[0.5, 0.0, 0.0, 0.0])
    thrashing = _roller_env(
        command=[0.4, 0.0, 0.0], forward_vel=0.2, contact_time=[0.5, 0.0, 0.0, 0.0], leg_vel=[5.0, 0.0]
    )
    double = _roller_env(command=[0.4, 0.0, 0.0], forward_vel=0.2, contact_time=[0.5, 0.0, 0.5, 0.0])
    braking = _roller_env(command=[-0.4, 0.0, 0.0], forward_vel=0.2, contact_time=[0.5, 0.0, 0.0, 0.0])

    torch.testing.assert_close(mdp.glide_reward(gliding.as_env(), **params), torch.tensor([1.0]))
    torch.testing.assert_close(mdp.glide_reward(thrashing.as_env(), **params), torch.tensor([math.exp(-1.0)]))
    torch.testing.assert_close(mdp.glide_reward(double.as_env(), **params), torch.tensor([0.0]))
    # silent while braking, so it never fights the braking reward
    torch.testing.assert_close(mdp.glide_reward(braking.as_env(), **params), torch.tensor([0.0]))


def _gait_symmetry_term(env: _DummyEnv):
    cfg = RewardTermCfg(
        func=mdp.gait_symmetry_penalty,
        weight=-1.0,
        params={"sensor_cfg": _tire_sensor_cfg(), "bodies_per_foot": 2},
    )
    return mdp.gait_symmetry_penalty(cfg, env.as_env()), cfg


def test_gait_symmetry_penalty_accumulates_the_swing_imbalance_over_the_episode():
    """The instantaneous asymmetry of a real stride is free; a one-legged push is not."""
    env = _roller_env(air_time=[0.3, 0.3, 0.0, 0.0], step_dt=0.02)
    term, cfg = _gait_symmetry_term(env)

    # only the left foot ever swings, so the imbalance saturates at 1
    for _ in range(5):
        cost = term(env.as_env(), **cfg.params)
    torch.testing.assert_close(cost, torch.tensor([0.1 / (0.1 + 1e-3)]))

    # the accumulator is cleared for the environments the reward manager resets
    term.reset()
    torch.testing.assert_close(term(env.as_env(), **cfg.params), torch.tensor([0.02 / (0.02 + 1e-3)]))


def test_gait_symmetry_penalty_is_zero_for_a_balanced_stride():
    """Equal swing time on both blades costs nothing however it is distributed in time."""
    left = _roller_env(air_time=[0.3, 0.3, 0.0, 0.0])
    right = _roller_env(air_time=[0.0, 0.0, 0.3, 0.3])
    term, cfg = _gait_symmetry_term(left)

    for env in (left, right, left, right):
        cost = term(env.as_env(), **cfg.params)

    torch.testing.assert_close(cost, torch.tensor([0.0]))


def test_gait_symmetry_penalty_rejects_a_sensor_that_is_not_a_pair_of_feet():
    """The imbalance is defined between exactly two feet, so three has to fail loudly."""
    env = _roller_env(air_time=[0.0] * 4)
    cfg = RewardTermCfg(
        func=mdp.gait_symmetry_penalty,
        weight=-1.0,
        params={"sensor_cfg": _tire_sensor_cfg(), "bodies_per_foot": 1},
    )
    with pytest.raises(ValueError, match="left and a right"):
        mdp.gait_symmetry_penalty(cfg, env.as_env())


def test_forward_lean_reward_peaks_at_a_nose_down_trunk_and_only_while_pushing():
    """Upstream's docstring negates the quantity and its code does not; the code is what is ported."""
    params = {"command_name": "twist", "target_pitch": 0.262, "std": 0.1}
    # a forward pitch of theta puts sin(theta) on the forward component of projected gravity
    leaning = _roller_env(command=[0.5, 0.0, 0.0], projected_gravity_b=[0.262, 0.0, -0.965])
    upright = _roller_env(command=[0.5, 0.0, 0.0], projected_gravity_b=[0.0, 0.0, -1.0])
    leaning_back = _roller_env(command=[0.5, 0.0, 0.0], projected_gravity_b=[-0.262, 0.0, -0.965])
    coasting = _roller_env(command=[0.0, 0.0, 0.0], projected_gravity_b=[0.262, 0.0, -0.965])

    torch.testing.assert_close(mdp.forward_lean_reward(leaning.as_env(), **params), torch.tensor([0.5]))
    assert float(mdp.forward_lean_reward(upright.as_env(), **params)) < 0.5 * math.exp(-6.0)
    # the reward is not symmetric about zero: leaning back scores far below leaning forward
    assert float(mdp.forward_lean_reward(leaning_back.as_env(), **params)) < float(
        mdp.forward_lean_reward(upright.as_env(), **params)
    )
    torch.testing.assert_close(mdp.forward_lean_reward(coasting.as_env(), **params), torch.tensor([0.0]))


def _heading_hold_term(env: _DummyEnv):
    cfg = RewardTermCfg(func=mdp.heading_hold_reward, weight=1.0, params={"std": 0.4})
    return mdp.heading_hold_reward(cfg, env.as_env()), cfg


def test_heading_hold_reward_anchors_on_the_first_step_of_an_episode():
    """The reference is the heading the episode started with, whatever that heading was."""
    spawned = _roller_env(heading=2.0)
    term, cfg = _heading_hold_term(spawned)

    torch.testing.assert_close(term(spawned.as_env(), **cfg.params), torch.tensor([1.0]))
    # drifting 0.4 rad off the spawn heading costs exactly one Gaussian width
    drifted = _roller_env(heading=2.4)
    torch.testing.assert_close(term(drifted.as_env(), **cfg.params), torch.tensor([math.exp(-1.0)]))
    # and the error wraps, so a heading of -pi is next to one of +pi rather than 2 pi away
    term.reset()
    at_pi = _roller_env(heading=math.pi)
    torch.testing.assert_close(term(at_pi.as_env(), **cfg.params), torch.tensor([1.0]))
    just_past = _roller_env(heading=-math.pi + 0.4)
    torch.testing.assert_close(term(just_past.as_env(), **cfg.params), torch.tensor([math.exp(-1.0)]))
