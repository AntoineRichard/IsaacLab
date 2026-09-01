# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the MicroDuck pick-and-place MDP terms.

This task has **no upstream counterpart**, so there is no reference implementation to transcribe an
expected value from. Every expectation below is therefore derived independently, in the test, from
the design document (``artifacts/microduck/pickplace/DESIGN.md``) -- the latch state machine's
transition table, the spring's stability budget, and the potential-based shape of the two distance
rewards. That document is this task's parity table.

The terms read a handful of articulation, rigid-object and command tensors and return a number per
environment, so the suite runs against doubles rather than a simulated scene. The end-to-end
behaviour under physics is asserted in ``test_microduck_pickplace_env.py``, which is where an
acceptance that actually scores rewards belongs.
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


MOUTH_BODY_IDS = [0]
"""Body slot the doubles fill with ``jaw_soft``, the body the mouth tip hangs off."""

HOLD_DISTANCE = 0.040
"""Mouth-tip-to-object-centre distance [m] the latch spring holds at, in the tests below."""

STIFFNESS = 40.0
DAMPING = 0.5
BREAK_FORCE = 2.0
LATCH_RADIUS = 0.055
MAX_REL_SPEED = 0.30
PLACE_TOLERANCE = 0.05
PLACE_MAX_HEIGHT = 0.06
"""The shipped latch constants, transcribed from the design document rather than imported.

A table that read the configuration it checks would agree with itself. The environment test asserts
that these are the values the environment actually ships.
"""


##
# Doubles
##


class _DummyTensorView:
    """Stands in for a ``ProxyArray``, which exposes its contents under ``.torch``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.torch = tensor


class _DummyData:
    def __init__(self, **fields: torch.Tensor | None) -> None:
        for name, value in fields.items():
            setattr(self, name, None if value is None else _DummyTensorView(value))


class _DummyComposer:
    """Wrench composer double that records the last write instead of launching a kernel."""

    def __init__(self) -> None:
        self.forces: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.body_ids = None
        self.env_ids = None
        self.is_global: bool | None = None
        self.calls = 0

    def set_forces_and_torques_index(
        self, forces=None, torques=None, positions=None, body_ids=None, env_ids=None, is_global=False
    ) -> None:
        self.forces = forces
        self.positions = positions
        self.body_ids = body_ids
        self.env_ids = env_ids
        self.is_global = is_global
        self.calls += 1


class _DummyAsset:
    """Articulation or rigid-object double, with a recording wrench composer."""

    def __init__(self, **fields: torch.Tensor | None) -> None:
        self.data = _DummyData(**fields)
        self.permanent_wrench_composer = _DummyComposer()
        self.written_pose: torch.Tensor | None = None
        self.written_velocity: torch.Tensor | None = None

    def write_root_link_pose_to_sim_index(self, root_pose: torch.Tensor, env_ids) -> None:
        self.written_pose = root_pose

    def write_root_com_velocity_to_sim_index(self, root_velocity: torch.Tensor, env_ids) -> None:
        self.written_velocity = root_velocity


class _DummyCommandTerm:
    def __init__(self, target_pos_w: torch.Tensor) -> None:
        self.target_pos_w = target_pos_w


class _DummyCommandManager:
    def __init__(self, commands: dict[str, torch.Tensor], terms: dict | None = None) -> None:
        self._commands = commands
        self._terms = terms or {}

    def get_command(self, name: str) -> torch.Tensor:
        return self._commands[name]

    def get_term(self, name: str):
        return self._terms[name]


class _DummyScene:
    def __init__(self, assets: dict, env_origins: torch.Tensor) -> None:
        self._assets = assets
        self.sensors: dict = {}
        self.env_origins = env_origins

    def __getitem__(self, name: str):
        return self._assets[name]


class _DummyEnv:
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        assets: dict | None = None,
        command_terms: dict | None = None,
        env_origins: torch.Tensor | None = None,
        step_dt: float = 0.02,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self.scene = _DummyScene(
            assets or {},
            torch.zeros(num_envs, 3, device=device) if env_origins is None else env_origins,
        )
        self.command_manager = _DummyCommandManager({}, command_terms)

    def as_env(self) -> ManagerBasedRLEnv:
        return cast("ManagerBasedRLEnv", self)


def _entity(name: str, body_ids=None, body_names=None, joint_ids=None, joint_names=None) -> SceneEntityCfg:
    """Build an already-resolved scene entity configuration, as the managers hand terms."""
    cfg = SceneEntityCfg(name)
    if body_ids is not None:
        cfg.body_ids = body_ids
        cfg.body_names = body_names
    if joint_ids is not None:
        cfg.joint_ids = joint_ids
        cfg.joint_names = joint_names
    return cfg


def _identity_quat(num_envs: int) -> torch.Tensor:
    """The identity orientation in Isaac Lab's ``(x, y, z, w)`` layout -- scalar **last**.

    Erratum E-1 of the design document: the pick-and-place hand-off states the opposite convention,
    and writing the scalar into slot 3 stands the robot on its head. Every quaternion in this file
    goes through here so that the layout is stated once.
    """
    quat = torch.zeros(num_envs, 4)
    quat[:, 3] = 1.0
    return quat


def _yaw_quat(yaw: float, num_envs: int = 1) -> torch.Tensor:
    """A pure-yaw rotation in the ``(x, y, z, w)`` layout."""
    quat = torch.zeros(num_envs, 4)
    quat[:, 2] = math.sin(0.5 * yaw)
    quat[:, 3] = math.cos(0.5 * yaw)
    return quat


def _latch_env(
    *,
    num_envs: int = 1,
    mouth_pos: list[list[float]],
    object_pos: list[list[float]],
    target_pos: list[list[float]],
    mouth_quat: torch.Tensor | None = None,
    mouth_lin_vel: list[list[float]] | None = None,
    mouth_ang_vel: list[list[float]] | None = None,
    object_lin_vel: list[list[float]] | None = None,
) -> _DummyEnv:
    """Build an environment whose mouth tip sits at ``mouth_pos``.

    The mouth-tip offset is passed as zero by every caller below, so the ``jaw_soft`` body frame and
    the mouth tip coincide and the geometry under test is the latch's, not the offset's. The offset
    itself is exercised by ``_mouth_tip_pose_w``'s own tests on the ground-pick task.
    """
    zeros = [[0.0, 0.0, 0.0] for _ in range(num_envs)]
    robot = _DummyAsset(
        body_link_pos_w=torch.tensor(mouth_pos).unsqueeze(1),
        body_link_quat_w=(_identity_quat(num_envs) if mouth_quat is None else mouth_quat).unsqueeze(1),
        body_link_lin_vel_w=torch.tensor(mouth_lin_vel or zeros).unsqueeze(1),
        body_link_ang_vel_w=torch.tensor(mouth_ang_vel or zeros).unsqueeze(1),
        root_link_pos_w=torch.zeros(num_envs, 3),
        root_link_quat_w=_identity_quat(num_envs),
    )
    obj = _DummyAsset(
        root_link_pos_w=torch.tensor(object_pos),
        root_link_lin_vel_w=torch.tensor(object_lin_vel or zeros),
    )
    return _DummyEnv(
        num_envs=num_envs,
        assets={"robot": robot, "object": obj},
        command_terms={"place_target": _DummyCommandTerm(torch.tensor(target_pos))},
    )


def _update_latch(env: _DummyEnv, **overrides) -> None:
    """Run the latch state machine once with the shipped constants."""
    params = dict(
        asset_cfg=_entity("robot", body_ids=MOUTH_BODY_IDS, body_names=["jaw_soft"]),
        object_cfg=_entity("object"),
        command_name="place_target",
        mouth_offset_b=(0.0, 0.0, 0.0),
        mouth_axis_b=(0.0, 0.0, -1.0),
        hold_distance=HOLD_DISTANCE,
        latch_radius=LATCH_RADIUS,
        max_rel_speed=MAX_REL_SPEED,
        stiffness=STIFFNESS,
        damping=DAMPING,
        break_force=BREAK_FORCE,
        place_tolerance=PLACE_TOLERANCE,
        place_max_height=PLACE_MAX_HEIGHT,
    )
    params.update(overrides)
    mdp.update_pickplace_latch(env.as_env(), None, **params)


##
# The latch state machine
##


def test_the_latch_state_is_allocated_once_and_shared():
    """The event that drives the latch and the terms that read it must address one buffer.

    Same lazy-allocation contract as ``mouth_payload`` and ``ball_kick_direction``: the first caller
    creates it and every later caller gets that same tensor, not a copy.
    """
    env = _DummyEnv(num_envs=3).as_env()

    first = mdp.pickplace_latch_state(env)
    second = mdp.pickplace_latch_state(env)

    assert first is second
    assert first.latched.shape == (3,)
    assert first.latched.dtype is torch.bool
    assert not first.latched.any()
    assert not first.succeeded.any()


def test_the_mouth_latches_onto_an_object_within_reach_and_at_rest():
    """The latch edge fires once, on the step the two conditions first hold together."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[1.0, 0.0, 0.035]])

    _update_latch(env)
    state = mdp.pickplace_latch_state(env.as_env())

    assert state.latched.tolist() == [True]
    assert state.latch_edge.tolist() == [True]

    # a second step with the object still held is not a second edge
    _update_latch(env)
    assert state.latched.tolist() == [True]
    assert state.latch_edge.tolist() == [False]


def test_an_object_out_of_reach_does_not_latch():
    """The gate is the mouth-tip-to-centre distance against ``latch_radius``, nothing else."""
    just_outside = LATCH_RADIUS + 1e-3
    env = _latch_env(mouth_pos=[[0.0, 0.0, just_outside]], object_pos=[[0.0, 0.0, 0.0]], target_pos=[[1.0, 0.0, 0.035]])

    _update_latch(env)

    assert mdp.pickplace_latch_state(env.as_env()).latched.tolist() == [False]


def test_an_object_being_swiped_at_does_not_latch():
    """You cannot pick up what you are moving past: the relative-speed gate denies it.

    Without it the policy could farm the latch bonus by running the mouth through the object.
    """
    env = _latch_env(
        mouth_pos=[[0.0, 0.0, 0.03]],
        object_pos=[[0.0, 0.0, 0.0]],
        target_pos=[[1.0, 0.0, 0.035]],
        object_lin_vel=[[MAX_REL_SPEED + 0.01, 0.0, 0.0]],
    )

    _update_latch(env)

    assert mdp.pickplace_latch_state(env.as_env()).latched.tolist() == [False]


def test_the_held_object_is_pulled_toward_the_anchor_with_the_reaction_on_the_jaw():
    """The spring is a *virtual weld*: equal and opposite, so momentum is conserved.

    A kinematic slave would move the object without the robot feeling it, which is the physical
    content this design exists to keep (design document, ruling R-PP1).
    """
    # mouth at 0.10 m pointing down, object 0.05 m below it -- inside the 0.055 m latch radius
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.10]], object_pos=[[0.0, 0.0, 0.05]], target_pos=[[1.0, 0.0, 0.035]])
    _update_latch(env)  # latch
    _update_latch(env)  # hold

    robot_force = env.scene["robot"].permanent_wrench_composer.forces
    object_force = env.scene["object"].permanent_wrench_composer.forces

    # anchor is 0.040 m below the mouth tip, at z = 0.060; the object is at z = 0.050, so the spring
    # is stretched by 0.010 m and pulls the object *up* by 40 * 0.010 = 0.4 N
    torch.testing.assert_close(object_force.squeeze(1), torch.tensor([[0.0, 0.0, 0.4]]))
    torch.testing.assert_close(robot_force.squeeze(1), -object_force.squeeze(1))


def test_the_reaction_acts_at_the_anchor_in_world_coordinates():
    """The robot-side wrench is placed at the point the spring pulls from, not at the body origin."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.10]], object_pos=[[0.0, 0.0, 0.05]], target_pos=[[1.0, 0.0, 0.035]])
    _update_latch(env)
    _update_latch(env)

    composer = env.scene["robot"].permanent_wrench_composer

    assert composer.is_global is True
    torch.testing.assert_close(composer.positions.squeeze(1), torch.tensor([[0.0, 0.0, 0.10 - HOLD_DISTANCE]]))


def test_a_latch_can_never_break_on_the_step_it_forms():
    """Derived, not assumed: the spring cannot reach the break force inside the latch radius.

    Worst case at the moment of latching is the largest of the two ends of the admissible distance
    band, plus the damping term at the fastest admissible relative speed.
    """
    worst_spring = STIFFNESS * max(HOLD_DISTANCE, LATCH_RADIUS - HOLD_DISTANCE)
    worst_damping = DAMPING * MAX_REL_SPEED

    assert worst_spring + worst_damping < BREAK_FORCE

    # and the state machine agrees: latching just inside the band holds. The gate is a strict
    # inequality, so the far edge itself is outside it.
    env = _latch_env(
        mouth_pos=[[0.0, 0.0, LATCH_RADIUS - 1e-4]], object_pos=[[0.0, 0.0, 0.0]], target_pos=[[1.0, 0.0, 0.035]]
    )
    _update_latch(env)
    assert mdp.pickplace_latch_state(env.as_env()).latched.tolist() == [True]

    _update_latch(env)
    assert mdp.pickplace_latch_state(env.as_env()).latched.tolist() == [True]


def test_the_latch_breaks_rather_than_saturating_when_it_is_overloaded():
    """A clamped constraint would let the policy use the spring as a winch (ruling R-PP2)."""
    # latch first, then teleport the mouth far away so the spring is overstretched
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[1.0, 0.0, 0.035]])
    _update_latch(env)
    assert mdp.pickplace_latch_state(env.as_env()).latched.tolist() == [True]

    # required force is 40 * (0.5 - 0.040) = 18.4 N, far past the 2 N grip
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, 2] = 0.52
    _update_latch(env)

    state = mdp.pickplace_latch_state(env.as_env())
    assert state.latched.tolist() == [False]
    assert state.succeeded.tolist() == [False]
    torch.testing.assert_close(env.scene["object"].permanent_wrench_composer.forces, torch.zeros(1, 1, 3))


def test_a_broken_latch_cannot_re_form_on_the_same_step():
    """Otherwise a break is a no-op and the grip limit means nothing.

    The state machine evaluates break, then release, then latch, and the latch gate reads the state
    the break wrote.
    """
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[1.0, 0.0, 0.035]])
    _update_latch(env)

    # overload the spring while leaving the object well inside the latch radius, so the only thing
    # that can keep it unlatched is the ordering
    _update_latch(env, break_force=1e-6)

    state = mdp.pickplace_latch_state(env.as_env())
    assert state.latched.tolist() == [False]
    assert state.latch_edge.tolist() == [False]


def test_the_object_is_released_at_the_target_and_the_success_is_sticky():
    """The release edge is the success edge, and it fires once per episode (ruling R-PP8)."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[0.4, 0.0, 0.035]])
    _update_latch(env)
    assert mdp.pickplace_latch_state(env.as_env()).latched.tolist() == [True]

    # carry the mouth and the object to the target
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, 0] = 0.4
    env.scene["object"].data.root_link_pos_w.torch[:, 0] = 0.4
    _update_latch(env)

    state = mdp.pickplace_latch_state(env.as_env())
    assert state.latched.tolist() == [False]
    assert state.release_edge.tolist() == [True]
    assert state.succeeded.tolist() == [True]

    # and it cannot be farmed: the object is back under the mouth, but nothing re-latches
    _update_latch(env)
    assert state.latched.tolist() == [False]
    assert state.latch_edge.tolist() == [False]
    assert state.release_edge.tolist() == [False]
    assert state.succeeded.tolist() == [True]


def test_an_object_held_high_over_the_target_is_not_released():
    """Placing is setting it down, not dropping it from a height."""
    env = _latch_env(
        mouth_pos=[[0.4, 0.0, PLACE_MAX_HEIGHT + 0.06]],
        object_pos=[[0.4, 0.0, PLACE_MAX_HEIGHT + 0.02]],
        target_pos=[[0.4, 0.0, 0.035]],
    )
    _update_latch(env)
    assert mdp.pickplace_latch_state(env.as_env()).latched.tolist() == [True]

    _update_latch(env)

    state = mdp.pickplace_latch_state(env.as_env())
    assert state.latched.tolist() == [True]
    assert state.release_edge.tolist() == [False]


def test_the_release_gate_measures_the_planar_distance_from_the_environment_origin():
    """The height half of the gate is measured above the environment's own ground, not world zero."""
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.5]])
    env = _latch_env(
        num_envs=2,
        mouth_pos=[[0.0, 0.0, 0.05], [10.0, 0.0, 0.55]],
        object_pos=[[0.0, 0.0, 0.02], [10.0, 0.0, 0.52]],
        target_pos=[[0.0, 0.0, 0.035], [10.0, 0.0, 0.535]],
    )
    env.scene.env_origins = origins

    _update_latch(env)
    _update_latch(env)

    state = mdp.pickplace_latch_state(env.as_env())
    # both environments are at their own target and just above their own floor
    assert state.release_edge.tolist() == [True, True]


def test_resetting_clears_the_latch_for_the_selected_environments_only():
    """A stale latch must not survive a reset that moved the object."""
    env = _latch_env(
        num_envs=2,
        mouth_pos=[[0.0, 0.0, 0.05], [0.0, 0.0, 0.05]],
        object_pos=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]],
        target_pos=[[1.0, 0.0, 0.035], [1.0, 0.0, 0.035]],
    )
    _update_latch(env)
    state = mdp.pickplace_latch_state(env.as_env())
    state.succeeded[:] = True

    mdp.reset_pickplace_latch(env.as_env(), torch.tensor([1]))

    assert state.latched.tolist() == [True, False]
    assert state.succeeded.tolist() == [True, False]


##
# Object placement at reset
##


def test_the_object_is_placed_in_the_robots_own_yaw_frame():
    """A world-frame offset would put the object behind a robot that spawned facing the other way.

    The reset draws a uniformly random yaw, so this is half the episodes, not an edge case.
    """
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.1]], object_pos=[[0.0, 0.0, 0.0]], target_pos=[[1.0, 0.0, 0.035]])
    robot = env.scene["robot"]
    robot.data.root_link_pos_w.torch[:] = torch.tensor([[1.0, 2.0, 0.125]])
    robot.data.root_link_quat_w.torch[:] = _yaw_quat(math.pi / 2.0)

    mdp.reset_object_in_reach(
        env.as_env(),
        None,
        distance_range=(0.2, 0.2),
        bearing_range=(0.0, 0.0),
        object_radius=0.035,
        asset_cfg=_entity("object"),
    )

    pose = env.scene["object"].written_pose
    # a zero bearing at a 90-degree yaw is +y in the world
    torch.testing.assert_close(pose[:, :3], torch.tensor([[1.0, 2.2, 0.035]]))


def test_the_placed_object_is_at_rest_and_carries_the_identity_orientation():
    """Erratum E-1: the scalar component lives in slot 6, not slot 3."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.1]], object_pos=[[0.0, 0.0, 0.0]], target_pos=[[1.0, 0.0, 0.035]])

    mdp.reset_object_in_reach(
        env.as_env(),
        None,
        distance_range=(0.1, 0.3),
        bearing_range=(-1.0, 1.0),
        object_radius=0.035,
        asset_cfg=_entity("object"),
    )

    pose = env.scene["object"].written_pose
    torch.testing.assert_close(pose[:, 3:], torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(env.scene["object"].written_velocity, torch.zeros(1, 6))


def test_the_object_rests_exactly_on_the_ground_of_its_own_environment():
    """Its centre is one radius above the environment origin, so it neither floats nor penetrates."""
    env = _latch_env(
        num_envs=2,
        mouth_pos=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]],
        object_pos=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        target_pos=[[1.0, 0.0, 0.035], [1.0, 0.0, 0.035]],
    )
    env.scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.25]])

    mdp.reset_object_in_reach(
        env.as_env(),
        None,
        distance_range=(0.2, 0.2),
        bearing_range=(0.0, 0.0),
        object_radius=0.035,
        asset_cfg=_entity("object"),
    )

    torch.testing.assert_close(env.scene["object"].written_pose[:, 2], torch.tensor([0.035, 0.285]))


def test_the_drawn_distance_and_bearing_stay_inside_the_configured_bands():
    """The curriculum widens these bands, so a draw that escaped them would silently skip a stage."""
    num_envs = 512
    env = _latch_env(
        num_envs=num_envs,
        mouth_pos=[[0.0, 0.0, 0.1]] * num_envs,
        object_pos=[[0.0, 0.0, 0.0]] * num_envs,
        target_pos=[[1.0, 0.0, 0.035]] * num_envs,
    )

    mdp.reset_object_in_reach(
        env.as_env(),
        None,
        distance_range=(0.15, 0.32),
        bearing_range=(-1.2, 1.2),
        object_radius=0.035,
        asset_cfg=_entity("object"),
    )

    placed = env.scene["object"].written_pose[:, :2]
    distance = torch.linalg.norm(placed, dim=1)
    bearing = torch.atan2(placed[:, 1], placed[:, 0])

    assert float(distance.min()) >= 0.15 - 1e-6
    assert float(distance.max()) <= 0.32 + 1e-6
    assert float(bearing.abs().max()) <= 1.2 + 1e-6
    # and the band is actually sampled rather than collapsed onto one value
    assert float(distance.max() - distance.min()) > 0.1


##
# Observations
##


def test_the_object_position_is_reported_in_the_robot_base_frame():
    """This is the one row a camera has to supply in v2, so it is the robot's own frame."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.1]], object_pos=[[0.3, 0.0, 0.035]], target_pos=[[1.0, 0.0, 0.035]])
    robot = env.scene["robot"]
    robot.data.root_link_pos_w.torch[:] = torch.tensor([[0.1, 0.0, 0.125]])
    robot.data.root_link_quat_w.torch[:] = _yaw_quat(math.pi / 2.0)

    obs = mdp.object_pos_in_base(env.as_env(), asset_cfg=_entity("object"))

    # world offset is +0.2 x; the robot faces +y, so that is 0.2 to its right, i.e. -y in base
    torch.testing.assert_close(obs, torch.tensor([[0.0, -0.2, -0.09]]))


def test_the_object_observations_are_guarded_against_a_solver_that_ejected_it():
    """Nothing NaN-checks a free rigid body on this stack, so the terms guard themselves."""
    env = _latch_env(
        mouth_pos=[[0.0, 0.0, 0.1]],
        object_pos=[[float("nan"), 0.0, 0.035]],
        target_pos=[[1.0, 0.0, 0.035]],
        object_lin_vel=[[float("inf"), 0.0, 0.0]],
    )

    assert torch.isfinite(mdp.object_pos_in_base(env.as_env(), asset_cfg=_entity("object"))).all()
    assert torch.isfinite(mdp.object_vel_in_base(env.as_env(), asset_cfg=_entity("object"))).all()


def test_the_latched_flag_is_a_single_float_column():
    """It is an observation, so it is float and two-dimensional, not a boolean vector."""
    env = _DummyEnv(num_envs=2)
    mdp.pickplace_latch_state(env.as_env()).latched[1] = True

    flag = mdp.pickplace_latched_flag(env.as_env())

    assert flag.shape == (2, 1)
    torch.testing.assert_close(flag, torch.tensor([[0.0], [1.0]]))


##
# Rewards
##


def _reward_term(cls, env: _DummyEnv):
    """Instantiate a stateful reward term against a double, as the reward manager would."""
    return cls(RewardTermCfg(func=cls, params={}), env.as_env())


def test_the_approach_progress_pays_the_distance_closed_and_charges_the_distance_opened():
    """Potential-based, so a closed path sums to exactly zero (ruling R-PP7).

    A Gaussian on the distance would pay for standing at the object forever; this pays nothing for
    standing anywhere.
    """
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.1]], object_pos=[[0.4, 0.0, 0.035]], target_pos=[[1.0, 0.0, 0.035]])
    term = _reward_term(mdp.pickplace_approach_progress, env)
    root = env.scene["robot"].data.root_link_pos_w.torch

    # the first step after a reset re-baselines rather than paying for the phantom approach
    torch.testing.assert_close(term(env.as_env()), torch.tensor([0.0]))

    root[:, 0] = 0.1  # closed 0.1 m
    first = term(env.as_env())
    root[:, 0] = 0.0  # opened it again
    second = term(env.as_env())

    torch.testing.assert_close(first, torch.tensor([0.1]))
    torch.testing.assert_close(second, torch.tensor([-0.1]))
    torch.testing.assert_close(first + second, torch.tensor([0.0]))


def test_the_approach_progress_is_silent_once_the_object_is_held():
    """Its job is over at the latch; the carry terms take it from there."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[1.0, 0.0, 0.035]])
    term = _reward_term(mdp.pickplace_approach_progress, env)
    term(env.as_env())

    _update_latch(env)
    env.scene["robot"].data.root_link_pos_w.torch[:, 0] = 0.5

    torch.testing.assert_close(term(env.as_env()), torch.tensor([0.0]))


def test_the_carry_progress_pays_only_while_the_object_is_held():
    """An object rolling to the target on its own is not a placement."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.5]], object_pos=[[0.0, 0.0, 0.035]], target_pos=[[0.5, 0.0, 0.035]])
    term = _reward_term(mdp.pickplace_carry_progress, env)
    obj = env.scene["object"].data.root_link_pos_w.torch

    term(env.as_env())
    obj[:, 0] = 0.2
    torch.testing.assert_close(term(env.as_env()), torch.tensor([0.0]))

    # now hold it and move it the same distance again
    mdp.pickplace_latch_state(env.as_env()).latched[:] = True
    term(env.as_env())
    obj[:, 0] = 0.4
    torch.testing.assert_close(term(env.as_env()), torch.tensor([0.2]))


def test_the_carry_hold_outbids_hovering_at_the_object():
    """The reward-hacking audit in one assertion (ruling R-PP6).

    ``mouth_to_object`` pays up to 1.0 and is gated off once latched, so a carry bonus below it would
    make *refusing to pick the object up* strictly dominant. This asserts the kernel is a plain
    indicator; the environment test asserts the weights that make the inequality hold.
    """
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[1.0, 0.0, 0.035]])

    torch.testing.assert_close(mdp.pickplace_carry_hold(env.as_env()), torch.tensor([0.0]))

    _update_latch(env)

    torch.testing.assert_close(mdp.pickplace_carry_hold(env.as_env()), torch.tensor([1.0]))


def test_the_carry_hold_stops_paying_after_a_successful_placement():
    """Otherwise standing next to a placed object earns the carry bonus for the rest of the episode."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[0.0, 0.0, 0.035]])
    _update_latch(env)
    _update_latch(env)  # releases at the target
    state = mdp.pickplace_latch_state(env.as_env())
    assert state.succeeded.tolist() == [True]

    torch.testing.assert_close(mdp.pickplace_carry_hold(env.as_env()), torch.tensor([0.0]))


def test_the_mouth_to_object_reward_peaks_on_contact_and_is_gated_off_once_held():
    """The fine bend-onto-it term. It is a hover basin by construction; the audit prices it."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.04]], object_pos=[[0.0, 0.0, 0.04]], target_pos=[[1.0, 0.0, 0.035]])
    params = dict(
        asset_cfg=_entity("robot", body_ids=MOUTH_BODY_IDS, body_names=["jaw_soft"]),
        object_cfg=_entity("object"),
        mouth_offset_b=(0.0, 0.0, 0.0),
        std=0.05,
    )

    torch.testing.assert_close(mdp.pickplace_mouth_to_object(env.as_env(), **params), torch.tensor([1.0]))

    env.scene["object"].data.root_link_pos_w.torch[:, 2] = 0.04 - 0.05
    torch.testing.assert_close(mdp.pickplace_mouth_to_object(env.as_env(), **params), torch.tensor([math.exp(-1.0)]))

    mdp.pickplace_latch_state(env.as_env()).latched[:] = True
    torch.testing.assert_close(mdp.pickplace_mouth_to_object(env.as_env(), **params), torch.tensor([0.0]))


def test_the_success_bonus_fires_exactly_once_on_the_release_edge():
    """150 of one-shot mass, so a second firing would be worth more than the whole episode."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[0.0, 0.0, 0.035]])
    _update_latch(env)
    torch.testing.assert_close(mdp.pickplace_place_success(env.as_env()), torch.tensor([0.0]))

    _update_latch(env)
    torch.testing.assert_close(mdp.pickplace_place_success(env.as_env()), torch.tensor([1.0]))

    _update_latch(env)
    torch.testing.assert_close(mdp.pickplace_place_success(env.as_env()), torch.tensor([0.0]))


def test_the_placement_precision_scores_the_error_at_the_moment_of_release():
    """Zero on every other step, so it cannot be integrated by loitering on the target.

    The edge buffers are rewritten by the state machine, not by the reward, so "another step" means
    another :func:`update_pickplace_latch`. In a rollout that is exactly one per reward evaluation:
    the interval events fire once at the tail of every ``step``.
    """
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[0.03, 0.0, 0.035]])
    _update_latch(env)
    _update_latch(env)  # object is 0.03 m from the target, inside the 0.05 m tolerance

    reward = mdp.pickplace_place_precision(env.as_env(), command_name="place_target", std=PLACE_TOLERANCE)

    torch.testing.assert_close(reward, torch.tensor([math.exp(-((0.03 / PLACE_TOLERANCE) ** 2))]))

    _update_latch(env)
    torch.testing.assert_close(
        mdp.pickplace_place_precision(env.as_env(), command_name="place_target", std=PLACE_TOLERANCE),
        torch.tensor([0.0]),
    )


def test_the_latch_bonus_fires_on_the_latch_edge_only():
    """One-shot, so holding the object is not itself worth 30 a step."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.05]], object_pos=[[0.0, 0.0, 0.02]], target_pos=[[1.0, 0.0, 0.035]])

    _update_latch(env)
    torch.testing.assert_close(mdp.pickplace_latch_bonus(env.as_env()), torch.tensor([1.0]))

    _update_latch(env)
    torch.testing.assert_close(mdp.pickplace_latch_bonus(env.as_env()), torch.tensor([0.0]))


def test_the_object_clearance_prefers_a_lift_to_a_drag_and_only_while_held():
    """Weak on purpose: dragging the object to the target is still a solved task."""
    env = _latch_env(mouth_pos=[[0.0, 0.0, 0.1]], object_pos=[[0.0, 0.0, 0.045]], target_pos=[[1.0, 0.0, 0.035]])
    params = dict(asset_cfg=_entity("object"), target_height=0.045, std=0.03)

    torch.testing.assert_close(mdp.pickplace_object_clearance(env.as_env(), **params), torch.tensor([0.0]))

    mdp.pickplace_latch_state(env.as_env()).latched[:] = True
    torch.testing.assert_close(mdp.pickplace_object_clearance(env.as_env(), **params), torch.tensor([1.0]))

    env.scene["object"].data.root_link_pos_w.torch[:, 2] = 0.045 + 0.03
    torch.testing.assert_close(mdp.pickplace_object_clearance(env.as_env(), **params), torch.tensor([math.exp(-1.0)]))
