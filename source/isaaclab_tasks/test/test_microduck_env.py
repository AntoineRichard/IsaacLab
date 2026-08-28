# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke and recipe-parity tests for the contributed MicroDuck velocity-tracking environments.

The smoke tests spawn :data:`~isaaclab_assets.MICRODUCK_CFG`, whose USD is generated rather than
committed, so they skip when that asset is absent -- the same condition the asset fidelity tests
skip on. Generate it with ``uv run --extra importers python scripts/tools/convert_microduck.py``.

The parity test needs neither the asset nor the simulator: it reads the assembled configuration and
compares it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference.md``. The expected values are spelled out rather than
imported from the configuration under test, so that a drifting value fails rather than agrees with
itself.
"""

import math
import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym
import pytest

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MicroDuckVelocityRoughEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_REGENERATE_COMMAND, MICRODUCK_USD_PATH

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_usd = pytest.mark.skipif(
    not os.path.isfile(MICRODUCK_USD_PATH),
    reason=f"MicroDuck USD asset is missing: {MICRODUCK_USD_PATH}. Generate it with '{MICRODUCK_REGENERATE_COMMAND}'.",
)
"""Skips the tests that spawn the robot. The parity test does not need the asset."""

TASK_NAMES = ["IsaacContrib-Velocity-Flat-MicroDuck", "IsaacContrib-Velocity-Rough-MicroDuck"]
"""The registered MicroDuck velocity tasks."""

ACTOR_OBSERVATION_DIM = 61
"""Actor observation width the deployed MicroDuck policy expects.

48 proprioception values plus the 13-wide command block ``[twist(3), head_pose(4), body_pose(6)]``;
see ``artifacts/microduck/upstream_reference.md`` section 7. The head and body pose commands are
not part of the task skeleton yet, so this is the contract the port is being built towards.
"""


@pytest.mark.integration
@requires_microduck_usd
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_environment_steps_with_random_actions(task_name):
    """Each registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(task_name, device="cuda", num_envs=2, num_steps=10)


@pytest.mark.integration
@requires_microduck_usd
@pytest.mark.xfail(reason="obs contract lands in Task 9", strict=False)
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_actor_observation_width_matches_the_deploy_contract(task_name):
    """The actor group is as wide as the policy deployed on the robot expects."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(task_name, device="cuda", num_envs=2)
        env = gym.make(task_name, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        obs, _ = env.reset()

        assert obs["policy"].shape[-1] == ACTOR_OBSERVATION_DIM
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


##
# Recipe parity against upstream
##

_SQRT_0_1 = math.sqrt(0.1)
_SQRT_0_5 = math.sqrt(0.5)
_SQRT_0_05 = math.sqrt(0.05)

_SOLE_TARGET_HEIGHT = 0.02
"""Upstream's foot-height target [m], measured at the sole site."""

_MEASURED_SOLE_TO_ANKLE_OFFSET = 0.022496
"""Height [m] of the ankle body frame above the foot site at the STAND2 home pose.

Measured independently of the port, with MuJoCo forward kinematics on the pinned
``robot_walk.xml``: ``data.xpos[ankle_left][2] - data.site_xpos[left_foot][2]``.
"""

_SOLE_TO_ANKLE_OFFSET = 0.0225
"""The same offset [m], rounded to a tenth of a millimetre, which is the form the recipe carries."""

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "track_lin_vel": (2.0, {"command_name": "base_velocity", "std": _SQRT_0_1}),
    "track_ang_vel": (2.0, {"command_name": "base_velocity", "std": _SQRT_0_5}),
    "upright": (2.0, {"std": _SQRT_0_05}),
    "pose": (1.0, {"command_name": "base_velocity", "walking_threshold": 0.01, "running_threshold": 1.5}),
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    "dof_pos_limits": (-1.0, {}),
    "action_rate_l2": (-0.1, {}),
    "air_time": (
        3.0,
        {
            "command_name": "base_velocity",
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_threshold": 0.01,
        },
    ),
    "foot_clearance": (
        -2.0,
        {
            "command_name": "base_velocity",
            "command_threshold": 0.01,
            "target_height": _SOLE_TARGET_HEIGHT + _SOLE_TO_ANKLE_OFFSET,
        },
    ),
    "foot_swing_height": (
        # the relative-error form is not invariant under moving the measurement frame, so the
        # weight carries the compensating factor -- see MICRODUCK_FOOT_TARGET_HEIGHT
        -0.25 * ((_SOLE_TARGET_HEIGHT + _SOLE_TO_ANKLE_OFFSET) / _SOLE_TARGET_HEIGHT) ** 2,
        {
            "command_name": "base_velocity",
            "command_threshold": 0.01,
            "target_height": _SOLE_TARGET_HEIGHT + _SOLE_TO_ANKLE_OFFSET,
        },
    ),
    "foot_slip": (-0.1, {"command_name": "base_velocity", "command_threshold": 0.01}),
    "self_collisions": (-1.0, {}),
    "head_pose_tracking": (2.0, {"command_name": "head_pose", "std": 0.5}),
    "head_pose_bias": (0.0, {"command_name": "head_pose", "tau_s": 1.0}),
    "body_pose_tracking": (
        0.0,
        {
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15.0),
        },
    ),
}
"""Upstream's reward recipe (reference sections 2.4 and 6), keyed by term name.

``soft_landing`` is absent because upstream removes it from its own recipe, and there is no energy
or torque penalty because the mjlab base template has none.
"""

EXPECTED_EVENTS = {
    # name: (mode, scalar params)
    "foot_friction": ("startup", {"static_friction_range": (0.7, 1.3), "restitution_range": (0.0, 0.0)}),
    "encoder_bias": ("startup", {"bias_range": (-0.015, 0.015)}),
    "mass_inertia": ("startup", {"mass_distribution_params": (0.95, 1.05), "operation": "scale"}),
    "reset_base": (
        "reset",
        {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    ),
    "reset_robot_joints": ("reset", {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}),
    "randomize_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_head_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_armature": ("reset", {"armature_distribution_params": (0.9, 1.1), "operation": "scale"}),
    "push_robot": ("interval", {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}),
}
"""Upstream's domain-randomization suite (reference section 2.6), keyed by term name.

Upstream's ``base_com`` is absent because it selects zero bodies upstream; its BAM friction terms
are absent because the BAM actuator is not ported yet; and the three terms it ships disabled are
absent because a term that is never enabled is not part of the recipe.
"""

EXPECTED_STAGE_TABLES = {
    "action_rate_weight": (
        "weight_stages",
        "weight",
        [-0.1, -0.2, -0.4, -0.6, -0.8, -1.0],
        [0, 500, 750, 1000, 1250, 1500],
    ),
    "head_pose_bias_weight": ("weight_stages", "weight", [0.0, 1.0, 2.0, 3.0], [0, 600, 1000, 1500]),
    "standing_envs": (
        "standing_stages",
        "rel_standing_envs",
        [0.02, 0.05, 0.1, 0.15, 0.2, 0.25],
        [0, 500, 750, 1000, 1500, 2000],
    ),
    "com_range": ("range_stages", "range", [0.003, 0.005, 0.01, 0.015], [0, 500, 1000, 1500]),
    "head_com_range": ("range_stages", "range", [0.003, 0.005, 0.01], [0, 500, 1000]),
}
"""Upstream's scalar curriculum tables (reference section 6): payloads and iteration boundaries."""

EXPECTED_HEAD_POSE_STAGES = [
    (0, ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (500, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
    (1000, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
    (1500, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))),
    (2000, ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))),
]
"""Upstream's head-pose range ramp, per ``(neck_pitch, head_pitch, head_yaw, head_roll)``."""

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, upstream's ``num_steps_per_env``."""


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which are compared separately and are not upstream's values."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


@pytest.mark.unit
def test_the_reward_recipe_matches_upstream_term_for_term():
    """Every reward slot upstream trains with is present, with its weight and its parameters."""
    rewards = MicroDuckVelocityRoughEnvCfg().rewards

    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_foot_height_targets_are_offset_to_the_ankle_frame_they_are_measured_in():
    """The sole-height target is re-based, not copied, and the swing weight compensates for it."""
    rewards = MicroDuckVelocityRoughEnvCfg().rewards

    # the rounded offset the recipe uses is the offset MuJoCo reports, to a tenth of a millimetre
    assert pytest.approx(_MEASURED_SOLE_TO_ANKLE_OFFSET, abs=5e-5) == _SOLE_TO_ANKLE_OFFSET
    # a target copied verbatim would ask for a foot 2 cm below the ground it stands on
    assert rewards.foot_clearance.params["target_height"] > _MEASURED_SOLE_TO_ANKLE_OFFSET
    # the swing-height term charges a relative error, so re-basing it also rescales its gradient;
    # weight * (peak / target - 1)^2 must equal upstream's -0.25 * (peak_sole / 0.02 - 1)^2
    target = rewards.foot_swing_height.params["target_height"]
    peak_sole = 0.031
    ported = rewards.foot_swing_height.weight * ((peak_sole + _SOLE_TO_ANKLE_OFFSET) / target - 1.0) ** 2
    upstream = -0.25 * (peak_sole / _SOLE_TARGET_HEIGHT - 1.0) ** 2
    assert ported == pytest.approx(upstream)


@pytest.mark.unit
def test_the_head_pose_terms_pin_the_joint_order_the_command_columns_assume():
    """The head rewards index command columns positionally, so their joint order is load-bearing."""
    rewards = MicroDuckVelocityRoughEnvCfg().rewards

    for name in ("head_pose_tracking", "head_pose_bias"):
        asset_cfg = getattr(rewards, name).params["asset_cfg"]
        assert asset_cfg.joint_names == ["neck_pitch", "head_pitch", "head_yaw", "head_roll"], name
        assert asset_cfg.preserve_order, name


@pytest.mark.unit
def test_the_randomization_suite_matches_upstream_term_for_term():
    """Every domain randomization upstream enables is present, in its mode, with its ranges."""
    events = MicroDuckVelocityRoughEnvCfg().events

    assert set(vars(events)) == set(EXPECTED_EVENTS)
    for name, (mode, params) in EXPECTED_EVENTS.items():
        term = getattr(events, name)
        assert term.mode == mode, name
        actual = _scalar_params(term)
        for key, value in params.items():
            assert actual[key] == value, f"{name}.{key}"
    assert events.push_robot.interval_range_s == (3.0, 6.0)


@pytest.mark.unit
def test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link():
    """``bearing_roll`` is the right hip-yaw link; upstream lists it among the head bodies in error."""
    body_names = MicroDuckVelocityRoughEnvCfg().events.randomize_head_com.params["asset_cfg"].body_names

    assert "bearing_roll" not in body_names
    assert len(body_names) == 4


@pytest.mark.unit
def test_the_scalar_curricula_reproduce_upstream_stage_tables():
    """The staged schedules carry upstream's payloads at upstream's iteration boundaries."""
    curriculum = MicroDuckVelocityRoughEnvCfg().curriculum

    for name, (params_key, payload_key, payloads, iterations) in EXPECTED_STAGE_TABLES.items():
        stages = getattr(curriculum, name).params[params_key]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage[payload_key] for stage in stages] == pytest.approx(payloads), name


@pytest.mark.unit
def test_the_head_pose_range_curriculum_reproduces_upstream_stage_table():
    """The head command opens through upstream's five stages, keeping its final caps verbatim."""
    stages = MicroDuckVelocityRoughEnvCfg().curriculum.head_pose_range.params["range_stages"]

    assert len(stages) == len(EXPECTED_HEAD_POSE_STAGES)
    for stage, (iteration, ranges) in zip(stages, EXPECTED_HEAD_POSE_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        assert stage["ranges"] == ranges


@pytest.mark.unit
def test_the_pose_commands_are_registered_at_their_initial_ranges():
    """The head and body pose commands exist, are as wide as their reward terms index, and resample."""
    commands = MicroDuckVelocityRoughEnvCfg().commands

    assert commands.head_pose.ranges == EXPECTED_HEAD_POSE_STAGES[0][1]
    assert commands.head_pose.resampling_time_range == (2.0, 5.0)
    assert commands.body_pose.ranges == ((-0.005, 0.005),) * 3 + ((-0.05, 0.05),) * 3
    assert commands.body_pose.resampling_time_range == (2.0, 5.0)
    # the velocity command's own ranges are fixed; only the standing fraction is on a curriculum
    assert commands.base_velocity.ranges.lin_vel_x == (-0.4, 0.4)
    assert commands.base_velocity.ranges.lin_vel_y == (-0.3, 0.3)
    assert commands.base_velocity.ranges.ang_vel_z == (-1.0, 1.0)


@pytest.mark.unit
def test_the_terrain_generator_carries_upstream_sub_terrain_mix():
    """The rough terrain is upstream's gentle mix, and the flat task keeps a plain plane."""
    rough = MicroDuckVelocityRoughEnvCfg()
    generator = rough.scene.terrain.terrain_generator

    assert rough.scene.terrain.terrain_type == "generator"
    assert {name: cfg.proportion for name, cfg in generator.sub_terrains.items()} == {
        "flat": 0.25,
        "pyramid_stairs": 0.25,
        "random_grid": 0.30,
        "pyramid_slope": 0.20,
    }
    # the robot lifts its feet 1-2 cm, so nothing it walks over may be taller than that
    assert generator.sub_terrains["pyramid_stairs"].step_height_range == (0.0, 0.015)
    assert generator.sub_terrains["random_grid"].grid_height_range == (0.0, 0.010)
    assert generator.sub_terrains["pyramid_slope"].slope_range == (0.03, 0.10)

    flat = MicroDuckVelocityFlatEnvCfg()
    assert flat.scene.terrain.terrain_type == "plane"
    assert flat.scene.terrain.terrain_generator is None
    # the terrain-level curriculum reads the generator's configuration, so it cannot stay behind
    assert flat.curriculum.terrain_levels is None
    assert rough.curriculum.terrain_levels is not None


@pytest.mark.unit
def test_the_self_collision_sensor_reports_a_force_matrix():
    """``self_collision_cost`` reads a filtered force matrix, which needs a dedicated sensor."""
    scene = MicroDuckVelocityRoughEnvCfg().scene

    assert scene.self_collision.filter_prim_paths_expr
    # one sensing body against one filter body: the two soles are the only collidable geometries on
    # the robot, so this is the whole self-collision signal, counted once rather than from both ends
    assert "ankle_left" in scene.self_collision.prim_path
    assert "ankle_right" in scene.self_collision.filter_prim_paths_expr[0]
    assert not scene.contact_forces.filter_prim_paths_expr
