# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe and acceptance tests for the contributed MicroDuck lift environment.

Like its pick-and-place sibling this task has no upstream counterpart, so acceptance is internal and
the design is the parity table. What it inherits from that task is not values but *discipline*: the
reward budget is asserted in episode-return units rather than argued in prose, because on the
pick-and-place task prose is exactly what let three reward defects through.
"""

import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.lift.agents.rsl_rl_ppo_cfg import MicroDuckLiftPPORunnerCfg
from isaaclab_tasks.contrib.microduck.lift.lift_env_cfg import (
    MICRODUCK_LIFT_REACH_RANGE,
    MICRODUCK_LIFT_TARGET_HEIGHT,
    MicroDuckLiftFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.pickplace.pickplace_env_cfg import MicroDuckPickPlaceFlatEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_BEAK_USD_PATH, MICRODUCK_MARBLE_RADIUS

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_beak_usd = pytest.mark.skipif(
    not os.path.isfile(MICRODUCK_BEAK_USD_PATH),
    reason=(
        f"MicroDuck beak USD asset is missing: {MICRODUCK_BEAK_USD_PATH}. Generate it with"
        " 'uv run --extra importers python scripts/tools/convert_microduck.py --model beak'."
    ),
)

TASK_NAME = "IsaacContrib-Lift-Flat-MicroDuck"

CONTROL_DT = 0.02
EPISODE_STEPS = 250
"""A 5 s episode at 50 Hz. Deliberately not the family's 20 s; see the environment configuration."""

ACTOR_OBSERVATION_TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("object_position", 3),
    ("latched", 1),
]
ACTOR_OBSERVATION_DIM = 52

EXPECTED_REWARD_WEIGHTS = {
    "mouth_to_object": 1.0,
    "mouth_down": 0.5,
    "latch_bonus": 1000.0,
    "lift": 30.0,
    "carry_upright": 2.0,
    "upright": 0.2,
    "feet_grounded": 1.0,
    "head_impact_penalty": -2.0,
    "self_collisions": -1.0,
    "dof_pos_limits": -1.0,
    "fell_penalty": -5000.0,
    "body_ang_vel": -0.05,
    "angular_momentum": -0.02,
    "action_rate_l2": -0.8,
    "joint_torques_l2": -5e-3,
}
"""The whole stack, transcribed rather than imported so a weight change has to be made twice."""


def _term_names(group) -> list[str]:
    return [name for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)]


def _returns(rewards) -> dict[str, float]:
    """Episode-return contribution of each strategy, in the units the learner optimizes."""

    def r(name: str, occurrences: float) -> float:
        return getattr(rewards, name).weight * CONTROL_DT * occurrences

    return {
        "lift_and_hold": (
            r("latch_bonus", 1)
            + r("lift", 180)
            + r("carry_upright", 180)
            + r("upright", EPISODE_STEPS)
            + r("feet_grounded", EPISODE_STEPS)
        ),
        "grab_but_never_lift": r("latch_bonus", 1) + r("upright", EPISODE_STEPS) + r("feet_grounded", EPISODE_STEPS),
        "hover_never_grab": (
            r("mouth_to_object", EPISODE_STEPS)
            + r("mouth_down", EPISODE_STEPS)
            + r("upright", EPISODE_STEPS)
            + r("feet_grounded", EPISODE_STEPS)
        ),
        "stand_and_ignore": r("upright", EPISODE_STEPS) + r("feet_grounded", EPISODE_STEPS),
        "fold_onto_it_and_topple": (
            r("mouth_to_object", 60) + r("latch_bonus", 1) + r("lift", 20) + r("fell_penalty", 1)
        ),
    }


##
# Recipe
##


@pytest.mark.unit
def test_the_reward_stack_is_the_designs_table():
    """Every term, every weight, nothing extra."""
    rewards = MicroDuckLiftFlatEnvCfg().rewards
    declared = {name: term.weight for name, term in vars(rewards).items() if term is not None}

    assert declared == pytest.approx(EXPECTED_REWARD_WEIGHTS)


@pytest.mark.unit
def test_lifting_pays_more_than_any_shortcut():
    """The audit in the units the learner sees, which is the whole of what this stack is checked on."""
    returns = _returns(MicroDuckLiftFlatEnvCfg().rewards)

    assert returns["lift_and_hold"] > 4.0 * returns["grab_but_never_lift"]
    assert returns["lift_and_hold"] > 8.0 * returns["hover_never_grab"]
    # grabbing still beats hovering, so a policy that has not learned to lift is pulled toward the
    # grab rather than toward standing inert
    assert returns["grab_but_never_lift"] > returns["hover_never_grab"] > returns["stand_and_ignore"]
    # and toppling onto it is worse than doing nothing at all
    assert returns["fold_onto_it_and_topple"] < 0.0


@pytest.mark.unit
def test_the_dominant_term_is_the_objective_and_is_unfarmable():
    """``lift`` is 78 % of the positive budget, and that is fine *because of what it is*.

    The pick-and-place task's first run failed with one term at 88 % of the return, so the rule
    written there was that no term may exceed half the budget. That rule is wrong as stated: what
    made that case a defect was not the share but that the term was a **re-triggerable event with no
    per-episode cap**. A dominant *objective* is what a well-shaped task looks like.

    So what is asserted here is the property that actually matters -- the dominant term is bounded
    per step, saturating, and can only be earned by holding the object off the floor.
    """
    rewards = MicroDuckLiftFlatEnvCfg().rewards
    ceilings = {
        "mouth_to_object": EPISODE_STEPS,
        "mouth_down": EPISODE_STEPS,
        "latch_bonus": 1,
        "lift": EPISODE_STEPS,
        "carry_upright": EPISODE_STEPS,
        "upright": EPISODE_STEPS,
        "feet_grounded": EPISODE_STEPS,
    }
    contributions = {n: getattr(rewards, n).weight * CONTROL_DT * c for n, c in ceilings.items()}
    dominant = max(contributions, key=contributions.get)

    assert dominant == "lift"
    assert rewards.lift.func is mdp.lift_height
    # bounded and saturating: the ramp tops out, so there is no reward for throwing the object
    assert rewards.lift.params["lift_height"] > rewards.lift.params["rest_height"]
    # and the only re-triggerable event in the stack is capped at one payment per episode
    assert rewards.latch_bonus.func is mdp.pickplace_latch_bonus


@pytest.mark.unit
def test_the_objective_is_dense_rather_than_a_one_shot_bonus():
    """No spike. The pick-and-place task paid 100 return in a single control step for its success,
    which is ~400x its typical per-step reward and the leading suspect for the advantage variance
    that stopped its policy annealing. Here the largest single-step payment is the latch bonus."""
    rewards = MicroDuckLiftFlatEnvCfg().rewards
    per_step_ceiling = (
        max(
            rewards.lift.weight,
            rewards.mouth_to_object.weight,
            rewards.carry_upright.weight,
            rewards.feet_grounded.weight,
            rewards.upright.weight,
        )
        * CONTROL_DT
    )
    spike = rewards.latch_bonus.weight * CONTROL_DT

    assert spike / per_step_ceiling < 40.0
    # and the pick-and-place task's spike was far larger, which is the comparison being made
    assert spike < MicroDuckPickPlaceFlatEnvCfg().rewards.place_success.weight * CONTROL_DT


@pytest.mark.unit
def test_the_task_needs_no_walking_and_therefore_no_curriculum():
    """Taking the locomotion out is the point; a reach the robot must walk to would put it back."""
    cfg = MicroDuckLiftFlatEnvCfg()

    assert MICRODUCK_LIFT_REACH_RANGE[1] <= 0.15
    assert cfg.events.reset_object.params["distance_range"] == MICRODUCK_LIFT_REACH_RANGE
    # no curriculum and no command manager at all -- both exist on the sibling only to grow a
    # distance the robot has to walk and to name a drop point, and this task has neither
    assert cfg.curriculum is None
    assert cfg.commands is None
    assert MicroDuckPickPlaceFlatEnvCfg().commands is not None


@pytest.mark.unit
def test_the_beak_never_releases_because_there_is_nowhere_to_put_it():
    """A lift has no release edge. The state machine's is switched off by leaving the drop-point
    command unset, rather than by a second code path that could drift from the first."""
    params = MicroDuckLiftFlatEnvCfg().events.update_latch.params

    assert "command_name" not in params or params["command_name"] is None
    assert MicroDuckLiftFlatEnvCfg().events.drive_beak.func is mdp.drive_beak_from_latch


@pytest.mark.unit
def test_the_episode_is_five_seconds_not_the_familys_twenty():
    """The gesture takes about three. The pick-and-place task's 20 s left a fifteen-second dead zone
    after its objective was met, which the trained policy filled by wandering about two metres."""
    cfg = MicroDuckLiftFlatEnvCfg()

    assert cfg.episode_length_s == pytest.approx(5.0)
    assert int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) == EPISODE_STEPS
    assert cfg.episode_length_s < MicroDuckPickPlaceFlatEnvCfg().episode_length_s


@pytest.mark.unit
def test_the_lift_target_is_above_the_marbles_resting_height():
    """A ramp that saturated at rest height would pay full marks for not lifting at all."""
    params = MicroDuckLiftFlatEnvCfg().rewards.lift.params

    assert params["rest_height"] == pytest.approx(MICRODUCK_MARBLE_RADIUS)
    assert params["lift_height"] == pytest.approx(MICRODUCK_LIFT_TARGET_HEIGHT)
    assert params["lift_height"] > 5.0 * params["rest_height"]


@pytest.mark.unit
def test_the_actor_layout_is_the_camera_contract_minus_the_drop_point():
    """One camera-replaceable row, as on the sibling: the object's position in the base frame."""
    policy = MicroDuckLiftFlatEnvCfg().observations.policy
    names = _term_names(policy)

    assert names == [n for n, _ in ACTOR_OBSERVATION_TERMS]
    assert sum(w for _, w in ACTOR_OBSERVATION_TERMS) == ACTOR_OBSERVATION_DIM
    assert policy.object_position.func is mdp.object_pos_in_base
    for name in names:
        assert "sensor_cfg" not in getattr(policy, name).params


@pytest.mark.unit
def test_the_iteration_budget_is_sized_from_evidence():
    """The pick-and-place logs settle it: that task had solved its objective by iteration 2000 and
    the remaining 18000 bought almost nothing. This task is strictly simpler."""
    runner = MicroDuckLiftPPORunnerCfg()

    assert runner.experiment_name == "microduck_lift"
    assert runner.max_iterations == 4000
    assert runner.max_iterations < 6000


@pytest.mark.unit
def test_the_task_runs_the_beak_robot_and_the_marble():
    """Neither is optional: the gesture is a beak closing on something it can hold."""
    cfg = MicroDuckLiftFlatEnvCfg()

    assert "beak" in cfg.scene.robot.spawn.usd_path
    assert set(cfg.scene.robot.actuators) == {"servos", "beak"}
    assert cfg.scene.object.spawn.radius == pytest.approx(MICRODUCK_MARBLE_RADIUS)
    # the beak is not a policy output, on this task as on the hardware
    assert len(cfg.actions.joint_pos.joint_names) == 14


##
# Simulator-backed acceptance
##


@pytest.mark.integration
@requires_microduck_beak_usd
def test_the_observation_groups_are_the_widths_the_tables_name():
    """The actor width, measured."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        observations, _ = env.reset()

        assert observations["policy"].shape[1] == ACTOR_OBSERVATION_DIM
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_beak_usd
def test_a_scripted_lift_scores_the_stack_end_to_end():
    """The acceptance: grab the marble under physics and check the objective actually pays.

    An acceptance that never evaluates a reward is not acceptance -- and on this family a scripted
    rollout has twice caught a defect the recipe tests could not see.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env_cfg.terminations.fell_over = None
        env_cfg.terminations.fell_low = None
        env_cfg.rewards.fell_penalty = None
        for name in ("foot_friction", "mass_inertia", "randomize_com", "randomize_head_com", "push_robot"):
            setattr(env_cfg.events, name, None)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        state = mdp.pickplace_latch_state(unwrapped)
        obj = unwrapped.scene["object"]
        hook = unwrapped.event_manager.get_term_cfg("update_latch")

        def score() -> dict[str, torch.Tensor]:
            manager = unwrapped.reward_manager
            manager.reset()
            manager.compute(dt=unwrapped.step_dt)
            out = {n: manager._episode_sums[n].clone() / unwrapped.step_dt for n in manager.active_terms}
            manager.reset()
            return out

        assert not bool(state.latched.any())
        assert float(score()["lift"].abs().max()) == pytest.approx(0.0)

        # put the marble on the latch anchor and run the state machine, as the interval event does
        robot = unwrapped.scene["robot"]
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import math as math_utils

        mouth = SceneEntityCfg("robot", body_names=["jaw_soft"])
        mouth.resolve(unwrapped.scene)
        bp = robot.data.body_link_pos_w.torch[:, mouth.body_ids].squeeze(1)
        bq = robot.data.body_link_quat_w.torch[:, mouth.body_ids].squeeze(1)
        off = torch.tensor(hook.params["mouth_offset_b"], device=unwrapped.device).expand_as(bp)
        axis = torch.tensor(hook.params["mouth_axis_b"], device=unwrapped.device).expand_as(bp)
        anchor = bp + math_utils.quat_apply(bq, off) + hook.params["hold_distance"] * math_utils.quat_apply(bq, axis)
        pose = torch.zeros(unwrapped.num_envs, 7, device=unwrapped.device)
        pose[:, :3] = anchor
        pose[:, 6] = 1.0  # (x, y, z, w) identity
        obj.write_root_link_pose_to_sim(pose)
        obj.write_root_com_velocity_to_sim(torch.zeros(unwrapped.num_envs, 6, device=unwrapped.device))
        unwrapped.scene.write_data_to_sim()
        unwrapped.scene.update(dt=unwrapped.physics_dt)
        hook.func(unwrapped, None, **hook.params)

        assert bool(state.latched.all())
        grabbed = score()
        assert float(grabbed["latch_bonus"].min()) > 0.0
        # the marble is up at mouth height, so the lift ramp is paying
        assert float(grabbed["lift"].min()) > 0.0
        # and there is no release on this task, however long it is held
        for _ in range(20):
            hook.func(unwrapped, None, **hook.params)
        assert bool(state.latched.all())
        assert not bool(state.succeeded.any())
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_beak_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
