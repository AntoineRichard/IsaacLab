# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke and recipe-parity tests for the contributed MicroDuck velocity-plus-recovery environment.

The smoke tests spawn :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG`, whose USD is generated
rather than committed, so they skip when that asset is absent. Generate it with
``uv run --extra importers python scripts/tools/convert_microduck.py --model allcollisions``.

The parity tests need neither the asset nor the simulator: they read the assembled configuration and
compare it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference_tasks3.md`` section 3.

This task is the one in the family that **derives from another**, so the suite has two halves and
they are checked differently on purpose:

* the **recovery layer** is spelled out against the extraction, as every sibling suite spells its
  own recipe out, so that a drifting value fails rather than agrees with itself;
* the **walking layer** is compared against the velocity task's own assembled configuration, because
  "the proven velocity recipe, verbatim" is the contract, and restating its values here would let
  the two drift apart while both suites passed. Those values are pinned against upstream by
  ``test_microduck_env.py``.
"""

import math
import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR
from isaaclab_tasks.contrib.microduck.velstand.agents.rsl_rl_ppo_cfg import MicroDuckVelStandPPORunnerCfg
from isaaclab_tasks.contrib.microduck.velstand.velstand_env_cfg import (
    MICRODUCK_CROUCH_JOINT_POS,
    MICRODUCK_FALLEN_TILT_DEG,
    MICRODUCK_RECOVERED_HEIGHT,
    MICRODUCK_RECOVERED_TILT_DEG,
    MICRODUCK_STAND_HEIGHT,
    MICRODUCK_TERMINATION_Z,
    MicroDuckVelStandFlatEnvCfg,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_USD_PATH

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_allcollisions_usd = pytest.mark.skipif(
    not os.path.isfile(MICRODUCK_ALLCOLLISIONS_USD_PATH),
    reason=(
        f"MicroDuck all-collisions USD asset is missing: {MICRODUCK_ALLCOLLISIONS_USD_PATH}. Generate it with"
        " 'uv run --extra importers python scripts/tools/convert_microduck.py --model allcollisions'."
    ),
)
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-VelStand-Flat-MicroDuck"

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, upstream's ``num_steps_per_env``."""

ACTOR_OBSERVATION_TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
]
"""The deployed MicroDuck observation layout, shared across the whole policy family.

Addendum section 11.1: every upstream MicroDuck task presents the same 61-wide vector, because one
runtime on the robot feeds every policy from the same buffer. The table is deliberately a copy of
the velocity task's rather than a reference to it -- if the two drift apart, both suites fail.
"""

ACTOR_OBSERVATION_DIM = 61
"""Actor observation width the deployed MicroDuck policy expects."""

CRITIC_OBSERVATION_TERMS = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("foot_height", 2),
    ("foot_air_time", 2),
    ("foot_contact", 2),
    ("foot_contact_forces", 6),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
]
"""The privileged critic layout, which is not a deploy contract.

Upstream adds no privileged observation for the recovery layer, so this is the velocity task's
critic unchanged -- including its two ``foot_height`` columns, which the stand-up task drops.
"""

CRITIC_OBSERVATION_DIM = 76
"""Critic observation width, measured from the assembled group."""

EXPECTED_SERVO_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 14 servos in upstream's MJCF actuator order: 0-4 left leg, 5-8 neck/head, 9-13 right leg."""

EXPECTED_LEG_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 10 leg joints the posture reward holds at the stand pose."""

EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The 4 head servos, in the order the head-pose command indexes its columns."""

EXPECTED_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot bodies in upstream's ``[left, right]`` site order."""

EXPECTED_TRUNK_BODY_NAMES = ["trunk_base"]
"""The body every recovery term measures the height, tilt and rise speed on."""

##
# The recovery layer (addendum sections 3.1 to 3.6)
##

EXPECTED_RECOVERY_REWARDS = {
    # name: (weight, scalar params)
    "upright_progress": (5.0, {}),
    "height_progress": (30.0, {"ceiling": 0.115}),
    "com_upward_velocity": (0.0, {"max_height": 0.125, "gate_tilt_above_deg": 40.0}),
    "fallen_tax": (
        0.0,
        {"gate_tilt_above_deg": 40.0, "release_tilt_below_deg": 25.0, "release_z_above": 0.09},
    ),
    "recovery_success": (
        0.0,
        {"fallen_tilt_deg": 40.0, "min_fallen_s": 0.5, "up_tilt_deg": 25.0, "up_z": 0.09},
    ),
    "joint_torque_rate_l2": (-2e-3, {}),
}
"""The six terms upstream adds to the walking recipe (addendum sections 3.2 and 3.3).

Three ship at weight zero and are ramped in together at iteration 1200, so these are the *initial*
weights and the schedules below are as load-bearing as they are.

``up_z`` is 0.09 rather than the 0.105 m upstream's own function defaults to: the extraction records
that the default demanded standing taller than the policy ever is, so the bounty never fired.
"""

EXPECTED_EDITED_REWARDS = {
    # name: the parameters the recovery layer adds to an inherited walking term
    "head_pose_bias": {
        "gate_height_low": 0.09,
        "gate_height_high": 0.11,
        "gate_tilt_full_deg": 20.0,
        "gate_tilt_zero_deg": 40.0,
    },
    "air_time": {"gate_tilt_above_deg": 40.0},
    "self_collisions": {"saturate": True},
}
"""The three inherited terms the recovery layer touches, and only what it adds to them.

``head_pose_bias`` and ``air_time`` are upstream's own two edits. ``self_collisions`` is the port's:
the wide sensor this task can afford reports a contact from each side of a pair, so the cost has to
saturate to stay on upstream's 0-or-1 scale.
"""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params)
    "time_out": (True, {}),
    "fell_over": (False, {"limit_angle": math.radians(70.0)}),
    "out_of_terrain_bounds": (True, {"distance_buffer": 20.3}),
    "nan_state": (False, {"sensor_names": ("contact_forces",)}),
    "fallen_too_long": (False, {"gate_z_below": 0.08, "gate_tilt_above_deg": 40.0, "max_duration_s": 8.0}),
}
"""Upstream's terminations (addendum section 3.5).

``fell_over`` is kept at its walking limit angle and *widened* by curriculum rather than deleted, and
``fallen_too_long`` is the backstop that makes that safe.
"""

EXPECTED_GROUND_STATE_PARAMS = {
    # the first curriculum stage: nothing but ordinary upright spawns
    "face_down_prob": 0.0,
    "face_up_prob": 0.0,
    "sitting_prob": 0.0,
    "standing_prob": 1.0,
    "crouch_prob": 0.0,
    "prone_z_range": (0.05, 0.09),
    "standing_z_range": (0.12, 0.13),
    "crouch_z_range": (0.06, 0.115),
    "crouch_joint_pos": MICRODUCK_CROUCH_JOINT_POS,
    "crouch_depth_range": (0.35, 1.0),
    "crouch_pitch_max": math.radians(55.0),
    "crouch_joint_noise": 0.12,
}
"""The ground-state reset's configuration (addendum section 3.4).

The height bands and the crouch anchor are configured from the first step even though the buckets
they belong to open hundreds of iterations later, because the curriculum flips a bucket live and a
live bucket with no band to spawn from raises.
"""

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "randomize_com",
    "randomize_head_com",
    "randomize_joint_friction",
    "randomize_armature",
    "set_ground_state",
]
"""The reset chain, in the order it fires.

This is behaviour, not housekeeping: ``set_ground_state`` overwrites the root height and orientation
``reset_base`` wrote, so it has to run after it. Upstream appends its equivalent last too.
"""

EXPECTED_CURRICULUM_TERMS = {
    # inherited from the walking recipe
    "terrain_levels",
    "action_rate_weight",
    "head_pose_bias_weight",
    "standing_envs",
    "head_pose_range",
    "body_pose_range",
    "com_range",
    "head_com_range",
    # the recovery layer's own
    "fell_over_disable",
    "ground_state_mix",
    "fallen_tax_weight",
    "recovery_success_weight",
    "com_upward_weight",
}
"""Every curriculum term (addendum section 3.6). ``terrain_levels`` is disabled by the flat task."""

EXPECTED_RECOVERY_WEIGHT_STAGES = {
    "fallen_tax_weight": ([0.0, -0.5], [0, 1200]),
    "recovery_success_weight": ([0.0, 10.0], [0, 1200]),
    "com_upward_weight": ([0.0, 2.0], [0, 1200]),
}
"""The recovery economics ramp: payloads and PPO-iteration boundaries, all three together."""

EXPECTED_GROUND_STATE_STAGES = [
    # (iteration, upstream's (prone_prob, face_down share, crouch_prob))
    (0, (0.00, 1.00, 0.00)),
    (800, (0.00, 1.00, 0.15)),
    (1500, (0.15, 0.80, 0.15)),
    (2000, (0.30, 0.65, 0.15)),
    (2500, (0.45, 0.50, 0.15)),
]
"""Upstream's ``PRONE_RAMP_STAGES`` (addendum section 3.1), in upstream's own parameterization.

The port stores the same mixture as four bucket probabilities, so the test converts rather than
restating the converted numbers -- otherwise the conversion would be checked against itself.
"""

EXPECTED_FELL_OVER_STAGES = [(0, math.radians(70.0)), (500, math.pi)]
"""The tilt termination's limit angle over the run: the walk-to-fall phase boundary."""

EXPECTED_ENTITY_SELECTIONS = {
    # "<manager>.<term>.<param>": (kind, expected names, preserve_order)
    "rewards.upright_progress.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.height_progress.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.com_upward_velocity.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.fallen_tax.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.recovery_success.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.joint_torque_rate_l2.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "events.set_ground_state.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
}
"""The entity selections the recovery layer makes, outside the observation groups.

Only the layer's own terms are listed: everything else is inherited and is pinned by
``test_microduck_env.py``, and the "verbatim" test below is what proves it is still inherited. A
term that measures the wrong joints is as wrong as one carrying the wrong weight -- the torque-rate
penalty sizes its state from its selection, and the ground-state reset's joint noise reaches
whatever it names.
"""

EXPECTED_OBSERVATION_SELECTIONS = {
    # "<group>.<term>": (kind, expected names)
    "policy.joint_pos": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "policy.joint_vel": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "critic.joint_pos": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "critic.joint_vel": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    # the foot height is read off the articulation's ankle bodies; the other three off the sensor
    "critic.foot_height": ("body", EXPECTED_FOOT_BODY_NAMES),
    "critic.foot_air_time": ("sensor", EXPECTED_FOOT_BODY_NAMES),
    "critic.foot_contact": ("sensor", EXPECTED_FOOT_BODY_NAMES),
    "critic.foot_contact_forces": ("sensor", EXPECTED_FOOT_BODY_NAMES),
}
"""Entity selections inside the two observation groups, all of which are ordering contracts.

The joint blocks are the deployed vector's own layout, which the runtime on the robot rebuilds by
hand from its sensor reads; the foot blocks are four consecutive critic terms that must agree on
which column is the left foot. They are inherited here, but they are the contract this task ships,
so they are asserted rather than assumed.
"""

EXPECTED_NJMAX = 128
EXPECTED_NCONMAX = 32
"""The measured solver budget: 82 constraints and 27 contacts per environment at the peak."""

MEASURED_PEAK_CONSTRAINTS = 82
MEASURED_PEAK_CONTACTS = 27
"""Profiled peaks under random actions with the tilt termination dropped and the pushes at full
magnitude, identical at 256 and at 2048 environments. Logs:
``artifacts/microduck/profile_microduck_contacts_velstand_{256,2048}envs.log``."""


def _entity_cfg_of(term_cfg, key: str) -> SceneEntityCfg:
    """Fetch a term's entity selection, looking inside a delayed term's wrapped parameters."""
    if key in term_cfg.params:
        return term_cfg.params[key]
    return term_cfg.params["term_params"][key]


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which carry no upstream scalar to compare against.

    They are not left unchecked: :data:`EXPECTED_ENTITY_SELECTIONS` and
    :data:`EXPECTED_OBSERVATION_SELECTIONS` pin them by name, and the two ``*_select_*`` tests below
    are what assert it.
    """
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


def _bucket_mix(prone_prob: float, face_down_share: float, crouch_prob: float) -> dict[str, float]:
    """Upstream's ``(prone, face-down share, crouch)`` triple as the bucket probabilities it means.

    Worked out here from upstream's own description rather than imported from the configuration, so
    that the conversion is checked rather than restated.
    """
    return {
        "face_down_prob": prone_prob * face_down_share,
        "face_up_prob": prone_prob * (1.0 - face_down_share),
        "crouch_prob": crouch_prob,
        "standing_prob": 1.0 - prone_prob - crouch_prob,
    }


##
# The walking layer: inherited verbatim
##


@pytest.mark.unit
def test_the_walking_recipe_is_the_velocity_task_verbatim():
    """The contract this task is built on: the proven walker, with three parameter edits on top.

    Restating the walking weights in this file would let the two recipes drift apart while both
    suites still passed, which is the one failure mode a derived task has and a standalone one does
    not.
    """
    velstand = MicroDuckVelStandFlatEnvCfg()
    velocity = MicroDuckVelocityFlatEnvCfg()

    inherited = set(vars(velocity.rewards))
    assert set(vars(velstand.rewards)) == inherited | set(EXPECTED_RECOVERY_REWARDS)
    for name in inherited:
        ours, theirs = getattr(velstand.rewards, name), getattr(velocity.rewards, name)
        assert ours.func is theirs.func, name
        assert ours.weight == pytest.approx(theirs.weight), name
        added = EXPECTED_EDITED_REWARDS.get(name, {})
        assert set(_scalar_params(ours)) == set(_scalar_params(theirs)) | set(added), name
        for key, value in _scalar_params(theirs).items():
            if key not in added:
                assert _scalar_params(ours)[key] == value, f"{name}.{key}"

    # the commands, the actions and both observation groups are inherited untouched
    assert vars(velstand.commands).keys() == vars(velocity.commands).keys()
    assert vars(velstand.actions) == vars(velocity.actions)
    for group in ("policy", "critic"):
        ours = _observation_terms(getattr(velstand.observations, group))
        theirs = _observation_terms(getattr(velocity.observations, group))
        assert list(ours) == list(theirs), group


@pytest.mark.unit
def test_the_three_edited_walking_terms_gain_exactly_the_recovery_parameters():
    """Two edits are upstream's and one is the port's; all three are gates, not retunes."""
    rewards = MicroDuckVelStandFlatEnvCfg().rewards

    for name, added in EXPECTED_EDITED_REWARDS.items():
        actual = _scalar_params(getattr(rewards, name))
        for key, value in added.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"
    # the gates all close at the same tilt the recovery rewards open at, so one number moves them
    assert rewards.air_time.params["gate_tilt_above_deg"] == pytest.approx(MICRODUCK_FALLEN_TILT_DEG)
    assert rewards.head_pose_bias.params["gate_tilt_zero_deg"] == pytest.approx(MICRODUCK_FALLEN_TILT_DEG)


##
# The recovery layer
##


@pytest.mark.unit
def test_the_recovery_rewards_match_upstream_term_for_term():
    """Every recovery slot upstream trains with is present, with its weight and its parameters."""
    rewards = MicroDuckVelStandFlatEnvCfg().rewards

    for name, (weight, params) in EXPECTED_RECOVERY_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_recovery_layer_pays_nothing_at_all_during_clean_walking():
    """Its whole design claim: two potential terms and four gated ones, so the walk is untouched.

    A recovery reward that paid a standing robot would change the walking recipe's balance without
    changing any of its weights, which is exactly the failure the extraction records upstream
    chasing across seven runs.
    """
    rewards = MicroDuckVelStandFlatEnvCfg().rewards

    # potential-based: the change in a scalar, so any held pose is worth zero
    for name in ("upright_progress", "height_progress"):
        assert getattr(rewards, name).func in (mdp.upright_progress, mdp.height_progress), name
    # explicitly gated on being toppled
    for name in ("com_upward_velocity", "fallen_tax", "recovery_success"):
        gate = _scalar_params(getattr(rewards, name))
        opening = gate.get("gate_tilt_above_deg", gate.get("fallen_tilt_deg"))
        assert opening == pytest.approx(MICRODUCK_FALLEN_TILT_DEG), name
        # gating on height too would pay a robot for sitting, which is upstream's run-1 lesson
        assert "gate_z_below" not in gate, name
    # and the two that do act everywhere are a cost, not a subsidy
    assert rewards.joint_torque_rate_l2.weight < 0.0


@pytest.mark.unit
def test_the_recovery_economics_are_signed_the_way_their_kernels_return():
    """The tax returns a positive magnitude and the bounty a positive event, so the signs differ."""
    curriculum = MicroDuckVelStandFlatEnvCfg().curriculum

    tax = curriculum.fallen_tax_weight.params["weight_stages"]
    bounty = curriculum.recovery_success_weight.params["weight_stages"]
    assert all(stage["weight"] <= 0.0 for stage in tax)
    assert all(stage["weight"] >= 0.0 for stage in bounty)
    # and the bounty has to outweigh a whole timeout of tax, or waiting stays rational: eight
    # seconds at 50 Hz is 400 steps of -0.5
    assert bounty[-1]["weight"] > 0.0
    assert tax[-1]["weight"] < 0.0


@pytest.mark.unit
def test_the_recovery_gates_are_ordered_the_way_a_recovery_passes_through_them():
    """A robot must clear the arming gate before it can clear the release one, or the tax never lifts."""
    assert MICRODUCK_RECOVERED_TILT_DEG < MICRODUCK_FALLEN_TILT_DEG
    # the release height sits inside the policy's real standing envelope (0.084-0.096 m measured),
    # below the settled stand the rise potential tops out at
    assert MICRODUCK_RECOVERED_HEIGHT < MICRODUCK_STAND_HEIGHT
    # the termination's height gate is below the release height, so a completed recovery is never
    # simultaneously "recovered" and "still down"
    assert MICRODUCK_TERMINATION_Z < MICRODUCK_RECOVERED_HEIGHT


@pytest.mark.unit
def test_the_ground_state_reset_configures_every_bucket_the_curriculum_will_open():
    """The curriculum flips buckets live, so their bands have to be there from the first step."""
    params = _scalar_params(MicroDuckVelStandFlatEnvCfg().events.set_ground_state)

    assert set(params) == set(EXPECTED_GROUND_STATE_PARAMS)
    for key, value in EXPECTED_GROUND_STATE_PARAMS.items():
        assert params[key] == pytest.approx(value), key
    # the crouch anchor is a fold of the sagittal chain only: hip yaw, hip roll and the neck stay at
    # the stand pose, and every angle is inside the model's +/-1.571 rad limits
    assert set(MICRODUCK_CROUCH_JOINT_POS) < set(EXPECTED_LEG_JOINT_NAMES)
    assert all(abs(angle) < 1.571 for angle in MICRODUCK_CROUCH_JOINT_POS.values())


@pytest.mark.unit
def test_the_ground_state_reset_runs_after_the_resets_it_overwrites():
    """Isaac Lab fires reset events in declaration order, and this order is the spawn distribution."""
    events = MicroDuckVelStandFlatEnvCfg().events

    reset_terms = [name for name, term in vars(events).items() if getattr(term, "mode", None) == "reset"]
    assert reset_terms == EXPECTED_RESET_EVENT_ORDER


@pytest.mark.unit
def test_the_terminations_keep_the_tilt_check_and_add_the_recovery_backstop():
    """Unlike the stand-up task, this one starts upright, so the tilt termination is what walks first."""
    terminations = MicroDuckVelStandFlatEnvCfg().terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        assert _scalar_params(term) == pytest.approx(params), name
    # the backstop is a real failure, not a time-out: the value must not bootstrap over lying down
    assert terminations.fallen_too_long.time_out is False


@pytest.mark.unit
def test_the_termination_gate_is_height_or_tilt_where_the_reward_gates_are_tilt_alone():
    """The asymmetry is upstream's: a sitter is recycled as stuck rather than paid as fallen."""
    cfg = MicroDuckVelStandFlatEnvCfg()

    termination = cfg.terminations.fallen_too_long.params
    assert termination["gate_z_below"] == pytest.approx(MICRODUCK_TERMINATION_Z)
    assert termination["gate_tilt_above_deg"] == pytest.approx(MICRODUCK_FALLEN_TILT_DEG)
    for name in ("com_upward_velocity", "fallen_tax", "recovery_success"):
        assert "gate_z_below" not in _scalar_params(getattr(cfg.rewards, name)), name


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """The three-phase schedule *is* the task; a stage boundary that moved would be a different run."""
    curriculum = MicroDuckVelStandFlatEnvCfg().curriculum

    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS
    for name, (weights, iterations) in EXPECTED_RECOVERY_WEIGHT_STAGES.items():
        stages = getattr(curriculum, name).params["weight_stages"]
        assert [stage["weight"] for stage in stages] == pytest.approx(weights), name
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
    # all three switch on together, because the tax-free window is what the gap between the
    # fall phase and the economics phase buys
    boundaries = {
        getattr(curriculum, name).params["weight_stages"][-1]["step"] for name in EXPECTED_RECOVERY_WEIGHT_STAGES
    }
    assert boundaries == {1200 * STEPS_PER_ITERATION}


@pytest.mark.unit
def test_the_fall_phase_opens_before_the_economics_and_the_prone_ramp():
    """Order the three phases wrong and prone recovery never bootstraps -- upstream's run-6 lesson."""
    curriculum = MicroDuckVelStandFlatEnvCfg().curriculum

    fell_over_off = curriculum.fell_over_disable.params["param_stages"][-1]["step"]
    economics = curriculum.fallen_tax_weight.params["weight_stages"][-1]["step"]
    first_prone = next(
        stage["step"]
        for stage in curriculum.ground_state_mix.params["param_stages"]
        if stage["params"]["face_down_prob"] > 0.0
    )
    assert fell_over_off < economics < first_prone


@pytest.mark.unit
def test_the_fell_over_curriculum_widens_the_limit_rather_than_deleting_the_term():
    """Deleting it would change the termination manager's shape mid-run; widening it does not."""
    curriculum = MicroDuckVelStandFlatEnvCfg().curriculum

    stages = curriculum.fell_over_disable.params["param_stages"]
    assert curriculum.fell_over_disable.params["term_name"] == "fell_over"
    assert [(stage["step"] // STEPS_PER_ITERATION, stage["params"]["limit_angle"]) for stage in stages] == [
        (iteration, pytest.approx(angle)) for iteration, angle in EXPECTED_FELL_OVER_STAGES
    ]
    # half a turn is past any orientation, so the termination is inert rather than merely generous
    assert stages[-1]["params"]["limit_angle"] >= math.pi


@pytest.mark.unit
def test_the_reset_mix_ramps_toward_the_prone_poses_without_starving_the_walk():
    """Capped at 45 % prone, so at least 55 % of the experience is still clean walking."""
    curriculum = MicroDuckVelStandFlatEnvCfg().curriculum

    stages = curriculum.ground_state_mix.params["param_stages"]
    assert curriculum.ground_state_mix.params["event_name"] == "set_ground_state"
    assert len(stages) == len(EXPECTED_GROUND_STATE_STAGES)
    for stage, (iteration, triple) in zip(stages, EXPECTED_GROUND_STATE_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        expected = _bucket_mix(*triple)
        assert set(stage["params"]) == set(expected)
        for key, value in expected.items():
            assert stage["params"][key] == pytest.approx(value), f"{iteration}.{key}"
        # the buckets partition the resets, so they have to sum to one at every stage
        assert sum(stage["params"].values()) == pytest.approx(1.0)

    prone = [stage["params"]["face_down_prob"] + stage["params"]["face_up_prob"] for stage in stages]
    assert max(prone) == pytest.approx(0.45)
    # face-down first, face-up mixed in later: the easier recovery is learned before the harder one
    face_up = [stage["params"]["face_up_prob"] for stage in stages]
    assert face_up == sorted(face_up)
    # and the crouch slice opens before either prone pose, at a stage that adds nothing else
    assert stages[1]["params"]["crouch_prob"] > 0.0 and prone[1] == pytest.approx(0.0)


@pytest.mark.unit
def test_the_terms_select_the_joints_and_bodies_upstream_measures():
    """A term that measures the wrong joints is as wrong as one carrying the wrong weight."""
    cfg = MicroDuckVelStandFlatEnvCfg()

    for path, (kind, expected, preserve_order) in EXPECTED_ENTITY_SELECTIONS.items():
        manager, term_name, key = path.split(".")
        entity_cfg = getattr(getattr(cfg, manager), term_name).params[key]
        assert entity_cfg.name == "robot", path
        if kind == "joint":
            assert entity_cfg.joint_names == expected, path
            assert entity_cfg.body_names is None, path
        else:
            assert entity_cfg.body_names == expected, path
            assert entity_cfg.joint_names is None, path
        assert entity_cfg.preserve_order is preserve_order, path

    # every recovery term this task adds carries a selection, and every one of them is in the table
    measured = {
        f"rewards.{name}.{key}"
        for name in EXPECTED_RECOVERY_REWARDS
        for key, value in getattr(cfg.rewards, name).params.items()
        if isinstance(value, SceneEntityCfg)
    }
    assert measured == {path for path in EXPECTED_ENTITY_SELECTIONS if path.startswith("rewards.")}


@pytest.mark.unit
def test_both_observation_groups_read_the_servos_and_the_feet_in_the_deploy_order():
    """Isaac Lab resolves joints and bodies in USD order; the deployed vector is in MJCF order."""
    observations = MicroDuckVelStandFlatEnvCfg().observations
    groups = {"policy": _observation_terms(observations.policy), "critic": _observation_terms(observations.critic)}

    for path, (kind, expected) in EXPECTED_OBSERVATION_SELECTIONS.items():
        group, term_name = path.split(".")
        term = groups[group][term_name]
        # the delayed actor terms hold their selection inside the wrapped term's parameters
        entity_cfg = _entity_cfg_of(term, "sensor_cfg" if kind == "sensor" else "asset_cfg")
        if kind == "joint":
            assert entity_cfg.name == "robot", path
            assert entity_cfg.joint_names == expected, path
        else:
            assert entity_cfg.name == ("contact_forces" if kind == "sensor" else "robot"), path
            assert entity_cfg.body_names == expected, path
        assert entity_cfg.preserve_order, path

    # the head rewards index their command's columns positionally, so their joint order is the
    # command's; the observation joint order is the deployed vector's. Different contracts, same
    # articulation, and the head block of one is the other's.
    assert EXPECTED_SERVO_JOINT_NAMES[5:9] == EXPECTED_HEAD_JOINT_NAMES


@pytest.mark.unit
def test_the_actor_observation_layout_is_the_shared_deploy_contract():
    """One runtime on the robot feeds every MicroDuck policy from the same 61-wide buffer."""
    terms = _observation_terms(MicroDuckVelStandFlatEnvCfg().observations.policy)

    assert list(terms) == [name for name, _ in ACTOR_OBSERVATION_TERMS]
    assert sum(width for _, width in ACTOR_OBSERVATION_TERMS) == ACTOR_OBSERVATION_DIM


@pytest.mark.unit
def test_the_critic_group_keeps_the_walking_tasks_foot_height_columns():
    """Upstream adds no privileged observation here, so the critic is the walking task's unchanged."""
    terms = _observation_terms(MicroDuckVelStandFlatEnvCfg().observations.critic)

    assert list(terms) == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert sum(width for _, width in CRITIC_OBSERVATION_TERMS) == CRITIC_OBSERVATION_DIM


@pytest.mark.unit
def test_the_task_runs_the_all_collisions_robot_on_a_plane():
    """A robot that lies down needs the ten world colliders the walking model does not have."""
    scene = MicroDuckVelStandFlatEnvCfg().scene

    assert scene.robot.spawn.usd_path.endswith("microduck_allcollisions.usd")
    assert scene.terrain.terrain_type == "plane"
    assert scene.terrain.terrain_generator is None


@pytest.mark.unit
def test_the_self_collision_sensor_senses_every_collider_against_every_other():
    """The walking scene can only watch sole against sole; this model makes the wide sensor reachable."""
    cfg = MicroDuckVelStandFlatEnvCfg()

    sensor = cfg.scene.self_collision
    assert sensor.sensor_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    assert sensor.filter_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    # sensing both sides of a pair reports one contact twice, so the cost saturates to upstream's
    # single-slot 0-or-1 signal rather than becoming a per-collider tariff
    assert cfg.rewards.self_collisions.params["saturate"] is True


@pytest.mark.unit
def test_the_solver_budget_covers_the_measured_contact_peak():
    """The walking budget is sized for two soles on a plane; this robot lies on ten colliders."""
    solver = MicroDuckVelStandFlatEnvCfg().sim.physics.newton_mjwarp.solver_cfg

    assert solver.njmax == EXPECTED_NJMAX
    assert solver.nconmax == EXPECTED_NCONMAX
    # a floor under the measurement, so a later retune cannot silently drop below it
    assert solver.njmax >= MEASURED_PEAK_CONSTRAINTS
    assert solver.nconmax >= MEASURED_PEAK_CONTACTS
    # and above the walking task's, which is what says the swap was accounted for
    assert solver.nconmax > MicroDuckVelocityFlatEnvCfg().sim.physics.newton_mjwarp.solver_cfg.nconmax


@pytest.mark.unit
def test_the_episode_and_simulation_rates_match_upstream():
    """Upstream overrides neither, so the walking task's 20 s episode at 50 Hz carries over."""
    cfg = MicroDuckVelStandFlatEnvCfg()

    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.episode_length_s == pytest.approx(20.0)
    # the failed-recovery backstop has to fit inside the episode with room to walk afterwards
    assert cfg.terminations.fallen_too_long.params["max_duration_s"] < cfg.episode_length_s / 2.0


@pytest.mark.unit
def test_the_runner_differs_from_the_velocity_one_in_two_fields():
    """Upstream shares the network, the optimizer and the rollout across the whole family."""
    from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg

    runner, velocity = MicroDuckVelStandPPORunnerCfg(), MicroDuckPPORunnerCfg()

    assert runner.experiment_name == "microduck_velstand"
    assert runner.max_iterations == 20000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION == velocity.num_steps_per_env
    assert runner.save_interval == velocity.save_interval == 250
    assert runner.actor.hidden_dims == velocity.actor.hidden_dims == [512, 256, 128]
    assert runner.critic.hidden_dims == velocity.critic.hidden_dims
    assert runner.obs_groups == velocity.obs_groups
    for field in ("clip_param", "entropy_coef", "learning_rate", "gamma", "lam", "desired_kl"):
        assert getattr(runner.algorithm, field) == getattr(velocity.algorithm, field), field
    # the budget is the one field that is not the walking task's
    assert runner.max_iterations != velocity.max_iterations


##
# Simulator-backed acceptance
##


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_observation_groups_are_the_widths_their_contracts_name():
    """The actor width is a deploy contract; the critic width is measured against the term table."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        observations, _ = env.reset()

        assert observations["policy"].shape[1] == ACTOR_OBSERVATION_DIM
        assert observations["critic"].shape[1] == CRITIC_OBSERVATION_DIM
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_reset_spawns_the_prone_and_crouch_poses_the_recovery_layer_trains_on():
    """The reset distribution is what this task trains on; a mix that collapses to one is a failure."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=8)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        # every bucket equally likely, in the *live* managers: the curriculum rewrites the event's
        # probabilities on every reset, so flattening its schedule is what makes the mix stick
        mix = {"face_down_prob": 0.25, "face_up_prob": 0.25, "crouch_prob": 0.25, "standing_prob": 0.25}
        index = unwrapped.curriculum_manager._term_names.index("ground_state_mix")
        unwrapped.curriculum_manager._term_cfgs[index].params["param_stages"] = [{"step": 0, "params": mix}]
        unwrapped.event_manager.get_term_cfg("set_ground_state").params.update(mix)

        robot = unwrapped.scene["robot"]
        origins = unwrapped.scene.env_origins
        knee_ids, _ = robot.find_joints(["left_knee"], preserve_order=True)
        default_knee = robot.data.default_joint_pos.torch[0, knee_ids[0]].item()

        # cloned, not referenced: the articulation's data tensors are written in place on every
        # reset, so keeping the views would collapse every sample onto the last one
        heights, cos_tilts, knees = [], [], []
        for _ in range(12):
            unwrapped.reset()
            heights.append((robot.data.root_link_pos_w.torch[:, 2] - origins[:, 2]).clone())
            quat = robot.data.root_link_quat_w.torch
            cos_tilts.append((1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)).clone())
            knees.append(robot.data.joint_pos.torch[:, knee_ids[0]].clone())
        height, cos_tilt, knee = torch.cat(heights), torch.cat(cos_tilts), torch.cat(knees)

        # every spawn lands inside one of the configured bands
        assert height.min().item() >= EXPECTED_GROUND_STATE_PARAMS["prone_z_range"][0] - 1e-4
        assert height.max().item() <= EXPECTED_GROUND_STATE_PARAMS["standing_z_range"][1] + 1e-4
        # prone spawns are lying down: cos(tilt) near zero rather than near one
        assert (cos_tilt.abs() < 0.2).any(), "no face-down or face-up spawn"
        # ordinary spawns stand upright at walking height
        upright = cos_tilt > 0.99
        assert (height[upright] > EXPECTED_GROUND_STATE_PARAMS["standing_z_range"][0] - 1e-4).any()
        # only the crouch bucket folds the knees, and it folds them far past the stand pose while
        # leaning the trunk forward -- a partly-tilted, partly-lowered pose neither other bucket makes
        folded = (knee - default_knee).abs() > 0.3
        assert folded.any(), "no crouch spawn folded its knees"
        assert (cos_tilt[folded] < 0.999).all(), "a crouch spawn was left perfectly upright"
        assert height[folded].max().item() <= EXPECTED_GROUND_STATE_PARAMS["crouch_z_range"][1] + 0.011
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
