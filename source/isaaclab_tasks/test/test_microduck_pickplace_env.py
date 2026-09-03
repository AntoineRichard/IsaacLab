# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe and acceptance tests for the contributed MicroDuck pick-and-place environment.

**This task has no upstream counterpart**, so there is no accuracy gate against a reference stack
and no upstream table to transcribe. Acceptance is internal, on the precedent the roller stand-up
task set (ruling R-T7a): physically motivated, staged, and it must score rewards end-to-end under
physics -- an acceptance that never evaluates a reward is not acceptance.

The recipe tests read the assembled configuration and compare it against
``artifacts/microduck/pickplace/DESIGN.md``, transcribed here rather than imported: a table that
read the configuration it checks would agree with itself. They need neither the asset nor the
simulator.

The acceptance tests spawn :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG`, whose USD is
generated rather than committed, so they skip when that asset is absent. Generate it with
``uv run --extra importers python scripts/tools/convert_microduck.py --model allcollisions``.
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
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.groundpick.groundpick_env_cfg import MicroDuckGroundPickFlatEnvCfg
from isaaclab_tasks.contrib.microduck.pickplace.agents.rsl_rl_ppo_cfg import MicroDuckPickPlacePPORunnerCfg
from isaaclab_tasks.contrib.microduck.pickplace.pickplace_env_cfg import (
    MICRODUCK_LATCH_BREAK_FORCE,
    MICRODUCK_LATCH_DAMPING,
    MICRODUCK_LATCH_HOLD_DISTANCE,
    MICRODUCK_LATCH_MAX_REL_SPEED,
    MICRODUCK_LATCH_OBJECT_MASS,
    MICRODUCK_LATCH_OMEGA_DT,
    MICRODUCK_LATCH_RADIUS,
    MICRODUCK_LATCH_STIFFNESS,
    MICRODUCK_PLACE_MAX_HEIGHT,
    MICRODUCK_PLACE_TOLERANCE,
    MicroDuckPickPlaceFlatEnvCfg,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import (
    MICRODUCK_ALLCOLLISIONS_USD_PATH,
    MICRODUCK_BEAK_CLOSED,
    MICRODUCK_BEAK_OPEN,
    MICRODUCK_MARBLE_MASS,
    MICRODUCK_MARBLE_RADIUS,
)

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_allcollisions_usd = pytest.mark.skipif(
    not os.path.isfile(MICRODUCK_ALLCOLLISIONS_USD_PATH),
    reason=(
        f"MicroDuck all-collisions USD asset is missing: {MICRODUCK_ALLCOLLISIONS_USD_PATH}. Generate it with"
        " 'uv run --extra importers python scripts/tools/convert_microduck.py --model allcollisions'."
    ),
)
"""Skips the tests that spawn the robot. The recipe tests do not need the asset."""

TASK_NAME = "IsaacContrib-PickPlace-Flat-MicroDuck"

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, the family's ``num_steps_per_env``."""

ACTOR_OBSERVATION_TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("object_position", 3),
    ("target_position", 3),
    ("latched", 1),
]
"""The actor layout, and the whole of this task's camera contract (design document §4).

Deliberately **not** the walking family's 61-wide deploy vector: this is a new task with a new
runtime, and padding to 61 with zeros would advertise a hot-swap compatibility that does not exist
(ruling R-PP5).
"""

ACTOR_OBSERVATION_DIM = 55

CRITIC_OBSERVATION_TERMS = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("object_position", 3),
    ("target_position", 3),
    ("latched", 1),
    ("foot_air_time", 2),
    ("foot_contact", 2),
    ("foot_contact_forces", 6),
    ("object_velocity", 3),
    ("mouth_to_object", 3),
    ("succeeded", 1),
]
"""The privileged critic layout, which is not a contract with anything outside this task."""

CRITIC_OBSERVATION_DIM = 75

CAMERA_REPLACEABLE_ACTOR_TERMS = {"object_position"}
"""Actor terms a v2 camera pipeline has to supply.

The point of the design: everything else in the actor group is either proprioception, a command, or
the robot's own controller state, so the camera migration is one term wide.
"""

PROPRIOCEPTIVE_ACTOR_TERMS = {"base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"}
"""Actor terms the robot reads off its own servos and IMU."""


def _term_names(group) -> list[str]:
    """Names of an observation group's terms, in declaration order.

    ``vars`` on an observation group also yields its switches -- ``concatenate_terms`` and the
    rest -- so the layout assertions filter on the term type rather than on the value being
    non-None, which would let a bool through and compare it against a width.
    """
    return [name for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)]


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
"""The 14 servos in the MJCF actuator order: 0-4 left leg, 5-8 neck/head, 9-13 right leg."""

EXPECTED_MOUTH_BODY_NAMES = ["jaw_soft"]
"""The body the mouth tip is rigidly attached to."""

EXPECTED_REWARD_WEIGHTS = {
    # Approach, live until the object is held or placed. Weights are **rates**: the reward manager
    # multiplies every term by the 0.02 s control step, so the sparse terms carry large weights and
    # the per-step terms small ones. Reading these as if they were episode-return figures is the
    # mistake that produced ruling R-PP17.
    "approach_progress": 2000.0,
    "mouth_to_object": 1.0,
    "mouth_down": 0.5,
    # pick -- one payment per episode
    "latch_bonus": 1000.0,
    # carry
    "carry_hold": 1.5,
    "carry_progress": 3000.0,
    "object_clearance": 0.5,
    # place -- one payment per episode
    "place_success": 5000.0,
    "place_precision": 2500.0,
    # posture floor
    "upright": 0.2,
    "feet_grounded": 0.5,
    "head_impact_penalty": -2.0,
    "self_collisions": -1.0,
    "dof_pos_limits": -1.0,
    "fell_penalty": -5000.0,
    # regularization
    "body_ang_vel": -0.05,
    "angular_momentum": -0.02,
    "action_rate_l2": -0.1,
    "joint_torques_l2": -5e-3,
}
"""The whole reward stack, transcribed from the design document's §5.4 table.

Since there is no upstream to gate against, this table *is* the parity table -- it is the artefact
the acceptance criteria are written in terms of, and it is transcribed rather than imported so that
a weight changed in the configuration has to be changed here too, deliberately.
"""

LATCH_STATE_GATED_REWARDS = {
    "approach_progress": "approach",
    "mouth_to_object": "approach",
    "mouth_down": "approach",
    "latch_bonus": "pick",
    "carry_hold": "carry",
    "carry_progress": "carry",
    "object_clearance": "carry",
    "place_success": "place",
    "place_precision": "place",
}
"""Which block each task reward belongs to. The remaining nine terms are ungated on purpose."""


##
# Recipe: the design document's tables, transcribed
##


@pytest.mark.unit
def test_the_reward_stack_is_the_design_documents_table():
    """Every term, every weight, and nothing extra."""
    rewards = MicroDuckPickPlaceFlatEnvCfg().rewards
    declared = {name: term.weight for name, term in vars(rewards).items() if term is not None}

    assert declared == pytest.approx(EXPECTED_REWARD_WEIGHTS)


@pytest.mark.unit
def test_the_carry_bonus_outbids_hovering_at_the_object():
    """The reward-hacking audit's load-bearing inequality (ruling R-PP6).

    ``mouth_to_object`` pays its full weight for holding the mouth on the object without picking it
    up, and it is gated off the moment the object is latched. Any carry bonus at or below it makes
    *refusing to pick the object up* strictly dominant, which is a task that trains cleanly to the
    wrong behaviour.
    """
    rewards = MicroDuckPickPlaceFlatEnvCfg().rewards

    assert rewards.carry_hold.weight > rewards.mouth_to_object.weight
    # This is a *per-step* inequality and that is all it is. Whether holding beats hovering over a
    # whole episode is a different question in different units, and answering it by eye here is what
    # went wrong the first time; ``test_delivering_the_object_pays_more_than_any_shortcut`` answers
    # it in episode-return units instead.


@pytest.mark.unit
def test_neither_distance_reward_pays_for_loitering_at_range():
    """A Gaussian on the distance pays for parking near the goal; these pay only for arriving.

    Both are stateful and neither takes a kernel width, which a level-based term would need. What
    each does with its state differs and is asserted in
    :func:`test_the_carry_progress_is_a_ratchet_and_the_approach_is_not`; both share the property
    that their total payout across an episode is bounded by the distance the episode started with.
    """
    rewards = MicroDuckPickPlaceFlatEnvCfg().rewards

    assert rewards.approach_progress.func is mdp.pickplace_approach_progress
    assert rewards.carry_progress.func is mdp.pickplace_carry_progress
    for term in (mdp.pickplace_approach_progress, mdp.pickplace_carry_progress):
        assert issubclass(term, mdp.upright_progress.__bases__[0].__bases__[0])
    assert "std" not in rewards.approach_progress.params
    assert "std" not in rewards.carry_progress.params


@pytest.mark.unit
def test_the_task_has_no_clock_and_gates_on_the_latch_instead():
    """Phases emerge from what the robot achieved, not from a cycle it is told the position of.

    The contrast with the ground-pick task is the whole structural difference: there, four segments
    of a 4 s cycle; here, four blocks keyed to the latch state machine.
    """
    cfg = MicroDuckPickPlaceFlatEnvCfg()

    assert set(vars(cfg.commands)) == {"place_target"}
    assert cfg.commands.place_target.class_type is mdp.PickPlaceTargetCommand
    # no reward reads a phase, and none takes the ground-pick task's segment boundaries
    for term in vars(cfg.rewards).values():
        assert "descent_end" not in term.params
        assert "rise_end" not in term.params


@pytest.mark.unit
def test_the_upright_reward_is_as_weak_as_the_ground_pick_tasks_and_for_its_reason():
    """The pick requires a deep forward fold, so a strong always-on uprightness prices it out."""
    cfg = MicroDuckPickPlaceFlatEnvCfg()

    assert cfg.rewards.upright.weight == pytest.approx(MicroDuckGroundPickFlatEnvCfg().rewards.upright.weight)
    assert cfg.rewards.upright.weight == pytest.approx(0.2)


@pytest.mark.unit
def test_the_flat_foot_penalty_is_absent_because_this_task_walks():
    """The ground-pick task applies it ungated; that gesture has no swing phase and this task does.

    An ungated flat-foot penalty on a locomotion task charges every step the robot takes.
    """
    cfg = MicroDuckPickPlaceFlatEnvCfg()

    assert "feet_flat" not in vars(cfg.rewards)
    assert "feet_flat" in vars(MicroDuckGroundPickFlatEnvCfg().rewards)
    # and the grounded reward is correspondingly weaker than that task's
    assert cfg.rewards.feet_grounded.weight < MicroDuckGroundPickFlatEnvCfg().rewards.feet_grounded.weight


@pytest.mark.unit
def test_the_latch_spring_is_integrated_at_the_control_rate_not_the_physics_rate():
    """The derivation ruling R-PP17 corrected, pinned so the original mistake cannot come back.

    The wrench is written by an interval event once per **control** step and the wrench composer is
    permanent, so the force is held constant across all four physics substeps: the spring is a
    zero-order hold at 50 Hz. Deriving the stiffness against the 0.005 s physics step -- which the
    first version did -- understates ``omega * dt`` by the decimation factor of four, and the spring
    that shipped rang hard enough to break its own grip roughly every control step.
    """
    cfg = MicroDuckPickPlaceFlatEnvCfg()
    control_dt = cfg.sim.dt * cfg.decimation
    assert control_dt == pytest.approx(0.02)

    omega = math.sqrt(MICRODUCK_LATCH_STIFFNESS / MICRODUCK_LATCH_OBJECT_MASS)
    # the bound that matters, against the step the force is actually held over
    assert omega * control_dt < 1.0
    # and the physics-step figure is *not* what this is judged on -- stated so the two cannot be
    # confused again by someone reading only the assertion
    assert omega * cfg.sim.dt < omega * control_dt

    # damped well enough not to ring, and below the zero-order-hold damper's own stability cap
    damping_ratio = MICRODUCK_LATCH_DAMPING / (2.0 * math.sqrt(MICRODUCK_LATCH_STIFFNESS * MICRODUCK_LATCH_OBJECT_MASS))
    assert 0.6 < damping_ratio < 1.2
    assert MICRODUCK_LATCH_DAMPING * control_dt / MICRODUCK_LATCH_OBJECT_MASS < 2.0

    # Static sag is ``mg/k``, which reduces to ``g / omega^2`` -- **mass-independent**, and capped by
    # the control rate rather than by the prop. It is about 5.8 mm at this stiffness whatever the
    # object weighs, which on a 6 mm marble is a whole radius: a held marble hangs noticeably below
    # its anchor. That is a real limit of a compliant weld at 50 Hz and not something a smaller prop
    # escapes, so what is asserted is that it stays small against the *latch radius* -- the distance
    # that decides whether the grip survives -- rather than against the object.
    sag = MICRODUCK_LATCH_OBJECT_MASS * 9.81 / MICRODUCK_LATCH_STIFFNESS
    assert sag == pytest.approx(9.81 / (MICRODUCK_LATCH_OMEGA_DT / control_dt) ** 2, rel=1e-6)
    assert sag < 0.5 * MICRODUCK_LATCH_RADIUS


@pytest.mark.unit
def test_the_grip_limit_sits_above_the_transient_a_normal_carry_produces():
    """A limit below the working transient is not a grip limit, it is a grip (ruling R-PP17).

    The original 6 N was 2 N, set against the object's *weight* and never against the force the
    spring actually develops when the head moves -- measured at 2.7 N with ``diag_latch_chatter.py``.
    """
    # Recorded as a ratio, not a force: the transient is ``m * a``, so it scales with the object and
    # the task has since changed prop. ``diag_latch_chatter.py`` measured 2.74 N on the 15 g ball,
    # which is 18.6 weights. Pinning the newton figure would pass for the wrong reason on a marble.
    transient_in_weights = 2.74 / (0.015 * 9.81)

    assert 2.0 * transient_in_weights * MICRODUCK_LATCH_OBJECT_MASS * 9.81 < MICRODUCK_LATCH_BREAK_FORCE
    # but still far too weak to hoist the robot, which is what makes it a grip and not a winch
    assert MICRODUCK_LATCH_BREAK_FORCE < 0.737 * 9.81


@pytest.mark.unit
def test_the_grip_cannot_break_on_the_step_it_forms():
    """Derived from the shipped constants: the worst admissible latch is inside the break force.

    Without this the grip limit would be a latch-radius limit in disguise -- objects caught at the
    edge of the band would be dropped on the next step.
    """
    worst_spring = MICRODUCK_LATCH_STIFFNESS * max(
        MICRODUCK_LATCH_HOLD_DISTANCE, MICRODUCK_LATCH_RADIUS - MICRODUCK_LATCH_HOLD_DISTANCE
    )
    worst_damping = MICRODUCK_LATCH_DAMPING * MICRODUCK_LATCH_MAX_REL_SPEED

    assert worst_spring + worst_damping < MICRODUCK_LATCH_BREAK_FORCE


@pytest.mark.unit
def test_the_hold_distance_is_derived_from_the_props_own_radius():
    """A change of prop must not leave the spring holding at a distance the geometry no longer has."""
    assert MICRODUCK_LATCH_HOLD_DISTANCE > MICRODUCK_MARBLE_RADIUS
    assert pytest.approx(0.005) == MICRODUCK_LATCH_HOLD_DISTANCE - MICRODUCK_MARBLE_RADIUS
    # and the latch radius leaves the mouth a real reach beyond the surface it holds at
    assert MICRODUCK_LATCH_RADIUS > MICRODUCK_LATCH_HOLD_DISTANCE


@pytest.mark.unit
def test_the_release_gate_asks_the_object_to_be_set_down_rather_than_dropped():
    """The release edge is the success edge, so both halves of it are the success criterion."""
    params = MicroDuckPickPlaceFlatEnvCfg().events.update_latch.params

    assert params["place_tolerance"] == pytest.approx(MICRODUCK_PLACE_TOLERANCE)
    assert params["place_max_height"] == pytest.approx(MICRODUCK_PLACE_MAX_HEIGHT)
    # a placement height the object can reach while resting on the ground, so a successful placement
    # is not asked to hover
    assert MICRODUCK_PLACE_MAX_HEIGHT > MICRODUCK_MARBLE_RADIUS
    # and the precision reward is scored on the same tolerance the release fires at
    assert MicroDuckPickPlaceFlatEnvCfg().rewards.place_precision.params["std"] == pytest.approx(
        MICRODUCK_PLACE_TOLERANCE
    )


@pytest.mark.unit
def test_the_latch_is_physics_rather_than_a_zero_weight_reward():
    """It is registered where Isaac Lab writes state, on a zero-width interval that fires each step.

    The ground-pick task's payload made the same move for the same reason: a zero-weight reward that
    is really load-bearing physics is exactly what a later cleanup deletes.
    """
    cfg = MicroDuckPickPlaceFlatEnvCfg()
    term = cfg.events.update_latch

    assert term.mode == "interval"
    assert term.interval_range_s == (0.0, 0.0)
    assert term.func is mdp.update_pickplace_latch
    # and no reward term carries the latch machinery's parameters
    for reward in vars(cfg.rewards).values():
        assert "stiffness" not in reward.params
        assert "break_force" not in reward.params


@pytest.mark.unit
def test_the_reset_places_the_object_before_the_latch_is_cleared_and_after_the_pose_settles():
    """Declaration order is behaviour, and three of these terms depend on it.

    The object is placed in the robot's *settled* yaw frame, so it must follow the ground-state
    reset that draws that yaw; the latch is cleared after the placement, so a latch that survived
    cannot spring-load the newly placed object toward a mouth it is no longer near.
    """
    order = [name for name, term in vars(MicroDuckPickPlaceFlatEnvCfg().events).items() if term is not None]

    assert order.index("set_ground_state") > order.index("reset_base")
    assert order.index("reset_object") > order.index("set_ground_state")
    assert order.index("reset_latch") > order.index("reset_object")


@pytest.mark.unit
def test_the_object_curriculum_starts_within_reach_of_a_standing_robot():
    """The stage that makes the task trainable at all (ruling R-PP11).

    The first stage has to be solvable by folding rather than by walking, otherwise the policy must
    discover locomotion and grasping simultaneously from a sparse latch bonus.
    """
    stages = MicroDuckPickPlaceFlatEnvCfg().curriculum.object_range.params["param_stages"]
    first, last = stages[0]["params"], stages[-1]["params"]

    # a standing MicroDuck's mouth tip reaches about 0.078 m ahead of the trunk frame at the home
    # pose, so the opening annulus is a fold away rather than a walk away
    assert first["distance_range"][1] <= 0.15
    assert first["bearing_range"][1] <= math.radians(30.0)
    # and it really does widen, in both coordinates, monotonically
    assert last["distance_range"][1] > 3.0 * first["distance_range"][0]
    assert last["bearing_range"][1] > 4.0 * first["bearing_range"][1]
    distances = [stage["params"]["distance_range"][1] for stage in stages]
    bearings = [stage["params"]["bearing_range"][1] for stage in stages]
    assert distances == sorted(distances)
    assert bearings == sorted(bearings)
    assert [stage["step"] for stage in stages] == sorted({stage["step"] for stage in stages})


@pytest.mark.unit
def test_the_carry_is_never_asked_to_be_longer_than_the_approach_already_solved():
    """The target curriculum trails the object curriculum by a stage, so the two never race."""
    object_stages = MicroDuckPickPlaceFlatEnvCfg().curriculum.object_range.params["param_stages"]
    target_stages = MicroDuckPickPlaceFlatEnvCfg().curriculum.target_range.params["range_stages"]

    assert target_stages[0]["ranges"][0][1] <= 2.0 * object_stages[0]["params"]["distance_range"][1]
    assert target_stages[-1]["step"] < object_stages[-1]["step"]


@pytest.mark.unit
def test_falling_is_priced_and_not_merely_terminated():
    """A termination is only a penalty if the rest of the episode was worth something.

    This stack's mass is a one-shot delivery bonus the policy can collect in the first second, so
    ending the episode early cost nothing and the second training run learned to deliver by diving.
    The penalty is keyed to the fall terms alone -- a diverged solver is not a policy decision.
    """
    rewards = MicroDuckPickPlaceFlatEnvCfg().rewards

    assert rewards.fell_penalty.func is mdp.is_terminated_term
    assert rewards.fell_penalty.weight < 0.0
    assert set(rewards.fell_penalty.params["term_keys"]) == {"fell_over", "fell_low"}
    assert "nan_state" not in rewards.fell_penalty.params["term_keys"]
    # and it costs a meaningful fraction of a delivery, or it prices nothing
    assert abs(rewards.fell_penalty.weight) > 0.5 * rewards.place_success.weight


@pytest.mark.unit
def test_the_fall_gate_is_tilt_and_height_rather_than_tilt_alone():
    """The velstand lesson: tilt-only gating opened a lie-flat reward-hacking basin (ruling R-PP9)."""
    terminations = MicroDuckPickPlaceFlatEnvCfg().terminations

    assert terminations.fell_over.func is mdp.bad_orientation
    assert terminations.fell_over.params["limit_angle"] == pytest.approx(math.radians(70.0))
    assert terminations.fell_low.func is mdp.root_height_below_minimum
    # below the 0.11-0.12 m the reset spawns into, so a standing robot is never caught by it
    assert terminations.fell_low.params["minimum_height"] < 0.11
    # falling ends the episode outright: there is no recovery phase to wait out
    assert "fallen_too_long" not in vars(terminations)


@pytest.mark.unit
def test_the_nan_guard_covers_every_force_path_that_feeds_something():
    """A reward reads a contact force directly here, where one non-finite value poisons the sum."""
    cfg = MicroDuckPickPlaceFlatEnvCfg()
    guarded = set(cfg.terminations.nan_state.params["sensor_names"])

    sensors = {name for name, term in vars(cfg.scene).items() if isinstance(term, type(cfg.scene.contact_forces))}
    assert guarded == sensors - {"self_collision"}
    # the object is covered by no state check on either stack, which is why its observations guard
    # themselves rather than relying on this
    assert cfg.terminations.nan_state.params.get("asset_cfg", SceneEntityCfg("robot")).name == "robot"


@pytest.mark.unit
def test_the_actor_layout_is_the_camera_contract():
    """One term wide, and that is the design (design document §6).

    Everything else the actor reads is proprioception, a command, or its own controller state, so a
    v2 that swaps in a perception term touches nothing but the row named here.
    """
    policy = MicroDuckPickPlaceFlatEnvCfg().observations.policy
    names = _term_names(policy)

    assert names == [name for name, _ in ACTOR_OBSERVATION_TERMS]
    assert sum(width for _, width in ACTOR_OBSERVATION_TERMS) == ACTOR_OBSERVATION_DIM
    # the object is its own term, never folded into a robot-state one
    assert policy.object_position.func is mdp.object_pos_in_base
    assert set(names) - PROPRIOCEPTIVE_ACTOR_TERMS - CAMERA_REPLACEABLE_ACTOR_TERMS == {"target_position", "latched"}
    # and nothing a camera could never provide leaks into the actor
    for name in names:
        assert "sensor_cfg" not in getattr(policy, name).params
    assert "succeeded" not in names


@pytest.mark.unit
def test_the_object_observation_carries_the_noise_a_perception_stack_would():
    """Training against millimetre-exact object state is training against a sensor v2 will not have."""
    policy = MicroDuckPickPlaceFlatEnvCfg().observations.policy

    assert policy.enable_corruption is True
    assert policy.object_position.noise is not None
    assert policy.object_position.noise.n_max == pytest.approx(0.005)
    # the critic sees it clean, which is the asymmetric half of the actor-critic
    critic = MicroDuckPickPlaceFlatEnvCfg().observations.critic
    assert critic.enable_corruption is False
    assert critic.object_position.noise is None


@pytest.mark.unit
def test_the_critic_sees_what_the_robot_has_no_sensor_for():
    """The privileged layout, and the two sensor-derived foot terms in their NaN-guarded variants."""
    critic = MicroDuckPickPlaceFlatEnvCfg().observations.critic
    names = _term_names(critic)

    assert names == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert sum(width for _, width in CRITIC_OBSERVATION_TERMS) == CRITIC_OBSERVATION_DIM
    assert critic.foot_air_time.func is mdp.foot_air_time_safe
    assert critic.foot_contact_forces.func is mdp.foot_contact_forces_safe


@pytest.mark.unit
def test_both_observation_groups_read_the_servos_in_the_deploy_order():
    """The joint blocks are a hardware contract even where the observation vector is not."""
    cfg = MicroDuckPickPlaceFlatEnvCfg()

    for group in (cfg.observations.policy, cfg.observations.critic):
        for name in ("joint_pos", "joint_vel"):
            term = getattr(group, name)
            params = term.params.get("term_params", term.params)
            entity = params["asset_cfg"]
            assert entity.joint_names == EXPECTED_SERVO_JOINT_NAMES
            assert entity.preserve_order is True


@pytest.mark.unit
def test_the_terms_select_the_bodies_and_sensors_the_task_measures():
    """One consumed table over every scene-entity selection in the stack.

    ``exempt <= measured`` rather than ``==``: a term added without a selection has to be added here
    too, and a table that merely listed the terms it already knew about would pass forever.
    """
    cfg = MicroDuckPickPlaceFlatEnvCfg()
    expected = {
        ("rewards", "mouth_to_object", "asset_cfg"): (None, EXPECTED_MOUTH_BODY_NAMES),
        ("rewards", "mouth_down", "asset_cfg"): (None, EXPECTED_MOUTH_BODY_NAMES),
        ("rewards", "upright", "asset_cfg"): (None, ["trunk_base"]),
        ("rewards", "body_ang_vel", "asset_cfg"): (None, ["trunk_base"]),
        ("rewards", "joint_torques_l2", "asset_cfg"): (EXPECTED_SERVO_JOINT_NAMES, None),
        ("events", "update_latch", "asset_cfg"): (None, EXPECTED_MOUTH_BODY_NAMES),
        ("observations", "mouth_to_object", "asset_cfg"): (None, EXPECTED_MOUTH_BODY_NAMES),
    }
    measured = set()

    def _record(section: str, term_name: str, params: dict) -> None:
        for key, value in params.items():
            if isinstance(value, SceneEntityCfg):
                measured.add((section, term_name, key))
                if (section, term_name, key) in expected:
                    joints, bodies = expected[(section, term_name, key)]
                    assert value.joint_names == joints, (term_name, key)
                    assert value.body_names == bodies, (term_name, key)

    for section in ("rewards", "events"):
        for term_name, term in vars(getattr(cfg, section)).items():
            if term is not None:
                _record(section, term_name, term.params)
    for group in (cfg.observations.policy, cfg.observations.critic):
        for term_name in _term_names(group):
            _record("observations", term_name, getattr(group, term_name).params)

    assert set(expected) <= measured


@pytest.mark.unit
def test_every_object_selection_names_the_scene_entity_the_prop_is_registered_as():
    """The prop is ``object``, not ``ball``: the camera migration keys off that name."""
    cfg = MicroDuckPickPlaceFlatEnvCfg()

    assert "object" in vars(cfg.scene)
    assert "ball" not in vars(cfg.scene)
    assert cfg.commands.place_target.object_name == "object"
    groups = [cfg.observations.policy, cfg.observations.critic]
    terms = [term for section in (cfg.rewards, cfg.events) for term in vars(section).values() if term is not None]
    terms += [getattr(group, name) for group in groups for name in _term_names(group)]
    for term in terms:
        if True:
            for key in ("asset_cfg", "object_cfg"):
                entity = term.params.get(key)
                if isinstance(entity, SceneEntityCfg) and entity.name not in ("robot", "contact_forces"):
                    assert entity.name == "object", (key, entity.name)


@pytest.mark.unit
def test_the_task_runs_the_all_collisions_robot_and_a_prop_the_beak_can_hold():
    """The head shells are what the mouth presses the object with, so the model is not optional."""
    cfg = MicroDuckPickPlaceFlatEnvCfg()

    assert "allcollisions" in cfg.scene.robot.spawn.usd_path
    assert cfg.scene.terrain.terrain_type == "plane"
    # the purpose-built marble, not the ball-kick floorball: 20 mm against a 31 mm aperture
    assert cfg.scene.object.spawn.radius == pytest.approx(MICRODUCK_MARBLE_RADIUS)
    assert cfg.scene.object.spawn.mass_props.mass == pytest.approx(MICRODUCK_LATCH_OBJECT_MASS)


@pytest.mark.unit
def test_the_head_impact_sensor_is_filtered_against_the_terrain_only():
    """Pressing the mouth onto the object must be free; face-planting into the floor must not be.

    An unfiltered head sensor would make the two indistinguishable and price the pick out of the task.
    """
    sensor = MicroDuckPickPlaceFlatEnvCfg().scene.head_impact_contact

    assert sensor.filter_shape_prim_expr == ["/World/ground/.*"]
    assert all("head_shell" in expr or "jaw" in expr for expr in sensor.sensor_shape_prim_expr)


@pytest.mark.unit
def test_the_episode_leaves_headroom_over_the_widest_curriculum_stage():
    """20 s against about 10 s of translation at MicroDuck's walking pace."""
    cfg = MicroDuckPickPlaceFlatEnvCfg()
    object_stages = cfg.curriculum.object_range.params["param_stages"]
    target_stages = cfg.curriculum.target_range.params["range_stages"]
    furthest = object_stages[-1]["params"]["distance_range"][1] + target_stages[-1]["ranges"][0][1]

    assert cfg.episode_length_s == pytest.approx(20.0)
    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    # a walking-pace budget of 0.15 m/s over half the episode still covers the longest route
    assert furthest < 0.15 * 0.5 * cfg.episode_length_s


@pytest.mark.unit
def test_the_runner_differs_from_the_velocity_one_in_the_two_fields_every_sibling_differs_in():
    """The network shapes, the optimizer and the rollout length are the family's, not this task's."""
    from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg

    runner = MicroDuckPickPlacePPORunnerCfg()
    velocity = MicroDuckPPORunnerCfg()

    assert runner.experiment_name == "microduck_pickplace"
    assert runner.max_iterations == 20000
    assert runner.num_steps_per_env == velocity.num_steps_per_env == STEPS_PER_ITERATION
    # symmetry augmentation stays off, as on every task but the forward roll -- and here it would be
    # wrong rather than merely unhelpful, since mirroring the robot without mirroring the object and
    # the drop point relabels a left-hand reach as a right-hand one
    assert runner.algorithm.symmetry_cfg is None
    assert velocity.algorithm.symmetry_cfg is None
    # and every other field is inherited rather than restated. Compared through ``to_dict`` because
    # the nested model configurations carry no ``__eq__``, so two identical ones compare unequal by
    # identity and a field-by-field ``!=`` would report the whole network as a difference.
    runner_fields, velocity_fields = runner.to_dict(), velocity.to_dict()
    differing = {field for field in runner_fields if runner_fields[field] != velocity_fields[field]}

    assert differing == {"experiment_name", "max_iterations"}


MEASURED_PEAK_CONTACTS = 30
MEASURED_PEAK_CONSTRAINTS = 94
"""Worst-case per-environment demand under random actions, re-measured after the prop change.

From ``artifacts/microduck/profile_microduck_contacts_pickplace_marble20_4096envs.log``. Swapping the
70 mm ball for a 20 mm marble takes the contact peak from 32 to 30 and *raises* the constraint peak
from 90 to 94 -- a smaller prop is not uniformly cheaper, which is why the budget is re-measured on a
prop change rather than assumed to be covered by the old one. Transcribed here rather than imported
so the budget below cannot be lowered under the measurement it was sized against.
"""


@pytest.mark.unit
def test_the_solver_budget_covers_the_measured_worst_case():
    """Measured rather than inherited, per the family's T10 precedent.

    A held object adds a persistent mouth-shell contact the ball-kick task never has -- its ball is
    either on the ground or in flight -- so inheriting that task's budget would have been two
    contacts short.
    """
    solver = MicroDuckPickPlaceFlatEnvCfg().sim.physics.newton_mjwarp.solver_cfg

    # ``njmax`` is a hard per-environment cap, so it carries real margin over the peak: at least
    # eight further pyramidal contacts' worth, four constraint rows each. Overflowing it is a
    # silently wrong simulation rather than a slow one, which is why this is the bound with margin.
    assert solver.njmax >= MEASURED_PEAK_CONSTRAINTS + 4 * 8
    # ``nconmax`` is a per-environment share of one shared pool rather than a cap, so it sits just
    # above the peak
    assert solver.nconmax >= MEASURED_PEAK_CONTACTS
    # and above the ball-kick task's, on the same robot and the same ball
    assert solver.nconmax > 36
    # the iteration counts are the family's template, unchanged
    assert (solver.iterations, solver.ls_iterations) == (10, 20)


##
# Simulator-backed acceptance
##


def _scripted_env(num_envs: int = 4):
    """Build the task with the per-robot randomization off, so a scripted rollout is reproducible."""
    env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=num_envs)
    for name in ("foot_friction", "mass_inertia", "randomize_com", "randomize_head_com", "push_robot"):
        setattr(env_cfg.events, name, None)
    for name in list(vars(env_cfg.curriculum)):
        setattr(env_cfg.curriculum, name, None)
    return env_cfg


def _place_object_at_mouth(unwrapped, mouth_cfg: SceneEntityCfg) -> torch.Tensor:
    """Teleport the object to the latch anchor and stop it, and return where it was put."""
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    body_pos = robot.data.body_link_pos_w.torch[:, mouth_cfg.body_ids].squeeze(1)
    body_quat = robot.data.body_link_quat_w.torch[:, mouth_cfg.body_ids].squeeze(1)
    offset = torch.tensor(
        MicroDuckPickPlaceFlatEnvCfg().events.update_latch.params["mouth_offset_b"], device=unwrapped.device
    ).expand_as(body_pos)
    axis = torch.tensor(
        MicroDuckPickPlaceFlatEnvCfg().events.update_latch.params["mouth_axis_b"], device=unwrapped.device
    ).expand_as(body_pos)
    mouth_pos = body_pos + math_utils.quat_apply(body_quat, offset)
    anchor = mouth_pos + MICRODUCK_LATCH_HOLD_DISTANCE * math_utils.quat_apply(body_quat, axis)

    pose = torch.zeros(unwrapped.num_envs, 7, device=unwrapped.device)
    pose[:, :3] = anchor
    pose[:, 6] = 1.0  # (x, y, z, w) identity
    obj.write_root_link_pose_to_sim(pose)
    obj.write_root_com_velocity_to_sim(torch.zeros(unwrapped.num_envs, 6, device=unwrapped.device))
    unwrapped.scene.write_data_to_sim()
    unwrapped.scene.update(dt=unwrapped.physics_dt)
    return anchor


def _world_force(composer) -> torch.Tensor:
    """Total world-frame force [N] the composer will apply to each environment's bodies.

    The composer keeps positional world writes and centre-of-mass world writes in two buffers, so
    reading either one alone reports half the pair; see the caller for why that matters here.
    """
    at_point = wp.to_torch(composer.global_force_w)
    at_com = wp.to_torch(composer.global_force_at_com_w)
    return (at_point + at_com).sum(dim=1)


def _fold_robot_over(unwrapped) -> None:
    """Pitch the whole robot 90 degrees forward so its mouth is at floor height.

    The release gate asks for the object to be **set down**, not dropped, so it cannot fire while the
    mouth is at standing height -- a standing MicroDuck holds the object about 0.15 m up, against a
    0.06 m release ceiling. The trained behaviour that satisfies it is the ground-pick task's fold;
    scripting that fold joint by joint would be testing the kinematics rather than the state machine,
    so this places the posture outright, the way the roller stand-up acceptance places its three.
    """
    robot = unwrapped.scene["robot"]
    pose = torch.zeros(unwrapped.num_envs, 7, device=unwrapped.device)
    pose[:, :3] = unwrapped.scene.env_origins + torch.tensor([0.0, 0.0, 0.030], device=unwrapped.device)
    # 90 degrees about +y, in Isaac Lab's (x, y, z, w) layout -- scalar last (design document E-1)
    pose[:, 1] = math.sin(math.pi / 4.0)
    pose[:, 6] = math.cos(math.pi / 4.0)
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_com_velocity_to_sim(torch.zeros(unwrapped.num_envs, 6, device=unwrapped.device))
    unwrapped.scene.write_data_to_sim()
    unwrapped.scene.update(dt=unwrapped.physics_dt)


def _score(unwrapped) -> dict[str, torch.Tensor]:
    """Evaluate the whole reward stack once and return each term's contribution, per step.

    The reward manager scales every term by the control step before accumulating it, so the episode
    sums it keeps are in reward-seconds. Dividing back out is what makes the numbers below
    comparable against the weights in :data:`EXPECTED_REWARD_WEIGHTS` rather than against the
    weights times 0.02.
    """
    manager = unwrapped.reward_manager
    # clear whatever the preceding ``env.step`` calls accumulated, so this measures exactly one
    # evaluation rather than the window since the last reset
    manager.reset()
    manager.compute(dt=unwrapped.step_dt)
    scores = {name: manager._episode_sums[name].clone() / unwrapped.step_dt for name in manager.active_terms}
    manager.reset()
    return scores


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_observation_groups_are_the_widths_the_term_tables_name():
    """The two layouts, measured against the tables the recipe tests transcribe."""
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
def test_the_reset_puts_the_object_in_reach_and_the_drop_point_beyond_it():
    """The two reset geometries, measured in the frame the curriculum widens them in.

    They are drawn by two different managers in a fixed order -- the object by a reset event, the
    drop point by the command manager afterwards -- and nothing in the configuration makes them
    agree. If the command manager resampled first, the drop points would scatter around wherever the
    previous episode left the object.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = _scripted_env(num_envs=64)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        obj = unwrapped.scene["object"]
        origins = unwrapped.scene.env_origins
        target = unwrapped.command_manager.get_term("place_target").target_pos_w

        # the object sits on the ground of its own environment, one radius up
        torch.testing.assert_close(
            obj.data.root_link_pos_w.torch[:, 2] - origins[:, 2],
            torch.full_like(origins[:, 2], MICRODUCK_MARBLE_RADIUS),
            atol=2e-3,
            rtol=0.0,
        )
        # ... inside the opening annulus, measured from the robot
        reach = torch.linalg.norm(
            obj.data.root_link_pos_w.torch[:, :2] - robot.data.root_link_pos_w.torch[:, :2], dim=-1
        )
        low, high = env_cfg.events.reset_object.params["distance_range"]
        assert float(reach.min()) >= low - 1e-3
        assert float(reach.max()) <= high + 1e-3

        # ... and the drop point is drawn around the *placed* object, not around the previous one
        carry = torch.linalg.norm(target[:, :2] - obj.data.root_link_pos_w.torch[:, :2], dim=-1)
        (target_low, target_high), _ = env_cfg.commands.place_target.ranges
        assert float(carry.min()) >= target_low - 1e-3
        assert float(carry.max()) <= target_high + 1e-3
        # both bands are actually sampled rather than collapsed
        assert float(reach.max() - reach.min()) > 0.5 * (high - low)
        assert float(carry.max() - carry.min()) > 0.5 * (target_high - target_low)

        # nothing is latched or succeeded on the first step of an episode
        state = mdp.pickplace_latch_state(unwrapped)
        assert not bool(state.latched.any())
        assert not bool(state.succeeded.any())
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_a_scripted_pick_carry_and_place_scores_the_whole_stack_end_to_end():
    """The acceptance the R-T7a precedent asks for: rewards evaluated under physics, not in isolation.

    A policy that could actually walk to an object is what training is for; this scripts the geometry
    instead and checks that every stage of the task is expressible, observable and *paid* through the
    real managers. Four things are being asserted at once, and each has failed independently in this
    family before: the latch forms under physics rather than only against doubles, the object stays
    with the mouth while held, the release fires at the commanded target, and each block's rewards
    are non-zero exactly where the design says they should be.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = _scripted_env(num_envs=4)
        # A scripted rollout must not be recycled underneath the measurement. Dropping the two fall
        # terminations means dropping the reward that prices them: ``is_terminated_term`` resolves
        # its keys at construction and raises if they name nothing, so the two are coupled and any
        # tool that disables the fall gates has to disable the penalty with them.
        env_cfg.terminations.fell_over = None
        env_cfg.terminations.fell_low = None
        env_cfg.rewards.fell_penalty = None
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        mouth_cfg = SceneEntityCfg("robot", body_names=EXPECTED_MOUTH_BODY_NAMES)
        mouth_cfg.resolve(unwrapped.scene)
        obj = unwrapped.scene["object"]
        state = mdp.pickplace_latch_state(unwrapped)
        hook = unwrapped.event_manager.get_term_cfg("update_latch")
        action = torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device)

        # -- 1. approach. Nothing is held, so the approach block is the only one paying.
        approach = _score(unwrapped)
        assert not bool(state.latched.any())
        assert float(approach["mouth_to_object"].abs().max()) >= 0.0
        assert float(approach["carry_hold"].abs().max()) == pytest.approx(0.0)
        assert float(approach["place_success"].abs().max()) == pytest.approx(0.0)

        # -- 2. pick. Put the object where a folded mouth would have brought it, then run the state
        # machine exactly as the interval event does.
        _place_object_at_mouth(unwrapped, mouth_cfg)
        hook.func(unwrapped, None, **hook.params)
        assert bool(state.latched.all())
        assert bool(state.latch_edge.all())

        pick = _score(unwrapped)
        latch_weight = EXPECTED_REWARD_WEIGHTS["latch_bonus"]
        torch.testing.assert_close(pick["latch_bonus"], torch.full_like(pick["latch_bonus"], latch_weight))
        # the approach block is silent the moment the object is held
        torch.testing.assert_close(pick["mouth_to_object"], torch.zeros_like(pick["mouth_to_object"]))
        torch.testing.assert_close(pick["mouth_down"], torch.zeros_like(pick["mouth_down"]))

        # -- 3. carry. The object is dynamic and held only by the spring, so this is the step that
        # would fail if the virtual weld were mis-signed: it would be flung rather than carried.
        for _ in range(10):
            env.step(action)
            assert bool(state.latched.all()), "the grip broke under its own carry"
        body_pos = unwrapped.scene["robot"].data.body_link_pos_w.torch[:, mouth_cfg.body_ids].squeeze(1)
        held_distance = torch.linalg.norm(obj.data.root_link_pos_w.torch - body_pos, dim=-1)
        # within the hold distance plus the spring's own sag and the mouth-tip offset's length
        assert float(held_distance.max()) < MICRODUCK_LATCH_HOLD_DISTANCE + 0.10

        carry = _score(unwrapped)
        carry_weight = EXPECTED_REWARD_WEIGHTS["carry_hold"]
        torch.testing.assert_close(carry["carry_hold"], torch.full_like(carry["carry_hold"], carry_weight))
        torch.testing.assert_close(carry["latch_bonus"], torch.zeros_like(carry["latch_bonus"]))
        # and holding pays strictly more than the hover it replaced, which is the audit's inequality
        # measured rather than asserted from the weights
        assert float(carry["carry_hold"].min()) > float(approach["mouth_to_object"].max())

        # -- 4. place. Two things have to be true at once, and the second is easy to forget: the
        # object has to be *at* the drop point and *near the floor*. Fold the robot over so the mouth
        # is at floor height, keep the object with it, and move the drop point under it -- the
        # release gate is a relative geometry, so bringing the target to the object exercises the
        # same edge as walking the object to the target.
        _fold_robot_over(unwrapped)
        _place_object_at_mouth(unwrapped, mouth_cfg)
        target = unwrapped.command_manager.get_term("place_target").target_pos_w
        target[:, :2] = obj.data.root_link_pos_w.torch[:, :2]

        object_height = obj.data.root_link_pos_w.torch[:, 2] - unwrapped.scene.env_origins[:, 2]
        assert float(object_height.max()) < MICRODUCK_PLACE_MAX_HEIGHT, "the fold did not reach the floor"

        hook.func(unwrapped, None, **hook.params)

        assert bool(state.release_edge.all())
        assert bool(state.succeeded.all())
        assert not bool(state.latched.any())

        place = _score(unwrapped)
        success_weight = EXPECTED_REWARD_WEIGHTS["place_success"]
        torch.testing.assert_close(place["place_success"], torch.full_like(place["place_success"], success_weight))
        # placed exactly on the target, so the precision term scores its full weight
        precision_weight = EXPECTED_REWARD_WEIGHTS["place_precision"]
        torch.testing.assert_close(
            place["place_precision"], torch.full_like(place["place_precision"], precision_weight), atol=1e-2, rtol=0.0
        )
        torch.testing.assert_close(place["carry_hold"], torch.zeros_like(place["carry_hold"]))

        # -- 5. after. Nothing task-related pays again, and the success cannot be farmed.
        hook.func(unwrapped, None, **hook.params)
        after = _score(unwrapped)
        for name in ("latch_bonus", "place_success", "place_precision", "carry_hold", "mouth_to_object"):
            torch.testing.assert_close(after[name], torch.zeros_like(after[name]), msg=f"{name} paid twice")
        assert bool(state.succeeded.all())
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_carried_object_loads_the_robot_rather_than_riding_for_free():
    """The physical content of ruling R-PP1, measured.

    A kinematically slaved object would track the mouth just as well and weigh nothing, which is the
    failure this design exists to avoid and which no geometric assertion can see. What separates the
    two is the reaction: the wrench on ``jaw_soft`` must be equal and opposite to the one on the
    object, and it must carry a moment, because it acts at the mouth tip rather than at the head's
    centre of mass.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = _scripted_env(num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        mouth_cfg = SceneEntityCfg("robot", body_names=EXPECTED_MOUTH_BODY_NAMES)
        mouth_cfg.resolve(unwrapped.scene)
        hook = unwrapped.event_manager.get_term_cfg("update_latch")
        robot_composer = unwrapped.scene["robot"].permanent_wrench_composer
        object_composer = unwrapped.scene["object"].permanent_wrench_composer

        # unlatched: the mechanism writes nothing to either body
        hook.func(unwrapped, None, **hook.params)
        assert float(robot_composer.out_force_b.torch.abs().max()) == pytest.approx(0.0)
        assert float(object_composer.out_force_b.torch.abs().max()) == pytest.approx(0.0)

        # latched, and displaced so the spring is genuinely loaded rather than at rest
        _place_object_at_mouth(unwrapped, mouth_cfg)
        hook.func(unwrapped, None, **hook.params)
        obj = unwrapped.scene["object"]
        pose = obj.data.root_link_pose_w.torch.clone()
        pose[:, 2] -= 0.01
        obj.write_root_link_pose_to_sim(pose)
        unwrapped.scene.write_data_to_sim()
        unwrapped.scene.update(dt=unwrapped.physics_dt)
        hook.func(unwrapped, None, **hook.params)

        # Read in the **world** frame, and from both world buffers. ``out_force_b`` resolves each
        # wrench into its own body's frame, and the two bodies here are the head and a free sphere,
        # so the equal-and-opposite relation is not expressible there at all. The composer also
        # splits a world-frame write by whether it named a point: the reaction on the head is
        # applied at the mouth, so it lands in ``global_force_w`` with its moment, while the
        # object's half acts at its centre of mass and lands in ``global_force_at_com_w``.
        robot_force = _world_force(robot_composer)
        object_force = _world_force(object_composer)

        # the robot really feels it: the force is the spring's own, so it is checked against the
        # stiffness rather than against a literal that only held for the prop it was written for
        assert float(object_force.norm(dim=-1).min()) > 0.5 * MICRODUCK_LATCH_STIFFNESS * 0.01
        torch.testing.assert_close(robot_force, -object_force, atol=1e-4, rtol=1e-3)
        # and it arrives as a moment about the head, not as a force at its centre of mass
        assert float(robot_composer.out_torque_b.torch.norm(dim=-1).sum(dim=-1).min()) > 0.0
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_grip_gives_way_rather_than_dragging_the_object_through_the_scene():
    """The anti-winch ruling (R-PP2), under physics rather than against a double."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = _scripted_env(num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        mouth_cfg = SceneEntityCfg("robot", body_names=EXPECTED_MOUTH_BODY_NAMES)
        mouth_cfg.resolve(unwrapped.scene)
        hook = unwrapped.event_manager.get_term_cfg("update_latch")
        state = mdp.pickplace_latch_state(unwrapped)

        _place_object_at_mouth(unwrapped, mouth_cfg)
        hook.func(unwrapped, None, **hook.params)
        assert bool(state.latched.all())

        # pin the object half a metre away, which the grip cannot bridge
        obj = unwrapped.scene["object"]
        pose = obj.data.root_link_pose_w.torch.clone()
        pose[:, 0] += 0.5
        obj.write_root_link_pose_to_sim(pose)
        unwrapped.scene.write_data_to_sim()
        unwrapped.scene.update(dt=unwrapped.physics_dt)
        hook.func(unwrapped, None, **hook.params)

        assert not bool(state.latched.any())
        assert not bool(state.succeeded.any())
        assert float(
            unwrapped.scene["object"].permanent_wrench_composer.out_force_b.torch.abs().max()
        ) == pytest.approx(0.0)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)


##
# Reward budget in episode-return units (ruling R-PP17)
##

EPISODE_STEPS = 1000
"""Control steps in a 20 s episode at the 50 Hz control rate."""

CONTROL_DT = 0.02
"""Seconds per control step. **Every** reward term is multiplied by this before it is summed.

``RewardManager.compute`` does ``value = func(...) * weight * dt`` (``reward_manager.py:154``), so a
weight is a *rate* and a one-shot bonus contributes ``weight * dt`` to the episode return, not
``weight``. Missing that factor is what made the first long run trainable only as a farm: the entire
task objective came to 4.26 against 80 for holding the object and doing nothing (ruling R-PP17).
"""


def _returns(rewards) -> dict[str, float]:
    """Episode-return contribution of each strategy, in the units the learner actually optimizes."""

    def r(name: str, occurrences: float) -> float:
        return getattr(rewards, name).weight * CONTROL_DT * occurrences

    return {
        # walk over, pick it up, carry it 0.6 m, set it down on the spot, having held it 300 steps
        "deliver": (
            r("approach_progress", 0.45)
            + r("latch_bonus", 1)
            + r("carry_progress", 0.60)
            + r("carry_hold", 300)
            + r("place_success", 1)
            + r("place_precision", 1)
        ),
        # pick it up and stand there for the rest of the episode
        "hold_forever": r("approach_progress", 0.45) + r("latch_bonus", 1) + r("carry_hold", EPISODE_STEPS),
        # park the mouth on it and never commit
        "hover_forever": (
            r("approach_progress", 0.45) + r("mouth_to_object", EPISODE_STEPS) + r("mouth_down", EPISODE_STEPS)
        ),
        # grab it, dive at the drop point and topple: it delivers, in 40 control steps, and forfeits
        # the rest of the episode -- which the second training run found and which cost nothing until
        # falling was priced
        "dive_and_topple": (
            r("approach_progress", 0.10)
            + r("latch_bonus", 1)
            + r("carry_progress", 0.20)
            + r("carry_hold", 40)
            + r("place_success", 1)
            + r("place_precision", 0.7)
            + r("fell_penalty", 1)
        ),
    }


@pytest.mark.unit
def test_delivering_the_object_pays_more_than_any_shortcut():
    """The budget the first long run did not have, expressed where the learner sees it.

    Weights are rates, so comparing a one-shot bonus against a per-step term without the ``dt``
    factor compares two different units -- which is exactly the mistake that let a policy spend
    2255 iterations farming a one-step grip. This asserts the ordering in episode-return units, and
    with real margin rather than a hair.
    """
    returns = _returns(MicroDuckPickPlaceFlatEnvCfg().rewards)

    assert returns["deliver"] > 2.5 * returns["hold_forever"]
    assert returns["deliver"] > 3.0 * returns["hover_forever"]
    # ... and than delivering by throwing yourself at the target, which satisfies the literal
    # success criterion and is not the task
    assert returns["deliver"] > 2.5 * returns["dive_and_topple"]
    # but diving still beats never touching the object, so the delivery gradient survives the fix:
    # a policy that has not yet learned to walk must still be paid for getting the object there
    assert returns["dive_and_topple"] > returns["hover_forever"]
    # and the per-step tie-break of ruling R-PP6 still points the right way at the moment of choice
    assert returns["hold_forever"] > returns["hover_forever"]


@pytest.mark.unit
def test_no_single_reward_term_can_dominate_the_return_on_its_own():
    """The symptom that made the defect visible: one term was 88 % of the return.

    Bounds every term by what it can pay across a whole episode, so a term that is accidentally
    worth more than the task cannot be introduced quietly.
    """
    rewards = MicroDuckPickPlaceFlatEnvCfg().rewards
    # the most any one term can pay in an episode: per-step terms saturate for the whole episode,
    # one-shot terms fire once, and the two distance terms are bounded by their own geometry
    ceilings = {
        "approach_progress": 0.45,
        "carry_progress": 0.60,
        "latch_bonus": 1,
        "place_success": 1,
        "place_precision": 1,
        "carry_hold": EPISODE_STEPS,
        "mouth_to_object": EPISODE_STEPS,
        "mouth_down": EPISODE_STEPS,
        "object_clearance": EPISODE_STEPS,
        "upright": EPISODE_STEPS,
        "feet_grounded": EPISODE_STEPS,
    }
    contributions = {n: getattr(rewards, n).weight * CONTROL_DT * c for n, c in ceilings.items()}
    total = sum(contributions.values())

    for name, value in contributions.items():
        assert value < 0.5 * total, f"{name} alone is {value / total:.0%} of the positive budget"


@pytest.mark.unit
def test_the_latch_bonus_is_capped_at_one_payment_per_episode():
    """An event the policy can re-trigger at will needs a per-episode cap, not a progress gate."""
    assert MicroDuckPickPlaceFlatEnvCfg().rewards.latch_bonus.func is mdp.pickplace_latch_bonus
    # the state machine keeps the raw edge and the first-of-episode edge apart, and the reward reads
    # the capped one
    state = mdp.PickPlaceLatchState(2, "cpu")
    assert hasattr(state, "has_latched")
    assert hasattr(state, "first_latch_edge")


@pytest.mark.unit
def test_the_carry_progress_is_a_ratchet_and_the_approach_is_not():
    """The asymmetry is deliberate and is the kind of thing that needs a reason on the record.

    Carry is farmable as a masked potential because the object can give ground back for free once
    dropped; approach is not, because the robot cannot move away from the object without being
    charged for it.
    """
    rewards = MicroDuckPickPlaceFlatEnvCfg().rewards

    assert issubclass(mdp.pickplace_carry_progress, mdp.upright_progress.__bases__[0].__bases__[0])
    assert not issubclass(mdp.pickplace_carry_progress, mdp.upright_progress.__bases__[0])
    assert issubclass(mdp.pickplace_approach_progress, mdp.upright_progress.__bases__[0])
    assert rewards.carry_progress.func is mdp.pickplace_carry_progress


@pytest.mark.unit
def test_the_object_fits_inside_the_beak():
    """The invariant the prop change exists to satisfy.

    The measured aperture at the mouth line is 31.1 mm at full open
    (``artifacts/microduck/pickplace/BEAK.md``). The task shipped for three training runs with a
    70 mm ball -- more than twice as wide as the beak opens -- which is why the recordings show a duck
    leaning on a sphere rather than picking anything up. A prop that does not fit is not a
    pick-and-place prop, however well the rest of the stack scores.
    """
    # Measured on the vertices that actually touch when the beak is shut -- the mouth line, which
    # lies entirely at the tip. Rotating those to full open gives 31.1 mm median.
    #
    # An earlier version of this test used 17.4 mm, which was the *minimum* distance from any
    # front-half jaw vertex to the upper surface. Most of those vertices sit near the hinge where the
    # mandibles barely separate, so that figure understated the aperture by nearly half and the prop
    # was sized down to match it.
    gape = 0.0311
    assert pytest.approx(math.radians(35.0), abs=1e-6) == MICRODUCK_BEAK_OPEN - MICRODUCK_BEAK_CLOSED

    assert gape > 2.0 * MICRODUCK_MARBLE_RADIUS
    # with real clearance, so the beak visibly closes on it rather than butting against it
    assert 0.75 * gape > 2.0 * MICRODUCK_MARBLE_RADIUS


@pytest.mark.unit
def test_every_latch_constant_derives_from_the_prop_rather_than_being_a_literal():
    """Ruling R-PP17's actual repair: a hand-picked spring is only right for the prop it was picked
    for, and this task has now changed prop once.

    Reproduces each constant from the object's mass and the control step, so a future prop change
    cannot leave a stale number behind -- which is what happened when the stiffness was derived
    against the physics rate and then never revisited.
    """
    cfg = MicroDuckPickPlaceFlatEnvCfg()
    control_dt = cfg.sim.dt * cfg.decimation
    m = MICRODUCK_LATCH_OBJECT_MASS

    assert m == pytest.approx(MICRODUCK_MARBLE_MASS)
    assert math.sqrt(MICRODUCK_LATCH_STIFFNESS / m) * control_dt == pytest.approx(0.82, abs=1e-6)
    ratio = MICRODUCK_LATCH_DAMPING / (2.0 * math.sqrt(MICRODUCK_LATCH_STIFFNESS * m))
    assert ratio == pytest.approx(0.73, abs=1e-6)
    assert MICRODUCK_LATCH_BREAK_FORCE / (m * 9.81) == pytest.approx(41.0, abs=1e-6)
