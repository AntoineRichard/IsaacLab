# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe-parity and acceptance tests for the contributed MicroDuck gear-backlash velocity twin.

The twin is an A/B experiment on the plant rather than a new task, so the load-bearing property is
**how little it differs** from the base flat task
(:class:`~isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg.MicroDuckVelocityFlatEnvCfg`): a
difference in a training curve has to be attributable to the gear play and to nothing else. The
headline test therefore enumerates every leaf of the assembled configuration that differs between
the two and compares that set against a table spelled out here, so an unintended edit fails rather
than passing unnoticed.

The parity tests need neither the asset nor the simulator. The integration tests spawn
:data:`~isaaclab_assets.MICRODUCK_BACKLASH_CFG`, whose USD is generated rather than committed, so
they skip when it is absent; generate it with
``uv run --extra importers python scripts/tools/convert_microduck.py --model walk_backlash``.

Upstream's recipe is ``make_backlash_variant`` (``tasks/backlash.py`` at the pinned checkout),
transcribed into ``artifacts/microduck/backlash_investigation.md`` section 1.4. The expected values
are spelled out here rather than imported from the configuration under test, so that a drifting
value fails rather than agrees with itself -- except for the delta table, which is *about* the
relationship between two configurations and therefore reads both.
"""

import itertools
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
from isaaclab.actuators import BamActuatorCfg, BamBacklashActuatorCfg
from isaaclab.actuators.actuator_bam_cfg import BACKLASH_JOINT_TEMPLATE
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.string import resolve_matching_names

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg
from isaaclab_tasks.contrib.microduck.velocity.backlash_env_cfg import (
    MICRODUCK_BACKLASH_NJMAX,
    MicroDuckVelocityBacklashFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_BACKLASH_USD_PATH

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_backlash_usd = pytest.mark.skipif(
    not os.path.isfile(MICRODUCK_BACKLASH_USD_PATH),
    reason=(
        f"MicroDuck backlash USD asset is missing: {MICRODUCK_BACKLASH_USD_PATH}. Generate it with"
        " 'uv run --extra importers python scripts/tools/convert_microduck.py --model walk_backlash'."
    ),
)
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-Velocity-Flat-MicroDuck-Backlash"

BASE_TASK_NAME = "IsaacContrib-Velocity-Flat-MicroDuck"
"""The task this one is a twin of, and the only thing it may be compared against."""

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

EXPECTED_PLAY_JOINT_NAMES = [BACKLASH_JOINT_TEMPLATE.format(joint=name) for name in EXPECTED_SERVO_JOINT_NAMES]
"""The 14 unactuated gear-play hinges, one per servo, built from the shipped naming contract.

Built rather than spelled out on purpose: this list is only meaningful if it is the same convention
the Newton BAM binding and the encoder observations resolve, and reproducing it by hand here would
let the two drift apart silently. That the converted robot really carries these names is asserted
against the spawned articulation, not against this list.
"""

EXPECTED_ALL_JOINT_NAMES = EXPECTED_SERVO_JOINT_NAMES + EXPECTED_PLAY_JOINT_NAMES
"""All 28 hinges of the played model. The real order interleaves them; only membership is used here."""

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
"""The deployed MicroDuck observation layout, which is shared across the whole policy family.

It is the base task's, unchanged, which is the headline design constraint of the backlash variant:
the robot gains 14 joints and the two joint blocks stay 14 wide, because they report the 14 servos'
*encoders* rather than every degree of freedom. A 28-wide block would make the actor 89 and
invalidate every trained checkpoint and the deployed runtime with it.
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
"""The privileged critic layout, the base task's unchanged. Not a deploy contract, but pinned here
because its two joint blocks are remapped as well and a width change would mean they were not."""

CRITIC_OBSERVATION_DIM = 76
"""Critic observation width, measured from the assembled group."""

ACTION_DIM = 14
"""Actions on the played model: still one per servo. The play hinges are joints nothing drives."""

NOMINAL_PLAY_RAD = math.radians(1.0)
"""Half the authored gear play [rad]: upstream's ``--backlash-deg 2.0`` is peak to peak."""

PLAY_OVERSHOOT_BOUND = 3.0
"""How far past its nominal play a loaded hinge may ride before the summand is something else."""

STRUCTURAL_CONSTRAINT_BOUND = 68
"""Constraint rows one played environment can demand at once: the base task's 54 plus 14.

The base task's peak is structural -- 4 pyramidal rows on each of 10 contacts, plus the 14 servo
limits -- and a gear-play hinge adds one permanently active limit row each, because its limits *are*
the gear teeth. Profiling measured 65 at 256 environments and 66 at 2048, i.e. the bound is not
quite reached but is approached from below at both scales.
"""

EXPECTED_OBSERVATION_SELECTIONS = {
    # "<group>.<term>": (the joints it reports, in the order it reports them)
    "policy.joint_pos": EXPECTED_SERVO_JOINT_NAMES,
    "policy.joint_vel": EXPECTED_SERVO_JOINT_NAMES,
    "critic.joint_pos": EXPECTED_SERVO_JOINT_NAMES,
    "critic.joint_vel": EXPECTED_SERVO_JOINT_NAMES,
}
"""The four joint blocks the encoder view replaces, and the selection each keeps.

Consumed by :func:`test_both_observation_groups_read_the_servos_through_the_play`. Upstream injects
an ``^(?!passive_).*`` selection into any of these terms that had none, because a term with no
selection would report all 28 joints; here every one is already spelled out as exact names, and this
table is what says so. Two properties ride on it: the widths stay 14, and the order stays the deploy
order the runtime on the robot rebuilds by hand.
"""

EXPECTED_OBSERVATION_FUNCS = {
    "policy.joint_pos": mdp.joint_pos_rel_backlash,
    "policy.joint_vel": mdp.joint_vel_rel_backlash,
    "critic.joint_pos": mdp.joint_pos_rel_backlash,
    "critic.joint_vel": mdp.joint_vel_rel_backlash,
}
"""The encoder-view term each of those blocks is computed by, after upstream's edit 2."""

EXPECTED_CFG_DELTAS = {
    # the plant: a different model, a servo group that reads through the play, and the play
    # hinges' own initial angle -- which needs its own configuration, since Isaac Lab rejects an
    # initial state whose patterns overlap and raises on an unmatched one
    "cfg.scene.robot.spawn.usd_path",
    "cfg.scene.robot.actuators.servos.__class__",
    "cfg.scene.robot.init_state.joint_pos..*_backlash",
    # the encoder view, in both groups. The actor's velocity block is wrapped in the bus-latency
    # term, so its swap lands on the wrapped term rather than on the term itself.
    "cfg.observations.policy.joint_pos.func",
    "cfg.observations.policy.joint_vel.params.term_func",
    "cfg.observations.critic.joint_pos.func",
    "cfg.observations.critic.joint_vel.func",
    # the soft-limit penalty, which had no selection at all and would otherwise charge the 14
    # hinges that ride their limits by construction
    "cfg.rewards.dof_pos_limits.params.asset_cfg",
    # the constraint budget the always-active limit rows need
    "cfg.sim.physics.newton_mjwarp.solver_cfg.njmax",
    "cfg.sim.physics.default.solver_cfg.njmax",
    # the configuration class itself
    "cfg.__class__",
}
"""Every leaf of the assembled configuration that may differ from the base flat task's.

This is the discrepancy table of the whole port: upstream's four edits are two here (report section
1.4), the third is the solver budget, and there is nothing else. Consumed two-sidedly by
:func:`test_the_configuration_differs_from_the_base_flat_task_in_exactly_these_places`, so a leaf
that stops differing fails as loudly as one that starts.
"""

ARMATURE_EXEMPTION = "events.randomize_armature.asset_cfg"
"""The one selection that deliberately reaches the play hinges (report section 1.4, gap 2)."""

EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The 4 head servos, in the order the head-pose command indexes its columns."""

EXPECTED_LEG_JOINT_NAMES = [
    name for name in EXPECTED_SERVO_JOINT_NAMES if "hip" in name or "knee" in name or "ankle" in name
]
"""The 10 leg servos the posture reward holds at the stand pose."""

EXPECTED_JOINT_SELECTIONS = {
    # "<path>": what it resolves to against the played model's 28 joint names
    "actions.joint_pos.joint_names": EXPECTED_SERVO_JOINT_NAMES,
    "observations.policy.joint_pos.asset_cfg": EXPECTED_SERVO_JOINT_NAMES,
    "observations.policy.joint_vel.asset_cfg": EXPECTED_SERVO_JOINT_NAMES,
    "observations.critic.joint_pos.asset_cfg": EXPECTED_SERVO_JOINT_NAMES,
    "observations.critic.joint_vel.asset_cfg": EXPECTED_SERVO_JOINT_NAMES,
    "rewards.pose.asset_cfg": EXPECTED_LEG_JOINT_NAMES,
    "rewards.dof_pos_limits.asset_cfg": EXPECTED_SERVO_JOINT_NAMES,
    "rewards.head_pose_tracking.asset_cfg": EXPECTED_HEAD_JOINT_NAMES,
    "rewards.head_pose_bias.asset_cfg": EXPECTED_HEAD_JOINT_NAMES,
    ARMATURE_EXEMPTION: EXPECTED_ALL_JOINT_NAMES,
}
"""Every selection in the configuration that names joints, and what it selects on this model.

Consumed two-sidedly by :func:`test_no_term_selection_resolves_a_play_hinge`, which resolves each
one against the 28 joint names the way the managers do rather than eyeballing them as "exact names,
so they must be fine". A selection that appears or disappears fails here, which is the point: the
twin's 14-wide contracts rest on the fact that nothing but the last entry reaches a play hinge.
"""


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


def _entity_cfg_of(term_cfg, key: str) -> SceneEntityCfg:
    """Fetch a term's entity selection, looking inside a delayed term's wrapped parameters."""
    if key in term_cfg.params:
        return term_cfg.params[key]
    return term_cfg.params["term_params"][key]


def _term_func(term_cfg):
    """The function a term computes, seen through the bus-latency wrapper where there is one."""
    if term_cfg.func is mdp.delayed_observation:
        return term_cfg.params["term_func"]
    return term_cfg.func


def _flatten(value, path: str, out: dict, depth: int = 0) -> None:
    """Reduce an assembled configuration to ``dotted path -> comparable leaf``.

    Functions and classes are compared by qualified name rather than by identity, so a reloaded
    module cannot make two identical configurations differ, and entity selections are compared by
    what they select rather than by object identity.
    """
    if depth > 25:  # pragma: no cover - the configurations are far shallower than this
        out[path] = "<too deep>"
        return
    if isinstance(value, SceneEntityCfg):
        out[path] = (
            value.name,
            tuple(value.joint_names or ()),
            tuple(value.body_names or ()),
            value.preserve_order,
        )
        return
    if isinstance(value, type) or callable(value):
        out[path] = f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', value)}"
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{path}.{key}", out, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(item, f"{path}[{index}]", out, depth + 1)
        return
    if hasattr(value, "__dict__"):
        out[f"{path}.__class__"] = type(value).__qualname__
        for key, item in vars(value).items():
            if not key.startswith("_"):
                _flatten(item, f"{path}.{key}", out, depth + 1)
        return
    out[path] = value


def _differing_leaves(base, twin) -> set[str]:
    """Paths that were added, removed, or changed between two assembled configurations."""
    base_leaves: dict = {}
    twin_leaves: dict = {}
    _flatten(base, "cfg", base_leaves)
    _flatten(twin, "cfg", twin_leaves)
    symmetric = set(base_leaves) ^ set(twin_leaves)
    changed = {key for key in set(base_leaves) & set(twin_leaves) if base_leaves[key] != twin_leaves[key]}
    return symmetric | changed


##
# Recipe parity against upstream (report section 1.4)
##


@pytest.mark.unit
def test_the_twin_makes_the_edits_upstream_makes_and_skips_the_two_it_does_not_need():
    """Upstream's ``make_backlash_variant`` performs four edits; two of them are no-ops here.

    Edit 2 injects a servo-only selection into any joint observation that never narrowed one, and
    edit 4 prepends a backlash exclusion to the posture reward's standard-deviation patterns. Both
    exist because upstream selects joints with regular expressions that a ``passive_`` joint can
    match. Every selection in this package is spelled out as exact joint names and Isaac Lab matches
    names in full, so neither has anything to do -- and this test is what says that is a fact about
    the configuration rather than an oversight.
    """
    cfg = MicroDuckVelocityBacklashFlatEnvCfg()

    # 1. the matching robot: the walking collision model, so an A/B against the base task is
    # unconfounded, with the servo group that reads its encoder through the play
    assert cfg.scene.robot.spawn.usd_path == MICRODUCK_BACKLASH_USD_PATH
    assert isinstance(cfg.scene.robot.actuators["servos"], BamBacklashActuatorCfg)

    # 2. the encoder view in both groups -- and no injected selection, because there is no term
    # whose selection was ever missing
    for path, func in EXPECTED_OBSERVATION_FUNCS.items():
        group, term_name = path.split(".")
        assert _term_func(_observation_terms(getattr(cfg.observations, group))[term_name]) is func, path

    # 3. the soft-limit penalty scoped to the servos
    assert cfg.rewards.dof_pos_limits.params["asset_cfg"].joint_names == EXPECTED_SERVO_JOINT_NAMES

    # 4. the posture reward is untouched, because no play hinge is in its selection to disambiguate
    base_pose = MicroDuckVelocityFlatEnvCfg().rewards.pose
    assert cfg.rewards.pose.params["asset_cfg"].joint_names == base_pose.params["asset_cfg"].joint_names
    for key in ("std_standing", "std_walking", "std_running"):
        assert cfg.rewards.pose.params[key] == base_pose.params[key], key


@pytest.mark.unit
def test_the_configuration_differs_from_the_base_flat_task_in_exactly_these_places():
    """The twin is an experiment on the plant: every other difference would confound it."""
    differing = _differing_leaves(MicroDuckVelocityFlatEnvCfg(), MicroDuckVelocityBacklashFlatEnvCfg())

    assert differing == EXPECTED_CFG_DELTAS

    # ... and the flattening is sensitive enough for that to mean something: two independently
    # built base configurations differ nowhere at all
    assert _differing_leaves(MicroDuckVelocityFlatEnvCfg(), MicroDuckVelocityFlatEnvCfg()) == set()


@pytest.mark.unit
def test_deriving_the_twin_leaves_the_base_task_untouched():
    """Upstream warns that its base templates share configuration objects across variants.

    Isaac Lab's ``configclass`` deep-copies its defaults per instance, so the twin's edits should
    stay in the twin -- but "should" is the word that makes this worth a test: a leak would silently
    train the base task on the encoder view and on a plant it does not have, and the first symptom
    would be an A/B comparison of two identical recipes.
    """
    MicroDuckVelocityBacklashFlatEnvCfg()
    base = MicroDuckVelocityFlatEnvCfg()

    terms = _observation_terms(base.observations.policy) | _observation_terms(base.observations.critic)
    assert _term_func(terms["joint_pos"]) is mdp.joint_pos_rel_biased
    assert _term_func(terms["joint_vel"]) is mdp.joint_vel_rel
    assert "asset_cfg" not in base.rewards.dof_pos_limits.params
    assert base.sim.physics.newton_mjwarp.solver_cfg.njmax == 64
    assert isinstance(base.scene.robot.actuators["servos"], BamActuatorCfg)
    assert not isinstance(base.scene.robot.actuators["servos"], BamBacklashActuatorCfg)


@pytest.mark.unit
def test_both_observation_groups_read_the_servos_through_the_play():
    """The encoder view is the actor's *and* the critic's: a value function that read the motor
    angle would be valuing a state the actor can neither observe nor act on."""
    observations = MicroDuckVelocityBacklashFlatEnvCfg().observations
    groups = {
        "policy": _observation_terms(observations.policy),
        "critic": _observation_terms(observations.critic),
    }

    for path, expected in EXPECTED_OBSERVATION_SELECTIONS.items():
        group, term_name = path.split(".")
        term = groups[group][term_name]
        entity_cfg = _entity_cfg_of(term, "asset_cfg")
        assert entity_cfg.name == "robot", path
        assert entity_cfg.joint_names == expected, path
        assert entity_cfg.preserve_order, path
        assert _term_func(term) is EXPECTED_OBSERVATION_FUNCS[path], path
    # the actor keeps the encoder-bias half of the calibration loop, which the encoder view composes
    # on the servo reading only
    assert groups["policy"]["joint_pos"].params["biased"] is True
    assert groups["critic"]["joint_pos"].params["biased"] is False


@pytest.mark.unit
def test_no_term_selection_resolves_a_play_hinge():
    """The ``passive_`` prefix is what keeps the twin's observations, actions and rewards 14 wide.

    Every joint selection in the configuration is enumerated and resolved against the played model's
    28 names, the way the managers resolve them. Upstream gets this property from a
    ``^(?!passive_).*`` lookahead on every selector; this port gets it from exact names, which is a
    stronger guarantee but an invisible one until it is measured.
    """
    cfg = MicroDuckVelocityBacklashFlatEnvCfg()

    selections = {
        f"{manager}.{term_name}.{key}": value
        for manager in ("rewards", "events", "terminations", "curriculum")
        for term_name, term in vars(getattr(cfg, manager)).items()
        for key, value in (getattr(term, "params", None) or {}).items()
        if isinstance(value, SceneEntityCfg) and value.joint_names is not None
    }
    for group_name in ("policy", "critic"):
        for term_name, term in _observation_terms(getattr(cfg.observations, group_name)).items():
            entity_cfg = term.params.get("term_params", term.params).get("asset_cfg")
            if isinstance(entity_cfg, SceneEntityCfg) and entity_cfg.joint_names is not None:
                selections[f"observations.{group_name}.{term_name}.asset_cfg"] = entity_cfg
    selections["actions.joint_pos.joint_names"] = SceneEntityCfg(
        "robot", joint_names=list(cfg.actions.joint_pos.joint_names)
    )

    assert set(selections) == set(EXPECTED_JOINT_SELECTIONS)

    played = set()
    for path, entity_cfg in selections.items():
        _, resolved = resolve_matching_names(entity_cfg.joint_names, EXPECTED_ALL_JOINT_NAMES, preserve_order=True)
        assert sorted(resolved) == sorted(EXPECTED_JOINT_SELECTIONS[path]), path
        if set(resolved) & set(EXPECTED_PLAY_JOINT_NAMES):
            played.add(path)

    # exactly one selection reaches them, and it does so on purpose
    assert played == {ARMATURE_EXEMPTION}


@pytest.mark.unit
def test_the_soft_limit_penalty_is_scoped_to_the_servos():
    """A gear-play hinge lives on its limits, so an unscoped penalty is a tax nothing can avoid.

    That is upstream's edit 3, and its reasoning verbatim: "backlash joints spend their life pinned
    against their +/-1 degree limits (that is the point of backlash)". Fourteen saturated rows would
    swamp the signal the term exists for, on every step of every episode.
    """
    cfg = MicroDuckVelocityBacklashFlatEnvCfg()
    asset_cfg = cfg.rewards.dof_pos_limits.params["asset_cfg"]

    assert asset_cfg.name == "robot"
    assert asset_cfg.joint_names == EXPECTED_SERVO_JOINT_NAMES
    assert not set(asset_cfg.joint_names) & set(EXPECTED_PLAY_JOINT_NAMES)
    # the base task leaves it unscoped, which is what made the edit necessary
    assert "asset_cfg" not in MicroDuckVelocityFlatEnvCfg().rewards.dof_pos_limits.params
    assert cfg.rewards.dof_pos_limits.weight == pytest.approx(-1.0)


@pytest.mark.unit
def test_the_armature_randomization_still_covers_the_play_hinges():
    """The one deliberate exception to the exclusion, and it is upstream's.

    ``make_backlash_variant`` edits the reward set and never the event set, so upstream's ``.*``
    armature randomization scales the play hinges' 0.001 conditioning armature along with the
    servos' rotor inertia. It is reproduced rather than narrowed: the comparison this twin exists to
    make is against upstream's plant, and narrowing it is a retune with its own training run.
    """
    events = MicroDuckVelocityBacklashFlatEnvCfg().events

    assert events.randomize_armature.params["asset_cfg"].joint_names == [".*"]
    assert events.randomize_armature.params["armature_distribution_params"] == (0.9, 1.1)
    assert events.randomize_armature.params["operation"] == "scale"


@pytest.mark.unit
def test_the_solver_budget_covers_the_always_active_limit_rows():
    """The base task's ``njmax`` was profiled on a plant with 14 fewer permanently active rows."""
    cfg = MicroDuckVelocityBacklashFlatEnvCfg()
    solver = cfg.sim.physics.default.solver_cfg
    base_solver = MicroDuckVelocityFlatEnvCfg().sim.physics.default.solver_cfg

    assert solver.njmax == MICRODUCK_BACKLASH_NJMAX
    assert solver.njmax > STRUCTURAL_CONSTRAINT_BOUND
    # the shipped budget of the base task does not, which is why this is not inherited
    assert base_solver.njmax < STRUCTURAL_CONSTRAINT_BOUND
    # the play hinges are joints and not colliders, so the contact budget is untouched
    assert solver.nconmax == base_solver.nconmax
    # and the rest of the solver profile is the base task's
    assert (solver.iterations, solver.ls_iterations) == (base_solver.iterations, base_solver.ls_iterations)
    assert solver.use_mujoco_default_joint_limit_solref is True


@pytest.mark.unit
def test_the_registered_task_reuses_the_base_velocity_runner():
    """Upstream reuses the base task's runner for its twins, and for the same reason.

    Different hyper-parameters would confound the comparison the twin exists to make, so the two
    tasks differ in their plant and in nothing a learner can see.
    """
    spec = gym.spec(TASK_NAME)
    base_spec = gym.spec(BASE_TASK_NAME)

    assert spec.kwargs["env_cfg_entry_point"].endswith("backlash_env_cfg:MicroDuckVelocityBacklashFlatEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == base_spec.kwargs["rsl_rl_cfg_entry_point"]
    assert spec.kwargs["default_agent"] == "rsl_rl"
    assert spec.entry_point == base_spec.entry_point
    # the experiment name is the base task's too, so the twin's runs land beside the ones they are
    # to be compared with
    assert MicroDuckPPORunnerCfg().experiment_name == "microduck_velocity"


@pytest.mark.unit
def test_the_episode_simulation_and_actuator_settings_are_the_base_task_s():
    """Nothing about the control loop moves: only the plant does."""
    cfg = MicroDuckVelocityBacklashFlatEnvCfg()
    base = MicroDuckVelocityFlatEnvCfg()

    assert (cfg.decimation, cfg.sim.dt, cfg.episode_length_s) == (base.decimation, base.sim.dt, base.episode_length_s)
    # the played servo group is Newton-native only and raises on the Isaac Lab-executed path
    assert cfg.sim.use_newton_actuators is True
    # an even decimation is what lets the stateful servo delay line be CUDA-graph-captured
    assert cfg.decimation % 2 == 0


##
# Environment smoke and acceptance tests
##


@pytest.mark.integration
@requires_microduck_backlash_usd
def test_the_observation_and_action_widths_are_the_ones_their_contracts_name():
    """28 joints, 14 actions and the same 61-wide actor: the whole point of the encoder view."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        obs, _ = env.reset()

        assert obs["policy"].shape[-1] == ACTOR_OBSERVATION_DIM
        assert obs["critic"].shape[-1] == CRITIC_OBSERVATION_DIM
        # per-term widths as well as the total, so two compensating drifts cannot agree on 61
        manager = env.unwrapped.observation_manager
        for group, expected in (("policy", ACTOR_OBSERVATION_TERMS), ("critic", CRITIC_OBSERVATION_TERMS)):
            measured = [
                (name, dim[0]) for name, dim in zip(manager.active_terms[group], manager.group_obs_term_dim[group])
            ]
            assert measured == expected, group

        # the converted robot carries the names the whole pairing is built on
        robot = env.unwrapped.scene["robot"]
        assert robot.num_joints == len(EXPECTED_ALL_JOINT_NAMES)
        assert set(robot.joint_names) == set(EXPECTED_ALL_JOINT_NAMES)
        # 28 joints, 14 of them driven, and none of the driven ones a play hinge
        assert env.unwrapped.action_manager.total_action_dim == ACTION_DIM
        action_joints = [robot.joint_names[int(i)] for i in env.unwrapped.action_manager._terms["joint_pos"]._joint_ids]
        assert action_joints == EXPECTED_SERVO_JOINT_NAMES
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_backlash_usd
def test_the_policy_reads_each_servo_through_the_hinge_in_series_with_it():
    """The acceptance test for the thing that makes this a backlash twin.

    The critic's joint block is the uncorrupted one -- no bias, no noise, no bus latency -- so it can
    be checked against the articulation state directly. Every one of the 14 columns has to be the
    servo *plus its own twin*, which a mis-paired or an unpaired block would both fail; and the
    summand has to be non-zero, or the model is not played and the test is vacuous.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()
        robot = unwrapped.scene["robot"]

        # let the robot load its gear teeth: from a centred reset the play is exactly zero and the
        # encoder view is trivially the servo reading
        action = torch.zeros((unwrapped.num_envs, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
        with torch.inference_mode():
            for _ in range(50):
                obs, *_ = env.step(action)

        servo_ids = [list(robot.joint_names).index(name) for name in EXPECTED_SERVO_JOINT_NAMES]
        play_ids = [list(robot.joint_names).index(name) for name in EXPECTED_PLAY_JOINT_NAMES]
        joint_pos = robot.data.joint_pos.torch
        default = robot.data.default_joint_pos.torch
        expected = joint_pos[:, servo_ids] + joint_pos[:, play_ids] - default[:, servo_ids]

        # the critic's joint_pos block, located by the layout table rather than by a magic offset
        start = sum(
            width for name, width in itertools.takewhile(lambda term: term[0] != "joint_pos", CRITIC_OBSERVATION_TERMS)
        )
        measured = obs["critic"][:, start : start + len(servo_ids)]

        torch.testing.assert_close(measured, expected, rtol=1e-4, atol=1e-5)
        # and the play is actually carrying something, so the equality above is not 14 zeros
        play = joint_pos[:, play_ids]
        assert float(play.abs().max()) > 1e-3, f"the gear play never opened: {float(play.abs().max())}"
        # ... and what it carries is the gear play rather than some other joint's motion. The bound
        # is deliberately loose: a joint limit is a soft constraint, so a loaded hinge rides *past*
        # its nominal degree -- the investigation measured up to 2.2x with MuJoCo's default limit
        # solref, and the asset's stiffened one still admits about 1.2x here. This assertion
        # identifies the summand, it does not pin the constraint stiffness (which is
        # ``artifacts/microduck/golden_trajectories/backlash/``'s job).
        assert float(play.abs().max()) < PLAY_OVERSHOOT_BOUND * NOMINAL_PLAY_RAD
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_backlash_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
