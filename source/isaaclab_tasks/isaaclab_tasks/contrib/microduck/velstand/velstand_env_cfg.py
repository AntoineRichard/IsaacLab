# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-plus-fall-recovery environment for the Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``artifacts/microduck/upstream_reference_tasks3.md``, the extraction of the pinned upstream
checkouts; the sections that describe the shared mjlab base template live in the companion
``artifacts/microduck/upstream_reference.md`` and are cited as "reference section N".

One policy walks *and* gets back up. That is the whole task, and it is why this is the only
MicroDuck environment in the port that **derives from another one**: the walking half is
:class:`~isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg.MicroDuckVelocityFlatEnvCfg`
verbatim, as upstream's own factory takes the velocity recipe verbatim, so a change to the proven
walking recipe reaches this task rather than being restated and left to drift. Everything below is
the recovery layer on top of it.

Three phases, and the boundaries between them are the design (addendum section 3.6):

* **Walk**, iterations 0-500. The inherited tilt termination ends a fall, so the rollout is clean
  walking data and nothing else.
* **Fall**, from iteration 500. ``fell_over``'s limit angle is widened to half a turn, so a fall is
  no longer an episode end but a recovery to attempt. ``fallen_too_long`` is the backstop: eight
  seconds continuously down and the episode is recycled instead of being spent on the floor. The
  recovery *economics* stay off until 1200, which buys a tax-free window where a get-up attempt
  costs nothing and the two potential-based progress terms alone can teach it.
* **Prone**, from iteration 1500. The reset distribution ramps toward the prone poses, capped at
  45 % so at least 55 % of the experience is still clean walking -- upstream's earlier design put
  two thirds of resets prone and starved the walk.

Two structural differences from :mod:`..velocity` follow from that and are worth stating up front:

* The robot is :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG`, not the walking model. A robot
  that lies down and pushes itself up needs a trunk, hips, shins and head that touch the floor.
  Swapping it in also makes the wide self-collision sensor reachable, which the walking scene has
  no colliders for.
* Every recovery reward is either **potential-based** or **gated on actually being toppled**, so the
  layer contributes exactly zero during clean walking. Upstream's own run notes record what the
  alternative costs: any positive reward for *being* in a fallen-ish state gets farmed from some
  comfortable pose -- sitting, lying, and a head tripod at 55 degrees were all observed.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
    MicroDuckSceneCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    CurriculumCfg as MicroDuckVelocityCurriculumCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    EventsCfg as MicroDuckVelocityEventsCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    RewardsCfg as MicroDuckVelocityRewardsCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    TerminationsCfg as MicroDuckVelocityTerminationsCfg,
)

from isaaclab_assets import MICRODUCK_ALLCOLLISIONS_CFG

##
# Fallen gates (addendum section 3.1)
##

MICRODUCK_FALLEN_TILT_DEG = 40.0
"""Trunk tilt [deg] beyond which the *recovery rewards* treat the robot as fallen.

Tilt only, never height. Upstream reached the same thing by passing a height gate of 0.0 m, which
the trunk never goes below, and its comment records why the height half was removed: gating on low
height made **sitting** open the gate -- trunk upright at about 0.07 m -- so the policy learned to
sit and farm the rising rewards while bobbing. A gate at 40 degrees of tilt cannot be opened from a
comfortable pose.
"""

MICRODUCK_TERMINATION_Z = 0.08
"""Trunk height [m] below which the *termination* treats the robot as down, whatever its tilt.

The asymmetry with :data:`MICRODUCK_FALLEN_TILT_DEG` is deliberate: a sitter is not paid as fallen
but is recycled as stuck, so sitting is neither rewarded nor a comfortable place to wait. 0.08 m
rather than 0.10 m because a normally wobbling upright MicroDuck dips to 0.084-0.096 m, and a gate
inside that envelope recycled crouch-walking explorers every few seconds; 0.08 m still catches
sitting at about 0.07 m and prone at about 0.05 m.
"""

MICRODUCK_FALLEN_TIMEOUT_S = 8.0
"""Continuous time [s] spent down before the episode is recycled.

Upstream raised it from 5.0 s after measuring that a face-down recovery spent most of a 5 s budget
reaching the deep crouch and was then recycled right at the frontier, leaving the crouch-to-stand
last mile with almost no on-policy data.
"""

##
# "Recovery complete" (addendum section 3.1). Shared by the bounty and the tax release.
##

MICRODUCK_RECOVERED_TILT_DEG = 25.0
MICRODUCK_RECOVERED_HEIGHT = 0.09
"""Trunk tilt [deg] and height [m] at which a recovery counts as finished.

The height has to sit **inside** the policy's real standing envelope rather than at the keyframe:
a normally wobbling upright robot measures 0.084 to 0.096 m where the full stand keyframe settles at
about 0.117 m. Upstream's earlier 0.105 m demanded standing taller than the policy ever is, so the
bounty never fired and recoveries converged on a deep crouch just past the reward gates. 0.09 m is
reachable on every stand and still 2 cm above sitting and 4 cm above prone.
"""

MICRODUCK_STAND_HEIGHT = 0.115
"""Trunk height [m] the rise potential stops paying at.

The settled stand, measured on a velocity policy holding the robot still; the model reaches
0.11718 m geometrically at the stand pose and the servos sag the rest under load. Above it the
potential is flat, so a standing robot cannot farm the term by bouncing.
"""

##
# Reset buckets (addendum section 3.4)
##

MICRODUCK_PRONE_Z_RANGE = (0.05, 0.09)
"""Trunk height band [m] a prone spawn lands in.

The function upstream inherits defaulted to 0.20-0.25 m, which opened every prone episode with a
15-20 cm free fall; its own audit calls that a bug and passes this band instead. A face-down trunk
rests at about 0.044 m, so this drops it a centimetre rather than a hand's width.
"""

MICRODUCK_STANDING_Z_RANGE = (0.12, 0.13)
"""Trunk height band [m] an ordinary upright spawn lands in.

Upstream leaves these episodes *untouched* by the prone event, so they keep what the inherited
velocity root reset wrote -- and that reset samples exactly this band. Routing them through the
standing bucket instead of leaving them alone is what lets one event own the whole mixture, and it
reproduces the same distribution.
"""

MICRODUCK_CROUCH_Z_RANGE = (0.06, MICRODUCK_STAND_HEIGHT)
"""Trunk height band [m] the crouch bucket interpolates across, deepest first.

Not sampled independently: the same depth draw picks the height, the forward lean and the leg fold,
so a crouch spawn is a consistent pose part-way along the recovery's last mile rather than three
independent draws.
"""

MICRODUCK_CROUCH_DEPTH_RANGE = (0.35, 1.0)
"""Bounds [-] on the crouch depth draw, 0 being the stand pose and 1 the deep-crouch anchor.

The floor at 0.35 keeps the bucket on the stretch it was added for; a draw near zero would just be
another standing spawn, which the standing bucket already supplies.
"""

MICRODUCK_CROUCH_PITCH_MAX = math.radians(55.0)
"""Forward trunk lean [rad] of the deepest crouch. The stuck basin is a forward crouch whichever
direction the fall came from, so the lean is one-sided."""

MICRODUCK_CROUCH_JOINT_NOISE = 0.12
"""Half-width [rad] of the uniform joint noise on a crouch spawn, about 7 degrees per joint."""

MICRODUCK_CROUCH_JOINT_POS = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}
"""The deep-crouch anchor [rad]: knees folded under the body, feet flat, hips leaning in.

Upstream derives it by extending the stand pose's hip-forward / knee-back / ankle-forward zig-zag
into deep flexion, staying inside the model's +/-1.571 rad sagittal limits. Hip yaw, hip roll and
the neck are absent and therefore stay at the stand pose. Keyed by name here where upstream keys by
servo index, because the converted asset resolves joints in Newton's order rather than the MJCF's.
"""

##
# Phase boundaries (addendum section 3.1). Steps are PPO iterations; ``_iterations`` converts.
##

MICRODUCK_FELL_OVER_DISABLE_ITER = 500
"""Iteration the tilt termination is widened at, so a fall becomes a recovery to train on."""

MICRODUCK_RECOVERY_ECON_ITER = 1200
"""Iteration the fallen tax, the recovery bounty and the rise reward all switch on at.

It is not about the walk being ready -- upstream tried 800 for that reason and prone recovery never
bootstrapped. What the gap between 500 and 1200 buys is a **tax-free window**: falls are already
survivable, so get-up attempts cost nothing beyond the shared regularizers and the two progress
terms alone can teach them. With the tax live from 800, hopeless prone episodes bled -0.5 per step
for the full timeout and the policy learned to avoid tilting instead of to recover.
"""

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])


def _iterations(count: int) -> int:
    """Convert an upstream PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


def _ground_state_mix(prone_prob: float, face_down_share: float, crouch_prob: float) -> dict[str, float]:
    """Upstream's ``(prone_prob, face_down_prob, crouch_prob)`` triple as bucket probabilities.

    Upstream's event draws one uniform and partitions it into prone / crouch / *untouched*, splitting
    the prone slice again by ``face_down_prob``. The port's
    :func:`~isaaclab_tasks.contrib.microduck.mdp.events.reset_ground_state` names the two prone poses
    separately and has no untouched bucket, so the untouched slice becomes a **standing** spawn --
    which is what it already was, since the inherited root reset places it upright in
    :data:`MICRODUCK_STANDING_Z_RANGE`. The partition stays exclusive and the four probabilities sum
    to one, so the mixture is upstream's rather than an approximation of it.

    Args:
        prone_prob: Upstream's share of episodes that start lying down.
        face_down_share: Upstream's split of that share into face-down rather than face-up.
        crouch_prob: Upstream's share of episodes that start part-way through a recovery.

    Returns:
        The four live bucket probabilities, as ``reset_ground_state`` parameters.
    """
    return {
        "face_down_prob": prone_prob * face_down_share,
        "face_up_prob": prone_prob * (1.0 - face_down_share),
        "crouch_prob": crouch_prob,
        "standing_prob": 1.0 - prone_prob - crouch_prob,
    }


MICRODUCK_GROUND_STATE_STAGES = [
    {"step": _iterations(0), "params": _ground_state_mix(0.00, 1.00, 0.00)},
    {"step": _iterations(800), "params": _ground_state_mix(0.00, 1.00, 0.15)},
    {"step": _iterations(1500), "params": _ground_state_mix(0.15, 0.80, 0.15)},
    {"step": _iterations(2000), "params": _ground_state_mix(0.30, 0.65, 0.15)},
    {"step": _iterations(2500), "params": _ground_state_mix(0.45, 0.50, 0.15)},
]
"""Upstream's reset-distribution ramp (addendum section 3.1, ``PRONE_RAMP_STAGES``).

The crouch slice comes first, at iteration 800: those spawns are near-upright, they are harmless
before the economics switch on, and they double as full-stand posture data. The prone slice waits
until 1500 -- after the tax-free window -- and is capped at 45 %, face-down first because it is the
easier recovery.
"""


##
# Scene definition
##


@configclass
class MicroDuckVelStandSceneCfg(MicroDuckSceneCfg):
    """The velocity scene on the all-collisions robot, with the self-collision sensor widened.

    Two fields change and the rest -- terrain, foot contact sensor, lighting -- is the walking
    scene's.
    """

    # The all-collisions model: the walking robot with six more colliders and with both shins
    # promoted from self-collision-only to world contact, so ten of its eleven colliders reach the
    # ground. A task whose episodes start and end on the floor needs all of them, and the walking
    # model's two soles would let the trunk sink through the plane.
    robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The many-to-many self-collision sensor the stand-up and forward-roll tasks use, which is
    # reachable here for the first time on a velocity-derived task: the walking model's other
    # colliders are disabled by the converter, so its sensor can only watch sole against sole.
    #
    # ``prim_path`` is ignored for the sensing objects once ``sensor_shape_prim_expr`` is set, but
    # the base sensor still requires one; the trunk is the cheapest expression that resolves.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
    )


##
# MDP settings
##


@configclass
class RewardsCfg(MicroDuckVelocityRewardsCfg):
    """The walking reward recipe plus the recovery layer (addendum section 3.2).

    Nothing is removed. Six terms are added and three inherited ones have parameters updated in
    :meth:`MicroDuckVelStandFlatEnvCfg.__post_init__`, which is where upstream makes the same three
    edits -- restating the walking terms here would let the two recipes drift apart, which is the one
    thing a task built on "the proven walker, verbatim" must not allow.

    The added terms come in three shapes, and the shape is what keeps them from being farmed:

    * **Potential-based** (:attr:`upright_progress`, :attr:`height_progress`). They pay the *change*
      in a scalar and exactly zero for holding any pose, so no comfortable posture earns anything.
      Both are ungated, which also pays for catching a stumble mid-gait.
    * **Fallen-gated** (:attr:`com_upward_velocity`, :attr:`fallen_tax`, :attr:`recovery_success`).
      They contribute nothing at all while the robot is upright, so the walking recipe's balance is
      untouched, and all three are held at weight zero until the tax-free window closes.
    * **Ungated regularization** (:attr:`joint_torque_rate_l2`). Upstream deliberately ships **no**
      impact penalties here: the duck's recovery pushes off with its head and trunk, and a head
      impact penalty taxed exactly that strategy until falling stayed cheaper than getting up.
    """

    ##
    # Potential-based progress: the dense signal a recovery follows.
    ##

    # A full prone-to-stand recovery collects Delta ~ +1, so about +5 in total.
    upright_progress = RewTerm(func=mdp.upright_progress, weight=5.0, params={"asset_cfg": _TRUNK_BODY_CFG})
    # The z-axis companion. The crouch-to-stand last mile is mostly height change at modest tilt,
    # where the orientation potential barely moves and every Gaussian posture term is flat. A full
    # rise from 0.05 m collects Delta ~ +0.065, so about +2 in total; the last mile alone is +1.
    height_progress = RewTerm(
        func=mdp.height_progress,
        weight=30.0,
        params={"asset_cfg": _TRUNK_BODY_CFG, "ceiling": MICRODUCK_STAND_HEIGHT},
    )

    ##
    # Recovery economics. All three ship at weight zero and are ramped in together at iteration 1200.
    ##

    # The ceiling sits just above standing so the rise keeps paying until the stand is finished; it
    # is the fallen gate, not the ceiling, that stops a walking robot farming gait bounce.
    com_upward_velocity = RewTerm(
        func=mdp.com_upward_velocity,
        weight=0.0,
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "max_height": 0.125,
            "gate_tilt_above_deg": MICRODUCK_FALLEN_TILT_DEG,
        },
    )
    # A flat tax while down, so lying still is strictly worse than trying: without it, waiting for
    # the failed-recovery timeout is rational, because an attempt costs action-rate and torque-rate
    # penalties where waiting costs nothing. The hysteresis keeps it charging through the sub-40
    # degree crouch, which is otherwise a zero-cost rest state just past every reward gate.
    fallen_tax = RewTerm(
        func=mdp.fallen_state_penalty,
        weight=0.0,
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "gate_tilt_above_deg": MICRODUCK_FALLEN_TILT_DEG,
            "release_tilt_below_deg": MICRODUCK_RECOVERED_TILT_DEG,
            "release_z_above": MICRODUCK_RECOVERED_HEIGHT,
        },
    )
    # The sharp endpoint signal the dense terms lack, one-shot so gate oscillation pays once.
    recovery_success = RewTerm(
        func=mdp.recovery_success,
        weight=0.0,
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "fallen_tilt_deg": MICRODUCK_FALLEN_TILT_DEG,
            "min_fallen_s": 0.5,
            "up_tilt_deg": MICRODUCK_RECOVERED_TILT_DEG,
            "up_z": MICRODUCK_RECOVERED_HEIGHT,
        },
    )

    ##
    # Transfer smoothness.
    ##

    # The stand-up specialist's anti-jitter term, at a flat weight and with no curriculum. It prices
    # torque *change* rather than magnitude or rotation, so it smooths the transfer without taxing
    # the recovery flip the way an impact penalty would.
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=-2e-3, params={"asset_cfg": _SERVO_JOINT_CFG})


@configclass
class EventsCfg(MicroDuckVelocityEventsCfg):
    """The walking randomization suite plus the ground-state reset (addendum section 3.4).

    The domain randomization is the velocity task's, term for term and range for range, so a policy
    trained here meets the same hardware spread as one trained only to walk.

    **Declaration order is behaviour.** Isaac Lab fires reset events in declaration order, and
    :attr:`set_ground_state` overwrites the root height and orientation that ``reset_base`` wrote.
    Inheriting appends it after every reset the walking recipe declares, which is where upstream
    inserts it too; only the horizontal spread of the root reset survives.
    """

    # Upstream calls this term ``random_prone_init`` and gives it a prone-or-untouched function of
    # its own; the port routes it through the shared ``reset_ground_state`` the stand-up task
    # already uses, whose crouch bucket was added for this task. See ``_ground_state_mix`` for why
    # the two parameterizations describe the same mixture.
    #
    # The probabilities here are the ``ground_state_mix`` curriculum's first stage. The height bands
    # and the crouch anchor are left alone by that curriculum, so they are configured once -- and
    # they must be configured from the start even though the buckets they belong to open later,
    # because the curriculum flips a bucket live and a bucket with no band to spawn from raises.
    set_ground_state = EventTerm(
        func=mdp.reset_ground_state,
        mode="reset",
        params={
            **MICRODUCK_GROUND_STATE_STAGES[0]["params"],
            # this task has no seated keyframe: its recovery hand-off is the crouch continuum
            "sitting_prob": 0.0,
            "prone_z_range": MICRODUCK_PRONE_Z_RANGE,
            "standing_z_range": MICRODUCK_STANDING_Z_RANGE,
            "crouch_z_range": MICRODUCK_CROUCH_Z_RANGE,
            "crouch_joint_pos": MICRODUCK_CROUCH_JOINT_POS,
            "crouch_depth_range": MICRODUCK_CROUCH_DEPTH_RANGE,
            "crouch_pitch_max": MICRODUCK_CROUCH_PITCH_MAX,
            "crouch_joint_noise": MICRODUCK_CROUCH_JOINT_NOISE,
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )


@configclass
class TerminationsCfg(MicroDuckVelocityTerminationsCfg):
    """The walking terminations plus the failed-recovery backstop (addendum section 3.5).

    ``fell_over`` is **kept** rather than deleted, unlike on the stand-up task: it is what makes the
    first five hundred iterations clean walking, and the ``fell_over_disable`` curriculum widens its
    limit angle to half a turn afterwards rather than removing the term.
    """

    # Once a fall no longer ends the episode, a robot that never gets up would spend the remaining
    # eighteen seconds on the floor. The gate is height *or* tilt where the reward gates are tilt
    # alone, so a stuck sitter is recycled rather than paid.
    fallen_too_long = DoneTerm(
        func=mdp.fallen_too_long,
        time_out=False,
        params={
            "gate_z_below": MICRODUCK_TERMINATION_Z,
            "gate_tilt_above_deg": MICRODUCK_FALLEN_TILT_DEG,
            "max_duration_s": MICRODUCK_FALLEN_TIMEOUT_S,
        },
    )


@configclass
class CurriculumCfg(MicroDuckVelocityCurriculumCfg):
    """The walking curricula plus the three-phase recovery schedule (addendum section 3.6).

    The walking recipe's own ramps -- action rate, head-pose bias and range, standing fraction,
    centre-of-mass spread -- are inherited unchanged and are what keeps the walk on the same schedule
    it was proven on.

    Note:
        :attr:`fell_over_disable` names a termination term, so removing that term -- as an ablation
        or a play variant would -- means removing this curriculum with it, exactly as the flat
        walking task removes ``terrain_levels`` along with its terrain generator. Upstream instead
        makes its equivalent tolerate a missing term, which would turn a mistyped term name into a
        schedule that silently never fires.
    """

    ##
    # Walk to fall, at iteration 500.
    ##

    # Widened rather than deleted, so the termination manager's shape does not change mid-run.
    fell_over_disable = CurrTerm(
        func=mdp.termination_param_stages,
        params={
            "term_name": "fell_over",
            "param_stages": [
                {"step": _iterations(0), "params": {"limit_angle": math.radians(70.0)}},
                {"step": _iterations(MICRODUCK_FELL_OVER_DISABLE_ITER), "params": {"limit_angle": math.pi}},
            ],
        },
    )

    ##
    # Fall to prone: the reset distribution, from iteration 800.
    ##

    ground_state_mix = CurrTerm(
        func=mdp.event_param_stages,
        params={"event_name": "set_ground_state", "param_stages": MICRODUCK_GROUND_STATE_STAGES},
    )

    ##
    # Recovery economics, all three at iteration 1200.
    ##

    fallen_tax_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "fallen_tax",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(MICRODUCK_RECOVERY_ECON_ITER), "weight": -0.5},
            ],
        },
    )

    recovery_success_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "recovery_success",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(MICRODUCK_RECOVERY_ECON_ITER), "weight": 10.0},
            ],
        },
    )

    com_upward_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "com_upward_velocity",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(MICRODUCK_RECOVERY_ECON_ITER), "weight": 2.0},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckVelStandFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """MicroDuck velocity-tracking-with-fall-recovery environment on flat ground."""

    scene: MicroDuckVelStandSceneCfg = MicroDuckVelStandSceneCfg(num_envs=4096, env_spacing=2.0)
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # The three edits upstream makes to inherited terms, made the same way: in place, so the
        # walking recipe stays the single source of the parameters they do not change.
        #
        # The head-pose bias gets an upright gate. In the walking task the ungated moving average is
        # safe because the tilt termination ends a fallen episode; here episodes survive falls, so
        # the ungated average would charge head "droop" through the whole ground phase -- a flat tax
        # on being fallen that the recovery economics never priced in. The gate multiplies the error
        # entering the average as well as the output, so arriving upright starts the clock from
        # roughly zero rather than paying the ground phase's accumulated error at the finish line.
        self.rewards.head_pose_bias.params.update(
            {
                "gate_height_low": 0.09,
                "gate_height_high": 0.11,
                "gate_tilt_full_deg": 20.0,
                "gate_tilt_zero_deg": MICRODUCK_FALLEN_TILT_DEG,
            }
        )
        # Air time is zeroed while toppled: a robot lying on its trunk can rhythmically tap its feet
        # through the swing window, which is the "shaking a leg" farm upstream observed.
        self.rewards.air_time.params["gate_tilt_above_deg"] = MICRODUCK_FALLEN_TILT_DEG
        # The wide sensor reports one contact from each side of a pair, so the cost saturates to
        # upstream's 0-or-1 "is the robot touching itself" signal rather than becoming a per-collider
        # tariff. The walking scene's one-against-one sensor needs no saturation and does not set it.
        self.rewards.self_collisions.params["saturate"] = True

        # Measured, not inherited: the all-collisions robot spends part of every episode on the
        # floor, so the walking task's budget -- sized for two soles on a plane, 10 contacts and 54
        # constraints -- does not cover it. Profiling under random actions with the tilt termination
        # dropped and the pushes forced to full magnitude peaks at **27 contacts and 82 constraints**
        # per environment, and the profile was run at both 256 and 2048 environments, which agree to
        # one contact -- that agreement is what says the tail has been sampled, because this peak
        # moves with the pose rather than being structural the way the walking task's is. Logs:
        # ``artifacts/microduck/profile_microduck_contacts_velstand_{256,2048}envs.log``, from
        # ``artifacts/microduck/profile_microduck_contacts.py``. The numbers match the stand-up
        # task's exactly, which is the corroboration: same model, same floor-contact regime.
        #
        # ``njmax`` is a hard per-environment cap and carries the margin; ``nconmax`` is a
        # per-environment share of one shared buffer and cannot overflow at the measured peak, so it
        # sits just above it.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 128
        newton_mjwarp.solver_cfg.nconmax = 32
        self.sim.physics.default = newton_mjwarp
