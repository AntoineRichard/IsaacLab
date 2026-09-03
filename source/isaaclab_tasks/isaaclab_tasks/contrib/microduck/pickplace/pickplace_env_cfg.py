# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-place environment for the Pollen Robotics MicroDuck biped.

**This task has no upstream counterpart.** Every other MicroDuck environment in this package is a
port of `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and cites
the extraction of the pinned upstream checkout for each of its numbers. This one is designed here,
so the document it cites instead is ``artifacts/microduck/pickplace/DESIGN.md``: the term table, the
rulings behind each judgment call, and the reward-hacking audit the weights were chosen from. What
it inherits from the family is conventions, not values -- the robot, the servo deployment, the
domain-randomization suite, the regularizers and the solver profile are the siblings'.

The robot spawns standing at a random heading with an object on the ground in front of it and a drop
point commanded somewhere else. It has to walk to the object, put its mouth on it, carry it, and set
it down on the target.

Three things are worth knowing before reading the stack:

* **The mouth is not a gripper.** There is no actuated jaw close on this robot -- the head degrees of
  freedom are neck pitch and head pitch/yaw/roll. What holds the object is a *compliant virtual
  weld*: a spring-damper between the mouth tip and the object with equal and opposite wrenches, run
  by :func:`~isaaclab_tasks.contrib.microduck.mdp.events.update_pickplace_latch`. The object stays a
  fully dynamic rigid body, so the robot really carries its weight; ruling R-PP1 records the two
  alternatives and why neither survived.
* **Nothing is scheduled.** Unlike the ground-pick task there is no phase clock: approach, pick,
  carry and place are gated on the *latch state*, so the phases emerge from what the robot has
  achieved rather than from a cycle it is told the position of.
* **v1 is state-based and shaped so that v2 is a one-term change.** The actor reads the object's
  position as its own observation term, never folded into a robot-state term, expressed in the robot
  base frame -- which is what a camera bolted to the head would produce. Swapping
  :attr:`ObservationsCfg.PolicyCfg.object_position` for a perception term is the whole of the camera
  migration.

The 61-wide deployed observation contract the walking family shares deliberately does **not** apply
here (design document ruling R-PP5): this is a new task with a new runtime, and padding the actor to
61 with zeros would advertise a hot-swap compatibility that does not exist.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.microduck.mdp as mdp

# The mouth-tip site adaptation, measured off the pinned upstream MJCF by the ground-pick task and
# imported rather than restated -- the same relationship that task has with the forward-roll task's
# head-body constant.
from isaaclab_tasks.contrib.microduck.groundpick.groundpick_env_cfg import (
    MICRODUCK_MOUTH_TIP_AXIS,
    MICRODUCK_MOUTH_TIP_OFFSET,
)
from isaaclab_tasks.contrib.microduck.roulade.roulade_env_cfg import MICRODUCK_HEAD_BODY_NAME
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR,
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import MICRODUCK_ALLCOLLISIONS_CFG, MICRODUCK_MARBLE_CFG
from isaaclab_assets.robots.microduck import MICRODUCK_MARBLE_MASS, MICRODUCK_MARBLE_RADIUS

##
# The latch (design document §1)
##

MICRODUCK_LATCH_STANDOFF = 0.005
"""Clearance [m] between the object's surface and the mouth tip when it is held.

Small and positive on purpose. At zero the spring would pull the object's surface onto the jaw shell
and fight the contact solver for the whole carry; at the shipped value the object rests *lightly*
against the shell, which is what holding something in a mouth looks like and what keeps the extra
contact in the budget profiled below.
"""

MICRODUCK_LATCH_OBJECT_MASS = MICRODUCK_MARBLE_MASS
"""Mass [kg] of the object the latch is sized for. Every spring constant below derives from it."""

MICRODUCK_LATCH_HOLD_DISTANCE = MICRODUCK_MARBLE_RADIUS + MICRODUCK_LATCH_STANDOFF
"""Mouth-tip-to-object-centre distance [m] the latch spring holds at, along the mouth axis.

Derived from the object's own radius, so a change of prop cannot leave the spring holding at a
distance the geometry no longer has.
"""

MICRODUCK_LATCH_OMEGA_DT = 0.82
"""Dimensionless stiffness of the virtual weld: ``omega * dt`` at the **control** step.

This is the one number that is chosen rather than derived, and everything else about the spring
falls out of it. It is expressed this way because of ruling R-PP17: the wrench is written once per
control step by an interval event and the wrench composer is permanent, so the force is a zero-order
hold at 50 Hz, not at the 200 Hz physics rate. The original constants were derived against the
physics step, which understated this figure by the decimation factor of four -- the shipped spring
was really at 1.03, four times past the accuracy bound -- and it rang hard enough to break its own
grip roughly every control step.

0.82 is what the measurement settled on: 138 control steps of mean grip life against 20 at the
original constants (``artifacts/microduck/pickplace/diag_latch_chatter.py``).

Note the consequence, which no choice of prop escapes: static sag is ``g / omega^2``, so it is
**mass-independent** and capped by the control rate at about 5.9 mm whatever the object weighs.
"""

MICRODUCK_LATCH_DAMPING_RATIO = 0.73
"""Damping ratio of the virtual weld, against the 0.32 that rang.

Bounded above as well as below: a zero-order-hold damper goes unstable once ``c * dt / m`` reaches 2,
which caps the ratio near 1.1 at this stiffness.
"""

MICRODUCK_LATCH_BREAK_WEIGHTS = 41.0
"""Grip strength, as a multiple of the object's own weight.

Expressed as a ratio rather than a force because the transient the grip has to survive is
``m * a`` -- it scales with the object, and so must the limit. **Ruling R-PP17 was partly this
mistake**: 2 N was set against the ball's weight and never against the transient a carry produces,
which measured 2.74 N, so the limit sat *below* normal working load. 41x weight is what the corrected
6 N was for the ball, carried across as the invariant.
"""

MICRODUCK_LATCH_STIFFNESS = MICRODUCK_LATCH_OBJECT_MASS * (MICRODUCK_LATCH_OMEGA_DT / 0.02) ** 2
"""Spring constant [N/m] of the virtual weld, derived from the object's mass and the control step.

**Derived, not chosen** -- which is the actual repair for R-PP17. A hand-picked stiffness is only
correct for the prop it was picked against, and this task has now changed prop once.
"""

MICRODUCK_LATCH_DAMPING = (
    2.0 * MICRODUCK_LATCH_DAMPING_RATIO * (MICRODUCK_LATCH_STIFFNESS * MICRODUCK_LATCH_OBJECT_MASS) ** 0.5
)
"""Damping coefficient [N.s/m] of the virtual weld, derived from the ratio above."""

MICRODUCK_LATCH_BREAK_FORCE = MICRODUCK_LATCH_BREAK_WEIGHTS * MICRODUCK_LATCH_OBJECT_MASS * 9.81
"""Force [N] above which the grip gives way and the object is dropped.

The grip is force-**limited**, not force-clamped, and that is the anti-exploit (ruling R-PP2): a
clamped constraint is a winch, and a policy that found one would drag the object through the scene
rather than carry it. Derived from the object's weight so the ratio, which is the physical invariant,
survives a change of prop.
"""

MICRODUCK_LATCH_RADIUS = MICRODUCK_MARBLE_RADIUS + 0.020
"""Mouth-tip-to-centre distance [m] within which the object can be picked up.

The object's radius plus 20 mm, so the mouth tip has to be within 20 mm of the object's
*surface*. It is the knob the pick difficulty is tuned with, and it is deliberately not on a
curriculum: what the curriculum widens is where the object is, not how accurately it must be met.
"""

MICRODUCK_LATCH_MAX_REL_SPEED = 0.30
"""Relative speed [m/s] above which the object is moving too fast to be caught.

Without it the policy could farm the latch bonus by swinging its head through the object, which is
the cheapest motion that satisfies a distance gate.
"""

##
# The placement (design document §5.5)
##

MICRODUCK_PLACE_TOLERANCE = 0.05
"""Planar distance [m] from the drop point within which the object is released.

The release edge **is** the success edge: a release cannot fire away from the target, so "dropping
short" is not a thing the policy can be paid for -- it can only break the latch, which pays nothing.
"""

MICRODUCK_PLACE_CLEARANCE = 0.025
"""How close the object's *surface* must get to the floor before a release counts [m]."""

MICRODUCK_PLACE_MAX_HEIGHT = MICRODUCK_MARBLE_RADIUS + MICRODUCK_PLACE_CLEARANCE
"""Object-centre height [m] above the ground below which the release may fire.

Placing is setting the object down, not dropping it from head height. **Derived from the prop**: the
literal 0.06 this replaced was sized against a 35 mm-radius ball, where it meant "surface within
25 mm of the floor", and on a 6 mm marble the same number would have meant 54 mm -- loose enough to
count a marble still held at head height as placed. Caught by the scripted acceptance test on the
prop change, which is what that test is for.
"""

MICRODUCK_CARRY_HEIGHT = 0.045
"""Object-centre height [m] the carry-clearance reward peaks at.

One centimetre above where the object rests on the ground, which is a lift the neck can hold through
a walk rather than a display of strength.
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_MOUTH_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_HEAD_BODY_NAME])
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
# no ``preserve_order``: the material randomization resolves body IDs into backend shape ranges and
# documents that callers must not pre-swizzle them
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")
_FEET_GROUND_SENSOR_CFG = SceneEntityCfg("feet_ground_contact")
_HEAD_IMPACT_SENSOR_CFG = SceneEntityCfg("head_impact_contact")
_OBJECT_CFG = SceneEntityCfg("object")

_TERRAIN_PRIM_PATH = "/World/ground"

_FOOT_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?(left_foot_collision|right_foot_collision)/[^/]*"
)
"""Shape-level expression selecting the two sole colliders, as on every sibling that senses them."""

_HEAD_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?(top_head_shell_1|jaw_1|bottom_head_shell_1)/[^/]*"
)
"""Shape-level expression selecting the three head shells, the ground-pick task's ``neck`` subtree."""

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle, the family's velocity-matched value."""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency (reference section 8)."""

_IMU_MAX_LAG = 1
"""Worst-case IMU latency [control steps].

The velocity recipe's post-audit value, measured against the real Dynamixel IMU path at a +/-20 ms
envelope. The ground-pick task's 3 is a transcribed pre-audit number this task has no reason to
inherit.
"""


def _iterations(count: int) -> int:
    """Convert a PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Physics preset
##


@configclass
class MicroDuckPickPlacePhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck pick-and-place environment.

    MJWarp is the only backend offered, as on every sibling: the environment sets
    ``sim.use_newton_actuators = True`` and the solver-hosted BAM model is rejected on the PhysX
    family's host adapter. See
    :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.MicroDuckPhysicsCfg`.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Measured, not inherited, and **re-measured after the prop changed**. Profiling under
            # random actions -- the regime where the robots collapse onto the floor and grind every
            # collider and the object into it, with both fall terminations dropped and the pushes
            # forced to full magnitude -- peaks at **30 contacts and 94 constraints** per
            # environment at 4096. Log:
            # ``artifacts/microduck/profile_microduck_contacts_pickplace_marble20_4096envs.log``.
            #
            # ==================  =========  =========  ===========
            # prop                peak ncon  peak nefc  median ncon
            # ==================  =========  =========  ===========
            # 70 mm ball                 32         90           27
            # 20 mm marble               30         94           25
            # ==================  =========  =========  ===========
            #
            # Note the constraint peak went **up** while the contact peak went down: a smaller prop
            # is not uniformly cheaper, so a prop change re-measures rather than assuming the old
            # budget covers it.
            #
            # ``njmax`` is a hard per-environment cap and the margin is now the tight one: 128
            # against 94 is 34 constraints, about eight further pyramidal contacts' worth, which is
            # the floor the test asserts. ``nconmax`` is a per-environment *share* of one shared
            # pool rather than a cap, so it sits just above the peak.
            njmax=128,
            nconmax=40,
            # the mjlab template's flat solver profile, which every sibling inherits unchanged
            iterations=10,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        # the family steps MuJoCo once per 0.005 s physics tick (reference section 1)
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MicroDuckPickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene with the all-collisions MicroDuck robot, an object to carry, and a ground plane.

    The object is declared after the robot because the reset event that places it reads the robot's
    settled pose, and it is named ``object`` rather than ``ball`` because the camera migration keys
    off that name: what v2 replaces is the *object* observation, whatever prop is in the scene.
    """

    terrain = TerrainImporterCfg(
        prim_path=_TERRAIN_PRIM_PATH,
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # neutral, so the friction the robot pushes off is the 1.0 the MJCF authors on the foot
            # soles and the foot-friction randomization scales
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # The all-collisions model is **required** here, not preferred. The whole task funnels through
    # putting the head on the floor next to an object, and on the walking model ``jaw_soft`` carries
    # no collider at all: the mouth would pass through the object it is supposed to press on, and
    # ``head_impact_penalty`` would read zero forever.
    robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The prop, reused rather than authored (ruling R-PP4): at 15 g it sits inside the 10-40 g
    # mouth-payload band the ground-pick task already validates at the same attachment point, and its
    # mass, hollow-shell inertia, collision and material are pinned by the asset's own fidelity
    # tests. Its position here is only what it has before the first reset.
    object = MICRODUCK_MARBLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

    # Feet and trunk, as every task in the family does. This one feeds the three critic foot terms.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    # "Both soles on the floor", filtered against the terrain specifically. Filtering is load-bearing
    # on this task in the way it is on the ground-pick one and then some: a robot folded far enough
    # to put its mouth on an object has its own shins within reach of its soles, *and* there is a
    # 70 mm ball rolling around at sole height for an unfiltered sensor to call ground.
    feet_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_FOOT_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
    )

    # The head shells against the **terrain only**, which is the whole point of this sensor here:
    # pressing the mouth onto the object must be free, and face-planting into the floor must be
    # charged. An unfiltered head sensor would make the two indistinguishable and price the pick out.
    head_impact_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_HEAD_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
    )

    # Self-collision sensor, identical to the rest of the family's: the model's ten enabled colliders
    # against each other, many-to-many. The reward saturates it back to a 0-or-1 scale.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    One term, and no velocity command anywhere: the robot decides its own gait to reach the object
    and the drop point, so there is nothing to steer it with. That is the structural difference from
    the walking family, where the twist slot is the task.

    The ranges are the *first* curriculum stage rather than the widest, because
    :func:`~isaaclab_tasks.contrib.microduck.mdp.curriculums.command_range_stages` replaces them
    from its own table before the first episode is scored.
    """

    place_target = mdp.PickPlaceTargetCommandCfg(
        asset_name="robot",
        object_name="object",
        # resampled at reset only: a drop point that moved mid-episode would be a different task
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=((0.10, 0.20), (-math.radians(30.0), math.radians(30.0))),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    The family's 14 servos, in the deploy order, at unit scale, closing the encoder-bias loop --
    unchanged from every sibling. In particular there is **no release action** (ruling R-PP3): the
    release is geometric, so the action vector stays the hardware contract it is everywhere else.
    """

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    **The actor layout is this task's camera contract.** Every object and target quantity is its own
    term; none of them is folded into a robot-state term. A v2 that replaces
    :attr:`PolicyCfg.object_position` with a camera-derived pose term changes nothing else in the
    MDP -- :attr:`PolicyCfg.target_position` is a command and :attr:`PolicyCfg.latched` is the
    robot's own controller state, so neither needs perceiving.

    The actor is 55 wide and is deliberately *not* padded to the walking family's 61-wide deploy
    contract; see the module docstring.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: proprioception, the object, the goal, the latch.

        The five proprioceptive terms and their corruptions are the velocity task's, term for term:
        the encoder bias, the IMU mounting misalignment and the bus latency model the hardware this
        policy would deploy to, and they do not become less true because the task changed. See
        :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.ObservationsCfg`.
        """

        base_ang_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.base_ang_vel_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": _IMU_MAX_LAG,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        projected_gravity = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.projected_gravity_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": _IMU_MAX_LAG,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_biased,
            params={"asset_cfg": _SERVO_JOINT_CFG, "biased": True},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.joint_vel_rel,
                "term_params": {"asset_cfg": _SERVO_JOINT_CFG},
                "min_lag": 1,
                "max_lag": 1,
                "update_period": 0,
            },
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.last_action)
        # The camera surface. In the robot's own base frame -- the frame a head-mounted camera
        # reports in -- and carrying the same positional noise a perception stack would, so a policy
        # trained here is not relying on millimetre-exact object state it will never have.
        object_position = ObsTerm(
            func=mdp.object_pos_in_base,
            params={"asset_cfg": _OBJECT_CFG},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        # A command, not a percept, so it stays exactly as it is in v2.
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "place_target"})
        # The robot's own controller state, so a deployed runtime knows it without sensing anything.
        latched = ObsTerm(func=mdp.pickplace_latched_flag)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function.

        The actor's terms with every corruption and delay removed, plus what the robot has no sensor
        for: its own base velocity, the three foot terms, the object's velocity, the latch geometry,
        and whether the episode has already succeeded.

        The two sensor-derived foot terms are the NaN-guarded variants, as on every sibling.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        object_position = ObsTerm(func=mdp.object_pos_in_base, params={"asset_cfg": _OBJECT_CFG})
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "place_target"})
        latched = ObsTerm(func=mdp.pickplace_latched_flag)
        foot_air_time = ObsTerm(func=mdp.foot_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        object_velocity = ObsTerm(func=mdp.object_vel_in_base, params={"asset_cfg": _OBJECT_CFG})
        # The latch geometry, so the value function can see a pick-up coming instead of inferring it
        # from the object position and fourteen joint angles.
        mouth_to_object = ObsTerm(
            func=mdp.mouth_to_object_in_base,
            params={
                "asset_cfg": _MOUTH_BODY_CFG,
                "object_cfg": _OBJECT_CFG,
                "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            },
        )
        succeeded = ObsTerm(func=mdp.pickplace_succeeded_flag)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events.

    The domain-randomization suite is the velocity family's, term for term and range for range, so a
    policy trained here meets the same hardware spread as one trained to walk. What this task adds is
    the object placement and the latch machinery.

    **Declaration order is behaviour.** Isaac Lab fires reset events in declaration order, and three
    of the terms below depend on it:

    * :attr:`set_ground_state` overwrites the height and orientation :attr:`reset_base` wrote.
    * :attr:`reset_object` reads the *settled* robot pose to place the object in its yaw frame.
      Ahead of the ground-state reset it would place the object at a heading the robot no longer
      has -- and with a uniformly random yaw drawn there, that is behind the robot half the time.
    * :attr:`reset_latch` must follow :attr:`reset_object`: a latch that survived the placement
      would spring-load the newly placed object back toward a mouth it is no longer near.

    There is a fourth ordering dependency that is *not* expressed here and cannot be, because it is
    between managers rather than between terms: the drop-point command reads the object's placed
    pose, which works because Isaac Lab applies reset events before it resets the command manager.
    :class:`~isaaclab_tasks.contrib.microduck.mdp.commands.PickPlaceTargetCommand` documents it.
    """

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": _FOOT_MATERIAL_CFG,
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.015, 0.015)},
    )

    mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    ##
    # Reset: redrawn every episode, in this order.
    ##

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )

    # Non-zero, as on the ball-kick task and for its reason: this policy would be handed off from a
    # walking or standing one at deployment, so it has to start from a stand it did not choose.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )

    # Standing-only, exactly as the ball-kick task uses it: this reuses the stand-up task's
    # ground-state machinery purely for its noisy upright spawn. The three ground buckets are off, so
    # their bands and keyframe are left unconfigured rather than filled with values nothing samples.
    set_ground_state = EventTerm(
        func=mdp.reset_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "standing_z_range": (0.11, 0.12),
            # spelled ``sitting_tilt_max`` upstream and applied to the standing bucket too, so at
            # ``standing_prob = 1.0`` it is the +/-5 degrees of pitch and roll the stand spawns with
            "sitting_tilt_max": math.radians(5.0),
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )

    # Must follow ``set_ground_state``: the object is placed in the robot's *reset* yaw frame. The
    # ranges here are the first curriculum stage; ``object_range`` replaces them from its own table.
    reset_object = EventTerm(
        func=mdp.reset_object_in_reach,
        mode="reset",
        params={
            "distance_range": (0.06, 0.12),
            "bearing_range": (-math.radians(20.0), math.radians(20.0)),
            "object_radius": MICRODUCK_MARBLE_RADIUS,
            "asset_cfg": _OBJECT_CFG,
        },
    )

    # Must follow ``reset_object``; see the class docstring.
    reset_latch = EventTerm(func=mdp.reset_pickplace_latch, mode="reset")

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _HEAD_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    ##
    # Interval.
    ##

    # Every control step, for every environment: a zero-width interval is how Isaac Lab spells "each
    # step". This is the task's entire grasp mechanism and it lives here, where Isaac Lab writes
    # state, for the reason the ground-pick task's payload does -- a zero-weight reward that is
    # really load-bearing physics is what a later cleanup deletes.
    update_latch = EventTerm(
        func=mdp.update_pickplace_latch,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": _MOUTH_BODY_CFG,
            "object_cfg": _OBJECT_CFG,
            "command_name": "place_target",
            "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            "mouth_axis_b": MICRODUCK_MOUTH_TIP_AXIS,
            "hold_distance": MICRODUCK_LATCH_HOLD_DISTANCE,
            "latch_radius": MICRODUCK_LATCH_RADIUS,
            "max_rel_speed": MICRODUCK_LATCH_MAX_REL_SPEED,
            "stiffness": MICRODUCK_LATCH_STIFFNESS,
            "damping": MICRODUCK_LATCH_DAMPING,
            "break_force": MICRODUCK_LATCH_BREAK_FORCE,
            "place_tolerance": MICRODUCK_PLACE_TOLERANCE,
            "place_max_height": MICRODUCK_PLACE_MAX_HEIGHT,
        },
    )

    # Half the walking family's magnitude, the ground-pick task's value, and for its reason: a robot
    # folded over an object with its centre of mass forward cannot absorb +/-0.3 m/s. Ramped from
    # nothing by ``push_magnitude``, because a robot still learning to walk cannot also be shoved.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Four blocks gated by the *latch state* rather than by a clock, so the phases emerge from what the
    robot has achieved:

    * **Approach** -- ``approach_progress``, ``mouth_to_object``, ``mouth_down``. Live until the
      object is in the mouth or already placed.
    * **Pick** -- ``latch_bonus``, one-shot on the edge.
    * **Carry** -- ``carry_hold``, ``carry_progress``, ``object_clearance``. Live while the object is
      held.
    * **Place** -- ``place_success`` and ``place_precision``, one-shot on the release edge.

    plus an ungated posture floor and the family's regularizers.

    Three of the weights are load-bearing and are not taste:

    * **``carry_hold`` at 4.0 is set by the reward-hacking audit** (design document §8, ruling
      R-PP6). Hovering the mouth at the object without latching pays ``mouth_to_object``'s 3.0 a
      step, and that term is gated off the moment the object is latched -- so at any carry bonus
      below 3.0 *refusing to pick the object up* is strictly dominant. This is the most fragile
      number in the stack.
    * **``upright`` at 0.2 is deliberately weak**, the ground-pick task's value and its reason: the
      pick requires a deep forward fold, so a strong always-on uprightness reward prices the task
      out.
    * **Both distance terms are potential-based, not Gaussians on distance** (ruling R-PP7). A
      closed path sums to exactly zero and standing anywhere pays exactly zero, so there is no range
      at which loitering is profitable.

    Two terms the ground-pick task carries are deliberately absent. ``feet_flat`` is ungated there
    because that gesture has no swing phase; this task has to walk, and an ungated flat-foot penalty
    charges every step it takes. And there is no explicit return-to-stand block: after a placement
    every task term is zero and the posture floor is all that is left, which is the return incentive
    without a second objective competing with the first.
    """

    ##
    # Approach: get there, and put your mouth on it.
    ##

    approach_progress = RewTerm(
        func=mdp.pickplace_approach_progress,
        weight=2000.0,
        params={"object_cfg": _OBJECT_CFG},
    )
    mouth_to_object = RewTerm(
        func=mdp.pickplace_mouth_to_object,
        weight=1.0,
        params={
            "asset_cfg": _MOUTH_BODY_CFG,
            "object_cfg": _OBJECT_CFG,
            "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            "std": 0.05,
        },
    )
    # Signed, the ground-pick task's shape: a mouth pointing up next to the object is charged rather
    # than merely unpaid, because reaching an object mouth-up is a different, useless posture.
    mouth_down = RewTerm(
        func=mdp.pickplace_mouth_down,
        weight=0.5,
        params={
            "asset_cfg": _MOUTH_BODY_CFG,
            "object_cfg": _OBJECT_CFG,
            "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            "mouth_axis_b": MICRODUCK_MOUTH_TIP_AXIS,
            "std": 0.15,
        },
    )

    ##
    # Pick.
    ##

    latch_bonus = RewTerm(func=mdp.pickplace_latch_bonus, weight=1000.0)

    ##
    # Carry.
    ##

    carry_hold = RewTerm(func=mdp.pickplace_carry_hold, weight=1.5)
    carry_progress = RewTerm(
        func=mdp.pickplace_carry_progress,
        weight=3000.0,
        params={"command_name": "place_target", "object_cfg": _OBJECT_CFG},
    )
    object_clearance = RewTerm(
        func=mdp.pickplace_object_clearance,
        weight=0.5,
        params={"asset_cfg": _OBJECT_CFG, "target_height": MICRODUCK_CARRY_HEIGHT, "std": 0.03},
    )

    ##
    # Place.
    ##

    place_success = RewTerm(func=mdp.pickplace_place_success, weight=5000.0)
    place_precision = RewTerm(
        func=mdp.pickplace_place_precision,
        weight=2500.0,
        params={"command_name": "place_target", "object_cfg": _OBJECT_CFG, "std": MICRODUCK_PLACE_TOLERANCE},
    )

    ##
    # The posture floor: whatever you are doing, stay on your feet.
    ##

    upright = RewTerm(
        func=mdp.upright,
        weight=0.2,
        params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG},
    )
    # Weaker than the ground-pick task's 3.0, because that task has no swing phase and this one has
    # to walk: a strong both-soles-down reward is a standing-still reward on a locomotion task.
    feet_grounded = RewTerm(
        func=mdp.feet_grounded_reward,
        weight=0.5,
        params={"sensor_cfg": _FEET_GROUND_SENSOR_CFG},
    )
    # Terrain-filtered, so this charges face-planting into the floor and never charges pressing the
    # mouth onto the object. ``saturate`` is not used: this is a force, and its magnitude is the
    # signal.
    head_impact_penalty = RewTerm(
        func=mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_cfg": _HEAD_IMPACT_SENSOR_CFG, "threshold": 1.0},
    )
    # ``saturate`` keeps the many-to-many sensor on the family's 0-or-1 scale, so the weight is the
    # penalty for touching yourself at all rather than a per-collider tariff.
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG, "saturate": True},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # **Falling has to be priced, not merely terminated** (ruling R-PP19). Ending an episode is only
    # a cost when the rest of the episode was worth something, and this stack's mass sits in a
    # one-shot delivery bonus the policy can collect in the first second -- so a fall cost nothing,
    # and the second training run duly learned to grab the object, dive at the drop point and topple.
    # It delivered on 93 % of episodes, in 40 control steps, with ``upright`` at 0.0003.
    #
    # Keyed to the two fall terms only. ``nan_state`` is deliberately excluded: a diverged solver is
    # not a policy decision and charging for it teaches avoidance of nothing. The stock term *sums*
    # its keys, so an episode that trips both fall gates on the same step is charged twice; that is
    # rare, and it only makes falling more expensive, which is the direction of travel.
    fell_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-5000.0,
        params={"term_keys": ["fell_over", "fell_low"]},
    )

    ##
    # Regularization, the velocity family's and lighter than the ground-pick task's.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    # Ramped to -0.6 by iteration 1500, where the ground-pick task ramps to -2.0. That task is a slow
    # quasi-static gesture; this one has a locomotion sub-problem, and the family's own stand-up
    # experience is that heavy smoothness regularization blocks a dynamic motion outright.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5e-3, params={"asset_cfg": _SERVO_JOINT_CFG})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Falling is failure here rather than a phase to be recovered from, so it terminates immediately
    and there is no ``fallen_too_long`` clock.

    **The fall gate is tilt and height, as two terms.** The family's velstand investigation found
    that tilt-only gating opened a lie-flat reward-hacking basin: a robot folded flat can stay
    nominally inside a 70-degree tilt bound while doing nothing the task asks for. Both stock terms
    exist, and each is independently testable, so they are configured separately rather than fused
    into a new one.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    fell_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.06})

    # Catches a *broken* robot rather than a fallen one, with the closed sensor list every sibling
    # port carries. This task needs the longest form of it: a reward reads a contact force directly,
    # where a single non-finite value poisons the episode sum.
    #
    # It reads the robot, as the family's does. The object is not covered by any state check on this
    # stack, which is why the object observations guard themselves.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("contact_forces", "feet_ground_contact", "head_impact_contact")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    **``object_range`` is what makes this task trainable** (ruling R-PP11). The object starts
    effectively at the mouth -- 6 to 12 cm, within 20 degrees of straight ahead, which a standing
    robot can reach by folding -- and only widens to a walk once the pick has been learned. Without
    it the policy has to discover locomotion and grasping simultaneously from a sparse latch bonus,
    which is the regime the family measured its structural blindness in.

    ``target_range`` widens second and one stage behind, so the carry is never asked to be longer
    than the approach the policy has already solved.
    """

    object_range = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "reset_object",
            "param_stages": [
                {
                    "step": _iterations(0),
                    "params": {"distance_range": (0.06, 0.12), "bearing_range": (-0.35, 0.35)},
                },
                {
                    "step": _iterations(500),
                    "params": {"distance_range": (0.10, 0.22), "bearing_range": (-0.79, 0.79)},
                },
                {
                    "step": _iterations(1000),
                    "params": {"distance_range": (0.15, 0.32), "bearing_range": (-1.22, 1.22)},
                },
                {
                    "step": _iterations(2000),
                    "params": {"distance_range": (0.15, 0.45), "bearing_range": (-1.57, 1.57)},
                },
            ],
        },
    )

    target_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "place_target",
            "range_stages": [
                {"step": _iterations(0), "ranges": ((0.10, 0.20), (-0.52, 0.52))},
                {"step": _iterations(500), "ranges": ((0.15, 0.35), (-1.57, 1.57))},
                {"step": _iterations(1500), "ranges": ((0.20, 0.60), (-3.14, 3.14))},
            ],
        },
    )

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -0.1},
                {"step": _iterations(500), "weight": -0.2},
                {"step": _iterations(1000), "weight": -0.4},
                {"step": _iterations(1500), "weight": -0.6},
            ],
        },
    )

    com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
                {"step": _iterations(1500), "range": 0.015},
            ],
        },
    )

    head_com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_head_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
            ],
        },
    )

    # ``inclusive=False`` matches the family's transcription of upstream's exclusive step comparison
    # on this helper, so a stage table written for a sibling schedules identically here.
    push_magnitude = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "push_robot",
            "inclusive": False,
            "param_stages": [
                {"step": _iterations(0), "params": {"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}},
                {"step": _iterations(500), "params": {"velocity_range": {"x": (-0.06, 0.06), "y": (-0.06, 0.06)}}},
                {"step": _iterations(1000), "params": {"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}}},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckPickPlaceFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck pick-and-place environment on flat ground.

    There is no rough variant. The object is a free-rolling sphere and the drop point is a position
    on a plane; on generated terrain the object would roll away downhill and the target would be
    underground, which is a different task rather than a harder one.
    """

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckPickPlacePhysicsCfg())
    # ``env_spacing`` is the family's scene extent, widened because this task's object and drop point
    # can be 0.45 m and 0.60 m from a robot that itself spawns anywhere in a 1 m square.
    scene: MicroDuckPickPlaceSceneCfg = MicroDuckPickPlaceSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 0.005 s physics steps decimated by 4 give the 50 Hz control rate the family deploys at.
        # The episode is the family's inherited 20 s -- 1000 control steps. At the widest curriculum
        # stage that is a 0.45 m approach, a fold, a 0.60 m carry and a placement, which is about
        # 10 s of translation at MicroDuck's walking pace: the same ~2x headroom the ground-pick
        # task's 20 s leaves its five cycles.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        # Run the BAM servos through the backend-native path, as the rest of the family does. The
        # decimation above is even, which is what lets the stateful servo delay line be
        # CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.feet_ground_contact is not None:
            self.scene.feet_ground_contact.update_period = self.sim.dt
        if self.scene.head_impact_contact is not None:
            self.scene.head_impact_contact.update_period = self.sim.dt
        if self.scene.self_collision is not None:
            self.scene.self_collision.update_period = self.sim.dt
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
