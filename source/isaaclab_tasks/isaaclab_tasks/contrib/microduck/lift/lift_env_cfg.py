# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift environment for the Pollen Robotics MicroDuck biped.

A marble lies on the ground within reach. The robot folds down, closes its beak on it, and stands
back up holding it. That is the whole task.

**This is the behaviour the hardware ships with**, which is the argument for building it before
anything more elaborate. Pollen describe it as "beak to the floor, one button" and "dips the beak to
the ground, scoops, and pops back upright". The pick-and-place task
(:mod:`~isaaclab_tasks.contrib.microduck.pickplace`) bolted locomotion-while-carrying and a
commanded drop point onto that, and every failure it produced was a carry-or-place failure rather
than a grasp failure: a policy that farmed a chattering grip, one that dived at the target and
toppled, and one that shuffled the object along the floor on its belly. The grasp itself always
worked.

So this task is the pick-and-place stack with everything after the grab removed:

* **no drop point** -- no command term, no ``carry_progress``, no placement rewards, and the latch
  state machine's release is switched off, so the robot grabs and holds and never lets go;
* **no walking** -- the marble spawns inside the fold, so there is no approach to reward and no
  curriculum to widen;
* **no 20 s episode** -- the gesture takes about three seconds, and the pick-and-place task's long
  episode left a fifteen-second dead zone the policy filled by wandering.

What it keeps, all of it already measured and asserted: the beak variant of the robot and its
hinge, the marble sized to fit that beak, the latch state machine with every constant derived from
the prop, latch-gated uprightness, and the priced fall.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_tasks.contrib.microduck.groundpick.groundpick_env_cfg import (
    MICRODUCK_MOUTH_TIP_AXIS,
    MICRODUCK_MOUTH_TIP_OFFSET,
)
from isaaclab_tasks.contrib.microduck.pickplace.beak_env_cfg import MICRODUCK_BEAK_OPEN_DISTANCE
from isaaclab_tasks.contrib.microduck.pickplace.pickplace_env_cfg import (
    MICRODUCK_LATCH_BREAK_FORCE,
    MICRODUCK_LATCH_DAMPING,
    MICRODUCK_LATCH_HOLD_DISTANCE,
    MICRODUCK_LATCH_MAX_REL_SPEED,
    MICRODUCK_LATCH_RADIUS,
    MICRODUCK_LATCH_STIFFNESS,
)
from isaaclab_tasks.contrib.microduck.roulade.roulade_env_cfg import MICRODUCK_HEAD_BODY_NAME
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR,
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import MICRODUCK_BEAK_CFG, MICRODUCK_MARBLE_CFG
from isaaclab_assets.robots.microduck import (
    MICRODUCK_BEAK_CLOSED,
    MICRODUCK_BEAK_JOINT_NAME,
    MICRODUCK_BEAK_OPEN,
    MICRODUCK_MARBLE_RADIUS,
)

##
# The lift
##

MICRODUCK_LIFT_REACH_RANGE = (0.06, 0.15)
"""Distance [m] from the robot root the marble spawns at.

Inside the fold, so there is nothing to walk to. The mouth tip sits about 78 mm ahead of the head at
the stand pose, and the ground-pick task shows the same robot bringing it to the floor from there,
so this band is reachable by bending rather than by stepping. No curriculum widens it: taking the
walking out is the point of this task, and a range the robot must walk to would put it back.
"""

MICRODUCK_LIFT_BEARING = math.radians(25.0)
"""Half-width [rad] of the bearing the marble spawns at, from the robot's heading.

Wide enough that the robot cannot solve the task with one memorised fold, narrow enough that it
never has to turn on the spot to find it.
"""

MICRODUCK_LIFT_TARGET_HEIGHT = 0.12
"""Object-centre height [m] the lift ramp saturates at.

Roughly where the mouth tip sits with the robot standing, so a saturated ramp means the marble is up
at carrying height rather than merely off the floor. Above this there is nothing more to gain, which
is what stops the objective rewarding a throw.
"""


##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_MOUTH_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_HEAD_BODY_NAME])
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")
_FEET_GROUND_SENSOR_CFG = SceneEntityCfg("feet_ground_contact")
_HEAD_IMPACT_SENSOR_CFG = SceneEntityCfg("head_impact_contact")
_OBJECT_CFG = SceneEntityCfg("object")
_BEAK_CFG = SceneEntityCfg(
    "robot", body_names=["jaw_soft"], joint_names=[MICRODUCK_BEAK_JOINT_NAME], preserve_order=True
)

_TERRAIN_PRIM_PATH = "/World/ground"
_FOOT_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?(left_foot_collision|right_foot_collision)/[^/]*"
)
_HEAD_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?(top_head_shell_1|jaw_1|bottom_head_shell_1)/[^/]*"
)
_IMU_MISALIGNMENT_DEG = 6.0
_IMU_DELAY_UPDATE_PERIOD = 64
_IMU_MAX_LAG = 1


@configclass
class MicroDuckLiftPhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck lift environment.

    MJWarp only, as on every sibling: the environment sets ``sim.use_newton_actuators = True`` and
    the solver-hosted BAM model is rejected on the PhysX family's host adapter.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # The pick-and-place task's measured budget, which bounds this one: same robot, same
            # marble, same plane, and strictly less going on -- no walking, no second contact-rich
            # phase. It is inherited rather than re-measured on that argument, and the inheritance is
            # asserted rather than assumed; see the environment test.
            njmax=128,
            nconmax=40,
            iterations=10,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


@configclass
class MicroDuckLiftSceneCfg(InteractiveSceneCfg):
    """The beak robot, a marble, and a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path=_TERRAIN_PRIM_PATH,
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # The variant whose beak opens. On this task it is not a nicety: the whole behaviour is the beak
    # closing on something, and on every other model the jaw is welded shut.
    robot = MICRODUCK_BEAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = MICRODUCK_MARBLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )
    feet_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_FOOT_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
    )
    # Terrain-filtered, so pressing the beak onto the marble is free and only hitting the *floor* is
    # charged. On a task whose whole gesture is bringing the head to the ground, an unfiltered head
    # sensor would charge the objective.
    head_impact_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_HEAD_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
    )
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class ActionsCfg:
    """The family's 14 servos, in the deploy order. The beak is not among them."""

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observations. The camera contract of the pick-and-place task, minus the drop point.

    The actor is 51 wide: proprioception, the marble's position in the base frame, and the latch
    flag. As on the pick-and-place task the marble is its own term, expressed in the robot base
    frame, so a v2 that reads it from a camera changes one row and nothing else.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """The deployed vector: proprioception, the object, and the robot's own latch state."""

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
        object_position = ObsTerm(
            func=mdp.object_pos_in_base,
            params={"asset_cfg": _OBJECT_CFG},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        latched = ObsTerm(func=mdp.pickplace_latched_flag)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """The actor's terms uncorrupted, plus what the robot has no sensor for."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        object_position = ObsTerm(func=mdp.object_pos_in_base, params={"asset_cfg": _OBJECT_CFG})
        latched = ObsTerm(func=mdp.pickplace_latched_flag)
        foot_air_time = ObsTerm(func=mdp.foot_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        object_velocity = ObsTerm(func=mdp.object_vel_in_base, params={"asset_cfg": _OBJECT_CFG})
        mouth_to_object = ObsTerm(
            func=mdp.mouth_to_object_in_base,
            params={
                "asset_cfg": _MOUTH_BODY_CFG,
                "object_cfg": _OBJECT_CFG,
                "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Events. The velocity family's randomization, plus the marble and the two latch hooks.

    Declaration order is behaviour: :attr:`reset_object` reads the robot's settled pose, and
    :attr:`reset_latch` must follow it so a latch cannot survive into an episode whose marble has
    moved.
    """

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
    encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="startup", params={"bias_range": (-0.015, 0.015)})
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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )
    set_ground_state = EventTerm(
        func=mdp.reset_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "standing_z_range": (0.11, 0.12),
            "sitting_tilt_max": math.radians(5.0),
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )
    reset_object = EventTerm(
        func=mdp.reset_object_in_reach,
        mode="reset",
        params={
            "distance_range": MICRODUCK_LIFT_REACH_RANGE,
            "bearing_range": (-MICRODUCK_LIFT_BEARING, MICRODUCK_LIFT_BEARING),
            "object_radius": MICRODUCK_MARBLE_RADIUS,
            "asset_cfg": _OBJECT_CFG,
        },
    )
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
        func=mdp.randomize_bam_friction, mode="reset", params={"scale_range": (0.9, 1.1)}
    )

    # Every control step. ``command_name`` is left unset, which switches the release off: this task
    # has nothing to place the marble on, so once the beak closes it stays closed.
    update_latch = EventTerm(
        func=mdp.update_pickplace_latch,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": _MOUTH_BODY_CFG,
            "object_cfg": _OBJECT_CFG,
            "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            "mouth_axis_b": MICRODUCK_MOUTH_TIP_AXIS,
            "hold_distance": MICRODUCK_LATCH_HOLD_DISTANCE,
            "latch_radius": MICRODUCK_LATCH_RADIUS,
            "max_rel_speed": MICRODUCK_LATCH_MAX_REL_SPEED,
            "stiffness": MICRODUCK_LATCH_STIFFNESS,
            "damping": MICRODUCK_LATCH_DAMPING,
            "break_force": MICRODUCK_LATCH_BREAK_FORCE,
        },
    )
    drive_beak = EventTerm(
        func=mdp.drive_beak_from_latch,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": _BEAK_CFG,
            "object_cfg": _OBJECT_CFG,
            "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            "open_distance": MICRODUCK_BEAK_OPEN_DISTANCE,
            "closed_angle": MICRODUCK_BEAK_CLOSED,
            "open_angle": MICRODUCK_BEAK_OPEN,
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
    )


@configclass
class RewardsCfg:
    """Reward terms. Five task terms and a posture floor -- the smallest stack in this package.

    The whole objective is :attr:`lift`, and it is **dense**: it pays every step the marble is held
    clear of the floor, in proportion to how high. That is deliberate on two counts learned from the
    pick-and-place task. A sustained hold should be worth more than an instant, which paying per step
    gives for free; and there is no large one-shot bonus anywhere in this stack, so there is no
    reward spike of a thousand times the typical step to destabilise the advantage estimate.

    The audit that matters is in the environment test, in episode-return units: lifting must beat
    grabbing-and-not-lifting, which must beat hovering, which must beat doing nothing.
    """

    ##
    # The reach: find the marble with the beak. Gated off once it is held.
    ##

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
    # Signed, the ground-pick task's shape: reaching the marble beak-up is a different and useless
    # posture, and a term that merely failed to pay for it would leave the proximity reward free to
    # find it.
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
    # The grab and the lift.
    ##

    # One payment per episode. It is a discovery bootstrap, not a subsidy, and the pick-and-place
    # task showed what an uncapped version buys: a policy that re-grasped 444 times an episode.
    latch_bonus = RewTerm(func=mdp.pickplace_latch_bonus, weight=1000.0)
    lift = RewTerm(
        func=mdp.lift_height,
        weight=30.0,
        params={
            "asset_cfg": _OBJECT_CFG,
            "rest_height": MICRODUCK_MARBLE_RADIUS,
            "lift_height": MICRODUCK_LIFT_TARGET_HEIGHT,
        },
    )
    # Standing while holding. Weak on its own -- lifting the marble to carrying height already
    # requires standing, so this mostly breaks ties between postures that lift equally well.
    carry_upright = RewTerm(
        func=mdp.pickplace_upright_while_carrying,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG},
    )

    ##
    # The posture floor.
    ##

    # Deliberately weak, the ground-pick task's value and its reason: the reach *requires* a deep
    # forward fold, and a strong always-on uprightness reward would price the task out.
    upright = RewTerm(func=mdp.upright, weight=0.2, params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG})
    feet_grounded = RewTerm(func=mdp.feet_grounded_reward, weight=1.0, params={"sensor_cfg": _FEET_GROUND_SENSOR_CFG})
    head_impact_penalty = RewTerm(
        func=mdp.body_impact_cost, weight=-2.0, params={"sensor_cfg": _HEAD_IMPACT_SENSOR_CFG, "threshold": 1.0}
    )
    self_collisions = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG, "saturate": True}
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Falling has to be priced, not merely terminated: a termination only costs something when the
    # rest of the episode was worth something.
    fell_penalty = RewTerm(func=mdp.is_terminated_term, weight=-5000.0, params={"term_keys": ["fell_over", "fell_low"]})

    ##
    # Regularization, heavier than the pick-and-place task's.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    # The ground-pick task's argument applies here and did not apply to pick-and-place: this is a
    # slow, precise, quasi-static gesture with no gait to block, so strong smoothness helps transfer
    # rather than preventing the motion. That task ramps to -2.0; this starts where its ramp ends
    # for the first stage, without a curriculum, because there is no locomotion to learn first.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.8)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5e-3, params={"asset_cfg": _SERVO_JOINT_CFG})


@configclass
class TerminationsCfg:
    """Terminations. Falling is failure; there is no recovery phase to wait out."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    fell_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.06})
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("contact_forces", "feet_ground_contact", "head_impact_contact")},
    )


@configclass
class MicroDuckLiftFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck lift environment on flat ground."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckLiftPhysicsCfg())
    scene: MicroDuckLiftSceneCfg = MicroDuckLiftSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.sim.dt = 0.005
        # **5 s, not the family's 20.** The gesture takes about three, and the pick-and-place task's
        # long episode left a fifteen-second dead zone after the objective was met which the policy
        # filled by wandering roughly two metres. A short episode also means more resets per
        # iteration, which is the cheap way to buy sample diversity on a task with no locomotion.
        self.episode_length_s = 5.0
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        for sensor in (
            self.scene.contact_forces,
            self.scene.feet_ground_contact,
            self.scene.head_impact_contact,
            self.scene.self_collision,
        ):
            if sensor is not None:
                sensor.update_period = self.sim.dt
        # Beak scale: the marble is 20 mm on a 25 cm robot.
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.5, 0.5, 0.28), lookat=(0.0, 0.0, 0.08))
