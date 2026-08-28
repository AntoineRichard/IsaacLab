# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-tracking environment for the Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``artifacts/microduck/upstream_reference.md``, the extraction of the pinned upstream checkouts.

This is the terrain-generator variant of the task; :mod:`.flat_env_cfg` derives the flat one.
Rewards, commands, events and observations here are the subset that upstream's own base template
provides. The MicroDuck-specific head-pose and body-pose commands, the sensor-driven foot rewards
and the domain-randomization suite are added on top of this skeleton.
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
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import MICRODUCK_CFG

##
# Shared constants
##

MICRODUCK_JOINT_NAMES = [
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
"""The 14 servo joints in upstream MJCF order: left leg, then neck/head, then right leg.

Reference section 6b. This order is a deploy contract, not a convenience: the policy trained here
is run on the robot against a flat observation vector whose joint blocks follow the MJCF actuator
indices, and Isaac Lab resolves joints in USD order, which differs. Every joint-block observation
and the action term therefore pin it with ``preserve_order=True``.

Spelling the joints out also reproduces upstream's ``^(?!passive_).*`` selector: name resolution
matches in full, so an exact name can never pick up a ``passive_`` joint. The walk model has none
(reference section 6b), but the roller and backlash variants of the robot do.
"""

MICRODUCK_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot bodies in upstream's ``[left, right]`` site order (reference section 2.9)."""

MICRODUCK_TRUNK_BODY_NAME = "trunk_base"
"""Body upstream measures the base pose, upright reward and self-collisions on."""

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)


##
# Physics presets
##


@configclass
class MicroDuckPhysicsCfg(PresetCfg):
    """Backend presets for the MicroDuck velocity environments.

    MicroDuck is trained on MuJoCo upstream, so MJWarp is the reference backend. The solver limits
    are upstream's rough-terrain profile (reference section 1): ``njmax`` 1500, ``nconmax`` 200,
    30 solver iterations and 50 line-search iterations. The flat variant tightens them.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1500,
            nconmax=200,
            iterations=30,
            ls_iterations=50,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        # upstream steps MuJoCo once per 0.005 s physics tick, so the substep count stays at one
        # and ``sim.dt`` is the upstream timestep (reference section 1)
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MicroDuckSceneCfg(InteractiveSceneCfg):
    """Scene with the MicroDuck robot on a ground plane."""

    # Placeholder terrain: upstream's rough profile uses a terrain generator, which is wired up
    # together with the terrain-level curriculum.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # neutral, so the friction the robot walks on is the 1.0 the MJCF authors on the foot
            # soles and the foot-friction randomization scales
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Feet and trunk only: upstream tracks ground contact on the two soles and self-collision on
    # the trunk subtree (reference section 2.9). No height scanner -- upstream drops the terrain
    # scan and the matching observation for MicroDuck (reference sections 2.3 and 2.9).
    #
    # The pattern is matched against whole body paths, and the MJCF import nests every link under
    # the trunk, so the ankle links the soles hang off are reached through a wildcard rather than
    # as direct children.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
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
    """Command specifications for the MDP."""

    # Reference section 2.7. The resampling range, heading controller and heading range are
    # inherited from upstream's base template; the velocity ranges and the standing fraction are
    # MicroDuck's own. ``rel_heading_envs = 0.0`` leaves the heading controller configured but
    # inert, which is what upstream ships.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Reference sections 3 and 9: the target is ``default_joint_pos + action * scale``, and
    # MicroDuck raises the base template's 0.5 scale to 1.0, so an action is a joint-space offset
    # from the stand pose in radians.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group.

        Term order is the deployed observation layout (reference section 7), which the runtime
        rebuilds by hand: base angular velocity, projected gravity, joint positions, joint
        velocities, the previous action, then the command block. The head-pose and body-pose
        commands that complete the 61-wide contract are appended after the twist command.

        Noise magnitudes are MicroDuck's own overrides of the base template (reference section
        2.3). The IMU misalignment, observation delays and encoder bias upstream also applies are
        not modelled yet.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _SERVO_JOINT_CFG},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _SERVO_JOINT_CFG},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Configuration for events.

    Reference section 2.6, restricted to the reset and push terms of upstream's base template.
    Upstream's ``base_com`` term is deliberately absent: it selects zero bodies upstream and
    MicroDuck never fills it in (reference section 2.6a). The startup randomization MicroDuck adds
    on top -- foot friction, encoder bias, centre of mass, mass and inertia, armature and joint
    friction -- is not part of this skeleton.
    """

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Upstream samples the absolute base height in (0.12, 0.13) m; Isaac Lab samples an
            # offset from the configured default, which is the 0.125 m midpoint of that band.
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

    # Zero-width on purpose: upstream resets every joint exactly to the stand pose.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    The four terms of upstream's recipe that map onto stock Isaac Lab rewards (reference section
    2.4). The gait, foot-clearance, posture, self-collision and pose-tracking terms upstream also
    uses need MicroDuck-specific implementations and are added later.
    """

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    # Stand-in for upstream's ``upright`` term, a Gaussian on the trunk's gravity tilt with weight
    # 2.0 and ``std = sqrt(0.05)``. The stock term is an L2 penalty on the same quantity, so the
    # magnitude carries over but the sign flips.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # Upstream starts here and ramps the weight to -1.0 by iteration 1500 through a curriculum.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Reference section 2.5. Upstream's ``out_of_terrain_bounds`` term arrives with the terrain
    generator, and its NaN guard is a MicroDuck-specific term. Upstream terminates on tilt rather
    than on trunk contact.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})


##
# Environment configuration
##


@configclass
class MicroDuckVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck velocity-tracking environment on rough terrain."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckPhysicsCfg())
    # ``env_spacing`` is upstream's scene extent (reference section 1).
    scene: MicroDuckSceneCfg = MicroDuckSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization."""
        # 0.005 s physics steps decimated by 4 give the 50 Hz control rate the deployed policy runs
        # at, and 20 s episodes are 1000 control steps (reference section 1).
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
