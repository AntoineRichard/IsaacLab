# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, PhoenXSolverCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=95,
            nconmax=10,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_phoenx = NewtonCfg(
        solver_cfg=PhoenXSolverCfg(
            # Validated tuning from newton_2's example_robot_anymal_c_walk and
            # example_robot_h1: 4 internal substeps + 8 PGS iterations track
            # MuJoCo's PD response to ~0.01 rad RMS at outer dt=5ms.
            substeps=4,
            solver_iterations=8,
            velocity_iterations=1,
            # ``substep_end`` was empirically the best of the three modes on
            # the G1 standing pose: ``finite_difference`` and
            # ``substep_average`` both invert the trunk-pitch direction
            # relative to MuJoCo Warp, which turns the policy's
            # projected-gravity observation into an off-distribution signal.
            # See newton_2/newton/examples/robot/example_robot_policy.py.
            velocity_readout="substep_end",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # ``feet_air_time`` is dropped by the parent post-init when the active
        # physics preset has no contact sensor (``newton_phoenx``).
        if self.rewards.feet_air_time is not None:
            self.rewards.feet_air_time.weight = 0.75
            self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
