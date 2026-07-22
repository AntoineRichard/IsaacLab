# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DR Legs physics presets."""

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

from isaaclab_tasks.contrib.dr_legs.hold_pose_env_cfg import DrLegsHoldPoseEnvCfg, DrLegsPhysicsCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_dr_legs_preserves_kamino_padmm_default() -> None:
    """The DR Legs default and control must remain task-specific P-ADMM presets."""
    physics = DrLegsPhysicsCfg()

    assert isinstance(physics.default, NewtonCfg)
    assert isinstance(physics.newton_kamino.solver_cfg, KaminoSolverCfg)
    assert physics.newton_kamino.solver_cfg.dynamics_solver is None
    assert physics.default.solver_cfg.dynamics_solver is None


def test_dr_legs_exposes_closed_loop_kamino_dvi_preset() -> None:
    """The DR Legs DVI preset must preserve task settings and use tuned DVI."""
    physics = DrLegsPhysicsCfg()
    solver = physics.newton_kamino_dvi.solver_cfg

    assert isinstance(physics.newton_kamino_dvi, NewtonCfg)
    assert isinstance(solver, KaminoSolverCfg)
    assert solver.dynamics_solver == "dvi"
    assert solver.integrator == "moreau"
    assert solver.use_collision_detector is False
    assert solver.use_fk_solver is True
    assert solver.sparse_jacobian is True
    assert solver.sparse_dynamics is True
    assert solver.constraints_alpha == 0.1
    assert solver.max_contacts_per_world == 32
    assert solver.dynamics_preconditioning is False
    assert solver.dynamics_linear_solver_type == "CR"
    assert solver.dynamics_linear_solver_max_iterations == 9
    assert solver.dvi_block_iterations == 16
    assert solver.dvi_contact_iterations == 2
    assert solver.dvi_bilateral_solve_period == 2
    assert solver.dvi_omega == 0.3
    assert solver.dvi_contact_jacobi_omega == 0.45
    assert solver.dvi_contact_jacobi_relaxation == 0.9
    assert solver.dvi_contact_block_preconditioner is False
    assert solver.dvi_warmstart_mode == "containers"
    assert physics.newton_kamino_dvi.use_cuda_graph is True
    assert physics.newton_kamino_dvi.default_shape_cfg == physics.newton_kamino.default_shape_cfg


def test_dr_legs_dvi_resolves_stable_closed_loop_reset_profile() -> None:
    """DVI must start assembled while retaining base-velocity and startup randomization."""
    env_cfg = resolve_presets(DrLegsHoldPoseEnvCfg(), selected=("newton_kamino_dvi",))

    assert env_cfg.sim.physics.use_cuda_graph is True
    assert set(env_cfg.events.reset_base.params["pose_range"].values()) == {(0.0, 0.0)}
    assert env_cfg.events.reset_robot_joints.params["position_range"] == (0.0, 0.0)
    assert env_cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["velocity_range"]["x"] == (-0.2, 0.2)
    assert env_cfg.events.randomize_joint_params is not None
    assert env_cfg.events.randomize_actuator_gains is not None
    assert env_cfg.events.physics_material is not None


def test_dr_legs_padmm_preserves_randomized_reset_profile() -> None:
    """The DVI-specific reset must not change the P-ADMM task distribution."""
    env_cfg = resolve_presets(DrLegsHoldPoseEnvCfg(), selected=("newton_kamino",))

    assert env_cfg.events.reset_base.params["pose_range"] == {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (0.01, 0.015),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-3.14159, 3.14159),
    }
    assert env_cfg.events.reset_robot_joints.params["position_range"] == (-0.1, 0.1)
    assert env_cfg.events.reset_robot_joints.params["velocity_range"] == (-0.05, 0.05)
