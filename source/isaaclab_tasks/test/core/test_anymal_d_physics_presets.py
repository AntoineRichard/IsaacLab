# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ANYmal-D Flat physics presets."""

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import PhysicsCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import VelocityEnvContactSensorCfg


def test_anymal_d_flat_exposes_kamino_benchmark_preset():
    """ANYmal-D Flat must expose the approved common Kamino configuration."""
    physics = PhysicsCfg()

    assert isinstance(physics.newton_kamino, NewtonCfg)
    assert isinstance(physics.newton_kamino.solver_cfg, KaminoSolverCfg)
    assert physics.newton_kamino.solver_cfg.max_contacts_per_world == 64
    assert physics.newton_kamino.num_substeps == 1
    assert physics.newton_kamino.debug_mode is False


def test_anymal_d_flat_exposes_sparse_kamino_dvi_preset() -> None:
    """ANYmal-D Flat must expose the tuned DVI preset with its contact capacity."""
    physics = PhysicsCfg()
    solver = physics.newton_kamino_dvi.solver_cfg

    assert isinstance(physics.newton_kamino_dvi, NewtonCfg)
    assert isinstance(solver, KaminoSolverCfg)
    assert solver.max_contacts_per_world == 64
    assert solver.dynamics_solver == "dvi"
    assert solver.integrator == "moreau"
    assert solver.sparse_jacobian is True
    assert solver.sparse_dynamics is True
    assert solver.dynamics_preconditioning is False
    assert solver.dynamics_linear_solver_type == "CR"
    assert solver.dynamics_linear_solver_max_iterations == 9
    assert solver.dvi_block_iterations == 16
    assert solver.dvi_contact_iterations == 2
    assert solver.dvi_bilateral_solve_period == 2
    assert solver.dvi_omega == 0.3
    assert solver.dvi_contact_jacobi_omega == 0.45
    assert solver.dvi_contact_jacobi_relaxation == 0.9


def test_anymal_d_flat_preserves_existing_physics_presets():
    """Adding Kamino must not replace the existing ANYmal-D Flat presets."""
    physics = PhysicsCfg()

    assert isinstance(physics.default, PhysxCfg)
    assert isinstance(physics.physx, PhysxCfg)
    assert isinstance(physics.newton_mjwarp, NewtonCfg)
    assert isinstance(physics.newton_mjwarp.solver_cfg, MJWarpSolverCfg)
    assert isinstance(physics.ovphysx, OvPhysxCfg)


def test_anymal_d_flat_uses_newton_contact_sensor_for_kamino_dvi() -> None:
    """ANYmal-D Flat must use its existing Newton sensor for the DVI preset."""
    sensors = VelocityEnvContactSensorCfg()

    assert isinstance(sensors.newton_kamino_dvi, NewtonContactSensorCfg)
    assert sensors.newton_kamino_dvi == sensors.newton_kamino
