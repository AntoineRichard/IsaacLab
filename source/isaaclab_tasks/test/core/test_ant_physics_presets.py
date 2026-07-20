# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Ant physics presets."""

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

from isaaclab_tasks.core.locomotion.ant.ant_direct_env_cfg import AntPhysicsCfg


def test_ant_preserves_kamino_padmm_preset() -> None:
    """The existing Ant Kamino preset must remain the P-ADMM control."""
    physics = AntPhysicsCfg()

    assert isinstance(physics.newton_kamino, NewtonCfg)
    assert isinstance(physics.newton_kamino.solver_cfg, KaminoSolverCfg)
    assert physics.newton_kamino.solver_cfg.dynamics_solver is None


def test_ant_exposes_sparse_kamino_dvi_preset() -> None:
    """The Ant DVI preset must select the optimized sparse configuration."""
    physics = AntPhysicsCfg()
    solver = physics.newton_kamino_dvi.solver_cfg

    assert isinstance(physics.newton_kamino_dvi, NewtonCfg)
    assert isinstance(solver, KaminoSolverCfg)
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
