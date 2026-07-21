# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Fourbar Pole physics presets."""

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

from isaaclab_tasks.core.fourbar_pole.fourbar_pole_manager_env_cfg import FourbarPolePhysicsCfg


def test_fourbar_pole_preserves_kamino_padmm_default() -> None:
    """The Fourbar Pole default and control must remain task-specific P-ADMM presets."""
    physics = FourbarPolePhysicsCfg()

    assert isinstance(physics.default, NewtonCfg)
    assert isinstance(physics.newton_kamino.solver_cfg, KaminoSolverCfg)
    assert physics.newton_kamino.solver_cfg.dynamics_solver is None
    assert physics.default.solver_cfg.dynamics_solver is None


def test_fourbar_pole_exposes_closed_loop_kamino_dvi_preset() -> None:
    """The Fourbar Pole DVI preset must retain FK and use tuned DVI settings."""
    physics = FourbarPolePhysicsCfg()
    solver = physics.newton_kamino_dvi.solver_cfg

    assert isinstance(physics.newton_kamino_dvi, NewtonCfg)
    assert isinstance(solver, KaminoSolverCfg)
    assert solver.dynamics_solver == "dvi"
    assert solver.integrator == "moreau"
    assert solver.use_fk_solver is True
    assert solver.sparse_jacobian is True
    assert solver.sparse_dynamics is True
    assert solver.constraints_alpha == 0.1
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
    assert physics.newton_kamino_dvi.num_substeps == 1
    assert physics.newton_kamino_dvi.debug_mode is False
    assert physics.newton_kamino_dvi.use_cuda_graph is True
