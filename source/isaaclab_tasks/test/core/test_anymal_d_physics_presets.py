# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ANYmal-D Flat physics presets."""

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import PhysicsCfg


def test_anymal_d_flat_exposes_kamino_benchmark_preset():
    """ANYmal-D Flat must expose the approved common Kamino configuration."""
    physics = PhysicsCfg()

    assert isinstance(physics.newton_kamino, NewtonCfg)
    assert isinstance(physics.newton_kamino.solver_cfg, KaminoSolverCfg)
    assert physics.newton_kamino.solver_cfg.max_contacts_per_world == 64
    assert physics.newton_kamino.num_substeps == 1
    assert physics.newton_kamino.debug_mode is False


def test_anymal_d_flat_preserves_existing_physics_presets():
    """Adding Kamino must not replace the existing ANYmal-D Flat presets."""
    physics = PhysicsCfg()

    assert isinstance(physics.default, PhysxCfg)
    assert isinstance(physics.physx, PhysxCfg)
    assert isinstance(physics.newton_mjwarp, NewtonCfg)
    assert isinstance(physics.newton_mjwarp.solver_cfg, MJWarpSolverCfg)
    assert isinstance(physics.ovphysx, OvPhysxCfg)
