# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""


import pytest

from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sim.schemas import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.shapes import ConeCfg, CuboidCfg, SphereCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.sim.utils.prims import create_prim
from isaaclab.sim.utils.queries import find_matching_prim_paths
from isaaclab.sim.utils.stage import create_new_stage, update_stage


@pytest.fixture
def sim():
    """Create a simulation context."""
    create_new_stage()
    dt = 0.1
    sim = SimulationContext(SimulationCfg(dt=dt))
    update_stage()
    yield sim
    sim.stop()
    sim.clear_instance()


def test_spawn_multiple_shapes_with_regex_prefix(sim):
    """Ensure assets are spawned and cloned when using regex prefix paths."""
    num_envs = 3
    num_assets = 3
    for env_idx in range(num_envs):
        env_path = f"/World/env_{env_idx}"
        create_prim(env_path, "Xform", translation=(0, 0, 0))
        create_prim(f"{env_path}/Cone", "Xform")

    cfg = MultiAssetSpawnerCfg(
        assets_cfg=[
            ConeCfg(
                radius=0.3,
                height=0.6,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                mass_props=MassPropertiesCfg(mass=100.0),  # this one should get overridden
            ),
            CuboidCfg(
                size=(0.3, 0.3, 0.3),
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
            SphereCfg(
                radius=0.3,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
        ],
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        mass_props=MassPropertiesCfg(mass=1.0),
        collision_props=CollisionPropertiesCfg(),
    )

    prim = cfg.func("/World/env_.*/Cone/asset_.*", cfg)
    assert str(prim.GetPath()) == "/World/env_0/Cone/asset_0"

    prim_paths = find_matching_prim_paths("/World/env_.*/Cone/asset_.*")
    assert len(prim_paths) == num_assets * num_envs

    for env_idx in range(num_envs):
        for asset_idx in range(num_assets):
            path = f"/World/env_{env_idx}/Cone/asset_{asset_idx}"
            assert path in prim_paths
            assert sim.stage.GetPrimAtPath(path).GetAttribute("physics:mass").Get() == cfg.mass_props.mass


def test_spawn_multiple_shapes_with_global_settings(sim):
    """Test spawning of shapes randomly with global rigid body settings."""
    create_prim("/World/template", "Xform", translation=(0, 0, 0))

    cfg = MultiAssetSpawnerCfg(
        assets_cfg=[
            ConeCfg(
                radius=0.3,
                height=0.6,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                mass_props=MassPropertiesCfg(mass=100.0),  # this one should get overridden
            ),
            CuboidCfg(
                size=(0.3, 0.3, 0.3),
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
            SphereCfg(
                radius=0.3,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
        ],
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        mass_props=MassPropertiesCfg(mass=1.0),
        collision_props=CollisionPropertiesCfg(),
    )
    prim = cfg.func("/World/template/Cone/asset_.*", cfg)

    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/template/Cone/asset_0"
    prim_paths = find_matching_prim_paths("/World/template/Cone/asset_.*")
    assert len(prim_paths) == 3

    for prim_path in prim_paths:
        prim = sim.stage.GetPrimAtPath(prim_path)
        assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass


def test_spawn_multiple_shapes_with_individual_settings(sim):
    """Test spawning of shapes randomly with individual rigid object settings."""
    create_prim("/World/template", "Xform", translation=(0, 0, 0))

    mass_variations = [2.0, 3.0, 4.0]
    cfg = MultiAssetSpawnerCfg(
        assets_cfg=[
            ConeCfg(
                radius=0.3,
                height=0.6,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                rigid_props=RigidBodyPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=mass_variations[0]),
                collision_props=CollisionPropertiesCfg(),
            ),
            CuboidCfg(
                size=(0.3, 0.3, 0.3),
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                rigid_props=RigidBodyPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=mass_variations[1]),
                collision_props=CollisionPropertiesCfg(),
            ),
            SphereCfg(
                radius=0.3,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                rigid_props=RigidBodyPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=mass_variations[2]),
                collision_props=CollisionPropertiesCfg(),
            ),
        ],
    )
    prim = cfg.func("/World/template/Cone/asset_.*", cfg)

    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/template/Cone/asset_0"
    prim_paths = find_matching_prim_paths("/World/template/Cone/asset_.*")
    assert len(prim_paths) == 3

    for prim_path in prim_paths:
        prim = sim.stage.GetPrimAtPath(prim_path)
        assert prim.GetAttribute("physics:mass").Get() in mass_variations


"""
Tests - Multiple USDs.
"""


def test_spawn_multiple_files_with_global_settings(sim):
    """Test spawning of files randomly with global articulation settings."""
    create_prim("/World/template", "Xform", translation=(0, 0, 0))

    cfg = MultiUsdFileCfg(
        usd_path=[
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        ],
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=True,
    )
    prim = cfg.func("/World/template/Robot/asset_.*", cfg)

    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/template/Robot/asset_0"
    prim_paths = find_matching_prim_paths("/World/template/Robot/asset_.*")
    assert len(prim_paths) == 2
