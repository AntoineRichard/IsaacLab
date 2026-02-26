# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for simulation context with stage in memory."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
# FIXME (mmittal): Stage in memory requires cameras to be enabled.
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""


import pytest

import omni.physx
import omni.usd
import usdrt
from isaacsim.core.cloner import GridCloner

from isaaclab.sim.simulation_context import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.version import get_isaac_sim_version
from isaaclab.sim.schemas import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.materials import MdlFileCfg, PreviewSurfaceCfg, RigidBodyMaterialCfg
from isaaclab.sim.spawners.meshes import MeshCuboidCfg
from isaaclab.sim.spawners.shapes import ConeCfg, SphereCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.sim.utils.prims import create_prim
from isaaclab.sim.utils.queries import find_matching_prim_paths
from isaaclab.sim.utils.stage import get_current_stage_id, is_current_stage_in_memory, update_stage, use_stage


@pytest.fixture
def sim():
    """Create a simulation context."""
    cfg = SimulationCfg(create_stage_in_memory=True)
    sim = SimulationContext(cfg=cfg)
    update_stage()
    yield sim
    omni.physx.get_physx_simulation_interface().detach_stage()
    sim.stop()
    sim.clear_instance()


"""
Tests
"""


def test_stage_in_memory_with_shapes(sim):
    """Test spawning of shapes with stage in memory."""

    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # grab stage in memory and set as current stage via the with statement
    stage_in_memory = sim.stage
    with use_stage(stage_in_memory):
        # create parent prim for shape prototypes
        create_prim("/World/Cone", "Xform")
        num_shape_prototypes = 3

        cfg = MultiAssetSpawnerCfg(
            assets_cfg=[
                ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    physics_material=RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=1.0,
                        dynamic_friction=1.0,
                    ),
                ),
                MeshCuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=MdlFileCfg(
                        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                        project_uvw=True,
                        texture_scale=(0.25, 0.25),
                    ),
                ),
                SphereCfg(
                    radius=0.3,
                    visual_material=MdlFileCfg(
                        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                        project_uvw=True,
                        texture_scale=(0.25, 0.25),
                    ),
                    physics_material=RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=1.0,
                        dynamic_friction=1.0,
                    ),
                ),
            ],
            random_choice=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=MassPropertiesCfg(mass=1.0),
            collision_props=CollisionPropertiesCfg(),
        )
        prim_path_regex = "/World/Cone/asset_.*"
        cfg.func(prim_path_regex, cfg)

        # verify prims exist in stage
        prims = find_matching_prim_paths(prim_path_regex)
        assert len(prims) == num_shape_prototypes

    # verify stage is no longer in memory
    assert not is_current_stage_in_memory()

    # verify prims now exist in context stage
    prims = find_matching_prim_paths(prim_path_regex)
    assert len(prims) == num_shape_prototypes


def test_stage_in_memory_with_usds(sim):
    """Test spawning of USDs with stage in memory."""

    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # define parameters
    num_robot_prototypes = 2
    usd_paths = [
        f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
    ]

    # verify stage is attached to USD context (happens automatically now with create_stage_in_memory)
    assert not is_current_stage_in_memory()

    # grab stage and set as current stage via the with statement
    stage_in_memory = sim.stage
    with use_stage(stage_in_memory):
        # create parent prim for robot prototypes
        create_prim("/World/Robot", "Xform")

        cfg = MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=True,
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
        prim_path_regex = "/World/Robot/asset_.*"
        cfg.func(prim_path_regex, cfg)

        # verify prims exist in stage
        prims = find_matching_prim_paths(prim_path_regex)
        assert len(prims) == num_robot_prototypes

    # verify stage is no longer in memory
    assert not is_current_stage_in_memory()

    # verify prims now exist in context stage
    prims = find_matching_prim_paths(prim_path_regex)
    assert len(prims) == num_robot_prototypes


def test_stage_in_memory_with_clone_in_fabric(sim):
    """Test cloning in fabric with stage in memory."""

    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # define parameters
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    num_clones = 100

    # verify stage is attached to USD context (happens automatically now with create_stage_in_memory)
    assert not is_current_stage_in_memory()

    # grab stage and set as current stage via the with statement
    stage_in_memory = sim.stage
    with use_stage(stage_in_memory):
        # set up paths
        base_env_path = "/World/envs"
        source_prim_path = f"{base_env_path}/env_0"

        # create cloner
        cloner = GridCloner(spacing=3, stage=stage_in_memory)
        cloner.define_base_env(base_env_path)

        # create source prim
        create_prim(f"{source_prim_path}/Robot", "Xform", usd_path=usd_path)

        # generate target paths
        target_paths = cloner.generate_paths("/World/envs/env", num_clones)

        # clone robots at target paths
        cloner.clone(
            source_prim_path=source_prim_path,
            base_env_path=base_env_path,
            prim_paths=target_paths,
            replicate_physics=True,
        )

    # verify prims exist in fabric stage using usdrt apis
    stage_id = get_current_stage_id()
    usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)
    for i in range(num_clones):
        prim = usdrt_stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        assert prim.IsValid()
