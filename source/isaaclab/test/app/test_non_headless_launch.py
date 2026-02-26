# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script checks if the app can be launched with non-headless app and start the simulation.
"""

"""Launch Isaac Sim Simulator first."""


import pytest

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(experience="isaaclab.python.kit", headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""


from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=GroundPlaneCfg())


def run_simulator(
    sim: SimulationContext,
):
    """Run the simulator."""

    count = 0

    # Simulate physics
    while simulation_app.is_running() and count < 100:
        # perform step
        sim.step()
        count += 1


@pytest.mark.isaacsim_ci
def test_non_headless_launch():
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print(scene)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim)
