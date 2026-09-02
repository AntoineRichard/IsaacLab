# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open and shut the MicroDuck's beak, so the hinge can be looked at rather than trusted.

The beak variant is the one MicroDuck model with no upstream MJCF. The real robot's fifteenth servo
drives a grasping beak; every upstream RL model welds that jaw on as a fixed geom and says so --
``scripts/bake-duck-mesh.py`` in ``pollen-robotics/microduck`` notes that "``mouth`` is a servo
without an MJCF joint (the jaw is a fixed geom), so it never appears in a bake". The hinge this demo
exercises was measured from the pinned meshes; the derivation and its cross-checks against Pollen's
own firmware constants are in ``artifacts/microduck/pickplace/BEAK.md``.

The robot is held at its stand pose with its root pinned, so the only thing that moves is the jaw,
and the camera sits at beak scale -- the gape is 17 mm on a 25 cm robot, so a body-scale view shows
nothing.

.. code-block:: bash

    uv run --no-sync python scripts/demos/microduck_beak.py --physics newton_mjwarp

"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="Open and shut the MicroDuck's beak.",
    conflict_handler="resolve",
)
parser.add_argument(
    "--physics", default="newton_mjwarp", choices=["isaacsim_physx", "newton_mjwarp"], help="Physics backend."
)
parser.add_argument("--period", type=float, default=2.0, help="Seconds per open-and-shut cycle.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsCfg

from isaaclab_assets.robots.microduck import (  # isort:skip
    MICRODUCK_BEAK_CFG,
    MICRODUCK_BEAK_CLOSED,
    MICRODUCK_BEAK_JOINT_NAME,
    MICRODUCK_BEAK_OPEN,
)


def main():
    """Spawn the beak variant and drive its jaw through the full measured travel."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        # Beak scale, not body scale: the robot is 25 cm tall and the gape is 17 mm.
        sim.set_camera_view(eye=[0.22, 0.20, 0.20], target=[0.0, 0.0, 0.15])

        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95))
        cfg.func("/World/Light", cfg)

        robot_cfg = MICRODUCK_BEAK_CFG.replace(prim_path="/World/Robot")
        robot = robot_cfg.class_type(robot_cfg)

        sim.reset()
        beak_id = robot.find_joints(MICRODUCK_BEAK_JOINT_NAME)[0][0]
        print(f"[beak] '{MICRODUCK_BEAK_JOINT_NAME}' is joint {beak_id} of {robot.num_joints}")
        print(
            f"[beak] travel {math.degrees(MICRODUCK_BEAK_CLOSED):+.1f} deg .. {math.degrees(MICRODUCK_BEAK_OPEN):+.1f} deg"
        )

        sim_dt = sim.get_physics_dt()
        rest_pose = robot.data.default_root_pose.torch.clone()
        rest_vel = torch.zeros_like(robot.data.default_root_vel.torch)
        count = 0
        while sim.is_headless_or_exist_active_visualizer():
            # A raised cosine, so the shut pose is held long enough to read on a frame.
            frac = 0.5 * (1.0 - math.cos(2.0 * math.pi * count * sim_dt / args_cli.period))
            targets = robot.data.default_joint_pos.torch.clone()
            targets[:, beak_id] = MICRODUCK_BEAK_CLOSED + frac * (MICRODUCK_BEAK_OPEN - MICRODUCK_BEAK_CLOSED)
            robot.set_joint_position_target_index(target=targets)
            # This demo is about the jaw, not about balance: pin the root so the duck cannot topple
            # out of a close-up frame.
            robot.write_root_pose_to_sim_index(root_pose=rest_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=rest_vel)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
            if count % 100 == 0:
                measured = math.degrees(float(robot.data.joint_pos.torch[0, beak_id]))
                asked = math.degrees(float(targets[0, beak_id]))
                print(f"[beak] t={count * sim_dt:5.2f}s  target {asked:+6.1f} deg  measured {measured:+6.1f} deg")
            count += 1


if __name__ == "__main__":
    main()
