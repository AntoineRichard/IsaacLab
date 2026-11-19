# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a scripted Franka FR3 fill a Christmas stocking in Isaac Lab.

.. code-block:: bash

    uv run isaaclab demo christmas-atelier
"""

import argparse
import math
import sys
from pathlib import Path

from isaaclab_newton.physics import NewtonCfg, NewtonCollisionPipelineCfg, NewtonSoftContactCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.visualizers import VisualizerCfg

# The catalog runs this file as a script in checkout and wheel layouts.
_examples_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _examples_path)
try:
    from christmas_atelier import workcell
finally:
    sys.path.remove(_examples_path)


def main() -> None:
    """Launch the workcell and run one complete stocking fill."""
    parser = argparse.ArgumentParser(description="Fill a Christmas stocking with a scripted Franka FR3 in Isaac Lab.")
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Limit the run to this many frames. Zero runs it in full."
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max_steps must be nonnegative.")

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / workcell.FPS / workcell.SUBSTEPS,
        render_interval=workcell.SUBSTEPS,
        device=args.device,
        physics=NewtonCfg(
            solver_cfg=workcell.AtelierVBDSolverCfg(
                iterations=15,
                update_joint_state=False,
                particle_enable_self_contact=True,
                particle_self_contact_margin=2 * workcell.PARTICLE_RADIUS,
                particle_enable_tile_solve=args.device.startswith("cuda"),
                rigid_compliant_alm=True,
                rigid_joint_linear_ke=1.0e7,
                rigid_joint_angular_ke=1.0e6,
                rigid_body_contact_buffer_size=256,
                rigid_body_particle_contact_buffer_size=4096,
                friction_epsilon=1.0e-3,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="sap", soft_contact_margin=0.004, enable_rigid_soft_full_surface_contact=True
            ),
            soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=2000.0, soft_contact_kd=0.1, soft_contact_mu=0.5),
        ),
        visualizer_cfgs=VisualizerCfg(
            eye=(1.65, -2.25, 1.9),
            lookat=(0.34, 0.0, 1.0),
            focal_length=34.0,
            enable_live_plots=False,
            backend=NewtonGLBackendCfg(show_particles=False),
        ),
    )
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = workcell.Atelier(sim)
        frames = math.ceil(scene.times[-1] * workcell.FPS)
        if args.max_steps:
            frames = min(frames, args.max_steps)
        for _ in range(frames):
            if not sim.is_running():
                break
            scene.step()


if __name__ == "__main__":
    main()
