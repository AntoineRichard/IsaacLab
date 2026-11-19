# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a scripted Franka FR3 fill a Christmas stocking in Isaac Lab.

.. code-block:: bash

    uv run isaaclab demo christmas-atelier
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from isaaclab_newton.physics import NewtonCfg, NewtonCollisionPipelineCfg, NewtonSoftContactCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.visualizers import VisualizerCfg

# The catalog runs the launcher as a script in both checkout and wheel layouts.
_examples_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _examples_path)
try:
    from christmas_atelier import assets, workcell
finally:
    sys.path.remove(_examples_path)

logger = logging.getLogger(__name__)


def _run(sim: sim_utils.SimulationContext, args: argparse.Namespace) -> None:
    scene = workcell.Atelier(sim, args)
    frames = args.max_steps or math.ceil(scene.duration * workcell.FPS)
    recorder = None
    if args.output is not None:
        from newton.viewer import ViewerUSD  # noqa: PLC0415

        args.output.parent.mkdir(parents=True, exist_ok=True)
        recorder = ViewerUSD(output_path=str(args.output), num_frames=frames)
        recorder.set_model(scene.backend.model)
    try:
        while scene.frame < frames and sim.is_running():
            scene.step()
            if recorder is not None:
                recorder.begin_frame(scene.sim_time)
                recorder.log_state(scene.state)
                recorder.end_frame()
    finally:
        if recorder is not None:
            recorder.close()

    if args.metrics is not None or args.check:
        report = scene.report()
        if args.metrics is not None:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            logger.info("Measurements: %s", args.metrics)
        if args.check and not report["passed"]:
            raise SystemExit("Physical sequence did not pass. Use --metrics to save the rollout measurements.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill a Christmas stocking with a scripted Franka FR3 in Isaac Lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("probe", "drape", "drop", "fill"), default="fill")
    parser.add_argument(
        "--robot-urdf", type=Path, help="Local FR3 + Franka hand URDF; otherwise use Newton's asset cache."
    )
    parser.add_argument("--candies", type=int, default=3)
    parser.add_argument(
        "--candy-collider",
        choices=("capsule", "cylinder"),
        default="capsule",
        help="Rigid collider for the candy core.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Limit the number of 60 Hz frames. Zero runs the complete sequence."
    )
    parser.add_argument("--arm-stiffness", type=float, default=32000.0)
    parser.add_argument("--arm-damping", type=float, default=240.0)
    parser.add_argument(
        "--grip-contact-damping",
        type=float,
        default=workcell.RIGID_CONTACT_DAMPING,
        help="Contact damping for the finger and candy core materials [N s/m].",
    )
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument(
        "--iterations",
        type=int,
        default=argparse.SUPPRESS,
        help="VBD iterations per substep. Defaults to 15 for the sewn stocking and 60 for option B.",
    )
    parser.add_argument("--grasp-iterations", type=int, default=15, help="Cap VBD iterations during pickup and carry.")
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable physics graph capture for comparison."
    )
    parser.add_argument("--stocking-pattern", choices=("B", "sewn"), default="sewn")
    parser.add_argument("--mesh-rings", type=int, default=28)
    parser.add_argument("--mesh-sides", type=int, default=32)
    parser.add_argument("--areal-density", type=float, default=0.25, help="Cloth surface density [kg/m^2].")
    parser.add_argument("--stretch-stiffness", type=float, default=1500.0)
    parser.add_argument("--stretch-damping", type=float, default=0.01)
    parser.add_argument("--bend-stiffness", type=float, default=5.0e-7)
    parser.add_argument("--bend-damping", type=float, default=4.0e-3)
    parser.add_argument(
        "--cuff-height", type=float, default=workcell.MOUTH[2], help="Supported cuff height above ground [m]."
    )
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--particle-radius", type=float, default=0.002)
    parser.add_argument("--contact-stiffness", type=float, default=2000.0)
    parser.add_argument("--output", type=Path, help="Optional animated USD recording path.")
    parser.add_argument("--metrics", type=Path, help="Optional JSON rollout measurements path.")
    parser.add_argument("--export-assets", type=Path, help="Export USD art assets and exit without simulating.")
    parser.add_argument(
        "--check", action="store_true", help="Exit unsuccessfully unless the complete physical sequence passes."
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if not hasattr(args, "iterations"):
        args.iterations = 15 if args.stocking_pattern == "sewn" else 60
    if args.export_assets:
        return args
    if not 1 <= args.candies <= 5:
        parser.error("--candies must be between 1 and 5.")
    if args.substeps < 1 or args.iterations < 1 or args.grasp_iterations < 1 or args.max_steps < 0:
        parser.error("Substeps and iterations must be positive. --max_steps must be nonnegative.")
    if args.mesh_rings < 8 or args.mesh_sides < 8:
        parser.error("The stocking requires at least eight mesh rings and sides.")
    for name in (
        "arm_stiffness",
        "arm_damping",
        "areal_density",
        "stretch_stiffness",
        "bend_stiffness",
        "friction",
        "particle_radius",
        "contact_stiffness",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite.")
    for name in ("bend_damping", "stretch_damping", "grip_contact_damping"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative and finite.")
    mesh = assets.panel_stocking_mesh if args.stocking_pattern == "sewn" else assets.stocking_mesh
    lowest = float(mesh(args.mesh_rings, args.mesh_sides)[0][:, 2].min())
    if not math.isfinite(args.cuff_height) or args.cuff_height + lowest <= workcell.TABLE_TOP:
        parser.error("--cuff-height must leave the undeformed stocking above the bench.")
    return args


def _simulation_cfg(args: argparse.Namespace) -> sim_utils.SimulationCfg:
    return sim_utils.SimulationCfg(
        dt=1.0 / workcell.FPS / args.substeps,
        render_interval=args.substeps,
        device=args.device,
        physics=NewtonCfg(
            use_cuda_graph=not args.disable_cuda_graph,
            solver_cfg=workcell.AtelierVBDSolverCfg(
                iterations=args.iterations,
                update_joint_state=False,
                particle_enable_self_contact=args.mode != "probe",
                particle_self_contact_margin=2 * args.particle_radius,
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
            soft_contact_cfg=NewtonSoftContactCfg(
                soft_contact_ke=args.contact_stiffness, soft_contact_kd=0.1, soft_contact_mu=args.friction
            ),
        ),
        visualizer_cfgs=VisualizerCfg(
            eye=(1.65, -2.25, 1.9),
            lookat=(0.34, 0.0, 1.0),
            focal_length=34.0,
            enable_live_plots=False,
            backend=NewtonGLBackendCfg(show_particles=False),
        ),
    )


def main() -> None:
    """Launch the filling demo or a shorter calibration scene."""
    args = _parse_args()
    if args.export_assets is not None:
        assets.export_assets(args.export_assets, sewn=args.stocking_pattern == "sewn")
        return

    sim_cfg = _simulation_cfg(args)
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        try:
            _run(sim, args)
        finally:
            sim_utils.SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
