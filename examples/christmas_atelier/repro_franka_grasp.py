# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare Franka grasping a single rigid candy in Newton on flat ground.

Run this file directly from a Newton environment with the ``importers`` and
``sim`` extras and GitPython for asset download. Tested at Newton revision
405ec9b54c3743b126cafe3ea59416a921ac36c8 with Warp 1.17.0 and MuJoCo Warp 3.17.0.
Use ``--usd pickup.usd`` to save an animated recording.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import newton
import newton.utils
import newton.viewer
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo, SolverVBD

FPS = 60
SUBSTEPS = 12
ITERATIONS = 15
OBJECT_START = (0.375, -0.23, 0.014)
CLOSE_START = 5.0
LIFT_START = 5.8
LIFT_END = 7.8
FRANKA_REVISION = "a96f0973b6ae69c90609f9fafef9d2c1db2d6431"

# IK at the home, approach, pick and lift poses, with the Franka base at ground level.
HOME = (-0.013765690, -0.556726217, -0.028884282, -2.207333326, -0.015310135, 1.650784969, 0.748331785)
APPROACH = (-0.117156506, -0.389616609, -0.342368990, -1.966722608, -0.127911627, 1.597449064, 0.353197902)
PICK = (-0.271077037, 0.434059948, -0.195775643, -2.531195879, 0.422424048, 2.940695286, -0.078577317)
LIFT = (-0.152631506, -0.385116309, -0.314951152, -1.967067480, -0.116681263, 1.599001408, 0.342843592)
TIMES = np.array([0.0, 1.0, 3.0, CLOSE_START, LIFT_START, LIFT_END])
TARGETS = np.array(
    [
        (*HOME, 0.04, 0.04),
        (*HOME, 0.04, 0.04),
        (*APPROACH, 0.04, 0.04),
        (*PICK, 0.04, 0.04),
        (*PICK, 0.01, 0.01),
        (*LIFT, 0.01, 0.01),
    ],
    dtype=np.float32,
)


def build_scene(
    urdf: Path,
    collider: str,
    device: wp.Device,
    *,
    stock_contacts: bool = False,
    pad_margin: float | None = None,
    finger_gap: float | None = None,
    pad_stiffness: float | None = None,
) -> tuple[newton.Model, int]:
    """Build a fixed-base Franka, ground plane and 20 g candy."""
    builder = newton.ModelBuilder()
    builder.add_urdf(str(urdf), floating=False, enable_self_collisions=False)
    builder.collapse_fixed_joints(joints_to_keep=["fr3/fr3_hand_joint"])
    builder.joint_q[:] = TARGETS[0].tolist()
    builder.joint_target_ke[:] = [32000.0] * 7 + [1500.0] * 2
    builder.joint_target_kd[:] = [240.0] * 7 + [8.0] * 2

    def material(mu: float, gap: float) -> newton.ModelBuilder.ShapeConfig:
        if stock_contacts:
            return newton.ModelBuilder.ShapeConfig(density=0.0)
        return newton.ModelBuilder.ShapeConfig(mu=mu, ke=1.0e4, kd=30.0, gap=gap, density=0.0)

    for shape, body in enumerate(builder.shape_body):
        is_finger = body >= 0 and builder.body_label[body].endswith(("leftfinger", "rightfinger"))
        if not stock_contacts:
            builder.shape_material_mu[shape] = 1.5 if is_finger else 0.5
            builder.shape_material_ke[shape] = 1.0e4
            builder.shape_material_kd[shape] = 30.0
        if is_finger and builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            if finger_gap is not None:
                builder.shape_gap[shape] = finger_gap
            # The FR3 URDF places each rubber-tip box 45.25 mm down its finger link.
            if builder.shape_type[shape] == newton.GeoType.BOX and math.isclose(
                builder.shape_transform[shape][2], 0.04525, abs_tol=1.0e-6
            ):
                if pad_margin is not None:
                    builder.shape_margin[shape] = pad_margin
                if pad_stiffness is not None:
                    builder.shape_material_ke[shape] = pad_stiffness

    builder.add_ground_plane(cfg=material(mu=0.5, gap=0.001))

    candy = builder.add_body(
        xform=wp.transform(OBJECT_START, wp.quat_identity()),
        mass=0.020,
        inertia=wp.diag(wp.vec3(2.0e-6)),
        label="candy",
    )
    if collider == "cube":
        add_shape = builder.add_shape_box
        dimensions = {"hx": 0.014, "hy": 0.014, "hz": 0.014}
    else:
        cylinder = collider == "cylinder"
        add_shape = builder.add_shape_cylinder if cylinder else builder.add_shape_capsule
        dimensions = {
            "xform": wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)),
            "radius": 0.014,
            "half_height": 0.014 if cylinder else 0.005,
        }
    add_shape(
        candy,
        **dimensions,
        cfg=material(mu=1.2, gap=0.0005),
        color=(0.85, 0.06, 0.09),
        label="candy_collider",
    )
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    return model, candy


def run(args: argparse.Namespace, device: wp.Device) -> None:
    """Run one scripted pickup and print the candy's measured rise."""
    urdf = args.robot_urdf
    if urdf is None:
        assets = newton.utils.download_asset("franka_emika_panda", ref=FRANKA_REVISION)
        urdf = assets / "urdf/fr3_franka_hand.urdf"

    model, candy = build_scene(
        urdf,
        args.collider,
        device,
        stock_contacts=args.stock_contacts,
        pad_margin=None if args.pad_margin_mm is None else args.pad_margin_mm / 1000.0,
        finger_gap=None if args.finger_gap_mm is None else args.finger_gap_mm / 1000.0,
        pad_stiffness=args.pad_stiffness,
    )
    state, next_state, control = model.state(), model.state(), model.control()
    newton_contacts = args.solver != "mujoco-native"
    pipeline = newton.CollisionPipeline(model, broad_phase="sap") if newton_contacts else None
    contacts = pipeline.contacts() if pipeline is not None else None
    if args.solver == "vbd":
        solver = SolverVBD(
            model,
            iterations=ITERATIONS,
            rigid_joint_linear_ke=1.0e7,
            rigid_joint_angular_ke=1.0e6,
            rigid_body_contact_buffer_size=256,
            friction_epsilon=1.0e-3,
        )
    else:
        solver = SolverMuJoCo(
            model,
            use_mujoco_contacts=not newton_contacts,
            enable_multiccd=True,
            cone=args.cone,
            impratio=args.impratio,
            # Multi-CCD against the ground plane can exceed the native default capacity.
            nconmax=256 if not newton_contacts else None,
            integrator="implicitfast",
            iterations=ITERATIONS,
            update_data_interval=0,
        )
        solver.reset(state)

    def step_frame() -> None:
        nonlocal state, next_state
        for _ in range(SUBSTEPS):
            state.clear_forces()
            if pipeline is not None:
                pipeline.collide(state, contacts)
            solver.step(state, next_state, control, contacts, 1.0 / (FPS * SUBSTEPS))
            state, next_state = next_state, state

    frame_count = round(FPS * (LIFT_END + 0.1))
    viewer = newton.viewer.ViewerUSD(output_path=str(args.usd), num_frames=frame_count) if args.usd else None
    if viewer is not None:
        viewer.set_model(model)
    graph = None
    candy_positions = []
    try:
        for frame in range(frame_count):
            t = frame / FPS
            segment = min(int(np.searchsorted(TIMES, t, side="right") - 1), len(TIMES) - 2)
            alpha = min((t - TIMES[segment]) / (TIMES[segment + 1] - TIMES[segment]), 1.0)
            blend = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
            control.joint_target_q.assign(TARGETS[segment] + blend * (TARGETS[segment + 1] - TARGETS[segment]))
            if graph is None:
                step_frame()
                if device.is_cuda:
                    # Both buffers return to their starting roles after twelve substeps.
                    with wp.ScopedCapture() as capture:
                        step_frame()
                    graph = capture.graph
            else:
                wp.capture_launch(graph)
            candy_positions.append(state.body_q.numpy()[candy, :3].copy())
            if viewer is not None:
                viewer.begin_frame((frame + 1) / FPS)
                viewer.log_state(state)
                viewer.end_frame()
    finally:
        if viewer is not None:
            viewer.close()

    positions = np.array(candy_positions)
    if not np.isfinite(positions).all():
        raise RuntimeError("The solver produced a non-finite candy position.")
    before_grasp = positions[round(FPS * CLOSE_START) - 1]
    during_lift = positions[round(FPS * LIFT_START) - 1 : round(FPS * LIFT_END)]
    drift = np.linalg.norm(before_grasp[:2] - OBJECT_START[:2])
    print(f"{args.solver} / {args.collider}")
    print(f"  XY drift before grasp: {1000.0 * drift:.3f} mm")
    print(f"  Peak rise during lift: {1000.0 * (during_lift[:, 2].max() - before_grasp[2]):.2f} mm")
    print(f"  Rise at end of lift:   {1000.0 * (during_lift[-1, 2] - before_grasp[2]):.2f} mm")


def main() -> None:
    """Parse the collider and solver choices for the standalone reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collider", choices=("capsule", "cylinder", "cube"), default="cylinder")
    parser.add_argument("--solver", choices=("vbd", "mujoco-newton", "mujoco-native"), default="vbd")
    parser.add_argument("--cone", choices=("pyramidal", "elliptic"), help="Friction cone for MuJoCo Warp.")
    parser.add_argument("--impratio", type=float, help="Friction-to-normal impedance ratio for MuJoCo Warp.")
    parser.add_argument("--stock-contacts", action="store_true", help="Use Newton's default materials and gaps.")
    parser.add_argument("--pad-margin-mm", type=float, help="Contact margin on each rubber tip in millimeters.")
    parser.add_argument("--finger-gap-mm", type=float, help="Detection gap on finger colliders in millimeters.")
    parser.add_argument("--pad-stiffness", type=float, help="Rubber-tip normal contact stiffness in N/m.")
    parser.add_argument("--device", help="Warp device, for example cuda:0 or cpu")
    parser.add_argument("--robot-urdf", type=Path, help="Path to the FR3 hand URDF. Defaults to Newton's Franka asset.")
    parser.add_argument("--usd", type=Path, help="Write the simulated pickup to this animated USD file.")
    args = parser.parse_args()
    if args.impratio is not None and (not math.isfinite(args.impratio) or args.impratio <= 0):
        parser.error("--impratio must be positive and finite.")
    for option, distance in (("--pad-margin-mm", args.pad_margin_mm), ("--finger-gap-mm", args.finger_gap_mm)):
        if distance is not None and (not math.isfinite(distance) or distance < 0):
            parser.error(f"{option} must be non-negative and finite.")
    if args.pad_stiffness is not None and (not math.isfinite(args.pad_stiffness) or args.pad_stiffness <= 0):
        parser.error("--pad-stiffness must be positive and finite.")
    if args.solver == "vbd" and (args.cone is not None or args.impratio is not None):
        parser.error("--cone and --impratio apply to the MuJoCo Warp modes.")
    with wp.ScopedDevice(args.device) as device:
        if args.solver != "vbd" and not device.is_cuda:
            parser.error("The MuJoCo Warp modes require a CUDA device.")
        run(args, device)


if __name__ == "__main__":
    main()
