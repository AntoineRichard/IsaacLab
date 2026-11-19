# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton VBD prototype for the Christmas stocking-filling atelier.

Run from the repository root with ``uv run python -m examples.christmas_atelier.run``.
All dynamics, including the Franka joints, run in SolverVBD. No external rigid solver,
kinematic grasp, candy teleportation, or force attachment is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import time
from collections import deque
from pathlib import Path

import newton
import newton.ik as ik
import newton.utils
import newton.viewer
import numpy as np
import warp as wp
from newton.solvers import SolverVBD

from .assets import (
    CANDY_CAPSULE_HALF_LENGTH,
    CANDY_CORE_HALF_LENGTH,
    CANDY_CORE_RADIUS,
    CREAM,
    GOLD,
    GREEN,
    RED,
    candy_mesh,
    export_assets,
    panel_stocking_mesh,
    stocking_mesh,
)
from .motion import choreography, measure_stocking, sample_motion

logger = logging.getLogger(__name__)
TABLE_TOP = 0.72
MOUTH = (0.48, 0.22, 1.20)
FPS = 60
TCP_OFFSET = wp.vec3(0.0, 0.0, 0.1034)
GRIP_FRICTION = 1.5
# Resolve the light candy/rubber contact without the stiff, underdamped pinch that ejects it.
RIGID_CONTACT_STIFFNESS = 1.0e4
RIGID_CONTACT_DAMPING = 30.0
FRANKA_ASSET_REVISION = "a96f0973b6ae69c90609f9fafef9d2c1db2d6431"


def _robot_builder(
    urdf: Path, arm_stiffness: float, arm_damping: float, grip_contact_damping: float
) -> newton.ModelBuilder:
    """Import the complete Franka chain and explicitly drive both fingers for VBD."""
    builder = newton.ModelBuilder()
    builder.add_urdf(
        str(urdf),
        floating=False,
        xform=wp.transform((0.0, 0.0, TABLE_TOP), wp.quat_identity()),
        enable_self_collisions=False,
    )
    # A massless URDF frame becomes immovable in a maximal-coordinate solve. Merge those
    # frames into their physical links, retaining the massive hand as our measured TCP parent.
    builder.collapse_fixed_joints(joints_to_keep=["fr3/fr3_hand_joint"])
    # VBD does not solve the URDF mimic equality. Give each physical finger its own target.
    for index in range(len(builder.joint_target_ke)):
        builder.joint_target_ke[index] = arm_stiffness if index < 7 else 1500.0
        builder.joint_target_kd[index] = arm_damping if index < 7 else 8.0
    builder.joint_q[:] = [0.0, -0.5, 0.0, -2.1, 0.0, 1.6, math.pi / 4, 0.04, 0.04]
    # The stocking only contacts analytic finger/candy geometry; avoid mesh volume SDFs on CPU.
    finger_bodies = {
        index for index, label in enumerate(builder.body_label) if label.endswith(("leftfinger", "rightfinger"))
    }
    for shape, body in enumerate(builder.shape_body):
        if body not in finger_bodies:
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        builder.shape_material_mu[shape] = GRIP_FRICTION if body in finger_bodies else 0.5
        builder.shape_material_ke[shape] = RIGID_CONTACT_STIFFNESS
        builder.shape_material_kd[shape] = grip_contact_damping if body in finger_bodies else RIGID_CONTACT_DAMPING
    return builder


def _solve_waypoints(builder: newton.ModelBuilder, positions: np.ndarray, fingers: np.ndarray, device: str) -> tuple:
    """Solve IK on a separate reference model; never overwrite simulated body poses."""
    model = builder.finalize(device=device)
    tcp = next(index for index, label in enumerate(model.body_label) if label.endswith("fr3_hand"))
    position = ik.IKObjectivePosition(
        link_index=tcp,
        link_offset=TCP_OFFSET,
        target_positions=wp.array([positions[0]], dtype=wp.vec3, device=device),
    )
    rotation = ik.IKObjectiveRotation(
        link_index=tcp,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.vec4, device=device),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[position, rotation, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    q = wp.clone(model.joint_q).reshape((1, model.joint_coord_count))
    state = model.state()
    targets = []
    for target, opening in zip(positions, fingers, strict=True):
        position.set_target_position(0, wp.vec3(*map(float, target)))
        solver.step(q, q, iterations=80)
        values = q.numpy().ravel()
        values[7:9] = opening
        q.assign(values.reshape(1, -1))
        newton.eval_fk(model, q.flatten(), model.joint_qd, state)
        error = np.linalg.norm(
            np.array(wp.transform_point(wp.transform(*state.body_q.numpy()[tcp]), TCP_OFFSET)) - target
        )
        if error > 0.003:
            raise RuntimeError(f"Unreachable TCP waypoint {target.tolist()}: IK error {error:.4f} m")
        targets.append(values.copy())
    return model, tcp, np.array(targets, dtype=np.float32)


class Atelier:
    """Own the native scene, dynamic states, scripted controls and measured rollout."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mouth = (MOUTH[0], MOUTH[1], args.cuff_height)
        self.frame = 0
        self.sim_time = 0.0
        self.device = wp.get_device(args.device)
        self.has_robot = args.mode in ("probe", "fill")
        self.has_stocking = args.mode != "probe"
        self.candy_count = 0 if args.mode == "drape" else (1 if args.mode == "probe" else args.candies)
        self.initial_candies = [
            (0.45 + (index - (self.candy_count - 1) / 2) * 0.075, -0.23, TABLE_TOP + 0.028)
            for index in range(self.candy_count)
        ]
        builder = newton.ModelBuilder()
        self.reference_model = None
        if self.has_robot:
            urdf = args.robot_urdf
            if urdf is None:
                urdf = (
                    newton.utils.download_asset("franka_emika_panda", ref=FRANKA_ASSET_REVISION)
                    / "urdf/fr3_franka_hand.urdf"
                )
            robot = _robot_builder(urdf, args.arm_stiffness, args.arm_damping, args.grip_contact_damping)
            self.times, positions, fingers, self.phases = choreography(self.initial_candies, self.mouth)
            self.reference_model, self.tcp, self.targets = _solve_waypoints(robot, positions, fingers, str(self.device))
            self.reference_state = self.reference_model.state()
            robot.joint_q[:] = self.targets[0].tolist()
            builder.add_builder(robot)
            self.duration = float(self.times[-1])
        else:
            self.duration = 6.0
        self._build_workcell(builder)
        self.faces = self.cuff = None
        if self.has_stocking:
            self._build_stocking(builder)
        self.candy_bodies = []
        self._build_candies(builder)
        # Warp's balancing pass can stall on this closed-toe graph; coloring alone is sufficient.
        builder.color(include_bending=True, balance_colors=False)
        if self.has_robot and self.candy_bodies:
            # Solve candy updates after the robot so the grasp contact sees the updated fingertip poses.
            candy_ids = set(self.candy_bodies)
            body_groups = []
            for group in builder.body_color_groups:
                robot_bodies = [body for body in group if body not in candy_ids]
                if robot_bodies:
                    body_groups.append(np.asarray(robot_bodies, dtype=np.int32))
            body_groups.append(np.asarray(self.candy_bodies, dtype=np.int32))
            builder.body_color_groups = body_groups
        self.model = builder.finalize(device=self.device)
        self.model.soft_contact_ke = args.contact_stiffness
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = args.friction
        # ALM captures joint rest transforms at construction, so initialize FK BEFORE the solver.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model, broad_phase="sap", soft_contact_gap=0.004, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverVBD(
            self.model,
            iterations=args.iterations,
            rigid_compliant_alm=True,
            integrate_with_external_rigid_solver=False,
            rigid_joint_linear_ke=1.0e7,
            rigid_joint_angular_ke=1.0e6,
            rigid_body_contact_buffer_size=256,
            rigid_body_particle_contact_buffer_size=4096,
            particle_enable_self_contact=self.has_stocking,
            particle_self_contact_margin=2 * args.particle_radius,
            particle_self_contact_gap=0.003,
            particle_enable_tile_solve=self.device.is_cuda,
            friction_epsilon=1.0e-3,
        )
        self.control_targets = self.control.joint_target_q.numpy() if self.has_robot else None
        self.max_tcp_error = 0.0
        self.max_joint_error = 0.0
        self.lift_heights = np.zeros(self.candy_count)
        self.release_errors = np.full(self.candy_count, np.inf)
        self.last_phase = ""
        self.phase_samples = []
        self.cloth_speed_squared = deque(maxlen=FPS)
        self.graphs = {}
        self.initial_particles = self.state.particle_q.numpy().copy() if self.has_stocking else None
        self.started = time.perf_counter()

    def _build_workcell(self, builder: newton.ModelBuilder) -> None:
        """Build an open bin, wooden bench and brass cuff support from analytic shapes."""

        def box(name: str, center: tuple, half: tuple, color: tuple) -> None:
            builder.add_shape_box(
                -1,
                xform=wp.transform(center, wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                color=color,
                cfg=newton.ModelBuilder.ShapeConfig(
                    mu=0.5, ke=RIGID_CONTACT_STIFFNESS, kd=RIGID_CONTACT_DAMPING, gap=0.001
                ),
                label=name,
            )

        builder.add_ground_plane(
            color=(0.08, 0.12, 0.13),
            cfg=newton.ModelBuilder.ShapeConfig(ke=RIGID_CONTACT_STIFFNESS, kd=RIGID_CONTACT_DAMPING),
        )
        box("walnut_bench", (0.30, 0.0, TABLE_TOP - 0.03), (0.62, 0.53, 0.03), (0.23, 0.10, 0.045))
        for x in (-0.2, 0.8):
            for y in (-0.42, 0.42):
                box(f"leg_{x}_{y}", (x, y, 0.345), (0.035, 0.035, 0.345), GREEN)
        box("candy_bin_floor", (0.45, -0.23, TABLE_TOP + 0.007), (0.23, 0.085, 0.007), GREEN)
        for side in (-1, 1):
            box(f"bin_end_{side}", (0.45 + side * 0.23, -0.23, TABLE_TOP + 0.02), (0.006, 0.09, 0.02), GREEN)
            box(f"bin_side_{side}", (0.45, -0.23 + side * 0.09, TABLE_TOP + 0.02), (0.23, 0.006, 0.02), GREEN)
        if self.has_stocking:
            box("stand_base", (self.mouth[0], self.mouth[1] + 0.15, TABLE_TOP + 0.012), (0.12, 0.07, 0.012), GOLD)
            post_bottom = TABLE_TOP + 0.024
            post_top = self.mouth[2] + 0.01
            box(
                "stand_post",
                (self.mouth[0], self.mouth[1] + 0.15, (post_bottom + post_top) / 2),
                (0.008, 0.008, (post_top - post_bottom) / 2),
                GOLD,
            )
            box("stand_arm", (self.mouth[0], self.mouth[1] + 0.105, self.mouth[2]), (0.008, 0.045, 0.008), GOLD)
            # Match the elliptical cloth mouth; the rim is visual, supported particles enforce it.
            for segment in range(32):
                angles = np.array([segment, segment + 1]) * 2 * np.pi / 32
                points = np.column_stack(
                    [
                        self.mouth[0] + 0.075 * np.cos(angles),
                        self.mouth[1] + 0.0585 * np.sin(angles),
                        np.full(2, self.mouth[2]),
                    ]
                )
                delta = points[1] - points[0]
                builder.add_shape_capsule(
                    -1,
                    xform=wp.transform(points.mean(axis=0), wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*delta))),
                    radius=0.002,
                    half_height=float(np.linalg.norm(delta) / 2),
                    color=GOLD,
                    cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False),
                    label=f"cuff_rim_{segment}",
                )

    def _build_stocking(self, builder: newton.ModelBuilder) -> None:
        mesh = panel_stocking_mesh if self.args.stocking_pattern == "sewn" else stocking_mesh
        vertices, self.faces, colors, self.cuff = mesh(self.args.mesh_rings, self.args.mesh_sides)
        builder.add_cloth_mesh(
            pos=wp.vec3(*self.mouth),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=vertices.tolist(),
            indices=self.faces.ravel().tolist(),
            density=self.args.areal_density,
            tri_ke=self.args.stretch_stiffness,
            tri_ka=self.args.stretch_stiffness,
            tri_kd=self.args.stretch_damping,
            edge_ke=self.args.bend_stiffness,
            edge_kd=self.args.bend_damping,
            particle_radius=self.args.particle_radius,
            color=colors,
            validate_mesh=True,
            label="christmas_stocking",
        )
        for particle in self.cuff:
            builder.particle_mass[int(particle)] = 0.0

    def _build_candies(self, builder: newton.ModelBuilder) -> None:
        vertices, faces, colors = candy_mesh()
        # The closed, unsplit mesh supplies cloth contacts. The two color meshes stay visual.
        wrapper = newton.Mesh(vertices, faces.ravel(), compute_inertia=False)
        wrapper_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=True,
            is_visible=False,
            mu=1.2,
            ke=RIGID_CONTACT_STIFFNESS,
            kd=RIGID_CONTACT_DAMPING,
        )
        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, has_shape_collision=False, has_particle_collision=False
        )
        for index in range(self.candy_count):
            position = self.initial_candies[index]
            if self.args.mode == "drop":
                position = (self.mouth[0], self.mouth[1], self.mouth[2] + 0.08 + index * 0.075)
            body = builder.add_body(
                xform=wp.transform(position, wp.quat_identity()),
                mass=0.02,
                inertia=wp.mat33(2.0e-6, 0, 0, 0, 2.0e-6, 0, 0, 0, 2.0e-6),
                label=f"candy_{index}",
            )
            self.candy_bodies.append(body)
            if self.args.candy_collider == "cylinder":
                add_core = builder.add_shape_cylinder
                half_height = CANDY_CORE_HALF_LENGTH
            else:
                add_core = builder.add_shape_capsule
                half_height = CANDY_CAPSULE_HALF_LENGTH
            add_core(
                body,
                radius=CANDY_CORE_RADIUS,
                half_height=half_height,
                xform=wp.transform((0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    mu=1.2,
                    ke=RIGID_CONTACT_STIFFNESS,
                    kd=self.args.grip_contact_damping,
                    gap=0.0005,
                    is_visible=False,
                    has_particle_collision=False,
                ),
                label=f"candy_core_{index}",
            )
            if self.has_stocking:
                builder.add_shape_mesh(body, mesh=wrapper, cfg=wrapper_cfg, label=f"wrapper_contact_{index}")
            white = np.all(np.isclose(colors, CREAM), axis=1)
            for mask, color in ((white, CREAM), (~white, (RED, GREEN, GOLD)[index % 3])):
                mesh = newton.Mesh(vertices, faces[mask].ravel(), compute_inertia=False)
                builder.add_shape_mesh(
                    body, mesh=mesh, cfg=visual_cfg, color=color, label=f"wrapper_{index}_{bool(mask[0])}"
                )

    def step(self) -> None:
        """Apply joint targets, then solve all contacts and dynamics together in VBD."""
        iterations = self.args.iterations
        if self.has_robot:
            target = sample_motion(self.sim_time, self.times, self.targets).astype(np.float32)
            # Choreography labels the destination of each interpolated segment.
            phase_index = min(int(np.searchsorted(self.times, self.sim_time, side="right")), len(self.phases) - 1)
            phase = self.phases[phase_index]
            pickup = phase.startswith(("approach_", "descend_", "grasp_", "lift_", "transport_", "lower_"))
            opening = phase.startswith("release_") and target[7] < 0.018
            # The rounded core lifts reliably at the qualified grasp budget. Switch to the
            # cloth budget once the fingers clear the core, before it reaches the stocking.
            if pickup or opening:
                iterations = min(iterations, self.args.grasp_iterations)
            self.control_targets[:9] = target
            self.control.joint_target_q.assign(self.control_targets)
            self.reference_state.joint_q.assign(target)
            newton.eval_fk(
                self.reference_model, self.reference_state.joint_q, self.reference_model.joint_qd, self.reference_state
            )
        self.solver.iterations = iterations
        graph = self.graphs.get(iterations)
        if graph is None:
            self._advance_physics()
            # Warm up allocations before capture. The even number of swaps returns to the
            # same state arrays, so the graphs can share both states across iteration budgets.
            if self.device.is_cuda and not self.args.disable_cuda_graph and self.args.substeps % 2 == 0:
                with wp.ScopedCapture() as capture:
                    self._advance_physics()
                self.graphs[iterations] = capture.graph
        else:
            wp.capture_launch(graph)
        self.sim_time += 1.0 / FPS
        self.frame += 1
        self._measure_frame(target if self.has_robot else None)

    def _advance_physics(self) -> None:
        """Advance one rendered frame using device-resident physics operations only."""
        dt = 1.0 / FPS / self.args.substeps
        for _ in range(self.args.substeps):
            self.state.clear_forces()
            if self.has_stocking:
                self.solver.rebuild_bvh(self.state)
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, dt)
            self.state, self.next_state = self.next_state, self.state

    def _measure_frame(self, target: np.ndarray | None) -> None:
        """Measure the physical rollout after physics has completed."""
        poses = self.state.body_q.numpy() if self.state.body_q is not None else np.empty((0, 7))
        if not np.isfinite(poses).all():
            raise RuntimeError(f"Non-finite rigid state at frame {self.frame}")
        if self.has_stocking:
            velocities = self.state.particle_qd.numpy()
            self.cloth_speed_squared.append(float(np.mean(np.sum(velocities**2, axis=1))))
        if self.has_robot:
            expected = np.array(
                wp.transform_point(wp.transform(*self.reference_state.body_q.numpy()[self.tcp]), TCP_OFFSET)
            )
            actual = np.array(wp.transform_point(wp.transform(*poses[self.tcp]), TCP_OFFSET))
            self.max_tcp_error = max(self.max_tcp_error, float(np.linalg.norm(actual - expected)))
            newton.eval_ik(self.model, self.state, self.state.joint_q, self.state.joint_qd)
            self.max_joint_error = max(
                self.max_joint_error, float(np.max(np.abs(self.state.joint_q.numpy()[:7] - target[:7])))
            )
            phase_index = min(int(np.searchsorted(self.times, self.sim_time, side="right")), len(self.phases) - 1)
            phase = self.phases[phase_index]
            if phase != self.last_phase:
                logger.info("t=%.2f %s", self.sim_time, phase)
                self.last_phase = phase
                self.phase_samples.append(
                    {
                        "time_s": self.sim_time,
                        "phase": phase,
                        "tcp_m": actual.tolist(),
                        "candies_m": poses[self.candy_bodies, :3].tolist(),
                    }
                )
            for index, body in enumerate(self.candy_bodies):
                if phase in (f"lift_{index}", f"transport_{index}", f"lower_{index}"):
                    self.lift_heights[index] = max(
                        self.lift_heights[index], poses[body, 2] - self.initial_candies[index][2]
                    )
                if phase == f"release_{index}":
                    self.release_errors[index] = min(
                        self.release_errors[index], np.linalg.norm(poses[body, :2] - np.array(self.mouth[:2]))
                    )

    def report(self) -> dict:
        """Report evidence, distinguishing truncated runs from complete successful cycles."""
        poses = self.state.body_q.numpy() if self.state.body_q is not None else np.empty((0, 7))
        complete = self.sim_time + 1.0e-6 >= self.duration
        source_root = Path(newton.__file__).resolve().parent.parent
        revision = None
        if (source_root / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                revision = result.stdout.strip()
        report = {
            "mode": self.args.mode,
            "solver": "SolverVBD",
            "external_rigid_solver": False,
            "device": str(self.device),
            "frames": self.frame,
            "cuda_graph": bool(self.graphs),
            "simulated_seconds": self.sim_time,
            "wall_seconds": time.perf_counter() - self.started,
            "complete": complete,
            "candies": self.candy_count,
            "max_tcp_error_m": self.max_tcp_error,
            "max_arm_joint_error_rad": self.max_joint_error,
            "grip_friction": GRIP_FRICTION,
            "rigid_contact_stiffness": RIGID_CONTACT_STIFFNESS,
            "rigid_contact_damping": RIGID_CONTACT_DAMPING,
            "lift_heights_m": self.lift_heights.tolist(),
            "release_xy_errors_m": [float(value) if np.isfinite(value) else None for value in self.release_errors],
            "phase_samples": self.phase_samples,
            "newton_source": str(Path(newton.__file__).parent),
            "newton_version": newton.__version__,
            "newton_source_revision": revision,
            "parameters": {
                key: getattr(self.args, key)
                for key in (
                    "arm_stiffness",
                    "arm_damping",
                    "grip_contact_damping",
                    "candy_collider",
                    "substeps",
                    "iterations",
                    "grasp_iterations",
                    "stocking_pattern",
                    "mesh_rings",
                    "mesh_sides",
                    "areal_density",
                    "stretch_stiffness",
                    "stretch_damping",
                    "bend_stiffness",
                    "bend_damping",
                    "cuff_height",
                    "friction",
                    "particle_radius",
                    "contact_stiffness",
                )
            },
        }
        passed = complete and np.isfinite(poses).all()
        if self.has_robot:
            passed = (
                passed
                and self.max_tcp_error < 0.025
                and bool(np.all(self.lift_heights > 0.12))
                and bool(np.all(self.release_errors < 0.035))
            )
        if self.has_stocking:
            report.update(
                measure_stocking(
                    self.state.particle_q.numpy(),
                    self.initial_particles,
                    self.faces,
                    self.cuff,
                    poses[self.candy_bodies],
                    candy_collider=self.args.candy_collider,
                )
            )
            report["cloth_table_clearance_m"] = (
                float(self.state.particle_q.numpy()[:, 2].min()) - TABLE_TOP if report["cloth_finite"] else None
            )
            speed_squared = np.asarray(self.cloth_speed_squared)
            report["final_second_cloth_rms_speed_m_s"] = (
                float(np.sqrt(speed_squared.mean()))
                if len(speed_squared) and np.isfinite(speed_squared).all()
                else None
            )
            passed = (
                passed
                and report["cloth_finite"]
                and report["cuff_drift_m"] < 1.0e-5
                and report["max_cloth_edge_stretch_ratio"] < 1.35
                and all(report["contained"])
                and all(report["wrapper_contained"])
                and report["cloth_table_clearance_m"] > 0.01
                and len(speed_squared) == FPS
                and report["final_second_cloth_rms_speed_m_s"] is not None
                and report["final_second_cloth_rms_speed_m_s"] < 0.03
            )
        else:
            # The probe must also release the candy, then retreat away from it.
            tcp_position = np.array(wp.transform_point(wp.transform(*poses[self.tcp]), TCP_OFFSET))
            report["released"] = [
                bool(np.linalg.norm(poses[body, :3] - tcp_position) > 0.15) for body in self.candy_bodies
            ]
            passed = passed and all(report["released"])
        report["passed"] = bool(passed)
        return report


def main() -> None:
    """Run an isolated solver probe, drape/drop calibration, or full filling sequence."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=("probe", "drape", "drop", "fill"), default="fill")
    parser.add_argument("--viewer", choices=("gl", "usd", "null"), default="gl")
    parser.add_argument("--device", default=None)
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
        "--num-frames", type=int, default=0, help="Zero runs the complete sequence; positive values truncate it."
    )
    parser.add_argument("--arm-stiffness", type=float, default=32000.0)
    parser.add_argument("--arm-damping", type=float, default=240.0)
    parser.add_argument(
        "--grip-contact-damping",
        type=float,
        default=RIGID_CONTACT_DAMPING,
        help="Contact damping for the finger and candy core materials [N s/m].",
    )
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument(
        "--iterations",
        type=int,
        default=argparse.SUPPRESS,
        help="VBD iterations per substep. Defaults to 15 for the sewn-stocking drop and 60 otherwise.",
    )
    parser.add_argument("--grasp-iterations", type=int, default=15, help="Cap VBD iterations during pickup and carry.")
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable physics graph capture for comparison."
    )
    parser.add_argument("--stocking-pattern", choices=("B", "sewn"), default="B")
    parser.add_argument("--mesh-rings", type=int, default=28)
    parser.add_argument("--mesh-sides", type=int, default=32)
    parser.add_argument("--areal-density", type=float, default=0.25, help="Cloth surface density [kg/m^2].")
    parser.add_argument("--stretch-stiffness", type=float, default=1500.0)
    parser.add_argument("--stretch-damping", type=float, default=0.01)
    parser.add_argument("--bend-stiffness", type=float, default=5.0e-7)
    parser.add_argument("--bend-damping", type=float, default=4.0e-3)
    parser.add_argument("--cuff-height", type=float, default=MOUTH[2], help="Supported cuff height above ground [m].")
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--particle-radius", type=float, default=0.002)
    parser.add_argument("--contact-stiffness", type=float, default=2000.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/christmas_atelier.usd"))
    parser.add_argument("--metrics", type=Path, default=Path("/tmp/christmas_atelier.json"))
    parser.add_argument("--export-assets", type=Path, help="Export USD art assets and exit without simulating.")
    parser.add_argument(
        "--check", action="store_true", help="Exit unsuccessfully unless the complete physical sequence passes."
    )
    args = parser.parse_args()
    if not hasattr(args, "iterations"):
        args.iterations = 15 if args.mode == "drop" and args.stocking_pattern == "sewn" else 60
    if args.export_assets:
        export_assets(args.export_assets, sewn=args.stocking_pattern == "sewn")
        return
    if not 1 <= args.candies <= 5:
        parser.error("--candies must be between 1 and 5.")
    if args.substeps < 1 or args.iterations < 1 or args.grasp_iterations < 1 or args.num_frames < 0:
        parser.error("Substeps and iterations must be positive; num-frames must be nonnegative.")
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
    mesh = panel_stocking_mesh if args.stocking_pattern == "sewn" else stocking_mesh
    lowest = float(mesh(args.mesh_rings, args.mesh_sides)[0][:, 2].min())
    if not math.isfinite(args.cuff_height) or args.cuff_height + lowest <= TABLE_TOP:
        parser.error("--cuff-height must leave the undeformed stocking above the bench.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with wp.ScopedDevice(args.device):
        scene = Atelier(args)
        frames = args.num_frames or math.ceil(scene.duration * FPS)
        if args.viewer == "gl":
            viewer = newton.viewer.ViewerGL(num_frames=frames)
        elif args.viewer == "usd":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            viewer = newton.viewer.ViewerUSD(output_path=str(args.output), num_frames=frames)
        else:
            viewer = newton.viewer.ViewerNull(num_frames=frames)
        viewer.set_model(scene.model)
        viewer.set_camera(pos=wp.vec3(1.65, -1.9, 1.65), pitch=-18.0, yaw=124.0)
        try:
            while scene.frame < frames and viewer.is_running():
                if viewer.should_step():
                    scene.step()
                viewer.begin_frame(scene.sim_time)
                viewer.log_state(scene.state)
                viewer.end_frame()
        finally:
            viewer.close()
            report = scene.report()
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            logger.info("Measurements: %s", args.metrics)
        if args.check and not report["passed"]:
            raise SystemExit("Physical sequence did not pass; inspect the measurements JSON.")


if __name__ == "__main__":
    main()
