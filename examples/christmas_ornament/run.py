# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A fixed-base Agile One hangs a glass ornament on a VBD Christmas tree.

Robot, ornament and tree share one ``SolverVBD``. The ornament starts hanging from the robot's
extended right index finger; the robot threads the ornament's rigid ring over the tip of one
outer branch, lowers its finger so the branch takes the weight, and withdraws. Nothing welds,
attaches or kinematically drives the ornament.

``--mode probe`` uses one standalone branch on a post instead of the full tree, to qualify the
hand-over quickly. ``--mode hang`` (default) is the full scene.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp

from examples.christmas_ornament.assets import (
    FINGER_CAPSULE,
    FINGER_LINKS,
    HOOK_LINK,
    HOOK_OFFSET,
    PELVIS_HEIGHT,
    RING_INNER_RADIUS,
    RING_TUBE_RADIUS,
    add_ornament,
    agile_one_urdf,
    body_index,
    hook_rotation,
    robot_builder,
    solve_home,
    solve_waypoints,
)
from examples.christmas_ornament.motion import (
    PHASES,
    choreography,
    creep,
    densify,
    finger_rotation,
    min_distance_points_to_capsule,
    ring_along_twig,
    ring_encircles,
    sample_linear,
)
from examples.christmas_ornament.tree import (
    NEEDLE_RADIUS,
    Wind,
    build_branch,
    build_tree,
    configure_builder,
    make_needles,
    needle_meshes,
    place_needles,
)

FPS = 60
DIP_GATE = 0.003  # [m] visible but modest: a stiffer tip keeps the ring from creeping off
CREEP_GATE = 0.002  # [m] the ring may not walk along the twig during admire
TIP_MARGIN_GATE = 0.01  # [m] twig left between the ring and the end of its segment
# Where the target branch's hang point sits horizontally (world frame; the tree is laid out for a
# robot near the origin facing +x). The tree stands on the ground, so the height is the tree's
# own; the probe's standalone branch is placed at exactly the same point.
HANG_XY = np.array([0.40, -0.10])
BRANCH_YAW = math.pi  # the target branch points from its trunk back toward the robot
# Where the robot stands: only its right arm moves, so it stands where that arm alone can hold
# the finger level along the branch with its nail up.
ROBOT_XY = np.array([-0.10, -0.02])  # hang point 0.50 m ahead, 0.08 m right of the pelvis
ROBOT_YAW = 0.0  # [rad] about +z
HOME_LOCAL = np.array([0.38, -0.22, 0.92])  # hook point at rest in the robot's frame


def robot_xform() -> wp.transform:
    return wp.transform(
        wp.vec3(float(ROBOT_XY[0]), float(ROBOT_XY[1]), PELVIS_HEIGHT),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), ROBOT_YAW),
    )


def robot_to_world(local) -> np.ndarray:
    """A point in the robot's frame (origin under the pelvis on the ground, +x forward)."""
    c, s = math.cos(ROBOT_YAW), math.sin(ROBOT_YAW)
    x, y, z = local
    return np.array([ROBOT_XY[0] + c * x - s * y, ROBOT_XY[1] + s * x + c * y, z])


HOME = robot_to_world(HOME_LOCAL)  # hook point at rest, ornament hanging from the finger
UP = np.array([0.0, 0.0, 1.0])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.array(wp.quat_rotate(wp.quat(*q), wp.vec3(*v)))


def _point(body_q: np.ndarray, body: int, local) -> np.ndarray:
    return np.array(wp.transform_point(wp.transform(*body_q[body]), wp.vec3(*local)))


def _branch_root(whorl_xform_builder) -> np.ndarray:
    """Hang point of a branch built at the origin, i.e. the offset from its root to its hang point."""
    builder = newton.ModelBuilder()
    configure_builder(builder)
    target = whorl_xform_builder(builder)
    return _point(np.array(builder.body_q), target.hang_body, target.hang_local)


def clearance_window_start() -> float:
    """Finger-ring clearance is gated from the end of withdraw on: a finger that drags the ring
    is still next to it when the retreat starts."""
    names = [name for name, _ in PHASES]
    return float(sum(d for _, d in PHASES[: names.index("withdraw") + 1]))


def creep_window_start() -> float:
    """Ring creep is gated over the last 2 s of admire: settling after the drop is fine, drift
    still going at the end is the ring walking off the twig."""
    return float(sum(d for _, d in PHASES)) - 2.0


class OrnamentScene:
    """Scene assembly, scripted rollout and measurements for both modes."""

    def __init__(self, args: argparse.Namespace, needles_as_shapes: bool = False):
        """``needles_as_shapes`` bakes the needles into one visual-only mesh shape per segment, for
        USD-based renderers (``ViewerRTX``); otherwise the viewer draws them as instances."""
        self.args = args
        self.device = wp.get_device(args.device)
        self.sim_dt = 1.0 / FPS / args.substeps
        self.sim_time = 0.0
        self.frame = 0
        self.total_time = sum(d for _, d in PHASES)

        builder = newton.ModelBuilder()
        configure_builder(builder)

        # Robot first, so its joint coordinates are the first slice of joint_target_q.
        robot = self._robot_builder()
        self.robot_coords = robot.joint_coord_count
        self.home_rotation = finger_rotation(np.array([math.cos(BRANCH_YAW), math.sin(BRANCH_YAW), 0.0]))
        robot.joint_q[:] = solve_home(robot, HOME, self.home_rotation, device=self.device).tolist()
        self.reference_model, self.hook_ref, q_home = solve_waypoints(
            robot, HOME[None, :], self.home_rotation[None, :], device=self.device
        )
        self.q_home = q_home[0]
        reference_state = self.reference_model.state()
        newton.eval_fk(
            self.reference_model,
            wp.array(self.q_home, device=self.device),
            self.reference_model.joint_qd,
            reference_state,
        )
        home_body_q = reference_state.body_q.numpy()
        robot.joint_q[:] = self.q_home.tolist()
        robot.joint_target_q[: self.robot_coords] = self.q_home.tolist()
        for body in range(robot.body_count):
            robot.body_q[body] = wp.transform(*home_body_q[body])
        builder.add_builder(robot)
        self.hook_body = body_index(builder, HOOK_LINK)
        self.finger_bodies = [body_index(builder, name) for name in FINGER_LINKS]
        self.hand_shapes = [
            s
            for s in range(builder.shape_count)
            if builder.shape_body[s] >= 0 and builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ]
        builder.add_ground_plane()

        # The full tree's hang point, built at the origin, fixes where the target branch sits.
        def make_tree(b, origin=(0.0, 0.0, 0.0)):
            tree = build_tree(b, origin, BRANCH_YAW, branch_modulus=args.branch_modulus)
            self.tree = tree
            return tree.target

        tree_offset = _branch_root(make_tree)
        hang_world = np.array([*HANG_XY, tree_offset[2]])
        self.tree = None
        if args.mode == "probe":

            def make_branch(b, root=wp.vec3(0.0, 0.0, 0.0)):
                yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), BRANCH_YAW)
                return build_branch(b, wp.transform(root, yaw), branch_modulus=args.branch_modulus)[0]

            branch_offset = _branch_root(make_branch)
            self.target = make_branch(builder, wp.vec3(*(hang_world - branch_offset)))
        else:
            self.target = make_tree(builder, (*(HANG_XY - tree_offset[:2]), 0.0))
            if needles_as_shapes:
                visual = newton.ModelBuilder.ShapeConfig(
                    density=0.0, has_shape_collision=False, has_particle_collision=False
                )
                for body, mesh in needle_meshes(self.tree, np.random.default_rng(11)).items():
                    shape_mesh = newton.Mesh(mesh.vertices, mesh.triangles.flatten(), compute_inertia=False)
                    builder.add_shape_mesh(body, mesh=shape_mesh, cfg=visual, color=tuple(float(c) for c in mesh.color))
            self._filter_hand_from_tree(builder)

        # The ornament hangs from the finger at home: ring inner top on the hook point, ring
        # axis along the finger. Ornament local Y is the ring axis; the hook frame's X the finger.
        hook_xform = wp.transform(*home_body_q[self.hook_ref])
        hook_world = np.array(wp.transform_point(hook_xform, HOOK_OFFSET))
        hook_q = hook_rotation(home_body_q[self.hook_ref, 3:])
        orn_rot = wp.mul(wp.quat(*hook_q), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5 * math.pi))
        back = _quat_rotate(hook_q, UP)
        self.ornament = add_ornament(
            builder,
            wp.transform(wp.vec3(*(hook_world - (RING_INNER_RADIUS - 0.0005) * back)), orn_rot),
            (0.85, 0.1, 0.1),
        )

        builder.color(balance_colors=False)
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.iterations,
            rigid_joint_linear_ke=1.0e7,
            rigid_joint_angular_ke=1.0e6,
            rigid_body_contact_buffer_size=256,
            friction_epsilon=1.0e-3,
        )
        self.control_targets = self.control.joint_target_q.numpy()
        self.reference_state = self.reference_model.state()
        shape_body = self.model.shape_body.numpy()
        twig_shape = int(np.flatnonzero(shape_body == self.target.hang_body)[0])
        self.twig_half_length = float(self.model.shape_scale.numpy()[twig_shape][1])

        # Motion: hold home through settle; the rest is planned from the settled branch.
        self.times = np.array([0.0, PHASES[0][1]])
        self.q_targets = np.stack([self.q_home, self.q_home])
        self.phase_names = ["start", PHASES[0][0]]
        self.planned = False
        self.hook_positions = None

        self.wind = None
        if self.tree is not None:
            if args.wind > 0.0:
                self.wind = Wind(self.tree.wind_bodies, self.tree.wind_area, args.wind, self.device)
            self.needles = (
                []
                if needles_as_shapes
                else [
                    (
                        batch.half_length,
                        wp.array(batch.body, dtype=wp.int32, device=self.device),
                        wp.array(batch.local, dtype=wp.transform, device=self.device),
                        wp.array(batch.color, dtype=wp.vec3, device=self.device),
                        wp.empty(len(batch.body), dtype=wp.transform, device=self.device),
                    )
                    for batch in make_needles(self.tree, np.random.default_rng(11))
                ]
            )
            self.trunk_top0 = self.state.body_q.numpy()[self.tree.trunk_top_body, :3].copy()
        else:
            self.needles = []

        # Measurements.
        self.max_hook_error = 0.0
        self.hang_settled = None
        self.finite = True
        self.hung_frames = 0
        self.admire_frames = 0
        self.min_release_clearance = np.inf
        self.clearance_from = clearance_window_start()
        self.creep_from = creep_window_start()
        self.creep_samples: list[tuple[float, float]] = []
        self.tip_margin = None
        self.nontarget_contact_frames = 0
        self.max_trunk_disp = 0.0
        self.phase_samples = []
        self.last_phase = ""
        self.graph = None
        self.started = time.perf_counter()

    def _filter_hand_from_tree(self, builder: newton.ModelBuilder) -> None:
        """The hand touches only the ornament and the target branch."""
        target = set(self.target.segment_bodies) | set(self.target.hang_neighbours)
        tree_shapes = [
            s
            for s in range(builder.shape_count)
            if builder.shape_body[s] in set(self.tree.bodies) | set(self.tree.hangers)
        ]
        pairs = builder.shape_collision_filter_pairs
        for t in tree_shapes:
            if builder.shape_body[t] in target:
                continue
            for h in self.hand_shapes:
                pairs.append((min(h, t), max(h, t)))

    # ------------------------------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------------------------------

    def phase_at(self, t: float) -> str:
        ends = np.cumsum([d for _, d in PHASES])
        index = int(np.searchsorted(ends, t, side="right"))
        return PHASES[min(index, len(PHASES) - 1)][0]

    def _hang_frame(self, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hang = _point(body_q, self.target.hang_body, self.target.hang_local)
        tangent = _quat_rotate(body_q[self.target.hang_body, 3:], self.target.tangent_local)
        return hang, tangent / np.linalg.norm(tangent)

    def _plan(self) -> None:
        """Solve the remaining waypoints from the settled branch (end of the settle phase)."""
        body_q = self.state.body_q.numpy()
        hang, tangent = self._hang_frame(body_q)
        self.hang_settled = hang
        # Where the hanging ring actually rides on the finger (contact, not the nominal hook point).
        hook = _point(body_q, self.hook_body, HOOK_OFFSET)
        back = _quat_rotate(hook_rotation(body_q[self.hook_body, 3:]), UP)
        ring_top = (
            _point(body_q, self.ornament.body, self.ornament.ring_center_local) + self.ornament.ring_inner_radius * back
        )
        ring_top_offset = float(np.dot(ring_top - hook, back))
        self.ring_top_offset = ring_top_offset
        times, positions, rotations, phases = choreography(
            hang,
            tangent,
            HOME,
            ring_inner_radius=self.ornament.ring_inner_radius,
            twig_radius=self.target.twig_radius,
            ring_top_offset=ring_top_offset,
        )
        dense_t, dense_p, dense_q, names = densify(times, positions, rotations, phases, HOME, samples=8)
        _, _, q = solve_waypoints(self._ik_robot(), dense_p[1:], dense_q[1:], device=self.device)
        self.times = dense_t
        self.q_targets = np.vstack([self.q_home, q])
        self.hook_positions = dense_p
        self.phase_names = names
        self.planned = True

    def _robot_builder(self) -> newton.ModelBuilder:
        return robot_builder(self.args.robot_urdf or agile_one_urdf(), robot_xform())

    def _ik_robot(self) -> newton.ModelBuilder:
        robot = self._robot_builder()
        robot.joint_q[:] = self.q_home.tolist()
        return robot

    def _advance_physics(self) -> None:
        for _ in range(self.args.substeps):
            self.state.clear_forces()
            if self.wind is not None:
                self.wind.apply(self.state, self.sim_dt)
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, self.sim_dt)
            self.state, self.next_state = self.next_state, self.state

    def step(self) -> None:
        if not self.planned and self.sim_time >= PHASES[0][1] - 1e-9:
            self._plan()
        target = sample_linear(self.sim_time, self.times, self.q_targets).astype(np.float32)
        self.control_targets[: self.robot_coords] = target
        self.control.joint_target_q.assign(self.control_targets)
        if self.graph is None:
            self._advance_physics()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += 1.0 / FPS
        self.frame += 1
        if self.frame == 1 and self.device.is_cuda and not self.args.disable_cuda_graph and self.args.substeps % 2 == 0:
            with wp.ScopedCapture(device=self.device) as capture:
                self._advance_physics()
            self.graph = capture.graph  # recorded only; replays from the next frame
        self._measure(target)

    def _measure(self, target: np.ndarray) -> None:
        body_q = self.state.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            self.finite = False
            return
        newton.eval_fk(
            self.reference_model,
            wp.array(target, device=self.device),
            self.reference_model.joint_qd,
            self.reference_state,
        )
        reference_hook = _point(self.reference_state.body_q.numpy(), self.hook_ref, HOOK_OFFSET)
        live_hook = _point(body_q, self.hook_body, HOOK_OFFSET)
        self.max_hook_error = max(self.max_hook_error, float(np.linalg.norm(live_hook - reference_hook)))

        ring_center = _point(body_q, self.ornament.body, self.ornament.ring_center_local)
        ring_axis = _quat_rotate(body_q[self.ornament.body, 3:], self.ornament.ring_axis_local)
        half = self.twig_half_length
        twig_a = _point(body_q, self.target.hang_body, (0.0, 0.0, -half))
        twig_b = _point(body_q, self.target.hang_body, (0.0, 0.0, half))
        hung = ring_encircles(ring_center, ring_axis, self.ornament.ring_inner_radius, twig_a, twig_b)
        phase = self.phase_at(self.sim_time)
        if phase == "admire":
            self.admire_frames += 1
            self.hung_frames += int(hung)
            along = ring_along_twig(ring_center, twig_a, twig_b)
            if self.sim_time >= self.creep_from:
                self.creep_samples.append((self.sim_time, along))
            self.tip_margin = 2.0 * half - along
        if self.tree is not None:
            self.nontarget_contact_frames += int(self._ornament_touches_other_tree())
        if self.sim_time >= self.clearance_from:
            self.min_release_clearance = min(
                self.min_release_clearance, self._ring_finger_clearance(body_q, ring_center, ring_axis)
            )
        if self.tree is not None:
            self.max_trunk_disp = max(
                self.max_trunk_disp, float(np.linalg.norm(body_q[self.tree.trunk_top_body, :3] - self.trunk_top0))
            )
        if phase != self.last_phase:
            self.phase_samples.append(
                {
                    "phase": phase,
                    "time_s": round(self.sim_time, 3),
                    "hook_error_m": float(np.linalg.norm(live_hook - reference_hook)),
                    "ring_center": ring_center.round(4).tolist(),
                    "hang_point": _point(body_q, self.target.hang_body, self.target.hang_local).round(4).tolist(),
                    "hung": bool(hung),
                }
            )
            self.last_phase = phase

    def _ornament_touches_other_tree(self) -> bool:
        """Whether the ornament touches tree parts other than the target branch this frame."""
        if not hasattr(self, "_other_tree"):
            self._shape_body = self.model.shape_body.numpy()
            target = set(self.target.segment_bodies) | set(self.target.hang_neighbours)
            self._other_tree = np.array(sorted((set(self.tree.bodies) | set(self.tree.hangers)) - target))
        n = int(self.contacts.rigid_contact_count.numpy()[0])
        a = self._shape_body[self.contacts.rigid_contact_shape0.numpy()[:n]]
        b = self._shape_body[self.contacts.rigid_contact_shape1.numpy()[:n]]
        orn = self.ornament.body
        return bool(np.any(((a == orn) & np.isin(b, self._other_tree)) | ((b == orn) & np.isin(a, self._other_tree))))

    def _ring_finger_clearance(self, body_q, ring_center, ring_axis) -> float:
        """Distance between the ring tube and the index finger, both as swept primitives."""
        e1 = np.cross(ring_axis, UP)
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(ring_axis, np.array([1.0, 0.0, 0.0]))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ring_axis, e1)
        major = self.ornament.ring_inner_radius + RING_TUBE_RADIUS
        angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
        ring = ring_center + major * (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)
        a_local, b_local, radius = FINGER_CAPSULE
        distance = np.inf
        for body in self.finger_bodies:
            a, b = _point(body_q, body, a_local), _point(body_q, body, b_local)
            distance = min(distance, min_distance_points_to_capsule(ring, a, b, radius) - RING_TUBE_RADIUS)
        return float(distance)

    def render(self, viewer) -> None:
        viewer.begin_frame(self.sim_time)
        viewer.log_state(self.state)
        for size, (half, body, local, color, xform) in enumerate(self.needles):
            wp.launch(
                place_needles,
                dim=body.shape[0],
                inputs=[body, local, self.state.body_q],
                outputs=[xform],
                device=self.device,
            )
            viewer.log_shapes(f"/needles_{size}", newton.GeoType.CONE, (NEEDLE_RADIUS, half), xform, colors=color)
        viewer.end_frame()

    # ------------------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------------------

    def metrics(self) -> dict:
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()
        hang_end = _point(body_q, self.target.hang_body, self.target.hang_local) if self.finite else None
        return {
            "mode": self.args.mode,
            "newton_version": newton.__version__,
            "newton_path": newton.__file__,
            "warp_version": wp.config.version,
            "device": str(self.device),
            "substeps": self.args.substeps,
            "iterations": self.args.iterations,
            "branch_modulus_pa": self.args.branch_modulus,
            "sim_time_s": round(self.sim_time, 3),
            "wall_time_s": round(time.perf_counter() - self.started, 1),
            "finite": self.finite,
            "complete": self.sim_time >= self.total_time - 1e-6,
            "max_hook_error_m": self.max_hook_error,
            "ring_top_offset_m": getattr(self, "ring_top_offset", None),
            "dip_m": float(self.hang_settled[2] - hang_end[2])
            if self.hang_settled is not None and hang_end is not None
            else None,
            "hung": self.admire_frames > 0 and self.hung_frames == self.admire_frames,
            "hung_fraction": self.hung_frames / max(self.admire_frames, 1),
            "released_clearance_m": None if not np.isfinite(self.min_release_clearance) else self.min_release_clearance,
            "ornament_speed_end": float(np.linalg.norm(body_qd[self.ornament.body, :3])) if self.finite else None,
            "trunk_top_disp_m": self.max_trunk_disp if self.tree is not None else None,
            "admire_creep_m": creep(*np.array(self.creep_samples).T) if len(self.creep_samples) > 1 else None,
            "tip_margin_m": self.tip_margin,
            "nontarget_contacts": self.nontarget_contact_frames if self.tree is not None else None,
            "phase_samples": self.phase_samples,
        }

    def check(self, metrics: dict) -> list[str]:
        failures = []
        if not metrics["finite"]:
            return ["non-finite state"]
        if not metrics["complete"]:
            failures.append("sequence truncated")
        if metrics["max_hook_error_m"] >= 0.025:
            failures.append(f"hook tracking error {metrics['max_hook_error_m']:.4f} m >= 0.025")
        if not metrics["hung"]:
            failures.append(f"ornament not hung through admire ({metrics['hung_fraction']:.0%} of frames)")
        if metrics["released_clearance_m"] is None or metrics["released_clearance_m"] <= 0.005:
            failures.append(f"finger-ring clearance {metrics['released_clearance_m']} m <= 0.005")
        if metrics["dip_m"] is None or metrics["dip_m"] < DIP_GATE:
            failures.append(f"branch dip {metrics['dip_m']} m < {DIP_GATE}")
        if metrics["ornament_speed_end"] is None or metrics["ornament_speed_end"] > 0.05:
            failures.append(f"ornament still moving at {metrics['ornament_speed_end']} m/s")
        if metrics["admire_creep_m"] is None or metrics["admire_creep_m"] > CREEP_GATE:
            failures.append(f"ring creep along the twig during admire {metrics['admire_creep_m']} m > {CREEP_GATE}")
        if metrics["tip_margin_m"] is None or metrics["tip_margin_m"] < TIP_MARGIN_GATE:
            failures.append(f"ring {metrics['tip_margin_m']} m from the tip of its segment < {TIP_MARGIN_GATE}")
        if metrics["trunk_top_disp_m"] is not None and metrics["trunk_top_disp_m"] >= 0.05:
            failures.append(f"trunk top moved {metrics['trunk_top_disp_m']:.3f} m >= 0.05")
        return failures


def make_viewer(args: argparse.Namespace, num_frames: int):
    if args.viewer == "gl":
        return newton.viewer.ViewerGL()
    if args.viewer == "usd":
        return newton.viewer.ViewerUSD(output_path=str(args.output), fps=FPS)
    return newton.viewer.ViewerNull(num_frames=num_frames)  # its default stops at 1000 frames


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("probe", "hang"), default="hang")
    p.add_argument("--viewer", choices=("gl", "usd", "null"), default="gl")
    p.add_argument("--device", default=None)
    p.add_argument("--substeps", type=int, default=20)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--branch-modulus", type=float, default=None, help="Branch Young's modulus [Pa].")
    p.add_argument("--wind", type=float, default=0.0, help="Peak wind speed [m/s]; off by default.")
    p.add_argument("--num-frames", type=int, default=0, help="Truncate the sequence; 0 runs all of it.")
    p.add_argument("--check", action="store_true", help="Exit non-zero unless every gate passes.")
    p.add_argument("--metrics", type=Path, default=Path("/tmp/christmas_ornament.json"))
    p.add_argument("--output", type=Path, default=Path("/tmp/christmas_ornament.usd"))
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--robot-urdf", type=Path, default=None, help="Override the robot's URDF path.")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.branch_modulus is None:
        from examples.christmas_ornament.tree import BRANCH_YOUNG_MODULUS

        args.branch_modulus = BRANCH_YOUNG_MODULUS
    scene = OrnamentScene(args)
    total_frames = int(round(scene.total_time * FPS))
    limit = args.num_frames or total_frames
    viewer = make_viewer(args, limit)
    viewer.set_model(scene.model)
    while scene.frame < limit and viewer.is_running():
        scene.step()
        scene.render(viewer)
        if not scene.finite:
            break
    viewer.close()
    metrics = scene.metrics()
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    summary = {k: v for k, v in metrics.items() if k != "phase_samples"}
    print(json.dumps(summary, indent=2))
    if args.check:
        failures = scene.check(metrics)
        if failures:
            raise SystemExit("CHECK FAILED: " + "; ".join(failures))
        print("CHECK PASSED")


if __name__ == "__main__":
    main()
