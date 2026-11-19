# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workcell and scripted control for the Isaac Lab Christmas atelier."""

from __future__ import annotations

import math

import newton
import newton.ik as ik
import newton.utils
import numpy as np
import warp as wp
from isaaclab_newton.physics import VBDSolver, VBDSolverCfg
from newton.solvers import SolverVBD

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .assets import (
    CANDY_CAPSULE_HALF_LENGTH,
    CANDY_CORE_RADIUS,
    CREAM,
    GOLD,
    GREEN,
    RED,
    candy_mesh,
    stocking_mesh,
)

TABLE_TOP = 0.72
MOUTH = (0.48, 0.22, 1.20)
FPS = 60
SUBSTEPS = 12
PARTICLE_RADIUS = 0.002
CANDY_POSITIONS = [(0.45 + (index - 1) * 0.075, -0.23, TABLE_TOP + 0.028) for index in range(3)]
TCP_OFFSET = wp.vec3(0.0, 0.0, 0.1034)
GRIP_FRICTION = 1.5
# Resolve the light candy/rubber contact without the stiff, underdamped pinch that ejects it.
RIGID_CONTACT_STIFFNESS = 1.0e4
RIGID_CONTACT_DAMPING = 30.0
FRANKA_ASSET_REVISION = "a96f0973b6ae69c90609f9fafef9d2c1db2d6431"
FRANKA_ARM_JOINTS = tuple(f"fr3_joint{index}" for index in range(1, 8))
FRANKA_FINGER_JOINTS = ("fr3_finger_joint1", "fr3_finger_joint2")
FRANKA_JOINTS = (*FRANKA_ARM_JOINTS, *FRANKA_FINGER_JOINTS)
FRANKA_HOME = (0.0, -0.5, 0.0, -2.1, 0.0, 1.6, math.pi / 4, 0.04, 0.04)


class _AtelierVBDSolver(VBDSolver):
    """Initialize the authored posture and solve the loose candies after the robot."""

    @classmethod
    def prepare_builder(cls, cfg: VBDSolverCfg, builder: newton.ModelBuilder) -> None:
        super().prepare_builder(cfg, builder)
        candy_bodies = {index for index, label in enumerate(builder.body_label) if label.startswith("candy_")}
        # The candy contact solve sees the updated fingertip poses when these groups run separately.
        body_groups = []
        for group in builder.body_color_groups:
            robot_bodies = [body for body in group if body not in candy_bodies]
            if robot_bodies:
                body_groups.append(np.asarray(robot_bodies, dtype=np.int32))
        body_groups.append(np.asarray(sorted(candy_bodies), dtype=np.int32))
        builder.body_color_groups = body_groups

    def build(self, model: newton.Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverVBD:
        # VBD captures joint rest transforms when constructed, before Isaac Lab initializes the live state.
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        return super().build(model, mode, env_template)

    def create_collision_pipeline(self, **kwargs) -> newton.CollisionPipeline:
        # The Newton revision used by this demo names the detection-only distance a gap.
        kwargs["soft_contact_gap"] = kwargs.pop("soft_contact_margin")
        return super().create_collision_pipeline(**kwargs)


@configclass
class AtelierVBDSolverCfg(VBDSolverCfg):
    """VBD settings for the stocking mesh and individual candy bodies."""

    class_type: type = _AtelierVBDSolver
    particle_self_contact_radius: float | None = None
    particle_self_contact_gap: float = 0.003
    particle_collision_detection_interval: int | None = None
    particle_enable_tile_solve: bool = True


def _franka_joint_ids(builder: newton.ModelBuilder) -> list[int]:
    """Resolve joint labels within the imported FR3 and Franka hand."""
    by_name = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(builder.joint_label)}
    missing = set(FRANKA_JOINTS) - by_name.keys()
    if missing:
        raise ValueError(f"The Franka URDF is missing joints: {', '.join(sorted(missing))}")
    return [by_name[name] for name in FRANKA_JOINTS]


def _robot_builder() -> newton.ModelBuilder:
    """Import the complete Franka chain and explicitly drive both fingers for VBD."""
    urdf = newton.utils.download_asset("franka_emika_panda", ref=FRANKA_ASSET_REVISION) / "urdf/fr3_franka_hand.urdf"
    builder = newton.ModelBuilder()
    builder.add_urdf(
        str(urdf),
        floating=False,
        xform=wp.transform((0.0, 0.0, TABLE_TOP), wp.quat_identity()),
        enable_self_collisions=False,
    )
    # A massless URDF frame becomes immovable in a maximal-coordinate solve. Merge those
    # frames into their physical links, retaining the massive hand as the TCP parent.
    builder.collapse_fixed_joints(joints_to_keep=["fr3/fr3_hand_joint"])
    # VBD does not solve the URDF mimic equality. Give each physical finger its own target.
    for joint, name, position in zip(_franka_joint_ids(builder), FRANKA_JOINTS, FRANKA_HOME, strict=True):
        dof = builder.joint_qd_start[joint]
        is_finger = name in FRANKA_FINGER_JOINTS
        builder.joint_target_ke[dof] = 1500.0 if is_finger else 32000.0
        builder.joint_target_kd[dof] = 8.0 if is_finger else 240.0
        builder.joint_q[builder.joint_q_start[joint]] = position
    # Limit robot-cloth contacts to the fingertips.
    finger_bodies = {
        index for index, label in enumerate(builder.body_label) if label.endswith(("leftfinger", "rightfinger"))
    }
    for shape, body in enumerate(builder.shape_body):
        if body not in finger_bodies:
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        builder.shape_material_mu[shape] = GRIP_FRICTION if body in finger_bodies else 0.5
        builder.shape_material_ke[shape] = RIGID_CONTACT_STIFFNESS
        builder.shape_material_kd[shape] = RIGID_CONTACT_DAMPING
    return builder


def _solve_waypoints(
    builder: newton.ModelBuilder, positions: np.ndarray, fingers: np.ndarray, finger_coords: list[int], device: str
) -> np.ndarray:
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
        values[finger_coords] = opening
        q.assign(values.reshape(1, -1))
        newton.eval_fk(model, q.flatten(), model.joint_qd, state)
        error = np.linalg.norm(
            np.array(wp.transform_point(wp.transform(*state.body_q.numpy()[tcp]), TCP_OFFSET)) - target
        )
        if error > 0.003:
            raise RuntimeError(f"Unreachable TCP waypoint {target.tolist()}: IK error {error:.4f} m")
        targets.append(values.copy())
    return np.array(targets, dtype=np.float32)


def _choreography() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan the three pickups and releases, with a hold before each lift and after each drop."""
    home = (0.38, -0.02, 1.22)
    times, positions, openings = [0.0], [home], [0.04]

    def arrive(duration: float, position: tuple[float, float, float], opening: float) -> None:
        times.append(times[-1] + duration)
        positions.append(position)
        openings.append(opening)

    mouth_x, mouth_y, mouth_z = MOUTH
    above_stocking = (mouth_x, mouth_y, mouth_z + 0.15)
    release = (mouth_x, mouth_y, mouth_z + 0.075)
    arrive(1.0, home, 0.04)
    for pick in CANDY_POSITIONS:
        above_tray = (pick[0], pick[1], 1.25)
        arrive(2.0, above_tray, 0.04)
        arrive(2.0, pick, 0.04)
        arrive(0.8, pick, 0.010)
        arrive(2.0, above_tray, 0.010)
        arrive(2.5, above_stocking, 0.010)
        arrive(1.0, release, 0.010)
        arrive(0.8, release, 0.04)
        arrive(0.8, release, 0.04)
        arrive(1.0, above_stocking, 0.04)
    arrive(2.0, home, 0.04)
    arrive(3.0, home, 0.04)
    return np.array(times), np.array(positions), np.array(openings)


def _sample_motion(time: float, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate targets with zero velocity and acceleration at each waypoint."""
    segment = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
    alpha = np.clip((time - times[segment]) / (times[segment + 1] - times[segment]), 0.0, 1.0)
    blend = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
    return targets[segment] + blend * (targets[segment + 1] - targets[segment])


def _add_workcell_props(builder: newton.ModelBuilder) -> None:
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
    mouth_x, mouth_y, mouth_z = MOUTH
    box("stand_base", (mouth_x, mouth_y + 0.15, TABLE_TOP + 0.012), (0.12, 0.07, 0.012), GOLD)
    post_bottom = TABLE_TOP + 0.024
    post_top = mouth_z + 0.01
    box(
        "stand_post",
        (mouth_x, mouth_y + 0.15, (post_bottom + post_top) / 2),
        (0.008, 0.008, (post_top - post_bottom) / 2),
        GOLD,
    )
    box("stand_arm", (mouth_x, mouth_y + 0.105, mouth_z), (0.008, 0.045, 0.008), GOLD)
    # Match the elliptical cloth mouth; the rim is visual, supported particles enforce it.
    for segment in range(32):
        angles = np.array([segment, segment + 1]) * 2 * np.pi / 32
        points = np.column_stack(
            [
                mouth_x + 0.075 * np.cos(angles),
                mouth_y + 0.0585 * np.sin(angles),
                np.full(2, mouth_z),
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


def _add_stocking(builder: newton.ModelBuilder) -> None:
    """Add the sewn cloth shell and pin its cuff to the hoop."""
    vertices, faces, colors, cuff = stocking_mesh()
    builder.add_cloth_mesh(
        pos=wp.vec3(*MOUTH),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices.tolist(),
        indices=faces.ravel().tolist(),
        density=0.25,
        tri_ke=1500.0,
        tri_ka=1500.0,
        tri_kd=0.01,
        edge_ke=5.0e-7,
        edge_kd=4.0e-3,
        particle_radius=PARTICLE_RADIUS,
        color=colors,
        validate_mesh=True,
        label="christmas_stocking",
    )
    for particle in cuff:
        builder.particle_mass[int(particle)] = 0.0


def _add_candies(builder: newton.ModelBuilder) -> None:
    """Give each wrapped candy a capsule grasp collider and a closed cloth collider."""
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
    visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    white = np.all(np.isclose(colors, CREAM), axis=1)
    for index, (position, wrapper_color) in enumerate(zip(CANDY_POSITIONS, (RED, GREEN, GOLD), strict=True)):
        body = builder.add_body(
            xform=wp.transform(position, wp.quat_identity()),
            mass=0.02,
            inertia=wp.mat33(2.0e-6, 0, 0, 0, 2.0e-6, 0, 0, 0, 2.0e-6),
            label=f"candy_{index}",
        )
        builder.add_shape_capsule(
            body,
            radius=CANDY_CORE_RADIUS,
            half_height=CANDY_CAPSULE_HALF_LENGTH,
            xform=wp.transform((0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=1.2,
                ke=RIGID_CONTACT_STIFFNESS,
                kd=RIGID_CONTACT_DAMPING,
                gap=0.0005,
                is_visible=False,
                has_particle_collision=False,
            ),
            label=f"candy_core_{index}",
        )
        builder.add_shape_mesh(body, mesh=wrapper, cfg=wrapper_cfg, label=f"wrapper_contact_{index}")
        for mask, color in ((white, CREAM), (~white, wrapper_color)):
            mesh = newton.Mesh(vertices, faces[mask].ravel(), compute_inertia=False)
            builder.add_shape_mesh(
                body, mesh=mesh, cfg=visual_cfg, color=color, label=f"wrapper_{index}_{bool(mask[0])}"
            )


class Atelier:
    """Build the workcell and run the prescribed Franka motion through Isaac Lab."""

    def __init__(self, sim: sim_utils.SimulationContext):
        self.sim = sim
        self.sim_time = 0.0
        self.times, positions, openings = _choreography()

        robot = _robot_builder()
        self.reference_coords = [robot.joint_q_start[joint] for joint in _franka_joint_ids(robot)]
        finger_coords = self.reference_coords[len(FRANKA_ARM_JOINTS) :]
        self.targets = _solve_waypoints(robot, positions, openings, finger_coords, sim.device)
        robot.joint_q[:] = self.targets[0].tolist()

        builder = sim.backend_manager.builder
        builder.add_builder(robot)
        _add_workcell_props(builder)
        _add_stocking(builder)
        _add_candies(builder)
        self.drive_dofs = [builder.joint_qd_start[joint] for joint in _franka_joint_ids(builder)]
        sim.reset()

        self.joint_targets = sim.backend_manager.backend.control.joint_target_q
        self.control_targets = self.joint_targets.numpy()

    def step(self) -> None:
        """Apply one frame of scripted targets and advance the physics."""
        target = _sample_motion(self.sim_time, self.times, self.targets).astype(np.float32)
        self.control_targets[self.drive_dofs] = target[self.reference_coords]
        self.joint_targets.assign(self.control_targets)
        # Refresh rigid contacts and the cloth BVH at each physics tick.
        for substep in range(SUBSTEPS):
            self.sim.step(render=substep == SUBSTEPS - 1)
        self.sim_time += 1.0 / FPS
