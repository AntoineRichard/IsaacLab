# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The glass ornament with a rigid SDF ring, and the fixed-base Unitree G1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import warp as wp

RING_INNER_RADIUS = 0.020  # [m] clears a Dex3 finger (~20 mm) plus a twig
RING_TUBE_RADIUS = 0.0025  # [m]
CAP_RADIUS = 0.007  # [m]
CAP_HALF_HEIGHT = 0.004  # [m]
SPHERE_RADIUS = 0.030  # [m]
ORNAMENT_MASS = 0.035  # [kg]
RING_DENSITY = 1500.0  # [kg/m^3] lacquered wire
CAP_DENSITY = 2700.0  # [kg/m^3] aluminium
RIGID_CONTACT_STIFFNESS = 1.0e4  # [N/m] the atelier's qualified light-object contact
RIGID_CONTACT_DAMPING = 30.0  # [N s/m]
RING_FRICTION = 0.8


@dataclass
class Ornament:
    body: int
    ring_shape: int
    ring_center_local: np.ndarray  # (3,) body frame [m]
    ring_axis_local: np.ndarray  # (3,) unit normal of the ring plane, body frame
    ring_inner_radius: float  # [m]


def ring_mesh(
    inner_diameter: float = 2.0 * RING_INNER_RADIUS,
    tube_radius: float = RING_TUBE_RADIUS,
    major_segments: int = 48,
    minor_segments: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """A closed torus in the local XZ plane (axis +Y), centred on the origin."""
    major = 0.5 * inner_diameter + tube_radius
    u = np.linspace(0.0, 2.0 * math.pi, major_segments, endpoint=False)
    v = np.linspace(0.0, 2.0 * math.pi, minor_segments, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radial = major + tube_radius * np.cos(vv)
    vertices = np.stack([radial * np.cos(uu), tube_radius * np.sin(vv), radial * np.sin(uu)], axis=-1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(major_segments), np.arange(minor_segments), indexing="ij")
    a = i * minor_segments + j
    b = ((i + 1) % major_segments) * minor_segments + j
    c = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
    d = i * minor_segments + (j + 1) % minor_segments
    triangles = np.concatenate([np.stack([a, c, b], -1), np.stack([a, d, c], -1)]).reshape(-1, 3)
    return vertices.astype(np.float32), triangles.astype(np.int32)


def _contact_cfg(density: float, mu: float = 0.5) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(density=density, ke=RIGID_CONTACT_STIFFNESS, kd=RIGID_CONTACT_DAMPING, mu=mu)


def add_ornament(builder: newton.ModelBuilder, xform: wp.transform, color: tuple[float, float, float]) -> Ornament:
    """A free rigid ornament: ring at the body origin, cap and glass sphere hanging below it."""
    vertices, triangles = ring_mesh()
    mesh = newton.Mesh(vertices, triangles.flatten())
    mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.004, 0.004), margin=0.002)
    ring_volume = 2.0 * math.pi * (RING_INNER_RADIUS + RING_TUBE_RADIUS) * math.pi * RING_TUBE_RADIUS**2
    cap_volume = math.pi * CAP_RADIUS**2 * 2.0 * CAP_HALF_HEIGHT
    sphere_volume = 4.0 / 3.0 * math.pi * SPHERE_RADIUS**3
    sphere_density = (ORNAMENT_MASS - RING_DENSITY * ring_volume - CAP_DENSITY * cap_volume) / sphere_volume

    body = builder.add_body(xform=xform, label="ornament")
    ring_shape = builder.add_shape_mesh(
        body, mesh=mesh, cfg=_contact_cfg(RING_DENSITY, RING_FRICTION), color=(0.85, 0.75, 0.35), label="ornament_ring"
    )
    ring_bottom = -(RING_INNER_RADIUS + 2.0 * RING_TUBE_RADIUS)
    cap_z = ring_bottom - CAP_HALF_HEIGHT
    builder.add_shape_cylinder(
        body,
        xform=wp.transform(wp.vec3(0.0, 0.0, cap_z), wp.quat_identity()),
        radius=CAP_RADIUS,
        half_height=CAP_HALF_HEIGHT,
        cfg=_contact_cfg(CAP_DENSITY),
        color=(0.85, 0.75, 0.35),
        label="ornament_cap",
    )
    builder.add_shape_sphere(
        body,
        xform=wp.transform(wp.vec3(0.0, 0.0, cap_z - CAP_HALF_HEIGHT - SPHERE_RADIUS), wp.quat_identity()),
        radius=SPHERE_RADIUS,
        cfg=_contact_cfg(sphere_density),
        color=color,
        label="ornament_glass",
    )
    return Ornament(
        body=body,
        ring_shape=ring_shape,
        ring_center_local=np.zeros(3),
        ring_axis_local=np.array([0.0, 1.0, 0.0]),
        ring_inner_radius=RING_INNER_RADIUS,
    )


# ----------------------------------------------------------------------------------------------
# Unitree G1 with Dex3 hands, fixed lower body
# ----------------------------------------------------------------------------------------------

G1_ASSET_REVISION = "a96f0973b6ae69c90609f9fafef9d2c1db2d6431"
G1_URDF = "urdf/g1_29dof_with_hand_rev_1_0.urdf"
HOOK_LINK = "right_hand_index_0_link"
# Top of the proximal index phalanx, mid-length: the ring's inner top edge rests here. The link's
# +X runs along the finger and +Z is the finger's back when the palm faces down.
HOOK_OFFSET = wp.vec3(0.025, 0.0, 0.0129)
ARM_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# Index straight (the hook); thumb and middle curled to 80 % of their range, out of the ring's way.
HAND_POSE = {
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_0_joint": 0.8 * 1.57079632,
    "right_hand_middle_1_joint": 0.8 * 1.74532925,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.8 * -1.04719755,
    "right_hand_thumb_2_joint": 0.8 * -1.74532925,
}
RIGHT_HAND_LINKS = ("right_hand_", "right_wrist_yaw_link")
LEG_KE, LEG_KD = 5.0e4, 500.0
HAND_KE, HAND_KD = 1500.0, 8.0


def g1_urdf() -> Path:
    """The pinned G1 URDF from Newton Assets (downloaded on first use)."""
    return newton.utils.download_asset("unitree_g1", ref=G1_ASSET_REVISION) / G1_URDF


def _short(label: str) -> str:
    return label.split("/")[-1]


def robot_builder(
    urdf: Path, pelvis_xform: wp.transform, arm_ke: float = 32000.0, arm_kd: float = 240.0
) -> newton.ModelBuilder:
    """Import the G1 with its pelvis welded to the world and only the right hand colliding.

    ``collapse_fixed_joints`` merges massless URDF frames into physical links and the welded
    pelvis into the world, so the legs, waist and arms hang off the world body. Legs are held
    straight by stiff PD; the waist and arms by the arm gains; the hand by soft finger gains.
    """
    builder = newton.ModelBuilder()
    builder.add_urdf(str(urdf), floating=False, xform=pelvis_xform, enable_self_collisions=False)
    builder.collapse_fixed_joints()
    if any(int(t) == int(newton.JointType.FREE) for t in builder.joint_type):
        raise RuntimeError("G1 import kept a free base joint; the lower body must be fixed")

    for joint, label in enumerate(builder.joint_label):
        name = _short(label)
        q, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        if "hip" in name or "knee" in name or "ankle" in name:
            ke, kd = LEG_KE, LEG_KD
        elif "hand" in name:
            ke, kd = HAND_KE, HAND_KD
        else:
            ke, kd = arm_ke, arm_kd
        value = HAND_POSE.get(name, builder.joint_q[q])
        builder.joint_q[q] = value
        builder.joint_target_q[q] = value
        builder.joint_target_ke[dof] = ke
        builder.joint_target_kd[dof] = kd

    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for shape, body in enumerate(builder.shape_body):
        hand = body >= 0 and _short(builder.body_label[body]).startswith(RIGHT_HAND_LINKS)
        if not hand:
            builder.shape_flags[shape] &= ~collide
        builder.shape_material_ke[shape] = RIGID_CONTACT_STIFFNESS
        builder.shape_material_kd[shape] = RIGID_CONTACT_DAMPING
        builder.shape_material_mu[shape] = 0.5
    return builder


def body_index(model_or_builder, link: str) -> int:
    return next(i for i, label in enumerate(model_or_builder.body_label) if _short(label) == link)


def solve_waypoints(
    builder: newton.ModelBuilder, hook_targets: np.ndarray, hook_rotations: np.ndarray, device
) -> tuple[newton.Model, int, np.ndarray]:
    """Solve hook-point IK on a separate reference model; only waist and right-arm joints move.

    ``hook_rotations`` are world orientations (x, y, z, w) of ``HOOK_LINK``. Returns the
    reference model, the hook body index and one full joint-coordinate vector per waypoint.
    Raises if any waypoint misses by more than 3 mm.
    """
    from newton import ik

    model = builder.finalize(device=device)
    hook = body_index(model, HOOK_LINK)
    position = ik.IKObjectivePosition(
        link_index=hook,
        link_offset=HOOK_OFFSET,
        target_positions=wp.array([hook_targets[0]], dtype=wp.vec3, device=device),
    )
    rotation = ik.IKObjectiveRotation(
        link_index=hook,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([hook_rotations[0]], dtype=wp.vec4, device=device),
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
    rest = model.joint_q.numpy().copy()
    free = np.zeros_like(rest, dtype=bool)
    for joint, label in enumerate(model.joint_label):
        if _short(label) in ARM_JOINTS:
            free[builder.joint_q_start[joint]] = True
    q = wp.array(rest.reshape(1, -1), dtype=wp.float32, device=device)
    state = model.state()
    solutions = []
    for target, quat in zip(hook_targets, hook_rotations, strict=True):
        position.set_target_position(0, wp.vec3(*map(float, target)))
        rotation.set_target_rotation(0, wp.vec4(*map(float, quat)))
        for _ in range(20):
            solver.step(q, q, iterations=10)
            values = q.numpy().ravel()
            values[~free] = rest[~free]
            q.assign(values.reshape(1, -1))
        newton.eval_fk(model, q.flatten(), model.joint_qd, state)
        hook_world = np.array(wp.transform_point(wp.transform(*state.body_q.numpy()[hook]), HOOK_OFFSET))
        error = float(np.linalg.norm(hook_world - target))
        if error > 0.003:
            raise RuntimeError(f"Unreachable hook waypoint {np.round(target, 3).tolist()}: IK error {error:.4f} m")
        solutions.append(q.numpy().ravel().copy())
    return model, hook, np.array(solutions, dtype=np.float32)
