# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The glass ornament with a rigid SDF ring, and the fixed-base Agile One with Wuji hands."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import warp as wp

RING_INNER_RADIUS = 0.030  # [m] room to thread a twig under the 17 mm Wuji fingertip
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
# Glass baubles are mirrored: renderers take roughness/metallic from mesh shapes only, so each
# ball is drawn as a visual mesh while its sphere keeps colliding, hidden.
BAUBLE_ROUGHNESS = 0.12
BAUBLE_METALLIC = 0.85
RIGID_GAP = 0.002  # [m] speculative contact gap, matching tree.configure_builder


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


def add_bauble_visual(
    builder: newton.ModelBuilder,
    body: int,
    xform: wp.transform,
    radius: float,
    color: tuple[float, float, float],
    roughness: float = BAUBLE_ROUGHNESS,
    metallic: float = BAUBLE_METALLIC,
) -> int:
    """A glossy visual-only sphere mesh (no collision, no mass) for a bauble on ``body``."""
    mesh = newton.Mesh.create_sphere(radius, num_latitudes=24, num_longitudes=32, compute_inertia=False)
    mesh.roughness = roughness
    mesh.metallic = metallic
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    return builder.add_shape_mesh(body, xform=xform, mesh=mesh, cfg=cfg, color=color)


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
    glass_xform = wp.transform(wp.vec3(0.0, 0.0, cap_z - CAP_HALF_HEIGHT - SPHERE_RADIUS), wp.quat_identity())
    glass_cfg = _contact_cfg(sphere_density)
    glass_cfg.is_visible = False
    builder.add_shape_sphere(body, xform=glass_xform, radius=SPHERE_RADIUS, cfg=glass_cfg, label="ornament_glass")
    add_bauble_visual(builder, body, glass_xform, SPHERE_RADIUS, color)
    return Ornament(
        body=body,
        ring_shape=ring_shape,
        ring_center_local=np.zeros(3),
        ring_axis_local=np.array([0.0, 1.0, 0.0]),
        ring_inner_radius=RING_INNER_RADIUS,
    )


# ----------------------------------------------------------------------------------------------
# Agile One SN with Wuji hands, fixed lower body
# ----------------------------------------------------------------------------------------------

AGILE_URDF_ENV = "AGILE_ONE_URDF"
PELVIS_HEIGHT = 0.8096  # [m] the asset's own initial root height, legs straight

# The hook is the right index finger, measured from its Wuji collision meshes. Each phalanx runs
# along its link's -Z, flexes about +X and has its nail toward -Y; the distal phalanx is 26 mm
# long. The arm moves alone, and it cannot hold the finger level with the nail up in front of the
# chest (the forearm would have to roll past its limit), so the ring rides on the finger's
# thumb side (+X), which faces up when the hand points forward thumb-up, as it rests.
HOOK_LINK = "r_index_finger_distal"
# Thumb side of the distal phalanx, 10 mm past the last knuckle (16 mm from the tip): the ring's
# inner top edge rests here.
HOOK_OFFSET = wp.vec3(0.0065, 0.0, -0.010)
# The choreography plans in a hook frame with +X down the finger and +Z out of the surface the
# ring rests on, which is link (-Z, +Y, +X) as (X, Y, Z): this rotation, applied in the link frame.
HOOK_FRAME = wp.quat(0.0, 0.70710678, 0.0, 0.70710678)
# Index phalanges as one capsule (a, b, radius) in each link's frame, for finger-ring clearance.
FINGER_LINKS = ("r_index_finger_middle", "r_index_finger_distal")
FINGER_CAPSULE = ((0.0, 0.0, 0.004), (0.0, 0.0, -0.026), 0.0085)

# Only the right arm moves; the waist and head hold their zero pose.
ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_CURL = 0.8  # fraction of the flexion range for the digits kept out of the ring's way
# Joint positions that differ from the URDF default.
POSE = {
    **{f"r_index_finger_{j}": 0.0 for j in ("mcp_flex", "mcp_abd", "pip", "dip")},
    **{
        f"r_{finger}_{j}": _CURL * upper
        for finger in ("middle_finger", "ring_finger", "pinky")
        for j, upper in (("mcp_flex", 1.57), ("pip", 2.094), ("dip", 1.57))
    },
    "r_thumb_cmc_flex": _CURL * 1.291,
    "r_thumb_mcp": _CURL * 1.57,
    "r_thumb_ip": _CURL * 1.57,
    # The asset's zero pose holds both forearms forward; the idle left arm would reach into the
    # tree, so it hangs at the robot's side instead.
    "left_elbow_joint": 1.4,
}
FINGER_JOINT_PREFIXES = ("l_", "r_")  # Wuji finger joints, driven by the soft hand gains
COLLIDE_LINK_PREFIXES = ("r_", "right_wrist_yaw_link")  # only the right hand collides
LEG_KE, LEG_KD = 5.0e4, 500.0
HAND_KE, HAND_KD = 1500.0, 8.0


def agile_one_urdf() -> Path:
    """The Agile One SN with Wuji hands (``robot_menagerie``'s ``hands_wuji`` URDF).

    The asset is internal and has no public download: point ``AGILE_ONE_URDF`` at
    ``agile_robotics/agile_one_sn/generated/hands_wuji/urdf/hands_wuji.urdf`` in a checkout with
    its Git LFS meshes pulled.
    """
    path = os.environ.get(AGILE_URDF_ENV)
    if not path:
        raise FileNotFoundError(f"set {AGILE_URDF_ENV} to the Agile One hands_wuji URDF")
    return Path(path)


def _short(label: str) -> str:
    return label.split("/")[-1]


def robot_builder(
    urdf: Path, pelvis_xform: wp.transform, arm_ke: float = 32000.0, arm_kd: float = 240.0
) -> newton.ModelBuilder:
    """Import the Agile One with its pelvis welded to the world and only the right hand colliding.

    ``collapse_fixed_joints`` merges massless URDF frames into physical links and the welded
    pelvis into the world, so the legs, waist and arms hang off the world body. Legs are held
    straight by stiff PD; the waist, head and arms by the arm gains; the fingers by soft gains.
    """
    builder = newton.ModelBuilder()
    # Shapes keep the gap of the builder that created them; the default is far wider than the
    # millimetre clearances between ring, finger and twig and floods the per-body contact buffer.
    builder.rigid_gap = RIGID_GAP
    builder.add_urdf(str(urdf), floating=False, xform=pelvis_xform, enable_self_collisions=False)
    builder.collapse_fixed_joints()
    if any(int(t) == int(newton.JointType.FREE) for t in builder.joint_type):
        raise RuntimeError("Agile One import kept a free base joint; the lower body must be fixed")

    for joint, label in enumerate(builder.joint_label):
        name = _short(label)
        q, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        if "hip" in name or "knee" in name or "ankle" in name:
            ke, kd = LEG_KE, LEG_KD
        elif name.startswith(FINGER_JOINT_PREFIXES):
            ke, kd = HAND_KE, HAND_KD
        else:
            ke, kd = arm_ke, arm_kd
        value = POSE.get(name, builder.joint_q[q])
        builder.joint_q[q] = value
        builder.joint_target_q[q] = value
        builder.joint_target_ke[dof] = ke
        builder.joint_target_kd[dof] = kd

    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for shape, body in enumerate(builder.shape_body):
        hand = body >= 0 and _short(builder.body_label[body]).startswith(COLLIDE_LINK_PREFIXES)
        if not hand:
            builder.shape_flags[shape] &= ~collide
        builder.shape_material_ke[shape] = RIGID_CONTACT_STIFFNESS
        builder.shape_material_kd[shape] = RIGID_CONTACT_DAMPING
        builder.shape_material_mu[shape] = 0.5
    return builder


def body_index(model_or_builder, link: str) -> int:
    return next(i for i, label in enumerate(model_or_builder.body_label) if _short(label) == link)


def hook_rotation(body_rotation) -> np.ndarray:
    """World hook-frame orientation (x, y, z, w) from the hook link's world orientation."""
    return np.array(wp.mul(wp.quat(*body_rotation), HOOK_FRAME))


def _arm_coords(model: newton.Model) -> np.ndarray:
    """Joint-coordinate indices of the IK's free joints (the right arm)."""
    starts = model.joint_q_start.numpy()
    return np.array([starts[j] for j, label in enumerate(model.joint_label) if _short(label) in ARM_JOINTS])


def _hook_error(model: newton.Model, state, hook: int, coords, target, quat) -> tuple[float, float]:
    """Hook-point position error [m] and hook-frame angle error [rad] for ``coords``."""
    newton.eval_fk(model, wp.array(coords, dtype=wp.float32, device=model.device), model.joint_qd, state)
    body = state.body_q.numpy()[hook]
    error = float(np.linalg.norm(np.array(wp.transform_point(wp.transform(*body), HOOK_OFFSET)) - np.asarray(target)))
    dot = min(1.0, abs(float(np.dot(hook_rotation(body[3:]), np.asarray(quat)))))
    return error, 2.0 * math.acos(dot)


def _ik_solver(model: newton.Model, problems: int, target, quat):
    from newton import ik

    hook = body_index(model, HOOK_LINK)
    position = ik.IKObjectivePosition(
        link_index=hook,
        link_offset=HOOK_OFFSET,
        target_positions=wp.array([wp.vec3(*map(float, target))] * problems, dtype=wp.vec3, device=model.device),
    )
    rotation = ik.IKObjectiveRotation(
        link_index=hook,
        link_offset_rotation=HOOK_FRAME,
        target_rotations=wp.array([wp.vec4(*map(float, quat))] * problems, dtype=wp.vec4, device=model.device),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=problems,
        objectives=[position, rotation, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    return solver, position, rotation


def _multistart(model: newton.Model, rest: np.ndarray, target, quat, seeds: int = 128) -> list[np.ndarray]:
    """Converged (1 mm, 0.5 degree) IK solutions from random right-arm seeds; other joints at ``rest``."""
    solver, _, _ = _ik_solver(model, seeds, target, quat)
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    arm = _arm_coords(model)
    coords = np.tile(rest, (seeds, 1))
    coords[:, arm] = np.random.default_rng(0).uniform(lower[arm], upper[arm], size=(seeds, len(arm)))
    q = wp.array(coords.astype(np.float32), dtype=wp.float32, device=model.device)
    fixed = np.ones(rest.shape, dtype=bool)
    fixed[arm] = False
    for _ in range(60):
        solver.step(q, q, iterations=10)
        values = q.numpy()
        values[:, fixed] = rest[fixed]
        q.assign(values)
    state = model.state()
    hook = body_index(model, HOOK_LINK)
    converged = []
    for coords_k in q.numpy():
        error, angle = _hook_error(model, state, hook, coords_k, target, quat)
        if error < 0.001 and angle < math.radians(0.5):
            converged.append(coords_k.copy())
    return converged


def solve_home(builder: newton.ModelBuilder, hook_target: np.ndarray, hook_quat: np.ndarray, device) -> np.ndarray:
    """Rest-pose IK from many random right-arm seeds.

    The arm reaches most hook poses on several branches (elbow and wrist configurations), and
    only some continue smoothly along the choreography. Returns the full joint-coordinate vector
    of the converged seed whose arm joints stay farthest from their limits.
    """
    model = builder.finalize(device=device)
    solutions = _multistart(model, model.joint_q.numpy().copy(), hook_target, hook_quat)
    if not solutions:
        raise RuntimeError(f"Unreachable home {np.round(hook_target, 3).tolist()}")
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    arm = _arm_coords(model)
    return max(solutions, key=lambda c: float(np.min(np.minimum(c[arm] - lower[arm], upper[arm] - c[arm]))))


# Largest arm-joint change between consecutive waypoints [rad]: more means IK switched branch and
# the interpolated motion would swing the hand off its path.
MAX_WAYPOINT_JUMP = 0.5


def solve_waypoints(
    builder: newton.ModelBuilder, hook_targets: np.ndarray, hook_rotations: np.ndarray, device
) -> tuple[newton.Model, int, np.ndarray]:
    """Solve hook-point IK on a separate reference model; only the right-arm joints move.

    ``hook_rotations`` are world orientations (x, y, z, w) of the hook frame. Each waypoint
    starts from the previous solution (``builder.joint_q`` for the first). Where that stalls, it
    re-solves from many seeds and takes the solution nearest the previous one. Returns the
    reference model, the hook body index and one full joint-coordinate vector per waypoint.
    Raises if a waypoint misses by more than 3 mm or needs an arm-joint jump above
    ``MAX_WAYPOINT_JUMP``.
    """
    model = builder.finalize(device=device)
    hook = body_index(model, HOOK_LINK)
    solver, position, rotation = _ik_solver(model, 1, hook_targets[0], hook_rotations[0])
    rest = model.joint_q.numpy().copy()
    arm = _arm_coords(model)
    fixed = np.ones(rest.shape, dtype=bool)
    fixed[arm] = False
    q = wp.array(rest.reshape(1, -1), dtype=wp.float32, device=device)
    state = model.state()
    previous = rest
    solutions = []
    for target, quat in zip(hook_targets, hook_rotations, strict=True):
        position.set_target_position(0, wp.vec3(*map(float, target)))
        rotation.set_target_rotation(0, wp.vec4(*map(float, quat)))
        # 20 rounds converge most waypoints; awkward wrist poses get up to 80 to close the last mm.
        for round_ in range(80):
            solver.step(q, q, iterations=10)
            values = q.numpy().ravel()
            values[fixed] = rest[fixed]
            q.assign(values.reshape(1, -1))
            if round_ >= 19 and round_ % 10 == 9:
                error, _ = _hook_error(model, state, hook, values, target, quat)
                if error < 0.001:
                    break
        coords = q.numpy().ravel().copy()
        if error > 0.001:
            candidates = _multistart(model, rest, target, quat)
            if candidates:
                coords = min(candidates, key=lambda c: float(np.max(np.abs(c[arm] - previous[arm]))))
                error, _ = _hook_error(model, state, hook, coords, target, quat)
                q.assign(coords.reshape(1, -1))
        if error > 0.003:
            raise RuntimeError(f"Unreachable hook waypoint {np.round(target, 3).tolist()}: IK error {error:.4f} m")
        jump = float(np.max(np.abs(coords[arm] - previous[arm])))
        if solutions and jump > MAX_WAYPOINT_JUMP:
            raise RuntimeError(f"IK branch switch at {np.round(target, 3).tolist()}: arm joints jump {jump:.2f} rad")
        solutions.append(coords)
        previous = coords
    return model, hook, np.array(solutions, dtype=np.float32)
