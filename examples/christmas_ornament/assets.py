# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The glass ornament with a rigid SDF ring, and the fixed-base Unitree G1."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
