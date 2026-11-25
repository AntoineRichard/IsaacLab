# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the dynamic pompon guides and their skinned ribbon meshes."""

from __future__ import annotations

import math

import newton
import numpy as np
import warp as wp

GUIDE_COUNT = 32
RIBBONS_PER_GUIDE = 16
ROD_SEGMENTS = 7
RIBBON_SEGMENTS = 10
HUB_HEIGHT = 0.55
PALETTE = np.array(
    [(25, 100, 59), (31, 128, 56), (45, 150, 88), (81, 177, 59), (130, 204, 75), (192, 235, 135)],
    dtype=np.uint8,
)


def make_ribbons(
    centerline: np.ndarray,
    width_direction: np.ndarray,
    rng: np.random.Generator,
    extra_ribbon_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build flat strips with fixed variation in the rod's rest frame."""
    direction = centerline[-1] - centerline[0]
    tangent = direction / np.linalg.norm(direction)
    cross_direction = np.cross(tangent, width_direction)
    progress = np.linspace(0.0, 1.0, RIBBON_SEGMENTS + 1)
    guide_progress = np.linspace(0.0, 1.0, len(centerline))
    vertices, triangles, uvs = [], [], []
    for ribbon in range(RIBBONS_PER_GUIDE):
        # Use a fixed seed so the visual ribbons keep their guide assignments.
        ribbon_rng = rng if ribbon < 12 else extra_ribbon_rng
        stretch = ribbon_rng.uniform(0.76, 1.09)
        sampled = np.column_stack(
            [np.interp(progress * stretch, guide_progress, coordinate) for coordinate in centerline.T]
        )
        sampled += np.maximum(progress * stretch - 1.0, 0.0)[:, None] * direction

        root_offset = ribbon_rng.uniform(-0.004, 0.004, size=2)
        tip_offset = ribbon_rng.uniform(-0.026, 0.026, size=2)
        offsets = root_offset + progress[:, None] * tip_offset
        sampled += offsets[:, :1] * width_direction + offsets[:, 1:] * cross_direction
        angle = ribbon_rng.uniform(-math.pi, math.pi) + ribbon_rng.uniform(-2.1, 2.1) * progress
        across = np.cos(angle)[:, None] * width_direction + np.sin(angle)[:, None] * cross_direction
        width = ribbon_rng.uniform(0.006, 0.011) * (0.8 + 0.2 * np.sin(math.pi * progress))
        halves = across * (0.5 * width[:, None])
        stripe = ribbon_rng.integers(len(PALETTE))
        base = len(vertices)
        for center, half in zip(sampled, halves, strict=True):
            vertices.extend((center - half, center + half))
            uvs.extend((((stripe + 0.5) / len(PALETTE), 0.5),) * 2)
        for span in range(RIBBON_SEGMENTS):
            left = base + 2 * span
            triangles.extend((left, left + 1, left + 2, left + 1, left + 3, left + 2))
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32), np.asarray(uvs)


def make_pompon_builder(*, visual_meshes: bool = True) -> tuple[newton.ModelBuilder, list[int]]:
    """Create the hub, physical guides, and one skinned mesh per guide."""
    rng = np.random.default_rng(17)
    extra_ribbon_rng = np.random.default_rng(29)
    builder = newton.ModelBuilder()
    center = np.array([0.0, 0.0, HUB_HEIGHT])
    hub = builder.add_body(xform=wp.transform(wp.vec3(*center), wp.quat_identity()), label="hub")
    handle_cfg = newton.ModelBuilder.ShapeConfig(density=150.0, has_shape_collision=False, has_particle_collision=False)
    builder.add_shape_sphere(hub, radius=0.027, cfg=handle_cfg, color=(0.10, 0.17, 0.13))
    builder.add_shape_capsule(
        hub,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.092), wp.quat_identity()),
        radius=0.014,
        half_height=0.068,
        cfg=handle_cfg,
        color=(0.38, 0.35, 0.34),
    )
    rod_cfg = newton.ModelBuilder.ShapeConfig(is_visible=False, has_shape_collision=False, has_particle_collision=False)
    texture = np.repeat(np.repeat(PALETTE[None, :, :], 16, axis=0), 16, axis=1)
    roots = [hub]
    for i in range(GUIDE_COUNT):
        z = 0.98 - 1.60 * (i + 0.5) / GUIDE_COUNT
        azimuth = i * math.pi * (3.0 - math.sqrt(5.0))
        radial = math.sqrt(1.0 - z * z)
        direction = np.array([radial * math.cos(azimuth), radial * math.sin(azimuth), z])
        width_direction = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
        progress = np.linspace(0.0, 1.0, ROD_SEGMENTS + 1)[:, None]
        length = rng.uniform(0.14, 0.21)
        points = center + (0.023 + length * progress) * direction
        points += rng.uniform(-0.03, 0.03) * progress**2 * width_direction

        rod = newton.Rod([wp.vec3(*point) for point in points], radius=0.002)
        bodies, _ = builder.add_rod(
            rod=rod,
            cfg=rod_cfg,
            total_mass=0.001,
            stretch_stiffness=110.0,
            bend_stiffness=0.00145,
            bend_damping=0.00010,
            twist_stiffness=0.00065,
            twist_damping=0.000068,
            label=f"guide_{i:02}",
            color=(0.24, 0.72, 0.45),
        )
        roots.append(bodies[0])

        vertices, indices, uvs = make_ribbons(points, width_direction, rng, extra_ribbon_rng)
        if visual_meshes:
            builder.add_deformable_visual_mesh(
                vertices, indices, kind="body", bodies=bodies, uvs=uvs, texture=texture, label=f"ribbons_{i:02}"
            )

    return builder, roots
