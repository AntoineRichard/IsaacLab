# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the ornament: ring size, mass, and an SDF ring that holds on a 2 mm twig."""

import math

import newton
import numpy as np
import pytest
import trimesh
import warp as wp

from examples.christmas_ornament.assets import add_ornament, ring_mesh
from examples.christmas_ornament.tree import configure_builder

pytestmark = [pytest.mark.kitless, pytest.mark.unit]


def test_ring_geometry():
    """A 60 mm inner diameter passes a twig under the 17 mm Wuji fingertip with margin; the tube is 2.5 mm."""
    vertices, triangles = ring_mesh()
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent
    radial = np.hypot(vertices[:, 0], vertices[:, 2])
    assert np.min(radial) == pytest.approx(0.030, abs=0.0005)
    assert 0.5 * (np.max(radial) - np.min(radial)) == pytest.approx(0.0025, abs=0.0003)
    assert np.max(np.abs(vertices[:, 1])) == pytest.approx(0.0025, abs=0.0003)


def test_ornament_mass_and_layout():
    """35 g total, centre of mass well below the ring so it hangs upright."""
    builder = newton.ModelBuilder()
    ornament = add_ornament(builder, wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), (0.85, 0.1, 0.1))
    assert builder.body_mass[ornament.body] == pytest.approx(0.035, rel=0.1)
    com = np.asarray(builder.body_com[ornament.body])
    assert com[2] < np.asarray(ornament.ring_center_local)[2] - 0.03
    assert ornament.ring_inner_radius == pytest.approx(0.030)
    assert np.allclose(ornament.ring_axis_local, (0.0, 1.0, 0.0))


@pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA for the SDF contact rollout")
def test_ring_does_not_tunnel_thin_twig():
    """Thrown down at 1 m/s onto a 2 mm twig threaded through it, the ring catches and hangs."""
    twig_radius, twig_z = 0.002, 1.0
    builder = newton.ModelBuilder()
    configure_builder(builder)
    # Static twig along world Y (the ring's axis), threaded through the ring 10 mm above its centre.
    builder.add_shape_capsule(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, twig_z), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)),
        radius=twig_radius,
        half_height=0.06,
    )
    ornament = add_ornament(
        builder, wp.transform(wp.vec3(0.0, 0.0, twig_z - 0.010), wp.quat_identity()), (0.85, 0.1, 0.1)
    )
    builder.body_qd[ornament.body] = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    builder.color(balance_colors=False)
    model = builder.finalize(device="cuda:0")
    solver = newton.solvers.SolverVBD(model, iterations=20)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    s0, s1, control = model.state(), model.state(), model.control()
    dt = 1.0 / 60.0 / 20
    for _ in range(90 * 20):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    pose = wp.transform(*s0.body_q.numpy()[ornament.body])
    center = np.array(wp.transform_point(pose, wp.vec3(*ornament.ring_center_local)))
    assert np.all(np.isfinite(center))
    # Still threaded: the twig axis (x = 0, z = twig_z) passes through the ring's hole.
    assert math.hypot(center[0], center[2] - twig_z) < ornament.ring_inner_radius
    # Hanging: the ring's inner top edge rests on the twig.
    assert center[2] == pytest.approx(twig_z + twig_radius - ornament.ring_inner_radius, abs=0.003)


def _visual_meshes(builder, body):
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    visible = int(newton.ShapeFlags.VISIBLE)
    return [
        builder.shape_source[s]
        for s in range(builder.shape_count)
        if builder.shape_body[s] == body
        and builder.shape_type[s] == newton.GeoType.MESH
        and builder.shape_flags[s] & visible
        and not builder.shape_flags[s] & collide
    ]


def test_ornament_glass_is_shiny():
    """Renderers take roughness/metallic from meshes only, so the glass is drawn as a glossy mesh;
    the collision sphere stays but is hidden."""
    from examples.christmas_ornament.assets import BAUBLE_METALLIC, BAUBLE_ROUGHNESS

    builder = newton.ModelBuilder()
    ornament = add_ornament(builder, wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), (0.85, 0.1, 0.1))
    shiny = [m for m in _visual_meshes(builder, ornament.body) if m.roughness is not None and m.roughness <= 0.2]
    assert shiny and all(m.metallic >= 0.5 for m in shiny)
    assert BAUBLE_ROUGHNESS <= 0.2 and BAUBLE_METALLIC >= 0.5
    spheres = [
        s
        for s in range(builder.shape_count)
        if builder.shape_body[s] == ornament.body and builder.shape_type[s] == newton.GeoType.SPHERE
    ]
    assert spheres and all(not builder.shape_flags[s] & int(newton.ShapeFlags.VISIBLE) for s in spheres)
    assert all(builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES) for s in spheres)
