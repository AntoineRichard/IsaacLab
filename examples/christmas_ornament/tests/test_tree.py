# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the rod-graph fir: two moduli, a reachable target branch, a standing trunk."""

import math

import newton
import numpy as np
import pytest
import warp as wp

from examples.christmas_ornament.tree import BRANCH_YOUNG_MODULUS, TRUNK_YOUNG_MODULUS, build_tree, configure_builder

pytestmark = [pytest.mark.kitless, pytest.mark.unit]

ORIGIN = (0.8, 0.0, 0.0)
FACING = math.pi  # target branch points from the trunk toward the robot at the world origin


def _builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    configure_builder(builder)
    return builder


def _bend_ke(builder: newton.ModelBuilder, joint: int) -> float:
    linear, _ = builder.joint_dof_dim[joint]
    return builder.joint_target_ke[builder.joint_qd_start[joint] + linear]


def _segment_axis(builder: newton.ModelBuilder, body: int, length: float) -> tuple[np.ndarray, np.ndarray]:
    xform = builder.body_q[body]
    a = np.array(wp.transform_point(xform, wp.vec3(0.0, 0.0, -0.5 * length)))
    b = np.array(wp.transform_point(xform, wp.vec3(0.0, 0.0, 0.5 * length)))
    return a, b


def _segment_distance(p0, p1, q0, q1, samples: int = 24) -> float:
    """Minimum distance between two segments, by dense sampling (adequate at mm scale)."""
    t = np.linspace(0.0, 1.0, samples)[:, None]
    a = p0 + t * (p1 - p0)
    b = q0 + t * (q1 - q0)
    return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))


def _hang_point_world(builder, tree) -> np.ndarray:
    xform = builder.body_q[tree.target.hang_body]
    return np.array(wp.transform_point(xform, wp.vec3(*tree.target.hang_local)))


def test_two_moduli():
    """The trunk is stiff enough to stand; branches are young wood so an ornament dips them."""
    builder = _builder()
    tree = build_tree(builder, ORIGIN, FACING)
    trunk_joint = tree.trunk_joints[1]
    child_body = builder.joint_child[trunk_joint]
    parent_body = builder.joint_parent[trunk_joint]
    r = min(tree.wood_radius[child_body], tree.wood_radius[parent_body])
    span = 0.5 * (tree.segment_length[child_body] + tree.segment_length[parent_body])
    assert _bend_ke(builder, trunk_joint) == pytest.approx(TRUNK_YOUNG_MODULUS * math.pi * r**4 / 4.0 / span, rel=1e-6)

    branch_joint = tree.target.joints[1]
    child_body = builder.joint_child[branch_joint]
    parent_body = builder.joint_parent[branch_joint]
    r = min(tree.wood_radius[child_body], tree.wood_radius[parent_body])
    span = 0.5 * (tree.segment_length[child_body] + tree.segment_length[parent_body])
    assert _bend_ke(builder, branch_joint) == pytest.approx(
        BRANCH_YOUNG_MODULUS * math.pi * r**4 / 4.0 / span, rel=1e-6
    )
    assert BRANCH_YOUNG_MODULUS < TRUNK_YOUNG_MODULUS
    assert TRUNK_YOUNG_MODULUS == 5.0e9


def test_target_branch_reachable_and_clear():
    """The hang point is at hand height, unobstructed, and the robot's standing sector is free.

    Hand height is below about 1 m: the Agile One's right arm alone holds the finger level only
    that low in front of it."""
    builder = _builder()
    tree = build_tree(builder, ORIGIN, FACING)
    hang = _hang_point_world(builder, tree)
    assert 0.88 <= hang[2] <= 0.98
    assert tree.target.hang_body == tree.target.segment_bodies[-1]  # outer segment, halfway out
    assert tree.target.hang_local[2] == pytest.approx(0.0)

    tangent = np.array(wp.quat_rotate(builder.body_q[tree.target.hang_body].q, wp.vec3(*tree.target.tangent_local)))
    corridor = (hang, hang + 0.20 * tangent)
    target_bodies = set(tree.target.segment_bodies)
    for body in tree.bodies:
        if body in target_bodies or body in tree.target.hang_neighbours:
            continue
        a, b = _segment_axis(builder, body, tree.segment_length[body])
        assert _segment_distance(*corridor, a, b) > 0.06, f"body {body} obstructs the approach corridor"

    # Nothing pre-hung near the hang point.
    for hanger in tree.hangers:
        a, b = _segment_axis(builder, hanger, tree.segment_length[hanger])
        assert np.linalg.norm(a - hang) > 0.15

    # The robot stands in the facing sector, 0.6-1.0 m from the trunk: keep it free of long branches.
    trunk_xy = np.array(ORIGIN[:2])
    for body in tree.bodies:
        if body in target_bodies or body in tree.target.hang_neighbours:
            continue
        for p in _segment_axis(builder, body, tree.segment_length[body]):
            d = p[:2] - trunk_xy
            radius = np.linalg.norm(d)
            if radius < 0.6 or p[2] > 1.0:
                continue
            azimuth = math.atan2(d[1], d[0])
            off = abs((azimuth - FACING + math.pi) % (2 * math.pi) - math.pi)
            assert off > math.radians(20.0), f"body {body} enters the robot sector"


@pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA for a 2.4k-body VBD rollout")
def test_tree_stands_and_dips_under_load():
    """The trunk stands; a 35 g load dips the branch >= 3 mm without tipping its tip downhill."""

    def settle(with_load: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        builder = _builder()
        tree = build_tree(builder, ORIGIN, FACING)
        if with_load:
            # Test fixture only: a 35 g point load welded at the hang point. The demo never welds.
            load = builder.add_body(
                xform=wp.transform(wp.vec3(*_hang_point_world(builder, tree)), wp.quat_identity()),
                mass=0.035,
                inertia=wp.mat33(np.eye(3) * 1.0e-6),
            )
            hang_xform = wp.transform(wp.vec3(*tree.target.hang_local), wp.quat_identity())
            builder.add_joint_fixed(tree.target.hang_body, load, parent_xform=hang_xform)
        builder.add_ground_plane()
        builder.color(balance_colors=False)
        model = builder.finalize(device="cuda:0")
        solver = newton.solvers.SolverVBD(model, iterations=20)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        s0, s1, control = model.state(), model.state(), model.control()
        top0 = s0.body_q.numpy()[tree.trunk_top_body, :3].copy()
        dt = 1.0 / 60.0 / 20
        for _ in range(120 * 20):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        q = s0.body_q.numpy()
        assert np.all(np.isfinite(q))
        pose = wp.transform(*q[tree.target.hang_body])
        hang = np.array(wp.transform_point(pose, wp.vec3(*tree.target.hang_local)))
        tangent = np.array(wp.quat_rotate(pose.q, wp.vec3(*tree.target.tangent_local)))
        return q[tree.trunk_top_body, :3] - top0, hang, tangent

    top_disp, hang_free, _ = settle(False)
    assert np.linalg.norm(top_disp) < 0.05
    _, hang_loaded, tangent_loaded = settle(True)
    # A visible dip, but the loaded twig must not run downhill toward its tip: a ring on a
    # downhill twig creeps off the end.
    assert hang_free[2] - hang_loaded[2] >= 0.003
    assert tangent_loaded[2] >= 0.0


def test_wind_time_advances_so_gusts_vary():
    """The gust clock must tick each substep; frozen at t = 0 the wind is a constant push."""
    from examples.christmas_ornament.tree import Wind

    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.1)
    builder.add_shape_capsule(body, radius=0.01, half_height=0.05)
    model = builder.finalize(device="cpu")
    state = model.state()
    wind = Wind([body], [0.01], strength=5.0, device="cpu")
    forces = []
    for _ in range(3):
        state.clear_forces()
        wind.apply(state, dt=0.5)
        forces.append(state.body_f.numpy()[body, :3].copy())
    assert float(wind.time.numpy()[0]) == pytest.approx(1.5)
    assert not np.allclose(forces[0], forces[2])


def test_needle_meshes_bake_each_hosts_needles_into_one_mesh():
    """For USD/RTX: one cone mesh per host segment (~2.4k prims) instead of ~258k instances."""
    from examples.christmas_ornament.tree import NEEDLE_HALF_LENGTH, make_needles, needle_meshes

    builder = _builder()
    tree = build_tree(builder, ORIGIN, FACING)
    batches = make_needles(tree, np.random.default_rng(3))
    meshes = needle_meshes(tree, np.random.default_rng(3), cone_segments=6)
    hosts = {int(b) for batch in batches for b in batch.body}
    assert set(meshes) == hosts
    total = sum(len(batch.body) for batch in batches)
    assert sum(len(m.triangles) for m in meshes.values()) == total * 2 * 6  # side + base fan
    for body, mesh in list(meshes.items())[:50]:
        reach = tree.segment_vis_radius[body] + 2.0 * max(NEEDLE_HALF_LENGTH) + 1e-6
        radial = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
        assert np.max(radial) <= reach
        assert np.max(np.abs(mesh.vertices[:, 2])) <= 0.5 * tree.segment_length[body] + reach
        assert mesh.color.shape == (3,)


def test_tree_baubles_and_star_are_shiny():
    builder = _builder()
    tree = build_tree(builder, ORIGIN, FACING)
    visible, collide = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
    for body in [*tree.hangers, tree.trunk_top_body]:
        meshes = [
            builder.shape_source[s]
            for s in range(builder.shape_count)
            if builder.shape_body[s] == body
            and builder.shape_type[s] == newton.GeoType.MESH
            and builder.shape_flags[s] & visible
        ]
        assert meshes and meshes[0].roughness <= 0.2 and meshes[0].metallic >= 0.5
        spheres = [
            s
            for s in range(builder.shape_count)
            if builder.shape_body[s] == body and builder.shape_type[s] == newton.GeoType.SPHERE
        ]
        assert all(not builder.shape_flags[s] & visible and builder.shape_flags[s] & collide for s in spheres)
