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
    """The hang point is at hand height, unobstructed, and the robot's standing sector is free."""
    builder = _builder()
    tree = build_tree(builder, ORIGIN, FACING)
    hang = _hang_point_world(builder, tree)
    assert 1.00 <= hang[2] <= 1.15

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
    """The trunk stands within 5 cm; a 35 g load at the hang point dips the branch by >= 10 mm."""

    def settle(with_load: bool) -> tuple[np.ndarray, np.ndarray]:
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
        hang = np.array(wp.transform_point(wp.transform(*q[tree.target.hang_body]), wp.vec3(*tree.target.hang_local)))
        return q[tree.trunk_top_body, :3] - top0, hang

    top_disp, hang_free = settle(False)
    assert np.linalg.norm(top_disp) < 0.05
    _, hang_loaded = settle(True)
    assert hang_free[2] - hang_loaded[2] >= 0.010
