# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the fixed-base Agile One: no free base, an extended hook finger, arm-only IK.

The Agile One asset is internal, so everything but the hook-frame check needs ``AGILE_ONE_URDF``.
"""

import os

import newton
import numpy as np
import pytest
import trimesh
import warp as wp

from examples.christmas_ornament.assets import (
    AGILE_URDF_ENV,
    ARM_JOINTS,
    HOOK_FRAME,
    HOOK_LINK,
    HOOK_OFFSET,
    PELVIS_HEIGHT,
    POSE,
    agile_one_urdf,
    robot_builder,
    solve_waypoints,
)
from examples.christmas_ornament.motion import finger_rotation

pytestmark = [pytest.mark.kitless, pytest.mark.unit]

agile_asset = pytest.mark.skipif(not os.environ.get(AGILE_URDF_ENV), reason=f"{AGILE_URDF_ENV} not set")
PELVIS = wp.transform(wp.vec3(0.0, 0.0, PELVIS_HEIGHT), wp.quat_identity())


@pytest.fixture(scope="module")
def builder() -> newton.ModelBuilder:
    return robot_builder(agile_one_urdf(), PELVIS)


def _joint_index(builder: newton.ModelBuilder, name: str) -> int:
    return next(i for i, label in enumerate(builder.joint_label) if label.endswith("/" + name))


def test_hook_frame_maps_finger_and_thumb_side():
    """The hook frame's +X runs down the Wuji phalanx (link -Z) and its +Z out of the thumb side (link +X)."""
    assert np.allclose(wp.quat_rotate(HOOK_FRAME, wp.vec3(1.0, 0.0, 0.0)), (0.0, 0.0, -1.0), atol=1e-6)
    assert np.allclose(wp.quat_rotate(HOOK_FRAME, wp.vec3(0.0, 0.0, 1.0)), (1.0, 0.0, 0.0), atol=1e-6)


def test_only_the_right_arm_is_free():
    assert len(ARM_JOINTS) == 7 and all(name.startswith("right_") for name in ARM_JOINTS)


@agile_asset
def test_fixed_base_and_pose(builder):
    """Pelvis welded, every posed joint exists and is held there, and the index stays straight."""
    assert newton.JointType.FREE not in {int(t) for t in builder.joint_type}
    for name, value in POSE.items():
        start = builder.joint_q_start[_joint_index(builder, name)]
        assert builder.joint_q[start] == pytest.approx(value)
        assert builder.joint_target_q[start] == pytest.approx(value)
    for name in ("r_index_finger_mcp_flex", "r_index_finger_pip", "r_index_finger_dip"):
        assert POSE[name] == 0.0


@agile_asset
def test_robot_shapes_use_tight_contact_gap(builder):
    """A wide default gap floods the ornament's per-body contact buffer and drops real contacts."""
    assert builder.rigid_gap == pytest.approx(0.002)
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    gaps = [builder.shape_gap[s] for s in range(builder.shape_count) if builder.shape_flags[s] & collide]
    assert gaps and max(gaps) <= 0.002


def _shape_matrix(xform) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = np.array(wp.quat_to_matrix(wp.quat(*xform[3:]))).reshape(3, 3)
    m[:3, 3] = xform[:3]
    return m


@agile_asset
def test_hook_point_on_nail_side(builder):
    """The hook point sits on the distal phalanx's thumb side (+X), where the ring rests."""
    body = next(i for i, label in enumerate(builder.body_label) if label.endswith("/" + HOOK_LINK))
    colliders = [
        s
        for s in range(builder.shape_count)
        if builder.shape_body[s] == body and builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)
    ]
    assert colliders, "the hook link must collide with the ring"
    hook = np.array([list(HOOK_OFFSET)])
    distances = []
    for shape in colliders:
        source = builder.shape_source[shape]
        mesh = trimesh.Trimesh(np.asarray(source.vertices), np.asarray(source.indices).reshape(-1, 3), process=False)
        mesh.apply_transform(_shape_matrix(builder.shape_transform[shape]))
        distances.append(trimesh.proximity.closest_point(mesh, hook)[1][0])
    assert min(distances) < 0.002
    assert HOOK_OFFSET[0] > 0.0


@agile_asset
def test_waypoints_move_only_the_right_arm(builder):
    """Hook targets ahead of the right hip, finger level along +x: solved, and nothing else moves."""
    targets = np.array([[0.44, -0.12, 0.95], [0.40, -0.12, 0.92]])
    rotations = np.tile(finger_rotation(np.array([-1.0, 0.0, 0.0])), (len(targets), 1))
    model, hook_body, q = solve_waypoints(builder, targets, rotations, device="cpu")
    state = model.state()
    arm = {_joint_index(builder, name) for name in ARM_JOINTS}
    for target, coords in zip(targets, q, strict=True):
        newton.eval_fk(model, wp.array(coords, dtype=wp.float32, device="cpu"), model.joint_qd, state)
        hook = wp.transform_point(wp.transform(*state.body_q.numpy()[hook_body]), HOOK_OFFSET)
        assert np.linalg.norm(np.array(hook) - target) < 0.003
        for joint in range(builder.joint_count):
            if joint not in arm:
                start = builder.joint_q_start[joint]
                assert coords[start] == pytest.approx(builder.joint_q[start], abs=1e-6)
