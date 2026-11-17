# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the fixed-base G1: no free base, an extended hook finger, reachable waypoints."""

import newton
import numpy as np
import pytest
import trimesh
import warp as wp

from examples.christmas_ornament.assets import (
    ARM_JOINTS,
    HAND_POSE,
    HOOK_LINK,
    HOOK_OFFSET,
    g1_urdf,
    robot_builder,
    solve_waypoints,
)

pytestmark = [pytest.mark.kitless, pytest.mark.unit]

PELVIS = wp.transform(wp.vec3(0.0, 0.0, 0.793), wp.quat_identity())


@pytest.fixture(scope="module")
def builder() -> newton.ModelBuilder:
    return robot_builder(g1_urdf(), PELVIS)


def _joint_index(builder: newton.ModelBuilder, name: str) -> int:
    return next(i for i, label in enumerate(builder.joint_label) if label.endswith("/" + name))


def test_fixed_base_and_hand_pose(builder):
    """The pelvis is welded to the world; the right index is straight and held there."""
    assert newton.JointType.FREE not in {int(t) for t in builder.joint_type}
    assert all(
        builder.joint_parent[i] != -1
        or builder.joint_label[i].split("/")[-1].startswith(("left_hip", "right_hip", "waist"))
        for i in range(builder.joint_count)
    )
    for name in ("right_hand_index_0_joint", "right_hand_index_1_joint"):
        joint = _joint_index(builder, name)
        assert builder.joint_target_q[builder.joint_q_start[joint]] == 0.0
        assert builder.joint_q[builder.joint_q_start[joint]] == 0.0
    assert HAND_POSE["right_hand_middle_1_joint"] > 1.0  # curled out of the ring's way
    assert len(ARM_JOINTS) == 10 and ARM_JOINTS[0] == "waist_yaw_joint" and ARM_JOINTS[-1] == "right_wrist_yaw_joint"


def test_hook_point_on_finger(builder):
    """The hook point sits on the proximal phalanx's top surface, where the ring will rest."""
    body = next(i for i, label in enumerate(builder.body_label) if label.endswith("/" + HOOK_LINK))
    colliders = [
        s
        for s in range(builder.shape_count)
        if builder.shape_body[s] == body and builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)
    ]
    assert colliders, "the hook link must collide with the ring"
    source = builder.shape_source[colliders[0]]
    mesh = trimesh.Trimesh(np.asarray(source.vertices), np.asarray(source.indices).reshape(-1, 3), process=False)
    _, distance, _ = trimesh.proximity.closest_point(mesh, np.array([list(HOOK_OFFSET)]))
    assert distance[0] < 0.004


def test_waypoints_reachable(builder):
    """0.40 m in front of the right shoulder at 1.05 m, finger level and pointing at the tree (+x)."""
    targets = np.array([[0.40, -0.10, 1.05], [0.30, -0.10, 1.05], [0.30, -0.10, 1.015]])
    rotations = np.tile([0.0, 0.0, 0.0, 1.0], (len(targets), 1))
    model, hook_body, q = solve_waypoints(builder, targets, rotations, device="cpu")
    assert q.shape == (len(targets), model.joint_coord_count)
    state = model.state()
    arm = {_joint_index(builder, name) for name in ARM_JOINTS}
    for target, coords in zip(targets, q, strict=True):
        newton.eval_fk(model, wp.array(coords, dtype=wp.float32, device="cpu"), model.joint_qd, state)
        hook = wp.transform_point(wp.transform(*state.body_q.numpy()[hook_body]), HOOK_OFFSET)
        assert np.linalg.norm(np.array(hook) - target) < 0.003
        # Only waist and right-arm coordinates move.
        for joint in range(builder.joint_count):
            if joint not in arm:
                start = builder.joint_q_start[joint]
                assert coords[start] == pytest.approx(builder.joint_q[start], abs=1e-6)
