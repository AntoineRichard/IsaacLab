# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for contact choreography and containment measurements."""

import numpy as np
import pytest
import trimesh

from examples.christmas_atelier.assets import stocking_mesh
from examples.christmas_atelier.motion import candy_is_contained, choreography, sample_motion

pytestmark = [pytest.mark.kitless, pytest.mark.unit]


def _open_box(half_width):
    """Use known clearances independent of the stocking's evolving art geometry."""
    mesh = trimesh.creation.box(extents=(2 * half_width, 0.12, 0.35))
    vertices = np.asarray(mesh.vertices).copy()
    vertices[:, 2] -= 0.175
    faces = np.asarray(mesh.faces)[mesh.face_normals[:, 2] < 0.5]
    cuff = np.flatnonzero(np.isclose(vertices[:, 2], 0))
    cuff = cuff[np.argsort(np.arctan2(vertices[cuff, 1], vertices[cuff, 0]))]
    return vertices, faces, cuff


def test_grasp_precedes_lift_and_release_waits_for_destination():
    """A phase-order regression must not open the hand during transport."""
    times, positions, fingers, phases = choreography([(0.45, -0.22, 0.76)], (0.45, 0.20, 1.10))
    grasp = phases.index("grasp_0")
    lift = phases.index("lift_0")
    release = phases.index("release_0")
    assert grasp < lift < release
    np.testing.assert_allclose(positions[grasp], positions[grasp - 1])
    assert fingers[grasp] < 0.014
    assert fingers[lift] == fingers[grasp]
    assert fingers[release] == 0.04
    np.testing.assert_allclose(positions[release], positions[release - 1])
    assert positions[lift, 2] > positions[grasp, 2] + 0.1
    assert np.all(np.diff(times) > 0)
    targets = np.column_stack([positions, fingers])
    np.testing.assert_allclose(sample_motion(times[-1] + 100, times, targets), targets[-1])
    np.testing.assert_allclose(sample_motion(-1, times, targets), targets[0])


def test_containment_uses_closed_deformed_shell():
    """Candy below the opening but beside the sock must not count as filled."""
    vertices, faces, _, cuff = stocking_mesh()
    assert candy_is_contained(np.array([0.0, 0.0, -0.12]), vertices, faces, cuff)
    assert not candy_is_contained(np.array([0.3, 0.0, -0.12]), vertices, faces, cuff)
    assert not candy_is_contained(np.array([0.0, 0.0, 0.03]), vertices, faces, cuff)
    # Stretching the live cloth changes which points it contains.
    deformed = vertices.copy()
    deformed[:, 0] += 0.3
    assert candy_is_contained(np.array([0.3, 0.0, -0.12]), deformed, faces, cuff)
    assert not candy_is_contained(np.array([0.0, 0.0, -0.12]), deformed, faces, cuff)


def test_containment_respects_cylinder_orientation_in_narrow_stocking():
    """A flat cylinder end clears the wall when aligned, while a tilted rim intersects it."""
    vertices, faces, cuff = _open_box(0.017)
    center = np.array([0.0, 0.0, -0.12])
    assert candy_is_contained(center, vertices, faces, cuff, candy_collider="cylinder")
    tilt = np.array([0.0, np.sin(np.pi / 8), 0.0, np.cos(np.pi / 8)])
    assert not candy_is_contained(center, vertices, faces, cuff, tilt, candy_collider="cylinder")


def test_invalid_cloth_measurements_remain_json_serializable():
    """Diverged physics must produce a failed measurement record, including when JSON forbids NaN."""
    import json

    from examples.christmas_atelier.motion import measure_stocking

    vertices, faces, _, cuff = stocking_mesh()
    broken = vertices.copy()
    broken[0] = np.nan
    report = measure_stocking(broken, vertices, faces, cuff, np.array([[0.0, 0.0, -0.12, 0, 0, 0, 1]]))
    assert not report["cloth_finite"]
    assert report["cuff_drift_m"] is None
    assert report["max_cloth_edge_stretch_ratio"] is None
    assert report["contained"] == [False]
    json.dumps(report, allow_nan=False)


def test_wrapper_containment_rejects_a_tail_outside_the_sock():
    """Core-only validation misses the visible clipping at a wrapper tail."""
    from examples.christmas_atelier.motion import measure_stocking

    vertices, faces, cuff = _open_box(0.025)
    pose = np.array([[0.0, 0.0, -0.12, 0.0, 0.0, 0.0, 1.0]])
    report = measure_stocking(vertices, vertices, faces, cuff, pose)
    assert report["contained"] == [True]
    assert report["wrapper_contained"] == [False]
    pose[0, 3:] = [0.0, 0.0, 2**-0.5, 2**-0.5]
    rotated = measure_stocking(vertices, vertices, faces, cuff, pose)
    assert rotated["wrapper_contained"] == [True]
