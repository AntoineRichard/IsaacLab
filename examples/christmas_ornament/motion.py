# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted hook-point waypoints for hanging the ornament, and the geometry that judges it.

The ornament starts hanging from the extended index finger: the ring's inner top edge rests
on the finger's back (the hook point), and both the finger and the target twig pass through
the ring's hole along its axis. All heights below follow from that picture.
"""

import numpy as np

# Dex3 proximal index phalanx, measured from its collision mesh (+-12.9 mm about the axis).
FINGER_THICKNESS = 0.026  # [m]

PHASES = [
    ("settle", 2.0),
    ("present", 2.5),
    ("align", 1.5),
    ("thread", 2.0),
    ("lower", 1.5),
    ("withdraw", 2.0),
    ("retreat", 1.5),
    ("home", 2.0),
    ("admire", 5.0),
]


def _finger_rotation(tangent: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) mapping local +X to -tangent and local +Z as close to up as possible."""
    x_axis = -tangent / np.linalg.norm(tangent)
    up = np.array([0.0, 0.0, 1.0])
    z_axis = up - np.dot(up, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    m = np.column_stack([x_axis, y_axis, z_axis])
    w = 0.5 * np.sqrt(max(1.0 + np.trace(m), 1e-12))
    if w > 1e-3:
        q = np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w])
    else:  # 180-degree turn about an axis in the XY plane
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4)
        q[i], q[j], q[k], q[3] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s, (m[k, j] - m[j, k]) / s
    return q / np.linalg.norm(q)


def choreography(
    hang_point: np.ndarray,
    tangent: np.ndarray,
    home: np.ndarray,
    ring_inner_radius: float,
    twig_radius: float,
    finger_thickness: float = FINGER_THICKNESS,
    expected_dip: float = 0.015,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """End time, hook-point position and hook orientation (x, y, z, w) of every phase.

    The hook starts at ``home`` at t = 0; phase ``i`` moves it to ``positions[i]`` by
    ``times[i]``.

    ``hang_point`` lies on the twig's axis; ``tangent`` is the branch's last segment direction,
    pointing outward toward the tip. ``expected_dip`` is how far the twig drops once loaded.
    """
    hang_point, tangent, home = (np.asarray(v, dtype=float) for v in (hang_point, tangent, home))
    tangent = tangent / np.linalg.norm(tangent)
    up = np.array([0.0, 0.0, 1.0])
    diameter = 2.0 * ring_inner_radius
    if diameter - finger_thickness <= 2.0 * twig_radius:
        raise ValueError("ring too small to pass a twig beside the finger")
    # Threading: the twig runs midway between the finger's underside and the ring's bottom.
    thread_height = 0.5 * (finger_thickness + diameter)
    thread = hang_point + thread_height * up
    align = thread + 0.10 * tangent
    present = align + 0.05 * up
    # Hand-over: the loaded twig carries the ring; the finger sits centred in the ring below it.
    lower = hang_point + (-expected_dip + 0.5 * (finger_thickness - diameter)) * up
    withdraw = lower + 0.15 * tangent
    retreat = withdraw - 0.05 * up
    positions = np.array([home, present, align, thread, lower, withdraw, retreat, home, home])
    times = np.cumsum([d for _, d in PHASES])  # each phase ends on its waypoint
    rotations = np.tile(_finger_rotation(tangent), (len(PHASES), 1))
    return times, positions, rotations, [name for name, _ in PHASES]


def sample_motion(time: float, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate targets with zero velocity and acceleration at each waypoint."""
    segment = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
    alpha = np.clip((time - times[segment]) / (times[segment + 1] - times[segment]), 0.0, 1.0)
    blend = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
    return targets[segment] + blend * (targets[segment + 1] - targets[segment])


def ring_encircles(
    ring_center: np.ndarray, ring_axis: np.ndarray, inner_radius: float, seg_a: np.ndarray, seg_b: np.ndarray
) -> bool:
    """True when segment ``a``-``b`` crosses the ring plane inside the ring's hole."""
    n = ring_axis / np.linalg.norm(ring_axis)
    da, db = np.dot(seg_a - ring_center, n), np.dot(seg_b - ring_center, n)
    if da * db > 0.0 or da == db:
        return False
    hit = seg_a + (da / (da - db)) * (seg_b - seg_a)
    return bool(np.linalg.norm(hit - ring_center) < inner_radius)


def min_distance_points_to_capsule(points: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> float:
    """Smallest distance from ``points`` to the surface of the capsule ``a``-``b`` (negative inside)."""
    ab = b - a
    t = np.clip((points - a) @ ab / np.dot(ab, ab), 0.0, 1.0)
    closest = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(points - closest, axis=1)) - radius)
