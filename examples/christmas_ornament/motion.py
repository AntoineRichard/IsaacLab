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
# Hook point to fingertip along the distal phalanx [m] (hook at 35 mm of a 52 mm phalanx).
HOOK_TO_TIP = 0.017
# Pitched this far tip-down, the ring slides off the fingertip: past the ring-on-finger friction
# angle (~33 degrees for the 0.8 / 0.5 ring and finger friction).
FINGER_TIP_PITCH_DEG = 40.0
# The finger aims along the branch in plan view but stays pitched tip-up: outer branches slope
# upward toward their tips, and a finger copying that slope points down and sheds the ring.
FINGER_PITCH_DEG = 5.0

PHASES = [
    ("settle", 2.0),
    ("present", 2.5),
    ("align", 1.5),
    ("thread", 2.0),
    ("lower", 1.5),
    ("tip", 1.5),
    ("withdraw", 2.0),
    ("retreat", 1.5),
    ("home", 2.0),
    ("admire", 5.0),
]


def finger_rotation(tangent: np.ndarray, pitch_deg: float = FINGER_PITCH_DEG) -> np.ndarray:
    """Hook-link orientation (x, y, z, w): local +X along -tangent in plan view, pitched
    ``pitch_deg`` tip-up; local +Z (the finger's back) as close to up as possible."""
    inward = -np.asarray(tangent, dtype=float)[:2]
    inward /= np.linalg.norm(inward)
    pitch = np.radians(pitch_deg)
    x_axis = np.array([inward[0] * np.cos(pitch), inward[1] * np.cos(pitch), np.sin(pitch)])
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


def _x_axis(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])


def choreography(
    hang_point: np.ndarray,
    tangent: np.ndarray,
    home: np.ndarray,
    ring_inner_radius: float,
    twig_radius: float,
    finger_thickness: float = FINGER_THICKNESS,
    ring_top_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """End time, hook-point position and hook orientation (x, y, z, w) of every phase.

    The hook starts at ``home`` at t = 0; phase ``i`` moves it to ``positions[i]`` by
    ``times[i]``.

    ``hang_point`` lies on the twig's axis; ``tangent`` is the branch's last segment direction,
    pointing outward toward the tip. ``ring_top_offset`` is how far the ring's inner top rides
    above the hook point, as measured on the hanging ornament.

    Hand-over: thread the twig under the finger, midway down the ring; lower until the finger is
    just above the twig; then pitch the finger tip-down about its fingertip so the ring slides off
    the tip onto the twig, inboard of the finger. Withdrawing then cannot touch the ring.
    """
    hang_point, tangent, home = (np.asarray(v, dtype=float) for v in (hang_point, tangent, home))
    tangent = tangent / np.linalg.norm(tangent)
    up = np.array([0.0, 0.0, 1.0])
    space = 2.0 * ring_inner_radius - ring_top_offset - finger_thickness
    if space <= 2.0 * twig_radius:
        raise ValueError("ring too small to pass a twig under the finger")
    carry = finger_rotation(tangent)
    tipped = finger_rotation(tangent, -FINGER_TIP_PITCH_DEG)
    carry_dir, tipped_dir = _x_axis(carry), _x_axis(tipped)
    # Thread right over the hang point: the ring slides ~1-2 cm inward off the fingertip, landing
    # toward the fork; from further out, branch bounce walks it off the tip.
    over = hang_point  # the twig's axis under the hook
    thread = over + 0.5 * (finger_thickness + 2.0 * ring_inner_radius - ring_top_offset) * up
    align = thread + 0.10 * tangent
    present = align + 0.05 * up
    # 12 mm above the twig: the twig rises toward the palm ~15 degrees relative to the finger.
    lower = over + (twig_radius + 0.012 + finger_thickness) * up
    fingertip = lower + HOOK_TO_TIP * carry_dir
    tip = fingertip + 0.005 * up - HOOK_TO_TIP * tipped_dir  # pivot about the (lifted) fingertip
    withdraw = tip + 0.15 * tangent
    retreat = withdraw - 0.05 * up
    positions = np.array([home, present, align, thread, lower, tip, withdraw, retreat, home, home])
    rotations = np.array([tipped if name in ("tip", "withdraw") else carry for name, _ in PHASES])
    times = np.cumsum([d for _, d in PHASES])  # each phase ends on its waypoint
    return times, positions, rotations, [name for name, _ in PHASES]


def minimum_jerk(alpha):
    """Minimum-jerk progress for normalised time ``alpha`` in [0, 1]."""
    return alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)


def sample_motion(time: float, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate targets with zero velocity and acceleration at each waypoint."""
    segment = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
    alpha = np.clip((time - times[segment]) / (times[segment + 1] - times[segment]), 0.0, 1.0)
    return targets[segment] + minimum_jerk(alpha) * (targets[segment + 1] - targets[segment])


def sample_linear(time: float, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of ``targets`` rows (for densified paths)."""
    segment = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
    alpha = np.clip((time - times[segment]) / (times[segment + 1] - times[segment]), 0.0, 1.0)
    return targets[segment] + alpha * (targets[segment + 1] - targets[segment])


def _slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        theta = np.arccos(d)
        q = (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)


def densify(
    times: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    phases: list[str],
    home: np.ndarray,
    samples: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Expand each phase into ``samples`` poses on its straight segment at minimum-jerk progress.

    Joint-space interpolation between two IK solutions can swing the hand far off the segment
    joining them (and did: it pressed the twig down and catapulted the ornament). Solving IK on
    these dense poses and interpolating joints linearly between them keeps the hand on the path.
    Returns times, positions, rotations (x, y, z, w) and the phase name of every row; row 0 is
    ``home`` at t = 0.
    """
    t, p, q, names = [0.0], [np.asarray(home, dtype=float)], [rotations[0]], [phases[0]]
    t0, p0, q0 = 0.0, np.asarray(home, dtype=float), rotations[0]
    for i, name in enumerate(phases):
        for k in range(1, samples + 1):
            u = k / samples
            s = minimum_jerk(u)
            t.append(t0 + u * (times[i] - t0))
            p.append(p0 + s * (positions[i] - p0))
            q.append(_slerp(q0, rotations[i], s))
            names.append(name)
        t0, p0, q0 = times[i], positions[i], rotations[i]
    return np.array(t), np.array(p), np.array(q), names


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
