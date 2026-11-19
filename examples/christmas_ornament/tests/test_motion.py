# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the hanging choreography and the geometric checks that judge it."""

import numpy as np
import pytest

from examples.christmas_ornament.motion import (
    FINGER_PITCH_DEG,
    FINGER_THICKNESS,
    PHASES,
    choreography,
    densify,
    min_distance_points_to_capsule,
    ring_encircles,
    sample_motion,
    side_offset,
)

pytestmark = [pytest.mark.kitless, pytest.mark.unit]

HANG = np.array([0.45, -0.10, 1.08])
TANGENT = np.array([-0.97, 0.0, 0.24])  # last segment points outward (toward the robot), slightly up
TANGENT = TANGENT / np.linalg.norm(TANGENT)
HOME = np.array([0.25, -0.20, 0.95])
R, TWIG = 0.030, 0.004
RING_TOP = 0.0076  # measured: the ring's inner top rides this far above the hook point


def _plan(**kwargs):
    return choreography(HANG, TANGENT, HOME, ring_inner_radius=R, twig_radius=TWIG, ring_top_offset=RING_TOP, **kwargs)


def test_phase_order_and_durations():
    times, positions, rotations, phases = _plan()
    assert phases == [name for name, _ in PHASES]
    assert PHASES == [
        ("settle", 2.0),
        ("present", 2.5),
        ("align", 1.5),
        ("thread", 2.0),
        ("lower", 2.0),
        ("release", 1.0),
        ("withdraw", 2.0),
        ("retreat", 1.5),
        ("home", 2.0),
        ("admire", 5.0),
    ]
    assert np.allclose(np.diff(np.concatenate([[0.0], times])), [d for _, d in PHASES])
    assert positions.shape == (len(PHASES), 3) and rotations.shape == (len(PHASES), 4)
    assert np.allclose(positions[0], HOME) and np.allclose(positions[-1], HOME)


def _twig_z_under(point):
    """Height of the (straight) twig axis directly under ``point``."""
    t_h = TANGENT[:2]
    along = np.dot(point[:2] - HANG[:2], t_h) / np.dot(t_h, t_h)
    return HANG[2] + along * TANGENT[2]


def _axes(q):
    x, y, z, w = q
    finger = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
    back = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    return finger, back


def _side():
    side = np.cross([0.0, 0.0, 1.0], TANGENT)
    return side / np.linalg.norm(side)


def test_thread_puts_twig_beside_and_under_finger_mid_ring():
    _, positions, _, phases = _plan()
    align, thread = positions[phases.index("align")], positions[phases.index("thread")]
    step = align - thread
    assert np.linalg.norm(np.cross(step, TANGENT)) < 1e-3  # collinear with the branch
    assert np.dot(step, TANGENT) == pytest.approx(0.10, abs=1e-3)  # starts 10 cm outward
    # Thread ends beside the hang point, just far enough sideways for the finger to pass the twig.
    offset = side_offset(TWIG)
    assert offset == pytest.approx(TWIG + 0.5 * FINGER_THICKNESS + 0.004)
    assert np.allclose(thread[:2], HANG[:2] + offset * _side()[:2])
    # The twig runs midway between the finger's underside and the hanging ring's bottom.
    twig_z = _twig_z_under(thread)
    finger_bottom = thread[2] - FINGER_THICKNESS
    ring_bottom = thread[2] + RING_TOP - 2.0 * R
    assert twig_z == pytest.approx(0.5 * (finger_bottom + ring_bottom), abs=1e-4)
    assert min(finger_bottom - twig_z, twig_z - ring_bottom) - TWIG > 0.008


def test_lower_past_the_twig_then_slide_out_of_the_ring():
    _, positions, rotations, phases = _plan()
    thread, lower, release, withdraw = (positions[phases.index(p)] for p in ("thread", "lower", "release", "withdraw"))
    # Lower: straight down beside the twig until the finger sits at the centre of the ring's hole
    # as the ring will hang from the twig, so the finger touches neither.
    assert np.allclose(lower[:2], thread[:2])
    twig_z = _twig_z_under(lower)
    finger_centre = lower[2] - 0.5 * FINGER_THICKNESS
    ring_centre = twig_z + TWIG + RING_TOP - R
    assert finger_centre == pytest.approx(ring_centre, abs=1e-6)
    lateral = side_offset(TWIG)
    assert R - lateral - 0.5 * FINGER_THICKNESS > 0.005  # finger clear of the ring's inside
    assert lateral - TWIG - 0.5 * FINGER_THICKNESS >= 0.004  # and of the twig on the way down
    # Release: hold still while the ring settles on the twig.
    assert np.allclose(release, lower)
    # Withdraw: straight back along the finger's own axis, out of the ring.
    finger, _ = _axes(rotations[phases.index("withdraw")])
    out = withdraw - release
    assert np.allclose(out / np.linalg.norm(out), -finger, atol=1e-9)
    assert np.linalg.norm(out) == pytest.approx(0.15)


def test_finger_tip_up_while_carrying():
    """Aimed along the branch in plan view and pitched tip-up whenever it carries the ring."""
    _, _, rotations, phases = _plan()
    inward = -TANGENT[:2] / np.linalg.norm(TANGENT[:2])
    for phase, _ in PHASES:
        finger, back = _axes(rotations[phases.index(phase)])
        assert finger[2] == pytest.approx(np.sin(np.radians(FINGER_PITCH_DEG)), abs=1e-6)
        assert np.allclose(finger[:2] / np.linalg.norm(finger[:2]), inward, atol=1e-6)
        assert back[2] > 0.9 and abs(np.dot(back, finger)) < 1e-6


def test_sample_motion_hits_waypoints():
    times = np.array([0.0, 1.0, 3.0])
    targets = np.array([[0.0], [1.0], [-1.0]])
    for t, target in zip(times, targets, strict=True):
        assert sample_motion(t, times, targets) == pytest.approx(target)
    # Minimum jerk: zero velocity at waypoints, midpoint halfway.
    eps = 1e-4
    assert abs(sample_motion(1.0 + eps, times, targets)[0] - 1.0) < 1e-6
    assert sample_motion(0.5, times, targets)[0] == pytest.approx(0.5)


def test_ring_encircles():
    center, axis = np.zeros(3), np.array([0.0, 1.0, 0.0])
    through = (np.array([0.0, -0.05, 0.010]), np.array([0.0, 0.05, 0.010]))
    assert ring_encircles(center, axis, R, *through)
    outside = (np.array([0.0, -0.05, R + 0.001]), np.array([0.0, 0.05, R + 0.001]))
    assert not ring_encircles(center, axis, R, *outside)
    parallel = (np.array([-0.05, 0.0, 0.010]), np.array([0.05, 0.0, 0.010]))
    assert not ring_encircles(center, axis, R, *parallel)
    short = (np.array([0.0, 0.002, 0.0]), np.array([0.0, 0.05, 0.0]))  # never reaches the ring plane
    assert not ring_encircles(center, axis, R, *short)


def test_min_distance_points_to_capsule():
    points = np.array([[0.0, 0.05, 0.0], [0.5, 0.0, 0.03]])
    assert min_distance_points_to_capsule(points, np.zeros(3), np.array([0.1, 0.0, 0.0]), 0.01) == pytest.approx(0.04)


def test_densify_follows_cartesian_path_on_minimum_jerk_timing():
    """Each phase becomes K poses on its straight segment, so IK can't swing the hand off it."""
    times, positions, rotations, phases = _plan()
    t, p, q, names = densify(times, positions, rotations, phases, HOME, samples=8)
    assert len(t) == len(p) == len(q) == len(names) == 1 + 8 * len(PHASES)
    assert np.all(np.diff(t) > 0.0) and t[0] == 0.0
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    # Phase ends land exactly on the waypoints, with the waypoint's orientation.
    for i in range(len(PHASES)):
        row = 8 * (i + 1)
        assert t[row] == pytest.approx(times[i]) and np.allclose(p[row], positions[i])
        assert abs(np.dot(q[row], rotations[i])) == pytest.approx(1.0)
    # Inside a phase, samples are uniform in time and on the segment at minimum-jerk progress.
    i = phases.index("withdraw")
    start, end = positions[i - 1], positions[i]
    for k in range(1, 8):
        row = 8 * i + k
        u = k / 8
        s = u**3 * (10 - 15 * u + 6 * u * u)
        assert t[row] == pytest.approx(times[i - 1] + u * (times[i] - times[i - 1]))
        assert np.allclose(p[row], start + s * (end - start))
        assert names[row] == "withdraw"


def test_ring_along_twig_measures_position_on_segment():
    from examples.christmas_ornament.motion import ring_along_twig

    a, b = np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])
    assert ring_along_twig(np.array([0.03, 0.0, -0.025]), a, b) == pytest.approx(0.03)
    assert ring_along_twig(np.array([0.12, 0.01, -0.02]), a, b) == pytest.approx(0.12)


def test_creep_ignores_pendulum_swing_but_catches_drift():
    """The hanging ornament rocks ~0.5 s per swing; only a shift of its mean position is creep."""
    from examples.christmas_ornament.motion import creep

    t = np.arange(0.0, 2.0, 1.0 / 60.0)
    swing = 0.003 * np.sin(2.0 * np.pi * t / 0.49)
    assert creep(t, swing) < 0.001
    drift = swing + 0.004 * t  # 8 mm over the window
    assert creep(t, drift) > 0.005
