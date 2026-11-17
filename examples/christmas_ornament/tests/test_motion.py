# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the hanging choreography and the geometric checks that judge it."""

import numpy as np
import pytest

from examples.christmas_ornament.motion import (
    FINGER_THICKNESS,
    PHASES,
    choreography,
    min_distance_points_to_capsule,
    ring_encircles,
    sample_motion,
)

pytestmark = [pytest.mark.kitless, pytest.mark.unit]

HANG = np.array([0.45, -0.10, 1.08])
TANGENT = np.array([-0.97, 0.0, 0.24])  # last segment points outward (toward the robot), slightly up
TANGENT = TANGENT / np.linalg.norm(TANGENT)
HOME = np.array([0.25, -0.20, 0.95])
R, TWIG = 0.020, 0.004


def _plan(**kwargs):
    return choreography(HANG, TANGENT, HOME, ring_inner_radius=R, twig_radius=TWIG, **kwargs)


def test_phase_order_and_durations():
    times, positions, rotations, phases = _plan()
    assert phases == [name for name, _ in PHASES]
    assert PHASES == [
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
    assert np.allclose(np.diff(np.concatenate([[0.0], times])), [d for _, d in PHASES])
    assert positions.shape == (len(PHASES), 3) and rotations.shape == (len(PHASES), 4)
    assert np.allclose(positions[0], HOME) and np.allclose(positions[-1], HOME)


def test_thread_is_along_tangent_and_twig_clears_finger():
    times, positions, _, phases = _plan()
    align, thread = positions[phases.index("align")], positions[phases.index("thread")]
    step = align - thread
    assert np.linalg.norm(np.cross(step, TANGENT)) < 1e-3  # collinear with the branch
    assert np.dot(step, TANGENT) == pytest.approx(0.10, abs=1e-3)  # starts 10 cm outward
    # While threading, the twig runs midway between the finger's underside and the ring's bottom.
    finger_bottom = thread[2] - FINGER_THICKNESS
    ring_bottom = thread[2] - 2.0 * R
    assert HANG[2] == pytest.approx(0.5 * (finger_bottom + ring_bottom), abs=1e-4)
    assert finger_bottom - HANG[2] > TWIG and HANG[2] - ring_bottom > TWIG


def test_lower_hands_over_and_withdraw_slides_out():
    _, positions, _, phases = _plan(expected_dip=0.015)
    lower, withdraw = positions[phases.index("lower")], positions[phases.index("withdraw")]
    twig_loaded = HANG[2] - 0.015
    # After the hand-over the ring rests on the twig; the finger sits centred inside it, below the twig.
    ring_inner_top = twig_loaded + TWIG
    ring_inner_bottom = ring_inner_top - 2.0 * R
    assert lower[2] < twig_loaded - TWIG
    assert lower[2] - FINGER_THICKNESS > ring_inner_bottom
    assert (twig_loaded - TWIG) - lower[2] == pytest.approx(lower[2] - FINGER_THICKNESS - ring_inner_bottom, abs=1e-4)
    out = withdraw - lower
    assert np.linalg.norm(np.cross(out, TANGENT)) < 1e-3 and np.dot(out, TANGENT) == pytest.approx(0.15, abs=1e-3)


def test_finger_orientation_constant_and_along_branch():
    _, _, rotations, _ = _plan()
    assert np.allclose(rotations, rotations[0])
    x, y, z, w = rotations[0]
    # Local +X (the finger) maps to -tangent; local +Z (the finger's back) points up.
    finger = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
    back = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    assert np.allclose(finger, -TANGENT, atol=1e-6)
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
