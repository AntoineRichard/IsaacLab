# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the MicroDuck-specific MDP terms.

The command terms only need ``num_envs``, ``device`` and the commanded asset, so they run against
an environment double instead of a simulated scene. The bucket fractions are driven to their
extremes (0.0 or 1.0) so every assertion is exact rather than statistical; the one distributional
claim -- that the buckets are independent draws rather than a partition -- is pinned with enough
environments that a false negative is impossible in practice.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest
import torch

import isaaclab_tasks.contrib.microduck.mdp as mdp

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


HEAD_POSE_RANGES = ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
"""Initial head-pose command ranges [rad] for ``(neck_pitch, head_pitch, head_yaw, head_roll)``."""

BODY_POSE_RANGES = (
    (-0.005, 0.005),
    (-0.005, 0.005),
    (-0.005, 0.005),
    (-0.05, 0.05),
    (-0.05, 0.05),
    (-0.05, 0.05),
)
"""Initial body-pose command ranges for ``(x, y, z)`` [m] and ``(roll, pitch, yaw)`` [rad]."""

ANG_VEL_Z_RANGE = (-1.0, 1.0)
"""Yaw-rate command range [rad/s] MicroDuck trains with."""


class _DummyTensorView:
    """Stands in for a ``ProxyArray``, which exposes its contents under ``.torch``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.torch = tensor


class _DummyRobotData:
    """The articulation state the velocity command reads."""

    def __init__(self, num_envs: int, device: str) -> None:
        self.root_lin_vel_b = _DummyTensorView(torch.zeros(num_envs, 3, device=device))
        self.root_ang_vel_b = _DummyTensorView(torch.zeros(num_envs, 3, device=device))
        self.heading_w = _DummyTensorView(torch.zeros(num_envs, device=device))


class _DummyRobot:
    def __init__(self, num_envs: int, device: str) -> None:
        self.data = _DummyRobotData(num_envs, device)


class _DummyVisMarkerRegistry:
    """Absorbs the debug-visualization handshake ``CommandTerm.__init__`` performs."""

    def clear_debug_vis_callback(self, term) -> None:
        pass


class _DummySimulation:
    def __init__(self) -> None:
        self.vis_marker_registry = _DummyVisMarkerRegistry()


class _DummyEnv:
    """Minimal environment double for command terms."""

    def __init__(self, num_envs: int = 8, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        self.extras: dict = {}
        self.sim = _DummySimulation()
        self.scene = {"robot": _DummyRobot(num_envs, device)}


def _make_pose_command(ranges, resampling_time_range=(2.0, 5.0), num_envs=8) -> mdp.UniformPoseDeltaCommand:
    env = _DummyEnv(num_envs=num_envs)
    cfg = mdp.UniformPoseDeltaCommandCfg(resampling_time_range=resampling_time_range, ranges=ranges)
    return mdp.UniformPoseDeltaCommand(cfg, cast("ManagerBasedRLEnv", env))


def _make_velocity_command(
    num_envs: int = 8,
    rel_standing_envs: float = 0.0,
    rel_forward_envs: float = 0.0,
    rel_turn_in_place_envs: float = 0.0,
) -> mdp.MicroDuckVelocityCommand:
    env = _DummyEnv(num_envs=num_envs)
    cfg = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=rel_standing_envs,
        rel_heading_envs=0.0,
        rel_forward_envs=rel_forward_envs,
        rel_turn_in_place_envs=rel_turn_in_place_envs,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=ANG_VEL_Z_RANGE,
            heading=(-math.pi, math.pi),
        ),
    )
    return mdp.MicroDuckVelocityCommand(cfg, cast("ManagerBasedRLEnv", env))


def _all_env_ids(term) -> torch.Tensor:
    return torch.arange(term.num_envs, device=term.device)


##
# Uniform pose-delta command
##


@pytest.mark.parametrize(
    "ranges,expected_dim",
    [(HEAD_POSE_RANGES, 4), (BODY_POSE_RANGES, 6)],
    ids=["head", "body"],
)
def test_pose_delta_command_width_follows_the_range_tuple(ranges, expected_dim):
    """The command is as wide as the range tuple is long."""
    term = _make_pose_command(ranges)

    assert term.command.shape == (term.num_envs, expected_dim)


@pytest.mark.parametrize(
    "ranges",
    [HEAD_POSE_RANGES, BODY_POSE_RANGES],
    ids=["head", "body"],
)
def test_pose_delta_command_samples_each_dimension_inside_its_own_range(ranges):
    """Every dimension stays inside the range configured for it, not the widest one."""
    torch.manual_seed(0)
    term = _make_pose_command(ranges, num_envs=512)

    term._resample_command(_all_env_ids(term))

    for dim, (low, high) in enumerate(ranges):
        column = term.command[:, dim]
        assert torch.all(column >= low)
        assert torch.all(column <= high)
        # a per-dimension range that is merely inherited from a neighbour would fail this
        assert column.abs().max() > 0.5 * high


def test_pose_delta_command_holds_its_value_until_the_resampling_clock_expires():
    """The command is held between resamples and redrawn once the sampled interval elapses."""
    torch.manual_seed(0)
    term = _make_pose_command(HEAD_POSE_RANGES, resampling_time_range=(2.0, 5.0))
    term.reset(_all_env_ids(term))
    held = term.command.clone()

    assert torch.all(term.time_left >= 2.0)
    assert torch.all(term.time_left <= 5.0)

    # below the shortest sampled interval nothing may change
    for _ in range(19):
        term.compute(dt=0.1)
    torch.testing.assert_close(term.command, held)
    assert torch.all(term.command_counter == 1)

    # past the longest sampled interval every environment must have been redrawn
    for _ in range(31):
        term.compute(dt=0.1)
    assert torch.all(term.command_counter >= 2)
    assert not torch.allclose(term.command, held)


def test_pose_delta_command_accepts_a_widened_range_but_rejects_a_narrowed_one():
    """A curriculum may replace the ranges, but only with one range per commanded dimension."""
    torch.manual_seed(0)
    term = _make_pose_command(HEAD_POSE_RANGES, num_envs=64)

    # widening in place is the curriculum's whole job
    term.cfg.ranges = tuple((10.0 * low, 10.0 * high) for low, high in HEAD_POSE_RANGES)
    term._resample_command(_all_env_ids(term))
    assert term.command[:, 0].abs().max() > HEAD_POSE_RANGES[0][1]

    # dropping a dimension would otherwise leave the last one holding a stale value
    term.cfg.ranges = HEAD_POSE_RANGES[:3]
    with pytest.raises(AssertionError):
        term._resample_command(_all_env_ids(term))


##
# MicroDuck velocity command
##


def test_turn_in_place_envs_zero_the_linear_command_and_force_a_yaw_rate():
    """A turn-in-place environment spins on the spot at no less than 40% of the yaw-rate limit."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_turn_in_place_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    max_rate = max(abs(ANG_VEL_Z_RANGE[0]), abs(ANG_VEL_Z_RANGE[1]))
    torch.testing.assert_close(term.command[:, :2], torch.zeros_like(term.command[:, :2]))
    assert torch.all(term.command[:, 2].abs() >= 0.4 * max_rate)
    assert torch.all(term.command[:, 2].abs() <= max_rate)
    # the sign is drawn, not fixed
    assert torch.any(term.command[:, 2] > 0.0)
    assert torch.any(term.command[:, 2] < 0.0)


def test_standing_envs_hold_a_zero_command():
    """A standing environment is commanded to stand still."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_standing_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    torch.testing.assert_close(term.command, torch.zeros_like(term.command))


def test_forward_envs_walk_forward_at_a_minimum_speed():
    """A forward-only environment gets a positive surge command and no sway or yaw."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_forward_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    assert torch.all(term.command[:, 0] >= 0.3)
    assert torch.all(term.command[:, 0] <= 0.4)
    torch.testing.assert_close(term.command[:, 1:], torch.zeros_like(term.command[:, 1:]))


def test_standing_takes_precedence_over_forward_only():
    """Upstream zeroes standing environments every step, after the forward bucket is applied."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_standing_envs=1.0, rel_forward_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    torch.testing.assert_close(term.command, torch.zeros_like(term.command))


def test_turn_in_place_takes_precedence_over_standing_and_forward_only():
    """Turn-in-place is applied last and clears the standing flag, so it wins every overlap."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_standing_envs=1.0, rel_forward_envs=1.0, rel_turn_in_place_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    max_rate = max(abs(ANG_VEL_Z_RANGE[0]), abs(ANG_VEL_Z_RANGE[1]))
    torch.testing.assert_close(term.command[:, :2], torch.zeros_like(term.command[:, :2]))
    assert torch.all(term.command[:, 2].abs() >= 0.4 * max_rate)
    assert not torch.any(term.is_standing_env)


def test_buckets_are_independent_draws_rather_than_a_partition():
    """Each bucket is its own Bernoulli draw, so an environment can land in several at once."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=4096, rel_standing_envs=0.5, rel_forward_envs=0.5)

    term._resample_command(_all_env_ids(term))

    assert torch.any(term.is_standing_env & term.is_forward_env)
    assert torch.any(term.is_standing_env & ~term.is_forward_env)
    assert torch.any(~term.is_standing_env & term.is_forward_env)
    assert torch.any(~term.is_standing_env & ~term.is_forward_env)


def test_disabled_buckets_leave_the_uniform_sample_alone():
    """With every fraction at zero the term reproduces the stock uniform velocity command."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    assert not torch.any(term.is_forward_env)
    assert not torch.any(term.is_standing_env)
    # a uniform draw over (-0.3, 0.3) practically never lands on the forward-only signature
    assert torch.any(term.command[:, 1] != 0.0)
    assert torch.any(term.command[:, 0] < 0.0)
