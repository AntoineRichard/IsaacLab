# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused reset ABI tests for Newton-native actuator state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter


class _State:
    """Record the Newton per-DOF reset mask."""

    def __init__(self) -> None:
        self.masks: list[torch.Tensor | None] = []

    def reset(self, mask: wp.array | None) -> None:
        """Record a detached copy of the provided mask."""
        self.masks.append(None if mask is None else wp.to_torch(mask).clone())


def _adapter(device: str) -> tuple[NewtonActuatorAdapter, _State, _State]:
    """Build a minimal four-environment, two-DOF adapter reset seam."""
    adapter = object.__new__(NewtonActuatorAdapter)
    first, second = _State(), _State()
    adapter._device = device
    adapter._num_envs = 4
    adapter._dof_offset = 0
    adapter.num_joints = 2
    adapter.actuators = [SimpleNamespace(indices=wp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=wp.uint32, device=device))]
    adapter._states_a = [first]
    adapter._states_b = [second]
    return adapter, first, second


def _devices() -> tuple[str, ...]:
    """Return the CPU and any available CUDA device for ABI coverage."""
    return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("index_spec", "expected"),
    [
        (wp.int32, [False, False, True, True, False, False, True, True]),
        (wp.int64, [False, False, True, True, False, False, True, True]),
        (slice(1, 4, 2), [False, False, True, True, False, False, True, True]),
        (slice(3, None, -2), [False, False, True, True, False, False, True, True]),
    ],
)
def test_reset_accepts_warp_indices_and_partial_slices(device: str, index_spec: object, expected: list[bool]) -> None:
    """Native reset supports public Warp index arrays and non-full slices."""
    adapter, first, second = _adapter(device)
    env_ids = wp.array([1, 3], dtype=index_spec, device=device) if index_spec in (wp.int32, wp.int64) else index_spec

    adapter.reset(env_ids)

    expected_mask = torch.tensor(expected, dtype=torch.bool, device=device)
    assert len(first.masks) == 1
    assert len(second.masks) == 1
    torch.testing.assert_close(first.masks[0], expected_mask)
    torch.testing.assert_close(second.masks[0], expected_mask)
