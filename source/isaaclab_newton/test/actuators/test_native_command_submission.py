# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused submission tests for native Newton actuator targets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl

from isaaclab.assets.articulation import ordering_kernels


@pytest.mark.parametrize(
    ("backend_to_user_values", "has_joint_ordering"),
    [
        pytest.param([0, 1, 2, 3], False, id="identity"),
        pytest.param([2, 0, 3, 1], True, id="permuted"),
    ],
)
def test_native_submit_uses_one_fused_launch_without_copying_identity_or_permuted_targets(
    monkeypatch: pytest.MonkeyPatch,
    backend_to_user_values: list[int],
    has_joint_ordering: bool,
) -> None:
    """Native submission gathers all targets once without replacing backend buffers."""
    command = SimpleNamespace(
        position=SimpleNamespace(warp=wp.array([[10.0, 20.0, 30.0, 40.0]], dtype=wp.float32, device="cpu")),
        velocity=SimpleNamespace(warp=wp.array([[1.0, 2.0, 3.0, 4.0]], dtype=wp.float32, device="cpu")),
        effort=SimpleNamespace(warp=wp.array([[100.0, 200.0, 300.0, 400.0]], dtype=wp.float32, device="cpu")),
    )
    backend_to_user = wp.array(backend_to_user_values, dtype=wp.int32, device="cpu")
    backend_targets = tuple(wp.full((1, 4), -1.0, dtype=wp.float32, device="cpu") for _ in range(4))
    pointers = tuple(wp.to_torch(target).data_ptr() for target in backend_targets)

    def fail_assign(*_args, **_kwargs) -> None:
        raise AssertionError("native target submission must not use wp.array.assign()")

    for target in backend_targets:
        target.assign = fail_assign

    data = SimpleNamespace(
        has_joint_ordering=has_joint_ordering,
        _sim_bind_joint_position_target=backend_targets[0],
        _sim_bind_joint_velocity_target=backend_targets[1],
        _sim_bind_joint_act=backend_targets[2],
        _sim_bind_joint_effort=backend_targets[3],
    )
    articulation = SimpleNamespace(
        num_instances=1,
        num_joints=4,
        num_fixed_tendons=0,
        device="cpu",
        data=data,
        _joint_backend_to_user_map=lambda: backend_to_user,
    )
    control = NewtonActuatorControl(articulation)
    control._native_active = True
    collection = SimpleNamespace(joint_command=command)
    launched_kernels = []
    real_launch = wp.launch

    def recording_launch(kernel, *args, **kwargs):
        launched_kernels.append(kernel)
        return real_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", recording_launch)

    control.submit_commands(collection)

    assert launched_kernels == [ordering_kernels.reorder_joint_targets_user_to_backend]
    expected_indices = torch.tensor(backend_to_user_values)
    expected_effort = torch.tensor([[100.0, 200.0, 300.0, 400.0]])[:, expected_indices]
    expected_position = torch.tensor([[10.0, 20.0, 30.0, 40.0]])[:, expected_indices]
    expected_velocity = torch.tensor([[1.0, 2.0, 3.0, 4.0]])[:, expected_indices]
    torch.testing.assert_close(wp.to_torch(data._sim_bind_joint_position_target), expected_position)
    torch.testing.assert_close(wp.to_torch(data._sim_bind_joint_velocity_target), expected_velocity)
    torch.testing.assert_close(wp.to_torch(data._sim_bind_joint_act), expected_effort)
    torch.testing.assert_close(wp.to_torch(data._sim_bind_joint_effort), expected_effort)
    assert tuple(wp.to_torch(target).data_ptr() for target in backend_targets) == pointers
    assert len(set(pointers)) == 4
