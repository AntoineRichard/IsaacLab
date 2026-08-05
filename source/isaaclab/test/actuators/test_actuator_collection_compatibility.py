# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Characterization tests for the develop actuator compatibility surface."""

from __future__ import annotations

import re
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.actuators import (
    ActuatorCollection,
    ActuatorControl,
    ActuatorJointProperties,
    IdealPDActuatorCfg,
)
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.utils.warp import ProxyArray


class _CompatibilityControl(ActuatorControl):
    """Small control adapter that records solver and native actuator gain writes."""

    def __init__(self) -> None:
        self._num_instances = 2
        self._joint_names = ["shoulder_0", "shoulder_1", "wrist"]
        self._joint_pos = ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu"))
        self._joint_vel = ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu"))
        self.native_stiffness = torch.zeros((2, 3), dtype=torch.float32)

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def num_fixed_tendons(self) -> int:
        return 0

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def joint_pos(self) -> ProxyArray:
        return self._joint_pos

    @property
    def joint_vel(self) -> ProxyArray:
        return self._joint_vel

    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        expressions = [name_keys] if isinstance(name_keys, str) else list(name_keys)
        matches = [
            (index, name)
            for index, name in enumerate(self._joint_names)
            if any(re.fullmatch(expression, name) for expression in expressions)
        ]
        return [index for index, _ in matches], [name for _, name in matches]

    def resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_instances, dtype=torch.int32)
        return torch.as_tensor(env_ids, dtype=torch.int32)

    def resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor:
        if joint_ids is None:
            return torch.arange(self.num_joints, dtype=torch.int32)
        return torch.as_tensor(joint_ids, dtype=torch.int32)

    def resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        if env_mask is None:
            return wp.ones(self.num_instances, dtype=wp.bool, device=self.device)
        return env_mask

    def resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        if joint_mask is None:
            return wp.ones(self.num_joints, dtype=wp.bool, device=self.device)
        return joint_mask

    def assert_shape_and_dtype(
        self, tensor: torch.Tensor | wp.array | float, shape: tuple[int, ...], dtype: type, name: str
    ) -> None:
        if isinstance(tensor, (float, int)):
            return
        assert tuple(tensor.shape) == shape, name

    def assert_shape_and_dtype_mask(
        self, tensor: torch.Tensor | wp.array | float, masks: tuple[wp.array, ...], dtype: type, name: str
    ) -> None:
        self.assert_shape_and_dtype(tensor, tuple(mask.shape[0] for mask in masks), dtype, name)

    def get_default_joint_properties(self, joint_ids: torch.Tensor | wp.array | slice) -> ActuatorJointProperties:
        num_joints = self.num_joints if isinstance(joint_ids, slice) else joint_ids.shape[0]
        zeros = torch.zeros((self.num_instances, num_joints), dtype=torch.float32)
        return ActuatorJointProperties(
            stiffness=zeros,
            damping=zeros,
            armature=zeros,
            friction=zeros,
            dynamic_friction=zeros,
            viscous_friction=zeros,
            effort_limit=torch.full_like(zeros, 100.0),
            velocity_limit=torch.full_like(zeros, 10.0),
        )

    def write_resolved_joint_properties(self, actuator, *, native_managed: bool) -> None:
        pass

    def submit_commands(self, collection: ActuatorCollection) -> None:
        pass

    def write_native_actuator_gain(
        self, attr: str, values: torch.Tensor, env_ids: torch.Tensor, joint_ids: torch.Tensor
    ) -> None:
        assert attr == "kp"
        self.native_stiffness[env_ids[:, None], joint_ids] = values


def make_initialized_actuator_fixture() -> tuple[ActuatorCollection, _CompatibilityControl]:
    """Create an initialized production actuator collection with two logical groups."""
    control = _CompatibilityControl()
    actuators = ActuatorCollection(
        {
            "shoulder": IdealPDActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                stiffness=3.0,
                damping=4.0,
                effort_limit=50.0,
                velocity_limit=20.0,
            ),
            "wrist": IdealPDActuatorCfg(
                joint_names_expr=["wrist"],
                stiffness=5.0,
                damping=6.0,
                effort_limit=60.0,
                velocity_limit=30.0,
            ),
        },
        control,
    )
    return actuators, control


def test_develop_actuator_mapping_lookup_and_iteration_remain_group_scoped() -> None:
    actuators, _ = make_initialized_actuator_fixture()

    assert actuators["shoulder"] is actuators["shoulder"]
    assert list(actuators) == ["shoulder", "wrist"]
    assert list(actuators.items()) == [
        ("shoulder", actuators["shoulder"]),
        ("wrist", actuators["wrist"]),
    ]


def test_develop_actuator_group_parameters_remain_stable_inspection_views() -> None:
    actuators, _ = make_initialized_actuator_fixture()
    stiffness = actuators["shoulder"].stiffness

    assert actuators["shoulder"].stiffness.data_ptr() == stiffness.data_ptr()
    assert actuators["shoulder"].stiffness.stride() == stiffness.stride()


def test_develop_target_setters_write_command_aliases_and_warn() -> None:
    actuators, control = make_initialized_actuator_fixture()
    articulation = SimpleNamespace(actuators=actuators)
    values = torch.tensor([[3.0, 5.0]], device=control.device)
    env_ids = torch.tensor([1], device=control.device)
    joint_ids = torch.tensor([0, 2], device=control.device)

    for target_name, command_name in (("position", "position"), ("velocity", "velocity"), ("effort", "effort")):
        with pytest.warns(DeprecationWarning, match=f"set_joint_{target_name}_target is deprecated"):
            getattr(BaseArticulation, f"set_joint_{target_name}_target")(articulation, values, joint_ids, env_ids)
        command = getattr(actuators.command, command_name).torch
        torch.testing.assert_close(command[1, joint_ids], values[0])


def test_deprecated_constructor_exposes_no_unshipped_dense_actuator_surface() -> None:
    """The legacy constructor does not retain PR-only collection-wide aliases."""
    actuators, control = make_initialized_actuator_fixture()
    del control

    for name in (
        "actuator_stiffness",
        "actuator_damping",
        "soft_joint_vel_limits",
        "gear_ratio",
        "computed_torque",
        "applied_torque",
        "write_actuator_stiffness_to_sim",
        "write_actuator_damping_to_sim",
    ):
        assert not hasattr(actuators, name)
