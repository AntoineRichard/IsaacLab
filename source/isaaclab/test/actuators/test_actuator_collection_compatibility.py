# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Characterization tests for the develop actuator compatibility surface."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import pytest
import torch
import warp as wp

from isaaclab.actuators import (
    ActuatorCollection,
    ActuatorControl,
    ActuatorJointProperties,
    IdealPDActuatorCfg,
)
from isaaclab.utils.warp import ProxyArray


class _CompatibilityControl(ActuatorControl):
    """Small control adapter that records solver and native actuator gain writes."""

    def __init__(self) -> None:
        self._num_instances = 2
        self._joint_names = ["shoulder_0", "shoulder_1", "wrist"]
        self._joint_pos = ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu"))
        self._joint_vel = ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu"))
        self.native_stiffness = torch.zeros((2, 3), dtype=torch.float32)
        self.solver_stiffness = torch.zeros((2, 3), dtype=torch.float32)

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

    def write_solver_stiffness(self, stiffness: torch.Tensor, env_ids: torch.Tensor, joint_ids: torch.Tensor) -> None:
        self.solver_stiffness[env_ids[:, None], joint_ids] = stiffness


class _CompatibilityData:
    """Deprecated data aliases bound to an articulation-scoped actuator view."""

    def __init__(self, actuators: ActuatorCollection) -> None:
        self._actuators = actuators

    def _get_alias(self, name: str):
        command_field = {
            "joint_pos_target": "position",
            "joint_vel_target": "velocity",
            "joint_effort_target": "effort",
        }.get(name)
        replacement = f"command.{command_field}" if command_field is not None else name
        warnings.warn(
            f"ArticulationData.{name} is deprecated. Use articulation.actuators.{replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (
            getattr(self._actuators.command, command_field)
            if command_field is not None
            else getattr(self._actuators, name)
        )

    @property
    def joint_pos_target(self):
        return self._get_alias("joint_pos_target")

    @property
    def joint_vel_target(self):
        return self._get_alias("joint_vel_target")

    @property
    def joint_effort_target(self):
        return self._get_alias("joint_effort_target")

    @property
    def computed_torque(self):
        return self._get_alias("computed_torque")

    @property
    def applied_torque(self):
        return self._get_alias("applied_torque")

    @property
    def soft_joint_vel_limits(self):
        return self._get_alias("soft_joint_vel_limits")

    @property
    def gear_ratio(self):
        return self._get_alias("gear_ratio")


class _CompatibilityArticulation:
    """Focused articulation fake that preserves the public compatibility routes."""

    def __init__(self, actuators: ActuatorCollection, control: _CompatibilityControl) -> None:
        self.actuators = actuators
        self.data = _CompatibilityData(actuators)
        self._control = control
        self.device = control.device

    def write_actuator_stiffness_to_sim(self, *, stiffness, env_ids, joint_ids) -> None:
        warnings.warn(
            "Articulation.write_actuator_stiffness_to_sim is deprecated. Use"
            " articulation.actuators.write_actuator_stiffness_to_sim instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.actuators.write_actuator_stiffness_to_sim(stiffness=stiffness, env_ids=env_ids, joint_ids=joint_ids)

    def write_joint_stiffness_to_sim_index(self, stiffness, *, env_ids, joint_ids) -> None:
        self._control.write_solver_stiffness(stiffness, env_ids, joint_ids)

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None) -> None:
        self._set_target("position", target, joint_ids, env_ids)

    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None) -> None:
        self._set_target("velocity", target, joint_ids, env_ids)

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None) -> None:
        self._set_target("effort", target, joint_ids, env_ids)

    def _set_target(self, name: str, target, joint_ids, env_ids) -> None:
        warnings.warn(
            f"Articulation.set_joint_{name}_target is deprecated. Use "
            f"articulation.actuators.command.set_{name}_index instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        getattr(self.actuators.command, f"set_{name}_index")(value=target, joint_ids=joint_ids, env_ids=env_ids)


def make_initialized_articulation_fixture() -> tuple[_CompatibilityArticulation, _CompatibilityControl]:
    """Create an articulation compatibility fixture with two logical actuator groups."""
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
    return _CompatibilityArticulation(actuators, control), control


def test_develop_actuator_mapping_lookup_and_iteration_remain_group_scoped() -> None:
    articulation, _ = make_initialized_articulation_fixture()

    assert articulation.actuators["shoulder"] is articulation.actuators["shoulder"]
    assert list(articulation.actuators) == ["shoulder", "wrist"]
    assert list(articulation.actuators.items()) == [
        ("shoulder", articulation.actuators["shoulder"]),
        ("wrist", articulation.actuators["wrist"]),
    ]


def test_develop_actuator_group_parameters_remain_writable_in_place() -> None:
    articulation, _ = make_initialized_articulation_fixture()
    stiffness = articulation.actuators["shoulder"].stiffness

    stiffness[0, 1] = 29.0

    assert articulation.actuators["shoulder"].stiffness.data_ptr() == stiffness.data_ptr()
    assert articulation.actuators["shoulder"].stiffness[0, 1].item() == 29.0


def test_develop_target_setters_write_command_aliases_and_warn() -> None:
    articulation, _ = make_initialized_articulation_fixture()
    values = torch.tensor([[3.0, 5.0]], device=articulation.device)
    env_ids = torch.tensor([1], device=articulation.device)
    joint_ids = torch.tensor([0, 2], device=articulation.device)

    for target_name, command_name in (("position", "position"), ("velocity", "velocity"), ("effort", "effort")):
        with pytest.warns(DeprecationWarning, match=f"set_joint_{target_name}_target is deprecated"):
            getattr(articulation, f"set_joint_{target_name}_target")(values, joint_ids=joint_ids, env_ids=env_ids)
        command = getattr(articulation.actuators.command, command_name).torch
        torch.testing.assert_close(command[1, joint_ids], values[0])


def test_develop_data_aliases_preserve_identity_and_warn() -> None:
    articulation, _ = make_initialized_articulation_fixture()
    aliases = (
        ("joint_pos_target", articulation.actuators.command.position),
        ("joint_vel_target", articulation.actuators.command.velocity),
        ("joint_effort_target", articulation.actuators.command.effort),
        ("computed_torque", articulation.actuators.computed_torque),
        ("applied_torque", articulation.actuators.applied_torque),
        ("soft_joint_vel_limits", articulation.actuators.soft_joint_vel_limits),
        ("gear_ratio", articulation.actuators.gear_ratio),
    )

    for data_name, actuator_value in aliases:
        with pytest.warns(DeprecationWarning, match=f"ArticulationData.{data_name} is deprecated"):
            assert getattr(articulation.data, data_name) is actuator_value


def test_develop_actuator_gain_writers_update_groups_and_backend() -> None:
    articulation, control = make_initialized_articulation_fixture()
    values = torch.tensor([[11.0, 17.0]], device=articulation.device)
    with pytest.warns(DeprecationWarning, match="write_actuator_stiffness_to_sim is deprecated"):
        articulation.write_actuator_stiffness_to_sim(
            stiffness=values,
            env_ids=torch.tensor([0], device=articulation.device),
            joint_ids=torch.tensor([0, 1], device=articulation.device),
        )
    torch.testing.assert_close(articulation.actuators["shoulder"].stiffness[:1], values)
    torch.testing.assert_close(control.native_stiffness[:1, :2], values)


def test_solver_gain_writers_remain_distinct_from_actuator_parameters() -> None:
    articulation, control = make_initialized_articulation_fixture()
    before = articulation.actuators["shoulder"].stiffness.clone()
    articulation.write_joint_stiffness_to_sim_index(
        torch.tensor([[31.0]], device=articulation.device),
        env_ids=torch.tensor([0], device=articulation.device),
        joint_ids=torch.tensor([0], device=articulation.device),
    )
    torch.testing.assert_close(articulation.actuators["shoulder"].stiffness, before)
    assert control.solver_stiffness[0, 0].item() == 31.0
