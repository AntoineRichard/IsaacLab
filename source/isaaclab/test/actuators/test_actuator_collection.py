# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the actuator collection runtime."""

from __future__ import annotations

import gc
import re
import warnings
import weakref
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.actuators import (
    ActuatorCollection,
    ActuatorControl,
    ActuatorJointProperties,
    DCMotor,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuator,
    IdealPDActuatorCfg,
    ImplicitActuator,
    ImplicitActuatorCfg,
)
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.cloner import ClonePlan
from isaaclab.utils.warp import ProxyArray


def _implicit_cfg() -> ImplicitActuatorCfg:
    """Create a valid implicit actuator config for collection tests."""
    return ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)


class SelectorRecordingActuator(ImplicitActuator):
    """Custom actuator that records the selector supplied to :meth:`compute`."""

    def compute(self, control_action, joint_pos, joint_vel):
        self.observed_joint_indices = control_action.joint_indices
        return super().compute(control_action, joint_pos, joint_vel)


def _ideal_cfg(joints: list[str], *, stiffness: float, damping: float, effort_limit: float):
    return IdealPDActuatorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        velocity_limit=100.0,
    )


def _dc_cfg(
    joints: list[str],
    *,
    stiffness: float,
    damping: float,
    effort_limit: float,
    velocity_limit: float,
    saturation_effort: float,
):
    return DCMotorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        saturation_effort=saturation_effort,
    )


def _make_unbatched_reference(monkeypatch, actuator_type, cfgs, control):
    with monkeypatch.context() as patch:
        patch.setattr(actuator_type, "_supports_execution_aggregation", False)
        return ActuatorCollection(cfgs, control)


def _assign_deterministic_inputs(collection: ActuatorCollection, control: FakeActuatorControl) -> None:
    control.joint_pos.torch.copy_(
        torch.tensor(
            [
                [0.35, -0.80, 1.25, -1.60],
                [-0.45, 0.95, -1.35, 1.80],
            ],
            dtype=torch.float32,
        )
    )
    control.joint_vel.torch.copy_(
        torch.tensor(
            [
                [16.0, 31.0, -17.0, -32.0],
                [-18.0, -33.0, 19.0, 34.0],
            ],
            dtype=torch.float32,
        )
    )
    collection.command.position.torch.copy_(
        torch.tensor(
            [
                [1.40, -0.20, -0.75, 2.20],
                [0.15, -1.45, 2.05, -0.65],
            ],
            dtype=torch.float32,
        )
    )
    collection.command.velocity.torch.copy_(
        torch.tensor(
            [
                [-3.5, 4.25, 5.75, -6.5],
                [7.0, -8.5, -9.25, 10.75],
            ],
            dtype=torch.float32,
        )
    )
    collection.command.effort.torch.copy_(
        torch.tensor(
            [
                [2.25, -3.50, 4.75, -5.25],
                [-6.50, 7.75, -8.25, 9.50],
            ],
            dtype=torch.float32,
        )
    )


def _assert_collection_outputs_match_exactly(actual: ActuatorCollection, reference: ActuatorCollection) -> None:
    torch.testing.assert_close(
        actual.joint_command.position.torch,
        reference.joint_command.position.torch,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual.joint_command.velocity.torch,
        reference.joint_command.velocity.torch,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual.joint_command.effort.torch,
        reference.joint_command.effort.torch,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(actual.computed_torque.torch, reference.computed_torque.torch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.applied_torque.torch, reference.applied_torque.torch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual.soft_joint_vel_limits.torch,
        reference.soft_joint_vel_limits.torch,
        rtol=0.0,
        atol=0.0,
    )


class FakeActuatorControl(ActuatorControl):
    """Small backend-neutral control object used by collection unit tests."""

    def __init__(self, *, num_envs: int = 2, joint_names: list[str] | None = None, device: str = "cpu"):
        self._num_instances = num_envs
        self._joint_names = joint_names or ["joint_0", "joint_1", "joint_2"]
        self._device = device
        self._joint_pos = ProxyArray(wp.zeros((num_envs, len(self._joint_names)), dtype=wp.float32, device=device))
        self._joint_vel = ProxyArray(wp.zeros((num_envs, len(self._joint_names)), dtype=wp.float32, device=device))
        self.written_properties: list[tuple[str, bool]] = []
        self.native_gain_writes: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.staged_commands: list[str] = []
        self.submitted = False

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
        return self._device

    @property
    def joint_pos(self) -> ProxyArray:
        return self._joint_pos

    @property
    def joint_vel(self) -> ProxyArray:
        return self._joint_vel

    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        expressions = [name_keys] if isinstance(name_keys, str) else list(name_keys)
        matches = [
            (joint_id, joint_name)
            for joint_id, joint_name in enumerate(self._joint_names)
            if any(re.fullmatch(expression, joint_name) for expression in expressions)
        ]
        return [joint_id for joint_id, _ in matches], [joint_name for _, joint_name in matches]

    def resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor | wp.array:
        if env_ids is None:
            return wp.array(list(range(self.num_instances)), dtype=wp.int32, device=self.device)
        if isinstance(env_ids, torch.Tensor | wp.array):
            return env_ids
        return wp.array(list(env_ids), dtype=wp.int32, device=self.device)

    def resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor | wp.array:
        if joint_ids is None:
            return wp.array(list(range(self.num_joints)), dtype=wp.int32, device=self.device)
        if isinstance(joint_ids, torch.Tensor | wp.array):
            return joint_ids
        return wp.array(list(joint_ids), dtype=wp.int32, device=self.device)

    def resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        return (
            env_mask
            if env_mask is not None
            else wp.array([True] * self.num_instances, dtype=wp.bool, device=self.device)
        )

    def resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        return (
            joint_mask
            if joint_mask is not None
            else wp.array([True] * self.num_joints, dtype=wp.bool, device=self.device)
        )

    def assert_shape_and_dtype(
        self, tensor: torch.Tensor | wp.array | float, shape: tuple[int, ...], dtype: type, name: str
    ) -> None:
        if isinstance(tensor, (float, int)):
            return
        if isinstance(tensor, torch.Tensor):
            assert tuple(tensor.shape) == shape
            return
        assert tensor.shape == shape
        assert tensor.dtype == dtype

    def assert_shape_and_dtype_mask(
        self, tensor: torch.Tensor | wp.array | float, masks: tuple[wp.array, ...], dtype: type, name: str
    ) -> None:
        self.assert_shape_and_dtype(tensor, tuple(mask.shape[0] for mask in masks), dtype, name)

    def get_default_joint_properties(self, joint_ids: torch.Tensor | wp.array | slice) -> ActuatorJointProperties:
        if isinstance(joint_ids, slice):
            num_joints = self.num_joints
        else:
            num_joints = joint_ids.shape[0]
        shape = (self.num_instances, num_joints)
        zeros = torch.zeros(shape, dtype=torch.float32, device=self.device)
        ones = torch.ones(shape, dtype=torch.float32, device=self.device)
        return ActuatorJointProperties(
            stiffness=zeros,
            damping=zeros,
            armature=zeros,
            friction=zeros,
            dynamic_friction=zeros,
            viscous_friction=zeros,
            effort_limit=ones * 100.0,
            velocity_limit=ones * 10.0,
        )

    def write_resolved_joint_properties(self, actuator, *, native_managed: bool) -> None:
        self.written_properties.append((actuator.__class__.__name__, native_managed))

    def write_native_actuator_gain(self, attr, values, env_ids, joint_ids) -> None:
        self.native_gain_writes.append((attr, values.clone(), env_ids.clone(), joint_ids.clone()))

    def stage_user_command(
        self,
        command_name: str,
        collection: ActuatorCollection,
        env_ids: torch.Tensor | wp.array | None,
        joint_ids: torch.Tensor | wp.array | None,
        env_mask: wp.array | None,
        joint_mask: wp.array | None,
    ) -> None:
        self.staged_commands.append(command_name)

    def submit_commands(self, collection: ActuatorCollection) -> None:
        self.submitted = True


class _ScopedSimulation:
    """Clone-plan provider for scoped command and telemetry tests."""

    def __init__(self, *, device: str, num_worlds: int = 2) -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, num_worlds), dtype=torch.bool, device=device),
            cfg_rows={1: (0,), 2: (0,)},
        )

    def get_clone_plan(self) -> ClonePlan:
        """Return the fixed two-articulation clone plan."""
        return self._clone_plan


def _scoped_ideal_cfg(joint_names: list[str]) -> IdealPDActuatorCfg:
    """Create one managed exact-type group configuration."""
    cfg = _ideal_cfg(joint_names, stiffness=1.0, damping=2.0, effort_limit=3.0)
    cfg.class_type = IdealPDActuator
    return cfg


def _make_finalized_two_articulation_manager(*, device: str = "cpu"):
    """Build two finalized scoped views with distinct articulation ranges."""
    collection = ActuatorCollection(_ScopedSimulation(device=device))
    first = collection.register_articulation(
        key="first",
        cfgs={"all": _scoped_ideal_cfg(["joint_0", "joint_1", "joint_2"])},
        control=FakeActuatorControl(device=device),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    second = collection.register_articulation(
        key="second",
        cfgs={"all": _scoped_ideal_cfg(["joint_0", "joint_1", "joint_2"])},
        control=FakeActuatorControl(device=device),
        replication_cfg_id=2,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    return collection, first, second


def _command_pointers(view) -> tuple[int, ...]:
    """Return every raw and processed command pointer for stability assertions."""
    return (
        view.command.position.torch.data_ptr(),
        view.command.velocity.torch.data_ptr(),
        view.command.effort.torch.data_ptr(),
        view.joint_command.position.torch.data_ptr(),
        view.joint_command.velocity.torch.data_ptr(),
        view.joint_command.effort.torch.data_ptr(),
    )


class NativeFakeActuatorControl(FakeActuatorControl):
    """Control object that handles actuator execution natively."""

    @property
    def native_active(self) -> bool:
        return True

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        return True


class ProxyFinderActuatorControl(FakeActuatorControl):
    """Control object whose joint finder returns cached proxy indices."""

    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[ProxyArray, list[str]]:
        return ProxyArray(wp.array([0, 2], dtype=wp.int32, device=self.device)), ["joint_0", "joint_2"]


class FakeArticulationActuatorControl(ArticulationActuatorControl):
    """Concrete shared articulation-control test adapter."""

    def submit_commands(self, collection: ActuatorCollection) -> None:
        pass


class FakeArticulation:
    """Small articulation facade for shared control tests."""

    def __init__(self):
        self.num_instances = 2
        self.num_joints = 3
        self.num_fixed_tendons = 0
        self.device = "cpu"
        shape = (self.num_instances, self.num_joints)
        zeros = torch.zeros(shape, dtype=torch.float32)
        ones = torch.ones(shape, dtype=torch.float32)
        self.data = SimpleNamespace(
            joint_pos=ProxyArray(wp.zeros(shape, dtype=wp.float32, device=self.device)),
            joint_vel=ProxyArray(wp.zeros(shape, dtype=wp.float32, device=self.device)),
            joint_stiffness=SimpleNamespace(torch=zeros),
            joint_damping=SimpleNamespace(torch=zeros),
            joint_armature=SimpleNamespace(torch=zeros),
            joint_friction_coeff=SimpleNamespace(torch=zeros),
            joint_effort_limits=SimpleNamespace(torch=ones * 100.0),
            joint_vel_limits=SimpleNamespace(torch=ones * 10.0),
        )
        self.calls: list[tuple[str, dict]] = []

    def find_joints(
        self, name_keys: str | Sequence[str], *, as_proxy: bool = False
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        joint_ids = list(range(self.num_joints))
        if as_proxy:
            resolved_ids = ProxyArray(wp.array(joint_ids, dtype=wp.int32, device=self.device))
        else:
            resolved_ids = joint_ids
        return resolved_ids, ["joint_0", "joint_1", "joint_2"]

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        values = list(range(self.num_instances)) if env_ids is None else list(env_ids)
        return wp.array(values, dtype=wp.int32, device=self.device)

    def _resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        values = list(range(self.num_joints)) if joint_ids is None else list(joint_ids)
        return wp.array(values, dtype=wp.int32, device=self.device)

    def _resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        return (
            env_mask
            if env_mask is not None
            else wp.array([True] * self.num_instances, dtype=wp.bool, device=self.device)
        )

    def _resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        return (
            joint_mask
            if joint_mask is not None
            else wp.array([True] * self.num_joints, dtype=wp.bool, device=self.device)
        )

    def assert_shape_and_dtype(
        self, tensor: torch.Tensor | wp.array | float, shape: tuple[int, ...], dtype: type, name: str
    ) -> None:
        pass

    def assert_shape_and_dtype_mask(
        self, tensor: torch.Tensor | wp.array | float, masks: tuple[wp.array, ...], dtype: type, name: str
    ) -> None:
        pass

    def write_joint_effort_limit_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("effort_limit", kwargs))

    def write_joint_velocity_limit_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("velocity_limit", kwargs))

    def write_joint_armature_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("armature", kwargs))

    def write_joint_friction_coefficient_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("friction", kwargs))

    def write_joint_stiffness_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("stiffness", kwargs))

    def write_joint_damping_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("damping", kwargs))


def test_articulation_control_provides_common_forwarding_and_property_writes():
    articulation = FakeArticulation()
    control = FakeArticulationActuatorControl(articulation)

    assert control.num_instances == articulation.num_instances
    assert control.num_joints == articulation.num_joints
    assert control.device == articulation.device

    defaults = control.get_default_joint_properties(slice(None))
    torch.testing.assert_close(defaults.dynamic_friction, torch.zeros(2, 3))
    torch.testing.assert_close(defaults.viscous_friction, torch.zeros(2, 3))

    actuator = SimpleNamespace(
        effort_limit_sim=torch.ones((2, 3)),
        velocity_limit_sim=torch.ones((2, 3)) * 2.0,
        armature=torch.ones((2, 3)) * 3.0,
        friction=torch.ones((2, 3)) * 4.0,
        stiffness=torch.ones((2, 3)) * 5.0,
        damping=torch.ones((2, 3)) * 6.0,
        joint_indices=slice(None),
    )

    control.write_resolved_joint_properties(actuator, native_managed=False)

    assert [name for name, _ in articulation.calls] == [
        "effort_limit",
        "velocity_limit",
        "armature",
        "friction",
        "stiffness",
        "damping",
    ]
    assert articulation.calls[-2][1]["stiffness"] == 0.0
    assert articulation.calls[-1][1]["damping"] == 0.0


def test_collection_is_mapping_like_and_read_only():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)

    assert list(collection.keys()) == ["all"]
    assert collection["all"] is next(iter(collection.values()))
    assert list(collection.items())[0][0] == "all"
    with pytest.raises(TypeError, match="membership is fixed"):
        collection["new"] = collection["all"]


def test_singleton_all_joint_group_preserves_public_selector():
    collection = ActuatorCollection({"all": _implicit_cfg()}, FakeActuatorControl())

    assert collection["all"].joint_indices == slice(None)


def test_custom_singleton_compute_receives_original_selector():
    cfg = _implicit_cfg()
    cfg.class_type = SelectorRecordingActuator
    collection = ActuatorCollection({"all": cfg}, FakeActuatorControl())

    collection.compute()

    assert collection["all"].observed_joint_indices == slice(None)


def test_same_stateless_class_builds_one_execution_batch_with_group_views():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=10.0, damping=1.0, effort_limit=20.0),
            "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=30.0, damping=2.0, effort_limit=40.0),
        },
        control,
    )

    assert len(collection._execution_batches) == 1
    batch = collection._execution_batches[0]
    assert type(batch.actuator) is IdealPDActuator
    assert batch.group_names == ("hips", "knees")
    assert isinstance(collection["hips"], IdealPDActuator)
    assert collection["hips"].joint_names == ["joint_0", "joint_2"]
    assert collection["hips"].stiffness.shape == (2, 2)
    torch.testing.assert_close(batch.actuator.stiffness[:, :2], torch.full((2, 2), 10.0))
    torch.testing.assert_close(batch.actuator.stiffness[:, 2:], torch.full((2, 2), 30.0))

    collection["hips"].stiffness.fill_(17.0)
    torch.testing.assert_close(batch.actuator.stiffness[:, :2], torch.full((2, 2), 17.0))
    torch.testing.assert_close(batch.actuator.stiffness[:, 2:], torch.full((2, 2), 30.0))


def test_dc_motor_execution_batch_packs_different_saturation_efforts():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _dc_cfg(
                ["joint_0", "joint_1"],
                stiffness=20.0,
                damping=1.0,
                effort_limit=40.0,
                velocity_limit=10.0,
                saturation_effort=60.0,
            ),
            "knees": _dc_cfg(
                ["joint_2", "joint_3"],
                stiffness=30.0,
                damping=2.0,
                effort_limit=70.0,
                velocity_limit=20.0,
                saturation_effort=120.0,
            ),
        },
        control,
    )

    batch = collection._execution_batches[0]
    assert type(batch.actuator) is DCMotor
    torch.testing.assert_close(
        batch.actuator._saturation_effort,
        torch.tensor([[60.0, 60.0, 120.0, 120.0]]).expand(2, -1),
    )


def test_ideal_pd_aggregate_matches_independent_groups_exactly(monkeypatch):
    joint_names = [f"joint_{index}" for index in range(4)]
    cfgs = {
        "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=12.0, damping=1.5, effort_limit=18.0),
        "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=27.0, damping=2.25, effort_limit=31.0),
    }
    reference_control = FakeActuatorControl(joint_names=joint_names)
    actual_control = FakeActuatorControl(joint_names=joint_names)
    reference = _make_unbatched_reference(monkeypatch, IdealPDActuator, cfgs, reference_control)
    actual = ActuatorCollection(cfgs, actual_control)
    _assign_deterministic_inputs(reference, reference_control)
    _assign_deterministic_inputs(actual, actual_control)

    reference.compute()
    actual.compute()

    _assert_collection_outputs_match_exactly(actual, reference)


def test_dc_motor_aggregate_matches_independent_groups_exactly(monkeypatch):
    joint_names = [f"joint_{index}" for index in range(4)]
    cfgs = {
        "hips": _dc_cfg(
            ["joint_0", "joint_2"],
            stiffness=14.0,
            damping=1.25,
            effort_limit=20.0,
            velocity_limit=10.0,
            saturation_effort=40.0,
        ),
        "knees": _dc_cfg(
            ["joint_1", "joint_3"],
            stiffness=23.0,
            damping=2.5,
            effort_limit=30.0,
            velocity_limit=20.0,
            saturation_effort=60.0,
        ),
    }
    reference_control = FakeActuatorControl(joint_names=joint_names)
    actual_control = FakeActuatorControl(joint_names=joint_names)
    reference = _make_unbatched_reference(monkeypatch, DCMotor, cfgs, reference_control)
    actual = ActuatorCollection(cfgs, actual_control)
    _assign_deterministic_inputs(reference, reference_control)
    _assign_deterministic_inputs(actual, actual_control)

    reference.compute()
    actual.compute()

    _assert_collection_outputs_match_exactly(actual, reference)


def test_implicit_aggregate_matches_independent_groups_exactly(monkeypatch):
    joint_names = [f"joint_{index}" for index in range(4)]
    cfgs = {
        "hips": ImplicitActuatorCfg(
            joint_names_expr=["joint_0", "joint_2"],
            stiffness=9.0,
            damping=0.75,
            effort_limit_sim=16.0,
            velocity_limit=7.0,
            velocity_limit_sim=70.0,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=["joint_1", "joint_3"],
            stiffness=19.0,
            damping=1.75,
            effort_limit_sim=28.0,
            velocity_limit=11.0,
            velocity_limit_sim=110.0,
        ),
    }
    reference_control = FakeActuatorControl(joint_names=joint_names)
    actual_control = FakeActuatorControl(joint_names=joint_names)
    reference = _make_unbatched_reference(monkeypatch, ImplicitActuator, cfgs, reference_control)
    actual = ActuatorCollection(cfgs, actual_control)
    _assign_deterministic_inputs(reference, reference_control)
    _assign_deterministic_inputs(actual, actual_control)

    reference.compute()
    actual.compute()

    _assert_collection_outputs_match_exactly(actual, reference)


def test_implicit_batch_bypasses_torch_actuator_compute(monkeypatch):
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": ImplicitActuatorCfg(
                joint_names_expr=["joint_0", "joint_2"],
                stiffness=9.0,
                damping=0.75,
                effort_limit_sim=16.0,
            ),
            "knees": ImplicitActuatorCfg(
                joint_names_expr=["joint_1", "joint_3"],
                stiffness=19.0,
                damping=1.75,
                effort_limit_sim=28.0,
            ),
        },
        control,
    )
    _assign_deterministic_inputs(collection, control)

    def fail_compute(*args, **kwargs):
        raise AssertionError("Implicit batches must execute through the fused Warp path")

    monkeypatch.setattr(ImplicitActuator, "compute", fail_compute)

    collection.compute()

    position_error = collection.command.position.torch - control.joint_pos.torch
    velocity_error = collection.command.velocity.torch - control.joint_vel.torch
    expected_computed = (
        collection.actuator_stiffness.torch * position_error
        + collection.actuator_damping.torch * velocity_error
        + collection.command.effort.torch
    )
    expected_applied = torch.clamp(expected_computed, min=-torch.tensor(28.0), max=torch.tensor(28.0))
    expected_applied[:, [0, 2]] = torch.clamp(expected_computed[:, [0, 2]], min=-16.0, max=16.0)
    torch.testing.assert_close(collection.computed_torque.torch, expected_computed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.applied_torque.torch, expected_applied, rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.joint_command.position.torch, collection.command.position.torch)
    torch.testing.assert_close(collection.joint_command.velocity.torch, collection.command.velocity.torch)
    torch.testing.assert_close(collection.joint_command.effort.torch, collection.command.effort.torch)


def test_aggregate_computes_once_and_refreshes_group_outputs(monkeypatch):
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(6)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_3"], stiffness=8.0, damping=0.5, effort_limit=12.0),
            "knees": _ideal_cfg(["joint_1", "joint_4"], stiffness=13.0, damping=1.0, effort_limit=18.0),
            "ankles": _ideal_cfg(["joint_2", "joint_5"], stiffness=21.0, damping=1.5, effort_limit=27.0),
        },
        control,
    )
    compute_calls = 0
    scatter_calls = 0
    original_compute = IdealPDActuator.compute
    original_scatter = collection._scatter_actuator_output

    def counted_compute(self, control_action, joint_pos, joint_vel):
        nonlocal compute_calls
        compute_calls += 1
        return original_compute(self, control_action, joint_pos, joint_vel)

    def counted_scatter(actuator, control_action, joint_indices=None):
        nonlocal scatter_calls
        scatter_calls += 1
        if joint_indices is None:
            return original_scatter(actuator, control_action)
        return original_scatter(actuator, control_action, joint_indices)

    monkeypatch.setattr(IdealPDActuator, "compute", counted_compute)
    monkeypatch.setattr(collection, "_scatter_actuator_output", counted_scatter)
    collection.command.position.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) + 0.25)
    collection.command.velocity.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) * -0.5 - 0.75)
    collection.command.effort.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) + 1.5)

    collection.compute()

    assert compute_calls == 1
    assert scatter_calls == 1
    batch = collection._execution_batches[0]
    for group_name, group_slice in zip(batch.group_names, batch.group_slices):
        torch.testing.assert_close(
            collection[group_name].computed_effort,
            batch.actuator.computed_effort[:, group_slice],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            collection[group_name].applied_effort,
            batch.actuator.applied_effort[:, group_slice],
            rtol=0.0,
            atol=0.0,
        )
    first_hips_output = collection["hips"].computed_effort
    collection.command.position.torch.mul_(-1.25)
    collection.command.velocity.torch.add_(2.75)
    collection.command.effort.torch.sub_(4.5)

    collection.compute()

    assert compute_calls == 2
    assert scatter_calls == 2
    assert collection["hips"].computed_effort is first_hips_output
    for group_name, group_slice in zip(batch.group_names, batch.group_slices):
        torch.testing.assert_close(
            collection[group_name].computed_effort,
            batch.actuator.computed_effort[:, group_slice],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            collection[group_name].applied_effort,
            batch.actuator.applied_effort[:, group_slice],
            rtol=0.0,
            atol=0.0,
        )


def test_stateless_explicit_batch_preserves_output_storage():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=8.0, damping=0.5, effort_limit=12.0),
            "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=13.0, damping=1.0, effort_limit=18.0),
        },
        control,
    )
    _assign_deterministic_inputs(collection, control)
    batch = collection._execution_batches[0]

    collection.compute()
    computed_ptr = batch.actuator.computed_effort.data_ptr()
    applied_ptr = batch.actuator.applied_effort.data_ptr()
    collection.command.position.torch.mul_(-1.25)
    collection.command.velocity.torch.add_(2.75)
    collection.command.effort.torch.sub_(4.5)

    collection.compute()

    assert batch.actuator.computed_effort.data_ptr() == computed_ptr
    assert batch.actuator.applied_effort.data_ptr() == applied_ptr


def test_stateless_explicit_batch_preserves_input_staging_storage():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=8.0, damping=0.5, effort_limit=12.0),
            "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=13.0, damping=1.0, effort_limit=18.0),
        },
        control,
    )
    _assign_deterministic_inputs(collection, control)
    batch = collection._execution_batches[0]

    collection.compute()
    pointers = (
        batch.control_action.joint_positions.data_ptr(),
        batch.control_action.joint_velocities.data_ptr(),
        batch.control_action.joint_efforts.data_ptr(),
        batch.joint_pos.data_ptr(),
        batch.joint_vel.data_ptr(),
    )
    collection.command.position.torch.mul_(-1.25)
    collection.command.velocity.torch.add_(2.75)
    collection.command.effort.torch.sub_(4.5)
    control.joint_pos.torch.add_(0.125)
    control.joint_vel.torch.sub_(0.25)

    collection.compute()

    assert pointers == (
        batch.control_action.joint_positions.data_ptr(),
        batch.control_action.joint_velocities.data_ptr(),
        batch.control_action.joint_efforts.data_ptr(),
        batch.joint_pos.data_ptr(),
        batch.joint_vel.data_ptr(),
    )


def test_stateless_explicit_batch_routes_repeated_launches_through_cache(monkeypatch):
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=8.0, damping=0.5, effort_limit=12.0),
            "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=13.0, damping=1.0, effort_limit=18.0),
        },
        control,
    )
    _assign_deterministic_inputs(collection, control)
    launch_kinds = []
    original_launch = collection._launch_cache.launch

    def record_launch(key, *args, **kwargs):
        launch_kinds.append(key[0])
        return original_launch(key, *args, **kwargs)

    monkeypatch.setattr(collection._launch_cache, "launch", record_launch)

    collection.compute()

    assert launch_kinds == ["gather", "scatter_targets", "scatter_telemetry"]


def test_stateful_subclasses_and_overlapping_groups_remain_unbatched():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    delayed = ActuatorCollection(
        {
            "first": DelayedPDActuatorCfg(
                joint_names_expr=["joint_0", "joint_1"], stiffness=1.0, damping=1.0, max_delay=0
            ),
            "second": DelayedPDActuatorCfg(
                joint_names_expr=["joint_2", "joint_3"], stiffness=2.0, damping=2.0, max_delay=0
            ),
        },
        control,
    )
    assert len(delayed._execution_batches) == 2

    overlapping = ActuatorCollection(
        {
            "first": _ideal_cfg(["joint_0", "joint_1"], stiffness=1.0, damping=1.0, effort_limit=10.0),
            "second": _ideal_cfg(["joint_1", "joint_2"], stiffness=2.0, damping=2.0, effort_limit=20.0),
        },
        FakeActuatorControl(joint_names=["joint_0", "joint_1", "joint_2"]),
    )
    assert len(overlapping._execution_batches) == 2

    cross_class = ActuatorCollection(
        {
            "ideal_a": _ideal_cfg(["joint_0"], stiffness=1.0, damping=1.0, effort_limit=10.0),
            "dc": _dc_cfg(
                ["joint_1", "joint_2"],
                stiffness=2.0,
                damping=2.0,
                effort_limit=20.0,
                velocity_limit=10.0,
                saturation_effort=30.0,
            ),
            "ideal_b": _ideal_cfg(["joint_1"], stiffness=3.0, damping=3.0, effort_limit=30.0),
        },
        FakeActuatorControl(joint_names=["joint_0", "joint_1", "joint_2"]),
    )
    assert len(cross_class._execution_batches) == 3


def test_runtime_gains_route_into_aggregate_and_native_hook():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _dc_cfg(
                ["joint_0", "joint_1"],
                stiffness=20.0,
                damping=1.0,
                effort_limit=40.0,
                velocity_limit=10.0,
                saturation_effort=60.0,
            ),
            "knees": _dc_cfg(
                ["joint_2", "joint_3"],
                stiffness=30.0,
                damping=2.0,
                effort_limit=70.0,
                velocity_limit=20.0,
                saturation_effort=120.0,
            ),
        },
        control,
    )
    env_ids = torch.tensor([1], dtype=torch.long)

    collection.write_actuator_stiffness_to_sim(
        stiffness=torch.tensor([[71.0, 93.0]]),
        env_ids=env_ids,
        joint_ids=torch.tensor([0, 3], dtype=torch.long),
    )

    torch.testing.assert_close(collection["hips"].stiffness[1, 0], torch.tensor(71.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection["knees"].stiffness[1, 1], torch.tensor(93.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.actuator_stiffness.torch[1, 0], torch.tensor(71.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.actuator_stiffness.torch[1, 3], torch.tensor(93.0), rtol=0.0, atol=0.0)
    assert control.native_gain_writes[-1][0] == "kp"

    collection.write_actuator_damping_to_sim(
        damping=torch.tensor([[47.0, 29.0]]),
        env_ids=env_ids,
        joint_ids=torch.tensor([3, 0], dtype=torch.long),
    )

    torch.testing.assert_close(collection["knees"].damping[1, 1], torch.tensor(47.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection["hips"].damping[1, 0], torch.tensor(29.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.actuator_damping.torch[1, 3], torch.tensor(47.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.actuator_damping.torch[1, 0], torch.tensor(29.0), rtol=0.0, atol=0.0)
    assert control.native_gain_writes[-1][0] == "kd"


def test_aliased_runtime_gain_values_preserve_reordered_routing():
    control = FakeActuatorControl(num_envs=1, joint_names=["joint_0", "joint_1"])
    collection = ActuatorCollection(
        {
            "all": _ideal_cfg(
                ["joint_0", "joint_1"],
                stiffness=1.0,
                damping=1.0,
                effort_limit=10.0,
            )
        },
        control,
    )
    collection["all"].stiffness.copy_(torch.tensor([[11.0, 22.0]]))
    aliased_values = collection["all"].stiffness[:, :]
    env_ids = torch.tensor([0], dtype=torch.long)
    joint_ids = torch.tensor([1, 0], dtype=torch.long)

    collection.write_actuator_stiffness_to_sim(
        stiffness=aliased_values,
        env_ids=env_ids,
        joint_ids=joint_ids,
    )

    torch.testing.assert_close(
        collection["all"].stiffness,
        torch.tensor([[22.0, 11.0]]),
        rtol=0.0,
        atol=0.0,
    )
    assert control.native_gain_writes[-1][0] == "kp"
    torch.testing.assert_close(control.native_gain_writes[-1][2], env_ids, rtol=0.0, atol=0.0)
    torch.testing.assert_close(control.native_gain_writes[-1][3], joint_ids, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        torch.cat((collection.actuator_stiffness.torch, control.native_gain_writes[-1][1])),
        torch.tensor([[22.0, 11.0], [11.0, 22.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_native_execution_bypasses_lab_aggregation_and_keeps_group_gains_current(monkeypatch):
    control = NativeFakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _dc_cfg(
                ["joint_0", "joint_1"],
                stiffness=20.0,
                damping=1.0,
                effort_limit=40.0,
                velocity_limit=10.0,
                saturation_effort=60.0,
            ),
            "knees": _dc_cfg(
                ["joint_2", "joint_3"],
                stiffness=30.0,
                damping=2.0,
                effort_limit=70.0,
                velocity_limit=20.0,
                saturation_effort=120.0,
            ),
        },
        control,
    )

    assert len(collection._execution_batches) == 2
    assert all(len(batch.group_names) == 1 for batch in collection._execution_batches)

    def fail_compute(*args, **kwargs):
        raise AssertionError("Lab actuator execution must be bypassed")

    monkeypatch.setattr(DCMotor, "compute", fail_compute)
    collection.compute()
    collection.write_actuator_stiffness_to_sim(
        stiffness=torch.tensor([[71.0, 93.0]]),
        env_ids=torch.tensor([1], dtype=torch.long),
        joint_ids=torch.tensor([0, 3], dtype=torch.long),
    )

    torch.testing.assert_close(collection["hips"].stiffness[1, 0], torch.tensor(71.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection["knees"].stiffness[1, 1], torch.tensor(93.0), rtol=0.0, atol=0.0)


def test_collection_exports_proxy_arrays():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)

    assert collection.command.position.shape == (2, 3)
    assert collection.command.velocity.shape == (2, 3)
    assert collection.command.effort.shape == (2, 3)
    assert collection.joint_command.position.shape == (2, 3)
    assert collection.joint_command.velocity.shape == (2, 3)
    assert collection.joint_command.effort.shape == (2, 3)
    assert collection.computed_torque.shape == (2, 3)
    assert collection.applied_torque.shape == (2, 3)
    assert collection.soft_joint_vel_limits.shape == (2, 3)
    assert collection.gear_ratio.shape == (2, 3)


def test_collection_accepts_cached_proxy_joint_indices():
    control = ProxyFinderActuatorControl()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        collection = ActuatorCollection({"outer": _implicit_cfg()}, control)

    assert not [warning for warning in caught_warnings if warning.category is DeprecationWarning]
    torch.testing.assert_close(collection["outer"].joint_indices, torch.tensor([0, 2], dtype=torch.int32))


def test_write_command_index_updates_only_selected_cells():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    collection.command.set_position_index(value=value, env_ids=[1], joint_ids=[0, 2])

    expected = torch.zeros(2, 3)
    expected[1, 0] = 1.0
    expected[1, 2] = 2.0
    torch.testing.assert_close(collection.command.position.torch.cpu(), expected)
    assert control.staged_commands == ["position"]


def test_write_command_index_accepts_signed_int64_selectors():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    env_ids = torch.tensor([1], dtype=torch.int64)
    joint_ids = wp.array([0, 2], dtype=wp.int64, device="cpu")

    collection.command.set_position_index(value=value, env_ids=env_ids, joint_ids=joint_ids)

    expected = torch.zeros(2, 3)
    expected[1, 0] = 3.0
    expected[1, 2] = 4.0
    torch.testing.assert_close(collection.command.position.torch.cpu(), expected)


def test_write_command_mask_uses_full_sized_value():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    joint_mask = wp.array([False, True, True], dtype=wp.bool, device="cpu")

    collection.command.set_velocity_mask(value=value, env_mask=env_mask, joint_mask=joint_mask)

    expected = torch.zeros(2, 3)
    expected[0, 1:] = value[0, 1:]
    torch.testing.assert_close(collection.command.velocity.torch.cpu(), expected)
    assert control.staged_commands == ["velocity"]


def test_compute_submits_processed_commands():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.ones(2, 3, dtype=torch.float32)
    collection.command.set_position_index(value=value, full_data=True)

    collection.compute()
    collection.submit_commands()

    torch.testing.assert_close(collection.joint_command.position.torch.cpu(), value)
    assert control.submitted


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_scoped_joint_domain_commands_are_disjoint_live_and_pointer_stable(device: str) -> None:
    """Scoped articulations retain disjoint cached raw command views."""
    _, first, second = _make_finalized_two_articulation_manager(device=device)
    first_pointer = first.command.position.torch.data_ptr()
    second.command.position.torch.fill_(8.0)

    assert first.command.position.torch.data_ptr() == first_pointer
    assert first.command.position.torch.is_contiguous()
    assert second.command.position.torch.is_contiguous()
    assert torch.count_nonzero(first.command.position.torch) == 0
    assert torch.all(second.command.position.torch == 8.0)


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_scoped_raw_and_processed_commands_have_separate_stable_storage(device: str) -> None:
    """Raw writes retain cached storage and never alias processed commands."""
    _, view, _ = _make_finalized_two_articulation_manager(device=device)
    assert view.command.position.torch.data_ptr() != view.joint_command.position.torch.data_ptr()
    pointers = _command_pointers(view)
    assert len(set(pointers)) == len(pointers)
    assert view.command.position is view.command.position
    assert view.computed_effort is view.computed_effort

    view.command.set_position_index(
        value=torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device),
        env_ids=torch.tensor([0], dtype=torch.int64, device=device),
        joint_ids=torch.tensor([1, 2], dtype=torch.int64, device=device),
    )

    assert _command_pointers(view) == pointers
    torch.testing.assert_close(
        view.command.position.torch,
        torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
    )


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_scoped_command_facades_do_not_expose_public_close(device: str) -> None:
    """Scoped command facade lifetimes are owned exclusively by their view."""
    _, view, _ = _make_finalized_two_articulation_manager(device=device)

    assert not hasattr(view.command, "close")
    assert not hasattr(view.joint_command, "close")


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_scoped_command_default_selectors_include_uncovered_joints(device: str) -> None:
    """Default command selectors cover every articulation joint, not just actuator groups."""
    collection = ActuatorCollection(_ScopedSimulation(device=device))
    view = collection.register_articulation(
        key="robot",
        cfgs={"hip": _scoped_ideal_cfg(["joint_0"])},
        control=FakeActuatorControl(device=device),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    value = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device=device)

    view.command.set_position_index(value=value)

    torch.testing.assert_close(view.command.position.torch, value)
    assert view._control.staged_commands == []


def test_retained_scoped_command_facades_reject_dirty_and_stale_generations() -> None:
    """Command facade guards run before a dirty or stale write can launch."""
    collection, view, _ = _make_finalized_two_articulation_manager()
    command = view.command
    joint_command = view.joint_command
    with pytest.warns(DeprecationWarning):
        view["duplicate"] = view["all"]

    for operation in (lambda: command.position, lambda: joint_command.position, lambda: view.computed_effort):
        with pytest.raises(RuntimeError, match="rebuild"):
            operation()

    collection.clear_generation()
    for operation in (lambda: command.position, lambda: joint_command.position):
        with pytest.raises(RuntimeError, match="stale actuator view"):
            operation()


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("set_position_index", "position"),
        ("set_velocity_index", "velocity"),
        ("set_effort_index", "effort"),
        ("set_position_mask", "position"),
        ("set_velocity_mask", "velocity"),
        ("set_effort_mask", "effort"),
    ),
)
def test_scoped_command_setters_preserve_index_mask_and_full_data_contracts(
    device: str, method: str, expected: str
) -> None:
    """Every scoped command setter accepts the established selector forms."""
    _, view, _ = _make_finalized_two_articulation_manager(device=device)
    command = view.command
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device)

    if method.endswith("index"):
        getattr(command, method)(
            value=value,
            env_ids=torch.tensor([1], dtype=torch.int64, device=device),
            joint_ids=wp.array([0, 2], dtype=wp.int64, device=device),
        )
        actual = getattr(command, expected).torch
        torch.testing.assert_close(actual[1], torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32, device=device))
    else:
        full_value = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
        getattr(command, method)(
            value=full_value,
            env_mask=torch.tensor([True, False], device=device),
            joint_mask=wp.array([False, True, True], dtype=wp.bool, device=device),
        )
        actual = getattr(command, expected).torch
        torch.testing.assert_close(
            actual,
            torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
        )

    full_data = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3) + 10.0
    command.position.torch.zero_()
    command.set_position_index(
        value=full_data,
        env_ids=torch.tensor([1], dtype=torch.int64, device=device),
        joint_ids=wp.array([2, 0], dtype=wp.int64, device=device),
        full_data=True,
    )
    expected_full_data = torch.zeros_like(full_data)
    expected_full_data[1, 2] = full_data[1, 2]
    expected_full_data[1, 0] = full_data[1, 0]
    torch.testing.assert_close(command.position.torch, expected_full_data)

    command.effort.torch.zero_()
    command.set_effort_index(
        value=torch.tensor([[7.0, 8.0]], dtype=torch.float32, device=device),
        env_ids=[0],
        joint_ids=[0, 1],
    )
    torch.testing.assert_close(
        command.effort.torch[0], torch.tensor([7.0, 8.0, 0.0], dtype=torch.float32, device=device)
    )


@pytest.mark.parametrize("method", ("set_position_index", "set_velocity_mask"))
def test_retained_scoped_command_setters_reject_dirty_and_stale_views_before_launch(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Guard command writes before a dirty or stale facade can launch Warp."""
    collection, view, _ = _make_finalized_two_articulation_manager()
    command = view.command
    with pytest.warns(DeprecationWarning):
        view["duplicate"] = view["all"]

    def _launch_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("generation guard must run before wp.launch")

    monkeypatch.setattr(wp, "launch", _launch_forbidden)
    kwargs = {"value": torch.ones((1, 1), dtype=torch.float32), "env_ids": [0], "joint_ids": [0]}
    if method.endswith("mask"):
        kwargs = {"value": torch.ones((2, 3), dtype=torch.float32), "env_mask": None, "joint_mask": None}
    with pytest.raises(RuntimeError, match="rebuild"):
        getattr(command, method)(**kwargs)

    collection.clear_generation()
    with pytest.raises(RuntimeError, match="stale actuator view"):
        getattr(command, method)(**kwargs)


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_scoped_command_setters_reuse_cached_metadata_after_warmup(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Same-device command writes do not allocate selector or source staging storage."""
    _, view, _ = _make_finalized_two_articulation_manager(device=device)
    command = view.command
    selector_state = view._selector_state
    assert selector_state is not None
    index_value = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device)
    mask_value = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    env_ids = torch.tensor([1], dtype=torch.int64, device=device)
    joint_ids = wp.array([0, 2], dtype=wp.int64, device=device)
    env_mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    joint_mask = wp.array([False, True, True], dtype=wp.bool, device=device)
    command.set_position_index(value=index_value, env_ids=env_ids, joint_ids=joint_ids)
    command.set_velocity_mask(value=mask_value, env_mask=env_mask, joint_mask=joint_mask)
    pointers = (
        selector_state._int_slab.data_ptr(),
        selector_state._bool_slab.data_ptr(),
        selector_state._identity_ids_wp.ptr,
        selector_state._all_env_mask_wp.ptr,
        selector_state._all_joint_mask_wp.ptr,
        command.position.torch.data_ptr(),
        command.velocity.torch.data_ptr(),
        command.effort.torch.data_ptr(),
    )

    def _allocation_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("steady-state scoped command write allocated host-visible storage")

    monkeypatch.setattr(torch, "arange", _allocation_forbidden)
    monkeypatch.setattr(torch, "full", _allocation_forbidden)
    monkeypatch.setattr(torch, "tensor", _allocation_forbidden)
    monkeypatch.setattr(torch, "as_tensor", _allocation_forbidden)
    monkeypatch.setattr(torch.Tensor, "contiguous", _allocation_forbidden)
    command.set_effort_index(value=wp.from_torch(index_value, dtype=wp.float32), env_ids=env_ids, joint_ids=joint_ids)
    command.set_position_mask(
        value=wp.from_torch(mask_value, dtype=wp.float32), env_mask=env_mask, joint_mask=joint_mask
    )

    assert pointers == (
        selector_state._int_slab.data_ptr(),
        selector_state._bool_slab.data_ptr(),
        selector_state._identity_ids_wp.ptr,
        selector_state._all_env_mask_wp.ptr,
        selector_state._all_joint_mask_wp.ptr,
        command.position.torch.data_ptr(),
        command.velocity.torch.data_ptr(),
        command.effort.torch.data_ptr(),
    )


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
@pytest.mark.parametrize("method", ("index", "mask"))
def test_scoped_command_selectors_do_not_read_device_values_in_normal_mode(
    monkeypatch: pytest.MonkeyPatch, device: str, method: str
) -> None:
    """Normal scoped command writes never synchronize or inspect selector contents."""
    _, view, _ = _make_finalized_two_articulation_manager(device=device)
    command = view.command
    index_value = torch.tensor([[5.0], [7.0]], dtype=torch.float32, device=device)
    mask_value = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    env_ids = torch.tensor([0, 1], dtype=torch.int64, device=device)
    joint_ids = torch.tensor([1], dtype=torch.int64, device=device)
    env_mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    joint_mask = torch.tensor([False, True, False], dtype=torch.bool, device=device)

    def _read_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("normal scoped command path read a device tensor")

    monkeypatch.setattr(torch.Tensor, "cpu", _read_forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", _read_forbidden)
    monkeypatch.setattr(torch.Tensor, "item", _read_forbidden)
    if device.startswith("cuda"):
        monkeypatch.setattr(torch.cuda, "synchronize", _read_forbidden)
    for name in ("synchronize", "synchronize_device", "synchronize_stream"):
        if hasattr(wp, name):
            monkeypatch.setattr(wp, name, _read_forbidden)

    if method == "index":
        command.set_position_index(value=index_value, env_ids=env_ids, joint_ids=joint_ids)
    else:
        command.set_velocity_mask(value=mask_value, env_mask=env_mask, joint_mask=joint_mask)


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_scoped_effort_telemetry_is_persistent_and_in_articulation_order(device: str) -> None:
    """Type-order effort outputs scatter into stable articulation-order telemetry."""
    collection = ActuatorCollection(_ScopedSimulation(device=device))
    view = collection.register_articulation(
        key="robot",
        cfgs={"rear": _scoped_ideal_cfg(["joint_2"]), "front": _scoped_ideal_cfg(["joint_0"])},
        control=FakeActuatorControl(device=device),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    generation = collection._active_generation
    assert generation is not None
    binding = generation.bindings[0]
    type_layout = binding.layout.type_layouts[IdealPDActuator]
    store = generation.stores[IdealPDActuator]
    computed = store.type_proxy(type_layout, "computed_effort").torch
    applied = store.type_proxy(type_layout, "applied_effort").torch
    computed.copy_(torch.tensor([[30.0, 10.0], [31.0, 11.0]], device=device))
    applied.copy_(torch.tensor([[300.0, 100.0], [301.0, 101.0]], device=device))
    pointers = (view.computed_effort.torch.data_ptr(), view.applied_effort.torch.data_ptr())
    view.computed_effort.torch.fill_(999.0)
    view.applied_effort.torch.fill_(999.0)

    generation.publish_effort_telemetry()

    torch.testing.assert_close(
        view.computed_effort.torch,
        torch.tensor([[10.0, 0.0, 30.0], [11.0, 0.0, 31.0]], dtype=torch.float32, device=device),
    )
    torch.testing.assert_close(
        view.applied_effort.torch,
        torch.tensor([[100.0, 0.0, 300.0], [101.0, 0.0, 301.0]], dtype=torch.float32, device=device),
    )
    assert pointers == (view.computed_effort.torch.data_ptr(), view.applied_effort.torch.data_ptr())


def test_failed_scoped_finalization_releases_retained_command_aliases_and_rejects_aba_reuse() -> None:
    """Retained scoped commands cannot revive after rollback and retry."""

    class _CompletionControl(FakeActuatorControl):
        def __init__(self, *, fail: bool) -> None:
            super().__init__()
            self.fail = fail
            self.children = None
            self.alias_refs = None

        def bind_actuator_view(self, view) -> None:
            if self.children is None:
                command = view.command
                self.children = (command, view.joint_command)
                self.alias_refs = (
                    weakref.ref(command._position),
                    weakref.ref(command._position.warp),
                    weakref.ref(command._position.torch),
                )

        def complete_articulation_initialization(self) -> None:
            if self.fail:
                raise RuntimeError("intentional command rollback")

    collection = ActuatorCollection(_ScopedSimulation(device="cpu"))
    first_control = _CompletionControl(fail=False)
    second_control = _CompletionControl(fail=True)
    first = collection.register_articulation(
        key="first",
        cfgs={"all": _scoped_ideal_cfg(["joint_0", "joint_1", "joint_2"])},
        control=first_control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.register_articulation(
        key="second",
        cfgs={"all": _scoped_ideal_cfg(["joint_0", "joint_1", "joint_2"])},
        control=second_control,
        replication_cfg_id=2,
        debug_validation=False,
        debug_value_resolution=False,
    )

    with pytest.raises(RuntimeError, match="intentional command rollback"):
        collection.finalize()
    assert first_control.children is not None
    assert first_control.alias_refs is not None
    old_command, old_joint_command = first_control.children
    raw_proxy_ref, raw_warp_ref, raw_torch_ref = first_control.alias_refs

    second_control.fail = False
    collection.finalize()
    gc.collect()

    assert raw_proxy_ref() is None
    assert raw_warp_ref() is None
    assert raw_torch_ref() is None
    for command in (old_command, old_joint_command):
        with pytest.raises(RuntimeError, match="stale actuator view"):
            _ = command.position
    assert first.command is not old_command
    assert first.joint_command is not old_joint_command
