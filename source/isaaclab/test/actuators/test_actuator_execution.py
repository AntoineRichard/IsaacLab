# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for articulation-owned typed actuator execution plans."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping

import pytest
import torch
import warp as wp

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_control import ActuatorJointProperties
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.cloner import ClonePlan
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray


def _available_devices() -> tuple[str, ...]:
    return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


class _Simulation:
    """Minimal simulation source used to construct one canonical generation."""

    def __init__(self, device: str, num_worlds: int = 2) -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, num_worlds), dtype=torch.bool, device=device),
            cfg_rows={1: (0,)},
        )

    def get_clone_plan(self) -> ClonePlan:
        """Return the one-source clone plan."""
        return self._clone_plan


class _Control:
    """Backend-neutral control double with stable joint state buffers."""

    def __init__(self, device: str, num_worlds: int = 2) -> None:
        self._device = device
        self._num_worlds = num_worlds
        self._joint_names = ("hip", "knee", "ankle")
        self._joint_pos = ProxyArray(wp.zeros((num_worlds, len(self._joint_names)), dtype=wp.float32, device=device))
        self._joint_vel = ProxyArray(wp.zeros((num_worlds, len(self._joint_names)), dtype=wp.float32, device=device))
        self.submissions = 0
        self.native_resets = 0

    @property
    def num_instances(self) -> int:
        return self._num_worlds

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def device(self) -> str:
        return self._device

    @property
    def joint_pos(self) -> ProxyArray:
        return self._joint_pos

    @property
    def joint_vel(self) -> ProxyArray:
        return self._joint_vel

    def discover_native_actuators(self, cfgs: Mapping[str, object]) -> set[str]:
        del cfgs
        return set()

    def find_joints(self, names: str | list[str]) -> tuple[list[int], list[str]]:
        expressions = [names] if isinstance(names, str) else names
        matches = [
            (index, name)
            for index, name in enumerate(self._joint_names)
            if any(re.fullmatch(expression, name) for expression in expressions)
        ]
        return [index for index, _ in matches], [name for _, name in matches]

    def get_source_joint_properties(
        self, joint_ids: torch.Tensor, source_env_ids: torch.Tensor
    ) -> ActuatorJointProperties:
        shape = (source_env_ids.shape[0], joint_ids.shape[0])
        zeros = torch.zeros(shape, dtype=torch.float32, device=self.device)
        return ActuatorJointProperties(
            stiffness=zeros,
            damping=zeros,
            armature=zeros,
            friction=zeros,
            dynamic_friction=zeros,
            viscous_friction=zeros,
            effort_limit=torch.full_like(zeros, 100.0),
            velocity_limit=torch.full_like(zeros, 30.0),
        )

    def get_default_joint_properties(self, joint_ids: torch.Tensor | slice) -> ActuatorJointProperties:
        count = self.num_joints if isinstance(joint_ids, slice) else joint_ids.shape[0]
        zeros = torch.zeros((self.num_instances, count), dtype=torch.float32, device=self.device)
        return ActuatorJointProperties(
            stiffness=zeros,
            damping=zeros,
            armature=zeros,
            friction=zeros,
            dynamic_friction=zeros,
            viscous_friction=zeros,
            effort_limit=torch.full_like(zeros, 100.0),
            velocity_limit=torch.full_like(zeros, 30.0),
        )

    def prepare_actuator_binding(self, binding) -> None:
        del binding

    def bind_actuator_view(self, view) -> None:
        del view

    def complete_articulation_initialization(self) -> None:
        pass

    def invalidate_actuator_view(self) -> None:
        pass

    def write_resolved_joint_properties_staged(self, properties) -> None:
        del properties

    def validate_resolved_joint_properties(self) -> None:
        pass

    def restore_resolved_joint_properties(self) -> None:
        pass

    def commit_resolved_joint_properties(self) -> None:
        pass

    def submit_commands(self, view) -> None:
        del view
        self.submissions += 1

    def reset_native_actuators(self, env_ids) -> None:
        del env_ids
        self.native_resets += 1


class _NativeControl(_Control):
    """Control double that retains Task 10's articulation-wide native bypass."""

    def discover_native_actuators(self, cfgs: Mapping[str, object]) -> set[str]:
        return set(cfgs)

    def compute_native_actuators(self, view, dt: float) -> bool:
        del view, dt
        return True


class _OpaqueIdealPD(IdealPDActuator):
    """Opaque subclass which must remain an eager barrier."""


class _EffortOnlyOpaque(IdealPDActuator):
    """Opaque actuator whose ordinary output deliberately omits targets."""


def _ideal_cfg(joints: list[str], *, stiffness: float = 2.0, damping: float = 0.5) -> IdealPDActuatorCfg:
    return IdealPDActuatorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=60.0,
        velocity_limit=20.0,
    )


def _dc_cfg(joints: list[str], *, stiffness: float = 2.0, damping: float = 0.5) -> DCMotorCfg:
    return DCMotorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=15.0,
        velocity_limit=10.0,
        saturation_effort=25.0,
    )


def _implicit_cfg(joints: list[str], *, stiffness: float = 2.0, damping: float = 0.5) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=60.0,
        velocity_limit=20.0,
    )


def _delayed_cfg(joints: list[str]) -> DelayedPDActuatorCfg:
    return DelayedPDActuatorCfg(
        joint_names_expr=joints,
        stiffness=2.0,
        damping=0.5,
        effort_limit=60.0,
        velocity_limit=20.0,
        min_delay=0,
        max_delay=1,
    )


def _make_plan(
    groups: Mapping[str, object], *, device: str = "cpu", control: _Control | None = None
) -> tuple[ActuatorCollection, object, _Control]:
    control = _Control(device) if control is None else control
    collection = ActuatorCollection(_Simulation(device))
    view = collection.register_articulation(
        key="robot",
        cfgs=groups,
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    return collection, view, control


def _set_literal_inputs(view, control: _Control) -> None:
    control.joint_pos.torch.copy_(
        torch.tensor([[0.25, -0.75, 1.0], [-0.5, 0.75, -1.25]], dtype=torch.float32, device=control.device)
    )
    control.joint_vel.torch.copy_(
        torch.tensor([[1.5, -2.0, 2.5], [-3.0, 3.5, -4.0]], dtype=torch.float32, device=control.device)
    )
    view.command.position.torch.copy_(
        torch.tensor([[1.0, -1.5, 2.0], [-2.5, 3.0, -3.5]], dtype=torch.float32, device=control.device)
    )
    view.command.velocity.torch.copy_(
        torch.tensor([[0.5, -1.0, 1.5], [-2.0, 2.5, -3.0]], dtype=torch.float32, device=control.device)
    )
    view.command.effort.torch.copy_(
        torch.tensor([[1.25, -2.25, 3.25], [-4.25, 5.25, -6.25]], dtype=torch.float32, device=control.device)
    )


@pytest.mark.parametrize("device", _available_devices())
def test_plan_builds_one_range_per_exact_stateless_type(device: str) -> None:
    """Catch per-group execution plans for exact stateless built-ins."""
    groups = {
        **{f"pd_{index}": _ideal_cfg(["hip"], stiffness=float(index + 1)) for index in range(3)},
        **{f"dc_{index}": _dc_cfg(["knee"], stiffness=float(index + 4)) for index in range(3)},
        **{f"implicit_{index}": _implicit_cfg(["ankle"], stiffness=float(index + 7)) for index in range(2)},
    }
    _, view, _ = _make_plan(groups, device=device)

    plan = view._execution_plan
    assert [execution_range.actuator_type for execution_range in plan.stateless_ranges] == [
        IdealPDActuator,
        DCMotor,
        ImplicitActuator,
    ]
    assert tuple(len(execution_range.group_names) for execution_range in plan.stateless_ranges) == (3, 3, 2)
    assert len(plan.static_scatter_epochs) == 1


@pytest.mark.parametrize("device", _available_devices())
def test_plan_does_not_split_ranges_for_different_numeric_parameters(device: str) -> None:
    """Catch a range signature that includes numeric actuator parameters."""
    groups = {
        f"group_{index}": _ideal_cfg(["hip"], stiffness=float(index + 1), damping=float(index + 2))
        for index in range(12)
    }
    _, view, _ = _make_plan(groups, device=device)

    plan = view._execution_plan
    assert len(plan.stateless_ranges) == 1
    assert plan.stateless_ranges[0].group_names == tuple(f"group_{index}" for index in range(12))


@pytest.mark.parametrize("device", _available_devices())
def test_native_articulation_bypass_owns_no_lab_execution_schedule(device: str) -> None:
    """Keep the established whole-articulation native execution short circuit."""
    control = _NativeControl(device)
    _, view, _ = _make_plan({"drive": _ideal_cfg(["hip"])}, device=device, control=control)

    plan = view._execution_plan

    assert plan.stateless_ranges == ()
    assert plan.eager_segments == ()
    assert plan.static_scatter_epochs == ()
    view.compute(dt=0.005)


def _ordinary_outputs(actuator_type, cfgs: Mapping[str, object], control: _Control) -> tuple[torch.Tensor, ...]:
    """Run separate ordinary actuators as an aggregation-independent oracle."""
    expected_position = torch.zeros((control.num_instances, control.num_joints), device=control.device)
    expected_velocity = torch.zeros_like(expected_position)
    expected_effort = torch.zeros_like(expected_position)
    expected_computed = torch.zeros_like(expected_position)
    expected_applied = torch.zeros_like(expected_position)
    raw_position = torch.tensor([[1.0, -1.5], [-2.5, 3.0]], dtype=torch.float32, device=control.device)
    raw_velocity = torch.tensor([[0.5, -1.0], [-2.0, 2.5]], dtype=torch.float32, device=control.device)
    raw_effort = torch.tensor([[1.25, -2.25], [-4.25, 5.25]], dtype=torch.float32, device=control.device)
    joint_position = torch.tensor([[0.25, -0.75], [-0.5, 0.75]], dtype=torch.float32, device=control.device)
    joint_velocity = torch.tensor([[1.5, -2.0], [-3.0, 3.5]], dtype=torch.float32, device=control.device)

    for joint_id, (_, source_cfg) in enumerate(cfgs.items()):
        cfg = copy.deepcopy(source_cfg)
        actuator = actuator_type(
            cfg=cfg,
            joint_names=[("hip", "knee")[joint_id]],
            joint_ids=torch.tensor([joint_id], dtype=torch.int32, device=control.device),
            num_envs=control.num_instances,
            device=control.device,
            stiffness=torch.zeros((control.num_instances, 1), device=control.device),
            damping=torch.zeros((control.num_instances, 1), device=control.device),
            armature=torch.zeros((control.num_instances, 1), device=control.device),
            friction=torch.zeros((control.num_instances, 1), device=control.device),
            dynamic_friction=torch.zeros((control.num_instances, 1), device=control.device),
            viscous_friction=torch.zeros((control.num_instances, 1), device=control.device),
            effort_limit=torch.full((control.num_instances, 1), 100.0, device=control.device),
            velocity_limit=torch.full((control.num_instances, 1), 30.0, device=control.device),
        )
        action = actuator.compute(
            ArticulationActions(
                joint_positions=raw_position[:, joint_id : joint_id + 1],
                joint_velocities=raw_velocity[:, joint_id : joint_id + 1],
                joint_efforts=raw_effort[:, joint_id : joint_id + 1],
            ),
            joint_position[:, joint_id : joint_id + 1],
            joint_velocity[:, joint_id : joint_id + 1],
        )
        if action.joint_positions is not None:
            expected_position[:, joint_id] = action.joint_positions[:, 0]
        if action.joint_velocities is not None:
            expected_velocity[:, joint_id] = action.joint_velocities[:, 0]
        if action.joint_efforts is not None:
            expected_effort[:, joint_id] = action.joint_efforts[:, 0]
        expected_computed[:, joint_id] = actuator.computed_effort[:, 0]
        expected_applied[:, joint_id] = actuator.applied_effort[:, 0]
    return expected_position, expected_velocity, expected_effort, expected_computed, expected_applied


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("actuator_type", [ImplicitActuator, IdealPDActuator, DCMotor])
def test_aggregated_range_matches_independent_groups_exactly(actuator_type, device: str) -> None:
    """Catch aggregation that changes exact built-in actuator math or telemetry."""
    cfg_factory = {
        ImplicitActuator: _implicit_cfg,
        IdealPDActuator: _ideal_cfg,
        DCMotor: _dc_cfg,
    }[actuator_type]
    groups = {
        "first": cfg_factory(["hip"], stiffness=3.0, damping=0.75),
        "second": cfg_factory(["knee"], stiffness=5.0, damping=1.25),
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)

    expected = _ordinary_outputs(actuator_type, groups, control)
    view.compute(dt=0.005)

    torch.testing.assert_close(view.joint_command.position.torch[:, :2], expected[0][:, :2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(view.joint_command.velocity.torch[:, :2], expected[1][:, :2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(view.joint_command.effort.torch[:, :2], expected[2][:, :2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(view.computed_effort.torch[:, :2], expected[3][:, :2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(view.applied_effort.torch[:, :2], expected[4][:, :2], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", _available_devices())
def test_overlapping_stateless_groups_preserve_last_writer_order(device: str) -> None:
    """Catch aggregation that uses compact-slot order instead of config order."""
    groups = {
        "first": _ideal_cfg(["hip"], stiffness=1.0),
        "second": _ideal_cfg(["hip"], stiffness=3.0),
        "third": _ideal_cfg(["hip"], stiffness=7.0),
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)

    view.compute()

    torch.testing.assert_close(view.joint_command.effort.torch[:, 0], view["third"].applied_effort[:, 0])
    torch.testing.assert_close(view.computed_effort.torch[:, 0], view["third"].computed_effort[:, 0])
    torch.testing.assert_close(view.applied_effort.torch[:, 0], view["third"].applied_effort[:, 0])


@pytest.mark.parametrize("device", _available_devices())
def test_mixed_overlap_uses_last_writer_per_present_output_field(device: str) -> None:
    """Catch one owner table that lets a DC group erase implicit targets."""
    groups = {"implicit": _implicit_cfg(["hip"]), "dc": _dc_cfg(["hip"], stiffness=6.0)}
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)

    view.compute()

    torch.testing.assert_close(view.joint_command.position.torch[:, 0], view.command.position.torch[:, 0])
    torch.testing.assert_close(view.joint_command.velocity.torch[:, 0], view.command.velocity.torch[:, 0])
    torch.testing.assert_close(view.joint_command.effort.torch[:, 0], view["dc"].applied_effort[:, 0])
    torch.testing.assert_close(view.computed_effort.torch[:, 0], view["dc"].computed_effort[:, 0])
    torch.testing.assert_close(view.applied_effort.torch[:, 0], view["dc"].applied_effort[:, 0])


@pytest.mark.parametrize("device", _available_devices())
def test_delayed_and_subclass_groups_remain_ordered_eager_segments(device: str) -> None:
    """Catch inherited aggregation support that merges stateful or subclassed groups."""
    opaque_cfg = _ideal_cfg(["ankle"])
    opaque_cfg.class_type = _OpaqueIdealPD
    groups = {
        "implicit": _implicit_cfg(["hip"]),
        "delayed": _delayed_cfg(["knee"]),
        "opaque": opaque_cfg,
    }
    _, view, _ = _make_plan(groups, device=device)

    plan = view._execution_plan
    assert [segment.group_name for segment in plan.eager_segments] == ["delayed", "opaque"]
    assert all(segment.group_names == (segment.group_name,) for segment in plan.eager_segments)


@pytest.mark.parametrize("device", _available_devices())
def test_eager_none_outputs_preserve_static_field_owners_across_barriers(device: str) -> None:
    """Catch a static final owner table that ignores an eager runtime ``None`` output."""
    opaque_cfg = _ideal_cfg(["hip"], stiffness=4.0)
    opaque_cfg.class_type = _EffortOnlyOpaque
    groups = {
        "implicit": _implicit_cfg(["hip"], stiffness=2.0),
        "opaque": opaque_cfg,
        "last": _ideal_cfg(["hip"], stiffness=7.0),
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)

    view.compute()

    torch.testing.assert_close(view.joint_command.position.torch[:, 0], view.command.position.torch[:, 0])
    torch.testing.assert_close(view.joint_command.velocity.torch[:, 0], view.command.velocity.torch[:, 0])
    torch.testing.assert_close(view.joint_command.effort.torch[:, 0], view["last"].applied_effort[:, 0])
    torch.testing.assert_close(view.computed_effort.torch[:, 0], view["last"].computed_effort[:, 0])
    torch.testing.assert_close(view.applied_effort.torch[:, 0], view["last"].applied_effort[:, 0])


@pytest.mark.parametrize("device", _available_devices())
def test_plan_rejects_dirty_then_stale_generation(device: str) -> None:
    """Catch retained plans that survive a requested rebuild or STOP."""
    collection, view, _ = _make_plan({"drive": _ideal_cfg(["hip"])}, device=device)
    plan = view._execution_plan

    collection.stage_deprecated_mutation(view, "delete", "drive", None)
    with pytest.raises(RuntimeError, match="late registration"):
        plan.compute()
    collection.clear_generation()
    with pytest.raises(RuntimeError, match="stale"):
        plan.compute()


@pytest.mark.parametrize("device", _available_devices())
def test_plan_reuses_stable_staging_and_one_compute_per_stateless_type(monkeypatch, device: str) -> None:
    """Catch plan-owned scratch or dispatch that scales with homogeneous group count."""
    groups = {f"group_{index}": _ideal_cfg(["hip"], stiffness=float(index + 1)) for index in range(12)}
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)
    plan = view._execution_plan
    execution_range = plan.stateless_ranges[0]
    pointers_before = tuple(array.torch.data_ptr() for array in execution_range.staging.values())
    calls = 0
    original_compute = IdealPDActuator.compute

    def _count_compute(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(IdealPDActuator, "compute", _count_compute)
    view.compute()
    view.compute()

    assert calls == 2
    assert tuple(array.torch.data_ptr() for array in execution_range.staging.values()) == pointers_before
    assert len(plan.static_scatter_epochs) == 1
