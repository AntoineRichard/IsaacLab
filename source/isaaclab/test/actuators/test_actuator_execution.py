# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for articulation-owned typed actuator execution plans."""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from collections.abc import Mapping

import pytest
import torch
import warp as wp
from torch.utils._python_dispatch import TorchDispatchMode

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_control import ActuatorJointProperties
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.cloner import ClonePlan
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache


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
        self.native_reset_ids = []
        self.native_compute_calls = 0

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
        self.native_resets += 1
        self.native_reset_ids.append(env_ids)


class _NativeControl(_Control):
    """Control double with an articulation-wide native ownership set."""

    def discover_native_actuators(self, cfgs: Mapping[str, object]) -> set[str]:
        return set(cfgs)

    def compute_native_actuators(self, view, dt: float) -> None:
        del view, dt
        self.native_compute_calls += 1


class _PartialNativeControl(_Control):
    """Control double that owns one native explicit group."""

    def __init__(self, device: str, num_worlds: int = 2) -> None:
        super().__init__(device, num_worlds)
        self.native_seen_effort: torch.Tensor | None = None

    def discover_native_actuators(self, cfgs: Mapping[str, object]) -> set[str]:
        del cfgs
        return {"native_pd"}

    def compute_native_actuators(self, view, dt: float) -> None:
        del dt
        self.native_compute_calls += 1
        view.joint_command.effort.torch[:, 0].copy_(view.command.effort.torch[:, 0])
        self.native_seen_effort = view.joint_command.effort.torch.clone()


class _OpaqueIdealPD(IdealPDActuator):
    """Opaque subclass which must remain an eager barrier."""


class _OpaqueDCMotor(DCMotor):
    """Opaque DC subclass which must remain an eager barrier."""


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
    """All-native articulations retain an empty Lab execution schedule."""
    control = _NativeControl(device)
    _, view, _ = _make_plan({"drive": _ideal_cfg(["hip"])}, device=device, control=control)

    plan = view._execution_plan

    assert plan.stateless_ranges == ()
    assert plan.eager_segments == ()
    assert plan.static_scatter_epochs == ()
    view.compute(dt=0.005)
    assert control.native_compute_calls == 1


@pytest.mark.parametrize("device", _available_devices())
def test_reset_uses_torch_view_for_lab_actuators_and_preserves_native_warp_ids(device: str) -> None:
    """Warp reset IDs reset Lab state while reaching the native hook unchanged."""
    _, view, control = _make_plan({"delayed": _delayed_cfg(["hip"])}, device=device)
    env_ids = wp.array([1], dtype=wp.int32, device=device)

    view.reset(env_ids)

    assert control.native_resets == 1
    assert control.native_reset_ids == [env_ids]


@pytest.mark.parametrize("device", _available_devices())
def test_mixed_native_and_lab_ranges_compute_only_lab_owned_types(device: str) -> None:
    """Native ownership excludes only its exact type from the Lab plan."""
    control = _PartialNativeControl(device)
    _, view, _ = _make_plan(
        {"native_pd": _ideal_cfg(["hip"]), "lab_implicit": _implicit_cfg(["knee"])},
        device=device,
        control=control,
    )
    view.command.effort.torch.copy_(torch.tensor([[13.0, 17.0, 0.0], [19.0, 23.0, 0.0]], device=device))

    view.compute(dt=0.005)

    assert [item.actuator_type for item in view._execution_plan.stateless_ranges] == [ImplicitActuator]
    assert control.native_compute_calls == 1
    assert control.native_seen_effort is not None
    torch.testing.assert_close(view.joint_command.effort.torch[:, 0], view.command.effort.torch[:, 0])
    torch.testing.assert_close(view.joint_command.effort.torch[:, 1], view.command.effort.torch[:, 1])


@pytest.mark.parametrize("device", _available_devices())
def test_mixed_same_type_native_ownership_is_rejected_with_context(device: str) -> None:
    """Mixed ownership within an exact type needs explicit selection staging."""
    control = _PartialNativeControl(device)

    with pytest.raises(RuntimeError, match=r"native/Lab.*IdealPDActuator.*native_pd.*lab_pd"):
        _make_plan(
            {"native_pd": _ideal_cfg(["hip"]), "lab_pd": _ideal_cfg(["knee"])},
            device=device,
            control=control,
        )


def _ordinary_outputs(
    actuator_type,
    cfgs: Mapping[str, object],
    control: _Control,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run separate ordinary actuators as an aggregation-independent oracle."""
    expected_position = torch.zeros((control.num_instances, control.num_joints), device=control.device)
    expected_velocity = torch.zeros_like(expected_position)
    expected_effort = torch.zeros_like(expected_position)
    expected_computed = torch.zeros_like(expected_position)
    expected_applied = torch.zeros_like(expected_position)
    if inputs is None:
        raw_position = torch.tensor([[1.0, -1.5], [-2.5, 3.0]], dtype=torch.float32, device=control.device)
        raw_velocity = torch.tensor([[0.5, -1.0], [-2.0, 2.5]], dtype=torch.float32, device=control.device)
        raw_effort = torch.tensor([[1.25, -2.25], [-4.25, 5.25]], dtype=torch.float32, device=control.device)
        joint_position = torch.tensor([[0.25, -0.75], [-0.5, 0.75]], dtype=torch.float32, device=control.device)
        joint_velocity = torch.tensor([[1.5, -2.0], [-3.0, 3.5]], dtype=torch.float32, device=control.device)
    else:
        raw_position, raw_velocity, raw_effort, joint_position, joint_velocity = inputs

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


def _random_execution_inputs(control: _Control, seed: int) -> tuple[torch.Tensor, ...]:
    """Create deterministic high-entropy command and state tensors for one device."""
    generator = torch.Generator(device=control.device).manual_seed(seed)
    return tuple(
        torch.randn((control.num_instances, control.num_joints), generator=generator, device=control.device) * 11.0
        for _ in range(5)
    )


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
@pytest.mark.parametrize("seed", [0, 1, 17])
def test_implicit_range_matches_randomized_ordinary_oracle_exactly(device: str, seed: int) -> None:
    """Catch CUDA contraction that changes the ordinary implicit result by one ULP."""
    groups = {
        "first": _implicit_cfg(["hip"], stiffness=3.125, damping=0.875),
        "second": _implicit_cfg(["knee"], stiffness=5.75, damping=1.625),
    }
    _, view, control = _make_plan(groups, device=device)
    inputs = _random_execution_inputs(control, seed)
    view.command.position.torch.copy_(inputs[0])
    view.command.velocity.torch.copy_(inputs[1])
    view.command.effort.torch.copy_(inputs[2])
    control.joint_pos.torch.copy_(inputs[3])
    control.joint_vel.torch.copy_(inputs[4])

    expected = _ordinary_outputs(ImplicitActuator, groups, control, inputs)
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
def test_eager_none_outputs_preserve_static_field_owners_across_barriers(monkeypatch, device: str) -> None:
    """Catch a static final owner table that ignores an eager runtime ``None`` output."""
    opaque_cfg = _ideal_cfg(["hip"], stiffness=4.0)
    opaque_cfg.class_type = _EffortOnlyOpaque
    groups = {
        "implicit": _implicit_cfg(["hip"], stiffness=2.0),
        "opaque": opaque_cfg,
        "last": _ideal_cfg(["knee"], stiffness=7.0),
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)
    plan = view._execution_plan
    dispatches = []
    outputs = []
    original_scatter = plan._scatter_static_epoch
    original_eager = plan._run_eager
    original_compute = view["opaque"].compute

    def _record_scatter(epoch) -> None:
        dispatches.append(("static", epoch.group_names))
        original_scatter(epoch)

    def _record_eager(segment) -> None:
        dispatches.append(("eager", segment.group_name))
        original_eager(segment)

    def _record_none_outputs(*args, **kwargs):
        output = original_compute(*args, **kwargs)
        outputs.append(output)
        return output

    monkeypatch.setattr(plan, "_scatter_static_epoch", _record_scatter)
    monkeypatch.setattr(plan, "_run_eager", _record_eager)
    monkeypatch.setattr(view["opaque"], "compute", _record_none_outputs)

    view.compute()

    assert dispatches == [("static", ("implicit",)), ("eager", "opaque"), ("static", ("last",))]
    assert outputs[0].joint_positions is None
    assert outputs[0].joint_velocities is None
    torch.testing.assert_close(view.joint_command.position.torch[:, 0], view.command.position.torch[:, 0])
    torch.testing.assert_close(view.joint_command.velocity.torch[:, 0], view.command.velocity.torch[:, 0])
    torch.testing.assert_close(view.joint_command.effort.torch[:, 0], view["opaque"].applied_effort[:, 0])
    torch.testing.assert_close(view.computed_effort.torch[:, 0], view["opaque"].computed_effort[:, 0])
    torch.testing.assert_close(view.applied_effort.torch[:, 0], view["opaque"].applied_effort[:, 0])
    torch.testing.assert_close(view.joint_command.effort.torch[:, 1], view["last"].applied_effort[:, 0])
    torch.testing.assert_close(view.computed_effort.torch[:, 1], view["last"].computed_effort[:, 0])
    torch.testing.assert_close(view.applied_effort.torch[:, 1], view["last"].applied_effort[:, 0])


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
@pytest.mark.parametrize("group_count", [1, 3, 12])
def test_all_stateless_plan_reuses_one_compute_and_one_fused_scatter(
    monkeypatch, device: str, group_count: int
) -> None:
    """Catch group-count-dependent stateless compute or final scatter dispatch."""
    groups = {
        **{f"ideal_{index}": _ideal_cfg(["hip"], stiffness=float(index + 1)) for index in range(group_count)},
        **{f"dc_{index}": _dc_cfg(["knee"], stiffness=float(index + 4)) for index in range(group_count)},
        **{f"implicit_{index}": _implicit_cfg(["ankle"], stiffness=float(index + 7)) for index in range(group_count)},
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)
    plan = view._execution_plan
    range_calls = []
    scatter_calls = []
    launch_arguments = []
    original_run_range = plan._run_range
    original_scatter = plan._scatter_static_epoch
    original_launch = _WarpLaunchCache.launch

    def _count_range(execution_range) -> None:
        range_calls.append(execution_range.actuator_type)
        original_run_range(execution_range)

    def _count_scatter(epoch) -> None:
        scatter_calls.append(epoch.group_names)
        original_scatter(epoch)

    def _record_launch(self, key, kernel, *, dim, inputs, outputs) -> None:
        launch_arguments.append((key, id(key), id(dim), id(inputs), id(outputs)))
        original_launch(self, key, kernel, dim=dim, inputs=inputs, outputs=outputs)

    monkeypatch.setattr(plan, "_run_range", _count_range)
    monkeypatch.setattr(plan, "_scatter_static_epoch", _count_scatter)
    monkeypatch.setattr(_WarpLaunchCache, "launch", _record_launch)
    view.compute()
    view.compute()

    assert range_calls == [IdealPDActuator, DCMotor, ImplicitActuator] * 2
    assert scatter_calls == [tuple(groups)] * 2
    assert len(plan.static_scatter_epochs) == 1
    assert all(isinstance(execution_range.gather_dim, tuple) for execution_range in plan.stateless_ranges)
    assert all(isinstance(execution_range.gather_key, tuple) for execution_range in plan.stateless_ranges)
    implicit_range = next(
        execution_range
        for execution_range in plan.stateless_ranges
        if execution_range.actuator_type is ImplicitActuator
    )
    assert isinstance(implicit_range.implicit_inputs, tuple)
    assert isinstance(implicit_range.implicit_outputs, tuple)
    epoch = plan.static_scatter_epochs[0]
    assert isinstance(epoch.scatter_dim, tuple)
    assert isinstance(epoch.scatter_inputs, tuple)
    assert isinstance(epoch.scatter_outputs, tuple)

    by_key = {}
    for key, key_id, dim_id, inputs_id, outputs_id in launch_arguments:
        by_key.setdefault(key, []).append((key_id, dim_id, inputs_id, outputs_id))
    assert all(len({value[0] for value in values}) == 1 for values in by_key.values())
    assert all(len({value[1] for value in values}) == 1 for values in by_key.values())
    assert all(len({value[2] for value in values}) == 1 for values in by_key.values())
    assert all(len({value[3] for value in values}) == 1 for values in by_key.values())


@pytest.mark.parametrize("device", _available_devices())
def test_plan_reuses_selector_joint_id_aliases(device: str) -> None:
    """Use candidate selector aliases instead of duplicate per-plan joint-ID buffers."""
    opaque_cfg = _ideal_cfg(["knee"])
    opaque_cfg.class_type = _OpaqueIdealPD
    _, view, _ = _make_plan({"ideal": _ideal_cfg(["hip"]), "opaque": opaque_cfg}, device=device)

    plan = view._execution_plan
    selector_state = view._selector_state
    execution_range = plan.stateless_ranges[0]
    eager_segment = plan.eager_segments[0]

    assert execution_range.joint_indices is selector_state.type_joint_ids_wp(IdealPDActuator)
    assert execution_range.action.joint_indices is selector_state.type_joint_ids(IdealPDActuator)
    assert eager_segment.joint_indices is selector_state._group_joint_ids_wp["opaque"]
    assert eager_segment.action.joint_indices is selector_state._group_joint_ids["opaque"]


def _storage_metadata(value: object) -> object:
    """Describe a Torch, Warp, or proxy storage alias without reading its values."""
    if isinstance(value, ProxyArray):
        return ("proxy", id(value), _storage_metadata(value.warp), _storage_metadata(value.torch))
    if isinstance(value, torch.Tensor):
        return (
            "torch",
            id(value),
            value.untyped_storage().data_ptr(),
            value.data_ptr(),
            value.storage_offset(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            str(value.device),
        )
    if isinstance(value, wp.array):
        tensor = wp.to_torch(value)
        return (
            "warp",
            id(value),
            value.ptr,
            tuple(value.shape),
            value.dtype,
            str(value.device),
            tensor.untyped_storage().data_ptr(),
            tensor.data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            str(tensor.device),
        )
    return value


class _NoNewTensorStorage(TorchDispatchMode):
    """Reject dispatcher results whose storage was not finalized before execution."""

    def __init__(self, tensors: tuple[torch.Tensor, ...]):
        super().__init__()
        self._storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **({} if kwargs is None else kwargs))
        leaves, _ = torch.utils._pytree.tree_flatten(result)
        for tensor in leaves:
            if isinstance(tensor, torch.Tensor):
                assert tensor.untyped_storage().data_ptr() in self._storage_pointers, func
        return result


def _explicit_range_tensors(execution_range) -> tuple[torch.Tensor, ...]:
    """Return every exact built-in action, output, and execution-scratch tensor."""
    action = execution_range.action
    assert action is not None
    executor = execution_range.executor
    binding = executor.__dict__["_parameter_binding"]
    tensors = (
        action.joint_positions,
        action.joint_velocities,
        action.joint_efforts,
        action.joint_indices,
        execution_range.staging["joint_position"].torch,
        execution_range.staging["joint_velocity"].torch,
        executor.computed_effort,
        executor.applied_effort,
        executor._effort_limit_lower,
        *(array.torch for array in binding.arrays.values()),
    )
    if type(executor) is DCMotor:
        tensors += (
            executor._vel_at_effort_lim,
            executor._joint_vel,
            executor._torque_speed_top,
            executor._torque_speed_bottom,
            executor._max_effort,
            executor._min_effort,
        )
    return tensors


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize(
    ("actuator_type", "cfg_factory"),
    [(IdealPDActuator, _ideal_cfg), (DCMotor, _dc_cfg)],
)
def test_aggregated_exact_explicit_range_rebuilds_final_shape_scratch_and_keeps_fixed_views(
    actuator_type, cfg_factory, device: str
) -> None:
    """Catch an aggregated executor retaining one source group's scratch shape or replacing fixed views."""
    groups = {
        "first": cfg_factory(["hip"]),
        "second": cfg_factory(["knee"]),
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)
    execution_range = next(
        item for item in view._execution_plan.stateless_ranges if item.actuator_type is actuator_type
    )
    tensors = _explicit_range_tensors(execution_range)
    expected_shape = (control.num_instances, 2)

    assert all(
        tensor.shape == expected_shape for tensor in tensors if tensor is not execution_range.action.joint_indices
    )
    fingerprints = tuple(_storage_metadata(tensor) for tensor in tensors)
    view.compute()
    view.compute()

    assert tuple(_storage_metadata(tensor) for tensor in tensors) == fingerprints


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize(
    ("actuator_type", "cfg_factory"),
    [(IdealPDActuator, _ideal_cfg), (DCMotor, _dc_cfg)],
)
def test_warmed_aggregated_exact_explicit_range_allocates_no_tensor_storage(
    actuator_type, cfg_factory, device: str
) -> None:
    """Catch the aggregated hot path calling allocation-producing public explicit compute."""
    groups = {
        "first": cfg_factory(["hip"]),
        "second": cfg_factory(["knee"]),
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)
    execution_range = next(
        item for item in view._execution_plan.stateless_ranges if item.actuator_type is actuator_type
    )
    tensors = _explicit_range_tensors(execution_range)
    view.compute()

    with _NoNewTensorStorage(tensors):
        view.compute()


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize(
    ("actuator_type", "exact_cfg", "subclass_cfg", "subclass_type"),
    [
        (IdealPDActuator, _ideal_cfg, _ideal_cfg, _OpaqueIdealPD),
        (DCMotor, _dc_cfg, _dc_cfg, _OpaqueDCMotor),
    ],
)
def test_exact_explicit_ranges_use_private_execution_while_subclasses_use_public_compute(
    monkeypatch, actuator_type, exact_cfg, subclass_cfg, subclass_type, device: str
) -> None:
    """Catch private execution leaking to subclasses or exact built-ins falling back to public compute."""
    opaque = subclass_cfg(["knee"])
    opaque.class_type = subclass_type
    _, view, control = _make_plan({"exact": exact_cfg(["hip"]), "opaque": opaque}, device=device)
    _set_literal_inputs(view, control)
    execution_range = next(
        item for item in view._execution_plan.stateless_ranges if item.actuator_type is actuator_type
    )
    eager_segment = view._execution_plan.eager_segments[0]
    private_calls = 0
    public_calls = 0
    exact_private = execution_range.executor._compute_execution
    subclass_public = eager_segment.actuator.compute

    def record_private(*args, **kwargs):
        nonlocal private_calls
        private_calls += 1
        return exact_private(*args, **kwargs)

    def record_public(*args, **kwargs):
        nonlocal public_calls
        public_calls += 1
        return subclass_public(*args, **kwargs)

    def reject_wrong_path(*_args, **_kwargs):
        raise AssertionError("executor used the wrong public/private compute seam")

    monkeypatch.setattr(execution_range.executor, "_compute_execution", record_private)
    monkeypatch.setattr(execution_range.executor, "compute", reject_wrong_path)
    monkeypatch.setattr(eager_segment.actuator, "compute", record_public)
    monkeypatch.setattr(eager_segment.actuator, "_compute_execution", reject_wrong_path)

    view.compute()

    assert private_calls == 1
    assert public_calls == 1


def _plan_state_fingerprint(plan, view) -> tuple[tuple[str, object], ...]:
    """Capture every Task-10-owned launch alias by a stable descriptive name."""
    aliases = {}
    for execution_range in plan.stateless_ranges:
        range_name = execution_range.actuator_type.__name__
        for name, value in execution_range.staging.items():
            aliases[f"range:{range_name}:staging:{name}"] = _storage_metadata(value)
        aliases[f"range:{range_name}:selector_ids"] = _storage_metadata(execution_range.joint_indices)
        action = execution_range.action
        if action is not None:
            for name, value in (
                ("position", action.joint_positions),
                ("velocity", action.joint_velocities),
                ("effort", action.joint_efforts),
                ("joint_ids", action.joint_indices),
            ):
                aliases[f"range:{range_name}:action:{name}"] = _storage_metadata(value)
        binding = execution_range.executor.__dict__["_parameter_binding"]
        for name, value in binding.arrays.items():
            aliases[f"range:{range_name}:parameter:{name}"] = _storage_metadata(value)
    for epoch_index, epoch in enumerate(plan.static_scatter_epochs):
        for name, value in epoch.owner_slots_by_field.items():
            aliases[f"epoch:{epoch_index}:owner:{name}"] = _storage_metadata(value)
    for name, value in (
        ("raw_position", view.command.position),
        ("raw_velocity", view.command.velocity),
        ("raw_effort", view.command.effort),
        ("processed_position", view.joint_command.position),
        ("processed_velocity", view.joint_command.velocity),
        ("processed_effort", view.joint_command.effort),
        ("computed_effort", view.computed_effort),
        ("applied_effort", view.applied_effort),
    ):
        aliases[f"joint_domain:{name}"] = _storage_metadata(value)
    return tuple(sorted(aliases.items()))


@pytest.mark.parametrize("device", _available_devices())
def test_stateless_launch_arguments_and_cached_commands_stay_stable_after_parameter_write(
    monkeypatch, device: str
) -> None:
    """Keep fixed launch aliases and CUDA command objects across public parameter writes."""
    groups = {
        "ideal": _ideal_cfg(["hip"]),
        "dc": _dc_cfg(["knee"]),
        "implicit": _implicit_cfg(["ankle"]),
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)
    plan = view._execution_plan
    arguments = defaultdict(list)
    original_launch = _WarpLaunchCache.launch

    def _record_launch(self, key, kernel, *, dim, inputs, outputs) -> None:
        arguments[key].append(
            (
                id(key),
                id(dim),
                id(inputs),
                id(outputs),
                tuple(_storage_metadata(value) for value in inputs),
                tuple(_storage_metadata(value) for value in outputs),
            )
        )
        original_launch(self, key, kernel, dim=dim, inputs=inputs, outputs=outputs)

    monkeypatch.setattr(_WarpLaunchCache, "launch", _record_launch)
    view.compute()
    cached_commands = dict(plan._launch_cache._commands)
    warm_state = _plan_state_fingerprint(plan, view)
    view.by_type[IdealPDActuator].set_parameter_index("stiffness", 9.0)
    view.compute()

    assert arguments
    assert all(len(history) == 2 and history[0] == history[1] for history in arguments.values())
    assert all(plan._launch_cache._commands[key] is command for key, command in cached_commands.items())
    assert _plan_state_fingerprint(plan, view) == warm_state


def _eager_gather_fingerprint(segment) -> tuple[object, ...]:
    """Capture fixed eager-gather aliases without inspecting dynamic output scatters."""
    return (
        segment.gather_key,
        id(segment.gather_key),
        segment.gather_dim,
        id(segment.gather_dim),
        id(segment.gather_inputs),
        id(segment.gather_outputs),
        tuple((name, _storage_metadata(value)) for name, value in segment.staging.items()),
        _storage_metadata(segment.joint_indices),
        _storage_metadata(segment.action.joint_indices),
    )


@pytest.mark.parametrize("device", _available_devices())
def test_warmed_eager_gather_aliases_and_cuda_command_stay_stable(device: str) -> None:
    """Keep a static/eager/static plan's fixed eager gather aliases stable after warm-up."""
    opaque_cfg = _ideal_cfg(["knee"])
    opaque_cfg.class_type = _OpaqueIdealPD
    groups = {
        "first": _implicit_cfg(["hip"]),
        "opaque": opaque_cfg,
        "last": _ideal_cfg(["ankle"]),
    }
    _, view, control = _make_plan(groups, device=device)
    _set_literal_inputs(view, control)
    plan = view._execution_plan
    eager_segment = plan.eager_segments[0]
    eager_key = ("eager_gather", eager_segment.group_name)

    assert eager_segment.gather_key == eager_key
    view.compute()
    warm_fingerprint = _eager_gather_fingerprint(eager_segment)
    cached_command = plan._launch_cache._commands.get(eager_key)
    view.compute()

    assert _eager_gather_fingerprint(eager_segment) == warm_fingerprint
    if device == "cuda":
        assert cached_command is not None
        assert plan._launch_cache._commands[eager_key] is cached_command
    else:
        assert eager_key not in plan._launch_cache._commands


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cached launch commands")
def test_warmed_stateless_cuda_plan_constructs_no_plan_boundary_arrays_or_synchronizes(monkeypatch) -> None:
    """Keep a warmed all-stateless CUDA execution path allocation and synchronization free."""
    from isaaclab.actuators import actuator_execution

    groups = {
        "ideal": _ideal_cfg(["hip"]),
        "dc": _dc_cfg(["knee"]),
        "implicit": _implicit_cfg(["ankle"]),
    }
    _, view, control = _make_plan(groups, device="cuda")
    _set_literal_inputs(view, control)
    view.compute()

    def _forbid(*_args, **_kwargs):
        raise AssertionError("warmed Task-10 execution constructed an array or synchronized the host")

    original_to = torch.Tensor.to

    def _forbid_cpu_transfer(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "cpu":
            _forbid()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(actuator_execution.wp, "array", _forbid)
    monkeypatch.setattr(actuator_execution.wp, "empty", _forbid)
    monkeypatch.setattr(actuator_execution.wp, "zeros", _forbid)
    monkeypatch.setattr(actuator_execution.wp, "from_torch", _forbid)
    for name in ("synchronize", "synchronize_device", "synchronize_stream"):
        if hasattr(actuator_execution.wp, name):
            monkeypatch.setattr(actuator_execution.wp, name, _forbid)
    monkeypatch.setattr(torch.cuda, "synchronize", _forbid)
    monkeypatch.setattr(torch.Tensor, "cpu", _forbid)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid)
    monkeypatch.setattr(torch.Tensor, "item", _forbid)
    monkeypatch.setattr(torch.Tensor, "to", _forbid_cpu_transfer)

    view.compute()
    view.compute()
