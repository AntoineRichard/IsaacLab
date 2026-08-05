# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private execution plans for articulation-scoped actuator collections."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import warp as wp

from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache

from . import actuator_kernels
from .actuator_base import ActuatorBase
from .actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from .actuator_storage import _GroupBinding

if TYPE_CHECKING:
    from .actuator_collection import _ArticulationBinding, _SelectorState
    from .actuator_control import ActuatorControl


_FIELD_NAMES = ("position", "velocity", "effort", "computed_effort", "applied_effort")
_SUPPORTED_TYPES = (ImplicitActuator, IdealPDActuator, DCMotor)


@dataclass(frozen=True)
class _ExecutionRange:
    """One exact-type stateless range with fixed command/state staging."""

    actuator_type: type[ActuatorBase]
    group_names: tuple[str, ...]
    group_slices: tuple[slice, ...]
    joint_indices: wp.array(dtype=wp.int32)
    graphable: bool
    staging: Mapping[str, ProxyArray]
    executor: ActuatorBase
    action: ArticulationActions | None
    gather_key: tuple[str, type[ActuatorBase]]
    gather_dim: tuple[int, int]
    gather_inputs: tuple[object, ...]
    gather_outputs: tuple[object, ...]
    implicit_key: tuple[str, type[ActuatorBase]] | None
    implicit_dim: tuple[int, int] | None
    implicit_inputs: tuple[object, ...] | None
    implicit_outputs: tuple[object, ...] | None


@dataclass(frozen=True)
class _EagerSegment:
    """One non-aggregate actuator group preserving ordinary compute behavior."""

    group_name: str
    group_names: tuple[str, ...]
    actuator: ActuatorBase
    joint_indices: wp.array(dtype=wp.int32)
    staging: Mapping[str, ProxyArray]
    action: ArticulationActions
    gather_key: tuple[str, str]
    gather_dim: tuple[int, int]
    scatter_dim: tuple[int, int]
    gather_inputs: tuple[object, ...]
    gather_outputs: tuple[object, ...]
    action_scatter_outputs: tuple[object, ...]
    telemetry_scatter_outputs: tuple[object, ...]


@dataclass(frozen=True)
class _StaticScatterEpoch:
    """Configuration-order static writers between eager execution barriers."""

    group_names: tuple[str, ...]
    owner_slots_by_field: Mapping[str, wp.array(dtype=wp.int32)]
    scatter_key: tuple[str, tuple[str, ...]] | None = None
    scatter_dim: tuple[int, int] | None = None
    scatter_inputs: tuple[object, ...] | None = None
    scatter_outputs: tuple[object, ...] | None = None


class _ArticulationExecutionPlan:
    """Preallocated execution plan for one private articulation binding."""

    def __init__(
        self,
        *,
        control: ActuatorControl,
        stateless_ranges: tuple[_ExecutionRange, ...],
        eager_segments: tuple[_EagerSegment, ...],
        static_scatter_epochs: tuple[_StaticScatterEpoch, ...],
        schedule: tuple[_StaticScatterEpoch | _EagerSegment, ...],
    ) -> None:
        self._control: ActuatorControl | None = control
        self.stateless_ranges = stateless_ranges
        self.eager_segments = eager_segments
        self.static_scatter_epochs = static_scatter_epochs
        self._schedule = schedule
        self._launch_cache = _WarpLaunchCache(control.device)
        self._valid = True
        self._validate_execution: Callable[[], None] | None = None
        self._native_compute: Callable[[float], None] | None = None

    @classmethod
    def build(
        cls,
        *,
        binding: _ArticulationBinding,
        control: ActuatorControl,
        selector_state: _SelectorState,
    ) -> _ArticulationExecutionPlan:
        """Build a complete private plan without reading a public facade."""
        if binding.groups is None or binding.command is None or binding.joint_command is None:
            raise RuntimeError("Cannot build an execution plan from an incomplete candidate binding.")
        if binding.computed_effort is None or binding.applied_effort is None:
            raise RuntimeError("Cannot build an execution plan without candidate telemetry aliases.")

        layout = binding.layout
        groups = binding.groups
        targets = MappingProxyType(
            {
                "position": binding.joint_command.position,
                "velocity": binding.joint_command.velocity,
                "effort": binding.joint_command.effort,
                "computed_effort": binding.computed_effort,
                "applied_effort": binding.applied_effort,
            }
        )

        native_group_names = binding.native_group_names
        lab_type_groups: dict[type[ActuatorBase], tuple[Any, ...]] = {}
        type_offsets: dict[type[ActuatorBase], int] = {}
        offset = 0
        for actuator_type in _SUPPORTED_TYPES:
            type_layout = layout.type_layouts.get(actuator_type)
            type_offsets[actuator_type] = offset
            if type_layout is None:
                continue
            type_groups = tuple(group for group in layout.group_layouts if group.actuator_type is actuator_type)
            native_type_groups = tuple(group for group in type_groups if group.name in native_group_names)
            lab_groups = tuple(group for group in type_groups if group.name not in native_group_names)
            if native_type_groups and lab_groups:
                native_names = ", ".join(group.name for group in native_type_groups)
                lab_names = ", ".join(group.name for group in lab_groups)
                raise RuntimeError(
                    f"Mixed native/Lab ownership within exact type {actuator_type.__name__} "
                    f"for articulation {binding.registration.key!r}: native [{native_names}], Lab [{lab_names}]."
                )
            lab_type_groups[actuator_type] = lab_groups
            if lab_groups:
                offset += type_layout.num_dofs

        source_buffers: dict[type[ActuatorBase], Mapping[str, ProxyArray]] = {}
        output_buffers: dict[type[ActuatorBase], Mapping[str, ProxyArray]] = {}
        ranges: list[_ExecutionRange] = []
        for actuator_type, type_layout in layout.type_layouts.items():
            if actuator_type not in _SUPPORTED_TYPES:
                continue
            type_groups = lab_type_groups.get(actuator_type, ())
            if not type_groups:
                continue
            execution_range = cls._build_range(
                actuator_type=actuator_type,
                type_layout=type_layout,
                group_layouts=type_groups,
                groups=groups,
                command=binding.command,
                control=control,
                selector_state=selector_state,
            )
            ranges.append(execution_range)
            source_buffers[actuator_type] = execution_range.staging
            group_binding = execution_range.executor.__dict__["_parameter_binding"]
            output_buffers[actuator_type] = MappingProxyType(
                {
                    "computed_effort": group_binding.arrays["computed_effort"],
                    "applied_effort": group_binding.arrays["applied_effort"],
                }
            )

        eager_segments, epochs, schedule = cls._build_ordered_segments(
            binding=binding,
            groups=groups,
            type_offsets=type_offsets,
            native_group_names=native_group_names,
            control=control,
            selector_state=selector_state,
        )
        implicit_layout = layout.type_layouts.get(ImplicitActuator)
        ideal_layout = layout.type_layouts.get(IdealPDActuator)
        epochs, schedule = cls._bind_static_launches(
            epochs=epochs,
            schedule=schedule,
            source_buffers=source_buffers,
            output_buffers=output_buffers,
            targets=targets,
            num_worlds=layout.num_worlds,
            num_joints=layout.num_joints,
            device=control.device,
            implicit_count=0
            if not lab_type_groups.get(ImplicitActuator) or implicit_layout is None
            else implicit_layout.num_dofs,
            ideal_count=0
            if not lab_type_groups.get(IdealPDActuator) or ideal_layout is None
            else ideal_layout.num_dofs,
        )
        return cls(
            control=control,
            stateless_ranges=tuple(ranges),
            eager_segments=eager_segments,
            static_scatter_epochs=epochs,
            schedule=schedule,
        )

    @classmethod
    def _build_range(
        cls,
        *,
        actuator_type: type[ActuatorBase],
        type_layout: Any,
        group_layouts: tuple[Any, ...],
        groups: Mapping[str, ActuatorBase],
        command: Any,
        control: ActuatorControl,
        selector_state: _SelectorState,
    ) -> _ExecutionRange:
        device = control.device
        joint_indices_torch = selector_state.type_joint_ids(actuator_type)
        joint_indices = selector_state.type_joint_ids_wp(actuator_type)
        staging = MappingProxyType(
            {
                "position": ProxyArray(
                    wp.empty((type_layout.num_worlds, type_layout.num_dofs), dtype=wp.float32, device=device)
                ),
                "velocity": ProxyArray(
                    wp.empty((type_layout.num_worlds, type_layout.num_dofs), dtype=wp.float32, device=device)
                ),
                "effort": ProxyArray(
                    wp.empty((type_layout.num_worlds, type_layout.num_dofs), dtype=wp.float32, device=device)
                ),
                "joint_position": ProxyArray(
                    wp.empty((type_layout.num_worlds, type_layout.num_dofs), dtype=wp.float32, device=device)
                ),
                "joint_velocity": ProxyArray(
                    wp.empty((type_layout.num_worlds, type_layout.num_dofs), dtype=wp.float32, device=device)
                ),
            }
        )
        first_group = groups[group_layouts[0].name]
        first_binding = first_group.__dict__.get("_parameter_binding")
        if not isinstance(first_binding, _GroupBinding):
            raise RuntimeError(f"Stateless actuator {group_layouts[0].name!r} is missing canonical parameter storage.")
        executor = copy.copy(first_group)
        executor.__dict__["_parameter_binding"] = replace(
            first_binding,
            joint_indices=joint_indices_torch,
            joint_names=tuple(name for group in group_layouts for name in group.joint_names),
            type_slice=slice(0, type_layout.num_dofs),
        )
        executor.__dict__["_joint_indices"] = joint_indices_torch
        executor.__dict__["_joint_names"] = [name for group in group_layouts for name in group.joint_names]
        executor.__dict__.pop("_facade_view", None)
        executor.__dict__.pop("_facade_token", None)
        if actuator_type is DCMotor:
            executor._rebuild_managed_runtime_state()

        gather_inputs = (
            command.position.warp,
            command.velocity.warp,
            command.effort.warp,
            control.joint_pos.warp,
            control.joint_vel.warp,
            joint_indices,
        )
        gather_outputs = (
            staging["position"].warp,
            staging["velocity"].warp,
            staging["effort"].warp,
            staging["joint_position"].warp,
            staging["joint_velocity"].warp,
        )
        action = None
        if actuator_type is not ImplicitActuator:
            action = ArticulationActions(
                joint_positions=staging["position"].torch,
                joint_velocities=staging["velocity"].torch,
                joint_efforts=staging["effort"].torch,
                joint_indices=joint_indices_torch,
            )
        implicit_inputs = None
        implicit_outputs = None
        if actuator_type is ImplicitActuator:
            binding = executor.__dict__["_parameter_binding"]
            implicit_inputs = (
                staging["position"].warp,
                staging["velocity"].warp,
                staging["effort"].warp,
                staging["joint_position"].warp,
                staging["joint_velocity"].warp,
                binding.arrays["stiffness"].warp,
                binding.arrays["damping"].warp,
                binding.arrays["effort_limit"].warp,
            )
            implicit_outputs = (binding.arrays["computed_effort"].warp, binding.arrays["applied_effort"].warp)
        return _ExecutionRange(
            actuator_type=actuator_type,
            group_names=tuple(group.name for group in group_layouts),
            group_slices=tuple(group.type_slice for group in group_layouts),
            joint_indices=joint_indices,
            graphable=actuator_type._parameter_schema().graphable,
            staging=staging,
            executor=executor,
            action=action,
            gather_key=("gather", actuator_type),
            gather_dim=(type_layout.num_worlds, type_layout.num_dofs),
            gather_inputs=gather_inputs,
            gather_outputs=gather_outputs,
            implicit_key=("implicit", actuator_type) if implicit_inputs is not None else None,
            implicit_dim=(type_layout.num_worlds, type_layout.num_dofs) if implicit_inputs is not None else None,
            implicit_inputs=implicit_inputs,
            implicit_outputs=implicit_outputs,
        )

    @classmethod
    def _build_ordered_segments(
        cls,
        *,
        binding: _ArticulationBinding,
        groups: Mapping[str, ActuatorBase],
        type_offsets: Mapping[type[ActuatorBase], int],
        native_group_names: frozenset[str],
        control: ActuatorControl,
        selector_state: _SelectorState,
    ) -> tuple[
        tuple[_EagerSegment, ...],
        tuple[_StaticScatterEpoch, ...],
        tuple[_StaticScatterEpoch | _EagerSegment, ...],
    ]:
        eager_segments: list[_EagerSegment] = []
        ordered_entries: list[_StaticScatterEpoch | _EagerSegment] = []
        epoch_layouts: list[Any] = []
        for group_layout in binding.layout.group_layouts:
            if group_layout.name in native_group_names:
                continue
            exact_stateless = type(group := groups[group_layout.name]) in _SUPPORTED_TYPES
            if exact_stateless:
                epoch_layouts.append(group_layout)
                continue
            if epoch_layouts:
                ordered_entries.append(
                    _StaticScatterEpoch(
                        group_names=tuple(group.name for group in epoch_layouts),
                        owner_slots_by_field=cls._epoch_owner_rows(
                            group_layouts=tuple(epoch_layouts),
                            type_offsets=type_offsets,
                            num_joints=binding.layout.num_joints,
                            device=control.device,
                        ),
                    )
                )
                epoch_layouts.clear()
            segment = cls._build_eager_segment(
                group_layout=group_layout,
                actuator=group,
                command=binding.command,
                control=control,
                selector_state=selector_state,
                joint_command=binding.joint_command,
                computed_effort=binding.computed_effort,
                applied_effort=binding.applied_effort,
            )
            eager_segments.append(segment)
            ordered_entries.append(segment)
        if epoch_layouts:
            ordered_entries.append(
                _StaticScatterEpoch(
                    group_names=tuple(group.name for group in epoch_layouts),
                    owner_slots_by_field=cls._epoch_owner_rows(
                        group_layouts=tuple(epoch_layouts),
                        type_offsets=type_offsets,
                        num_joints=binding.layout.num_joints,
                        device=control.device,
                    ),
                )
            )
        return (
            tuple(eager_segments),
            tuple(entry for entry in ordered_entries if isinstance(entry, _StaticScatterEpoch)),
            tuple(ordered_entries),
        )

    @classmethod
    def _build_eager_segment(
        cls,
        group_layout: Any,
        actuator: ActuatorBase,
        command: Any,
        control: ActuatorControl,
        selector_state: _SelectorState,
        joint_command: Any,
        computed_effort: ProxyArray,
        applied_effort: ProxyArray,
    ) -> _EagerSegment:
        device = control.device
        joint_indices_torch = selector_state._group_joint_ids[group_layout.name]
        joint_indices = selector_state._group_joint_ids_wp[group_layout.name]
        staging = MappingProxyType(
            {
                "position": ProxyArray(
                    wp.empty((group_layout.num_worlds, group_layout.num_dofs), dtype=wp.float32, device=device)
                ),
                "velocity": ProxyArray(
                    wp.empty((group_layout.num_worlds, group_layout.num_dofs), dtype=wp.float32, device=device)
                ),
                "effort": ProxyArray(
                    wp.empty((group_layout.num_worlds, group_layout.num_dofs), dtype=wp.float32, device=device)
                ),
                "joint_position": ProxyArray(
                    wp.empty((group_layout.num_worlds, group_layout.num_dofs), dtype=wp.float32, device=device)
                ),
                "joint_velocity": ProxyArray(
                    wp.empty((group_layout.num_worlds, group_layout.num_dofs), dtype=wp.float32, device=device)
                ),
            }
        )
        return _EagerSegment(
            group_name=group_layout.name,
            group_names=(group_layout.name,),
            actuator=actuator,
            joint_indices=joint_indices,
            staging=staging,
            action=ArticulationActions(
                joint_positions=staging["position"].torch,
                joint_velocities=staging["velocity"].torch,
                joint_efforts=staging["effort"].torch,
                joint_indices=joint_indices_torch,
            ),
            gather_key=("eager_gather", group_layout.name),
            gather_dim=(group_layout.num_worlds, group_layout.num_dofs),
            scatter_dim=(group_layout.num_worlds, group_layout.num_dofs),
            gather_inputs=(
                command.position.warp,
                command.velocity.warp,
                command.effort.warp,
                control.joint_pos.warp,
                control.joint_vel.warp,
                joint_indices,
            ),
            gather_outputs=(
                staging["position"].warp,
                staging["velocity"].warp,
                staging["effort"].warp,
                staging["joint_position"].warp,
                staging["joint_velocity"].warp,
            ),
            action_scatter_outputs=(
                joint_command.position.warp,
                joint_command.velocity.warp,
                joint_command.effort.warp,
            ),
            telemetry_scatter_outputs=(computed_effort.warp, applied_effort.warp),
        )

    @staticmethod
    def _epoch_owner_rows(
        *,
        group_layouts: tuple[Any, ...],
        type_offsets: Mapping[type[ActuatorBase], int],
        num_joints: int,
        device: str,
    ) -> Mapping[str, wp.array(dtype=wp.int32)]:
        owners = {field: [-1] * num_joints for field in _FIELD_NAMES}
        for group_layout in group_layouts:
            field_names = (
                _FIELD_NAMES
                if group_layout.actuator_type is ImplicitActuator
                else ("effort", "computed_effort", "applied_effort")
            )
            type_offset = type_offsets[group_layout.actuator_type]
            for local_slot, joint_id in enumerate(group_layout.joint_indices):
                slot = type_offset + group_layout.type_slice.start + local_slot
                for field in field_names:
                    owners[field][joint_id] = slot
        return MappingProxyType(
            {field: wp.array(slots, dtype=wp.int32, device=device) for field, slots in owners.items()}
        )

    @classmethod
    def _bind_static_launches(
        cls,
        *,
        epochs: tuple[_StaticScatterEpoch, ...],
        schedule: tuple[_StaticScatterEpoch | _EagerSegment, ...],
        source_buffers: Mapping[type[ActuatorBase], Mapping[str, ProxyArray]],
        output_buffers: Mapping[type[ActuatorBase], Mapping[str, ProxyArray]],
        targets: Mapping[str, ProxyArray],
        num_worlds: int,
        num_joints: int,
        device: str,
        implicit_count: int,
        ideal_count: int,
    ) -> tuple[
        tuple[_StaticScatterEpoch, ...],
        tuple[_StaticScatterEpoch | _EagerSegment, ...],
    ]:
        """Bind immutable static scatter arguments after every stable alias exists."""
        if not epochs:
            return epochs, schedule

        needs_sentinel = any(
            not buffers
            for buffers in (
                source_buffers.get(ImplicitActuator),
                output_buffers.get(ImplicitActuator),
                output_buffers.get(IdealPDActuator),
                output_buffers.get(DCMotor),
            )
        )
        sentinel = ProxyArray(wp.zeros((1, 1), dtype=wp.float32, device=device)) if needs_sentinel else None

        def _source(actuator_type: type[ActuatorBase], name: str) -> ProxyArray:
            source = source_buffers.get(actuator_type, {}).get(name)
            if source is not None:
                return source
            assert sentinel is not None
            return sentinel

        def _output(actuator_type: type[ActuatorBase], name: str) -> ProxyArray:
            output = output_buffers.get(actuator_type, {}).get(name)
            if output is not None:
                return output
            assert sentinel is not None
            return sentinel

        updated_epochs = tuple(
            replace(
                epoch,
                scatter_key=("scatter", epoch.group_names),
                scatter_dim=(num_worlds, num_joints),
                scatter_inputs=(
                    _source(ImplicitActuator, "position").warp,
                    _source(ImplicitActuator, "velocity").warp,
                    _source(ImplicitActuator, "effort").warp,
                    _output(ImplicitActuator, "computed_effort").warp,
                    _output(ImplicitActuator, "applied_effort").warp,
                    _output(IdealPDActuator, "applied_effort").warp,
                    _output(IdealPDActuator, "computed_effort").warp,
                    _output(IdealPDActuator, "applied_effort").warp,
                    _output(DCMotor, "applied_effort").warp,
                    _output(DCMotor, "computed_effort").warp,
                    _output(DCMotor, "applied_effort").warp,
                    epoch.owner_slots_by_field["position"],
                    epoch.owner_slots_by_field["velocity"],
                    epoch.owner_slots_by_field["effort"],
                    epoch.owner_slots_by_field["computed_effort"],
                    epoch.owner_slots_by_field["applied_effort"],
                    implicit_count,
                    ideal_count,
                ),
                scatter_outputs=(
                    targets["position"].warp,
                    targets["velocity"].warp,
                    targets["effort"].warp,
                    targets["computed_effort"].warp,
                    targets["applied_effort"].warp,
                ),
            )
            for epoch in epochs
        )
        by_group_names = {epoch.group_names: epoch for epoch in updated_epochs}
        updated_schedule = tuple(
            by_group_names[entry.group_names] if isinstance(entry, _StaticScatterEpoch) else entry for entry in schedule
        )
        return updated_epochs, updated_schedule

    def set_runtime_hooks(
        self,
        *,
        validate_execution: Callable[[], None],
        native_compute: Callable[[float], None] | None,
    ) -> None:
        """Install publication-time guards without reading the facade during build."""
        self._validate_execution = validate_execution
        self._native_compute = native_compute

    def compute(self, dt: float = 0.0) -> None:
        """Execute fixed stateless ranges and ordered eager scatter epochs."""
        if not self._valid:
            raise RuntimeError("stale actuator execution plan")
        if self._validate_execution is not None:
            self._validate_execution()
        for execution_range in self.stateless_ranges:
            self._run_range(execution_range)
        for entry in self._schedule:
            if isinstance(entry, _StaticScatterEpoch):
                self._scatter_static_epoch(entry)
            else:
                self._run_eager(entry)
        if self._native_compute is not None:
            self._native_compute(dt)

    def _run_range(self, execution_range: _ExecutionRange) -> None:
        self._launch_cache.launch(
            execution_range.gather_key,
            actuator_kernels.gather_actuator_batch,
            dim=execution_range.gather_dim,
            inputs=execution_range.gather_inputs,
            outputs=execution_range.gather_outputs,
        )
        if execution_range.actuator_type is ImplicitActuator:
            if (
                execution_range.implicit_key is None
                or execution_range.implicit_dim is None
                or execution_range.implicit_inputs is None
                or execution_range.implicit_outputs is None
            ):
                raise RuntimeError("Implicit execution range is missing fixed launch arguments.")
            self._launch_cache.launch(
                execution_range.implicit_key,
                actuator_kernels.compute_implicit_actuator_range,
                dim=execution_range.implicit_dim,
                inputs=execution_range.implicit_inputs,
                outputs=execution_range.implicit_outputs,
            )
            return
        action = execution_range.action
        if action is None:
            raise RuntimeError("Explicit actuator execution range is missing its fixed action staging.")
        staging = execution_range.staging
        action.joint_positions = staging["position"].torch
        action.joint_velocities = staging["velocity"].torch
        action.joint_efforts = staging["effort"].torch
        try:
            execution_range.executor.compute(action, staging["joint_position"].torch, staging["joint_velocity"].torch)
        finally:
            action.joint_positions = staging["position"].torch
            action.joint_velocities = staging["velocity"].torch
            action.joint_efforts = staging["effort"].torch

    def _scatter_static_epoch(self, epoch: _StaticScatterEpoch) -> None:
        if (
            epoch.scatter_key is None
            or epoch.scatter_dim is None
            or epoch.scatter_inputs is None
            or epoch.scatter_outputs is None
        ):
            raise RuntimeError("Static scatter epoch is missing fixed launch arguments.")
        self._launch_cache.launch(
            epoch.scatter_key,
            actuator_kernels.scatter_execution_plan,
            dim=epoch.scatter_dim,
            inputs=epoch.scatter_inputs,
            outputs=epoch.scatter_outputs,
        )

    def _run_eager(self, segment: _EagerSegment) -> None:
        self._launch_cache.launch(
            segment.gather_key,
            actuator_kernels.gather_actuator_batch,
            dim=segment.gather_dim,
            inputs=segment.gather_inputs,
            outputs=segment.gather_outputs,
        )
        action = segment.action
        action.joint_positions = segment.staging["position"].torch
        action.joint_velocities = segment.staging["velocity"].torch
        action.joint_efforts = segment.staging["effort"].torch
        output = segment.actuator.compute(
            action,
            segment.staging["joint_position"].torch,
            segment.staging["joint_velocity"].torch,
        )
        control = self._control
        if control is None:
            raise RuntimeError("stale actuator execution plan")
        wp.launch(
            actuator_kernels.scatter_processed_targets,
            dim=segment.scatter_dim,
            inputs=[output.joint_positions, output.joint_velocities, output.joint_efforts, segment.joint_indices],
            outputs=segment.action_scatter_outputs,
            device=control.device,
        )
        wp.launch(
            actuator_kernels.scatter_eager_effort_telemetry,
            dim=segment.scatter_dim,
            inputs=[segment.actuator.computed_effort, segment.actuator.applied_effort, segment.joint_indices],
            outputs=segment.telemetry_scatter_outputs,
            device=control.device,
        )

    def reset(self, env_ids: Sequence[int] | slice) -> None:
        """Reset ordinary group state and the backend-native actuator state."""
        if not self._valid:
            raise RuntimeError("stale actuator execution plan")
        if self._validate_execution is not None:
            self._validate_execution()
        for execution_range in self.stateless_ranges:
            execution_range.executor.reset(env_ids)
        for segment in self.eager_segments:
            segment.actuator.reset(env_ids)
        control = self._control
        if control is None:
            raise RuntimeError("stale actuator execution plan")
        reset_native = getattr(control, "reset_native_actuators", None)
        if reset_native is not None:
            reset_native(env_ids)

    def invalidate(self) -> None:
        """Release cached launches and candidate-owned execution aliases."""
        if not self._valid:
            return
        self._valid = False
        self._launch_cache.clear()
        self._validate_execution = None
        self._native_compute = None
        self.stateless_ranges = ()
        self.eager_segments = ()
        self.static_scatter_epochs = ()
        self._schedule = ()
        self._control = None
