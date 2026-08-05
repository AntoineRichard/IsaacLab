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

import torch
import warp as wp

from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache

from . import actuator_kernels
from .actuator_base import ActuatorBase
from .actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from .actuator_storage import _GroupBinding

if TYPE_CHECKING:
    from .actuator_collection import _ArticulationBinding
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
    owner_slots_by_field: Mapping[str, wp.array(dtype=wp.int32)]
    graphable: bool
    staging: Mapping[str, ProxyArray]
    executor: ActuatorBase
    action: ArticulationActions | None
    gather_inputs: tuple[object, ...]
    gather_outputs: tuple[object, ...]


@dataclass(frozen=True)
class _EagerSegment:
    """One non-aggregate actuator group preserving ordinary compute behavior."""

    group_name: str
    group_names: tuple[str, ...]
    actuator: ActuatorBase
    joint_indices: wp.array(dtype=wp.int32)
    staging: Mapping[str, ProxyArray]
    action: ArticulationActions
    gather_inputs: tuple[object, ...]
    gather_outputs: tuple[object, ...]


@dataclass(frozen=True)
class _StaticScatterEpoch:
    """Configuration-order static writers between eager execution barriers."""

    group_names: tuple[str, ...]
    owner_slots_by_field: Mapping[str, wp.array(dtype=wp.int32)]


class _ArticulationExecutionPlan:
    """Preallocated execution plan for one private articulation binding."""

    def __init__(
        self,
        *,
        control: ActuatorControl,
        generation: int,
        num_worlds: int,
        num_joints: int,
        stateless_ranges: tuple[_ExecutionRange, ...],
        eager_segments: tuple[_EagerSegment, ...],
        static_scatter_epochs: tuple[_StaticScatterEpoch, ...],
        schedule: tuple[_StaticScatterEpoch | _EagerSegment, ...],
        source_buffers: Mapping[type[ActuatorBase], Mapping[str, ProxyArray]],
        output_buffers: Mapping[type[ActuatorBase], Mapping[str, ProxyArray]],
        targets: Mapping[str, ProxyArray],
        implicit_count: int,
        ideal_count: int,
    ) -> None:
        self._control = control
        self._generation = generation
        self._num_worlds = num_worlds
        self._num_joints = num_joints
        self.stateless_ranges = stateless_ranges
        self.eager_segments = eager_segments
        self.static_scatter_epochs = static_scatter_epochs
        self._schedule = schedule
        self._source_buffers = source_buffers
        self._output_buffers = output_buffers
        self._targets = targets
        self._implicit_count = implicit_count
        self._ideal_count = ideal_count
        self._launch_cache = _WarpLaunchCache(control.device)
        self._valid = True
        self._validate_execution: Callable[[], None] | None = None
        self._native_compute: Callable[[float], bool] | None = None

    @classmethod
    def build(
        cls,
        *,
        binding: _ArticulationBinding,
        control: ActuatorControl,
        generation: int,
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

        # Task 10 retains the backend's established whole-articulation native
        # bypass. The callback installed after publication skips every range.
        native_bypass = bool(binding.native_group_names)
        type_offsets: dict[type[ActuatorBase], int] = {}
        offset = 0
        for actuator_type in _SUPPORTED_TYPES:
            type_layout = layout.type_layouts.get(actuator_type)
            type_offsets[actuator_type] = offset
            if type_layout is not None and not native_bypass:
                offset += type_layout.num_dofs

        source_buffers: dict[type[ActuatorBase], Mapping[str, ProxyArray]] = {}
        output_buffers: dict[type[ActuatorBase], Mapping[str, ProxyArray]] = {}
        ranges_by_type: dict[type[ActuatorBase], _ExecutionRange] = {}
        ranges: list[_ExecutionRange] = []
        for actuator_type, type_layout in layout.type_layouts.items():
            if actuator_type not in _SUPPORTED_TYPES or native_bypass:
                continue
            type_groups = tuple(group for group in layout.group_layouts if group.actuator_type is actuator_type)
            if not type_groups:
                continue
            execution_range = cls._build_range(
                actuator_type=actuator_type,
                type_layout=type_layout,
                group_layouts=type_groups,
                groups=groups,
                command=binding.command,
                control=control,
                type_offset=type_offsets[actuator_type],
                num_joints=layout.num_joints,
            )
            ranges.append(execution_range)
            ranges_by_type[actuator_type] = execution_range
            source_buffers[actuator_type] = execution_range.staging
            group_binding = execution_range.executor.__dict__["_parameter_binding"]
            output_buffers[actuator_type] = MappingProxyType(
                {
                    "computed_effort": group_binding.arrays["computed_effort"],
                    "applied_effort": group_binding.arrays["applied_effort"],
                }
            )

        source_buffers, output_buffers = cls._add_missing_type_buffers(
            source_buffers=source_buffers,
            output_buffers=output_buffers,
            num_worlds=layout.num_worlds,
            device=control.device,
        )
        if native_bypass:
            eager_segments, epochs, schedule = (), (), ()
        else:
            eager_segments, epochs, schedule = cls._build_ordered_segments(
                binding=binding,
                groups=groups,
                ranges_by_type=ranges_by_type,
                type_offsets=type_offsets,
                control=control,
                native_bypass=False,
            )
        implicit_count = 0 if native_bypass else layout.type_layouts.get(ImplicitActuator, _EmptyTypeLayout()).num_dofs
        ideal_count = 0 if native_bypass else layout.type_layouts.get(IdealPDActuator, _EmptyTypeLayout()).num_dofs
        return cls(
            control=control,
            generation=generation,
            num_worlds=layout.num_worlds,
            num_joints=layout.num_joints,
            stateless_ranges=tuple(ranges),
            eager_segments=eager_segments,
            static_scatter_epochs=epochs,
            schedule=schedule,
            source_buffers=MappingProxyType(source_buffers),
            output_buffers=MappingProxyType(output_buffers),
            targets=targets,
            implicit_count=implicit_count,
            ideal_count=ideal_count,
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
        type_offset: int,
        num_joints: int,
    ) -> _ExecutionRange:
        device = control.device
        joint_indices_torch = torch.tensor(type_layout.compact_joint_indices, dtype=torch.int32, device=device)
        joint_indices = wp.from_torch(joint_indices_torch, dtype=wp.int32)
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

        owner_slots = cls._owner_rows(
            group_layouts=group_layouts,
            type_offset=type_offset,
            num_joints=num_joints,
            device=device,
        )
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
        return _ExecutionRange(
            actuator_type=actuator_type,
            group_names=tuple(group.name for group in group_layouts),
            group_slices=tuple(group.type_slice for group in group_layouts),
            joint_indices=joint_indices,
            owner_slots_by_field=owner_slots,
            graphable=actuator_type._parameter_schema().graphable,
            staging=staging,
            executor=executor,
            action=action,
            gather_inputs=gather_inputs,
            gather_outputs=gather_outputs,
        )

    @classmethod
    def _add_missing_type_buffers(
        cls,
        *,
        source_buffers: Mapping[type[ActuatorBase], Mapping[str, ProxyArray]],
        output_buffers: Mapping[type[ActuatorBase], Mapping[str, ProxyArray]],
        num_worlds: int,
        device: str,
    ) -> tuple[dict[type[ActuatorBase], Mapping[str, ProxyArray]], dict[type[ActuatorBase], Mapping[str, ProxyArray]]]:
        sources = dict(source_buffers)
        outputs = dict(output_buffers)
        for actuator_type in _SUPPORTED_TYPES:
            if actuator_type not in sources:
                sources[actuator_type] = MappingProxyType(
                    {
                        name: ProxyArray(wp.zeros((num_worlds, 1), dtype=wp.float32, device=device))
                        for name in ("position", "velocity", "effort")
                    }
                )
            if actuator_type not in outputs:
                outputs[actuator_type] = MappingProxyType(
                    {
                        name: ProxyArray(wp.zeros((num_worlds, 1), dtype=wp.float32, device=device))
                        for name in ("computed_effort", "applied_effort")
                    }
                )
        return sources, outputs

    @classmethod
    def _build_ordered_segments(
        cls,
        *,
        binding: _ArticulationBinding,
        groups: Mapping[str, ActuatorBase],
        ranges_by_type: Mapping[type[ActuatorBase], _ExecutionRange],
        type_offsets: Mapping[type[ActuatorBase], int],
        control: ActuatorControl,
        native_bypass: bool,
    ) -> tuple[
        tuple[_EagerSegment, ...],
        tuple[_StaticScatterEpoch, ...],
        tuple[_StaticScatterEpoch | _EagerSegment, ...],
    ]:
        eager_segments: list[_EagerSegment] = []
        ordered_entries: list[_StaticScatterEpoch | _EagerSegment] = []
        epoch_layouts: list[Any] = []
        for group_layout in binding.layout.group_layouts:
            exact_stateless = type(group := groups[group_layout.name]) in _SUPPORTED_TYPES
            if exact_stateless and not native_bypass:
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
            if group_layout.name in binding.native_group_names:
                continue
            segment = cls._build_eager_segment(group_layout, group, binding.command, control)
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
        cls, group_layout: Any, actuator: ActuatorBase, command: Any, control: ActuatorControl
    ) -> _EagerSegment:
        device = control.device
        joint_indices_torch = torch.tensor(group_layout.joint_indices, dtype=torch.int32, device=device)
        joint_indices = wp.from_torch(joint_indices_torch, dtype=wp.int32)
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
        )

    @classmethod
    def _owner_rows(
        cls,
        *,
        group_layouts: tuple[Any, ...],
        type_offset: int,
        num_joints: int,
        device: str,
    ) -> Mapping[str, wp.array(dtype=wp.int32)]:
        return cls._epoch_owner_rows(
            group_layouts=group_layouts,
            type_offsets={group_layouts[0].actuator_type: type_offset},
            num_joints=num_joints,
            device=device,
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

    def set_runtime_hooks(
        self,
        *,
        validate_execution: Callable[[], None],
        native_compute: Callable[[float], bool],
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
        if self._native_compute is not None and self._native_compute(dt):
            return
        for execution_range in self.stateless_ranges:
            self._run_range(execution_range)
        for entry in self._schedule:
            if isinstance(entry, _StaticScatterEpoch):
                self._scatter_static_epoch(entry)
            else:
                self._run_eager(entry)

    def _run_range(self, execution_range: _ExecutionRange) -> None:
        self._launch_cache.launch(
            ("gather", execution_range.actuator_type),
            actuator_kernels.gather_actuator_batch,
            dim=(self._num_worlds, execution_range.joint_indices.shape[0]),
            inputs=execution_range.gather_inputs,
            outputs=execution_range.gather_outputs,
        )
        staging = execution_range.staging
        if execution_range.actuator_type is ImplicitActuator:
            binding = execution_range.executor.__dict__["_parameter_binding"]
            self._launch_cache.launch(
                ("implicit", execution_range.actuator_type),
                actuator_kernels.compute_implicit_actuator_range,
                dim=(self._num_worlds, execution_range.joint_indices.shape[0]),
                inputs=[
                    staging["position"].warp,
                    staging["velocity"].warp,
                    staging["effort"].warp,
                    staging["joint_position"].warp,
                    staging["joint_velocity"].warp,
                    binding.arrays["stiffness"].warp,
                    binding.arrays["damping"].warp,
                    binding.arrays["effort_limit"].warp,
                ],
                outputs=[binding.arrays["computed_effort"].warp, binding.arrays["applied_effort"].warp],
            )
            return
        action = execution_range.action
        if action is None:
            raise RuntimeError("Explicit actuator execution range is missing its fixed action staging.")
        action.joint_positions = staging["position"].torch
        action.joint_velocities = staging["velocity"].torch
        action.joint_efforts = staging["effort"].torch
        execution_range.executor.compute(action, staging["joint_position"].torch, staging["joint_velocity"].torch)

    def _scatter_static_epoch(self, epoch: _StaticScatterEpoch) -> None:
        sources = self._source_buffers
        outputs = self._output_buffers
        implicit = sources[ImplicitActuator]
        ideal = outputs[IdealPDActuator]
        dc = outputs[DCMotor]
        self._launch_cache.launch(
            ("scatter", epoch.group_names),
            actuator_kernels.scatter_execution_plan,
            dim=(self._num_worlds, self._num_joints),
            inputs=[
                implicit["position"].warp,
                implicit["velocity"].warp,
                implicit["effort"].warp,
                outputs[ImplicitActuator]["computed_effort"].warp,
                outputs[ImplicitActuator]["applied_effort"].warp,
                ideal["applied_effort"].warp,
                ideal["computed_effort"].warp,
                ideal["applied_effort"].warp,
                dc["applied_effort"].warp,
                dc["computed_effort"].warp,
                dc["applied_effort"].warp,
                epoch.owner_slots_by_field["position"],
                epoch.owner_slots_by_field["velocity"],
                epoch.owner_slots_by_field["effort"],
                epoch.owner_slots_by_field["computed_effort"],
                epoch.owner_slots_by_field["applied_effort"],
                self._implicit_count,
                self._ideal_count,
            ],
            outputs=[
                self._targets["position"].warp,
                self._targets["velocity"].warp,
                self._targets["effort"].warp,
                self._targets["computed_effort"].warp,
                self._targets["applied_effort"].warp,
            ],
        )

    def _run_eager(self, segment: _EagerSegment) -> None:
        self._launch_cache.launch(
            ("eager_gather", segment.group_name),
            actuator_kernels.gather_actuator_batch,
            dim=(self._num_worlds, segment.joint_indices.shape[0]),
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
        wp.launch(
            actuator_kernels.scatter_processed_targets,
            dim=(self._num_worlds, segment.joint_indices.shape[0]),
            inputs=[output.joint_positions, output.joint_velocities, output.joint_efforts, segment.joint_indices],
            outputs=[self._targets["position"].warp, self._targets["velocity"].warp, self._targets["effort"].warp],
            device=self._control.device,
        )
        wp.launch(
            actuator_kernels.scatter_eager_effort_telemetry,
            dim=(self._num_worlds, segment.joint_indices.shape[0]),
            inputs=[segment.actuator.computed_effort, segment.actuator.applied_effort, segment.joint_indices],
            outputs=[self._targets["computed_effort"].warp, self._targets["applied_effort"].warp],
            device=self._control.device,
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
        reset_native = getattr(self._control, "reset_native_actuators", None)
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
        self._source_buffers = {}
        self._output_buffers = {}
        self._targets = {}


@dataclass(frozen=True)
class _EmptyTypeLayout:
    """Zero-width fallback for missing exact-type layouts."""

    num_dofs: int = 0
