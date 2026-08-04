# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private typed parameter storage used by managed actuator groups."""

from __future__ import annotations

import operator
from collections.abc import Callable, ItemsView, Iterator, KeysView, Mapping, MutableMapping, Sequence, ValuesView
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan

    from .actuator_base import ActuatorBase


class _GuardedIterator(Iterator[Any]):
    """Iterator that revalidates its owning generation on every operation."""

    def __init__(self, guard: Callable[[], None], iterator: Iterator[Any]) -> None:
        self._guard = guard
        self._iterator = iterator

    def __iter__(self) -> _GuardedIterator:
        self._guard()
        return self

    def __next__(self) -> Any:
        self._guard()
        return next(self._iterator)

    def __length_hint__(self) -> int:
        self._guard()
        return operator.length_hint(self._iterator)


class _GuardedSetOperations:
    """Set operations that validate a guarded mapping view at entry."""

    _guard: Callable[[], None]

    def _call_set_operation(self, operation: str, other: Any) -> Any:
        self._guard()
        return getattr(super(), operation)(other)

    def __le__(self, other: Any) -> bool:
        return self._call_set_operation("__le__", other)

    def __lt__(self, other: Any) -> bool:
        return self._call_set_operation("__lt__", other)

    def __gt__(self, other: Any) -> bool:
        return self._call_set_operation("__gt__", other)

    def __ge__(self, other: Any) -> bool:
        return self._call_set_operation("__ge__", other)

    def __eq__(self, other: object) -> bool:
        return self._call_set_operation("__eq__", other)

    def __ne__(self, other: object) -> bool:
        self._guard()
        result = super().__eq__(other)
        return result if result is NotImplemented else not result

    def __and__(self, other: Any) -> set[Any]:
        return self._call_set_operation("__and__", other)

    def __rand__(self, other: Any) -> set[Any]:
        return self._call_set_operation("__rand__", other)

    def isdisjoint(self, other: Any) -> bool:
        return self._call_set_operation("isdisjoint", other)

    def __or__(self, other: Any) -> set[Any]:
        return self._call_set_operation("__or__", other)

    def __ror__(self, other: Any) -> set[Any]:
        return self._call_set_operation("__ror__", other)

    def __sub__(self, other: Any) -> set[Any]:
        return self._call_set_operation("__sub__", other)

    def __rsub__(self, other: Any) -> set[Any]:
        return self._call_set_operation("__rsub__", other)

    def __xor__(self, other: Any) -> set[Any]:
        return self._call_set_operation("__xor__", other)

    def __rxor__(self, other: Any) -> set[Any]:
        return self._call_set_operation("__rxor__", other)


class _GuardedKeysView(_GuardedSetOperations, KeysView):
    """Live keys view that revalidates its owning generation when consumed."""

    def __init__(self, mapping: Mapping[Any, Any], guard: Callable[[], None]) -> None:
        super().__init__(mapping)
        self._guard = guard

    def __iter__(self) -> Iterator[Any]:
        self._guard()
        return _GuardedIterator(self._guard, super().__iter__())

    def __reversed__(self) -> Iterator[Any]:
        self._guard()
        return _GuardedIterator(self._guard, reversed(self._mapping))

    def __len__(self) -> int:
        self._guard()
        return super().__len__()

    def __contains__(self, key: object) -> bool:
        self._guard()
        return super().__contains__(key)

    @property
    def mapping(self) -> Mapping[Any, Any]:
        """Read-only guarded access to the underlying live mapping."""
        self._guard()
        return MappingProxyType(self._mapping)

    def __repr__(self) -> str:
        self._guard()
        return repr(dict(self._mapping).keys())


class _GuardedItemsView(_GuardedSetOperations, ItemsView):
    """Live items view that revalidates its owning generation when consumed."""

    def __init__(self, mapping: Mapping[Any, Any], guard: Callable[[], None]) -> None:
        super().__init__(mapping)
        self._guard = guard

    def __iter__(self) -> Iterator[Any]:
        self._guard()
        return _GuardedIterator(self._guard, super().__iter__())

    def __reversed__(self) -> Iterator[Any]:
        self._guard()
        iterator = ((key, self._mapping[key]) for key in reversed(self._mapping))
        return _GuardedIterator(self._guard, iterator)

    def __len__(self) -> int:
        self._guard()
        return super().__len__()

    def __contains__(self, item: object) -> bool:
        self._guard()
        return super().__contains__(item)

    @property
    def mapping(self) -> Mapping[Any, Any]:
        """Read-only guarded access to the underlying live mapping."""
        self._guard()
        return MappingProxyType(self._mapping)

    def __repr__(self) -> str:
        self._guard()
        return repr(dict(self._mapping).items())


class _GuardedValuesView(ValuesView):
    """Live values view that revalidates its owning generation when consumed."""

    def __init__(self, mapping: Mapping[Any, Any], guard: Callable[[], None]) -> None:
        super().__init__(mapping)
        self._guard = guard

    def __iter__(self) -> Iterator[Any]:
        self._guard()
        return _GuardedIterator(self._guard, super().__iter__())

    def __reversed__(self) -> Iterator[Any]:
        self._guard()
        iterator = (self._mapping[key] for key in reversed(self._mapping))
        return _GuardedIterator(self._guard, iterator)

    def __len__(self) -> int:
        self._guard()
        return super().__len__()

    def __contains__(self, value: object) -> bool:
        self._guard()
        return super().__contains__(value)

    @property
    def mapping(self) -> Mapping[Any, Any]:
        """Read-only guarded access to the underlying live mapping."""
        self._guard()
        return MappingProxyType(self._mapping)

    def __repr__(self) -> str:
        self._guard()
        return repr(dict(self._mapping).values())


@dataclass(frozen=True)
class _FieldSpec:
    """Description of one typed actuator-storage field."""

    name: str
    dtype: type
    unit: str
    role: Literal["parameter", "output", "scratch", "state"]
    fill: float
    backend_side_effect: str | None


@dataclass(frozen=True)
class _ActuatorSchema:
    """Typed storage contract declared by one exact built-in actuator class."""

    fields: tuple[_FieldSpec, ...]
    graphable: bool
    stateful: bool

    @property
    def parameter_names(self) -> frozenset[str]:
        """Names of fields used as actuator-model parameters."""
        return frozenset(field.name for field in self.fields if field.role == "parameter")


@dataclass(frozen=True)
class _GroupBinding:
    """Canonical typed-array binding for one logical actuator group."""

    generation: int
    joint_indices: torch.Tensor
    joint_names: tuple[str, ...]
    type_slice: slice
    arrays: Mapping[str, ProxyArray]
    parameter_proxies: Mapping[str, ProxyArray] | None = None


@dataclass(frozen=True)
class _GroupRegistration:
    """Resolved numeric metadata for one logical group on one source prototype."""

    name: str
    actuator_type: type[ActuatorBase]
    joint_indices: tuple[int, ...]
    values: Mapping[str, tuple[float, ...]]
    joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PrototypeRegistration:
    """Resolved actuator metadata for one clone-plan source prototype."""

    registration_key: object
    num_joints: int
    groups: tuple[_GroupRegistration, ...]


@dataclass(frozen=True)
class _GroupLayout:
    """Immutable compact-column layout for one logical actuator group."""

    name: str
    actuator_type: type[ActuatorBase]
    global_slice: slice
    articulation_offset: int
    num_worlds: int
    num_type_dofs: int
    type_slice: slice
    joint_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    prototype_values: Mapping[str, tuple[tuple[float, ...], ...]]
    prototype_assignment: torch.Tensor

    @property
    def num_dofs(self) -> int:
        """Number of compact DOFs in the logical group."""
        return len(self.joint_indices)


@dataclass(frozen=True)
class _TypeLayout:
    """Immutable contiguous layout for one exact actuator type in an articulation."""

    actuator_type: type[ActuatorBase]
    global_slice: slice
    articulation_offset: int
    num_worlds: int
    num_dofs: int
    compact_joint_indices: tuple[int, ...]
    articulation_to_compact_offsets: tuple[int, ...]
    articulation_to_compact_slots: tuple[int, ...]


@dataclass(frozen=True)
class _ArticulationLayout:
    """Immutable clone-aware typed-storage layout for one articulation."""

    registration_key: object
    num_worlds: int
    num_joints: int
    prototype_rows: tuple[int, ...]
    group_layouts: tuple[_GroupLayout, ...]
    type_layouts: Mapping[type[ActuatorBase], _TypeLayout]


def _build_articulation_layout(
    *,
    replication_cfg_id: int,
    clone_plan: ClonePlan,
    registrations: Sequence[_PrototypeRegistration],
    type_offsets: MutableMapping[type[ActuatorBase], int] | None = None,
) -> _ArticulationLayout:
    """Build one articulation layout from source-prototype registrations.

    The clone assignment remains a device tensor. Python work is bounded by the
    number of source prototypes, logical groups, and articulation joints.
    """
    prototype_rows = clone_plan.cfg_rows[replication_cfg_id]
    if len(registrations) != len(prototype_rows):
        raise ValueError(f"Expected {len(prototype_rows)} prototype registrations, got {len(registrations)}.")
    if not registrations:
        raise ValueError("At least one source-prototype registration is required.")

    first = registrations[0]
    for registration in registrations[1:]:
        if registration.registration_key is not first.registration_key or registration.num_joints != first.num_joints:
            raise ValueError("Source prototypes disagree on articulation identity or joint count.")
        if len(registration.groups) != len(first.groups):
            raise ValueError("Source prototypes disagree on logical actuator group count.")
        for expected, actual in zip(first.groups, registration.groups):
            if (
                actual.name != expected.name
                or actual.actuator_type is not expected.actuator_type
                or actual.joint_indices != expected.joint_indices
                or actual.joint_names != expected.joint_names
            ):
                raise ValueError(f"Source prototypes disagree on actuator group {expected.name!r} topology.")

    num_worlds = int(clone_plan.clone_mask.shape[1])
    prototype_row_indices = torch.tensor(prototype_rows, dtype=torch.long, device=clone_plan.clone_mask.device)
    selected_mask = clone_plan.clone_mask[prototype_row_indices]
    prototype_assignment = torch.argmax(selected_mask.to(dtype=torch.int32), dim=0).to(dtype=torch.int32)

    groups_by_type: dict[type[ActuatorBase], list[tuple[int, _GroupRegistration]]] = {}
    for group_index, group in enumerate(first.groups):
        groups_by_type.setdefault(group.actuator_type, []).append((group_index, group))

    if type_offsets is None:
        type_offsets = {}
    type_layouts: dict[type[ActuatorBase], _TypeLayout] = {}
    group_layout_by_index: dict[int, _GroupLayout] = {}
    for actuator_type, indexed_groups in groups_by_type.items():
        if actuator_type.__dict__.get("_parameter_schema") is None:
            for group_index, group in indexed_groups:
                group_layout_by_index[group_index] = _GroupLayout(
                    name=group.name,
                    actuator_type=actuator_type,
                    global_slice=slice(0, 0),
                    articulation_offset=0,
                    num_worlds=num_worlds,
                    num_type_dofs=len(group.joint_indices),
                    type_slice=slice(0, len(group.joint_indices)),
                    joint_indices=group.joint_indices,
                    joint_names=group.joint_names,
                    prototype_values=MappingProxyType({}),
                    prototype_assignment=prototype_assignment,
                )
            continue
        parameter_names = actuator_type._parameter_schema().parameter_names
        for group_index, group in indexed_groups:
            unknown_fields = (
                set().union(*(registration.groups[group_index].values.keys() for registration in registrations))
                - parameter_names
            )
            if unknown_fields:
                names = ", ".join(repr(name) for name in sorted(unknown_fields))
                raise ValueError(
                    f"Unknown field(s) {names} for {actuator_type.__name__} actuator group {group.name!r}."
                )
        compact_joint_indices = tuple(joint_id for _, group in indexed_groups for joint_id in group.joint_indices)
        slots_by_joint: list[list[int]] = [[] for _ in range(first.num_joints)]
        for compact_slot, articulation_joint_id in enumerate(compact_joint_indices):
            if articulation_joint_id < 0 or articulation_joint_id >= first.num_joints:
                raise ValueError(
                    f"Actuator group joint index {articulation_joint_id} is outside [0, {first.num_joints})."
                )
            slots_by_joint[articulation_joint_id].append(compact_slot)
        csr_offsets = [0]
        csr_slots: list[int] = []
        for slots in slots_by_joint:
            csr_slots.extend(slots)
            csr_offsets.append(len(csr_slots))

        num_type_dofs = len(compact_joint_indices)
        articulation_offset = type_offsets.get(actuator_type, 0)
        global_slice = slice(articulation_offset, articulation_offset + num_worlds * num_type_dofs)
        type_offsets[actuator_type] = global_slice.stop
        type_layouts[actuator_type] = _TypeLayout(
            actuator_type=actuator_type,
            global_slice=global_slice,
            articulation_offset=articulation_offset,
            num_worlds=num_worlds,
            num_dofs=num_type_dofs,
            compact_joint_indices=compact_joint_indices,
            articulation_to_compact_offsets=tuple(csr_offsets),
            articulation_to_compact_slots=tuple(csr_slots),
        )

        group_offset = 0
        for group_index, group in indexed_groups:
            prototype_fields: dict[str, tuple[tuple[float, ...], ...]] = {}
            field_names = tuple(
                dict.fromkeys(
                    name for registration in registrations for name in registration.groups[group_index].values
                )
            )
            for field_name in field_names:
                rows = []
                for registration in registrations:
                    values = registration.groups[group_index].values.get(field_name)
                    if values is None:
                        raise ValueError(
                            f"Source prototype omits field {field_name!r} for actuator group {group.name!r}."
                        )
                    if len(values) != len(group.joint_indices):
                        raise ValueError(
                            f"Field {field_name!r} for actuator group {group.name!r} has {len(values)} values; "
                            f"expected {len(group.joint_indices)}."
                        )
                    rows.append(tuple(float(value) for value in values))
                prototype_fields[field_name] = tuple(rows)
            group_layout_by_index[group_index] = _GroupLayout(
                name=group.name,
                actuator_type=actuator_type,
                global_slice=global_slice,
                articulation_offset=articulation_offset,
                num_worlds=num_worlds,
                num_type_dofs=num_type_dofs,
                type_slice=slice(group_offset, group_offset + len(group.joint_indices)),
                joint_indices=group.joint_indices,
                joint_names=group.joint_names,
                prototype_values=MappingProxyType(prototype_fields),
                prototype_assignment=prototype_assignment,
            )
            group_offset += len(group.joint_indices)

    return _ArticulationLayout(
        registration_key=first.registration_key,
        num_worlds=num_worlds,
        num_joints=first.num_joints,
        prototype_rows=prototype_rows,
        group_layouts=tuple(group_layout_by_index[index] for index in range(len(first.groups))),
        type_layouts=MappingProxyType(type_layouts),
    )


@wp.kernel
def _expand_prototype_field(
    output: wp.array(dtype=wp.float32),
    prototype_values: wp.array(dtype=wp.float32),
    prototype_assignment: wp.array(dtype=wp.int32),
    output_offsets: wp.array(dtype=wp.int64),
    assignment_offsets: wp.array(dtype=wp.int64),
    prototype_offsets: wp.array(dtype=wp.int64),
    block_num_worlds: wp.array(dtype=wp.int32),
    block_num_dofs: wp.array(dtype=wp.int32),
):
    """Expand source-prototype values into articulation-major flat storage."""
    block, world, dof = wp.tid()
    num_worlds = block_num_worlds[block]
    num_dofs = block_num_dofs[block]
    if world < num_worlds and dof < num_dofs:
        prototype = prototype_assignment[assignment_offsets[block] + wp.int64(world)]
        output[output_offsets[block] + wp.int64(world * num_dofs + dof)] = prototype_values[
            prototype_offsets[block] + wp.int64(prototype * num_dofs + dof)
        ]


class _TypedStore:
    """Warp-owned flat field storage for one exact actuator type."""

    def __init__(self, actuator_type: type[ActuatorBase]) -> None:
        """Create an unallocated exact-type store."""
        if actuator_type.__dict__.get("_parameter_schema") is None:
            raise TypeError(f"{actuator_type.__name__} does not opt into managed parameter storage.")
        self.actuator_type = actuator_type
        self._fields: dict[str, ProxyArray] = {}
        self._type_proxies: dict[tuple[int, str], ProxyArray] = {}
        self._group_proxies: dict[tuple[int, str], ProxyArray] = {}
        self._mapping_proxies: dict[int, tuple[ProxyArray, ProxyArray]] = {}
        self._initialization_buffers: list[object] = []

    def allocate(self, layouts: Sequence[_ArticulationLayout], *, device: str) -> None:
        """Allocate and initialize one flat array per exact-type schema field."""
        type_layouts = [
            layout.type_layouts[self.actuator_type] for layout in layouts if self.actuator_type in layout.type_layouts
        ]
        expected_offset = 0
        for layout in type_layouts:
            if layout.global_slice.start != expected_offset:
                raise ValueError("Exact-type articulation ranges must be contiguous and in registration order.")
            expected_offset = layout.global_slice.stop

        schema = self.actuator_type._parameter_schema()
        for field in schema.fields:
            if field.dtype is not torch.Tensor:
                raise TypeError(f"Unsupported managed field dtype for {field.name!r}: {field.dtype!r}.")
            self._fields[field.name] = ProxyArray(wp.empty(expected_offset, dtype=wp.float32, device=device))

        for layout in type_layouts:
            self._mapping_proxies[id(layout)] = (
                ProxyArray(wp.array(layout.articulation_to_compact_offsets, dtype=wp.int32, device=device)),
                ProxyArray(wp.array(layout.articulation_to_compact_slots, dtype=wp.int32, device=device)),
            )

        matching_layouts = [layout for layout in layouts if self.actuator_type in layout.type_layouts]
        groups_by_layout = [
            tuple(group for group in layout.group_layouts if group.actuator_type is self.actuator_type)
            for layout in matching_layouts
        ]
        varying_fields = {
            field.name
            for field in schema.fields
            if any(field.name in group.prototype_values for groups in groups_by_layout for group in groups)
        }
        if varying_fields:
            assignments = torch.cat(
                [groups[0].prototype_assignment.to(device=device, dtype=torch.int32) for groups in groups_by_layout]
            ).contiguous()
            assignment_offsets = []
            assignment_offset = 0
            prototype_offsets = []
            prototype_offset = 0
            for layout in matching_layouts:
                assignment_offsets.append(assignment_offset)
                assignment_offset += layout.num_worlds
                prototype_offsets.append(prototype_offset)
                type_layout = layout.type_layouts[self.actuator_type]
                prototype_offset += len(layout.prototype_rows) * type_layout.num_dofs
            output_offsets = [layout.type_layouts[self.actuator_type].global_slice.start for layout in matching_layouts]
            block_num_worlds = [layout.num_worlds for layout in matching_layouts]
            block_num_dofs = [layout.type_layouts[self.actuator_type].num_dofs for layout in matching_layouts]
            assignment = wp.from_torch(assignments, dtype=wp.int32)
            offset_metadata = tuple(
                wp.array(values, dtype=wp.int64, device=device)
                for values in (output_offsets, assignment_offsets, prototype_offsets)
            )
            shape_metadata = tuple(
                wp.array(values, dtype=wp.int32, device=device) for values in (block_num_worlds, block_num_dofs)
            )
            metadata = (*offset_metadata, *shape_metadata)
            self._initialization_buffers.extend((assignments, assignment, *metadata))

        for field in schema.fields:
            if field.name not in varying_fields:
                self._fields[field.name].warp.fill_(field.fill)
                continue

            flat_prototype_values: list[float] = []
            for layout, groups in zip(matching_layouts, groups_by_layout):
                for prototype in range(len(layout.prototype_rows)):
                    for group in groups:
                        rows = group.prototype_values.get(field.name)
                        if rows is None:
                            flat_prototype_values.extend((field.fill,) * group.num_dofs)
                        else:
                            flat_prototype_values.extend(rows[prototype])

            source = wp.array(flat_prototype_values, dtype=wp.float32, device=device)
            self._initialization_buffers.append(source)
            wp.launch(
                _expand_prototype_field,
                dim=(len(matching_layouts), max(block_num_worlds), max(block_num_dofs)),
                inputs=[self._fields[field.name].warp, source, assignment, *metadata],
                device=device,
            )

    def _view_proxy(
        self,
        *,
        field: str,
        offset: int,
        shape: tuple[int, int],
        strides: tuple[int, int],
    ) -> ProxyArray:
        flat = self._fields[field]
        item_size = wp.types.type_size_in_bytes(wp.float32)
        capacity = ((shape[0] - 1) * strides[0] + shape[1] * strides[1]) * item_size
        warp_view = wp.array(
            ptr=flat.warp.ptr + offset * item_size,
            dtype=wp.float32,
            shape=shape,
            strides=(strides[0] * item_size, strides[1] * item_size),
            capacity=capacity,
            device=flat.warp.device,
            copy=False,
        )
        proxy = ProxyArray(warp_view)
        proxy._torch_cache = flat.torch.as_strided(shape, strides, storage_offset=offset)
        return proxy

    def type_proxy(self, layout: _TypeLayout, field: str) -> ProxyArray:
        """Return the cached contiguous articulation/type view for ``field``."""
        if layout.actuator_type is not self.actuator_type:
            raise TypeError("Type layout does not belong to this exact-type store.")
        key = (id(layout), field)
        proxy = self._type_proxies.get(key)
        if proxy is None:
            proxy = self._view_proxy(
                field=field,
                offset=layout.global_slice.start,
                shape=(layout.num_worlds, layout.num_dofs),
                strides=(layout.num_dofs, 1),
            )
            self._type_proxies[key] = proxy
        return proxy

    def group_proxy(self, layout: _GroupLayout, field: str) -> ProxyArray:
        """Return the cached zero-copy strided logical-group view for ``field``."""
        if layout.actuator_type is not self.actuator_type:
            raise TypeError("Group layout does not belong to this exact-type store.")
        key = (id(layout), field)
        proxy = self._group_proxies.get(key)
        if proxy is None:
            proxy = self._view_proxy(
                field=field,
                offset=layout.global_slice.start + layout.type_slice.start,
                shape=(layout.num_worlds, layout.num_dofs),
                strides=(layout.num_type_dofs, 1),
            )
            self._group_proxies[key] = proxy
        return proxy

    def mapping_proxies(self, layout: _TypeLayout) -> tuple[ProxyArray, ProxyArray]:
        """Return immutable device copies of the articulation-to-compact CSR tables."""
        return self._mapping_proxies[id(layout)]

    def __getattr__(self, name: str) -> ProxyArray:
        """Expose exact-type flat fields by schema name."""
        try:
            return self._fields[name]
        except KeyError as error:
            raise AttributeError(name) from error


# Ownership rules:
# typed actuator parameters: stiffness, damping, actuator effort/velocity limits, saturation_effort
# typed outputs: computed_effort, applied_effort
# solver compatibility only: effort_limit_sim, velocity_limit_sim, armature, all friction fields
# structural/state: delay/history/recurrent buffers, network metadata, lookup tables
# legacy fill only when no type declares it: gear_ratio = 1.0
_PD_PARAMETERS = (
    _FieldSpec("stiffness", torch.Tensor, "[N/m or N·m/rad, depending on joint type]", "parameter", 0.0, None),
    _FieldSpec("damping", torch.Tensor, "[N·s/m or N·m·s/rad, depending on joint type]", "parameter", 0.0, None),
    _FieldSpec("effort_limit", torch.Tensor, "[N or N·m, depending on joint type]", "parameter", torch.inf, None),
    _FieldSpec("velocity_limit", torch.Tensor, "[m/s or rad/s, depending on joint type]", "parameter", torch.inf, None),
)
_MOTOR_PARAMETERS = _PD_PARAMETERS + (
    _FieldSpec("saturation_effort", torch.Tensor, "[N or N·m, depending on joint type]", "parameter", 0.0, None),
)
_NEURAL_PARAMETERS = (
    _FieldSpec("effort_limit", torch.Tensor, "[N or N·m, depending on joint type]", "parameter", torch.inf, None),
    _FieldSpec("velocity_limit", torch.Tensor, "[m/s or rad/s, depending on joint type]", "parameter", torch.inf, None),
    _FieldSpec("saturation_effort", torch.Tensor, "[N or N·m, depending on joint type]", "parameter", 0.0, None),
)
_OUTPUTS = (
    _FieldSpec("computed_effort", torch.Tensor, "[N or N·m, depending on joint type]", "output", 0.0, None),
    _FieldSpec("applied_effort", torch.Tensor, "[N or N·m, depending on joint type]", "output", 0.0, None),
)

_IMPLICIT_ACTUATOR_SCHEMA = _ActuatorSchema(_PD_PARAMETERS + _OUTPUTS, graphable=True, stateful=False)
_IDEAL_PD_ACTUATOR_SCHEMA = _ActuatorSchema(_PD_PARAMETERS + _OUTPUTS, graphable=True, stateful=False)
_DC_MOTOR_SCHEMA = _ActuatorSchema(_MOTOR_PARAMETERS + _OUTPUTS, graphable=True, stateful=False)
_DELAYED_PD_ACTUATOR_SCHEMA = _ActuatorSchema(_PD_PARAMETERS + _OUTPUTS, graphable=False, stateful=True)
_REMOTIZED_PD_ACTUATOR_SCHEMA = _ActuatorSchema(_PD_PARAMETERS + _OUTPUTS, graphable=False, stateful=True)
_ACTUATOR_NET_LSTM_SCHEMA = _ActuatorSchema(_NEURAL_PARAMETERS + _OUTPUTS, graphable=False, stateful=True)
_ACTUATOR_NET_MLP_SCHEMA = _ActuatorSchema(_NEURAL_PARAMETERS + _OUTPUTS, graphable=False, stateful=True)
