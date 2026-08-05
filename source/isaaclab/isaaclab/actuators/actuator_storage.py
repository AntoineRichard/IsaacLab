# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private typed parameter storage used by managed actuator groups."""

from __future__ import annotations

import operator
from collections.abc import Callable, ItemsView, Iterator, KeysView, Mapping, MutableMapping, Sequence, ValuesView
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from . import actuator_kernels

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan

    from .actuator_base import ActuatorBase


def _construction_stream(device: str) -> tuple[wp.Stream | None, torch.cuda.Stream | None, torch.cuda.Stream | None]:
    """Return a Warp stream ordered after the caller's current Torch stream.

    The caller may have just produced source tensors on a non-default Torch
    stream.  Warp cannot infer that dependency from a raw tensor pointer, so
    make it explicit before the first Warp consumer launch.  The caller stream
    is returned too, allowing the construction method to publish its Warp
    writes without a host synchronization.
    """
    if torch.device(device).type != "cuda":
        return None, None, None
    caller_stream = torch.cuda.current_stream(device)
    stream = wp.get_stream(device)
    torch_stream = wp.stream_to_torch(stream)
    if torch_stream.cuda_stream != caller_stream.cuda_stream:
        torch_stream.wait_stream(caller_stream)
    return stream, torch_stream, caller_stream


def _publish_construction_stream(
    torch_stream: torch.cuda.Stream | None, caller_stream: torch.cuda.Stream | None
) -> None:
    """Make Warp construction writes visible to the caller's Torch stream."""
    if torch_stream is not None and caller_stream is not None and torch_stream.cuda_stream != caller_stream.cuda_stream:
        caller_stream.wait_stream(torch_stream)


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
    solver_proxies: Mapping[str, ProxyArray] | None = None


class _BackendParameterStaging:
    """Candidate-owned articulation-order staging for backend parameter writes.

    A backend receives dense articulation-joint arrays while canonical actuator
    parameters remain compact exact-type arrays.  This object owns one stable
    dense target per backend route and uses the candidate's configuration-owner
    slots to patch only the selected joints.  It never stores caller-owned
    command values or selector tensors.
    """

    def __init__(
        self,
        *,
        num_worlds: int,
        num_joints: int,
        device: str,
        owner_slots: Mapping[tuple[type[ActuatorBase], str], torch.Tensor | ProxyArray],
        initial_values: Mapping[str, ProxyArray] | None = None,
        all_env_ids: torch.Tensor | None = None,
        all_joint_ids: torch.Tensor | None = None,
        all_env_mask: torch.Tensor | None = None,
        all_joint_mask: torch.Tensor | None = None,
        all_env_ids_wp: wp.array | None = None,
        all_joint_ids_wp: wp.array | None = None,
        all_env_mask_wp: wp.array | None = None,
        all_joint_mask_wp: wp.array | None = None,
    ) -> None:
        """Allocate pointer-stable staging targets for the supplied backend routes.

        Args:
            num_worlds: Number of articulation worlds.
            num_joints: Number of articulation joints.
            device: Device hosting the candidate allocations.
            owner_slots: Candidate-owned articulation-joint to compact-slot maps.
            all_env_ids: Optional selector-state view for every articulation world.
            all_joint_ids: Optional selector-state view for every articulation joint.
            all_env_mask: Optional selector-state true mask for every world.
            all_joint_mask: Optional selector-state true mask for every joint.
        """
        self._num_worlds = num_worlds
        self._num_joints = num_joints
        self._targets: dict[str, ProxyArray] = {}
        self._owner_slots: dict[tuple[type[ActuatorBase], str], ProxyArray] = {}
        self._all_env_ids = (
            torch.arange(num_worlds, dtype=torch.int32, device=device) if all_env_ids is None else all_env_ids
        )
        self._all_joint_ids = (
            torch.arange(num_joints, dtype=torch.int32, device=device) if all_joint_ids is None else all_joint_ids
        )
        self._all_env_mask = (
            torch.ones(num_worlds, dtype=torch.bool, device=device) if all_env_mask is None else all_env_mask
        )
        self._all_joint_mask = (
            torch.ones(num_joints, dtype=torch.bool, device=device) if all_joint_mask is None else all_joint_mask
        )
        self._selector_aliases: dict[tuple[int, int, torch.dtype, torch.device], wp.array] = {
            self._selector_key(self._all_env_ids): all_env_ids_wp
            if all_env_ids_wp is not None
            else wp.from_torch(self._all_env_ids, dtype=wp.int32),
            self._selector_key(self._all_joint_ids): all_joint_ids_wp
            if all_joint_ids_wp is not None
            else wp.from_torch(self._all_joint_ids, dtype=wp.int32),
            self._selector_key(self._all_env_mask): all_env_mask_wp
            if all_env_mask_wp is not None
            else wp.from_torch(self._all_env_mask, dtype=wp.bool),
            self._selector_key(self._all_joint_mask): all_joint_mask_wp
            if all_joint_mask_wp is not None
            else wp.from_torch(self._all_joint_mask, dtype=wp.bool),
        }
        # The ordinary ``None`` selector path is intentionally a direct route:
        # it neither builds selector keys nor performs a dictionary lookup.
        self._all_env_ids_wp = self._selector_aliases[self._selector_key(self._all_env_ids)]
        self._all_joint_ids_wp = self._selector_aliases[self._selector_key(self._all_joint_ids)]
        self._all_env_mask_wp = self._selector_aliases[self._selector_key(self._all_env_mask)]
        self._all_joint_mask_wp = self._selector_aliases[self._selector_key(self._all_joint_mask)]
        if (
            self._all_env_ids.shape != (num_worlds,)
            or self._all_joint_ids.shape != (num_joints,)
            or self._all_env_mask.shape != (num_worlds,)
            or self._all_joint_mask.shape != (num_joints,)
        ):
            raise ValueError("Backend staging selector-state views have invalid articulation shapes.")
        initial_values = {} if initial_values is None else initial_values
        for route, slots in owner_slots.items():
            if isinstance(slots, ProxyArray):
                slot_proxy = slots
            else:
                if slots.dtype is not torch.int32 or slots.ndim != 1 or slots.shape[0] != num_joints:
                    raise ValueError("Backend owner slots must be one-dimensional int32 articulation-joint maps.")
                requested_device = torch.device(device)
                if slots.device.type != requested_device.type or (
                    requested_device.index is not None and slots.device.index != requested_device.index
                ):
                    raise ValueError("Backend owner slots must be on the candidate device.")
                slot_proxy = ProxyArray(wp.from_torch(slots, dtype=wp.int32))
            self._owner_slots[route] = slot_proxy
            _, field_name = route
            if field_name not in self._targets:
                target = ProxyArray(wp.zeros((num_worlds, num_joints), dtype=wp.float32, device=device))
                initial = initial_values.get(field_name)
                if initial is not None:
                    if initial.torch.shape != (num_worlds, num_joints) or initial.torch.device != target.torch.device:
                        raise ValueError(
                            "Backend staging initial values must match the candidate articulation shape/device."
                        )
                    target.torch.copy_(initial.torch)
                self._targets[field_name] = target

    def target(self, actuator_type: type[ActuatorBase], name: str) -> ProxyArray:
        """Return the stable dense target for one exact backend parameter route."""
        del actuator_type
        return self._targets[name]

    def patch_index(
        self,
        *,
        actuator_type: type[ActuatorBase],
        name: str,
        canonical: ProxyArray,
        env_ids: torch.Tensor | wp.array,
        joint_ids: torch.Tensor | wp.array,
    ) -> None:
        """Patch selected signed articulation indices from canonical compact storage."""
        route = (actuator_type, name)
        target = self._targets[name]
        self._validate_canonical(canonical, target)
        env_ids_wp = self._index_warp(env_ids, target, "env_ids")
        joint_ids_wp = self._index_warp(joint_ids, target, "joint_ids")
        if env_ids.shape[0] == 0 or joint_ids.shape[0] == 0:
            return
        wp.launch(
            actuator_kernels.patch_backend_parameter_index,
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[
                canonical.warp,
                env_ids_wp,
                joint_ids_wp,
                self._owner_slots[route].warp,
                self._num_worlds,
                self._num_joints,
            ],
            outputs=[target.warp],
            device=target.warp.device,
        )

    def patch_mask(
        self,
        *,
        actuator_type: type[ActuatorBase],
        name: str,
        canonical: ProxyArray,
        env_mask: torch.Tensor | wp.array,
        joint_mask: torch.Tensor | wp.array,
    ) -> None:
        """Patch full articulation masks from canonical compact storage."""
        route = (actuator_type, name)
        target = self._targets[name]
        self._validate_canonical(canonical, target)
        env_mask_wp = self._mask_warp(env_mask, self._num_worlds, target, "env_mask")
        joint_mask_wp = self._mask_warp(joint_mask, self._num_joints, target, "joint_mask")
        wp.launch(
            actuator_kernels.patch_backend_parameter_mask,
            dim=(self._num_worlds, self._num_joints),
            inputs=[canonical.warp, env_mask_wp, joint_mask_wp, self._owner_slots[route].warp],
            outputs=[target.warp],
            device=target.warp.device,
        )

    def patch_write(self, *, actuator_type: type[ActuatorBase], name: str, write: Any) -> None:
        """Patch one backend write using its type-wide canonical source.

        ``write.value`` may be a strided logical-group view.  The explicit
        ``canonical`` alias always addresses the full exact-type compact block,
        which is the coordinate system used by backend owner slots.
        """
        canonical = getattr(write, "canonical", None)
        if not isinstance(canonical, ProxyArray):
            raise TypeError("Backend parameter writes require a canonical compact ProxyArray source.")
        env_mask = getattr(write, "env_mask", None)
        joint_mask = getattr(write, "joint_mask", None)
        if env_mask is not None or joint_mask is not None:
            self.patch_mask(
                actuator_type=actuator_type,
                name=name,
                canonical=canonical,
                env_mask=self._all_env_mask if env_mask is None else env_mask,
                joint_mask=self._all_joint_mask if joint_mask is None else joint_mask,
            )
            return
        env_ids = getattr(write, "env_ids", None)
        joint_ids = getattr(write, "joint_ids", None)
        if env_ids is None and joint_ids is None:
            route = (actuator_type, name)
            target = self._targets[name]
            self._validate_canonical(canonical, target)
            wp.launch(
                actuator_kernels.patch_backend_parameter_index,
                dim=(self._num_worlds, self._num_joints),
                inputs=[
                    canonical.warp,
                    self._all_env_ids_wp,
                    self._all_joint_ids_wp,
                    self._owner_slots[route].warp,
                    self._num_worlds,
                    self._num_joints,
                ],
                outputs=[target.warp],
                device=target.warp.device,
            )
            return
        self.patch_index(
            actuator_type=actuator_type,
            name=name,
            canonical=canonical,
            env_ids=self._all_env_ids if env_ids is None else env_ids,
            joint_ids=self._all_joint_ids if joint_ids is None else joint_ids,
        )

    def close(self) -> None:
        """Release candidate-owned staging targets and cached device aliases."""
        self._targets.clear()
        self._owner_slots.clear()
        self._all_env_ids = None
        self._all_joint_ids = None
        self._all_env_mask = None
        self._all_joint_mask = None
        self._all_env_ids_wp = None
        self._all_joint_ids_wp = None
        self._all_env_mask_wp = None
        self._all_joint_mask_wp = None
        self._selector_aliases.clear()

    def _index_warp(self, ids: torch.Tensor | wp.array, target: ProxyArray, name: str) -> wp.array:
        """Return a signed selector Warp alias without allocating device storage."""
        if isinstance(ids, wp.array):
            if ids.dtype not in (wp.int32, wp.int64):
                raise TypeError("Backend parameter indices must have a signed integer dtype.")
            if ids.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if str(ids.device) != str(target.warp.device):
                raise ValueError(f"{name} must be on the candidate device.")
            return ids
        if ids.dtype not in (torch.int32, torch.int64) or ids.ndim != 1:
            raise TypeError("Backend parameter indices must be one-dimensional signed integer tensors.")
        if ids.device != target.torch.device:
            raise ValueError(f"{name} must be on the candidate device.")
        alias = self._selector_aliases.get(self._selector_key(ids))
        return (
            alias if alias is not None else wp.from_torch(ids, dtype=wp.int32 if ids.dtype is torch.int32 else wp.int64)
        )

    def _mask_warp(self, mask: torch.Tensor | wp.array, size: int, target: ProxyArray, name: str) -> wp.array:
        """Return a full boolean-mask Warp alias without allocating device storage."""
        if isinstance(mask, wp.array):
            if mask.dtype is not wp.bool or mask.ndim != 1 or mask.shape[0] != size:
                raise ValueError(f"{name} must be a one-dimensional boolean mask with shape ({size},).")
            if str(mask.device) != str(target.warp.device):
                raise ValueError(f"{name} must be on the candidate device.")
            return mask
        if mask.dtype is not torch.bool or mask.ndim != 1 or mask.shape[0] != size:
            raise ValueError(f"{name} must be a one-dimensional boolean mask with shape ({size},).")
        if mask.device != target.torch.device:
            raise ValueError(f"{name} must be on the candidate device.")
        alias = self._selector_aliases.get(self._selector_key(mask))
        return alias if alias is not None else wp.from_torch(mask, dtype=wp.bool)

    def _validate_canonical(self, canonical: ProxyArray, target: ProxyArray) -> None:
        """Validate canonical source metadata without reading device values."""
        if canonical.torch.device != target.torch.device:
            raise ValueError("Backend canonical values must be on the candidate device.")
        if canonical.torch.dtype is not torch.float32:
            raise TypeError("Backend canonical values must have float32 dtype.")
        if canonical.torch.ndim != 2 or canonical.torch.shape[0] != self._num_worlds:
            raise ValueError("Backend canonical values must be two-dimensional with one row per candidate world.")

    @staticmethod
    def _selector_key(value: torch.Tensor) -> tuple[int, int, torch.dtype, torch.device]:
        """Return a shape-aware identity key for an owned selector alias."""
        return value.data_ptr(), value.numel(), value.dtype, value.device


@dataclass(frozen=True)
class _GroupRegistration:
    """Resolved numeric metadata for one logical group on one source prototype."""

    name: str
    actuator_type: type[ActuatorBase]
    joint_indices: tuple[int, ...]
    values: Mapping[str, tuple[float, ...]]
    joint_names: tuple[str, ...] = ()
    source_values: Mapping[str, torch.Tensor] = field(default_factory=dict)
    solver_values: Mapping[str, torch.Tensor] = field(default_factory=dict)
    resolved: Any | None = None


@dataclass(frozen=True)
class _PrototypeRegistration:
    """Resolved actuator metadata for one clone-plan source prototype."""

    registration_key: object
    num_joints: int
    groups: tuple[_GroupRegistration, ...]
    source_resolved: bool = False
    solver_default_values: Mapping[str, torch.Tensor] = field(default_factory=dict)


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
    prototype_values: Mapping[str, torch.Tensor | tuple[tuple[float, ...], ...]]
    solver_prototype_values: Mapping[str, torch.Tensor]
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
    prototype_rows: Sequence[int]
    prototype_assignment: torch.Tensor
    prototype_source_columns: torch.Tensor
    source_slot_by_backend_row: torch.Tensor | None
    clone_plan_metadata: object | None
    group_layouts: tuple[_GroupLayout, ...]
    type_layouts: Mapping[type[ActuatorBase], _TypeLayout]
    solver_default_values: Mapping[str, torch.Tensor]


def _build_articulation_layout(  # noqa: C901
    *,
    replication_cfg_id: int,
    clone_plan: ClonePlan | None,
    registrations: Sequence[_PrototypeRegistration],
    type_offsets: MutableMapping[type[ActuatorBase], int] | None = None,
    prototype_rows: Sequence[int] | None = None,
    prototype_assignment: torch.Tensor | None = None,
    prototype_source_columns: torch.Tensor | None = None,
    source_slot_by_backend_row: torch.Tensor | None = None,
    num_worlds: int | None = None,
    clone_plan_metadata: object | None = None,
) -> _ArticulationLayout:
    """Build one articulation layout from source-prototype registrations.

    The clone assignment remains host topology until matching consumers pack it
    into one device slab. Python work is bounded by the number of source
    prototypes, logical groups, and articulation joints.
    """
    if clone_plan is None:
        explicit_inputs = {
            "prototype_rows": prototype_rows,
            "prototype_assignment": prototype_assignment,
            "prototype_source_columns": prototype_source_columns,
            "source_slot_by_backend_row": source_slot_by_backend_row,
            "num_worlds": num_worlds,
        }
        missing_inputs = tuple(name for name, value in explicit_inputs.items() if value is None)
        if missing_inputs:
            raise ValueError(
                f"An articulation layout with no clone plan requires explicit values for: {', '.join(missing_inputs)}."
            )
    if prototype_rows is None:
        prototype_rows = clone_plan.cfg_rows[replication_cfg_id]
    source_resolved = len(registrations) == 1 and registrations[0].source_resolved
    if len(registrations) != len(prototype_rows) and not source_resolved:
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

    if num_worlds is None:
        num_worlds = int(clone_plan.clone_mask.shape[1])
    if prototype_assignment is None or prototype_source_columns is None:
        prototype_row_indices = torch.tensor(prototype_rows, dtype=torch.long, device=clone_plan.clone_mask.device)
        selected_mask = clone_plan.clone_mask[prototype_row_indices]
        if prototype_assignment is None:
            prototype_assignment = torch.argmax(selected_mask.to(dtype=torch.int32), dim=0).to(dtype=torch.int32)
        if prototype_source_columns is None:
            prototype_source_columns = torch.argmax(selected_mask.to(dtype=torch.int32), dim=1).to(dtype=torch.int64)
    if prototype_assignment.shape != (num_worlds,):
        raise ValueError("Prototype assignment must have one source slot per backend articulation row.")
    if prototype_source_columns.shape != (len(prototype_rows),):
        raise ValueError("Prototype source columns must have one representative per compact source slot.")

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
                    solver_prototype_values=MappingProxyType({}),
                    prototype_assignment=prototype_assignment,
                )
            continue
        parameter_names = actuator_type._parameter_schema().parameter_names
        for group_index, group in indexed_groups:
            unknown_fields = (
                set().union(
                    *(
                        set(registration.groups[group_index].values)
                        | set(registration.groups[group_index].source_values)
                        for registration in registrations
                    )
                )
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
            prototype_fields: dict[str, torch.Tensor | tuple[tuple[float, ...], ...]] = {}
            solver_fields: dict[str, torch.Tensor] = {}
            field_names = tuple(
                dict.fromkeys(
                    name
                    for registration in registrations
                    for name in (
                        *registration.groups[group_index].values,
                        *registration.groups[group_index].source_values,
                    )
                )
            )
            for field_name in field_names:
                if source_resolved:
                    source_values = group.source_values.get(field_name)
                    if source_values is None:
                        raise ValueError(
                            f"Source registration omits field {field_name!r} for actuator group {group.name!r}."
                        )
                    if (
                        source_values.dtype is not torch.float32
                        or source_values.ndim != 2
                        or source_values.shape != (len(prototype_rows), len(group.joint_indices))
                    ):
                        raise ValueError(
                            f"Source field {field_name!r} for actuator group {group.name!r} must have shape "
                            f"({len(prototype_rows)}, {len(group.joint_indices)}) and float32 dtype."
                        )
                    prototype_fields[field_name] = source_values
                    continue
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
            if source_resolved:
                for field_name, source_values in group.solver_values.items():
                    if (
                        source_values.dtype is not torch.float32
                        or source_values.ndim != 2
                        or source_values.shape != (len(prototype_rows), len(group.joint_indices))
                    ):
                        raise ValueError(
                            f"Source solver field {field_name!r} for actuator group {group.name!r} must have shape "
                            f"({len(prototype_rows)}, {len(group.joint_indices)}) and float32 dtype."
                        )
                    solver_fields[field_name] = source_values
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
                solver_prototype_values=MappingProxyType(solver_fields),
                prototype_assignment=prototype_assignment,
            )
            group_offset += len(group.joint_indices)

    return _ArticulationLayout(
        registration_key=first.registration_key,
        num_worlds=num_worlds,
        num_joints=first.num_joints,
        prototype_rows=prototype_rows,
        prototype_assignment=prototype_assignment,
        prototype_source_columns=prototype_source_columns,
        source_slot_by_backend_row=source_slot_by_backend_row,
        clone_plan_metadata=clone_plan_metadata,
        group_layouts=tuple(group_layout_by_index[index] for index in range(len(first.groups))),
        type_layouts=MappingProxyType(type_layouts),
        solver_default_values=MappingProxyType(dict(first.solver_default_values) if source_resolved else {}),
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
        self._initialization_records: list[tuple[wp.Stream, tuple[torch.Tensor, ...]]] = []

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
        for field_spec in schema.fields:
            if field_spec.dtype is not torch.Tensor:
                raise TypeError(f"Unsupported managed field dtype for {field_spec.name!r}: {field_spec.dtype!r}.")
            self._fields[field_spec.name] = ProxyArray(wp.empty(expected_offset, dtype=wp.float32, device=device))

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
            # Clone assignments are host topology.  Concatenate every matching
            # articulation before the sole H2D transfer for this exact type.
            assignment_blocks = [groups[0].prototype_assignment.to(dtype=torch.int32) for groups in groups_by_layout]
            assignment_source = torch.cat(assignment_blocks).contiguous()
            assignment_stream, assignment_torch_stream, caller_stream = _construction_stream(device)
            if assignment_torch_stream is not None:
                with torch.cuda.stream(assignment_torch_stream):
                    assignments = assignment_source.to(device=device)
            else:
                assignments = assignment_source.to(device=device)
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
            self._initialization_buffers.extend((assignment_source, assignments, assignment, *metadata))

        for schema_field in schema.fields:
            if schema_field.name not in varying_fields:
                self._fields[schema_field.name].warp.fill_(schema_field.fill)
                continue

            source_devices = {
                rows.device
                for groups in groups_by_layout
                for group in groups
                if isinstance((rows := group.prototype_values.get(schema_field.name)), torch.Tensor)
            }
            if len(source_devices) > 1:
                raise ValueError(
                    f"Source rows for {self.actuator_type.__name__}.{schema_field.name} must share one device."
                )
            source_device = next(iter(source_devices), torch.device(device))
            source_blocks: list[torch.Tensor] = []
            for layout, groups in zip(matching_layouts, groups_by_layout):
                group_blocks: list[torch.Tensor] = []
                for group in groups:
                    rows = group.prototype_values.get(schema_field.name)
                    if rows is None:
                        group_blocks.append(
                            torch.full(
                                (len(layout.prototype_rows), group.num_dofs),
                                schema_field.fill,
                                dtype=torch.float32,
                                device=source_device,
                            )
                        )
                    elif isinstance(rows, torch.Tensor):
                        if rows.device != source_device:
                            raise ValueError(
                                "Source rows for "
                                f"{self.actuator_type.__name__}.{schema_field.name} must share one device."
                            )
                        group_blocks.append(rows.to(dtype=torch.float32))
                    else:
                        group_blocks.append(torch.tensor(rows, dtype=torch.float32, device=source_device))
                source_blocks.append(torch.cat(group_blocks, dim=1))

            source_values = torch.cat([block.reshape(-1) for block in source_blocks]).contiguous()
            launch_stream, launch_torch_stream, caller_stream = _construction_stream(device)
            if source_values.device.type != torch.device(device).type or source_values.device != torch.device(device):
                if launch_torch_stream is not None:
                    with torch.cuda.stream(launch_torch_stream):
                        source_values = source_values.to(device=device)
                else:
                    source_values = source_values.to(device=device)
            source = wp.from_torch(source_values, dtype=wp.float32)
            self._initialization_buffers.extend((source_values, source))
            self._record_launch_buffers(launch_stream, source_values, assignments)
            wp.launch(
                _expand_prototype_field,
                dim=(len(matching_layouts), max(block_num_worlds), max(block_num_dofs)),
                inputs=[self._fields[schema_field.name].warp, source, assignment, *metadata],
                device=device,
                **({"stream": launch_stream} if launch_stream is not None else {}),
            )
            _publish_construction_stream(launch_torch_stream, caller_stream)

    def _release_initialization_buffers(self) -> None:
        """Release construction tensors after recording their final CUDA-stream use."""
        for stream, buffers in self._initialization_records:
            torch_stream = wp.stream_to_torch(stream)
            for buffer in buffers:
                buffer.record_stream(torch_stream)
        self._initialization_buffers.clear()
        self._initialization_records.clear()

    def _record_launch_buffers(self, stream: wp.Stream | None, *buffers: object) -> None:
        """Capture the Warp launch stream for every Torch input retained by a launch."""
        tensors = tuple(buffer for buffer in buffers if isinstance(buffer, torch.Tensor) and buffer.is_cuda)
        if stream is not None and tensors:
            self._initialization_records.append((stream, tensors))

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


class _SolverPropertyStore:
    """Candidate-owned compact solver-property rows and optional device targets.

    All fields first remain source-prototype rows.  CPU transports consume those
    compact rows directly; device transports materialize their dense candidate
    target only after opaque group construction has supplied the final
    configuration-order overlay.
    """

    _FIELD_NAMES = (
        "stiffness",
        "damping",
        "effort_limit_sim",
        "velocity_limit_sim",
        "armature",
        "friction",
        "dynamic_friction",
        "viscous_friction",
    )

    def __init__(self) -> None:
        """Create an unallocated solver-property staging store."""
        self._fields: dict[str, ProxyArray] = {}
        self._articulation_offsets: dict[int, int] = {}
        self._articulation_proxies: dict[tuple[int, str], ProxyArray] = {}
        self._source_rows: dict[tuple[int, str], torch.Tensor] = {}
        self._source_row_producers: dict[tuple[int, str], torch.cuda.Stream] = {}
        self._device_source_rows: dict[tuple[int, str], torch.Tensor] = {}
        self._device_source_slabs: dict[str, torch.Tensor] = {}
        self._device_source_offsets: dict[tuple[int, str], int] = {}
        self._source_owner_slots: dict[int, torch.Tensor] = {}
        self._device_assignments: dict[int, torch.Tensor] = {}
        self._device_assignment_slab: torch.Tensor | None = None
        self._device_assignment_offsets: dict[int, int] = {}
        self._device_assignment_aliases: dict[int, wp.array] = {}
        self._expansion_metadata: dict[tuple[str, tuple[int, ...]], tuple[wp.array, ...]] = {}
        self._expansion_metadata_signatures: dict[tuple[str, tuple[int, ...]], tuple[tuple[int, ...], ...]] = {}
        self._initialization_buffers: list[object] = []
        self._initialization_records: list[tuple[wp.Stream, tuple[torch.Tensor, ...]]] = []

    def allocate(self, layouts: Sequence[_ArticulationLayout], *, device: str) -> None:
        """Resolve all properties into compact source rows without dense allocation."""
        del device
        for layout in layouts:
            for field_name in self._FIELD_NAMES:
                key = (id(layout), field_name)
                source_rows = self._merged_source_rows(field_name, layout)
                self._source_rows[key] = source_rows
                if source_rows.is_cuda:
                    self._source_row_producers[key] = torch.cuda.current_stream(source_rows.device)

    def articulation_proxy(self, field: str, layout: _ArticulationLayout) -> ProxyArray:
        """Return the materialized device target for one property.

        CPU transports intentionally never create this target.
        """
        key = (id(layout), field)
        proxy = self._articulation_proxies.get(key)
        if proxy is None:
            raise RuntimeError(f"Solver property {field!r} has no device transport target.")
        return proxy

    def materialize_device_targets(self, layouts: Sequence[_ArticulationLayout], *, device: str) -> None:
        """Create one flat dense target per field across device-transport articulations."""
        self.prepare_device_assignments(layouts, device=device)
        expected_offset = 0
        for layout in layouts:
            self._articulation_offsets[id(layout)] = expected_offset
            expected_offset += layout.num_worlds * layout.num_joints
        allocation_size = max(expected_offset, 1)
        for field_name in self._FIELD_NAMES:
            if field_name not in self._fields:
                self._fields[field_name] = ProxyArray(wp.empty(allocation_size, dtype=wp.float32, device=device))
        for layout in layouts:
            for field_name in self._FIELD_NAMES:
                key = (id(layout), field_name)
                if key not in self._articulation_proxies:
                    flat = self._fields[field_name]
                    offset = self._articulation_offsets[id(layout)]
                    item_size = wp.types.type_size_in_bytes(wp.float32)
                    shape = (layout.num_worlds, layout.num_joints)
                    proxy = ProxyArray(
                        wp.array(
                            ptr=flat.warp.ptr + offset * item_size,
                            dtype=wp.float32,
                            shape=shape,
                            strides=(layout.num_joints * item_size, item_size),
                            capacity=layout.num_worlds * layout.num_joints * item_size,
                            device=flat.warp.device,
                            copy=False,
                        )
                    )
                    proxy._torch_cache = flat.torch.as_strided(shape, (layout.num_joints, 1), storage_offset=offset)
                    self._articulation_proxies[key] = proxy
        for field_name in self._FIELD_NAMES:
            self._prepare_device_source_rows(layouts, field_name, device=device)
            self._expand_packed_source_rows(field_name, layouts, device=device)

    def _prepare_device_source_rows(
        self, layouts: Sequence[_ArticulationLayout], field_name: str, *, device: str
    ) -> None:
        """Pack one source slab for a solver field without mixing source devices."""
        missing = tuple(layout for layout in layouts if (id(layout), field_name) not in self._device_source_rows)
        if not missing:
            return
        target_device = torch.device(device)
        source_blocks = [self.source_rows(field_name, layout).reshape(-1) for layout in missing]
        stream, torch_stream, caller_stream = _construction_stream(device)
        cursor = 0
        # Sources from each producer device are packed independently.  CPU
        # rows contribute exactly one H2D slab; CUDA rows copy on the chosen
        # construction stream and never require a device-to-host round trip.
        by_device: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
        for block in source_blocks:
            by_device.setdefault(block.device, []).append((cursor, block))
            cursor += block.numel()
        if torch_stream is not None:
            for layout in missing:
                producer = self._source_row_producers.get((id(layout), field_name))
                if producer is not None and producer.cuda_stream != torch_stream.cuda_stream:
                    torch_stream.wait_stream(producer)
        with torch.cuda.stream(torch_stream) if torch_stream is not None else torch.no_grad():
            source_slab = torch.empty(sum(block.numel() for block in source_blocks), dtype=torch.float32, device=device)
            for _source_device, blocks in by_device.items():
                packed = torch.cat([block for _, block in blocks]).contiguous()
                if packed.device != target_device:
                    packed = packed.to(device=device)
                packed_cursor = 0
                for output_offset, block in blocks:
                    next_cursor = packed_cursor + block.numel()
                    source_slab[output_offset : output_offset + block.numel()].copy_(packed[packed_cursor:next_cursor])
                    packed_cursor = next_cursor
        cursor = 0
        for layout in missing:
            size = len(layout.prototype_rows) * layout.num_joints
            self._device_source_offsets[(id(layout), field_name)] = cursor
            self._device_source_rows[(id(layout), field_name)] = source_slab[cursor : cursor + size].view(
                len(layout.prototype_rows), layout.num_joints
            )
            cursor += size
        self._initialization_buffers.append(source_slab)
        if field_name not in self._device_source_slabs:
            self._device_source_slabs[field_name] = source_slab
        self._record_launch_buffers(stream, source_slab)
        _publish_construction_stream(torch_stream, caller_stream)

    def _expand_packed_source_rows(
        self, field_name: str, layouts: Sequence[_ArticulationLayout], *, device: str
    ) -> None:
        """Expand every layout of one field with one bounded Warp launch."""
        source_slab = self._device_source_slabs[field_name]
        assignments = self._device_assignment_slab
        if assignments is None:
            raise RuntimeError("Solver expansion requires an owning assignment slab.")
        output_offsets = [self._articulation_offsets[id(layout)] for layout in layouts]
        assignment_offsets: list[int] = []
        prototype_offsets: list[int] = []
        for layout in layouts:
            assignment_offsets.append(self._device_assignment_offsets[id(layout)])
            prototype_offsets.append(self._device_source_offsets[(id(layout), field_name)])
        metadata = self._expansion_metadata_for(
            layouts=layouts,
            device=device,
            output_offsets=output_offsets,
            assignment_offsets=assignment_offsets,
            prototype_offsets=prototype_offsets,
        )
        stream, torch_stream, caller_stream = _construction_stream(device)
        source_wp = wp.from_torch(source_slab, dtype=wp.float32)
        assignment_wp = wp.from_torch(assignments, dtype=wp.int32)
        self._initialization_buffers.extend((source_slab, assignments, source_wp, assignment_wp))
        self._record_launch_buffers(stream, source_slab, assignments)
        wp.launch(
            _expand_prototype_field,
            dim=(
                len(layouts),
                max(layout.num_worlds for layout in layouts),
                max(layout.num_joints for layout in layouts),
            ),
            inputs=[self._fields[field_name].warp, source_wp, assignment_wp, *metadata],
            device=device,
            **({"stream": stream} if stream is not None else {}),
        )
        _publish_construction_stream(torch_stream, caller_stream)

    def _expansion_metadata_for(
        self,
        *,
        layouts: Sequence[_ArticulationLayout],
        device: str,
        output_offsets: Sequence[int],
        assignment_offsets: Sequence[int],
        prototype_offsets: Sequence[int],
    ) -> tuple[wp.array, ...]:
        """Return one retained metadata bundle for an ordered materialized layout set."""
        key = str(torch.device(device)), tuple(id(layout) for layout in layouts)
        signature = (
            tuple(output_offsets),
            tuple(assignment_offsets),
            tuple(prototype_offsets),
            tuple(layout.num_worlds for layout in layouts),
            tuple(layout.num_joints for layout in layouts),
        )
        existing_signature = self._expansion_metadata_signatures.get(key)
        if existing_signature is not None and existing_signature != signature:
            raise RuntimeError("Solver expansion metadata changed for an already prepared ordered layout set.")
        metadata = self._expansion_metadata.get(key)
        if metadata is None:
            metadata = (
                *(wp.array(values, dtype=wp.int64, device=device) for values in signature[:3]),
                *(wp.array(values, dtype=wp.int32, device=device) for values in signature[3:]),
            )
            self._expansion_metadata[key] = metadata
            self._expansion_metadata_signatures[key] = signature
            self._initialization_buffers.extend(metadata)
        return metadata

    def prepare_device_assignments(self, layouts: Sequence[_ArticulationLayout], *, device: str) -> None:
        """Pack host clone assignments once for all layouts using device expansion."""
        layouts = tuple(layout for layout in layouts if id(layout) not in self._device_assignments)
        if not layouts:
            return
        assignment_source = torch.cat([layout.prototype_assignment.to(dtype=torch.int32) for layout in layouts])
        launch_stream, torch_stream, caller_stream = _construction_stream(device)
        if torch_stream is not None:
            with torch.cuda.stream(torch_stream):
                assignment_slab = assignment_source.to(device=device)
        else:
            assignment_slab = assignment_source.to(device=device)
        assignment_offset = 0
        for layout in layouts:
            self._device_assignment_offsets[id(layout)] = assignment_offset
            assignment = assignment_slab[assignment_offset : assignment_offset + layout.num_worlds]
            assignment_offset += layout.num_worlds
            self._device_assignments[id(layout)] = assignment
            self._device_assignment_aliases[id(layout)] = wp.from_torch(assignment, dtype=wp.int32)
        self._initialization_buffers.extend(
            (assignment_source, assignment_slab, *self._device_assignment_aliases.values())
        )
        if self._device_assignment_slab is None:
            self._device_assignment_slab = assignment_slab
        self._record_launch_buffers(launch_stream, assignment_slab)
        _publish_construction_stream(torch_stream, caller_stream)

    def expand_source_rows_into(self, target: ProxyArray, field_name: str, layout: _ArticulationLayout) -> None:
        """Expand one final compact field into a caller-owned dense target."""
        source_rows = self._device_source_rows.get((id(layout), field_name), self.source_rows(field_name, layout))
        self._expand_source_rows(target, source_rows, layout)

    def _merged_source_rows(
        self,
        field_name: str,
        layout: _ArticulationLayout,
        opaque_values: Mapping[tuple[str, str], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return final source rows after applying every group in config order."""
        source_count = len(layout.prototype_rows)
        default_values = layout.solver_default_values.get(field_name)
        if default_values is None:
            source_device = next(
                (
                    values.device
                    for group in layout.group_layouts
                    if (values := group.solver_prototype_values.get(field_name)) is not None
                ),
                layout.prototype_assignment.device,
            )
            default_values = torch.zeros((source_count, layout.num_joints), dtype=torch.float32, device=source_device)
        if (
            default_values.dtype is not torch.float32
            or default_values.ndim != 2
            or default_values.shape != (source_count, layout.num_joints)
        ):
            raise ValueError("Solver source defaults have an invalid source-row shape or dtype.")

        source_rows = default_values.clone()
        for group in layout.group_layouts:
            values = group.solver_prototype_values.get(field_name)
            if opaque_values is not None:
                values = opaque_values.get((group.name, field_name), values)
            if values is None:
                continue
            if (
                values.dtype is not torch.float32
                or values.ndim != 2
                or values.shape != (source_count, group.num_dofs)
                or values.device != source_rows.device
            ):
                raise ValueError(f"Solver source values for actuator group {group.name!r} are invalid.")
            joint_indices = torch.tensor(group.joint_indices, dtype=torch.long, device=source_rows.device)
            source_rows[:, joint_indices] = values
        return source_rows

    def _expand_source_rows(
        self,
        target: ProxyArray,
        source_rows: torch.Tensor,
        layout: _ArticulationLayout,
    ) -> None:
        """Expand final compact source rows into one dense solver target."""
        assignment = self._device_assignments.get(id(layout))
        if assignment is None:
            if target.torch.device.type != "cpu":
                raise RuntimeError("Device solver expansion requires a packed source-assignment view.")
            assignment = layout.prototype_assignment
        owners = self._source_owner_slots.get(id(layout))
        if owners is None:
            owners = torch.arange(layout.num_joints, dtype=torch.int32, device=target.torch.device)
            self._source_owner_slots[id(layout)] = owners
        launch_stream, torch_stream, caller_stream = _construction_stream(str(target.torch.device))
        if source_rows.device == target.torch.device:
            expansion_rows = source_rows
        elif source_rows.device.type == "cpu" and target.torch.device.type != "cpu":
            # CPU source rows are the intentional PhysX/OV construction path.
            # This H2D copy exists only for the private dense canonical target;
            # the source rows retained for the backend remain host-resident.
            with torch.cuda.stream(torch_stream):
                expansion_rows = source_rows.to(device=target.torch.device)
        else:
            raise ValueError("Solver source rows cannot be copied from device to the canonical target device.")
        source_wp = wp.from_torch(expansion_rows.reshape(-1), dtype=wp.float32)
        assignment_wp = self._device_assignment_aliases.get(id(layout))
        if assignment_wp is None:
            assignment_wp = wp.from_torch(assignment, dtype=wp.int32)
        owners_wp = wp.from_torch(owners, dtype=wp.int32)
        self._initialization_buffers.extend((expansion_rows, assignment, source_wp, assignment_wp, owners_wp))
        self._record_launch_buffers(launch_stream, expansion_rows, assignment, owners)
        wp.launch(
            actuator_kernels.expand_source_property,
            dim=(layout.num_worlds, layout.num_joints),
            inputs=[target.warp, source_wp, assignment_wp, owners_wp, layout.num_joints],
            device=target.warp.device,
            **({"stream": launch_stream} if launch_stream is not None else {}),
        )
        _publish_construction_stream(torch_stream, caller_stream)

    def overlay_opaque_groups(self, layout: _ArticulationLayout, groups: Mapping[str, Any]) -> None:
        """Merge eager opaque-group rows with managed rows in config order.

        Opaque classes do not expose the managed source-resolution seam. Their
        one eager world-sized constructor is therefore allowed to provide
        source rows selected from the clone representatives. When the source
        transport is CPU but an opaque constructor only exposes CUDA tensors,
        this compatibility path packs every compact opaque property row into
        one bounded D2H transfer for the articulation. Managed classes never
        use that fallback. The complete source matrix is rebuilt so an opaque
        group cannot incorrectly win over a later managed group that owns the
        same joint.
        """
        from .actuator_pd import ImplicitActuator

        source_count = len(layout.prototype_rows)
        opaque_groups = tuple(
            group_layout
            for group_layout in layout.group_layouts
            if group_layout.actuator_type.__dict__.get("_parameter_schema") is None
        )
        if not opaque_groups:
            return

        opaque_values: dict[tuple[str, str], torch.Tensor] = {}
        transfer_blocks: list[torch.Tensor] = []
        transfer_keys: list[tuple[str, str]] = []
        source_columns_by_device: dict[torch.device, torch.Tensor] = {}
        for field_name in self._FIELD_NAMES:
            source_device = self.source_rows(field_name, layout).device
            for group_layout in opaque_groups:
                actuator = groups[group_layout.name]
                values = getattr(actuator, field_name, None)
                if field_name in {"stiffness", "damping"} and not issubclass(
                    group_layout.actuator_type, ImplicitActuator
                ):
                    opaque_values[(group_layout.name, field_name)] = torch.zeros(
                        (source_count, group_layout.num_dofs), dtype=torch.float32, device=source_device
                    )
                    continue
                if not isinstance(values, torch.Tensor):
                    continue
                if values.dtype is not torch.float32 or values.shape != (layout.num_worlds, group_layout.num_dofs):
                    raise ValueError(
                        f"Opaque actuator group {group_layout.name!r} has invalid solver field {field_name!r}."
                    )
                source_columns = source_columns_by_device.get(values.device)
                if source_columns is None:
                    source_columns = layout.prototype_source_columns.to(device=values.device, dtype=torch.long)
                    source_columns_by_device[values.device] = source_columns
                source_values = values.index_select(0, source_columns)
                if source_values.device != source_device:
                    if source_device.type != "cpu":
                        source_values = source_values.to(device=source_device)
                    else:
                        transfer_blocks.append(source_values)
                        transfer_keys.append((group_layout.name, field_name))
                        continue
                opaque_values[(group_layout.name, field_name)] = source_values

        if transfer_blocks:
            packed = torch.cat(transfer_blocks, dim=1)
            packed_cpu = packed.to(device="cpu")
            offset = 0
            for key, block in zip(transfer_keys, transfer_blocks, strict=True):
                next_offset = offset + block.shape[1]
                opaque_values[key] = packed_cpu[:, offset:next_offset]
                offset = next_offset

        if not opaque_values:
            return
        for field_name in self._FIELD_NAMES:
            source_rows = self._merged_source_rows(field_name, layout, opaque_values)
            key = (id(layout), field_name)
            self._source_rows[key] = source_rows
            if source_rows.is_cuda:
                self._source_row_producers[key] = torch.cuda.current_stream(source_rows.device)
            target = self._articulation_proxies.get((id(layout), field_name))
            if target is not None:
                self._expand_source_rows(target, source_rows, layout)

    def source_rows(self, field_name: str, layout: _ArticulationLayout) -> torch.Tensor:
        """Return the final config-order-merged source rows for one property."""
        return self._source_rows[(id(layout), field_name)]

    def source_assignment(self, layout: _ArticulationLayout) -> torch.Tensor:
        """Return the cached device assignment for a device solver payload."""
        return self._device_assignments[id(layout)]

    def close(self) -> None:
        """Release every candidate-owned solver staging allocation and alias."""
        self._release_initialization_buffers()
        self._articulation_proxies.clear()
        self._articulation_offsets.clear()
        self._source_rows.clear()
        self._source_row_producers.clear()
        self._device_source_rows.clear()
        self._device_source_slabs.clear()
        self._device_source_offsets.clear()
        self._source_owner_slots.clear()
        self._device_assignments.clear()
        self._device_assignment_slab = None
        self._device_assignment_offsets.clear()
        self._device_assignment_aliases.clear()
        self._expansion_metadata.clear()
        self._expansion_metadata_signatures.clear()
        self._fields.clear()

    def _release_initialization_buffers(self) -> None:
        """Release CUDA launch inputs only after associating their launch stream."""
        for stream, buffers in self._initialization_records:
            torch_stream = wp.stream_to_torch(stream)
            for buffer in buffers:
                buffer.record_stream(torch_stream)
        self._initialization_buffers.clear()
        self._initialization_records.clear()

    def _record_launch_buffers(self, stream: wp.Stream | None, *buffers: object) -> None:
        """Capture the Warp stream that consumes the retained Torch launch inputs."""
        tensors = tuple(buffer for buffer in buffers if isinstance(buffer, torch.Tensor) and buffer.is_cuda)
        if stream is not None and tensors:
            self._initialization_records.append((stream, tensors))


class _JointDomainStore:
    """Warp-owned articulation-major command and telemetry storage.

    Each field owns one flat allocation for the active generation. Cached
    articulation aliases remain contiguous without adding per-world Python or
    device metadata objects.
    """

    _FIELD_NAMES = (
        "raw_position",
        "raw_velocity",
        "raw_effort",
        "processed_position",
        "processed_velocity",
        "processed_effort",
        "computed_effort",
        "applied_effort",
    )

    def __init__(self) -> None:
        """Create an unallocated joint-domain store."""
        self._fields: dict[str, ProxyArray] = {}
        self._articulation_offsets: dict[int, int] = {}
        self._articulation_proxies: dict[tuple[int, str], ProxyArray] = {}
        self._compatibility_projections: dict[tuple[int, str], ProxyArray] = {}

    def allocate(self, layouts: Sequence[_ArticulationLayout], *, device: str) -> None:
        """Allocate every joint-domain field once for the supplied articulations.

        Args:
            layouts: Articulation layouts in deterministic registration order.
            device: Warp device on which to allocate the flat fields.
        """
        expected_offset = 0
        for layout in layouts:
            self._articulation_offsets[id(layout)] = expected_offset
            expected_offset += layout.num_worlds * layout.num_joints
        allocation_size = max(expected_offset, 1)
        self._fields = {
            name: ProxyArray(wp.zeros(allocation_size, dtype=wp.float32, device=device)) for name in self._FIELD_NAMES
        }
        for layout in layouts:
            for field_name in self._FIELD_NAMES:
                self.articulation_proxy(field_name, layout)

    def articulation_proxy(self, field: str, layout: _ArticulationLayout) -> ProxyArray:
        """Return one cached contiguous articulation alias for a joint-domain field.

        Args:
            field: Canonical joint-domain field name.
            layout: Articulation layout that determines the alias shape.

        Returns:
            A contiguous ``[num_worlds, num_joints]`` proxy alias.
        """
        if field not in self._fields:
            raise KeyError(f"Unknown joint-domain field {field!r}.")
        key = (id(layout), field)
        proxy = self._articulation_proxies.get(key)
        if proxy is None:
            offset = self._articulation_offsets[id(layout)]
            shape = (layout.num_worlds, layout.num_joints)
            item_size = wp.types.type_size_in_bytes(wp.float32)
            flat = self._fields[field]
            capacity = layout.num_worlds * layout.num_joints * item_size
            warp_view = wp.array(
                ptr=flat.warp.ptr + offset * item_size,
                dtype=wp.float32,
                shape=shape,
                strides=(layout.num_joints * item_size, item_size),
                capacity=capacity,
                device=flat.warp.device,
                copy=False,
            )
            proxy = ProxyArray(warp_view)
            proxy._torch_cache = flat.torch.as_strided(shape, (layout.num_joints, 1), storage_offset=offset)
            self._articulation_proxies[key] = proxy
        return proxy

    def compatibility_projection(self, name: str, layout: _ArticulationLayout) -> ProxyArray:
        """Return one lazy articulation-order legacy compatibility projection.

        Args:
            name: Legacy projection name.
            layout: Articulation layout that determines the projection shape.

        Returns:
            A stable ``[num_worlds, num_joints]`` proxy filled with the legacy
            default until compatible typed parameters are scattered into it.
        """
        fills = {"soft_joint_vel_limits": 0.0, "gear_ratio": 1.0}
        try:
            fill = fills[name]
        except KeyError as error:
            raise KeyError(f"Unknown actuator compatibility projection {name!r}.") from error
        key = (id(layout), name)
        projection = self._compatibility_projections.get(key)
        if projection is None:
            projection = ProxyArray(
                wp.full(
                    (layout.num_worlds, layout.num_joints),
                    fill,
                    dtype=wp.float32,
                    device=self._fields["raw_position"].warp.device,
                )
            )
            self._compatibility_projections[key] = projection
        return projection

    def refresh_compatibility_projection(
        self, name: str, layout: _ArticulationLayout, groups: Mapping[str, ActuatorBase]
    ) -> None:
        """Refresh an activated projection from configuration-order parameter owners.

        This intentionally does nothing before first access so regular typed
        actuator construction and execution do not allocate legacy storage.
        """
        projection = self._compatibility_projections.get((id(layout), name))
        if projection is None:
            return
        source_name, fill = {
            "soft_joint_vel_limits": ("velocity_limit", 0.0),
            "gear_ratio": ("gear_ratio", 1.0),
        }.get(name, (None, None))
        if source_name is None:
            raise KeyError(f"Unknown actuator compatibility projection {name!r}.")
        projection.warp.fill_(fill)
        for group_layout in layout.group_layouts:
            binding = groups[group_layout.name].__dict__.get("_parameter_binding")
            if binding is None or binding.parameter_proxies is None:
                continue
            source = binding.parameter_proxies.get(source_name)
            if source is not None:
                projection.torch[:, binding.joint_indices] = source.torch

    def refresh_compatibility_projections(
        self, layout: _ArticulationLayout, groups: Mapping[str, ActuatorBase]
    ) -> None:
        """Refresh every projection that has already been allocated for one articulation."""
        for name in ("soft_joint_vel_limits", "gear_ratio"):
            self.refresh_compatibility_projection(name, layout, groups)

    def close(self) -> None:
        """Release every flat allocation and cached articulation alias."""
        self._compatibility_projections.clear()
        self._articulation_proxies.clear()
        self._articulation_offsets.clear()
        self._fields.clear()

    def __getattr__(self, name: str) -> ProxyArray:
        """Expose one flat joint-domain field by its canonical name."""
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
