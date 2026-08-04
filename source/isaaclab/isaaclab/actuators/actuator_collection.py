# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime actuator collection for articulations."""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from prettytable import PrettyTable

from isaaclab.utils.string import ResolvableString, string_to_callable
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache

from . import actuator_kernels
from .actuator_base import ActuatorBase, _SolverCompatibilitySeed
from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_control import (
    ActuatorControl,
    ActuatorJointProperties,
    _ActuatorParameterWrite,
    _ResolvedSolverProperties,
    _ResolvedSolverProperty,
)
from .actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from .actuator_storage import (
    _ArticulationLayout,
    _BackendParameterStaging,
    _build_articulation_layout,
    _GroupBinding,
    _GroupRegistration,
    _GuardedItemsView,
    _GuardedIterator,
    _GuardedKeysView,
    _GuardedValuesView,
    _JointDomainStore,
    _PrototypeRegistration,
    _SolverPropertyStore,
    _TypedStore,
)

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan

logger = logging.getLogger(__name__)


def _copy_actuator_cfg(cfg: ActuatorBaseCfg) -> ActuatorBaseCfg:
    """Copy one actuator config without retaining mutable runtime state."""
    copy_method = getattr(cfg, "copy", None)
    if callable(copy_method):
        return copy_method()
    return copy.deepcopy(cfg)


def _resolve_actuator_type(cfg: ActuatorBaseCfg) -> type[ActuatorBase]:
    """Resolve and validate one configured actuator class without changing its config.

    Config classes deliberately retain :class:`~isaaclab.utils.string.ResolvableString`
    values for serialization.  Candidate construction needs the concrete class for
    schema discovery, though, so it resolves the value once at this boundary and
    carries the class in the private layout rather than writing it back to the
    caller-owned configuration.
    """
    configured_type = cfg.class_type
    if isinstance(configured_type, ResolvableString):
        actuator_type = configured_type._resolve()
    elif isinstance(configured_type, str):
        actuator_type = string_to_callable(configured_type)
    else:
        actuator_type = configured_type
    if not isinstance(actuator_type, type) or not issubclass(actuator_type, ActuatorBase):
        raise TypeError("Actuator configuration class_type must resolve to an ActuatorBase subclass.")
    return actuator_type


def _clone_source_assignment(clone_plan: Any, replication_cfg_id: int) -> object | None:
    """Return clone-plan source metadata without coupling consumers to its storage shape."""
    for accessor_name in ("get_source_assignment", "source_assignment"):
        accessor = getattr(clone_plan, accessor_name, None)
        if callable(accessor):
            return accessor(replication_cfg_id)
    assignments = getattr(clone_plan, "_source_assignments", None)
    return None if assignments is None else assignments.get(replication_cfg_id)


class _GuardedDictFromKeys:
    """Return ordinary dict snapshots while guarding instance-bound calls."""

    def __get__(self, instance: Any, owner: type | None = None):
        del owner

        def fromkeys(iterable, value=None) -> dict:
            if instance is not None:
                instance._require_current_generation()
            return dict.fromkeys(iterable, value)

        return fromkeys


@dataclass(frozen=True)
class _ArticulationRegistration:
    """Metadata registered by an articulation before collection publication."""

    key: object
    cfgs: Mapping[str, ActuatorBaseCfg]
    control: ActuatorControl
    replication_cfg_id: int
    debug_validation: bool
    debug_value_resolution: bool
    native_group_names: frozenset[str]


@dataclass(frozen=True)
class _ArticulationBinding:
    """Private candidate binding for one articulation registration."""

    registration: _ArticulationRegistration
    layout: _ArticulationLayout
    backend_parameter_staging: _BackendParameterStaging | None = None


@dataclass(frozen=True)
class _ManagedGroupResolution:
    """Source-bounded exact-class resolution retained by one candidate group."""

    configuration_index: int
    native_managed: bool
    source_env_ids: torch.Tensor
    resolved: Any
    backend_binding_metadata: Mapping[str, Any]


class _SelectorState:
    """Pointer-stable selector metadata owned by one candidate articulation.

    The three flat slabs are the only owning selector allocations. Logical group
    and exact-type routes retain Python slices into them rather than individual
    device tensors, which keeps allocation count independent of group count.
    """

    def __init__(self, binding: _ArticulationBinding) -> None:
        layout = binding.layout
        device = binding.registration.control.device
        self.token = object()
        self._closed = False
        self._num_worlds = layout.num_worlds
        self._num_joints = layout.num_joints
        self._max_scope_dofs = max((type_layout.num_dofs for type_layout in layout.type_layouts.values()), default=0)
        identity_size = max(layout.num_worlds, layout.num_joints, self._max_scope_dofs)
        opaque_group_layouts = tuple(
            group for group in layout.group_layouts if group.actuator_type not in layout.type_layouts
        )
        scope_size = sum(type_layout.num_dofs for type_layout in layout.type_layouts.values()) + sum(
            len(group.joint_indices) for group in opaque_group_layouts
        )
        group_inverse_size = len(layout.group_layouts) * layout.num_joints

        backend_routes: list[tuple[type[ActuatorBase], str]] = []
        for actuator_type in layout.type_layouts:
            fields: set[str] = set()
            if issubclass(actuator_type, ImplicitActuator):
                fields.update(("stiffness", "damping"))
            if any(
                group.name in binding.registration.native_group_names and group.actuator_type is actuator_type
                for group in layout.group_layouts
            ):
                fields.update(actuator_type._parameter_schema().parameter_names)
            backend_routes.extend((actuator_type, field) for field in sorted(fields))

        winner_by_field: dict[str, list[tuple[type[ActuatorBase], int] | None]] = {
            field: [None] * layout.num_joints for _, field in backend_routes
        }
        for group_layout in layout.group_layouts:
            fields: set[str] = set()
            if issubclass(group_layout.actuator_type, ImplicitActuator):
                fields.update(("stiffness", "damping"))
            if group_layout.name in binding.registration.native_group_names:
                fields.update(group_layout.actuator_type._parameter_schema().parameter_names)
            for field in fields:
                winners = winner_by_field.get(field)
                if winners is None:
                    continue
                for local_slot, joint_id in enumerate(group_layout.joint_indices):
                    winners[joint_id] = (group_layout.actuator_type, group_layout.type_slice.start + local_slot)
            # Every non-implicit group owns its configured joints with respect
            # to the solver gains, even when it has no backend route itself.
            # This config-order blocker prevents an earlier implicit group from
            # continuing to write gains after a later explicit/opaque group has
            # resolved the final solver source rows for that joint.
            if (
                not issubclass(group_layout.actuator_type, ImplicitActuator)
                and group_layout.name not in binding.registration.native_group_names
            ):
                for field in ("stiffness", "damping"):
                    winners = winner_by_field.get(field)
                    if winners is not None:
                        for joint_id in group_layout.joint_indices:
                            winners[joint_id] = None

        owner_rows_by_policy: dict[tuple[type[ActuatorBase], str], int] = {}
        owner_row_values: list[list[int]] = []
        for route in backend_routes:
            actuator_type, field = route
            owners = [-1] * layout.num_joints
            for joint_id, winner in enumerate(winner_by_field[field]):
                if winner is not None and winner[0] is actuator_type:
                    owners[joint_id] = winner[1]
            owner_row = next((index for index, values in enumerate(owner_row_values) if values == owners), None)
            if owner_row is None:
                owner_row = len(owner_row_values)
                owner_row_values.append(owners)
            owner_rows_by_policy[route] = owner_row
        backend_owner_size = len(owner_row_values) * layout.num_joints

        compact_values = [
            joint_id for type_layout in layout.type_layouts.values() for joint_id in type_layout.compact_joint_indices
        ] + [joint_id for group in opaque_group_layouts for joint_id in group.joint_indices]
        owner_values = [value for row in owner_row_values for value in row]

        # Layout: identity ids, compact joint ids, group inverse rows, duplicate
        # scratch rows, and configuration-order backend-owner rows. Build the
        # device slab from one host tensor: vectorized initialization avoids
        # per-world Python objects while keeping one retained int32 allocation.
        int_size = identity_size + scope_size + group_inverse_size + layout.num_worlds + layout.num_joints
        int_size += backend_owner_size
        int_host = torch.empty(int_size, dtype=torch.int32)
        cursor = 0
        torch.arange(identity_size, dtype=torch.int32, out=int_host[cursor : cursor + identity_size])
        cursor += identity_size
        if compact_values:
            int_host[cursor : cursor + scope_size].copy_(torch.as_tensor(compact_values, dtype=torch.int32))
        cursor += scope_size
        int_host[cursor : cursor + group_inverse_size].fill_(-1)
        for group_row, group_layout in enumerate(layout.group_layouts):
            start = cursor + group_row * layout.num_joints
            for local_slot, joint_id in enumerate(group_layout.joint_indices):
                int_host[start + joint_id] = local_slot
        cursor += group_inverse_size
        int_host[cursor : cursor + layout.num_worlds].fill_(-1)
        cursor += layout.num_worlds
        int_host[cursor : cursor + layout.num_joints].fill_(-1)
        cursor += layout.num_joints
        if owner_values:
            int_host[cursor : cursor + backend_owner_size].copy_(torch.as_tensor(owner_values, dtype=torch.int32))
        cursor += backend_owner_size
        assert cursor == int_size
        self._int_slab = int_host.to(device)
        self._bool_slab = torch.ones(max(layout.num_worlds, layout.num_joints), dtype=torch.bool, device=device)
        self._float_slab = torch.empty(layout.num_worlds * self._max_scope_dofs, dtype=torch.float32, device=device)

        cursor = 0
        self._identity_ids = self._int_slab[cursor : cursor + identity_size]
        self._identity_ids_wp = wp.from_torch(self._identity_ids, dtype=wp.int32)
        cursor += identity_size
        self._all_env_ids = self._identity_ids[: layout.num_worlds]
        self._all_env_ids_wp = wp.from_torch(self._all_env_ids, dtype=wp.int32)
        self._all_joint_ids = self._identity_ids[: layout.num_joints]
        self._all_joint_ids_wp = wp.from_torch(self._all_joint_ids, dtype=wp.int32)
        self._all_env_mask = self._bool_slab[: layout.num_worlds]
        self._all_joint_mask = self._bool_slab[: layout.num_joints]
        self._all_env_mask_wp = wp.from_torch(self._all_env_mask, dtype=wp.bool)
        self._all_joint_mask_wp = wp.from_torch(self._all_joint_mask, dtype=wp.bool)

        self._type_joint_ids: dict[type[ActuatorBase], torch.Tensor] = {}
        self._type_joint_ids_wp: dict[type[ActuatorBase], wp.array] = {}
        self._group_joint_ids: dict[str, torch.Tensor] = {}
        self._group_joint_ids_wp: dict[str, wp.array] = {}
        compact_ids = self._int_slab[cursor : cursor + scope_size]
        cursor += scope_size
        compact_cursor = 0
        for actuator_type, type_layout in layout.type_layouts.items():
            values = compact_ids[compact_cursor : compact_cursor + type_layout.num_dofs]
            compact_cursor += type_layout.num_dofs
            self._type_joint_ids[actuator_type] = values
            self._type_joint_ids_wp[actuator_type] = wp.from_torch(values, dtype=wp.int32)
        for group_layout in layout.group_layouts:
            if group_layout.actuator_type in self._type_joint_ids:
                values = self._type_joint_ids[group_layout.actuator_type][group_layout.type_slice]
            else:
                values = compact_ids[compact_cursor : compact_cursor + len(group_layout.joint_indices)]
                compact_cursor += len(group_layout.joint_indices)
            self._group_joint_ids[group_layout.name] = values
            self._group_joint_ids_wp[group_layout.name] = wp.from_torch(values, dtype=wp.int32)

        inverse_rows = self._int_slab[cursor : cursor + group_inverse_size].view(
            len(layout.group_layouts), layout.num_joints
        )
        cursor += group_inverse_size
        self._group_inverse: dict[str, torch.Tensor] = {}
        self._group_inverse_wp: dict[str, wp.array] = {}
        self._group_name_by_binding: dict[int, str] = {}
        for row, group_layout in enumerate(layout.group_layouts):
            inverse = inverse_rows[row]
            self._group_inverse[group_layout.name] = inverse
            self._group_inverse_wp[group_layout.name] = wp.from_torch(inverse, dtype=wp.int32)

        self._last_env_positions = self._int_slab[cursor : cursor + layout.num_worlds]
        cursor += layout.num_worlds
        self._last_joint_positions = self._int_slab[cursor : cursor + layout.num_joints]
        cursor += layout.num_joints
        self._last_env_positions_wp = wp.from_torch(self._last_env_positions, dtype=wp.int32)
        self._last_joint_positions_wp = wp.from_torch(self._last_joint_positions, dtype=wp.int32)

        owner_rows = self._int_slab[cursor : cursor + backend_owner_size].view(len(owner_row_values), layout.num_joints)
        cursor += backend_owner_size
        self._backend_owner_slots: dict[tuple[type[ActuatorBase], str], torch.Tensor] = {}
        self._active_backend_routes: set[tuple[type[ActuatorBase], str]] = set()
        for route, row in owner_rows_by_policy.items():
            self._backend_owner_slots[route] = owner_rows[row]
            if any(owner >= 0 for owner in owner_row_values[row]):
                self._active_backend_routes.add(route)

        self._alias_staging = self._float_slab.view(layout.num_worlds, self._max_scope_dofs)

    def bind_group(self, binding: _GroupBinding, group_name: str) -> None:
        """Associate one newly created logical group binding with packed metadata."""
        self._group_name_by_binding[id(binding)] = group_name

    def group_joint_ids(self, binding: _GroupBinding) -> torch.Tensor:
        """Return the packed joint-id view for a logical group binding."""
        return self._group_joint_ids[self._group_name_by_binding[id(binding)]]

    def group_joint_ids_wp(self, binding: _GroupBinding) -> wp.array:
        """Return the Warp alias for a logical group joint-id view."""
        return self._group_joint_ids_wp[self._group_name_by_binding[id(binding)]]

    def group_inverse_wp(self, binding: _GroupBinding) -> wp.array:
        """Return the Warp alias for one logical group's inverse row."""
        return self._group_inverse_wp[self._group_name_by_binding[id(binding)]]

    def type_joint_ids(self, actuator_type: type[ActuatorBase]) -> torch.Tensor:
        """Return the packed compact joint-id view for an exact actuator type."""
        return self._type_joint_ids[actuator_type]

    def type_joint_ids_wp(self, actuator_type: type[ActuatorBase]) -> wp.array:
        """Return the Warp alias for an exact actuator type's compact joint ids."""
        return self._type_joint_ids_wp[actuator_type]

    def default_joint_ids(self, count: int) -> torch.Tensor:
        """Return the shared compact identity-id prefix for one scope."""
        return self._identity_ids[:count]

    def alias_staging(self, shape: torch.Size) -> torch.Tensor:
        """Return a bounded shaped view into the preallocated float staging slab."""
        size = 1
        for dimension in shape:
            size *= dimension
        if size > self._float_slab.numel():
            raise ValueError("Aliased parameter values exceed the facade staging capacity.")
        return self._float_slab[:size].view(shape)

    def close(self) -> None:
        """Release all selector-owned slabs and aliases."""
        if self._closed:
            return
        self._closed = True
        self._type_joint_ids.clear()
        self._type_joint_ids_wp.clear()
        self._group_joint_ids.clear()
        self._group_joint_ids_wp.clear()
        self._group_inverse.clear()
        self._group_inverse_wp.clear()
        self._group_name_by_binding.clear()
        self._backend_owner_slots.clear()
        self._active_backend_routes.clear()
        self._identity_ids = None
        self._identity_ids_wp = None
        self._all_env_ids = None
        self._all_env_ids_wp = None
        self._all_joint_ids = None
        self._all_joint_ids_wp = None
        self._all_env_mask = None
        self._all_joint_mask = None
        self._all_env_mask_wp = None
        self._all_joint_mask_wp = None
        self._last_env_positions = None
        self._last_joint_positions = None
        self._last_env_positions_wp = None
        self._last_joint_positions_wp = None
        self._alias_staging = None
        self._int_slab = None
        self._bool_slab = None
        self._float_slab = None


class _CollectionGeneration:
    """Private, reversible collection generation prepared before publication."""

    def __init__(
        self,
        generation: int,
        bindings: tuple[_ArticulationBinding, ...],
        stores: dict[type, _TypedStore],
        joint_store: _JointDomainStore,
        managed_group_resolutions: Mapping[tuple[object, str], _ManagedGroupResolution],
    ) -> None:
        self.generation = generation
        self.bindings = bindings
        self.stores = stores
        self.joint_store = joint_store
        self.solver_store = _SolverPropertyStore()
        self.managed_group_resolutions = dict(managed_group_resolutions)
        self.groups: dict[object, dict[str, ActuatorBase]] = {}
        self.selector_states: dict[object, _SelectorState] = {}
        self.backend_parameter_staging: dict[object, _BackendParameterStaging] = {}
        self._solver_properties_written: list[_ArticulationBinding] = []

    @classmethod
    def build(
        cls,
        registrations: tuple[_ArticulationRegistration, ...],
        sim_context: Any,
        generation: int,
    ) -> _CollectionGeneration:
        clone_plan: ClonePlan | None = sim_context.get_clone_plan()
        if clone_plan is None:
            raise RuntimeError("Actuator collection finalization requires a published clone plan.")
        candidate_device = torch.device(registrations[0].control.device)
        if any(torch.device(registration.control.device) != candidate_device for registration in registrations[1:]):
            raise ValueError("All actuator registrations in one simulation must use the same device.")

        layouts: list[_ArticulationLayout] = []
        managed_group_resolutions: dict[tuple[object, str], _ManagedGroupResolution] = {}
        type_offsets: dict[type[ActuatorBase], int] = {}
        for registration in registrations:
            try:
                source_transport = getattr(
                    registration.control, "resolved_solver_property_transport", lambda: "device"
                )()
                if source_transport not in {"cpu", "device"}:
                    raise ValueError(f"Unsupported solver-property transport {source_transport!r}.")
                clone_metadata = _clone_source_assignment(clone_plan, registration.replication_cfg_id)
                source_slot_by_backend_row: torch.Tensor | None = None
                layout_assignment: torch.Tensor | None = None
                layout_source_columns: torch.Tensor | None = None
                layout_num_worlds: int | None = None
                if clone_metadata is None:
                    prototype_rows = clone_plan.cfg_rows[registration.replication_cfg_id]
                    prototype_row_indices = torch.tensor(
                        prototype_rows,
                        dtype=torch.long,
                        device=clone_plan.clone_mask.device,
                    )
                    source_mask = clone_plan.clone_mask[prototype_row_indices]
                    source_env_ids = torch.argmax(source_mask.to(dtype=torch.int32), dim=1).to(dtype=torch.int64)
                else:
                    if clone_metadata.local_source_slots.device.type != "cpu":
                        raise ValueError("Clone-plan local source slots must be host-resident.")
                    active_source_slots = clone_metadata.local_source_slots[clone_metadata.local_source_slots >= 0]
                    if active_source_slots.shape != (registration.control.num_instances,):
                        raise ValueError(
                            "Clone-plan local source slots must contain one active entry per backend articulation row."
                        )
                    prototype_rows = tuple(int(row) for row in clone_metadata.used_rows.tolist())
                    source_slot_by_backend_row = active_source_slots.to(dtype=torch.int64)
                    backend_rows = torch.arange(registration.control.num_instances, dtype=torch.int64)
                    source_env_ids_cpu = torch.full(
                        (len(prototype_rows),),
                        registration.control.num_instances,
                        dtype=torch.int64,
                    )
                    source_env_ids_cpu.scatter_reduce_(
                        0,
                        source_slot_by_backend_row,
                        backend_rows,
                        reduce="amin",
                        include_self=True,
                    )
                    if torch.any(source_env_ids_cpu == registration.control.num_instances):
                        raise ValueError("Clone-plan source slots omit a compact source row.")
                    layout_num_worlds = registration.control.num_instances
                    layout_assignment = source_slot_by_backend_row.to(
                        device=candidate_device,
                        dtype=torch.int32,
                    )
                    layout_source_columns = source_env_ids_cpu.to(device=candidate_device)
                    source_env_ids = (
                        source_env_ids_cpu
                        if source_transport == "cpu"
                        else source_env_ids_cpu.to(device=candidate_device)
                    )
                groups: list[_GroupRegistration] = []
                source_resolved = hasattr(registration.control, "get_source_joint_properties")
                solver_default_values: dict[str, torch.Tensor] = {}
                source_defaults: ActuatorJointProperties | None = None
                if registration.control.num_joints and source_resolved:
                    all_joint_ids = torch.arange(
                        registration.control.num_joints,
                        dtype=torch.int32,
                        device="cpu" if source_transport == "cpu" else registration.control.device,
                    )
                    source_defaults = registration.control.get_source_joint_properties(all_joint_ids, source_env_ids)
                    solver_default_values = {
                        "stiffness": source_defaults.stiffness,
                        "damping": source_defaults.damping,
                        "effort_limit_sim": source_defaults.effort_limit,
                        "velocity_limit_sim": source_defaults.velocity_limit,
                        "armature": source_defaults.armature,
                        "friction": source_defaults.friction,
                        "dynamic_friction": source_defaults.dynamic_friction,
                        "viscous_friction": source_defaults.viscous_friction,
                    }
                for configuration_index, (name, cfg) in enumerate(registration.cfgs.items()):
                    actuator_type = _resolve_actuator_type(cfg)
                    joint_ids, joint_names = registration.control.find_joints(cfg.joint_names_expr)
                    if not joint_names:
                        raise ValueError(f"{actuator_type.__name__} actuator group {name!r} resolved no joints.")
                    if isinstance(joint_ids, ProxyArray):
                        joint_indices = tuple(int(index) for index in joint_ids.torch.tolist())
                        joint_index_tensor = joint_ids.torch
                    elif isinstance(joint_ids, torch.Tensor):
                        joint_indices = tuple(int(index) for index in joint_ids.tolist())
                        joint_index_tensor = joint_ids
                    else:
                        joint_indices = tuple(int(index) for index in joint_ids)
                        joint_index_tensor = torch.tensor(
                            joint_indices, dtype=torch.int32, device=registration.control.device
                        )
                    source_values: Mapping[str, torch.Tensor] = {}
                    solver_values: Mapping[str, torch.Tensor] = {}
                    resolved = None
                    if actuator_type.__dict__.get("_parameter_schema") is not None and hasattr(
                        registration.control, "get_source_joint_properties"
                    ):
                        public_joint_indices: slice | torch.Tensor
                        if joint_indices == tuple(range(registration.control.num_joints)):
                            public_joint_indices = slice(None)
                        else:
                            public_joint_indices = joint_index_tensor
                        if source_defaults is None:
                            raise RuntimeError("Managed source resolution requires articulation source defaults.")
                        source_joint_ids = torch.tensor(
                            joint_indices,
                            dtype=torch.long,
                            device=source_defaults.stiffness.device,
                        )
                        source_properties = ActuatorJointProperties(
                            stiffness=source_defaults.stiffness.index_select(1, source_joint_ids),
                            damping=source_defaults.damping.index_select(1, source_joint_ids),
                            armature=source_defaults.armature.index_select(1, source_joint_ids),
                            friction=source_defaults.friction.index_select(1, source_joint_ids),
                            dynamic_friction=source_defaults.dynamic_friction.index_select(1, source_joint_ids),
                            viscous_friction=source_defaults.viscous_friction.index_select(1, source_joint_ids),
                            effort_limit=source_defaults.effort_limit.index_select(1, source_joint_ids),
                            velocity_limit=source_defaults.velocity_limit.index_select(1, source_joint_ids),
                        )
                        resolved = actuator_type._resolve_managed_registration(
                            cfg=_copy_actuator_cfg(cfg),
                            joint_names=list(joint_names),
                            joint_indices=public_joint_indices,
                            defaults_by_source=source_properties,
                            debug_value_resolution=registration.debug_value_resolution,
                        )
                        source_values = resolved.source_values
                        is_solver_implicit = (
                            issubclass(actuator_type, ImplicitActuator) and name not in registration.native_group_names
                        )
                        solver_values = {
                            "stiffness": (
                                resolved.source_shell.stiffness
                                if is_solver_implicit
                                else torch.zeros_like(resolved.source_shell.stiffness)
                            ),
                            "damping": (
                                resolved.source_shell.damping
                                if is_solver_implicit
                                else torch.zeros_like(resolved.source_shell.damping)
                            ),
                            "effort_limit_sim": resolved.source_shell.effort_limit_sim,
                            "velocity_limit_sim": resolved.source_shell.velocity_limit_sim,
                            "armature": resolved.source_shell.armature,
                            "friction": resolved.source_shell.friction,
                            "dynamic_friction": resolved.source_shell.dynamic_friction,
                            "viscous_friction": resolved.source_shell.viscous_friction,
                        }
                        managed_group_resolutions[(registration.key, name)] = _ManagedGroupResolution(
                            configuration_index=configuration_index,
                            native_managed=name in registration.native_group_names,
                            source_env_ids=source_env_ids,
                            resolved=resolved,
                            backend_binding_metadata={
                                "actuator_type": actuator_type,
                                "parameter_names": actuator_type._parameter_schema().parameter_names,
                            },
                        )
                    groups.append(
                        _GroupRegistration(
                            name=name,
                            actuator_type=actuator_type,
                            joint_indices=joint_indices,
                            values={},
                            joint_names=tuple(joint_names),
                            source_values=source_values,
                            solver_values=solver_values,
                            resolved=resolved,
                        ),
                    )
                layouts.append(
                    _build_articulation_layout(
                        replication_cfg_id=registration.replication_cfg_id,
                        clone_plan=clone_plan,
                        registrations=(
                            _PrototypeRegistration(
                                registration_key=registration.key,
                                num_joints=registration.control.num_joints,
                                groups=tuple(groups),
                                source_resolved=source_resolved,
                                solver_default_values=solver_default_values,
                            ),
                        ),
                        type_offsets=type_offsets,
                        prototype_rows=prototype_rows,
                        prototype_assignment=layout_assignment,
                        prototype_source_columns=layout_source_columns,
                        source_slot_by_backend_row=source_slot_by_backend_row,
                        num_worlds=layout_num_worlds,
                        clone_plan_metadata=clone_metadata,
                    )
                )
            except Exception as error:
                error.add_note(f"Failed to build actuator candidate for {registration.key!r}.")
                raise

        stores: dict[type, _TypedStore] = {}
        for layout in layouts:
            for actuator_type in layout.type_layouts:
                stores.setdefault(actuator_type, _TypedStore(actuator_type))
        candidate = cls(generation, (), stores, _JointDomainStore(), managed_group_resolutions)
        try:
            for store in stores.values():
                store.allocate(layouts, device=str(candidate_device))
            candidate.joint_store.allocate(layouts, device=str(candidate_device))
            candidate.solver_store.allocate(layouts, device=str(candidate_device))
            candidate.bindings = tuple(
                _ArticulationBinding(registration=registration, layout=layout)
                for registration, layout in zip(registrations, layouts, strict=True)
            )
            candidate.selector_states = {
                binding.registration.key: _SelectorState(binding) for binding in candidate.bindings
            }
        except Exception:
            candidate.close()
            raise
        return candidate

    def validate(self) -> None:
        """Validate candidate bindings without reading public facades."""
        for binding in self.bindings:
            if binding.layout.registration_key is not binding.registration.key:
                raise ValueError(f"Candidate binding for {binding.registration.key!r} has an invalid registration key.")

    def prepare_solver_properties(self) -> None:
        """Materialize final device targets and persistent gain staging after opaque overlays."""
        device_bindings = tuple(
            binding
            for binding in self.bindings
            if getattr(binding.registration.control, "resolved_solver_property_transport", lambda: "device")()
            == "device"
        )
        if device_bindings:
            self.solver_store.materialize_device_targets(
                tuple(binding.layout for binding in device_bindings),
                device=device_bindings[0].registration.control.device,
            )

        updated_bindings: list[_ArticulationBinding] = []
        for binding in self.bindings:
            selector_state = self.selector_states[binding.registration.key]
            active_owner_slots = {
                route: owner_slots
                for route, owner_slots in selector_state._backend_owner_slots.items()
                if (
                    route in selector_state._active_backend_routes
                    and issubclass(route[0], ImplicitActuator)
                    and route[1] in {"stiffness", "damping"}
                )
            }
            staging: _BackendParameterStaging | None = None
            if active_owner_slots:
                staging = _BackendParameterStaging(
                    num_worlds=binding.layout.num_worlds,
                    num_joints=binding.layout.num_joints,
                    device=binding.registration.control.device,
                    owner_slots=active_owner_slots,
                    all_env_ids=selector_state._all_env_ids,
                    all_joint_ids=selector_state._all_joint_ids,
                    all_env_mask=selector_state._all_env_mask,
                    all_joint_mask=selector_state._all_joint_mask,
                )
                for field_name in ("stiffness", "damping"):
                    route = next((route for route in active_owner_slots if route[1] == field_name), None)
                    if route is not None:
                        self.solver_store.expand_source_rows_into(
                            staging.target(route[0], field_name), field_name, binding.layout
                        )
                self.backend_parameter_staging[binding.registration.key] = staging
            updated_bindings.append(replace(binding, backend_parameter_staging=staging))
        self.bindings = tuple(updated_bindings)

    def write_solver_properties(self) -> None:
        """Synchronously offer one resolved articulation rowset to each control."""
        for binding in self.bindings:
            write = getattr(binding.registration.control, "write_resolved_joint_properties_staged", None)
            if write is None:
                # Pre-Task-8 duck-typed test/third-party controls remain valid
                # until their adapters adopt the explicit neutral contract.
                continue
            transport = getattr(binding.registration.control, "resolved_solver_property_transport", lambda: "device")()
            if transport not in {"cpu", "device"}:
                raise ValueError(f"Unsupported solver-property transport {transport!r}.")
            source_slot_by_backend_row = getattr(binding.layout, "source_slot_by_backend_row", None)
            properties: dict[str, _ResolvedSolverProperty] = {}
            for name in self.solver_store._FIELD_NAMES:
                source_rows = self.solver_store.source_rows(name, binding.layout)
                if transport == "cpu":
                    if source_rows.device.type != "cpu" or source_slot_by_backend_row is None:
                        raise RuntimeError(
                            "CPU solver-property transport requires clone-plan-provided CPU source rows and slots."
                        )
                    if source_slot_by_backend_row.device.type != "cpu":
                        raise ValueError("CPU solver-property source slots must be host-resident.")
                    properties[name] = _ResolvedSolverProperty(
                        transport="cpu",
                        source_rows=source_rows,
                        source_slot_by_backend_row=source_slot_by_backend_row,
                        source_assignment=None,
                        canonical_target=None,
                    )
                else:
                    canonical_target = self.solver_store.articulation_proxy(name, binding.layout)
                    if source_rows.device != canonical_target.torch.device:
                        raise RuntimeError(
                            "Device solver-property transport requires source rows on the canonical target device."
                        )
                    properties[name] = _ResolvedSolverProperty(
                        transport="device",
                        source_rows=source_rows,
                        source_slot_by_backend_row=None,
                        source_assignment=binding.layout.prototype_assignment,
                        canonical_target=canonical_target,
                    )
            self._solver_properties_written.append(binding)
            write(
                _ResolvedSolverProperties(properties=properties, clone_plan_metadata=binding.layout.clone_plan_metadata)
            )

    def validate_solver_properties(self) -> None:
        """Run backend validation after synchronous solver writes and before publication."""
        for binding in self._solver_properties_written:
            validate = getattr(binding.registration.control, "validate_resolved_joint_properties", None)
            if validate is not None:
                validate()

    def restore_solver_properties(self) -> None:
        """Restore backend solver defaults after a reversible candidate failure."""
        failures: list[Exception] = []
        try:
            for binding in reversed(self._solver_properties_written):
                try:
                    restore = getattr(binding.registration.control, "restore_resolved_joint_properties", None)
                    if restore is not None:
                        restore()
                except Exception as error:
                    error.add_note(f"Failed to restore solver properties for {_binding_context(binding)}.")
                    failures.append(error)
        finally:
            self._solver_properties_written.clear()
        if failures:
            first, *remaining = failures
            for error in remaining:
                first.add_note(f"Additional solver-property restore failure: {error}")
            raise first

    def commit_solver_properties(self) -> None:
        """Discard build-only solver staging after every control has completed."""
        try:
            for binding in self._solver_properties_written:
                try:
                    commit = getattr(binding.registration.control, "commit_resolved_joint_properties", None)
                    if commit is not None:
                        commit()
                except Exception:
                    logger.exception("Failed to commit solver properties for %s.", _binding_context(binding))
        finally:
            self._solver_properties_written.clear()
            self.managed_group_resolutions.clear()
            self.solver_store.close()
            for store in self.stores.values():
                store._release_initialization_buffers()

    def bind_facade_storage(self) -> None:
        """Construct exact logical groups and bind candidate-owned typed arrays."""
        for binding in self.bindings:
            groups: dict[str, ActuatorBase] = {}
            control = binding.registration.control
            selector_state = self.selector_states[binding.registration.key]
            for (name, source_cfg), group_layout in zip(
                binding.registration.cfgs.items(), binding.layout.group_layouts, strict=True
            ):
                joint_indices = selector_state._group_joint_ids[name]
                joint_names = group_layout.joint_names
                public_joint_indices: slice | torch.Tensor
                if group_layout.joint_indices == tuple(range(binding.layout.num_joints)):
                    public_joint_indices = slice(None)
                else:
                    public_joint_indices = joint_indices
                cfg = _copy_actuator_cfg(source_cfg)
                managed_resolution = self.managed_group_resolutions.get((binding.registration.key, name))
                store = self.stores.get(group_layout.actuator_type)
                parameter_binding = None
                if store is not None:
                    schema = group_layout.actuator_type._parameter_schema()
                    group_proxies = {field.name: store.group_proxy(group_layout, field.name) for field in schema.fields}
                    type_layout = binding.layout.type_layouts[group_layout.actuator_type]
                    parameter_binding = _GroupBinding(
                        generation=self.generation,
                        joint_indices=joint_indices,
                        joint_names=tuple(joint_names),
                        type_slice=group_layout.type_slice,
                        arrays={field.name: store.type_proxy(type_layout, field.name) for field in schema.fields},
                        parameter_proxies={
                            field.name: group_proxies[field.name]
                            for field in schema.fields
                            if field.role == "parameter"
                        },
                    )
                if managed_resolution is not None:
                    if parameter_binding is None:
                        raise RuntimeError(f"Managed actuator group {name!r} has no canonical parameter binding.")
                    actuator = group_layout.actuator_type._build_managed_runtime_shell(
                        resolved=managed_resolution.resolved,
                        binding=parameter_binding,
                        num_envs=group_layout.num_worlds,
                        device=control.device,
                        joint_indices=public_joint_indices,
                    )
                elif hasattr(control, "get_default_joint_properties") and hasattr(cfg, "effort_limit"):
                    defaults = control.get_default_joint_properties(joint_indices)
                    actuator = group_layout.actuator_type(
                        cfg=cfg,
                        joint_names=list(joint_names),
                        joint_ids=public_joint_indices,
                        num_envs=group_layout.num_worlds,
                        device=control.device,
                        stiffness=defaults.stiffness,
                        damping=defaults.damping,
                        armature=defaults.armature,
                        friction=defaults.friction,
                        dynamic_friction=defaults.dynamic_friction,
                        viscous_friction=defaults.viscous_friction,
                        effort_limit=defaults.effort_limit,
                        velocity_limit=defaults.velocity_limit,
                    )
                else:
                    actuator = object.__new__(group_layout.actuator_type)
                    actuator.cfg = cfg
                    actuator._num_envs = group_layout.num_worlds
                    actuator._device = control.device
                    actuator._joint_names = list(joint_names)
                    actuator._joint_indices = public_joint_indices

                if parameter_binding is not None:
                    if managed_resolution is None:
                        for field in schema.fields:
                            existing = actuator.__dict__.get(field.name)
                            if isinstance(existing, torch.Tensor):
                                group_proxies[field.name].torch.copy_(existing)
                        actuator._bind_parameter_storage(parameter_binding)
                    selector_state.bind_group(parameter_binding, name)
                groups[name] = actuator
            self.groups[binding.registration.key] = groups
            self.solver_store.overlay_opaque_groups(binding.layout, groups)
            for group_layout in binding.layout.group_layouts:
                actuator = groups[group_layout.name]
                if "_parameter_binding" not in actuator.__dict__:
                    continue
                source_joint_ids = torch.tensor(
                    group_layout.joint_indices,
                    dtype=torch.long,
                    device=self.solver_store.source_rows("armature", binding.layout).device,
                )
                actuator._bind_solver_compatibility_seeds(
                    {
                        field_name: _SolverCompatibilitySeed(
                            source_rows=self.solver_store.source_rows(field_name, binding.layout),
                            source_assignment=binding.layout.prototype_assignment,
                            source_joint_indices=source_joint_ids,
                        )
                        for field_name in actuator._SOLVER_COMPATIBILITY_PARAMETER_NAMES
                    }
                )
                actuator._release_managed_source_parameters()

    def publish_effort_telemetry(self) -> None:
        """Publish exact-type effort outputs in stable articulation joint order."""
        for binding in self.bindings:
            layout = binding.layout
            self.joint_store.articulation_proxy("computed_effort", layout).warp.fill_(0.0)
            self.joint_store.articulation_proxy("applied_effort", layout).warp.fill_(0.0)

        for binding in self.bindings:
            layout = binding.layout
            selector_state = self.selector_states[binding.registration.key]
            target_computed_effort = self.joint_store.articulation_proxy("computed_effort", layout)
            target_applied_effort = self.joint_store.articulation_proxy("applied_effort", layout)
            for actuator_type, type_layout in layout.type_layouts.items():
                store = self.stores[actuator_type]
                if "computed_effort" not in store._fields or "applied_effort" not in store._fields:
                    continue
                wp.launch(
                    actuator_kernels.scatter_type_effort_telemetry,
                    dim=(layout.num_worlds, type_layout.num_dofs),
                    inputs=[
                        store.type_proxy(type_layout, "computed_effort").warp,
                        store.type_proxy(type_layout, "applied_effort").warp,
                        selector_state.type_joint_ids_wp(actuator_type),
                    ],
                    outputs=[target_computed_effort.warp, target_applied_effort.warp],
                    device=target_computed_effort.warp.device,
                )

    def close(self) -> None:
        """Release every candidate-owned allocation and private reference."""
        for groups in self.groups.values():
            for group in groups.values():
                release = getattr(group, "_release_facade_storage", None)
                if release is not None:
                    release()
        for selector_state in self.selector_states.values():
            selector_state.close()
        self.selector_states.clear()
        for staging in self.backend_parameter_staging.values():
            staging.close()
        self.backend_parameter_staging.clear()
        self.solver_store.close()
        self.joint_store.close()
        for store in self.stores.values():
            store._release_initialization_buffers()
            store._fields.clear()
            store._type_proxies.clear()
            store._group_proxies.clear()
            store._mapping_proxies.clear()
        self.stores.clear()
        self.groups.clear()
        self.managed_group_resolutions.clear()
        self._solver_properties_written.clear()


def _binding_context(binding: _ArticulationBinding) -> str:
    """Describe the articulation groups involved in a binding failure."""
    groups = ", ".join(f"{group.name} ({group.actuator_type.__name__})" for group in binding.layout.group_layouts)
    return f"articulation {binding.registration.key!r}; actuator groups: {groups or '<none>'}"


class ActuatorCollection(Mapping[str, ActuatorBase]):
    """Simulation-scoped actuator registration manager.

    ``ActuatorCollection(sim_context)`` creates the lifecycle manager used by
    :class:`~isaaclab.sim.SimulationContext`.  The legacy two-argument
    constructor remains temporarily available for develop compatibility while
    backend integration is completed in a later task.

    The collection owns actuator command buffers, processed joint command buffers,
    actuator telemetry, and actuator-resolved gain/state buffers. Named mapping
    entries are stable logical configuration and access groups, and membership is
    fixed after construction. Compatible groups whose concrete type is the same
    supported stateless actuator class may share a private execution actuator while
    retaining their separate per-joint parameters and group-shaped public values.
    Execution batches are an implementation detail, and users must not depend on
    their count.

    The collection owns lifecycle execution for its managed groups. Calling
    :meth:`~isaaclab.actuators.ActuatorBase.compute` or
    :meth:`~isaaclab.actuators.ActuatorBase.reset` directly on a mapping value is
    unsupported.
    """

    class _GuardedMapping(Mapping[Any, Any]):
        """Read-only mapping whose owner validates each access."""

        def __init__(self, owner: Any, values: Mapping[Any, Any], token: object | None = None) -> None:
            self._owner = owner
            self._values = values
            self._token = token

        def _require_current_generation(self) -> None:
            if self._token is None:
                self._owner._require_current_generation()
            else:
                self._owner._require_current_generation(self._token)

        def __getitem__(self, key: Any) -> Any:
            self._require_current_generation()
            return self._values[key]

        def __iter__(self) -> Iterator[Any]:
            self._require_current_generation()
            return _GuardedIterator(self._require_current_generation, iter(self._values))

        def __reversed__(self) -> Iterator[Any]:
            self._require_current_generation()
            return _GuardedIterator(self._require_current_generation, reversed(self._values))

        def __len__(self) -> int:
            self._require_current_generation()
            return len(self._values)

        def __repr__(self) -> str:
            self._require_current_generation()
            return repr(self._values)

        def copy(self) -> dict[Any, Any]:
            """Return an ordinary dictionary snapshot of the guarded mapping."""
            self._require_current_generation()
            return dict(self._values)

        def keys(self) -> _GuardedKeysView:
            self._require_current_generation()
            return _GuardedKeysView(self, self._require_current_generation)

        def items(self) -> _GuardedItemsView:
            self._require_current_generation()
            return _GuardedItemsView(self, self._require_current_generation)

        def values(self) -> _GuardedValuesView:
            self._require_current_generation()
            return _GuardedValuesView(self, self._require_current_generation)

    class TypeView:
        """Compact exact-class actuator view for one articulation generation."""

        def __init__(
            self,
            facade: ActuatorCollection.ArticulationView,
            actuator_type: type[ActuatorBase],
            binding: _ArticulationBinding,
            store: _TypedStore,
            groups: Mapping[str, ActuatorBase],
            selector_state: _SelectorState,
        ) -> None:
            self._facade = facade
            self._install_token = selector_state.token
            self._actuator_type = actuator_type
            type_layout = binding.layout.type_layouts[actuator_type]
            group_layouts = tuple(
                group for group in binding.layout.group_layouts if group.actuator_type is actuator_type
            )
            self._joint_names = tuple(
                joint_name for group in group_layouts for joint_name in groups[group.name].__dict__["_joint_names"]
            )
            self._joint_indices = selector_state.type_joint_ids(actuator_type)
            self._csr_offsets, self._csr_slots = store.mapping_proxies(type_layout)
            self._max_csr_fanout = max(
                end - start
                for start, end in zip(
                    type_layout.articulation_to_compact_offsets,
                    type_layout.articulation_to_compact_offsets[1:],
                )
            )
            self._group_slices = {group.name: group.type_slice for group in group_layouts}
            self._num_instances = type_layout.num_worlds
            schema = actuator_type._parameter_schema()
            parameter_values = {
                field.name: store.type_proxy(type_layout, field.name)
                for field in schema.fields
                if field.role == "parameter"
            }
            self._parameters = ActuatorCollection._GuardedMapping(self, parameter_values)

        def _require_current_generation(self) -> None:
            self._facade._require_current_generation(self._install_token)

        def _release(self) -> None:
            """Drop canonical aliases while preserving an ABA-safe stale guard."""
            self._parameters._values.clear()
            self._joint_indices = None
            self._csr_offsets = None
            self._csr_slots = None

        @property
        def actuator_type(self) -> type[ActuatorBase]:
            """Exact managed actuator class represented by this view."""
            self._require_current_generation()
            return self._actuator_type

        @property
        def num_instances(self) -> int:
            """Number of articulation instances represented by this view."""
            self._require_current_generation()
            return self._num_instances

        @property
        def num_joints(self) -> int:
            """Number of compact actuator DOF occurrences represented by this view."""
            self._require_current_generation()
            return len(self._joint_names)

        @property
        def joint_names(self) -> tuple[str, ...]:
            """Compact joint names in group and configuration order."""
            self._require_current_generation()
            return self._joint_names

        @property
        def joint_indices(self) -> torch.Tensor:
            """Articulation joint indices for compact DOF occurrences."""
            self._require_current_generation()
            return self._joint_indices

        @property
        def group_slices(self) -> dict[str, slice]:
            """Compact column slices keyed by logical actuator group."""
            self._require_current_generation()
            return dict(self._group_slices)

        @property
        def parameter_names(self) -> tuple[str, ...]:
            """Managed parameter names in exact-schema declaration order."""
            self._require_current_generation()
            return tuple(self._parameters._values)

        @property
        def parameters(self) -> Mapping[str, ProxyArray]:
            """Contiguous mutable parameter arrays exposed through a read-only mapping."""
            self._require_current_generation()
            return self._parameters

        def set_parameter_index(
            self,
            name: str,
            value: float | torch.Tensor | wp.array(dtype=wp.float32) | Sequence[float] | Sequence[Sequence[float]],
            *,
            env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | wp.array(dtype=wp.int64) | None = None,
            joint_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | wp.array(dtype=wp.int64) | None = None,
        ) -> None:
            """Set one parameter using Cartesian articulation index selectors.

            Args:
                name: Exact-schema parameter name.
                value: A scalar; compact values with shape
                    ``[len(joint_ids)]``; or Cartesian values with shape
                    ``[len(env_ids), len(joint_ids)]``. When
                    :paramref:`joint_ids` is ``None``, its length is this exact
                    type's compact DOF count; when :paramref:`env_ids` is
                    ``None``, its length is ``num_worlds``. Units follow
                    :paramref:`name`:
                    stiffness [N/m or N·m/rad], damping [N·s/m or N·m·s/rad],
                    effort and saturation limits [N or N·m], and velocity limits
                    [m/s or rad/s], depending on joint type.
                env_ids: Signed articulation-world indices, or ``None`` for every
                    world. Shape ``[num_selected_worlds]``.
                joint_ids: Signed articulation-DOF indices, or ``None`` for every
                    compact type slot. Shape ``[num_selected_joints]``.

            The selected Cartesian pairs retain their supplied value rows and
            columns: filtering out-of-range worlds, out-of-range joints, or
            joints outside this exact type's scope does not shift source columns.
            In normal mode, duplicate environment or joint IDs use the last
            Cartesian occurrence. Explicit :paramref:`joint_ids` fan a supplied
            value out to every overlapping compact slot for that articulation
            DOF. ``joint_ids=None`` addresses compact type slots individually in
            stable configuration order. With debug validation enabled, bounds,
            duplicates, and ownership violations synchronously raise instead.

            Raises:
                KeyError: If :paramref:`name` is not an exact-schema parameter.
                TypeError: If selector or value dtypes are unsupported.
                ValueError: If metadata is malformed, values cannot broadcast, or
                    an overlapping value source exceeds the bounded staging
                    capacity, or debug validation rejects selector contents.
                RuntimeError: If this view is stale or not execution-ready.
            """
            self._require_current_generation()
            self._facade._write_scoped_parameter_index(
                scope="type",
                name=name,
                value=value,
                target=self._parameters.get(name),
                scope_joint_ids=self._joint_indices,
                csr_offsets=self._csr_offsets,
                csr_slots=self._csr_slots,
                max_csr_fanout=self._max_csr_fanout,
                actuator_type=self._actuator_type,
                group_binding=None,
                group_inverse=None,
                env_ids=env_ids,
                joint_ids=joint_ids,
            )

        def set_parameter_mask(
            self,
            name: str,
            value: float | torch.Tensor | wp.array(dtype=wp.float32) | Sequence[float] | Sequence[Sequence[float]],
            *,
            env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
        ) -> None:
            """Set one parameter using full-articulation masks and compact values.

            Args:
                name: Exact-schema parameter name.
                value: A scalar; compact values with shape ``[num_scope_dofs]``;
                    or world-by-compact values with shape
                    ``[num_worlds, num_scope_dofs]``. Units follow
                    :paramref:`name`: stiffness [N/m or N·m/rad], damping
                    [N·s/m or N·m·s/rad], effort and saturation limits [N or N·m],
                    and velocity limits [m/s or rad/s], depending on joint type.
                env_mask: Boolean articulation-world mask with shape
                    ``[num_worlds]``, or ``None`` for every world.
                joint_mask: Boolean articulation-DOF mask with shape
                    ``[num_joints]``, or ``None`` for every joint.

            Values are indexed by stable compact type slots, not by the count of
            ``True`` entries in :paramref:`joint_mask`. Every overlapping compact
            slot therefore remains eligible and may receive a distinct compact
            value. The masks select full articulation domains; entries outside
            this exact type's scope are ignored in every mode. Debug validation
            performs value-dependent bounds, ownership, and duplicate checks only
            for index selectors.

            Raises:
                KeyError: If :paramref:`name` is not an exact-schema parameter.
                TypeError: If selector or value dtypes are unsupported.
                ValueError: If metadata is malformed, values cannot broadcast, or
                    an overlapping value source exceeds the bounded staging
                    capacity.
                RuntimeError: If this view is stale or not execution-ready.
            """
            self._require_current_generation()
            self._facade._write_scoped_parameter_mask(
                scope="type",
                name=name,
                value=value,
                target=self._parameters.get(name),
                scope_joint_ids=self._joint_indices,
                actuator_type=self._actuator_type,
                group_binding=None,
                csr_offsets=self._csr_offsets,
                csr_slots=self._csr_slots,
                env_mask=env_mask,
                joint_mask=joint_mask,
            )

    class ArticulationView(dict[str, ActuatorBase]):
        """Guarded dictionary facade owned by one collection generation."""

        _MISSING = object()
        fromkeys = _GuardedDictFromKeys()

        class Command:
            """Raw commands received by the actuator models.

            Position and velocity commands use joint-side coordinates. All
            arrays are indexed in articulation joint order.
            """

            def __init__(
                self,
                view: ActuatorCollection.ArticulationView,
                install_token: object,
                position: ProxyArray,
                velocity: ProxyArray,
                effort: ProxyArray,
            ) -> None:
                self._view = view
                self._install_token = install_token
                self._position: ProxyArray | None = position
                self._velocity: ProxyArray | None = velocity
                self._effort: ProxyArray | None = effort

            def _require_current_generation(self) -> None:
                self._view._require_current_generation(self._install_token)
                if self._view._manager._dirty:
                    raise RuntimeError("late registration requires STOP-to-READY rebuild before actuator access.")

            @property
            def position(self) -> ProxyArray:
                """Desired positions [m or rad, depending on joint type]."""
                self._require_current_generation()
                if self._position is None:
                    raise RuntimeError("stale actuator view")
                return self._position

            @property
            def velocity(self) -> ProxyArray:
                """Desired velocities [m/s or rad/s, depending on joint type]."""
                self._require_current_generation()
                if self._velocity is None:
                    raise RuntimeError("stale actuator view")
                return self._velocity

            @property
            def effort(self) -> ProxyArray:
                """Effort commands [N or N·m, depending on joint type]."""
                self._require_current_generation()
                if self._effort is None:
                    raise RuntimeError("stale actuator view")
                return self._effort

            def set_position_index(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                env_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                full_data: bool = False,
            ) -> None:
                """Set desired positions using articulation index selectors.

                Args:
                    value: Desired positions [m or rad, depending on joint type].
                    joint_ids: Joint indices. Defaults to all joints.
                    env_ids: Environment indices. Defaults to all environments.
                    full_data: Whether :paramref:`value` has full articulation shape.
                """
                self._view._write_command_index(
                    value, env_ids, joint_ids, self.position, command_name="position", full_data=full_data
                )

            def set_velocity_index(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                env_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                full_data: bool = False,
            ) -> None:
                """Set desired velocities using articulation index selectors.

                Args:
                    value: Desired velocities [m/s or rad/s, depending on joint type].
                    joint_ids: Joint indices. Defaults to all joints.
                    env_ids: Environment indices. Defaults to all environments.
                    full_data: Whether :paramref:`value` has full articulation shape.
                """
                self._view._write_command_index(
                    value, env_ids, joint_ids, self.velocity, command_name="velocity", full_data=full_data
                )

            def set_effort_index(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                env_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                full_data: bool = False,
            ) -> None:
                """Set effort commands using articulation index selectors.

                Args:
                    value: Effort commands [N or N·m, depending on joint type].
                    joint_ids: Joint indices. Defaults to all joints.
                    env_ids: Environment indices. Defaults to all environments.
                    full_data: Whether :paramref:`value` has full articulation shape.
                """
                self._view._write_command_index(
                    value, env_ids, joint_ids, self.effort, command_name="effort", full_data=full_data
                )

            def set_position_mask(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
                env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            ) -> None:
                """Set desired positions using articulation masks.

                Args:
                    value: Full position commands [m or rad, depending on joint type].
                    joint_mask: Joint selection mask. Defaults to all joints.
                    env_mask: Environment selection mask. Defaults to all environments.
                """
                self._view._write_command_mask(value, env_mask, joint_mask, self.position, command_name="position")

            def set_velocity_mask(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
                env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            ) -> None:
                """Set desired velocities using articulation masks.

                Args:
                    value: Full velocity commands [m/s or rad/s, depending on joint type].
                    joint_mask: Joint selection mask. Defaults to all joints.
                    env_mask: Environment selection mask. Defaults to all environments.
                """
                self._view._write_command_mask(value, env_mask, joint_mask, self.velocity, command_name="velocity")

            def set_effort_mask(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
                env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            ) -> None:
                """Set effort commands using articulation masks.

                Args:
                    value: Full effort commands [N or N·m, depending on joint type].
                    joint_mask: Joint selection mask. Defaults to all joints.
                    env_mask: Environment selection mask. Defaults to all environments.
                """
                self._view._write_command_mask(value, env_mask, joint_mask, self.effort, command_name="effort")

            def _close(self) -> None:
                """Release every facade-owned command alias."""
                self._position = None
                self._velocity = None
                self._effort = None

        class JointCommand:
            """Processed commands produced for the simulated joints."""

            def __init__(
                self,
                view: ActuatorCollection.ArticulationView,
                install_token: object,
                position: ProxyArray,
                velocity: ProxyArray,
                effort: ProxyArray,
            ) -> None:
                self._view = view
                self._install_token = install_token
                self._position: ProxyArray | None = position
                self._velocity: ProxyArray | None = velocity
                self._effort: ProxyArray | None = effort

            def _require_current_generation(self) -> None:
                self._view._require_current_generation(self._install_token)
                if self._view._manager._dirty:
                    raise RuntimeError("late registration requires STOP-to-READY rebuild before actuator access.")

            @property
            def position(self) -> ProxyArray:
                """Processed positions [m or rad, depending on joint type]."""
                self._require_current_generation()
                if self._position is None:
                    raise RuntimeError("stale actuator view")
                return self._position

            @property
            def velocity(self) -> ProxyArray:
                """Processed velocities [m/s or rad/s, depending on joint type]."""
                self._require_current_generation()
                if self._velocity is None:
                    raise RuntimeError("stale actuator view")
                return self._velocity

            @property
            def effort(self) -> ProxyArray:
                """Processed efforts [N or N·m, depending on joint type]."""
                self._require_current_generation()
                if self._effort is None:
                    raise RuntimeError("stale actuator view")
                return self._effort

            def _close(self) -> None:
                """Release every facade-owned processed-command alias."""
                self._position = None
                self._velocity = None
                self._effort = None

        def __init__(self, manager: ActuatorCollection, key: object) -> None:
            dict.__init__(self)
            self._manager = manager
            self._key = key
            self._generation: int | None = None
            self._failure: str = "pending actuator view"
            self._by_type: Mapping[type[ActuatorBase], ActuatorCollection.TypeView] = {}
            self._selector_state: _SelectorState | None = None
            self._install_token: object | None = None
            self._command: ActuatorCollection.ArticulationView.Command | None = None
            self._joint_command: ActuatorCollection.ArticulationView.JointCommand | None = None
            self._computed_effort: ProxyArray | None = None
            self._applied_effort: ProxyArray | None = None
            self._backend_parameter_staging: _BackendParameterStaging | None = None

        @property
        def generation(self) -> int:
            """Published collection generation for this view."""
            self._require_current_generation()
            assert self._generation is not None
            return self._generation

        @property
        def is_ready(self) -> bool:
            """Whether this view belongs to the active generation."""
            return (
                self._generation is not None
                and self._manager.generation == self._generation
                and self._manager.is_finalized
            )

        @property
        def by_type(self) -> Mapping[type[ActuatorBase], ActuatorCollection.TypeView]:
            """Read-only exact managed class views for this articulation."""
            self._require_current_generation()
            return self._by_type

        @property
        def command(self) -> Command:
            """Raw commands received by the actuator models."""
            self._require_bindable()
            if self._command is None:
                raise RuntimeError("Actuator command storage is not available before the scoped facade is installed.")
            return self._command

        @property
        def joint_command(self) -> JointCommand:
            """Processed commands produced for the simulated joints."""
            self._require_bindable()
            if self._joint_command is None:
                raise RuntimeError("Actuator command storage is not available before the scoped facade is installed.")
            return self._joint_command

        @property
        def computed_effort(self) -> ProxyArray:
            """Computed effort [N or N·m, depending on joint type]."""
            self._require_bindable()
            if self._computed_effort is None:
                raise RuntimeError("Actuator telemetry storage is not available before the scoped facade is installed.")
            return self._computed_effort

        @property
        def applied_effort(self) -> ProxyArray:
            """Applied effort [N or N·m, depending on joint type]."""
            self._require_bindable()
            if self._applied_effort is None:
                raise RuntimeError("Actuator telemetry storage is not available before the scoped facade is installed.")
            return self._applied_effort

        def compute(self, dt: float = 0.0) -> None:
            """Reject execution while a topology mutation requires a safe rebuild."""
            del dt
            self._require_execution_ready()

        def __getitem__(self, name: str) -> ActuatorBase:
            self._require_current_generation()
            return dict.__getitem__(self, name)

        def __iter__(self) -> Iterator[str]:
            self._require_current_generation()
            return _GuardedIterator(self._require_current_generation, dict.__iter__(self))

        def __len__(self) -> int:
            self._require_current_generation()
            return dict.__len__(self)

        def __contains__(self, name: object) -> bool:
            self._require_current_generation()
            return dict.__contains__(self, name)

        def __repr__(self) -> str:
            self._require_current_generation()
            return dict.__repr__(self)

        def __reversed__(self) -> Iterator[str]:
            self._require_current_generation()
            return _GuardedIterator(self._require_current_generation, dict.__reversed__(self))

        def __or__(self, other: dict) -> dict:
            self._require_current_generation()
            return dict.__or__(self, other)

        def __ror__(self, other: dict) -> dict:
            self._require_current_generation()
            return dict.__ror__(self, other)

        def __eq__(self, other: object) -> bool:
            self._require_current_generation()
            return dict.__eq__(self, other)

        def __ne__(self, other: object) -> bool:
            self._require_current_generation()
            return dict.__ne__(self, other)

        def keys(self) -> _GuardedKeysView:
            self._require_current_generation()
            return _GuardedKeysView(self, self._require_current_generation)

        def items(self) -> _GuardedItemsView:
            self._require_current_generation()
            return _GuardedItemsView(self, self._require_current_generation)

        def values(self) -> _GuardedValuesView:
            self._require_current_generation()
            return _GuardedValuesView(self, self._require_current_generation)

        def get(self, name: str, default: Any = None) -> Any:
            self._require_current_generation()
            return dict.get(self, name, default)

        def copy(self) -> dict[str, ActuatorBase]:
            self._require_current_generation()
            return dict.copy(self)

        def __setitem__(self, name: str, actuator: ActuatorBase) -> None:
            self._require_current_generation()
            candidate = self._topology_snapshot()
            dict.__setitem__(candidate, name, actuator)
            self._commit_topology(candidate, "setitem", name, actuator)

        def __delitem__(self, name: str) -> None:
            self._require_current_generation()
            candidate = self._topology_snapshot()
            dict.__delitem__(candidate, name)
            self._commit_topology(candidate, "delitem", name, None)

        def clear(self) -> None:
            self._require_current_generation()
            candidate = self._topology_snapshot()
            dict.clear(candidate)
            self._commit_topology(candidate, "clear", "", None)

        def pop(self, name: str, default: Any = _MISSING) -> Any:
            self._require_current_generation()
            candidate = self._topology_snapshot()
            if default is self._MISSING:
                value = dict.pop(candidate, name)
            else:
                value = dict.pop(candidate, name, default)
            self._commit_topology(candidate, "pop", name, None)
            return value

        def popitem(self) -> tuple[str, ActuatorBase]:
            self._require_current_generation()
            candidate = self._topology_snapshot()
            value = dict.popitem(candidate)
            self._commit_topology(candidate, "popitem", value[0], None)
            return value

        def setdefault(self, name: str, default: ActuatorBase | None = None) -> ActuatorBase | None:
            self._require_current_generation()
            candidate = self._topology_snapshot()
            value = dict.setdefault(candidate, name, default)
            self._commit_topology(candidate, "setdefault", name, value)
            return value

        def update(self, *args, **kwargs) -> None:
            self._require_current_generation()
            candidate = self._topology_snapshot()
            dict.update(candidate, *args, **kwargs)
            self._commit_topology(candidate, "update", "", None)

        def __ior__(self, other: Mapping[str, ActuatorBase]):
            self._require_current_generation()
            candidate = self._topology_snapshot()
            dict.__ior__(candidate, other)
            self._commit_topology(candidate, "ior", "", None)
            return self

        def _install(
            self,
            generation: _CollectionGeneration,
            binding: _ArticulationBinding,
        ) -> None:
            groups = generation.groups[binding.registration.key]
            selector_state = generation.selector_states[binding.registration.key]
            self._selector_state = selector_state
            self._install_token = selector_state.token
            self._command = self.Command(
                self,
                selector_state.token,
                generation.joint_store.articulation_proxy("raw_position", binding.layout),
                generation.joint_store.articulation_proxy("raw_velocity", binding.layout),
                generation.joint_store.articulation_proxy("raw_effort", binding.layout),
            )
            self._joint_command = self.JointCommand(
                self,
                selector_state.token,
                generation.joint_store.articulation_proxy("processed_position", binding.layout),
                generation.joint_store.articulation_proxy("processed_velocity", binding.layout),
                generation.joint_store.articulation_proxy("processed_effort", binding.layout),
            )
            self._computed_effort = generation.joint_store.articulation_proxy("computed_effort", binding.layout)
            self._applied_effort = generation.joint_store.articulation_proxy("applied_effort", binding.layout)
            self._backend_parameter_staging = binding.backend_parameter_staging
            dict.__init__(self, groups)
            for group in groups.values():
                group._bind_facade_view(self, selector_state.token)
            type_views = {
                actuator_type: ActuatorCollection.TypeView(
                    self, actuator_type, binding, generation.stores[actuator_type], groups, selector_state
                )
                for actuator_type in binding.layout.type_layouts
            }
            self._by_type = ActuatorCollection._GuardedMapping(self, type_views, selector_state.token)
            self._debug_validation = binding.registration.debug_validation
            self._control = binding.registration.control
            self._native_group_binding_ids = {
                id(groups[name].__dict__["_parameter_binding"])
                for name in binding.registration.native_group_names
                if name in groups and "_parameter_binding" in groups[name].__dict__
            }

        @staticmethod
        def _as_command_value(value: torch.Tensor | wp.array, target: ProxyArray) -> torch.Tensor | wp.array:
            """Validate a command value without copying same-device tensor data."""
            if isinstance(value, ProxyArray):
                value = value.torch
            if isinstance(value, wp.array):
                if value.device != target.warp.device:
                    raise ValueError("Command values must be on the scoped articulation device.")
                if value.dtype is not wp.float32:
                    raise TypeError("Command values must have float32 dtype.")
                return value
            elif not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=torch.float32, device=target.torch.device)
            if not isinstance(value, torch.Tensor):
                raise TypeError("Command values must be float32 Torch or Warp arrays.")
            if value.device != target.torch.device:
                raise ValueError("Command values must be on the scoped articulation device.")
            if value.dtype is not torch.float32:
                raise TypeError("Command values must have float32 dtype.")
            return value

        @staticmethod
        def _as_command_index_selector(
            value: Sequence[int] | torch.Tensor | wp.array, target: torch.Tensor, name: str
        ) -> torch.Tensor | wp.array:
            """Validate one command index selector without converting Warp storage."""
            if isinstance(value, ProxyArray):
                value = value.torch
            if isinstance(value, wp.array):
                if len(value.shape) != 1:
                    raise ValueError(f"{name} must be one-dimensional.")
                if str(value.device) != str(target.device):
                    raise ValueError(f"{name} must be on the scoped articulation device.")
                if value.dtype not in (wp.int32, wp.int64):
                    raise TypeError(f"{name} must have a signed integer dtype.")
                return value
            return ActuatorCollection.ArticulationView._as_index_torch(value, target, name)

        @staticmethod
        def _as_command_mask_selector(
            value: torch.Tensor | wp.array, target: torch.Tensor, name: str
        ) -> torch.Tensor | wp.array:
            """Validate one command mask without converting Warp storage."""
            if isinstance(value, ProxyArray):
                value = value.torch
            if isinstance(value, wp.array):
                if value.shape != tuple(target.shape):
                    raise ValueError(f"{name} must have shape {tuple(target.shape)}.")
                if str(value.device) != str(target.device):
                    raise ValueError(f"{name} must be on the scoped articulation device.")
                if value.dtype is not wp.bool:
                    raise TypeError(f"{name} must have boolean dtype.")
                return value
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional boolean tensor.")
            if value.device != target.device:
                raise ValueError(f"{name} must be on the scoped articulation device.")
            if value.dtype is not torch.bool:
                raise TypeError(f"{name} must have boolean dtype.")
            if value.shape != target.shape:
                raise ValueError(f"{name} must have shape {tuple(target.shape)}.")
            return value

        def _write_command_index(
            self,
            value: torch.Tensor | wp.array,
            env_ids: Sequence[int] | torch.Tensor | wp.array | None,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None,
            target: ProxyArray,
            *,
            command_name: str,
            full_data: bool,
        ) -> None:
            """Validate and write one raw command with index selectors."""
            self._require_execution_ready()
            selector_state = self._require_selector_state()
            target_proxy = target
            env_selector = (
                selector_state._all_env_ids
                if env_ids is None
                else self._as_command_index_selector(env_ids, selector_state._all_env_ids, "env_ids")
            )
            joint_selector = (
                selector_state.default_joint_ids(selector_state._all_joint_mask.shape[0])
                if joint_ids is None
                else self._as_command_index_selector(joint_ids, selector_state._all_joint_mask, "joint_ids")
            )
            source = self._as_command_value(value, target_proxy)
            expected_shape = (
                (selector_state._all_env_ids.shape[0], selector_state._all_joint_mask.shape[0])
                if full_data
                else (env_selector.shape[0], joint_selector.shape[0])
            )
            if tuple(source.shape) != expected_shape:
                raise ValueError(f"Command index values must have shape {expected_shape}.")
            if env_selector.shape[0] == 0 or joint_selector.shape[0] == 0:
                self._stage_user_command(command_name, env_selector, joint_selector, None, None)
                return
            env_selector_wp = (
                selector_state._all_env_ids_wp
                if env_ids is None
                else (
                    env_selector
                    if isinstance(env_selector, wp.array)
                    else wp.from_torch(env_selector, dtype=wp.int32 if env_selector.dtype is torch.int32 else wp.int64)
                )
            )
            joint_selector_wp = (
                selector_state._identity_ids_wp
                if joint_ids is None
                else (
                    joint_selector
                    if isinstance(joint_selector, wp.array)
                    else wp.from_torch(
                        joint_selector, dtype=wp.int32 if joint_selector.dtype is torch.int32 else wp.int64
                    )
                )
            )
            wp.launch(
                actuator_kernels.write_2d_float_with_indices_kernel(env_selector, joint_selector),
                dim=(env_selector.shape[0], joint_selector.shape[0]),
                inputs=[
                    source if isinstance(source, wp.array) else wp.from_torch(source, dtype=wp.float32),
                    env_selector_wp,
                    joint_selector_wp,
                    full_data,
                ],
                outputs=[target_proxy.warp],
                device=target_proxy.warp.device,
            )
            self._stage_user_command(command_name, env_selector, joint_selector, None, None)

        def _write_command_mask(
            self,
            value: torch.Tensor | wp.array,
            env_mask: torch.Tensor | wp.array | None,
            joint_mask: torch.Tensor | wp.array | None,
            target: ProxyArray,
            *,
            command_name: str,
        ) -> None:
            """Validate and write one raw command with full-articulation masks."""
            self._require_execution_ready()
            selector_state = self._require_selector_state()
            target_proxy = target
            env_selector = (
                selector_state._all_env_mask
                if env_mask is None
                else self._as_command_mask_selector(env_mask, selector_state._all_env_mask, "env_mask")
            )
            joint_selector = (
                selector_state._all_joint_mask
                if joint_mask is None
                else self._as_command_mask_selector(joint_mask, selector_state._all_joint_mask, "joint_mask")
            )
            source = self._as_command_value(value, target_proxy)
            expected_shape = (selector_state._all_env_ids.shape[0], selector_state._all_joint_mask.shape[0])
            if tuple(source.shape) != expected_shape:
                raise ValueError(f"Command mask values must have shape {expected_shape}.")
            env_selector_wp = (
                selector_state._all_env_mask_wp
                if env_mask is None
                else env_selector
                if isinstance(env_selector, wp.array)
                else wp.from_torch(env_selector, dtype=wp.bool)
            )
            joint_selector_wp = (
                selector_state._all_joint_mask_wp
                if joint_mask is None
                else joint_selector
                if isinstance(joint_selector, wp.array)
                else wp.from_torch(joint_selector, dtype=wp.bool)
            )
            wp.launch(
                actuator_kernels.write_2d_float_with_mask,
                dim=expected_shape,
                inputs=[
                    source if isinstance(source, wp.array) else wp.from_torch(source, dtype=wp.float32),
                    env_selector_wp,
                    joint_selector_wp,
                ],
                outputs=[target_proxy.warp],
                device=target_proxy.warp.device,
            )
            self._stage_user_command(command_name, None, None, env_selector, joint_selector)

        def _stage_user_command(
            self,
            command_name: str,
            env_ids: torch.Tensor | wp.array | None,
            joint_ids: torch.Tensor | wp.array | None,
            env_mask: torch.Tensor | wp.array | None,
            joint_mask: torch.Tensor | wp.array | None,
        ) -> None:
            """Deliver a completed canonical raw-command write to its backend control."""
            control = self._control
            if control is None:
                raise RuntimeError("Actuator control is not available before the scoped facade is installed.")
            control.stage_user_command(command_name, self, env_ids, joint_ids, env_mask, joint_mask)

        def _write_group_parameter_index(
            self,
            group: ActuatorBase,
            name: str,
            value: float | torch.Tensor | wp.array | Sequence[float],
            env_ids: Sequence[int] | torch.Tensor | wp.array | None,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None,
        ) -> None:
            """Route one logical group parameter index write through canonical typed storage."""
            binding = group.__dict__.get("_parameter_binding")
            if binding is None or binding.parameter_proxies is None:
                raise RuntimeError("Actuator group does not have managed parameter storage.")
            selector_state = self._require_selector_state()
            self._write_scoped_parameter_index(
                scope="group",
                name=name,
                value=value,
                target=binding.parameter_proxies.get(name),
                scope_joint_ids=binding.joint_indices,
                csr_offsets=None,
                csr_slots=None,
                max_csr_fanout=0,
                actuator_type=type(group),
                group_binding=binding,
                group_inverse=selector_state.group_inverse_wp(binding),
                env_ids=env_ids,
                joint_ids=joint_ids,
            )

        def _write_group_parameter_mask(
            self,
            group: ActuatorBase,
            name: str,
            value: float | torch.Tensor | wp.array | Sequence[float],
            env_mask: torch.Tensor | wp.array | None,
            joint_mask: torch.Tensor | wp.array | None,
        ) -> None:
            """Route one logical group parameter mask write through canonical typed storage."""
            binding = group.__dict__.get("_parameter_binding")
            if binding is None or binding.parameter_proxies is None:
                raise RuntimeError("Actuator group does not have managed parameter storage.")
            self._write_scoped_parameter_mask(
                scope="group",
                name=name,
                value=value,
                target=binding.parameter_proxies.get(name),
                scope_joint_ids=binding.joint_indices,
                actuator_type=type(group),
                group_binding=binding,
                csr_offsets=None,
                csr_slots=None,
                env_mask=env_mask,
                joint_mask=joint_mask,
            )

        def _write_scoped_parameter_index(
            self,
            *,
            scope: str,
            name: str,
            value: float | torch.Tensor | wp.array | Sequence[float],
            target: ProxyArray | None,
            scope_joint_ids: torch.Tensor,
            csr_offsets: ProxyArray | None,
            csr_slots: ProxyArray | None,
            max_csr_fanout: int,
            actuator_type: type[ActuatorBase],
            group_binding: _GroupBinding | None,
            group_inverse: wp.array | None,
            env_ids: Sequence[int] | torch.Tensor | wp.array | None,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None,
        ) -> None:
            """Validate metadata and launch the synchronization-free indexed parameter writer."""
            self._require_execution_ready()
            if target is None:
                raise KeyError(f"Unknown parameter {name!r} for {scope} actuator facade.")
            selector_state = self._require_selector_state()
            explicit_env_ids = env_ids is not None
            env_selector = self._normalize_index_selector(env_ids, "env_ids", selector_state._all_env_ids, scope)
            explicit_joint_ids = joint_ids is not None
            if joint_ids is None:
                joint_selector = selector_state.default_joint_ids(scope_joint_ids.shape[0])
            else:
                joint_selector = self._as_index_torch(joint_ids, scope_joint_ids, "joint_ids")
            source, value_mode = self._normalize_index_value(
                value, env_selector.shape[0], joint_selector.shape[0], target
            )
            if self._debug_validation:
                self._validate_index_contents(
                    scope, name, env_selector, joint_selector, scope_joint_ids, explicit_joint_ids
                )
            if env_selector.shape[0] == 0 or joint_selector.shape[0] == 0:
                return
            source_wp = self._parameter_source_warp(source, target, selector_state)
            env_selector_wp = (
                selector_state._all_env_ids_wp
                if env_ids is None
                else wp.from_torch(env_selector, dtype=wp.int32 if env_selector.dtype is torch.int32 else wp.int64)
            )
            if joint_ids is None:
                joint_selector_wp = selector_state._identity_ids_wp
            else:
                joint_selector_wp = wp.from_torch(
                    joint_selector, dtype=wp.int32 if joint_selector.dtype is torch.int32 else wp.int64
                )
            scope_joint_ids_wp = (
                selector_state.group_joint_ids_wp(group_binding)
                if group_binding is not None
                else selector_state.type_joint_ids_wp(actuator_type)
            )
            type_scope = csr_offsets is not None and csr_slots is not None
            group_scope = group_inverse is not None
            num_candidates = max_csr_fanout if explicit_joint_ids and type_scope else scope_joint_ids.shape[0]
            if explicit_joint_ids and group_scope:
                num_candidates = 1
            if not explicit_joint_ids:
                num_candidates = 1
            backend_write = self._prepare_parameter_side_effect(
                actuator_type=actuator_type,
                scope=scope,
                name=name,
                value=target.torch,
                target=target,
                env_ids=None if env_ids is None else env_selector,
                joint_ids=None if joint_ids is None else joint_selector,
                csr_offsets=csr_offsets,
                csr_slots=csr_slots,
                group_binding=group_binding,
            )
            if backend_write is not None:
                getattr(self._control, "preflight_actuator_parameter_write", lambda *_: None)(name, backend_write)
            if explicit_env_ids:
                selector_state._last_env_positions_wp.fill_(-1)
                wp.launch(
                    actuator_kernels.record_last_scoped_parameter_position,
                    dim=env_selector.shape[0],
                    inputs=[env_selector_wp, selector_state._all_env_ids.shape[0]],
                    outputs=[selector_state._last_env_positions_wp],
                    device=target.warp.device,
                )
            if explicit_joint_ids:
                selector_state._last_joint_positions_wp.fill_(-1)
                wp.launch(
                    actuator_kernels.record_last_scoped_parameter_position,
                    dim=joint_selector.shape[0],
                    inputs=[
                        joint_selector_wp,
                        selector_state._all_joint_mask.shape[0],
                    ],
                    outputs=[selector_state._last_joint_positions_wp],
                    device=target.warp.device,
                )
            wp.launch(
                actuator_kernels.write_scoped_parameter_index,
                dim=(env_selector.shape[0], joint_selector.shape[0], num_candidates),
                inputs=[
                    source_wp,
                    env_selector_wp,
                    joint_selector_wp,
                    scope_joint_ids_wp,
                    None if csr_offsets is None else csr_offsets.warp,
                    None if csr_slots is None else csr_slots.warp,
                    group_inverse,
                    selector_state._last_env_positions_wp,
                    selector_state._last_joint_positions_wp,
                    selector_state._all_env_ids.shape[0],
                    selector_state._all_joint_mask.shape[0],
                    explicit_env_ids,
                    explicit_joint_ids,
                    type_scope,
                    group_scope,
                    value_mode,
                ],
                outputs=[target.warp],
                device=target.warp.device,
            )
            self._route_parameter_side_effect(name, backend_write)

        def _write_scoped_parameter_mask(
            self,
            *,
            scope: str,
            name: str,
            value: float | torch.Tensor | wp.array | Sequence[float],
            target: ProxyArray | None,
            scope_joint_ids: torch.Tensor,
            actuator_type: type[ActuatorBase],
            group_binding: _GroupBinding | None,
            csr_offsets: ProxyArray | None,
            csr_slots: ProxyArray | None,
            env_mask: torch.Tensor | wp.array | None,
            joint_mask: torch.Tensor | wp.array | None,
        ) -> None:
            """Validate metadata and launch the synchronization-free masked parameter writer."""
            self._require_execution_ready()
            if target is None:
                raise KeyError(f"Unknown parameter {name!r} for {scope} actuator facade.")
            selector_state = self._require_selector_state()
            env_selector = self._normalize_mask_selector(env_mask, "env_mask", selector_state._all_env_mask, scope)
            joint_selector = self._normalize_mask_selector(
                joint_mask, "joint_mask", selector_state._all_joint_mask, scope
            )
            source, value_mode = self._normalize_mask_value(value, scope_joint_ids.shape[0], target)
            source_wp = self._parameter_source_warp(source, target, selector_state)
            env_selector_wp = (
                selector_state._all_env_mask_wp if env_mask is None else wp.from_torch(env_selector, dtype=wp.bool)
            )
            joint_selector_wp = (
                selector_state._all_joint_mask_wp
                if joint_mask is None
                else wp.from_torch(joint_selector, dtype=wp.bool)
            )
            scope_joint_ids_wp = (
                selector_state.group_joint_ids_wp(group_binding)
                if group_binding is not None
                else selector_state.type_joint_ids_wp(actuator_type)
            )
            backend_write = self._prepare_parameter_side_effect(
                actuator_type=actuator_type,
                scope=scope,
                name=name,
                value=target.torch,
                target=target,
                env_ids=None,
                joint_ids=None,
                csr_offsets=csr_offsets,
                csr_slots=csr_slots,
                group_binding=group_binding,
                env_mask=None if env_mask is None else env_selector,
                joint_mask=None if joint_mask is None else joint_selector,
            )
            if backend_write is not None:
                getattr(self._control, "preflight_actuator_parameter_write", lambda *_: None)(name, backend_write)
            wp.launch(
                actuator_kernels.write_scoped_parameter_mask,
                dim=(selector_state._all_env_ids.shape[0], scope_joint_ids.shape[0]),
                inputs=[source_wp, env_selector_wp, joint_selector_wp, scope_joint_ids_wp, value_mode],
                outputs=[target.warp],
                device=target.warp.device,
            )
            self._route_parameter_side_effect(name, backend_write)

        @staticmethod
        def _as_torch(value: float | torch.Tensor | wp.array | Sequence[float], target: ProxyArray) -> torch.Tensor:
            """Convert compatibility inputs without copying device-resident selector contents to the host."""
            if isinstance(value, ProxyArray):
                value = value.torch
            elif isinstance(value, wp.array):
                value = wp.to_torch(value)
            elif not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=torch.float32, device=target.torch.device)
            if not isinstance(value, torch.Tensor):
                raise TypeError("Parameter values must be floating tensors, Warp arrays, scalars, or sequences.")
            if value.device != target.torch.device:
                raise ValueError("Parameter values must be on the same device as the actuator facade.")
            if value.dtype is not torch.float32:
                raise TypeError("Parameter values must have float32 dtype.")
            return value

        @staticmethod
        def _tensor_byte_span(value: torch.Tensor) -> tuple[int, int] | None:
            """Return a conservative visible byte interval from public tensor metadata."""
            if value.numel() == 0:
                return None
            min_offset = 0
            max_offset = 0
            for size, stride in zip(value.shape, value.stride(), strict=True):
                offset = (size - 1) * stride
                min_offset += min(offset, 0)
                max_offset += max(offset, 0)
            start = value.data_ptr() + min_offset * value.element_size()
            return start, value.data_ptr() + (max_offset + 1) * value.element_size()

        def _parameter_source_warp(
            self, source: torch.Tensor, target: ProxyArray, selector_state: _SelectorState
        ) -> wp.array:
            """Return a Warp source alias, staging overlapping values on Warp's stream."""
            source_span = self._tensor_byte_span(source)
            target_span = self._tensor_byte_span(target.torch)
            if source_span is None or target_span is None or source.device != target.torch.device:
                return wp.from_torch(source, dtype=wp.float32)
            overlaps = source_span[0] < target_span[1] and target_span[0] < source_span[1]
            if not overlaps:
                return wp.from_torch(source, dtype=wp.float32)
            staged = selector_state.alias_staging(source.shape)
            source_wp = wp.from_torch(source, dtype=wp.float32)
            staged_wp = wp.from_torch(staged, dtype=wp.float32)
            wp.launch(
                actuator_kernels.copy_scoped_parameter_source,
                dim=source.shape,
                inputs=[source_wp],
                outputs=[staged_wp],
                device=target.warp.device,
            )
            return staged_wp

        @staticmethod
        def _as_index_torch(
            value: torch.Tensor | wp.array | Sequence[int], target: torch.Tensor, name: str
        ) -> torch.Tensor:
            """Convert an index compatibility input while validating host-visible metadata only."""
            if isinstance(value, ProxyArray):
                value = value.torch
            elif isinstance(value, wp.array):
                value = wp.to_torch(value)
            elif not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=torch.int32, device=target.device)
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if value.device != target.device:
                raise ValueError(f"{name} must be on the actuator facade device.")
            if value.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must have a signed integer dtype.")
            return value

        def _normalize_index_selector(
            self,
            value: Sequence[int] | torch.Tensor | wp.array | None,
            name: str,
            default: torch.Tensor,
            scope: str,
        ) -> torch.Tensor:
            del scope
            return default if value is None else self._as_index_torch(value, default, name)

        def _default_scope_joint_ids(self, scope_joint_ids: torch.Tensor) -> torch.Tensor:
            """Return a pointer-stable compact selector owned by this facade."""
            return self._require_selector_state().default_joint_ids(scope_joint_ids.shape[0])

        def _normalize_mask_selector(
            self,
            value: torch.Tensor | wp.array | None,
            name: str,
            default: torch.Tensor,
            scope: str,
        ) -> torch.Tensor:
            del scope
            if value is None:
                return default
            if isinstance(value, ProxyArray):
                value = value.torch
            elif isinstance(value, wp.array):
                value = wp.to_torch(value)
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional boolean tensor.")
            if value.device != default.device:
                raise ValueError(f"{name} must be on the actuator facade device.")
            if value.dtype is not torch.bool:
                raise TypeError(f"{name} must have boolean dtype.")
            if value.shape != default.shape:
                raise ValueError(f"{name} must have shape {tuple(default.shape)}.")
            return value

        def _normalize_index_value(
            self,
            value: float | torch.Tensor | wp.array | Sequence[float],
            num_envs: int,
            num_joints: int,
            target: ProxyArray,
        ) -> tuple[torch.Tensor, int]:
            source = self._as_torch(value, target)
            if source.ndim == 0:
                return source.reshape(1, 1), 0
            if source.ndim == 1 and source.shape[0] == num_joints:
                return source.reshape(1, num_joints), 1
            if source.ndim == 2 and source.shape == (num_envs, num_joints):
                return source, 2
            raise ValueError("Parameter index values must be scalar, compact 1-D, or world-by-compact 2-D.")

        def _normalize_mask_value(
            self, value: float | torch.Tensor | wp.array | Sequence[float], num_slots: int, target: ProxyArray
        ) -> tuple[torch.Tensor, int]:
            source = self._as_torch(value, target)
            if source.ndim == 0:
                return source.reshape(1, 1), 0
            if source.ndim == 1 and source.shape[0] == num_slots:
                return source.reshape(1, num_slots), 1
            if source.ndim == 2 and source.shape == (self._require_selector_state()._all_env_ids.shape[0], num_slots):
                return source, 2
            raise ValueError("Parameter mask values must be scalar, compact 1-D, or world-by-compact 2-D.")

        def _validate_index_contents(
            self,
            scope: str,
            name: str,
            env_ids: torch.Tensor,
            joint_ids: torch.Tensor,
            scope_joint_ids: torch.Tensor,
            explicit_joint_ids: bool,
        ) -> None:
            """Perform opt-in synchronized bounds, ownership, and duplicate diagnostics."""
            env_values = env_ids.detach().cpu().tolist()
            joint_values = joint_ids.detach().cpu().tolist()
            selector_state = self._require_selector_state()
            if len(set(env_values)) != len(env_values) or len(set(joint_values)) != len(joint_values):
                raise ValueError(f"{name!r} {scope} selector contains duplicate ids.")
            if any(env_id < 0 or env_id >= selector_state._all_env_ids.shape[0] for env_id in env_values):
                raise ValueError(f"{name!r} {scope} selector contains an out-of-range environment id.")
            if not explicit_joint_ids:
                return
            scope_values = set(scope_joint_ids.detach().cpu().tolist())
            if any(joint_id < 0 or joint_id >= selector_state._all_joint_mask.shape[0] for joint_id in joint_values):
                raise ValueError(f"{name!r} {scope} selector contains an out-of-range joint id.")
            if any(joint_id not in scope_values for joint_id in joint_values):
                raise ValueError(f"{name!r} {scope} selector addresses a joint outside its ownership.")

        def _prepare_parameter_side_effect(
            self,
            *,
            actuator_type: type[ActuatorBase],
            scope: str,
            name: str,
            value: torch.Tensor,
            target: ProxyArray,
            env_ids: torch.Tensor | None,
            joint_ids: torch.Tensor | None,
            csr_offsets: ProxyArray | None,
            csr_slots: ProxyArray | None,
            group_binding: _GroupBinding | None,
            env_mask: torch.Tensor | None = None,
            joint_mask: torch.Tensor | None = None,
        ) -> _ActuatorParameterWrite | None:
            """Resolve one backend route before either canonical or backend mutation."""
            selector_state = self._require_selector_state()
            route = (actuator_type, name)
            owner_slots = selector_state._backend_owner_slots.get(route)
            if owner_slots is None or route not in selector_state._active_backend_routes:
                return None
            if group_binding is not None:
                is_implicit_drive = issubclass(actuator_type, ImplicitActuator) and name in {"stiffness", "damping"}
                is_native_group = id(group_binding) in self._native_group_binding_ids
                if not (is_implicit_drive or is_native_group):
                    return None
            return _ActuatorParameterWrite(
                value=value,
                canonical=target if group_binding is None else group_binding.arrays.get(name, target),
                env_ids=env_ids,
                joint_ids=joint_ids,
                env_mask=env_mask,
                joint_mask=joint_mask,
                group_binding=group_binding,
                type_csr_offsets=csr_offsets,
                type_csr_slots=csr_slots,
                backend_owner_slots=owner_slots,
                backend_parameter_staging=self._backend_parameter_staging,
                scope=scope,
            )

        def _route_parameter_side_effect(self, name: str, write: _ActuatorParameterWrite | None) -> None:
            """Forward a preflighted canonical write through the backend route."""
            if write is not None:
                self._control.write_actuator_parameter(name, write)

        def _publish(self, generation: int) -> None:
            self._generation = generation
            self._failure = ""

        def _invalidate(self, failure: str) -> None:
            self._failure = failure
            self._generation = None
            self._install_token = None
            if self._command is not None:
                self._command._close()
            if self._joint_command is not None:
                self._joint_command._close()
            self._command = None
            self._joint_command = None
            self._computed_effort = None
            self._applied_effort = None
            self._backend_parameter_staging = None
            for group in tuple(dict.values(self)):
                group._release_facade_storage()
            dict.clear(self)
            if isinstance(self._by_type, ActuatorCollection._GuardedMapping):
                for type_view in tuple(self._by_type._values.values()):
                    type_view._release()
                self._by_type._values.clear()
            self._by_type = {}
            selector_state = self._selector_state
            self._selector_state = None
            self._debug_validation = False
            self._control = None
            self._native_group_binding_ids = set()
            if selector_state is not None:
                selector_state.close()

        def _require_selector_state(self) -> _SelectorState:
            """Return the active selector state after generation validation."""
            self._require_current_generation()
            selector_state = self._selector_state
            if selector_state is None or selector_state._closed:
                raise RuntimeError("stale actuator view")
            return selector_state

        def _commit_topology(
            self,
            candidate: dict[str, ActuatorBase],
            operation: str,
            name: str,
            actuator: ActuatorBase | None,
        ) -> None:
            current_names = tuple(dict.keys(self))
            if tuple(candidate) == current_names and all(
                candidate[group_name] is dict.__getitem__(self, group_name) for group_name in current_names
            ):
                return
            self._manager.stage_deprecated_mutation(
                self,
                operation,
                name,
                actuator,
                candidate=candidate,
            )
            dict.clear(self)
            dict.update(self, candidate)

        def _topology_snapshot(self) -> dict[str, ActuatorBase]:
            return {group_name: group for group_name, group in dict.items(self)}

        def _require_current_generation(self, expected_token: object | None = None) -> None:
            if self._manager._closed:
                raise RuntimeError("Actuator collection is closed.")
            if self._failure:
                raise RuntimeError(self._failure)
            if self._generation is None or self._manager.generation != self._generation:
                raise RuntimeError("stale actuator view")
            if expected_token is not None and self._install_token is not expected_token:
                raise RuntimeError("stale actuator view")

        def _require_execution_ready(self, expected_token: object | None = None) -> None:
            self._require_current_generation(expected_token)
            if self._manager._dirty:
                raise RuntimeError("late registration requires STOP-to-READY rebuild before actuator execution.")
            if not self._manager.is_finalized:
                raise RuntimeError("actuator collection finalization is incomplete")

        def _require_bindable(self) -> None:
            """Allow aliases during completion, but not after a dirty rebuild request."""
            self._require_current_generation()
            if self._manager._dirty:
                raise RuntimeError("late registration requires STOP-to-READY rebuild before actuator access.")

    def _initialize_manager(self, sim_context: Any) -> None:
        self._sim_context = sim_context
        self._registrations: list[_ArticulationRegistration] = []
        self._views: dict[object, ActuatorCollection.ArticulationView] = {}
        self._active_generation: _CollectionGeneration | None = None
        self._next_generation = 0
        self._dirty = False
        self._finalizing = False
        self._closed = False
        self._deprecated_staged_topology_overrides: dict[object, object] = {}
        self._deprecated_topology_warning_emitted = False

    def register_articulation(
        self,
        *,
        key: object,
        cfgs: Mapping[str, ActuatorBaseCfg],
        control: ActuatorControl,
        replication_cfg_id: int,
        debug_validation: bool,
        debug_value_resolution: bool,
    ) -> ArticulationView:
        """Register one articulation for the next transactional generation."""
        if self._closed:
            raise RuntimeError("Actuator collection is closed and cannot accept registrations.")
        if key in self._views:
            raise RuntimeError(f"Actuator collection already registered {key!r} for this generation.")
        retained_override = self._deprecated_staged_topology_overrides.get(key)
        if retained_override is not None:
            cfgs = {name: _copy_actuator_cfg(cfg) for name, cfg in retained_override.items()}
        native_group_names = frozenset(control.discover_native_actuators(cfgs))
        unknown_native_groups = native_group_names.difference(cfgs)
        if unknown_native_groups:
            names = ", ".join(repr(name) for name in sorted(unknown_native_groups))
            raise ValueError(f"Native actuator discovery returned unknown group(s): {names}.")
        registration = _ArticulationRegistration(
            key=key,
            cfgs=cfgs,
            control=control,
            replication_cfg_id=replication_cfg_id,
            debug_validation=debug_validation,
            debug_value_resolution=debug_value_resolution,
            native_group_names=native_group_names,
        )
        view = self.ArticulationView(self, key)
        self._registrations.append(registration)
        self._views[key] = view
        if self._active_generation is not None:
            self._dirty = True
        return view

    @property
    def registration_keys(self) -> tuple[object, ...]:
        """Registered articulation keys in deterministic registration order."""
        return tuple(registration.key for registration in self._registrations)

    @property
    def generation(self) -> int | None:
        """Active published generation, if any."""
        return None if self._active_generation is None else self._active_generation.generation

    @property
    def is_finalized(self) -> bool:
        """Whether a clean generation is currently published."""
        return self._active_generation is not None and not self._dirty and not self._finalizing

    def finalize(self) -> None:
        """Build and atomically publish a complete registered generation."""
        if self._closed:
            raise RuntimeError("Actuator collection is closed.")
        if self._active_generation is not None:
            if self._dirty:
                raise RuntimeError("Late registration requires STOP-to-READY rebuild before finalization.")
            return
        if not self._registrations:
            return
        candidate: _CollectionGeneration | None = None
        try:
            candidate = _CollectionGeneration.build(
                tuple(self._registrations), self._sim_context, self._next_generation
            )
            candidate.validate()
            candidate.bind_facade_storage()
            candidate.prepare_solver_properties()
            candidate.write_solver_properties()
            candidate.validate_solver_properties()
            for binding in candidate.bindings:
                try:
                    binding.registration.control.prepare_actuator_binding(binding)
                except Exception as error:
                    error.add_note(f"Failed to prepare {_binding_context(binding)}.")
                    raise
        except Exception as error:
            if candidate is not None:
                try:
                    candidate.restore_solver_properties()
                except Exception as restore_error:
                    error.add_note(str(restore_error))
            self._invalidate_pending(error, failure="finalization failed")
            if candidate is not None:
                candidate.close()
            raise

        self._active_generation = candidate
        self._finalizing = True
        try:
            for binding in candidate.bindings:
                view = self._views[binding.registration.key]
                view._install(candidate, binding)
                view._publish(candidate.generation)
            for binding in candidate.bindings:
                try:
                    binding.registration.control.bind_actuator_view(self._views[binding.registration.key])
                except Exception as error:
                    error.add_note(f"Failed to bind actuator view for {_binding_context(binding)}.")
                    raise
            for binding in candidate.bindings:
                try:
                    binding.registration.control.complete_articulation_initialization()
                except Exception as error:
                    error.add_note(f"Failed to complete actuator initialization for {_binding_context(binding)}.")
                    raise
        except Exception as error:
            self._active_generation = None
            self._finalizing = False
            try:
                candidate.restore_solver_properties()
            except Exception as restore_error:
                error.add_note(str(restore_error))
            self._invalidate_pending(error, failure="finalization failed")
            candidate.close()
            error.add_note("Actuator collection finalization rolled back every registration.")
            raise
        candidate.commit_solver_properties()
        self._finalizing = False
        self._dirty = False
        for binding in candidate.bindings:
            self._deprecated_staged_topology_overrides.pop(binding.registration.key, None)

    def stage_deprecated_mutation(
        self,
        view: ArticulationView,
        operation: str,
        name: str,
        actuator: ActuatorBase | None,
        *,
        candidate: Mapping[str, ActuatorBase] | None = None,
    ) -> None:
        """Retain a copied ordered config override for deprecated facade mutation."""
        del operation, name, actuator
        view._require_current_generation()
        if candidate is None:
            candidate = view
        override: dict[str, ActuatorBaseCfg] = {}
        for group_name, group in candidate.items():
            if not isinstance(group, ActuatorBase):
                raise TypeError("Actuator facade values must be ActuatorBase instances.")
            override[group_name] = _copy_actuator_cfg(group.__dict__["cfg"])
        if not self._deprecated_topology_warning_emitted:
            warnings.warn(
                "Mutating Articulation.actuators is deprecated; rebuild the simulation from ArticulationCfg instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            self._deprecated_topology_warning_emitted = True
        self._deprecated_staged_topology_overrides[view._key] = override
        self._dirty = True

    def _invalidate_pending(self, error: Exception, *, failure: str) -> None:
        """Invalidate every control and facade without masking the triggering error."""
        for registration in self._registrations:
            try:
                registration.control.invalidate_actuator_view()
            except Exception as invalidation_error:
                error.add_note(f"Failed to invalidate backend control for {registration.key!r}: {invalidation_error}")
            try:
                self._views[registration.key]._invalidate(failure)
            except Exception as invalidation_error:
                error.add_note(f"Failed to invalidate actuator view for {registration.key!r}: {invalidation_error}")

    def clear_generation(self) -> None:
        """Invalidate the active generation and retain only staged topology overrides."""
        active = self._active_generation
        self._active_generation = None
        self._dirty = False
        self._finalizing = False
        for registration in self._registrations:
            registration.control.invalidate_actuator_view()
            self._views[registration.key]._invalidate("stale actuator view")
        if active is not None:
            active.close()
        self._next_generation += 1
        self._registrations.clear()
        self._views.clear()

    def close(self) -> None:
        """Permanently close this manager and reject later registration."""
        if self._closed:
            return
        self.clear_generation()
        self._deprecated_staged_topology_overrides.clear()
        self._closed = True

    @dataclass
    class _ExecutionBatch:
        actuator: ActuatorBase
        group_names: tuple[str, ...]
        group_slices: tuple[slice, ...]
        joint_indices: torch.Tensor
        joint_indices_wp: wp.array
        implicit_inputs: list[wp.array] | None = None
        implicit_outputs: list[wp.array] | None = None
        control_action: ArticulationActions | None = None
        joint_pos: torch.Tensor | None = None
        joint_vel: torch.Tensor | None = None
        gather_inputs: list[wp.array] | None = None
        gather_outputs: list[wp.array] | None = None

    class Command:
        """Commands received by the actuator models.

        Position and velocity commands use joint-side coordinates. All command
        arrays are indexed by articulation joint, not by motor shaft.
        """

        def __init__(self, collection: ActuatorCollection) -> None:
            """Initialize the command view.

            Args:
                collection: Owning actuator collection.
            """
            self._collection = collection

        @property
        def position(self) -> ProxyArray:
            """Desired positions [m or rad, depending on joint type]."""
            return self._collection._joint_pos_target_ta

        @property
        def velocity(self) -> ProxyArray:
            """Desired velocities [m/s or rad/s, depending on joint type]."""
            return self._collection._joint_vel_target_ta

        @property
        def effort(self) -> ProxyArray:
            """Effort commands [N or N·m, depending on joint type]."""
            return self._collection._joint_effort_target_ta

        def set_position_index(
            self,
            *,
            value: torch.Tensor | wp.array,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            full_data: bool = False,
        ) -> None:
            """Set desired positions using indices.

            Args:
                value: Desired positions [m or rad, depending on joint type].
                joint_ids: Joint indices. Defaults to all joints.
                env_ids: Environment indices. Defaults to all environments.
                full_data: Whether :paramref:`value` is a full articulation command buffer.
            """
            collection = self._collection
            env_ids_resolved = collection._control.resolve_env_ids(env_ids)
            joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
            collection._write_index_target(
                value,
                env_ids_resolved,
                joint_ids_resolved,
                collection._joint_pos_target,
                full_data=full_data,
                command_name="position",
            )

        def set_velocity_index(
            self,
            *,
            value: torch.Tensor | wp.array,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            full_data: bool = False,
        ) -> None:
            """Set desired velocities using indices.

            Args:
                value: Desired velocities [m/s or rad/s, depending on joint type].
                joint_ids: Joint indices. Defaults to all joints.
                env_ids: Environment indices. Defaults to all environments.
                full_data: Whether :paramref:`value` is a full articulation command buffer.
            """
            collection = self._collection
            env_ids_resolved = collection._control.resolve_env_ids(env_ids)
            joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
            collection._write_index_target(
                value,
                env_ids_resolved,
                joint_ids_resolved,
                collection._joint_vel_target,
                full_data=full_data,
                command_name="velocity",
            )

        def set_effort_index(
            self,
            *,
            value: torch.Tensor | wp.array,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            full_data: bool = False,
        ) -> None:
            """Set effort commands using indices.

            Args:
                value: Effort commands [N or N·m, depending on joint type].
                joint_ids: Joint indices. Defaults to all joints.
                env_ids: Environment indices. Defaults to all environments.
                full_data: Whether :paramref:`value` is a full articulation command buffer.
            """
            collection = self._collection
            env_ids_resolved = collection._control.resolve_env_ids(env_ids)
            joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
            collection._write_index_target(
                value,
                env_ids_resolved,
                joint_ids_resolved,
                collection._joint_effort_target,
                full_data=full_data,
                command_name="effort",
            )

        def set_position_mask(
            self,
            *,
            value: torch.Tensor | wp.array,
            joint_mask: wp.array | None = None,
            env_mask: wp.array | None = None,
        ) -> None:
            """Set desired positions using masks.

            Args:
                value: Full articulation position commands [m or rad, depending on joint type].
                joint_mask: Joint selection mask. Defaults to all joints.
                env_mask: Environment selection mask. Defaults to all environments.
            """
            collection = self._collection
            env_mask_resolved = collection._control.resolve_env_mask(env_mask)
            joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
            collection._write_mask_target(
                value,
                env_mask_resolved,
                joint_mask_resolved,
                collection._joint_pos_target,
                command_name="position",
            )

        def set_velocity_mask(
            self,
            *,
            value: torch.Tensor | wp.array,
            joint_mask: wp.array | None = None,
            env_mask: wp.array | None = None,
        ) -> None:
            """Set desired velocities using masks.

            Args:
                value: Full articulation velocity commands [m/s or rad/s, depending on joint type].
                joint_mask: Joint selection mask. Defaults to all joints.
                env_mask: Environment selection mask. Defaults to all environments.
            """
            collection = self._collection
            env_mask_resolved = collection._control.resolve_env_mask(env_mask)
            joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
            collection._write_mask_target(
                value,
                env_mask_resolved,
                joint_mask_resolved,
                collection._joint_vel_target,
                command_name="velocity",
            )

        def set_effort_mask(
            self,
            *,
            value: torch.Tensor | wp.array,
            joint_mask: wp.array | None = None,
            env_mask: wp.array | None = None,
        ) -> None:
            """Set effort commands using masks.

            Args:
                value: Full articulation effort commands [N or N·m, depending on joint type].
                joint_mask: Joint selection mask. Defaults to all joints.
                env_mask: Environment selection mask. Defaults to all environments.
            """
            collection = self._collection
            env_mask_resolved = collection._control.resolve_env_mask(env_mask)
            joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
            collection._write_mask_target(
                value,
                env_mask_resolved,
                joint_mask_resolved,
                collection._joint_effort_target,
                command_name="effort",
            )

    class JointCommand:
        """Processed commands produced for the simulated joints."""

        def __init__(self, collection: ActuatorCollection) -> None:
            """Initialize the joint command view.

            Args:
                collection: Owning actuator collection.
            """
            self._collection = collection

        @property
        def position(self) -> ProxyArray:
            """Processed position commands [m or rad, depending on joint type]."""
            return self._collection._joint_pos_target_sim_ta

        @property
        def velocity(self) -> ProxyArray:
            """Processed velocity commands [m/s or rad/s, depending on joint type]."""
            return self._collection._joint_vel_target_sim_ta

        @property
        def effort(self) -> ProxyArray:
            """Processed effort commands [N or N·m, depending on joint type]."""
            return self._collection._joint_effort_target_sim_ta

    def __init__(
        self,
        sim_context_or_actuator_cfgs: Any,
        control: ActuatorControl | None = None,
        *,
        debug_value_resolution: bool = False,
    ):
        """Initialize the actuator collection.

        Args:
            sim_context_or_actuator_cfgs: Simulation context for the scoped manager,
                or the deprecated mapping of actuator group names to configs.
            control: Backend control bridge for state reads and sim writes.
            debug_value_resolution: Whether to log actuator value resolution.
        """
        if control is None and not isinstance(sim_context_or_actuator_cfgs, Mapping):
            self._initialize_manager(sim_context_or_actuator_cfgs)
            return
        if control is None:
            raise TypeError("The deprecated ActuatorCollection constructor requires a control bridge.")
        actuator_cfgs = sim_context_or_actuator_cfgs
        if not isinstance(actuator_cfgs, Mapping):
            raise TypeError("Actuator configs must be a mapping in the deprecated constructor.")
        self._control = control
        self._groups: dict[str, ActuatorBase] = {}
        self._groups_by_class: dict[type[ActuatorBase], list[ActuatorBase]] = {}
        self._native_group_names: set[str] = set()
        self._has_implicit_actuators = False
        self._joint_indices_wp: dict[str, wp.array] = {}
        self._launch_cache = _WarpLaunchCache(self.device)

        self._allocate_buffers()
        self._command = self.Command(self)
        self._joint_command = self.JointCommand(self)
        self._native_group_names = self._control.prepare_native_actuators(self, actuator_cfgs)
        self._build_groups(actuator_cfgs)
        self._control.finalize_native_actuators(self)
        self._validate_coverage()
        self._build_execution_batches()
        if debug_value_resolution:
            self._print_value_resolution_table()

    """
    Mapping interface.
    """

    def __getitem__(self, name: str) -> ActuatorBase:
        return self._groups[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __setitem__(self, name: str, actuator: ActuatorBase) -> None:
        raise TypeError("ActuatorCollection membership is fixed after initialization.")

    """
    Properties.
    """

    @property
    def command(self) -> Command:
        """Commands received by the actuator models."""
        return self._command

    @property
    def joint_command(self) -> JointCommand:
        """Processed commands produced for the simulated joints."""
        return self._joint_command

    @property
    def num_instances(self) -> int:
        """Number of articulation instances."""
        return self._control.num_instances

    @property
    def num_joints(self) -> int:
        """Number of articulation joints."""
        return self._control.num_joints

    @property
    def device(self) -> str:
        """Warp/Torch device string."""
        return self._control.device

    @property
    def has_implicit_actuators(self) -> bool:
        """Whether any configured actuator group is implicit."""
        return self._has_implicit_actuators

    @property
    def computed_torque(self) -> ProxyArray:
        """Joint torques computed before clipping [N or N·m, depending on joint type]."""
        return self._computed_torque_ta

    @property
    def applied_torque(self) -> ProxyArray:
        """Joint torques applied after clipping [N or N·m, depending on joint type]."""
        return self._applied_torque_ta

    @property
    def actuator_stiffness(self) -> ProxyArray:
        """Actuator-resolved stiffness values [N/m or N·m/rad, depending on joint type]."""
        return self._actuator_stiffness_ta

    @property
    def actuator_damping(self) -> ProxyArray:
        """Actuator-resolved damping values [N·s/m or N·m·s/rad, depending on joint type]."""
        return self._actuator_damping_ta

    @property
    def soft_joint_vel_limits(self) -> ProxyArray:
        """Actuator-resolved soft joint velocity limits [m/s or rad/s, depending on joint type]."""
        return self._soft_joint_vel_limits_ta

    @property
    def gear_ratio(self) -> ProxyArray:
        """Gear ratio for relating motor torques to applied joint torques [dimensionless]."""
        return self._gear_ratio_ta

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset all actuator group states.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
        """
        if env_ids is None:
            env_ids = slice(None)
        for actuator in self._groups.values():
            actuator.reset(env_ids)
        self._control.reset_native_actuators(env_ids)

    def compute(self, dt: float = 0.0) -> None:
        """Compute processed actuator commands and telemetry.

        Args:
            dt: Physics step size [s].
        """
        if self._control.compute_native_actuators(self, dt):
            return

        for batch in self._execution_batches:
            actuator = batch.actuator
            if type(actuator) is ImplicitActuator:
                self._compute_implicit_batch(batch)
                continue
            if batch.control_action is not None:
                self._gather_explicit_batch(batch)
                control_action = batch.control_action
                command_pos = control_action.joint_positions
                command_vel = control_action.joint_velocities
                command_effort = control_action.joint_efforts
                control_action = actuator.compute(
                    control_action,
                    joint_pos=batch.joint_pos,
                    joint_vel=batch.joint_vel,
                )
                self._scatter_actuator_output(actuator, control_action, batch.joint_indices_wp)
                control_action.joint_positions = command_pos
                control_action.joint_velocities = command_vel
                control_action.joint_efforts = command_effort
                continue
            joint_indices = actuator.joint_indices if len(batch.group_names) == 1 else batch.joint_indices
            control_action = ArticulationActions(
                joint_positions=self.command.position.torch[:, joint_indices],
                joint_velocities=self.command.velocity.torch[:, joint_indices],
                joint_efforts=self.command.effort.torch[:, joint_indices],
                joint_indices=joint_indices,
            )
            control_action = actuator.compute(
                control_action,
                joint_pos=self._control.joint_pos.torch[:, joint_indices],
                joint_vel=self._control.joint_vel.torch[:, joint_indices],
            )
            self._scatter_actuator_output(actuator, control_action, batch.joint_indices_wp)

    def submit_commands(self) -> None:
        """Submit processed actuator command buffers through the backend control object."""
        self._control.submit_commands(self)

    def write_actuator_stiffness_to_sim(
        self,
        *,
        stiffness: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Write actuator stiffness values and propagate them to native controllers."""
        self._write_actuator_gain("kp", stiffness, env_ids, joint_ids, self._actuator_stiffness)

    def write_actuator_damping_to_sim(
        self,
        *,
        damping: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Write actuator damping values and propagate them to native controllers."""
        self._write_actuator_gain("kd", damping, env_ids, joint_ids, self._actuator_damping)

    """
    Internal helpers.
    """

    def _allocate_buffers(self) -> None:
        shape = (self.num_instances, self.num_joints)
        self._joint_pos_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_vel_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_effort_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_pos_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_vel_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_effort_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._computed_torque = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._applied_torque = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._actuator_stiffness = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._actuator_damping = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._soft_joint_vel_limits = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._gear_ratio = wp.ones(shape, dtype=wp.float32, device=self.device)
        self._all_env_ids = wp.array(list(range(self.num_instances)), dtype=wp.int32, device=self.device)
        self._all_joint_ids = wp.array(list(range(self.num_joints)), dtype=wp.int32, device=self.device)

        self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
        self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
        self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
        self._joint_pos_target_sim_ta = ProxyArray(self._joint_pos_target_sim)
        self._joint_vel_target_sim_ta = ProxyArray(self._joint_vel_target_sim)
        self._joint_effort_target_sim_ta = ProxyArray(self._joint_effort_target_sim)
        self._computed_torque_ta = ProxyArray(self._computed_torque)
        self._applied_torque_ta = ProxyArray(self._applied_torque)
        self._actuator_stiffness_ta = ProxyArray(self._actuator_stiffness)
        self._actuator_damping_ta = ProxyArray(self._actuator_damping)
        self._soft_joint_vel_limits_ta = ProxyArray(self._soft_joint_vel_limits)
        self._gear_ratio_ta = ProxyArray(self._gear_ratio)

    def _build_groups(self, actuator_cfgs: dict[str, ActuatorBaseCfg]) -> None:
        for actuator_name, actuator_cfg in actuator_cfgs.items():
            joint_ids, joint_names = self._control.find_joints(actuator_cfg.joint_names_expr)
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            if len(joint_names) == self.num_joints:
                actuator_joint_ids: slice | torch.Tensor = slice(None)
            elif isinstance(joint_ids, ProxyArray):
                actuator_joint_ids = joint_ids.torch
            else:
                actuator_joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.int32)

            defaults = self._control.get_default_joint_properties(actuator_joint_ids)
            cfg = actuator_cfg.copy() if hasattr(actuator_cfg, "copy") else actuator_cfg
            actuator: ActuatorBase = cfg.class_type(
                cfg=cfg,
                joint_names=joint_names,
                joint_ids=actuator_joint_ids,
                num_envs=self.num_instances,
                device=self.device,
                stiffness=defaults.stiffness,
                damping=defaults.damping,
                armature=defaults.armature,
                friction=defaults.friction,
                dynamic_friction=defaults.dynamic_friction,
                viscous_friction=defaults.viscous_friction,
                effort_limit=defaults.effort_limit,
                velocity_limit=defaults.velocity_limit,
            )

            self._groups[actuator_name] = actuator
            self._groups_by_class.setdefault(type(actuator), []).append(actuator)
            self._joint_indices_wp[actuator_name] = self._joint_indices_as_wp(actuator)
            self._has_implicit_actuators = self._has_implicit_actuators or isinstance(actuator, ImplicitActuator)

            self._scatter_resolved_gains(actuator_name, actuator)
            self._control.write_resolved_joint_properties(
                actuator,
                native_managed=actuator_name in self._native_group_names,
            )

    def _joint_indices_as_wp(self, actuator: ActuatorBase) -> wp.array:
        if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
            return self._all_joint_ids
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, wp.array):
            return joint_indices
        return wp.from_torch(joint_indices.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)

    def _joint_indices_as_torch(self, actuator: ActuatorBase) -> torch.Tensor:
        if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
            return torch.arange(self.num_joints, dtype=torch.int32, device=self.device)
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, wp.array):
            joint_indices = wp.to_torch(joint_indices)
        return joint_indices.to(self.device, dtype=torch.int32).contiguous()

    def _make_execution_batch(
        self,
        group_names: tuple[str, ...],
        groups: tuple[ActuatorBase, ...],
        joint_indices: torch.Tensor,
        *,
        executor: ActuatorBase | None = None,
    ) -> ActuatorCollection._ExecutionBatch:
        group_slices = []
        start = 0
        for group in groups:
            stop = start + group.num_joints
            group_slices.append(slice(start, stop))
            start = stop
        joint_indices = joint_indices.to(self.device, dtype=torch.int32).contiguous()
        if executor is None:
            executor = groups[0]
        else:
            executor._joint_names = [name for group in groups for name in group.joint_names]
            executor._joint_indices = joint_indices
        batch = self._ExecutionBatch(
            actuator=executor,
            group_names=group_names,
            group_slices=tuple(group_slices),
            joint_indices=joint_indices,
            joint_indices_wp=wp.from_torch(joint_indices, dtype=wp.int32),
        )
        if type(executor) is ImplicitActuator:
            batch.implicit_inputs = [
                self._joint_pos_target,
                self._joint_vel_target,
                self._joint_effort_target,
                self._control.joint_pos.warp,
                self._control.joint_vel.warp,
                wp.from_torch(executor.stiffness, dtype=wp.float32),
                wp.from_torch(executor.damping, dtype=wp.float32),
                wp.from_torch(executor.effort_limit, dtype=wp.float32),
                wp.from_torch(executor.velocity_limit, dtype=wp.float32),
                batch.joint_indices_wp,
            ]
            batch.implicit_outputs = [
                wp.from_torch(executor.computed_effort, dtype=wp.float32),
                wp.from_torch(executor.applied_effort, dtype=wp.float32),
                self._joint_pos_target_sim,
                self._joint_vel_target_sim,
                self._joint_effort_target_sim,
                self._computed_torque,
                self._applied_torque,
                self._soft_joint_vel_limits,
            ]
        elif type(executor) in (IdealPDActuator, DCMotor):
            shape = (self.num_instances, joint_indices.shape[0])
            command_pos = torch.empty(shape, dtype=torch.float32, device=self.device)
            command_vel = torch.empty_like(command_pos)
            command_effort = torch.empty_like(command_pos)
            joint_pos = torch.empty_like(command_pos)
            joint_vel = torch.empty_like(command_pos)
            batch.control_action = ArticulationActions(
                joint_positions=command_pos,
                joint_velocities=command_vel,
                joint_efforts=command_effort,
                joint_indices=joint_indices,
            )
            batch.joint_pos = joint_pos
            batch.joint_vel = joint_vel
            batch.gather_inputs = [
                self._joint_pos_target,
                self._joint_vel_target,
                self._joint_effort_target,
                self._control.joint_pos.warp,
                self._control.joint_vel.warp,
                batch.joint_indices_wp,
            ]
            batch.gather_outputs = [
                wp.from_torch(command_pos, dtype=wp.float32),
                wp.from_torch(command_vel, dtype=wp.float32),
                wp.from_torch(command_effort, dtype=wp.float32),
                wp.from_torch(joint_pos, dtype=wp.float32),
                wp.from_torch(joint_vel, dtype=wp.float32),
            ]
        return batch

    def _build_execution_batches(self) -> None:
        native_active = getattr(self._control, "native_active", False)
        batch_by_group: dict[str, ActuatorCollection._ExecutionBatch] = {}
        if not self._groups:
            self._execution_batches = []
            return
        group_joint_indices = {name: self._joint_indices_as_torch(group) for name, group in self._groups.items()}
        joint_use_count = torch.bincount(
            torch.cat(list(group_joint_indices.values())).to(dtype=torch.long),
            minlength=self.num_joints,
        )

        for actuator_type in self._groups_by_class:
            names = tuple(name for name, group in self._groups.items() if type(group) is actuator_type)
            groups = [self._groups[name] for name in names]
            joint_indices = [group_joint_indices[name] for name in names]
            supported = actuator_type.__dict__.get("_supports_execution_aggregation", False)

            if native_active or not supported:
                for name, group, indices in zip(names, groups, joint_indices):
                    batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
                continue

            safe = [
                (name, group, indices)
                for name, group, indices in zip(names, groups, joint_indices)
                if torch.all(joint_use_count[indices.to(dtype=torch.long)] == 1)
            ]
            safe_names_set = {name for name, _, _ in safe}
            unsafe = [
                (name, group, indices)
                for name, group, indices in zip(names, groups, joint_indices)
                if name not in safe_names_set
            ]
            for name, group, indices in unsafe:
                batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
            if len(safe) < 2:
                for name, group, indices in safe:
                    batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
                continue

            safe_names, safe_groups, safe_indices = zip(*safe)
            combined = torch.cat(safe_indices)
            executor = actuator_type._build_execution_actuator(safe_groups)
            batch = self._make_execution_batch(safe_names, safe_groups, combined, executor=executor)
            self._validate_execution_batch(batch, safe_groups)
            self._bind_execution_batch_parameters(batch, safe_groups)
            for name in safe_names:
                batch_by_group[name] = batch

        seen: set[int] = set()
        self._execution_batches = []
        for name in self._groups:
            batch = batch_by_group[name]
            if id(batch) not in seen:
                self._execution_batches.append(batch)
                seen.add(id(batch))

    def _validate_execution_batch(
        self, batch: ActuatorCollection._ExecutionBatch, groups: Sequence[ActuatorBase]
    ) -> None:
        expected_joint_names = [name for group in groups for name in group.joint_names]
        expected_num_joints = len(expected_joint_names)
        if len(batch.group_names) != len(groups) or len(batch.group_slices) != len(groups):
            raise ValueError("Execution batch group metadata is inconsistent.")
        if any(self._groups[name] is not group for name, group in zip(batch.group_names, groups)):
            raise ValueError("Execution batch group names do not match its logical groups.")
        if batch.actuator.joint_names != expected_joint_names:
            raise ValueError("Execution batch joint names do not match its logical groups.")
        if batch.joint_indices.ndim != 1 or batch.joint_indices.shape[0] != expected_num_joints:
            raise ValueError("Execution batch joint indices do not match its logical groups.")
        if (
            batch.joint_indices.dtype != torch.int32
            or batch.joint_indices.device != torch.device(self.device)
            or not batch.joint_indices.is_contiguous()
        ):
            raise ValueError("Execution batch joint indices use an unexpected dtype or device.")
        if not torch.equal(batch.actuator.joint_indices, batch.joint_indices):
            raise ValueError("Execution actuator joint indices do not match its batch.")
        if (
            batch.joint_indices_wp.shape[0] != expected_num_joints
            or batch.joint_indices_wp.dtype != wp.int32
            or batch.joint_indices_wp.device != wp.get_device(self.device)
        ):
            raise ValueError("Execution batch Warp joint indices do not match its logical groups.")

        expected_start = 0
        for group, group_slice in zip(groups, batch.group_slices):
            expected_stop = expected_start + group.num_joints
            if group_slice != slice(expected_start, expected_stop):
                raise ValueError("Execution batch group slices are not contiguous.")
            expected_start = expected_stop
        if expected_start != expected_num_joints:
            raise ValueError("Execution batch group slices do not cover all executor joints.")

        tensor_names = (*type(batch.actuator)._execution_parameter_names(), "computed_effort", "applied_effort")
        for name in tensor_names:
            value = getattr(batch.actuator, name)
            if value.shape != (self.num_instances, expected_num_joints):
                raise ValueError(f"Execution batch tensor '{name}' has an unexpected shape.")
            if value.device != torch.device(self.device) or value.dtype != getattr(groups[0], name).dtype:
                raise ValueError(f"Execution batch tensor '{name}' has an unexpected dtype or device.")

    def _bind_execution_batch_parameters(
        self, batch: ActuatorCollection._ExecutionBatch, groups: Sequence[ActuatorBase]
    ) -> None:
        tensor_names = (*type(batch.actuator)._execution_parameter_names(), "computed_effort", "applied_effort")
        bindings: list[tuple[ActuatorBase, str, torch.Tensor]] = []
        for group, group_slice in zip(groups, batch.group_slices):
            for name in tensor_names:
                original = getattr(group, name)
                view = getattr(batch.actuator, name)[:, group_slice]
                if view.shape != original.shape or view.dtype != original.dtype or view.device != original.device:
                    raise ValueError(f"Execution batch view for '{name}' is incompatible with its logical group.")
                bindings.append((group, name, view))

        for group, name, view in bindings:
            setattr(group, name, view)

    def _bind_execution_batch_outputs(self, batch: ActuatorCollection._ExecutionBatch) -> None:
        for group_name, group_slice in zip(batch.group_names, batch.group_slices):
            group = self._groups[group_name]
            group.computed_effort = batch.actuator.computed_effort[:, group_slice]
            group.applied_effort = batch.actuator.applied_effort[:, group_slice]

    def _compute_implicit_batch(self, batch: ActuatorCollection._ExecutionBatch) -> None:
        if batch.implicit_inputs is None or batch.implicit_outputs is None:
            raise RuntimeError("Implicit actuator execution batch was not initialized.")
        self._launch_cache.launch(
            ("implicit", id(batch)),
            actuator_kernels.compute_implicit_actuator_batch,
            dim=(self.num_instances, batch.joint_indices_wp.shape[0]),
            inputs=batch.implicit_inputs,
            outputs=batch.implicit_outputs,
        )

    def _gather_explicit_batch(self, batch: ActuatorCollection._ExecutionBatch) -> None:
        if batch.gather_inputs is None or batch.gather_outputs is None:
            raise RuntimeError("Explicit actuator execution batch was not initialized.")
        self._launch_cache.launch(
            ("gather", id(batch)),
            actuator_kernels.gather_actuator_batch,
            dim=(self.num_instances, batch.joint_indices_wp.shape[0]),
            inputs=batch.gather_inputs,
            outputs=batch.gather_outputs,
        )

    def _write_index_target(
        self,
        target: torch.Tensor | wp.array,
        env_ids: torch.Tensor | wp.array,
        joint_ids: torch.Tensor | wp.array,
        target_buffer: wp.array,
        *,
        full_data: bool,
        command_name: str,
    ) -> None:
        expected_shape = (self.num_instances, self.num_joints) if full_data else (env_ids.shape[0], joint_ids.shape[0])
        self._control.assert_shape_and_dtype(target, expected_shape, wp.float32, "target")
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(env_ids, joint_ids),
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[target, env_ids, joint_ids, full_data],
            outputs=[target_buffer],
            device=self.device,
        )
        self._control.stage_user_command(command_name, self, env_ids, joint_ids, None, None)

    def _write_mask_target(
        self,
        target: torch.Tensor | wp.array,
        env_mask: wp.array,
        joint_mask: wp.array,
        target_buffer: wp.array,
        *,
        command_name: str,
    ) -> None:
        self._control.assert_shape_and_dtype_mask(target, (env_mask, joint_mask), wp.float32, "target")
        wp.launch(
            actuator_kernels.write_2d_float_with_mask,
            dim=(env_mask.shape[0], joint_mask.shape[0]),
            inputs=[target, env_mask, joint_mask],
            outputs=[target_buffer],
            device=self.device,
        )
        self._control.stage_user_command(command_name, self, None, None, env_mask, joint_mask)

    def _scatter_resolved_gains(self, actuator_name: str, actuator: ActuatorBase) -> None:
        joint_indices = self._joint_indices_wp[actuator_name]
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(self._all_env_ids, joint_indices),
            dim=(self.num_instances, joint_indices.shape[0]),
            inputs=[actuator.stiffness, self._all_env_ids, joint_indices, False],
            outputs=[self._actuator_stiffness],
            device=self.device,
        )
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(self._all_env_ids, joint_indices),
            dim=(self.num_instances, joint_indices.shape[0]),
            inputs=[actuator.damping, self._all_env_ids, joint_indices, False],
            outputs=[self._actuator_damping],
            device=self.device,
        )

    def _scatter_actuator_output(
        self,
        actuator: ActuatorBase,
        control_action: ArticulationActions,
        joint_indices: wp.array | None = None,
    ) -> None:
        if joint_indices is None:
            joint_indices = self._joint_indices_as_wp(actuator)
        target_inputs = [
            control_action.joint_positions,
            control_action.joint_velocities,
            control_action.joint_efforts,
            joint_indices,
        ]
        target_outputs = [
            self._joint_pos_target_sim,
            self._joint_vel_target_sim,
            self._joint_effort_target_sim,
        ]
        stable_launch = type(actuator) in (IdealPDActuator, DCMotor)
        if stable_launch:
            self._launch_cache.launch(
                ("scatter_targets", id(actuator)),
                actuator_kernels.scatter_processed_targets,
                dim=(self.num_instances, joint_indices.shape[0]),
                inputs=target_inputs,
                outputs=target_outputs,
            )
        else:
            wp.launch(
                actuator_kernels.scatter_processed_targets,
                dim=(self.num_instances, joint_indices.shape[0]),
                inputs=target_inputs,
                outputs=target_outputs,
                device=self.device,
            )
        gear_ratio = getattr(actuator, "gear_ratio", None)
        has_gear_ratio = gear_ratio is not None
        if gear_ratio is None:
            gear_ratio = self._gear_ratio
        telemetry_inputs = [
            actuator.computed_effort,
            actuator.applied_effort,
            gear_ratio,
            actuator.velocity_limit,
            has_gear_ratio,
            joint_indices,
        ]
        telemetry_outputs = [
            self._computed_torque,
            self._applied_torque,
            self._gear_ratio,
            self._soft_joint_vel_limits,
        ]
        if stable_launch:
            self._launch_cache.launch(
                ("scatter_telemetry", id(actuator)),
                actuator_kernels.scatter_actuator_state_model,
                dim=(self.num_instances, joint_indices.shape[0]),
                inputs=telemetry_inputs,
                outputs=telemetry_outputs,
            )
        else:
            wp.launch(
                actuator_kernels.scatter_actuator_state_model,
                dim=(self.num_instances, joint_indices.shape[0]),
                inputs=telemetry_inputs,
                outputs=telemetry_outputs,
                device=self.device,
            )

    def _write_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
        target_buffer: wp.array,
    ) -> None:
        values_snapshot = values.to(self.device, dtype=torch.float32).contiguous().clone()
        actuator_attr = {"kp": "stiffness", "kd": "damping"}[attr]
        self._write_execution_parameter(actuator_attr, values_snapshot, env_ids, joint_ids)
        env_ids_wp = wp.from_torch(env_ids.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        joint_ids_wp = wp.from_torch(joint_ids.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        values_wp = wp.from_torch(values_snapshot, dtype=wp.float32)
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(env_ids_wp, joint_ids_wp),
            dim=(env_ids_wp.shape[0], joint_ids_wp.shape[0]),
            inputs=[values_wp, env_ids_wp, joint_ids_wp, False],
            outputs=[target_buffer],
            device=self.device,
        )
        self._control.write_native_actuator_gain(attr, values_snapshot, env_ids, joint_ids)

    def _write_execution_parameter(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        values = values.to(self.device, dtype=torch.float32)
        env_ids = env_ids.to(self.device, dtype=torch.long)
        joint_ids = joint_ids.to(self.device, dtype=torch.long)
        for batch in self._execution_batches:
            batch_joint_ids = batch.joint_indices.to(dtype=torch.long)
            requested_columns, batch_columns = torch.where(joint_ids[:, None] == batch_joint_ids[None, :])
            if requested_columns.numel() == 0:
                continue
            target = getattr(batch.actuator, attr)
            target[env_ids[:, None], batch_columns[None, :]] = values[:, requested_columns]

    def _validate_coverage(self) -> None:
        if self.num_joints == 0:
            return
        total_act_joints = sum(actuator.num_joints for actuator in self._groups.values())
        expected_joints = self.num_joints - self._control.num_fixed_tendons
        if total_act_joints != expected_joints:
            logger.warning(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                " joints available: %s != %s.",
                total_act_joints,
                expected_joints,
            )

    def _print_value_resolution_table(self) -> None:
        table = PrettyTable(["Group", "Property", "Name", "ID", "USD Value", "ActuatorCfg Value", "Applied"])
        for actuator_group, actuator in self._groups.items():
            group_count = 0
            for property_name, resolution_details in actuator.joint_property_resolution_table.items():
                for prop_idx, resolution_detail in enumerate(resolution_details):
                    actuator_group_str = actuator_group if group_count == 0 else ""
                    property_str = property_name if prop_idx == 0 else ""
                    fmt = [f"{value:.2e}" if isinstance(value, float) else str(value) for value in resolution_detail]
                    table.add_row([actuator_group_str, property_str, *fmt])
                    group_count += 1
        logger.warning("\nActuatorCfg-USD Value Discrepancy Resolution (matching values are skipped): \n%s", table)
