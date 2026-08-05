# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-actuator adapter shared by the Newton and PhysX backends.

The adapter owns actuator state and performs per-step ``step`` / ``reset``
calls.  Each articulation binding either aliases a whole exact-type canonical
range directly into the Newton controller, clamping, and output arrays, or
uses the model-global Newton arrays as persistent staging.  After each step,
computed and applied efforts are published into canonical storage and scattered
to the articulation and physical-backend force buffers according to the
resolved final-writer policy.

The :meth:`~NewtonActuatorAdapter.from_usd` classmethod parses
``NewtonActuator`` USD prims for the PhysX backend; Newton supplies the
already-built ``model.actuators``.  Dynamic actuator parameter writes are
routed directly to the controller and clamping arrays owned by the active
binding.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from newton.actuators import Actuator, Clamping, Delay

from isaaclab.actuators import ImplicitActuator

from .kernels import (
    build_implicit_dof_mask,
    build_per_dof_env_mask_kernel,
    gather_canonical_range_to_controller,
    recompute_dc_motor_corner_velocity,
    scatter_gain_kernel,
    set_mask_int64_kernel,
    set_mask_kernel,
    set_mask_slice_kernel,
    zero_at_indices_kernel,
)

_DEFAULTS_DEPRECATION_EMITTED = False


@wp.kernel(enable_backward=False)
def _expand_env_major_indices(
    local_indices: wp.array(dtype=wp.int32), env_stride: int, indices: wp.array(dtype=wp.uint32)
):
    """Expand compact local DOF IDs into an env-major physical index array."""
    index = wp.tid()
    local_count = local_indices.shape[0]
    indices[index] = wp.uint32(index // local_count * env_stride + local_indices[index % local_count])


@wp.kernel(enable_backward=False)
def _expand_env_major_values(local_values: wp.array(dtype=wp.float32), values: wp.array(dtype=wp.float32)):
    """Repeat compact per-DOF values over env-major controller storage."""
    values[wp.tid()] = local_values[wp.tid() % local_values.shape[0]]


@wp.kernel(enable_backward=False)
def _expand_env_major_int_values(local_values: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.int32)):
    """Repeat compact integer values over env-major controller storage."""
    values[wp.tid()] = local_values[wp.tid() % local_values.shape[0]]


if TYPE_CHECKING:
    from isaaclab.actuators.actuator_collection import _ArticulationBinding


@dataclass(frozen=True)
class _NativeRangeBinding:
    """One direct or staged native controller range for canonical storage.

    Direct ranges alias complete exact-type canonical parameter and output
    arrays.  Staged ranges retain compact routing into persistent model-global
    Newton arrays and publish their outputs after the physical actuator step.
    """

    group_names: tuple[str, ...]
    actuator_type: type
    actuator: Actuator
    direct: bool
    canonical_parameters: Mapping[str, wp.array]
    canonical_computed_effort: wp.array
    canonical_applied_effort: wp.array
    staging: object | None = None
    validation_error_flag: wp.array | None = None
    handle: object | None = None
    compact_joint_ids: wp.array | None = None
    canonical_slots: wp.array | None = None
    effort_owner_slots: wp.array | None = None
    computed_owner_slots: wp.array | None = None
    applied_owner_slots: wp.array | None = None
    controller_local_slots: wp.array | None = None
    controller_stride: int = 0
    dof_offset: int = 0
    has_joint_ordering: bool = False
    user_to_backend: wp.array | None = None


@dataclass
class _GlobalNativeActuatorBinding:
    """Registration state for one persistent model-global staged actuator."""

    actuator: Actuator
    parameters: list[tuple[object, str, wp.array]]
    computed_effort: wp.array | None
    applied_effort: wp.array | None
    registrations: set[object]


@dataclass
class _PointerMutation:
    """One reversible exact Newton pointer replacement for direct binding."""

    owner: object
    attr: str
    original: object
    installed: object
    registrations: set[object]


@dataclass
class _DirectPointerBinding:
    """Generation-scoped reverse mutation log for one direct actuator alias."""

    actuator: Actuator
    mutations: list[_PointerMutation]
    registrations: set[object]


@dataclass(frozen=True)
class _HostedDirectBinding:
    """Canonical arrays eligible for construction-time hosted direct aliases."""

    parameters: Mapping[str, wp.array]
    computed_effort: wp.array
    applied_effort: wp.array


@dataclass(frozen=True)
class _HostedActuatorRecipe:
    """One parsed joint actuator with all reusable resolutions cached."""

    parsed: Any
    signature: tuple
    controller_arguments: Mapping[str, Any]
    component_arguments: tuple[tuple[type, Mapping[str, Any]], ...]


# ---------------------------------------------------------------------------
# Abstract base — backend-independent logic
# ---------------------------------------------------------------------------


class NewtonActuatorAdapter:
    """Adapter that wraps a list of :class:`newton.actuators.Actuator`.

    Owns actuator-state lifecycle, DOF-to-actuator bookkeeping, stepping,
    reset, and canonical range publication.
    """

    @dataclass(frozen=True)
    class ArticulationBinding:
        """Newton setup state for one articulation.

        Returned by :meth:`bind_articulation`. Bundles the implicit-actuator
        compatibility mask and its memory owner with the direct or staged
        canonical ranges installed for this candidate binding. Treat
        :attr:`ranges` as opaque and pass it unchanged to the adapter's range
        methods. The ranges remain valid only until the articulation binding
        is invalidated.
        """

        implicit_dof_mask: wp.array(dtype=wp.int32)
        """Compatibility mask, shape ``(num_joints,)``: ``1`` for implicit-actuator DOFs, ``0`` otherwise."""

        implicit_dof_mask_owner: torch.Tensor
        """Torch owner of the memory aliased by :attr:`implicit_dof_mask`, shape ``(num_joints,)``."""

        ranges: tuple[_NativeRangeBinding, ...]
        """Opaque direct or staged native ranges retained for the binding lifetime."""

    def __init__(
        self,
        actuators: list[Actuator],
        num_envs: int,
        num_joints: int,
        dof_offset: int,
        device: str,
        actuator_keys: Sequence[tuple] | None = None,
        dof_signatures: Mapping[int, tuple | Sequence[tuple]] | None = None,
        actuator_dof_indices: Mapping[tuple, Sequence[int]] | None = None,
        owns_actuators: bool = False,
    ) -> None:
        """Initialize an adapter over finalized Newton actuator objects.

        Args:
            actuators: Finalized Newton actuators in model-builder order.
            num_envs: Number of simulation environments.
            num_joints: Number of model DOFs per environment used as the
                env-major actuator-index stride.
            dof_offset: First managed DOF in the adapter's env-major index space.
            device: Warp device string, for example ``"cuda:0"``.
            actuator_keys: Structural model-builder key for each item in
                :paramref:`actuators`. When omitted, keys are read from the
                actuator metadata installed during model construction.
            dof_signatures: Authored structural-key occurrences for each
                environment-zero DOF.
            actuator_dof_indices: Authored environment-zero DOF order for each
                structural actuator key.
            owns_actuators: Whether the adapter constructed the actuator objects
                and may discard their direct aliases instead of restoring
                manager-owned pointers.
        """
        self.actuators = actuators
        self.num_joints = num_joints

        self._num_envs = num_envs
        self._dof_offset = dof_offset
        self._device = device
        self._global_native_bindings: dict[int, _GlobalNativeActuatorBinding] = {}
        self._direct_pointer_bindings: dict[int, _DirectPointerBinding] = {}
        # Hosted adapters construct standalone Newton objects from USD during
        # candidate preparation. Their direct aliases do not need restoration:
        # invalidation discards the complete adapter. Native adapters borrow a
        # manager-owned model and retain reverse mutations instead.
        self._owns_actuators = owns_actuators
        keys = actuator_keys or tuple(getattr(actuator, "_isaaclab_structural_key", None) for actuator in actuators)
        if len(keys) != len(actuators) or any(key is None for key in keys):
            raise RuntimeError("Newton actuator structural keys are required for canonical range binding.")
        self._actuators_by_signature = dict(zip(keys, actuators, strict=True))
        if len(self._actuators_by_signature) != len(actuators):
            raise RuntimeError("Newton model contains duplicate actuator structural entries.")
        # Keep the caller-provided shape for compatibility, but derive an
        # immutable occurrence multimap for binding.  A physical DOF may be
        # authored more than once in one Newton controller (``[0, 0]`` is a
        # legal ModelBuilder entry), so a one-value ``dof -> signature`` map
        # is lossy.
        self._dof_signatures = dict(dof_signatures or {})
        self._actuator_dof_indices = {key: tuple(indices) for key, indices in (actuator_dof_indices or {}).items()}
        for key, actuator in self._actuators_by_signature.items():
            if key not in self._actuator_dof_indices:
                local_indices = getattr(actuator, "_isaaclab_env_zero_dof_indices", None)
                if local_indices is not None:
                    self._actuator_dof_indices[key] = tuple(local_indices)
        self._actuator_local_slots = {
            key: {dof_index: local_slot for local_slot, dof_index in enumerate(indices)}
            for key, indices in self._actuator_dof_indices.items()
        }
        self._actuator_occurrence_slots = {
            key: self._slots_by_dof(indices) for key, indices in self._actuator_dof_indices.items()
        }
        self._joint_signatures: dict[str, tuple] = {}

        # Compatibility-only; production routing uses canonical range maps.
        # Do not read world-sized actuator indices back from device here.
        self.joint_indices: torch.Tensor | slice = slice(None)

        self._states_a = [act.state() for act in actuators]
        self._states_b = [act.state() for act in actuators]

    @staticmethod
    def _slots_by_dof(indices: Sequence[int]) -> dict[int, tuple[int, ...]]:
        """Return immutable authored controller slots grouped by physical DOF."""
        slots: dict[int, list[int]] = {}
        for local_slot, dof_index in enumerate(indices):
            slots.setdefault(int(dof_index), []).append(local_slot)
        return {dof_index: tuple(dof_slots) for dof_index, dof_slots in slots.items()}

    @staticmethod
    def _signature_occurrences(value: tuple | Sequence[tuple] | None) -> tuple[tuple, ...]:
        """Normalize legacy scalar keys and occurrence-aware key sequences."""
        if value is None:
            return ()
        # Structural keys themselves are tuples whose first entry is a
        # controller class. A sequence of structural keys therefore starts
        # with a tuple, whereas a legacy scalar key starts with a class.
        if isinstance(value, tuple) and (not value or not isinstance(value[0], tuple)):
            return (value,)
        return tuple(value)

    def finalize(self, sim_control: Any) -> None:
        """Run the legacy no-op finalization hook.

        Args:
            sim_control: Retained for compatibility. Articulation binding owns
                output transport and does not add fields to a model-global
                Newton control object.
        """
        del sim_control

    def step(self, sim_state: Any, sim_control: Any, dt: float) -> None:
        """Clear actuator-owned force slots, step all actuators, and swap states.

        Args:
            sim_state: Object with ``joint_q``, ``joint_qd``, etc.
                Newton ``State`` on the Newton backend,
                :class:`~isaaclab_newton.actuators.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            sim_control: Object with ``joint_f``, ``joint_target_pos``, etc.
                Newton ``Control`` on the Newton backend,
                :class:`~isaaclab_newton.actuators.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            dt: Physics timestep [s].
        """
        for act in self.actuators:
            wp.launch(
                zero_at_indices_kernel,
                dim=act.effort_indices.shape[0],
                inputs=[sim_control.joint_f, act.effort_indices],
                device=self._device,
            )
        for act, sa, sb in zip(self.actuators, self._states_a, self._states_b):
            act.step(sim_state, sim_control, sa, sb, dt=dt)
        self._swap_state_buffers()

    def _swap_state_buffers(self) -> None:
        """Advance the actuator state ping-pong after an eager step or graph replay."""
        self._states_a, self._states_b = self._states_b, self._states_a

    def reset(
        self,
        env_ids: Sequence[int]
        | torch.Tensor
        | wp.array(dtype=wp.int32)
        | wp.array(dtype=wp.int64)
        | slice
        | None = None,
    ) -> None:
        """Reset actuator states for the given environments.

        Args:
            env_ids: Environment indices to reset. ``None`` (or
                ``slice(None)``, which IsaacLab callers sometimes pass)
                resets all environments. Otherwise accepts a Torch tensor,
                signed Warp index array, integer sequence, or slice.

        Newton's :meth:`Actuator.State.reset` expects a per-DOF boolean
        mask of length ``num_actuators`` (= ``num_envs * dofs_per_actuator``),
        not a per-env mask — each entry gates the corresponding column of
        the actuator's state buffers (delay queue, controller integral,
        etc.). We therefore build a per-actuator per-DOF mask from the
        env mask before delegating to each state.
        """
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            for sa, sb in zip(self._states_a, self._states_b):
                if sa is not None:
                    sa.reset(None)
                if sb is not None:
                    sb.reset(None)
            return

        env_mask = wp.zeros(self._num_envs, dtype=wp.bool, device=self._device)
        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self._num_envs)
            count = len(range(start, stop, step))
            if count == 0:
                return
            wp.launch(
                set_mask_slice_kernel,
                dim=count,
                inputs=[env_mask, start, step],
                device=self._device,
            )
        elif isinstance(env_ids, wp.array):
            if env_ids.ndim != 1:
                raise ValueError("Warp reset indices must be a one-dimensional array")
            if env_ids.dtype not in (wp.int32, wp.int64):
                raise TypeError("Warp reset indices must have dtype wp.int32 or wp.int64")
            if wp.get_device(env_ids.device) != wp.get_device(self._device):
                raise ValueError(f"Warp reset indices must be on adapter device {self._device!r}")
            if env_ids.shape[0] == 0:
                return
            set_mask = set_mask_kernel if env_ids.dtype == wp.int32 else set_mask_int64_kernel
            wp.launch(set_mask, dim=env_ids.shape[0], inputs=[env_mask, env_ids], device=self._device)
        elif isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            if env_ids.ndim != 1:
                raise ValueError("Torch reset indices must be a one-dimensional tensor")
            if env_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("Torch reset indices must have dtype torch.int32 or torch.int64")
            idx = wp.from_torch(env_ids.to(device=self._device).contiguous().to(torch.int32), dtype=wp.int32)
            wp.launch(set_mask_kernel, dim=idx.shape[0], inputs=[env_mask, idx], device=self._device)
        else:
            if len(env_ids) == 0:
                return
            idx = wp.array(list(env_ids), dtype=wp.int32, device=self._device)
            wp.launch(set_mask_kernel, dim=idx.shape[0], inputs=[env_mask, idx], device=self._device)

        for act, sa, sb in zip(self.actuators, self._states_a, self._states_b):
            per_dof_mask = wp.zeros(act.indices.shape[0], dtype=wp.bool, device=self._device)
            wp.launch(
                build_per_dof_env_mask_kernel,
                dim=act.indices.shape[0],
                inputs=[act.indices, env_mask, self._dof_offset, self.num_joints, per_dof_mask],
                device=self._device,
            )
            if sa is not None:
                sa.reset(per_dof_mask)
            if sb is not None:
                sb.reset(per_dof_mask)

    def bind_articulation(
        self,
        binding: _ArticulationBinding,
        *,
        dof_offset: int,
        joint_user_to_backend_indices: Sequence[int] | None = None,
        structural_occurrences: Mapping[int, Sequence[tuple]] | None = None,
    ) -> ArticulationBinding:
        """Bind private canonical ranges for one articulation.

        Full exact-type ranges install direct aliases for parameter, computed,
        and applied effort storage. Partial or reordered ranges retain
        model-global Newton storage as pointer-stable staging. Borrowed direct
        aliases are tracked in a reverse mutation log and restored when their
        final articulation registration is released.

        Args:
            binding: Unpublished private actuator binding. No facade is read
                while aliases are installed.
            dof_offset: Offset of this articulation's DOFs in the adapter's
                env-major global index space (``0`` on PhysX, view-dependent
                on Newton).
            joint_user_to_backend_indices: Complete permutation from public
                joint indices to adapter-local joint indices. ``None``
                preserves adapter-local order (the PhysX case, whose adapter
                is already built from public joint names).
            structural_occurrences: Binding-local authored structural keys by
                adapter-local joint index. Native Newton callers obtain this
                stream from the articulation's USD actuator prims before the
                builder groups compatible entries. ``None`` preserves the
                legacy metadata fallback for narrow test seams.

        Returns:
            The bundled :class:`ArticulationBinding` for this articulation.
        """
        if binding.groups is None:
            raise RuntimeError("Newton canonical binding requires private actuator groups.")
        ranges = self._bind_canonical_ranges(
            binding,
            dof_offset=dof_offset,
            joint_user_to_backend_indices=joint_user_to_backend_indices,
            structural_occurrences=structural_occurrences,
        )
        implicit_dof_mask, implicit_dof_mask_owner = build_implicit_dof_mask(
            dict(binding.groups),
            binding.layout.num_joints,
            self._device,
            group_layouts=binding.layout.group_layouts,
        )
        return self.ArticulationBinding(
            implicit_dof_mask=implicit_dof_mask,
            implicit_dof_mask_owner=implicit_dof_mask_owner,
            ranges=ranges,
        )

    def _bind_canonical_ranges(
        self,
        binding: _ArticulationBinding,
        *,
        dof_offset: int,
        joint_user_to_backend_indices: Sequence[int] | None,
        structural_occurrences: Mapping[int, Sequence[tuple]] | None,
    ) -> tuple[_NativeRangeBinding, ...]:
        """Install canonical ranges atomically and reverse partial aliases on failure."""
        installed_ranges: list[_NativeRangeBinding] = []
        try:
            self._bind_canonical_ranges_install(
                binding,
                dof_offset=dof_offset,
                joint_user_to_backend_indices=joint_user_to_backend_indices,
                structural_occurrences=structural_occurrences,
                installed_ranges=installed_ranges,
            )
        except Exception as error:
            try:
                self._rollback_articulation_ranges(tuple(reversed(installed_ranges)))
            except Exception as cleanup_error:
                error.add_note(f"Canonical-range rollback also failed: {cleanup_error!r}")
            raise
        return tuple(installed_ranges)

    def _bind_canonical_ranges_install(
        self,
        binding: _ArticulationBinding,
        *,
        dof_offset: int,
        joint_user_to_backend_indices: Sequence[int] | None,
        structural_occurrences: Mapping[int, Sequence[tuple]] | None,
        installed_ranges: list[_NativeRangeBinding],
    ) -> None:
        """Bind structural native segments to exact-type canonical storage."""
        assert binding.groups is not None
        # Build and validate immutable structural routing once before mutating
        # a controller pointer or registering staging. This makes a later
        # missing segment fail atomically instead of leaking an earlier direct
        # alias into candidate-owned canonical memory.
        occurrence_assignments = self._plan_structural_occurrences(
            binding,
            dof_offset=dof_offset,
            joint_user_to_backend_indices=joint_user_to_backend_indices,
            structural_occurrences=structural_occurrences,
        )
        for signature in {signature for signature, _ in occurrence_assignments.values()}:
            try:
                actuator = self._actuators_by_signature[signature]
            except KeyError as error:
                raise RuntimeError(f"No Newton actuator matches native structural occurrence {signature!r}.") from error
            effort_indices = getattr(actuator, "effort_indices", actuator.indices)
            if effort_indices is not actuator.indices:
                raise NotImplementedError(
                    "Newton native actuator binding does not support coupled effort routing: "
                    "actuator.effort_indices must be actuator.indices."
                )
        structural_plan: list[tuple[type, Any, tuple[Any, ...], tuple[Any, ...], wp.array]] = []
        for actuator_type, type_layout in binding.layout.type_layouts.items():
            native_groups = tuple(
                group
                for group in binding.layout.group_layouts
                if group.actuator_type is actuator_type and group.name in binding.native_group_names
            )
            if not native_groups:
                continue
            segments = self._structural_segments(
                binding,
                native_groups,
                occurrence_assignments=occurrence_assignments,
            )
            for signature, group_names, _, _, _ in segments:
                if signature not in self._actuators_by_signature:
                    raise RuntimeError(
                        f"No Newton actuator matches native structural segment {group_names} ({signature!r})."
                    )
            # Effort, computed-effort, and applied-effort have the same
            # configuration-order final-writer policy. Build and upload that
            # row once per exact type, then alias it from every segment.
            force_owner_slots = self._range_owner_slots(binding, actuator_type, "effort")
            structural_plan.append((actuator_type, type_layout, native_groups, segments, force_owner_slots))

        user_to_backend = wp.array(
            joint_user_to_backend_indices
            if joint_user_to_backend_indices is not None
            else list(range(binding.layout.num_joints)),
            dtype=wp.int32,
            device=self._device,
        )
        # Direct pointer mutations are deliberately delayed until every range
        # has allocated and validated its complete descriptor set.  Thus an
        # allocation or routing error in a later structural segment cannot
        # expose an earlier controller to a partially constructed binding.
        direct_pending: list[tuple[_NativeRangeBinding, tuple[_PointerMutation, ...]]] = []
        for actuator_type, type_layout, native_groups, segments, force_owner_slots in structural_plan:
            group = binding.groups[native_groups[0].name]
            parameter_binding = group.__dict__.get("_parameter_binding")
            if parameter_binding is None:
                raise RuntimeError(f"Native actuator type {actuator_type.__name__} has no canonical parameter storage.")
            arrays = parameter_binding.arrays
            for signature, group_names, compact_joint_ids, canonical_slots, controller_slots in segments:
                try:
                    actuator = self._actuators_by_signature[signature]
                except KeyError as error:
                    raise RuntimeError(
                        f"No Newton actuator matches native structural segment {group_names} ({signature!r})."
                    ) from error
                whole_type = tuple(canonical_slots) == tuple(range(type_layout.num_dofs))
                expected_size = type_layout.num_worlds * type_layout.num_dofs
                direct = (
                    whole_type
                    # A direct alias exposes every canonical occurrence to
                    # one controller array. Overlapping physical DOFs need
                    # staged occurrence routing so a losing group can retain
                    # its canonical value without changing the final owner.
                    and len(set(type_layout.compact_joint_indices)) == type_layout.num_dofs
                    and actuator.indices.shape[0] == expected_size
                    # Canonical direct aliases are only valid in public joint
                    # order.  Nonidentity articulation order uses staging;
                    # exact structural matching establishes the identity-order
                    # controller layout during builder construction.
                    and (
                        joint_user_to_backend_indices is None
                        or all(
                            user_joint == backend_joint
                            for user_joint, backend_joint in enumerate(joint_user_to_backend_indices)
                        )
                    )
                    and getattr(self, "_actuator_dof_indices", {}).get(signature)
                    == tuple(dof_offset + joint_id for joint_id in type_layout.compact_joint_indices)
                )
                canonical_parameters = {name: value.warp for name, value in arrays.items()}
                computed = arrays["computed_effort"].warp
                applied = arrays["applied_effort"].warp
                # Allocate every descriptor-owned array before mutating a
                # controller.  The outer range transaction can only restore
                # descriptors that exist, so a failing allocation must not
                # leave a direct alias or staged registration behind.
                compact_joint_ids_array = wp.array(compact_joint_ids, dtype=wp.int32, device=self._device)
                canonical_slots_array = wp.array(canonical_slots, dtype=wp.int32, device=self._device)
                controller_local_slots = None
                controller_stride = 0
                if not direct:
                    controller_indices = getattr(self, "_actuator_dof_indices", {}).get(signature)
                    if controller_indices is None:
                        raise RuntimeError(
                            f"No env-0 controller layout metadata is available for native segment {group_names}."
                        )
                    controller_local_slots = wp.array(controller_slots, dtype=wp.int32, device=self._device)
                    controller_stride = len(controller_indices)
                handle = object()
                staging = None
                direct_mutations: tuple[_PointerMutation, ...] = ()
                if direct:
                    direct_mutations = self._prepare_direct_pointer_mutations(
                        actuator, canonical_parameters, computed, applied
                    )
                try:
                    if not direct:
                        staging = self._register_staged_actuator(actuator, handle)
                    range_binding = _NativeRangeBinding(
                        group_names=group_names,
                        actuator_type=actuator_type,
                        actuator=actuator,
                        direct=direct,
                        canonical_parameters=canonical_parameters,
                        canonical_computed_effort=computed,
                        canonical_applied_effort=applied,
                        staging=staging,
                        handle=handle,
                        compact_joint_ids=compact_joint_ids_array,
                        canonical_slots=canonical_slots_array,
                        effort_owner_slots=force_owner_slots,
                        computed_owner_slots=force_owner_slots,
                        applied_owner_slots=force_owner_slots,
                        controller_local_slots=controller_local_slots,
                        controller_stride=controller_stride,
                        dof_offset=dof_offset,
                        has_joint_ordering=joint_user_to_backend_indices is not None,
                        user_to_backend=user_to_backend,
                    )
                    installed_ranges.append(range_binding)
                    if direct:
                        direct_pending.append((range_binding, direct_mutations))
                except Exception:
                    if staging is not None:
                        self._unregister_staged_handle(actuator, handle)
                    raise
        for range_binding, direct_mutations in direct_pending:
            self._install_direct_pointer_binding(range_binding.actuator, range_binding.handle, direct_mutations)
        if getattr(self, "_owns_actuators", False):
            for range_binding, _ in direct_pending:
                # Hosted actuators were constructed for this candidate alone.
                # Commit only after every direct install succeeds so a later
                # actuator failure can still roll the complete batch back.
                # Once the batch is published there is no external model to
                # restore, so do not retain throwaway original Newton arrays.
                self._discard_owned_direct_pointer_binding(range_binding.actuator, range_binding.handle)

    def _rollback_articulation_ranges(self, ranges: Sequence[_NativeRangeBinding]) -> None:
        """Undo a failed bind and reverse every installed borrowed alias."""
        errors: list[Exception] = []
        for range_binding in ranges:
            try:
                if range_binding.direct and range_binding.handle is not None:
                    self._release_direct_pointer_binding(
                        range_binding.actuator, range_binding.handle, force_restore=True
                    )
                elif not range_binding.direct and range_binding.handle is not None:
                    self._unregister_staged_handle(range_binding.actuator, range_binding.handle)
            except Exception as error:
                errors.append(error)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(f"Additional canonical-range rollback failure: {error!r}")
            raise primary

    def _discard_owned_direct_pointer_binding(self, actuator: Actuator, handle: object | None) -> None:
        """Discard a successful hosted direct log and its superseded owned arrays."""
        if handle is None:
            return
        binding = self._direct_pointer_bindings.get(id(actuator))
        if binding is None:
            return
        if binding.registrations != {handle}:
            raise RuntimeError("Hosted direct actuator unexpectedly has shared registrations.")
        del self._direct_pointer_bindings[id(actuator)]

    def _structural_segments(
        self,
        binding: _ArticulationBinding,
        groups: Sequence[Any],
        *,
        occurrence_assignments: Mapping[tuple[str, int], tuple[tuple, int]],
    ) -> tuple[tuple[tuple, tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]:
        """Collect occurrence-assigned exact-type slots in config order."""
        segments: dict[tuple, tuple[list[str], list[int], list[int], list[int]]] = {}
        for group in groups:
            for local_slot, (joint_id, joint_name) in enumerate(
                zip(group.joint_indices, group.joint_names, strict=True)
            ):
                del joint_name
                signature, controller_slot = occurrence_assignments[group.name, local_slot]
                canonical_slot = group.type_slice.start + local_slot
                names, joints, slots, controller_slots = segments.setdefault(signature, ([], [], [], []))
                if not names or names[-1] != group.name:
                    names.append(group.name)
                joints.append(joint_id)
                slots.append(canonical_slot)
                controller_slots.append(controller_slot)
        return tuple(
            (signature, tuple(names), tuple(joints), tuple(slots), tuple(controller_slots))
            for signature, (names, joints, slots, controller_slots) in segments.items()
        )

    def _plan_structural_occurrences(
        self,
        binding: _ArticulationBinding,
        *,
        dof_offset: int,
        joint_user_to_backend_indices: Sequence[int] | None,
        structural_occurrences: Mapping[int, Sequence[tuple]] | None,
    ) -> dict[tuple[str, int], tuple[tuple, int]]:
        """Match native canonical occurrences to authored Newton controller slots.

        The plan is intentionally completed before any descriptor allocation,
        pointer replacement, or staged registration. Both the source and
        destination order are global configuration order, not physical-DOF
        order, so duplicate physical DOFs retain separate controller state,
        gains, and outputs.
        """
        assignments: dict[tuple[str, int], tuple[tuple, int]] = {}
        signature_cursors: dict[int, int] = {}
        slot_cursors: dict[tuple[tuple, int], int] = {}
        required: dict[int, int] = {}
        available_occurrences: dict[int, tuple[tuple, ...]] = {}
        for group in binding.layout.group_layouts:
            for local_slot, (joint_id, joint_name) in enumerate(
                zip(group.joint_indices, group.joint_names, strict=True)
            ):
                backend_joint = (
                    joint_id if joint_user_to_backend_indices is None else joint_user_to_backend_indices[joint_id]
                )
                global_dof = dof_offset + backend_joint
                if structural_occurrences is not None:
                    try:
                        structural_keys = tuple(structural_occurrences[backend_joint])
                    except KeyError as error:
                        raise RuntimeError(
                            "Newton binding-local structural occurrence metadata is missing for native "
                            f"joint {joint_name!r} at local DOF {backend_joint}."
                        ) from error
                    builder_keys = self._signature_occurrences(self._dof_signatures.get(global_dof))
                    if Counter(structural_keys) != Counter(builder_keys):
                        raise RuntimeError(
                            "Newton controller occurrence metadata for Lab-covered "
                            f"joint {joint_name!r} at global DOF {global_dof} does not match "
                            "the finalized model actuator entries."
                        )
                else:
                    structural_keys = self._signature_occurrences(self._joint_signatures.get(joint_name))
                    if not structural_keys:
                        structural_keys = self._signature_occurrences(self._dof_signatures.get(global_dof))
                previous_keys = available_occurrences.setdefault(global_dof, structural_keys)
                if previous_keys != structural_keys:
                    raise RuntimeError(
                        "Newton structural occurrence sources disagree for Lab-covered "
                        f"joint {joint_name!r} at global DOF {global_dof}."
                    )
                if group.name not in binding.native_group_names:
                    continue
                occurrence_index = signature_cursors.get(global_dof, 0)
                if occurrence_index >= len(structural_keys):
                    raise RuntimeError(
                        "Newton controller occurrence cardinality is insufficient for native "
                        f"joint {joint_name!r} at global DOF {global_dof}."
                    )
                signature = structural_keys[occurrence_index]
                signature_cursors[global_dof] = occurrence_index + 1
                slots_by_dof = getattr(self, "_actuator_occurrence_slots", {}).get(signature)
                if slots_by_dof is None:
                    controller_indices = getattr(self, "_actuator_dof_indices", {}).get(signature)
                    if controller_indices is None:
                        raise RuntimeError(
                            f"No env-0 controller layout metadata is available for native joint {joint_name!r}."
                        )
                    slots_by_dof = self._slots_by_dof(controller_indices)
                slots = slots_by_dof.get(global_dof, ())
                slot_key = signature, global_dof
                slot_index = slot_cursors.get(slot_key, 0)
                if slot_index >= len(slots):
                    raise RuntimeError(
                        "Newton controller slot cardinality is insufficient for native "
                        f"joint {joint_name!r} at global DOF {global_dof}."
                    )
                slot_cursors[slot_key] = slot_index + 1
                assignments[group.name, local_slot] = signature, slots[slot_index]
                required[global_dof] = required.get(global_dof, 0) + 1

        for global_dof, keys in available_occurrences.items():
            count = required.get(global_dof, 0)
            if len(keys) != count:
                raise RuntimeError(
                    "Newton controller occurrence cardinality does not exactly match native canonical groups "
                    f"at global DOF {global_dof}: expected {len(keys)}, got {count}."
                )
        return assignments

    def _range_owner_slots(
        self,
        binding: _ArticulationBinding,
        actuator_type: type,
        field: str,
    ) -> wp.array:
        """Build one final-writer row for a native exact-type range.

        Every entry is the compact slot of the final native writer for this
        range, or ``-1``. A later Lab group clears the earlier native writer,
        so the resulting row can safely gate physical and telemetry scatter
        without depending on range iteration order.
        """
        owner_slots = [-1] * binding.layout.num_joints
        for group in binding.layout.group_layouts:
            native = group.name in binding.native_group_names
            group_fields = (
                {"position", "velocity", "effort", "computed_effort", "applied_effort"}
                if native or issubclass(group.actuator_type, ImplicitActuator)
                else {"effort", "computed_effort", "applied_effort"}
            )
            if field not in group_fields:
                continue
            joint_indices = getattr(group, "joint_indices", None)
            type_slice = getattr(group, "type_slice", None)
            if joint_indices is None:
                type_layout = binding.layout.type_layouts[group.actuator_type]
                compact_indices = type_layout.compact_joint_indices
                type_slice = type_slice or slice(0, len(compact_indices))
                joint_indices = compact_indices[type_slice]
            joint_ids = tuple(int(joint_id) for joint_id in joint_indices)
            if native and group.actuator_type is actuator_type:
                if type_slice is None:
                    type_slice = slice(0, len(joint_ids))
                for joint_id, compact_slot in zip(joint_ids, range(type_slice.start, type_slice.stop), strict=True):
                    owner_slots[joint_id] = compact_slot
            else:
                for joint_id in joint_ids:
                    owner_slots[joint_id] = -1
        return wp.array(owner_slots, dtype=wp.int32, device=self._device)

    def _register_staged_actuator(self, actuator: Actuator, handle: object) -> _GlobalNativeActuatorBinding:
        """Register an existing model-global controller as persistent staging.

        Newton already owns controller, clamping, computed, and applied arrays
        at model-global scope. Staged ranges gather into disjoint immutable
        slots of those arrays, then scatter the same slots back after one
        physical actuator step. Replacing them with clones adds allocation and
        makes multi-articulation ownership needlessly fragile.
        """
        registry = self._global_native_bindings
        key = id(actuator)
        global_binding = registry.get(key)
        if global_binding is None:
            parameters = [
                (component, name, value)
                for component in (actuator.controller, *(actuator.clamping or ()))
                for name in (
                    "kp",
                    "kd",
                    "max_effort",
                    "max_motor_effort",
                    "velocity_limit",
                    "saturation_effort",
                )
                if isinstance((value := getattr(component, name, None)), wp.array)
            ]
            global_binding = _GlobalNativeActuatorBinding(
                actuator=actuator,
                parameters=parameters,
                computed_effort=getattr(actuator, "_computed_forces", None),
                applied_effort=getattr(actuator, "_applied_forces", None),
                registrations=set(),
            )
            registry[key] = global_binding
        global_binding.registrations.add(handle)
        return global_binding

    def unregister_articulation_ranges(self, ranges: Sequence[_NativeRangeBinding]) -> None:
        """Release ranges and restore borrowed direct pointers after their final user.

        Persistent staged arrays remain Newton-owned throughout their
        registrations.  Teardown performs every inverse pointer write it can
        and retains failed direct restores so a later teardown can retry them.

        Args:
            ranges: Opaque ranges from :attr:`ArticulationBinding.ranges`.
        """
        errors: list[Exception] = []
        for range_binding in ranges:
            try:
                handle = range_binding.handle
                if range_binding.direct:
                    if handle is not None:
                        self._release_direct_pointer_binding(range_binding.actuator, handle)
                    continue
                if handle is None:
                    continue
                self._unregister_staged_handle(range_binding.actuator, handle)
            except Exception as error:
                errors.append(error)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(f"Additional canonical-range teardown failure: {error!r}")
            raise primary

    def _unregister_staged_handle(self, actuator: Actuator, handle: object) -> None:
        """Drop one staged registration and restore pointers after its last user."""
        key = id(actuator)
        global_binding = self._global_native_bindings.get(key)
        if global_binding is None:
            return
        global_binding.registrations.discard(handle)
        if global_binding.registrations:
            return
        del self._global_native_bindings[key]

    def gather_staged_ranges(self, ranges: Sequence[_NativeRangeBinding]) -> None:
        """Refresh persistent staged controller parameters from canonical storage.

        Args:
            ranges: Opaque ranges from :attr:`ArticulationBinding.ranges`.
        """
        for range_binding in ranges:
            if range_binding.direct or range_binding.staging is None:
                continue
            assert range_binding.compact_joint_ids is not None
            assert range_binding.canonical_slots is not None
            assert range_binding.controller_local_slots is not None
            refreshed_dc = False
            for component, name, staged in range_binding.staging.parameters:
                canonical_name = {
                    "kp": "stiffness",
                    "kd": "damping",
                    "max_effort": "effort_limit",
                    "max_motor_effort": "effort_limit",
                    "velocity_limit": "velocity_limit",
                    "saturation_effort": "saturation_effort",
                }.get(name)
                if canonical_name is None or canonical_name not in range_binding.canonical_parameters:
                    continue
                wp.launch(
                    gather_canonical_range_to_controller,
                    dim=(range_binding.canonical_computed_effort.shape[0], range_binding.compact_joint_ids.shape[0]),
                    inputs=[
                        range_binding.canonical_parameters[canonical_name],
                        range_binding.canonical_slots,
                        range_binding.controller_local_slots,
                        range_binding.controller_stride,
                        staged,
                    ],
                    device=self._device,
                )
                refreshed_dc = refreshed_dc or name in {
                    "max_motor_effort",
                    "velocity_limit",
                    "saturation_effort",
                }
            if refreshed_dc:
                self._refresh_dc_motor_corner_velocity(range_binding.actuator)

    def publish_outputs(
        self,
        ranges: Sequence[_NativeRangeBinding],
        joint_computed_effort: wp.array2d(dtype=wp.float32),
        joint_applied_effort: wp.array2d(dtype=wp.float32),
        joint_command_effort: wp.array2d(dtype=wp.float32) | None = None,
        backend_effort: wp.array2d(dtype=wp.float32) | None = None,
        backend_computed_effort: wp.array2d(dtype=wp.float32) | None = None,
        user_to_backend: wp.array(dtype=wp.int32) | None = None,
        sim_buffers_are_backend_order: bool = False,
    ) -> None:
        """Publish direct and staged outputs to canonical, joint, and physical buffers.

        Computed and applied effort are written independently to canonical and
        articulation joint storage.  When supplied, ``backend_effort`` is
        restored from canonical applied effort after the actuator step, while
        ``backend_computed_effort`` receives computed effort.  All scatters
        obey the range final-writer ownership slots.

        Args:
            ranges: Opaque ranges from :attr:`ArticulationBinding.ranges`.
            joint_computed_effort: Computed effort [N or N·m, depending on
                joint type] in public joint order, shape
                ``(num_envs, num_joints)``.
            joint_applied_effort: Applied effort [N or N·m, depending on
                joint type] in public joint order, shape
                ``(num_envs, num_joints)``.
            joint_command_effort: Optional processed effort command [N or N·m,
                depending on joint type] in public joint order, shape
                ``(num_envs, num_joints)``.
            backend_effort: Optional physical effort buffer [N or N·m,
                depending on joint type], shape ``(num_envs, num_joints)``.
            backend_computed_effort: Optional backend telemetry buffer [N or
                N·m, depending on joint type], shape
                ``(num_envs, num_joints)``.
            user_to_backend: Optional public-to-backend joint permutation,
                shape ``(num_joints,)``.
            sim_buffers_are_backend_order: Whether backend effort buffers use
                backend rather than public joint order.
        """
        from .kernels import (
            scatter_canonical_range_to_backend,
            scatter_canonical_range_to_joint,
            scatter_controller_range_to_canonical,
        )

        for range_binding in ranges:
            compact_joint_ids = range_binding.compact_joint_ids
            canonical_slots = range_binding.canonical_slots
            if compact_joint_ids is None or canonical_slots is None:
                continue
            if range_binding.direct:
                if range_binding.actuator._applied_forces is None:
                    wp.copy(range_binding.canonical_applied_effort, range_binding.canonical_computed_effort)
            else:
                assert range_binding.staging is not None
                assert range_binding.controller_local_slots is not None
                staging = range_binding.staging
                computed = staging.computed_effort
                if computed is not None:
                    wp.launch(
                        scatter_controller_range_to_canonical,
                        dim=(
                            range_binding.canonical_computed_effort.shape[0],
                            compact_joint_ids.shape[0],
                        ),
                        inputs=[
                            computed,
                            canonical_slots,
                            range_binding.controller_local_slots,
                            range_binding.controller_stride,
                            range_binding.canonical_computed_effort,
                        ],
                        device=self._device,
                    )
                applied = staging.applied_effort
                source = applied if applied is not None else computed
                if source is not None:
                    wp.launch(
                        scatter_controller_range_to_canonical,
                        dim=(
                            range_binding.canonical_applied_effort.shape[0],
                            compact_joint_ids.shape[0],
                        ),
                        inputs=[
                            source,
                            canonical_slots,
                            range_binding.controller_local_slots,
                            range_binding.controller_stride,
                            range_binding.canonical_applied_effort,
                        ],
                        device=self._device,
                    )
            wp.launch(
                scatter_canonical_range_to_joint,
                dim=(range_binding.canonical_computed_effort.shape[0], compact_joint_ids.shape[0]),
                inputs=[
                    range_binding.canonical_computed_effort,
                    compact_joint_ids,
                    canonical_slots,
                    range_binding.computed_owner_slots,
                ],
                outputs=[joint_computed_effort],
                device=self._device,
            )
            wp.launch(
                scatter_canonical_range_to_joint,
                dim=(range_binding.canonical_applied_effort.shape[0], compact_joint_ids.shape[0]),
                inputs=[
                    range_binding.canonical_applied_effort,
                    compact_joint_ids,
                    canonical_slots,
                    range_binding.applied_owner_slots,
                ],
                outputs=[joint_applied_effort],
                device=self._device,
            )
            if joint_command_effort is not None:
                wp.launch(
                    scatter_canonical_range_to_joint,
                    dim=(range_binding.canonical_applied_effort.shape[0], compact_joint_ids.shape[0]),
                    inputs=[
                        range_binding.canonical_applied_effort,
                        compact_joint_ids,
                        canonical_slots,
                        range_binding.effort_owner_slots,
                    ],
                    outputs=[joint_command_effort],
                    device=self._device,
                )
            if backend_effort is not None and user_to_backend is not None:
                wp.launch(
                    scatter_canonical_range_to_backend,
                    dim=(range_binding.canonical_applied_effort.shape[0], compact_joint_ids.shape[0]),
                    inputs=[
                        range_binding.canonical_applied_effort,
                        compact_joint_ids,
                        canonical_slots,
                        range_binding.effort_owner_slots,
                        user_to_backend,
                        sim_buffers_are_backend_order,
                    ],
                    outputs=[backend_effort],
                    device=self._device,
                )
            if backend_computed_effort is not None and user_to_backend is not None:
                wp.launch(
                    scatter_canonical_range_to_backend,
                    dim=(range_binding.canonical_computed_effort.shape[0], compact_joint_ids.shape[0]),
                    inputs=[
                        range_binding.canonical_computed_effort,
                        compact_joint_ids,
                        canonical_slots,
                        range_binding.computed_owner_slots,
                        user_to_backend,
                        sim_buffers_are_backend_order,
                    ],
                    outputs=[backend_computed_effort],
                    device=self._device,
                )

    @staticmethod
    def _prepare_direct_pointer_mutations(
        actuator: Actuator,
        parameters: Mapping[str, wp.array],
        computed_effort: wp.array,
        applied_effort: wp.array,
    ) -> tuple[_PointerMutation, ...]:
        """Build and validate every reversible direct pointer replacement.

        The returned objects are the complete transaction descriptor: no
        ``reshape`` or controller lookup is deferred until after the first
        pointer has been installed.  This is important for native model
        adapters, where an error in a later range must leave no alias into
        candidate-owned canonical storage.
        """

        def _flatten(array: wp.array) -> wp.array:
            return array if array.ndim == 1 else array.reshape(-1)

        names = {
            "stiffness": ("kp",),
            "damping": ("kd",),
            "effort_limit": ("max_effort", "max_motor_effort"),
            "velocity_limit": ("velocity_limit",),
            "saturation_effort": ("saturation_effort",),
        }
        direct_arrays = [computed_effort, applied_effort]
        direct_arrays.extend(parameter for name, parameter in parameters.items() if name in names)
        if any(not isinstance(array, wp.array) for array in direct_arrays):
            raise TypeError("Direct Newton actuator aliases must be Warp arrays.")
        if any(array.dtype != wp.float32 for array in direct_arrays):
            raise TypeError("Direct Newton actuator aliases must use Warp float32 arrays.")
        if any(array.device != computed_effort.device for array in direct_arrays):
            raise ValueError("Direct Newton actuator aliases must share one device.")
        if any(not array.is_contiguous for array in direct_arrays):
            raise ValueError("Direct Newton actuator aliases must be contiguous.")
        if any(array.size != computed_effort.size for array in direct_arrays):
            raise ValueError("Direct Newton actuator aliases must have identical flat sizes.")
        components = (actuator.controller, *(actuator.clamping or ()))
        mutations: list[_PointerMutation] = []
        for canonical_name, component_names in names.items():
            parameter = parameters.get(canonical_name)
            if parameter is None:
                continue
            installed = _flatten(parameter)
            for component in components:
                for component_name in component_names:
                    if not hasattr(component, component_name):
                        continue
                    original = getattr(component, component_name)
                    if isinstance(original, wp.array):
                        mutations.append(_PointerMutation(component, component_name, original, installed, set()))

        computed = _flatten(computed_effort)
        mutations.append(
            _PointerMutation(actuator, "_computed_forces", getattr(actuator, "_computed_forces"), computed, set())
        )
        if getattr(actuator, "_applied_forces", None) is not None:
            mutations.append(
                _PointerMutation(
                    actuator,
                    "_applied_forces",
                    getattr(actuator, "_applied_forces"),
                    _flatten(applied_effort),
                    set(),
                )
            )
        for mutation in mutations:
            if not isinstance(mutation.original, wp.array):
                raise TypeError(f"Newton direct pointer {mutation.attr!r} must originally reference a Warp array.")
            if mutation.original.dtype != wp.float32 or mutation.installed.dtype != wp.float32:
                raise TypeError(f"Newton direct pointer {mutation.attr!r} must use Warp float32 arrays.")
            if mutation.original.device != mutation.installed.device:
                raise ValueError(f"Newton direct pointer {mutation.attr!r} cannot cross devices.")
            if not mutation.original.is_contiguous or not mutation.installed.is_contiguous:
                raise ValueError(f"Newton direct pointer {mutation.attr!r} must be contiguous.")
            if mutation.original.size != mutation.installed.size:
                raise ValueError(f"Newton direct pointer {mutation.attr!r} must preserve flat size.")
        return tuple(mutations)

    def _install_direct_pointer_binding(
        self,
        actuator: Actuator,
        handle: object,
        mutations: Sequence[_PointerMutation],
    ) -> None:
        """Install a prevalidated direct alias transaction for one actuator.

        A second registration may share an already installed transaction only
        when every exact target pointer matches.  A different canonical range
        is rejected rather than silently replacing a live user's arrays.
        """
        registry = getattr(self, "_direct_pointer_bindings", None)
        if registry is None:
            # Preserve compatibility with lightweight adapters created by
            # extension tests before this private transaction state existed.
            registry = {}
            self._direct_pointer_bindings = registry
        key = id(actuator)
        existing = registry.get(key)
        if existing is not None and not existing.registrations:
            # A previous rollback failed. Never attach a new owner to stale
            # pointers: first retry the exact inverse log or surface its
            # primary cleanup error to the caller.
            self._restore_direct_pointer_binding(key, existing)
            existing = registry.get(key)
        if existing is not None:
            same_targets = len(existing.mutations) == len(mutations) and all(
                previous.owner is current.owner
                and previous.attr == current.attr
                and self._same_warp_array(previous.installed, current.installed)
                for previous, current in zip(existing.mutations, mutations, strict=True)
            )
            if not same_targets:
                raise RuntimeError("Newton actuator is already directly bound to another canonical range.")
            existing.registrations.add(handle)
            for mutation in existing.mutations:
                mutation.registrations.add(handle)
            return

        binding = _DirectPointerBinding(actuator=actuator, mutations=[], registrations={handle})
        registry[key] = binding
        try:
            for mutation in mutations:
                # Record responsibility before the assignment.  A native
                # descriptor is allowed to raise after changing its pointer;
                # conditional restoration handles both outcomes safely.
                mutation.registrations.add(handle)
                binding.mutations.append(mutation)
                setattr(mutation.owner, mutation.attr, mutation.installed)
            self._refresh_dc_motor_corner_velocity(actuator)
        except Exception as error:
            binding.registrations.clear()
            try:
                self._restore_direct_pointer_binding(key, binding)
            except Exception as cleanup_error:
                error.add_note(f"Direct-pointer rollback also failed: {cleanup_error!r}")
            raise

    @staticmethod
    def _same_warp_array(first: object, second: object) -> bool:
        """Return whether two Warp views name the exact same contiguous storage."""
        return (
            isinstance(first, wp.array)
            and isinstance(second, wp.array)
            and first.ptr == second.ptr
            and first.dtype == second.dtype
            and first.device == second.device
            and first.shape == second.shape
            and first.strides == second.strides
        )

    def _release_direct_pointer_binding(
        self,
        actuator: Actuator,
        handle: object,
        *,
        force_restore: bool = False,
    ) -> None:
        """Release one direct registration and restore only after the final user.

        Failed inverse writes remain in the adapter-global log with no
        registrations.  Calling this method again retries them; an external
        rebind is never overwritten because restoration requires identity with
        the exact object installed by this adapter.
        """
        key = id(actuator)
        registry = getattr(self, "_direct_pointer_bindings", None)
        if registry is None:
            return
        binding = registry.get(key)
        if binding is None:
            return
        binding.registrations.discard(handle)
        for mutation in binding.mutations:
            mutation.registrations.discard(handle)
        if binding.registrations:
            return
        if getattr(self, "_owns_actuators", False) and not force_restore:
            del registry[key]
            return
        self._restore_direct_pointer_binding(key, binding)

    def _restore_direct_pointer_binding(self, key: int, binding: _DirectPointerBinding) -> None:
        """Best-effort reverse one pointer log, retaining only failed entries.

        An attribute changed externally after installation is left untouched;
        only the exact pointer installed by this adapter is restored.
        """
        failed: list[_PointerMutation] = []
        errors: list[Exception] = []
        for mutation in reversed(binding.mutations):
            if getattr(mutation.owner, mutation.attr) is not mutation.installed:
                continue
            try:
                setattr(mutation.owner, mutation.attr, mutation.original)
            except Exception as error:
                failed.append(mutation)
                errors.append(error)
        if failed:
            failed_ids = {id(mutation) for mutation in failed}
            binding.mutations[:] = [mutation for mutation in binding.mutations if id(mutation) in failed_ids]
        else:
            binding.mutations.clear()
        try:
            self._refresh_dc_motor_corner_velocity(binding.actuator)
        except Exception as error:
            errors.append(error)
        if not binding.mutations and not errors:
            self._direct_pointer_bindings.pop(key, None)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(f"Additional direct-pointer restore failure: {error!r}")
            raise primary

    def _refresh_dc_motor_corner_velocity(self, actuator: Actuator) -> None:
        """Refresh the derived corner velocity of every DC motor clamping component."""
        for component in actuator.clamping or ():
            if not all(
                isinstance(getattr(component, name, None), wp.array)
                for name in ("saturation_effort", "velocity_limit", "max_motor_effort", "corner_velocity")
            ):
                continue
            wp.launch(
                recompute_dc_motor_corner_velocity,
                dim=component.corner_velocity.shape[0],
                inputs=[component.saturation_effort, component.velocity_limit, component.max_motor_effort],
                outputs=[component.corner_velocity],
                device=self._device,
            )

    @property
    def is_all_graphable(self) -> bool:
        """``True`` when all actuators are CUDA-graph-safe."""
        return len(self.actuators) > 0 and all(a.is_graphable() for a in self.actuators)

    @property
    def is_stateful(self) -> bool:
        """``True`` when any actuator maintains delay or controller state."""
        return any(a.is_stateful() for a in self.actuators)

    @classmethod
    def from_usd(
        cls,
        stage: Any,
        joint_names: list[str],
        num_envs: int,
        num_joints: int,
        device: str,
        articulation_prim_path: str | None = None,
    ) -> NewtonActuatorAdapter:
        """Build an adapter from ``NewtonActuator`` prims authored on *stage*.

        This is the PhysX-side counterpart of Newton's
        ``ModelBuilder.add_usd``. It reads the same prims and constructs matching
        :class:`~newton.actuators.Actuator` objects. Joints with the same
        controller and clamping structure are merged into one actuator with
        combined indices; authored per-joint values are expanded into its
        model-global arrays. Newton backends use ``model.actuators`` instead.

        On PhysX, :paramref:`joint_names` is in this adapter's local public order
        and defines the local indices assigned to parsed actuator targets.

        Args:
            stage: USD stage containing ``NewtonActuator`` prims.
            joint_names: All articulation joint names in adapter-local public order.
            num_envs: Number of environments.
            num_joints: Number of joints per environment.
            device: Warp device string, for example ``"cuda:0"``.
            articulation_prim_path: Root prim path of environment zero's
                articulation. When set, only prims under this subtree are
                considered; otherwise the whole stage is scanned.

        Returns:
            Adapter whose actuator indices use :paramref:`joint_names` order.

        Raises:
            ValueError: If no authored actuator targets a name in
                :paramref:`joint_names`.
        """
        actuators, joint_signatures = _create_actuators_from_usd(
            stage,
            joint_names,
            num_envs,
            num_joints,
            device,
            articulation_prim_path=articulation_prim_path,
        )
        adapter = cls(actuators, num_envs, num_joints, dof_offset=0, device=device, owns_actuators=True)
        adapter._joint_signatures = joint_signatures
        return adapter

    @classmethod
    def _from_usd_binding(
        cls,
        binding: _ArticulationBinding,
        stage: Any,
        joint_names: list[str],
        num_envs: int,
        num_joints: int,
        device: str,
        articulation_prim_path: str | None = None,
    ) -> NewtonActuatorAdapter:
        """Build a hosted adapter with private canonical binding context.

        The binding-aware materialization plan is deliberately private: public
        ``from_usd`` remains an ABI-compatible standalone parser. Whole
        exact-type ranges can be injected as canonical parameter and output
        aliases during construction; otherwise the hosted adapter retains its
        model-global staging arrays. Legacy target attributes remain bound to
        the PhysX wrapper protocol.
        """
        recipes_per_joint = _parse_actuators_from_usd(stage, joint_names, articulation_prim_path)
        direct_bindings = _plan_hosted_direct_bindings(binding, joint_names, recipes_per_joint)
        actuators, joint_signatures = _create_actuators_from_usd(
            stage,
            joint_names,
            num_envs,
            num_joints,
            device,
            articulation_prim_path=articulation_prim_path,
            recipes_per_joint=recipes_per_joint,
            direct_bindings=direct_bindings,
        )
        adapter = cls(actuators, num_envs, num_joints, dof_offset=0, device=device, owns_actuators=True)
        adapter._joint_signatures = joint_signatures
        return adapter


# ---------------------------------------------------------------------------
# Deprecated per-articulation initial-gain snapshot compatibility helper.
# ---------------------------------------------------------------------------


def build_newton_actuator_defaults(
    actuators: list[Actuator],
    num_envs: int,
    num_joints: int,
    dof_offset: int,
    env_stride: int,
    device: str,
    joint_user_to_backend_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | slice]:
    """Deprecated compatibility helper that snapshots initial Newton controller gains.

    .. deprecated:: 2.4.1
        Actuator defaults are owned by
        :class:`~isaaclab.actuators.ActuatorCollection`. Use the scoped
        actuator facade exposed by :class:`~isaaclab.assets.Articulation`
        instead of creating a separate Newton gain snapshot.

    Actuators are filtered to those whose environment-zero DOF lies in
    ``[dof_offset, dof_offset + num_joints)``. Their gains are scattered in the
    actuator adapter's local joint order. Without :paramref:`joint_user_to_backend_indices`,
    the output preserves that local order. PhysX builds its per-articulation adapter from
    public joint names, so its adapter-local order is public order. Newton's global adapter
    uses backend-local order; the optional map converts its gains and managed indices to
    public order.

    Args:
        actuators: Newton actuators visible to this articulation.
        num_envs: Number of environments.
        num_joints: Articulation-local joint count.
        dof_offset: Offset of this articulation's DOFs in the env-major
            global index space (``0`` on PhysX, view-dependent on Newton).
        env_stride: Whole-model per-env DOF count — the stride used to build
            each actuator's env-major ``indices``. Equals ``num_joints`` on
            PhysX, but exceeds it by the free-root DOFs on a floating-base
            Newton articulation, so it must be passed explicitly rather than
            assumed equal to ``num_joints``. The owning adapter's
            :attr:`NewtonActuatorAdapter.num_joints` is exactly this value.
        device: Warp device string (e.g. ``"cuda:0"``).
        joint_user_to_backend_indices: Complete permutation from public joint
            indices to adapter-local joint indices. For Newton's global adapter,
            adapter-local order is backend order. ``None`` preserves adapter-local order.

    Returns:
        Tuple containing the following values:

        * ``stiffness``: Initial gains [N/m or N·m/rad, depending on joint
          type], shape ``(num_envs, num_joints)``, dtype ``torch.float32``, on
          :paramref:`device`.
        * ``damping``: Initial gains [N·s/m or N·m·s/rad, depending on joint
          type], shape ``(num_envs, num_joints)``, dtype ``torch.float32``, on
          :paramref:`device`.
        * ``joint_indices``: ``slice(None)`` when every joint is managed;
          otherwise, a ``torch.int32`` tensor on :paramref:`device` containing
          managed columns in the same adapter-local or public order as the gain tensors.

    Raises:
        ValueError: If :paramref:`joint_user_to_backend_indices` is not a
            complete permutation of all adapter-local joint indices.
    """
    global _DEFAULTS_DEPRECATION_EMITTED
    if not _DEFAULTS_DEPRECATION_EMITTED:
        warnings.warn(
            "build_newton_actuator_defaults() is deprecated; actuator defaults are now managed by "
            "ActuatorCollection canonical storage.",
            DeprecationWarning,
            stacklevel=2,
        )
        _DEFAULTS_DEPRECATION_EMITTED = True

    user_to_backend: tuple[int, ...] | None = None
    if joint_user_to_backend_indices is not None:
        user_to_backend = tuple(int(index) for index in joint_user_to_backend_indices)
        if sorted(user_to_backend) != list(range(num_joints)):
            raise ValueError(
                "joint_user_to_backend_indices must contain each backend joint index exactly once; "
                f"expected a permutation of 0..{num_joints - 1}, got {user_to_backend}."
            )

    arti_actuators = [act for act in actuators if dof_offset <= int(act.indices.numpy()[0]) < dof_offset + num_joints]

    managed_local: set[int] = set()
    for act in arti_actuators:
        per_act = act.indices.shape[0] // num_envs
        for global_dof in act.indices.numpy()[:per_act]:
            local = int(global_dof) - dof_offset
            if 0 <= local < num_joints:
                managed_local.add(local)
    joint_indices: torch.Tensor | slice
    if len(managed_local) == num_joints:
        joint_indices = slice(None)
    else:
        joint_indices = torch.tensor(sorted(managed_local), dtype=torch.int32, device=device)

    wp_device = wp.get_device(device)
    flat_stiffness = wp.zeros(num_envs * num_joints, dtype=wp.float32, device=wp_device)
    flat_damping = wp.zeros(num_envs * num_joints, dtype=wp.float32, device=wp_device)
    for act in arti_actuators:
        ctrl = act.controller
        if hasattr(ctrl, "kp"):
            wp.launch(
                scatter_gain_kernel,
                dim=act.indices.shape[0],
                inputs=[ctrl.kp, flat_stiffness, act.indices, dof_offset, num_joints, env_stride],
                device=wp_device,
            )
        if hasattr(ctrl, "kd"):
            wp.launch(
                scatter_gain_kernel,
                dim=act.indices.shape[0],
                inputs=[ctrl.kd, flat_damping, act.indices, dof_offset, num_joints, env_stride],
                device=wp_device,
            )
    stiffness = wp.to_torch(flat_stiffness.reshape((num_envs, num_joints)))
    damping = wp.to_torch(flat_damping.reshape((num_envs, num_joints)))
    if user_to_backend is not None:
        # ``index_select(1, backend_column_indices)`` gathers backend-order columns into user-order
        # positions: for each user position ``u`` it holds the backend column ``user_to_backend[u]``.
        backend_column_indices = torch.tensor(user_to_backend, dtype=torch.long, device=device)
        stiffness = stiffness.index_select(1, backend_column_indices)
        damping = damping.index_select(1, backend_column_indices)
        if not isinstance(joint_indices, slice):
            backend_to_user = [0] * num_joints
            for user_index, backend_index in enumerate(user_to_backend):
                backend_to_user[backend_index] = user_index
            joint_indices = torch.tensor(
                sorted(backend_to_user[index] for index in managed_local),
                dtype=torch.int32,
                device=device,
            )
    return stiffness, damping, joint_indices


# ---------------------------------------------------------------------------
# PhysX-only USD parsing
# ---------------------------------------------------------------------------


def _actuator_signature(
    parsed: Any,
    controller_arguments: Mapping[str, Any] | None = None,
    component_arguments: Sequence[tuple[type, Mapping[str, Any]]] | None = None,
) -> tuple:
    """Build Newton's structural grouping key for one parsed actuator spec.

    Numeric arguments are intentionally excluded: Newton stores them as
    per-DOF arrays inside one actuator. Only controller identity, delay
    presence, ordered clamping structure, and explicit shared arguments split
    groups, matching :class:`newton.ModelBuilder`'s finalization ABI.
    """

    def _hashable(value: Any) -> Any:
        if isinstance(value, list | tuple):
            return tuple(_hashable(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
        return value

    ctrl_resolved = (
        parsed.controller_class.resolve_arguments(dict(parsed.controller_kwargs))
        if controller_arguments is None
        else controller_arguments
    )
    shared_ctrl = getattr(parsed.controller_class, "SHARED_PARAMS", set())
    ctrl_shared_key = tuple(sorted((key, _hashable(ctrl_resolved[key])) for key in shared_ctrl if key in ctrl_resolved))

    comp_keys: list[tuple] = []
    has_delay = False
    resolved_components = (
        tuple((comp_cls, comp_cls.resolve_arguments(comp_kwargs)) for comp_cls, comp_kwargs in parsed.component_specs)
        if component_arguments is None
        else component_arguments
    )
    for comp_cls, resolved in resolved_components:
        if issubclass(comp_cls, Delay):
            has_delay = True
            continue
        shared = getattr(comp_cls, "SHARED_PARAMS", set())
        comp_keys.append(
            (comp_cls, tuple(sorted((key, _hashable(resolved[key])) for key in shared if key in resolved)))
        )

    # Keep the exact ``ModelBuilder.add_actuator`` key shape/order. The same
    # key is retained by NewtonManager for native model binding.
    return (parsed.controller_class, has_delay, tuple(comp_keys), ctrl_shared_key)


def _parse_actuators_from_usd(
    stage: Any,
    joint_names: Sequence[str],
    articulation_prim_path: str | None,
    *,
    allow_empty: bool = False,
) -> dict[int, tuple[_HostedActuatorRecipe, ...]]:
    """Parse actuator prims into public articulation-joint slots."""
    from newton.actuators import parse_actuator_prim  # noqa: PLC0415

    from pxr import Usd  # noqa: PLC0415

    joint_name_to_idx = {name: index for index, name in enumerate(joint_names)}
    root_prim = stage.GetPrimAtPath(articulation_prim_path) if articulation_prim_path else stage.GetPseudoRoot()
    recipes_per_joint: dict[int, list[_HostedActuatorRecipe]] = {}
    for prim in Usd.PrimRange(root_prim):
        parsed = parse_actuator_prim(prim)
        if parsed is None:
            continue
        target_name = parsed.target_path.rsplit("/", 1)[-1]
        if target_name in joint_name_to_idx:
            controller_arguments = parsed.controller_class.resolve_arguments(dict(parsed.controller_kwargs))
            component_arguments = tuple(
                (component_class, component_class.resolve_arguments(component_kwargs))
                for component_class, component_kwargs in parsed.component_specs
            )
            recipes_per_joint.setdefault(joint_name_to_idx[target_name], []).append(
                _HostedActuatorRecipe(
                    parsed=parsed,
                    signature=_actuator_signature(parsed, controller_arguments, component_arguments),
                    controller_arguments=controller_arguments,
                    component_arguments=component_arguments,
                )
            )
    if not recipes_per_joint and not allow_empty:
        raise ValueError(f"No NewtonActuator prims found targeting any of: {joint_names}")
    return {index: tuple(recipes) for index, recipes in recipes_per_joint.items()}


def _structural_occurrences_from_usd(
    stage: Any,
    joint_names: Sequence[str],
    articulation_prim_path: str,
) -> dict[int, tuple[tuple, ...]]:
    """Recover one articulation's authored actuator stream before builder grouping.

    The returned values are keyed by adapter-local joint index and preserve
    USD prim traversal order. Empty streams are retained for untargeted joints
    so native binding can reject a configuration that claims an absent
    controller before it changes any Newton pointers.
    """
    recipes_per_joint = _parse_actuators_from_usd(stage, joint_names, articulation_prim_path, allow_empty=True)
    return {
        joint_index: tuple(recipe.signature for recipe in recipes_per_joint.get(joint_index, ()))
        for joint_index in range(len(joint_names))
    }


def _hosted_recipe_occurrences(
    recipes_per_joint: Mapping[int, _HostedActuatorRecipe | Sequence[_HostedActuatorRecipe]],
) -> tuple[tuple[int, _HostedActuatorRecipe], ...]:
    """Flatten hosted recipes while retaining authored duplicate occurrences."""
    occurrences: list[tuple[int, _HostedActuatorRecipe]] = []
    for index, recipes in sorted(recipes_per_joint.items()):
        if isinstance(recipes, _HostedActuatorRecipe):
            occurrences.append((index, recipes))
        else:
            occurrences.extend((index, recipe) for recipe in recipes)
    return tuple(occurrences)


def _plan_hosted_direct_bindings(
    binding: _ArticulationBinding,
    joint_names: Sequence[str],
    recipes_per_joint: Mapping[int, _HostedActuatorRecipe | Sequence[_HostedActuatorRecipe]],
) -> dict[tuple, _HostedDirectBinding]:
    """Select whole exact-type Newton signatures that can alias canonical storage."""
    if binding.groups is None:
        return {}

    occurrences = _hosted_recipe_occurrences(recipes_per_joint)
    signatures_by_joint_name = {
        joint_names[index]: tuple(recipe.signature for recipe_index, recipe in occurrences if recipe_index == index)
        for index in range(len(joint_names))
    }
    indices_by_signature: dict[tuple, list[int]] = {}
    types_by_signature: dict[tuple, set[type]] = {}
    for index, recipe in occurrences:
        indices_by_signature.setdefault(recipe.signature, []).append(index)
    for group in binding.layout.group_layouts:
        for joint_name in group.joint_names:
            for signature in signatures_by_joint_name.get(joint_name, ()):
                types_by_signature.setdefault(signature, set()).add(group.actuator_type)

    plans: dict[tuple, _HostedDirectBinding] = {}
    for signature, local_indices in indices_by_signature.items():
        actuator_types = types_by_signature.get(signature, set())
        if len(actuator_types) != 1:
            continue
        actuator_type = next(iter(actuator_types))
        type_layout = binding.layout.type_layouts.get(actuator_type)
        if type_layout is None:
            continue
        type_groups = tuple(group for group in binding.layout.group_layouts if group.actuator_type is actuator_type)
        if not type_groups or any(group.name not in binding.native_group_names for group in type_groups):
            continue
        canonical_joint_names = tuple(joint_name for group in type_groups for joint_name in group.joint_names)
        signature_joint_names = tuple(joint_names[index] for index in local_indices)
        if (
            tuple(local_indices) != tuple(type_layout.compact_joint_indices)
            or len(local_indices) != type_layout.num_dofs
            or len(set(local_indices)) != len(local_indices)
            or signature_joint_names != canonical_joint_names
        ):
            continue

        group_binding = binding.groups[type_groups[0].name].__dict__.get("_parameter_binding")
        arrays = getattr(group_binding, "arrays", None)
        if arrays is None or "computed_effort" not in arrays or "applied_effort" not in arrays:
            continue
        plans[signature] = _HostedDirectBinding(
            parameters={name: value.warp for name, value in arrays.items()},
            computed_effort=arrays["computed_effort"].warp,
            applied_effort=arrays["applied_effort"].warp,
        )
    return plans


def _hosted_flat_array(array: wp.array, expected_size: int, device: str, name: str) -> wp.array:
    """Validate and flatten one canonical hosted array without changing its pointer."""
    if array.dtype != wp.float32:
        raise TypeError(f"Hosted canonical {name} must have Warp float32 dtype.")
    if array.device != wp.get_device(device):
        raise ValueError(f"Hosted canonical {name} must be on device {device!r}.")
    if array.size != expected_size:
        raise ValueError(f"Hosted canonical {name} has {array.size} values; expected {expected_size}.")
    return array.reshape(-1)


def _create_actuators_from_usd(
    stage: Any,
    joint_names: list[str],
    num_envs: int,
    num_total_joints: int,
    device: str,
    articulation_prim_path: str | None = None,
    *,
    recipes_per_joint: Mapping[int, _HostedActuatorRecipe | Sequence[_HostedActuatorRecipe]] | None = None,
    direct_bindings: Mapping[tuple, _HostedDirectBinding] | None = None,
) -> tuple[list[Actuator], dict[str, tuple]]:
    """Parse ``NewtonActuator`` prims and instantiate standalone actuators.

    This mirrors the actuator construction that Newton's
    ``ModelBuilder.add_usd`` performs, but operates independently of a
    Newton ``Model``.  It is used on the PhysX backend where there is no
    Newton simulation — actuators are stepped manually via the adapter.

    Because PhysX articulations have no free or ball joints, every
    joint's coordinate count equals its DOF count.  A single
    ``indices`` array is therefore sufficient for all index roles
    (``indices``, ``pos_indices``, ``target_pos_indices``).

    Joints with compatible controller structure are merged into one
    :class:`Actuator` with combined indices. Numeric parameters remain
    per-DOF arrays, exactly as in Newton's model builder.

    Each per-DOF scalar parameter (``kp``, ``kd``, ``saturation_effort``,
    etc.) is constructed as a device tensor over the merged group.
    Parameters marked as ``SHARED_PARAMS`` on the controller or clamping
    class (e.g. ``model_path``, ``lookup_positions``) are passed through
    directly without broadcast.
    """
    from collections import defaultdict  # noqa: PLC0415

    if recipes_per_joint is None:
        recipes_per_joint = _parse_actuators_from_usd(stage, joint_names, articulation_prim_path)
    direct_bindings = direct_bindings or {}

    groups: dict[tuple, list[int]] = defaultdict(list)
    sig_to_parsed: dict[tuple, Any] = {}
    recipe_occurrences = _hosted_recipe_occurrences(recipes_per_joint)
    for local_idx, recipe in recipe_occurrences:
        sig = recipe.signature
        groups[sig].append(local_idx)
        if sig not in sig_to_parsed:
            sig_to_parsed[sig] = recipe

    actuators = []
    for sig, local_indices in groups.items():
        parsed = sig_to_parsed[sig]
        direct_binding = direct_bindings.get(sig)
        expected_size = num_envs * len(local_indices)

        torch_owners: list[torch.Tensor] = []
        local_indices_t = torch.tensor(local_indices, dtype=torch.int32, device=device)
        torch_owners.append(local_indices_t)
        local_indices_wp = wp.from_torch(local_indices_t, dtype=wp.int32)
        indices = wp.empty(num_envs * len(local_indices), dtype=wp.uint32, device=device)
        wp.launch(
            _expand_env_major_indices,
            dim=indices.shape[0],
            inputs=[local_indices_wp, num_total_joints],
            outputs=[indices],
            device=device,
        )
        group_parsed = tuple(
            recipe for index, recipe in recipe_occurrences if index in local_indices and recipe.signature == sig
        )

        def _direct_parameter(name: str) -> wp.array | None:
            if direct_binding is None or name not in direct_binding.parameters:
                return None
            return _hosted_flat_array(direct_binding.parameters[name], expected_size, device, name)

        def _expanded_values(values: list[float], dtype: torch.dtype = torch.float32) -> wp.array:
            local_values = torch.tensor(values, dtype=dtype, device=device)
            torch_owners.append(local_values)
            if dtype is torch.int32:
                local_values_wp = wp.from_torch(local_values, dtype=wp.int32)
                expanded = wp.empty(num_envs * len(values), dtype=wp.int32, device=device)
                wp.launch(
                    _expand_env_major_int_values,
                    dim=expanded.shape[0],
                    inputs=[local_values_wp],
                    outputs=[expanded],
                    device=device,
                )
            else:
                local_values_wp = wp.from_torch(local_values, dtype=wp.float32)
                expanded = wp.empty(num_envs * len(values), dtype=wp.float32, device=device)
                wp.launch(
                    _expand_env_major_values,
                    dim=expanded.shape[0],
                    inputs=[local_values_wp],
                    outputs=[expanded],
                    device=device,
                )
            return expanded

        # Controller
        resolved = parsed.controller_arguments
        controller_class = parsed.parsed.controller_class
        shared_ctrl = getattr(controller_class, "SHARED_PARAMS", set())
        ctrl_arrays = {}
        controller_parameter_names = {"kp": "stiffness", "kd": "damping"}
        for key, val in resolved.items():
            if key in shared_ctrl:
                ctrl_arrays[key] = val
            else:
                direct_value = _direct_parameter(controller_parameter_names.get(key, ""))
                if direct_value is not None:
                    ctrl_arrays[key] = direct_value
                else:
                    values = [group.controller_arguments[key] for group in group_parsed]
                    ctrl_arrays[key] = _expanded_values(values)
        controller = controller_class(**ctrl_arrays)

        # Components (delay + clampings)
        clampings = []
        delay = None
        clamping_specs = [
            (component_class, arguments)
            for component_class, arguments in parsed.component_arguments
            if issubclass(component_class, Clamping)
        ]
        delay_values = []
        has_delay = False
        for group in group_parsed:
            delay_spec = next(
                (
                    arguments
                    for component_class, arguments in group.component_arguments
                    if issubclass(component_class, Delay)
                ),
                None,
            )
            has_delay = has_delay or delay_spec is not None
            delay_values.append(int(delay_spec.get("delay_steps", 0)) if delay_spec is not None else 0)
        if has_delay:
            delay = Delay(delay_steps=_expanded_values(delay_values, torch.int32), max_delay=max(delay_values))

        for clamping_index, (comp_cls, comp_kwargs) in enumerate(clamping_specs):
            resolved_kw = comp_kwargs
            shared_clamp = getattr(comp_cls, "SHARED_PARAMS", set())
            clamp_arrays = {}
            clamping_parameter_names = {
                "max_effort": "effort_limit",
                "max_motor_effort": "effort_limit",
                "velocity_limit": "velocity_limit",
                "saturation_effort": "saturation_effort",
            }
            for key, value in resolved_kw.items():
                if key in shared_clamp:
                    clamp_arrays[key] = value
                else:
                    direct_value = _direct_parameter(clamping_parameter_names.get(key, ""))
                    if direct_value is not None:
                        clamp_arrays[key] = direct_value
                    else:
                        values = []
                        for group in group_parsed:
                            group_clamps = [
                                (component_class, arguments)
                                for component_class, arguments in group.component_arguments
                                if issubclass(component_class, Clamping)
                            ]
                            group_cls, group_kwargs = group_clamps[clamping_index]
                            if group_cls is not comp_cls:
                                raise RuntimeError("Newton actuator grouping produced incompatible clamping chains.")
                            values.append(group_kwargs[key])
                        clamp_arrays[key] = _expanded_values(values)
            clampings.append(comp_cls(**clamp_arrays))

        actuator = Actuator(
            indices=indices,
            controller=controller,
            delay=delay,
            clamping=clampings if clampings else None,
            control_target_pos_attr="joint_target_pos",
            control_target_vel_attr="joint_target_vel",
        )
        if direct_binding is not None:
            actuator._computed_forces = _hosted_flat_array(
                direct_binding.computed_effort, expected_size, device, "computed_effort"
            )
            if actuator._applied_forces is not None:
                actuator._applied_forces = _hosted_flat_array(
                    direct_binding.applied_effort, expected_size, device, "applied_effort"
                )
        actuator._isaaclab_structural_key = sig
        # Retain only the authored env-0 local order.  It proves direct alias
        # eligibility without reading the world-sized device ``indices``
        # array back to the host during binding.
        actuator._isaaclab_env_zero_dof_indices = tuple(local_indices)
        actuator._isaaclab_adapter_torch_owners = torch_owners
        actuators.append(actuator)

    joint_signatures: dict[str, list[tuple]] = {}
    for index, recipe in recipe_occurrences:
        joint_signatures.setdefault(joint_names[index], []).append(recipe.signature)
    return actuators, {name: tuple(signatures) for name, signatures in joint_signatures.items()}
