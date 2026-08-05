# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-actuator adapter shared by the Newton and PhysX backends.

Owns the actuator-state lifecycle, the pre-clamp computed-effort buffer,
and the per-step ``step`` / ``reset`` / ``finalize`` calls. The
:meth:`~NewtonActuatorAdapter.from_usd` classmethod parses
``NewtonActuator`` USD prims on the PhysX backend (Newton populates
``model.actuators`` itself).

DR gain updates bypass the adapter — the articulation writes straight
to controller arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from newton.actuators import Actuator, Clamping, Delay

from .kernels import (
    build_controller_slot_map,
    build_implicit_dof_mask,
    build_per_dof_env_mask_kernel,
    gather_canonical_range_to_controller,
    scatter_gain_kernel,
    set_mask_kernel,
    validate_direct_range_order,
    zero_at_indices_kernel,
)

if TYPE_CHECKING:
    from isaaclab.actuators.actuator_collection import _ArticulationBinding


@dataclass(frozen=True)
class _NativeRangeBinding:
    """One native controller range bound to canonical actuator storage."""

    group_names: tuple[str, ...]
    actuator: Actuator
    direct: bool
    canonical_parameters: Mapping[str, wp.array]
    canonical_computed_effort: wp.array
    canonical_applied_effort: wp.array
    staging: object | None = None
    validation_error_flag: wp.array | None = None
    handle: object | None = None
    compact_joint_ids: wp.array | None = None
    controller_slots: wp.array | None = None
    dof_offset: int = 0
    has_joint_ordering: bool = False
    user_to_backend: wp.array | None = None


@dataclass
class _GlobalNativeActuatorBinding:
    """Adapter-owned staged binding for one globally shared Newton actuator."""

    actuator: Actuator
    original_parameters: list[tuple[object, str, wp.array]]
    staged_parameters: list[tuple[object, str, wp.array]]
    original_computed_effort: wp.array | None
    original_applied_effort: wp.array | None
    staged_computed_effort: wp.array | None
    staged_applied_effort: wp.array | None
    controller_slots: wp.array
    registrations: set[object]


# ---------------------------------------------------------------------------
# Abstract base — backend-independent logic
# ---------------------------------------------------------------------------


class NewtonActuatorAdapter:
    """Adapter that wraps a list of :class:`newton.actuators.Actuator`.

    Owns the actuator-state lifecycle, DOF-to-actuator bookkeeping,
    stepping, reset, and the pre-clamp computed-effort buffer the
    in-graph telemetry kernel reads on the post-actuator hook.
    """

    @dataclass(frozen=True)
    class ArticulationBinding:
        """Newton fast-path init state for one articulation.

        Returned by :meth:`bind_articulation`. Bundles the pieces the
        articulation formerly assembled from separate free-function calls:
        the initial gain snapshot, the implicit-DOF mask, and the
        per-articulation view of the adapter's computed-effort buffer.
        """

        implicit_dof_mask: wp.array
        """Per-DOF mask consumed by ``sync_torque_telemetry``; ``1`` on implicit-actuator DOFs, ``0`` otherwise."""

        implicit_dof_mask_owner: torch.Tensor
        """Torch tensor owning the memory :attr:`implicit_dof_mask` aliases; keep referenced for the mask's lifetime."""

        computed_effort_view: wp.array
        """Private articulation telemetry destination in public joint order."""

        ranges: tuple[_NativeRangeBinding, ...]
        """Immutable direct or staged native ranges retained through binding lifetime."""

    def __init__(
        self,
        actuators: list[Actuator],
        num_envs: int,
        num_joints: int,
        dof_offset: int,
        device: str,
    ):
        self.actuators = actuators
        self.num_joints = num_joints

        self._num_envs = num_envs
        self._dof_offset = dof_offset
        self._device = device
        self._global_native_bindings: dict[int, _GlobalNativeActuatorBinding] = {}

        # Collect the set of local DOFs covered by some actuator. Only the
        # env-0 slice of each actuator's flat ``indices`` array is needed —
        # later envs are repeats with a constant ``num_joints`` stride.
        managed: set[int] = set()
        for act in actuators:
            all_indices = act.indices.numpy()
            num_per_act = len(all_indices) // num_envs
            for global_dof in all_indices[:num_per_act]:
                local_dof = global_dof - dof_offset
                if 0 <= local_dof < num_joints:
                    managed.add(local_dof)

        if len(managed) == num_joints:
            self.joint_indices: torch.Tensor | slice = slice(None)
        else:
            self.joint_indices = torch.tensor(sorted(managed), dtype=torch.int32, device=device)

        self._states_a = [act.state() for act in actuators]
        self._states_b = [act.state() for act in actuators]

        for act in actuators:
            act.control_computed_output_attr = "joint_computed_f"

    def finalize(self, sim_control: Any) -> None:
        """Bind the pre-clamp computed-effort buffer onto ``sim_control``.

        Args:
            sim_control: The ``sim_control`` object that will be passed
                to :meth:`step` for this adapter's lifetime. Newton's
                ``Control`` on the Newton backend, an
                :class:`~isaaclab_newton.actuators.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
        """
        del sim_control

    def step(self, sim_state: Any, sim_control: Any, dt: float) -> None:
        """Zero actuated DOFs, step all actuators, and swap state buffers.

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
                dim=act.indices.shape[0],
                inputs=[sim_control.joint_f, act.indices],
            )
        for act, sa, sb in zip(self.actuators, self._states_a, self._states_b):
            act.step(sim_state, sim_control, sa, sb, dt=dt)
        self._swap_state_buffers()

    def _swap_state_buffers(self) -> None:
        """Advance the actuator state ping-pong after an eager step or graph replay."""
        self._states_a, self._states_b = self._states_b, self._states_a

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset actuator states for the given environments.

        Args:
            env_ids: Environment indices to reset. ``None`` (or
                ``slice(None)``, which IsaacLab callers sometimes pass)
                resets all environments. Otherwise expects a torch tensor
                or sequence of int indices.

        Newton's :meth:`Actuator.State.reset` expects a per-DOF boolean
        mask of length ``num_actuators`` (= ``num_envs * dofs_per_actuator``),
        not a per-env mask — each entry gates the corresponding column of
        the actuator's state buffers (delay queue, controller integral,
        etc.). We therefore build a per-actuator per-DOF mask from the
        env mask before delegating to each state.
        """
        if env_ids is None or env_ids == slice(None):
            for sa, sb in zip(self._states_a, self._states_b):
                if sa is not None:
                    sa.reset(None)
                if sb is not None:
                    sb.reset(None)
            return

        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            idx = wp.from_torch(env_ids.to(device=self._device).contiguous().to(torch.int32), dtype=wp.int32)
        else:
            if len(env_ids) == 0:
                return
            idx = wp.array(list(env_ids), dtype=wp.int32, device=self._device)
        env_mask = wp.zeros(self._num_envs, dtype=wp.bool, device=self._device)
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
    ) -> ArticulationBinding:
        """Bind private canonical ranges for one articulation.

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

        Returns:
            The bundled :class:`ArticulationBinding` for this articulation.
        """
        if binding.groups is None:
            raise RuntimeError("Newton canonical binding requires private actuator groups.")
        ranges = self._bind_canonical_ranges(
            binding,
            dof_offset=dof_offset,
            joint_user_to_backend_indices=joint_user_to_backend_indices,
        )
        implicit_dof_mask, implicit_dof_mask_owner = build_implicit_dof_mask(
            dict(binding.groups), binding.layout.num_joints, self._device
        )
        return self.ArticulationBinding(
            implicit_dof_mask=implicit_dof_mask,
            implicit_dof_mask_owner=implicit_dof_mask_owner,
            computed_effort_view=binding.computed_effort.warp,
            ranges=ranges,
        )

    def _bind_canonical_ranges(
        self,
        binding: _ArticulationBinding,
        *,
        dof_offset: int,
        joint_user_to_backend_indices: Sequence[int] | None,
    ) -> tuple[_NativeRangeBinding, ...]:
        """Alias exact native type blocks when their 1-D controller ABI permits it."""
        assert binding.groups is not None
        ranges: list[_NativeRangeBinding] = []
        for actuator_type, type_layout in binding.layout.type_layouts.items():
            group_names = tuple(
                group.name
                for group in binding.layout.group_layouts
                if group.actuator_type is actuator_type and group.name in binding.native_group_names
            )
            if not group_names:
                continue
            group = binding.groups[group_names[0]]
            parameter_binding = group.__dict__.get("_parameter_binding")
            if parameter_binding is None:
                raise RuntimeError(f"Native actuator type {actuator_type.__name__} has no canonical parameter storage.")
            arrays = parameter_binding.arrays
            expected_size = type_layout.num_worlds * type_layout.num_dofs
            compatible = tuple(
                act
                for act in self.actuators
                if act.indices.shape[0] == expected_size
                and self._has_direct_range_order(
                    act,
                    type_layout.compact_joint_indices,
                    dof_offset=dof_offset,
                    joint_user_to_backend_indices=joint_user_to_backend_indices,
                )
            )
            direct = len(compatible) == 1
            actuator = compatible[0] if direct else self._select_staged_actuator()
            canonical_parameters = {name: value.warp for name, value in arrays.items()}
            computed = arrays["computed_effort"].warp
            applied = arrays["applied_effort"].warp
            handle = object()
            staging = None
            if direct:
                self._bind_direct_parameters(actuator, canonical_parameters)
                actuator._computed_forces = computed.reshape(-1)
                if getattr(actuator, "_applied_forces", None) is not None:
                    actuator._applied_forces = applied.reshape(-1)
            else:
                staging = self._register_staged_actuator(actuator, handle)
            ranges.append(
                _NativeRangeBinding(
                    group_names=group_names,
                    actuator=actuator,
                    direct=direct,
                    canonical_parameters=canonical_parameters,
                    canonical_computed_effort=computed,
                    canonical_applied_effort=applied,
                    staging=staging,
                    handle=handle,
                    compact_joint_ids=wp.array(type_layout.compact_joint_indices, dtype=wp.int32, device=self._device),
                    controller_slots=None if direct else staging.controller_slots,
                    dof_offset=dof_offset,
                    has_joint_ordering=joint_user_to_backend_indices is not None,
                    user_to_backend=(
                        wp.array(joint_user_to_backend_indices, dtype=wp.int32, device=self._device)
                        if joint_user_to_backend_indices is not None
                        else wp.array(list(range(binding.layout.num_joints)), dtype=wp.int32, device=self._device)
                    ),
                )
            )
        return tuple(ranges)

    def _select_staged_actuator(self) -> Actuator:
        """Select the global controller whose immutable slot map will be staged."""
        if len(self.actuators) != 1:
            raise RuntimeError("Cannot infer a staged Newton controller range from multiple global actuators.")
        return self.actuators[0]

    def _register_staged_actuator(self, actuator: Actuator, handle: object) -> _GlobalNativeActuatorBinding:
        """Install one persistent controller-sized staging set and retain an exact handle."""
        registry = self._global_native_bindings
        key = id(actuator)
        global_binding = registry.get(key)
        if global_binding is None:
            original_parameters: list[tuple[object, str, wp.array]] = []
            staged_parameters: list[tuple[object, str, wp.array]] = []
            for component in (actuator.controller, *(actuator.clamping or ())):
                for name in (
                    "kp",
                    "kd",
                    "max_effort",
                    "max_motor_effort",
                    "velocity_limit",
                    "saturation_effort",
                ):
                    value = getattr(component, name, None)
                    if isinstance(value, wp.array):
                        original_parameters.append((component, name, value))
                        staged = wp.clone(value)
                        staged_parameters.append((component, name, staged))
                        setattr(component, name, staged)
            original_computed = getattr(actuator, "_computed_forces", None)
            original_applied = getattr(actuator, "_applied_forces", None)
            staged_computed = wp.clone(original_computed) if isinstance(original_computed, wp.array) else None
            staged_applied = wp.clone(original_applied) if isinstance(original_applied, wp.array) else None
            if staged_computed is not None:
                actuator._computed_forces = staged_computed
            if staged_applied is not None:
                actuator._applied_forces = staged_applied
            global_binding = _GlobalNativeActuatorBinding(
                actuator=actuator,
                original_parameters=original_parameters,
                staged_parameters=staged_parameters,
                original_computed_effort=original_computed,
                original_applied_effort=original_applied,
                staged_computed_effort=staged_computed,
                staged_applied_effort=staged_applied,
                controller_slots=wp.full(self._num_envs * self.num_joints, -1, dtype=wp.int32, device=self._device),
                registrations=set(),
            )
            wp.launch(
                build_controller_slot_map,
                dim=actuator.indices.shape[0],
                inputs=[actuator.indices, global_binding.controller_slots],
                device=self._device,
            )
            registry[key] = global_binding
        global_binding.registrations.add(handle)
        return global_binding

    def unregister_articulation_ranges(self, ranges: Sequence[_NativeRangeBinding]) -> None:
        """Release exact staged registrations and restore pointers after the last user leaves."""
        for range_binding in ranges:
            handle = range_binding.handle
            if range_binding.direct or handle is None:
                continue
            key = id(range_binding.actuator)
            global_binding = self._global_native_bindings.get(key)
            if global_binding is None:
                continue
            global_binding.registrations.discard(handle)
            if global_binding.registrations:
                continue
            for component, name, value in global_binding.original_parameters:
                setattr(component, name, value)
            if global_binding.original_computed_effort is not None:
                global_binding.actuator._computed_forces = global_binding.original_computed_effort
            if global_binding.original_applied_effort is not None:
                global_binding.actuator._applied_forces = global_binding.original_applied_effort
            del self._global_native_bindings[key]

    def gather_staged_ranges(self, ranges: Sequence[_NativeRangeBinding]) -> None:
        """Refresh persistent staged controller parameters from canonical storage."""
        for range_binding in ranges:
            if range_binding.direct or range_binding.staging is None:
                continue
            assert range_binding.compact_joint_ids is not None
            assert range_binding.controller_slots is not None
            assert range_binding.user_to_backend is not None
            for component, name, staged in range_binding.staging.staged_parameters:
                canonical_name = {
                    "kp": "stiffness",
                    "kd": "damping",
                    "max_effort": "max_effort",
                    "max_motor_effort": "max_motor_effort",
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
                        range_binding.compact_joint_ids,
                        range_binding.user_to_backend,
                        range_binding.controller_slots,
                        range_binding.dof_offset,
                        self.num_joints,
                        range_binding.has_joint_ordering,
                        staged,
                    ],
                    device=self._device,
                )

    def _has_direct_range_order(
        self,
        actuator: Actuator,
        compact_joint_indices: Sequence[int],
        *,
        dof_offset: int,
        joint_user_to_backend_indices: Sequence[int] | None,
    ) -> bool:
        """Validate one candidate direct controller alias entirely on device."""
        compact_joint_ids = wp.array(compact_joint_indices, dtype=wp.int32, device=self._device)
        if joint_user_to_backend_indices is None:
            user_to_backend = wp.array(list(range(len(compact_joint_indices))), dtype=wp.int32, device=self._device)
            has_joint_ordering = False
        else:
            user_to_backend = wp.array(joint_user_to_backend_indices, dtype=wp.int32, device=self._device)
            has_joint_ordering = True
        error_flag = wp.zeros(1, dtype=wp.int32, device=self._device)
        wp.launch(
            validate_direct_range_order,
            dim=actuator.indices.shape[0],
            inputs=[
                actuator.indices,
                compact_joint_ids,
                user_to_backend,
                dof_offset,
                self.num_joints,
                has_joint_ordering,
                error_flag,
            ],
            device=self._device,
        )
        return int(wp.to_torch(error_flag)[0]) == 0

    @staticmethod
    def _bind_direct_parameters(actuator: Actuator, parameters: Mapping[str, wp.array]) -> None:
        """Rebind recognised Newton component arrays to canonical flat aliases."""
        names = {
            "stiffness": "kp",
            "damping": "kd",
            "effort_limit": "max_effort",
            "max_effort": "max_effort",
            "max_motor_effort": "max_motor_effort",
            "velocity_limit": "velocity_limit",
            "saturation_effort": "saturation_effort",
        }
        components = (actuator.controller, *(actuator.clamping or ()))
        for canonical_name, component_name in names.items():
            parameter = parameters.get(canonical_name)
            if parameter is None:
                continue
            for component in components:
                if hasattr(component, component_name):
                    setattr(component, component_name, parameter.reshape(-1))

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
        controller, gains, clamping, and delay are merged into one actuator with
        combined indices. Newton backends use ``model.actuators`` instead.

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
        actuators = _create_actuators_from_usd(
            stage,
            joint_names,
            num_envs,
            num_joints,
            device,
            articulation_prim_path=articulation_prim_path,
        )
        return cls(actuators, num_envs, num_joints, dof_offset=0, device=device)


# ---------------------------------------------------------------------------
# Per-articulation initial-gain snapshot — consumed by
# ``randomize_actuator_gains`` to seed ``default_joint_*`` baselines.
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
    """Snapshot initial Newton actuator gains for one articulation.

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


def _actuator_signature(parsed: Any) -> tuple:
    """Build a hashable key from a parsed actuator spec for grouping.

    Joints whose prims resolve to the same signature share identical
    controller type, gains, clamping chain, and delay configuration and
    can therefore be merged into a single :class:`~newton.actuators.Actuator`
    with combined index arrays.
    """
    ctrl_resolved = parsed.controller_class.resolve_arguments(
        dict(parsed.controller_kwargs),
    )
    ctrl_key = (parsed.controller_class, tuple(sorted(ctrl_resolved.items())))

    comp_keys: list[tuple] = []
    for comp_cls, comp_kwargs in parsed.component_specs:
        resolved = comp_cls.resolve_arguments(comp_kwargs)
        comp_keys.append((comp_cls, tuple(sorted(resolved.items()))))
    comp_keys.sort(key=lambda t: t[0].__name__)

    return (ctrl_key, tuple(comp_keys))


def _create_actuators_from_usd(
    stage: Any,
    joint_names: list[str],
    num_envs: int,
    num_total_joints: int,
    device: str,
    articulation_prim_path: str | None = None,
) -> list[Actuator]:
    """Parse ``NewtonActuator`` prims and instantiate standalone actuators.

    This mirrors the actuator construction that Newton's
    ``ModelBuilder.add_usd`` performs, but operates independently of a
    Newton ``Model``.  It is used on the PhysX backend where there is no
    Newton simulation — actuators are stepped manually via the adapter.

    Because PhysX articulations have no free or ball joints, every
    joint's coordinate count equals its DOF count.  A single
    ``indices`` array is therefore sufficient for all index roles
    (``indices``, ``pos_indices``, ``target_pos_indices``).

    Joints with identical controller type, gains, clamping chain, and
    delay are merged into one :class:`Actuator` with combined indices.

    Each per-DOF scalar parameter (``kp``, ``kd``, ``saturation_effort``,
    etc.) is broadcast via :func:`wp.full` to match the group size.
    Parameters marked as ``SHARED_PARAMS`` on the controller or clamping
    class (e.g. ``model_path``, ``lookup_positions``) are passed through
    directly without broadcast.
    """
    from collections import defaultdict  # noqa: PLC0415

    from newton.actuators import parse_actuator_prim  # noqa: PLC0415

    from pxr import Usd  # noqa: PLC0415

    wp_device = wp.get_device(device)

    joint_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(joint_names)}

    if articulation_prim_path is not None:
        root_prim = stage.GetPrimAtPath(articulation_prim_path)
    else:
        root_prim = stage.GetPseudoRoot()

    parsed_per_joint: dict[int, Any] = {}
    for prim in Usd.PrimRange(root_prim):
        parsed = parse_actuator_prim(prim)
        if parsed is None:
            continue
        target_name = parsed.target_path.rsplit("/", 1)[-1]
        if target_name in joint_name_to_idx:
            parsed_per_joint[joint_name_to_idx[target_name]] = parsed

    if not parsed_per_joint:
        raise ValueError(f"No NewtonActuator prims found targeting any of: {joint_names}")

    groups: dict[tuple, list[int]] = defaultdict(list)
    sig_to_parsed: dict[tuple, Any] = {}
    for local_idx, parsed in sorted(parsed_per_joint.items()):
        sig = _actuator_signature(parsed)
        groups[sig].append(local_idx)
        if sig not in sig_to_parsed:
            sig_to_parsed[sig] = parsed

    actuators = []
    for sig, local_indices in groups.items():
        parsed = sig_to_parsed[sig]

        flat_indices = np.array(
            [idx + e * num_total_joints for e in range(num_envs) for idx in local_indices],
            dtype=np.uint32,
        )
        indices = wp.array(flat_indices, device=wp_device)
        num_dofs_in_group = len(local_indices) * num_envs

        # Controller
        ctrl_kwargs = dict(parsed.controller_kwargs)
        resolved = parsed.controller_class.resolve_arguments(ctrl_kwargs)
        shared_ctrl = getattr(parsed.controller_class, "SHARED_PARAMS", set())
        ctrl_arrays = {}
        for key, val in resolved.items():
            if key in shared_ctrl:
                ctrl_arrays[key] = val
            else:
                ctrl_arrays[key] = wp.full(num_dofs_in_group, float(val), dtype=wp.float32, device=wp_device)
        controller = parsed.controller_class(**ctrl_arrays)

        # Components (delay + clampings)
        clampings = []
        delay = None
        for comp_cls, comp_kwargs in parsed.component_specs:
            if issubclass(comp_cls, Delay):
                resolved_kw = Delay.resolve_arguments(comp_kwargs)
                delay_steps = int(resolved_kw.get("delay_steps", 0))
                if delay_steps > 0:
                    delay_arr = wp.full(num_dofs_in_group, delay_steps, dtype=wp.int32, device=wp_device)
                    delay = Delay(delay_steps=delay_arr, max_delay=delay_steps)
            elif issubclass(comp_cls, Clamping):
                resolved_kw = comp_cls.resolve_arguments(comp_kwargs)
                shared_clamp = getattr(comp_cls, "SHARED_PARAMS", set())
                clamp_arrays = {}
                for k, v in resolved_kw.items():
                    if k in shared_clamp:
                        clamp_arrays[k] = v
                    else:
                        clamp_arrays[k] = wp.full(
                            num_dofs_in_group,
                            float(v),
                            dtype=wp.float32,
                            device=wp_device,
                        )
                clampings.append(comp_cls(**clamp_arrays))

        actuator = Actuator(
            indices=indices,
            controller=controller,
            delay=delay,
            clamping=clampings if clampings else None,
        )
        actuators.append(actuator)

    return actuators
