# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX actuator control adapter."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.actuators.actuator_pd import ImplicitActuator
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.utils.queries import find_first_matching_prim
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from isaaclab_physx.physics import PhysxManager as SimulationManager

if TYPE_CHECKING:
    from isaaclab.actuators.actuator_collection import _ArticulationBinding
    from isaaclab.actuators.actuator_control import _ActuatorParameterWrite, _ResolvedSolverProperties
    from isaaclab.actuators.actuator_storage import _BackendParameterStaging

    from .articulation import Articulation

_HAS_NEWTON_ACTUATORS = importlib.util.find_spec("isaaclab_newton.actuators") is not None

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _mark_native_parameter_selection(
    ids: wp.array(dtype=Any),
    selection: wp.array(dtype=wp.bool),
) -> None:
    """Mark one reusable native parameter selection mask."""
    selection[wp.int32(ids[wp.tid()])] = True


@wp.kernel(enable_backward=False)
def _patch_native_parameter_mask(
    indices: wp.array(dtype=wp.uint32),
    canonical: wp.array2d(dtype=wp.float32),
    owner_slots: wp.array(dtype=wp.int32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    num_joints: int,
    destination: wp.array(dtype=wp.float32),
) -> None:
    """Patch one hosted-native parameter through full articulation masks."""
    native_slot = wp.tid()
    flat_joint = wp.int32(indices[native_slot])
    env_id = flat_joint // num_joints
    joint_id = flat_joint - env_id * num_joints
    compact_slot = owner_slots[joint_id]
    if compact_slot >= 0 and env_mask[env_id] and joint_mask[joint_id]:
        destination[native_slot] = canonical[env_id, compact_slot]


_MARK_NATIVE_PARAMETER_SELECTION = IndexKernelDispatcher(
    _mark_native_parameter_selection,
    ("ids",),
)


class PhysxActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the PhysX backend."""

    def __init__(self, articulation: Articulation):
        """Initialize the control adapter.

        Args:
            articulation: PhysX articulation that owns backend simulation handles.
        """
        super().__init__(articulation)
        self._native_active = False
        self._physx_actuator_wrapper = None
        self._all_env_mask: wp.array | None = None
        self._all_joint_mask: wp.array | None = None
        self._native_actuator_graphs: tuple[wp.Graph, wp.Graph] | None = None
        self._native_actuator_graph_index = 0
        self._backend_parameter_staging: _BackendParameterStaging | None = None
        self._dirty_backend_parameters: set[str] = set()
        self._resolved_property_backend_snapshot: dict[str, wp.array] | None = None
        self._resolved_property_cache_snapshot: dict[str, wp.array] | None = None
        self._all_env_mask = wp.ones(self.num_instances, dtype=wp.bool, device=self.device)
        self._all_joint_mask = wp.ones(self.num_joints, dtype=wp.bool, device=self.device)
        self._native_env_selection = wp.zeros(self.num_instances, dtype=wp.bool, device=self.device)
        self._native_joint_selection = wp.zeros(self.num_joints, dtype=wp.bool, device=self.device)

    def invalidate_actuator_view(self) -> None:
        """Release every candidate-owned PhysX and hosted-native binding."""
        try:
            self._clear_native_actuator_state()
        finally:
            self._dirty_backend_parameters.clear()
            self._backend_parameter_staging = None
            super().invalidate_actuator_view()

    def write_resolved_joint_properties_staged(self, properties: _ResolvedSolverProperties) -> None:
        """Apply one reversible set of resolved solver properties before publication."""
        if self._resolved_property_backend_snapshot is not None:
            raise RuntimeError("PhysX solver-property staging already owns an uncommitted snapshot.")
        articulation = self._articulation
        data = articulation.data
        root_view = articulation.root_view
        backend_snapshot = {
            "stiffness": wp.clone(root_view.get_dof_stiffnesses(), device="cpu"),
            "damping": wp.clone(root_view.get_dof_dampings(), device="cpu"),
            "effort_limit_sim": wp.clone(root_view.get_dof_max_forces(), device="cpu"),
            "velocity_limit_sim": wp.clone(root_view.get_dof_max_velocities(), device="cpu"),
            "armature": wp.clone(root_view.get_dof_armatures(), device="cpu"),
            "friction_properties": wp.clone(root_view.get_dof_friction_properties(), device="cpu"),
        }
        cache_attrs = (
            "_joint_stiffness",
            "_joint_damping",
            "_joint_effort_limits",
            "_joint_vel_limits",
            "_joint_armature",
            "_joint_friction_coeff",
            "_joint_dynamic_friction_coeff",
            "_joint_viscous_friction_coeff",
            "_joint_stiffness_backend",
            "_joint_damping_backend",
            "_joint_effort_limits_backend",
            "_joint_vel_limits_backend",
            "_joint_armature_backend",
            "_joint_friction_props_user",
            "_joint_friction_props_backend",
        )
        cache_snapshot = {
            name: wp.clone(value) for name in cache_attrs if (value := getattr(data, name, None)) is not None
        }
        self._resolved_property_backend_snapshot = backend_snapshot
        self._resolved_property_cache_snapshot = cache_snapshot

        def target(name: str) -> wp.array:
            canonical_target = properties.properties[name].canonical_target
            if canonical_target is None:
                raise RuntimeError(f"PhysX requires a device target for resolved {name!r} properties.")
            return canonical_target.warp

        articulation.write_joint_effort_limit_to_sim_index(limits=target("effort_limit_sim"), full_data=True)
        articulation.write_joint_velocity_limit_to_sim_index(limits=target("velocity_limit_sim"), full_data=True)
        articulation.write_joint_armature_to_sim_index(armature=target("armature"), full_data=True)
        articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=target("friction"),
            joint_dynamic_friction_coeff=target("dynamic_friction"),
            joint_viscous_friction_coeff=target("viscous_friction"),
            full_data=True,
        )
        articulation.write_joint_stiffness_to_sim_index(stiffness=target("stiffness"), full_data=True)
        articulation.write_joint_damping_to_sim_index(damping=target("damping"), full_data=True)

    def validate_resolved_joint_properties(self) -> None:
        """Validate PhysX configuration after staged solver writes and before publication."""
        self._articulation._validate_cfg()

    def restore_resolved_joint_properties(self) -> None:
        """Restore exact backend and data-cache rows after failed finalization."""
        backend_snapshot = self._resolved_property_backend_snapshot
        cache_snapshot = self._resolved_property_cache_snapshot
        if backend_snapshot is None or cache_snapshot is None:
            return
        articulation = self._articulation
        root_view = articulation.root_view
        indices = articulation._cpu_env_ids_all
        failures: list[Exception] = []
        backend_restores = (
            (root_view.set_dof_max_forces, backend_snapshot["effort_limit_sim"]),
            (root_view.set_dof_max_velocities, backend_snapshot["velocity_limit_sim"]),
            (root_view.set_dof_armatures, backend_snapshot["armature"]),
            (root_view.set_dof_friction_properties, backend_snapshot["friction_properties"]),
            (root_view.set_dof_stiffnesses, backend_snapshot["stiffness"]),
            (root_view.set_dof_dampings, backend_snapshot["damping"]),
        )
        try:
            for restore, value in backend_restores:
                try:
                    restore(value, indices=indices)
                except Exception as error:
                    failures.append(error)
            for name, value in cache_snapshot.items():
                try:
                    wp.copy(getattr(articulation.data, name), value)
                except Exception as error:
                    failures.append(error)
            articulation.data._reset_dynamics(mass_matrix=True)
        finally:
            self._resolved_property_backend_snapshot = None
            self._resolved_property_cache_snapshot = None
        if failures:
            first, *remaining = failures
            for error in remaining:
                first.add_note(f"Additional PhysX solver-property restore failure: {error}")
            raise first

    def commit_resolved_joint_properties(self) -> None:
        """Release successful candidate snapshots after publication completes."""
        self._resolved_property_backend_snapshot = None
        self._resolved_property_cache_snapshot = None

    def complete_articulation_initialization(self) -> None:
        """Prime and complete an articulation already validated before publication."""
        if self._actuator_binding is None or self._actuator_view is None:
            raise RuntimeError("Actuator facade must be prepared and bound before articulation completion.")
        articulation = self._articulation
        articulation.update(0.0)
        articulation._log_articulation_info()
        articulation.data.is_primed = True
        articulation._complete_deferred_initialization()

    def preflight_actuator_parameter_write(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Reject deferred PhysX drive writes during CUDA graph capture."""
        if (
            name in {"stiffness", "damping"}
            and issubclass(write.actuator_type, ImplicitActuator)
            and write.backend_parameter_staging is not None
            and wp.get_device(self.device).is_capturing
        ):
            raise RuntimeError("PhysX implicit actuator drive updates are not supported during CUDA graph capture.")

    def write_actuator_parameter(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Route implicit drives or hosted-native parameters through candidate metadata."""
        if (
            name in {"stiffness", "damping"}
            and issubclass(write.actuator_type, ImplicitActuator)
            and write.backend_parameter_staging is not None
        ):
            write.backend_parameter_staging.patch_write(actuator_type=write.actuator_type, name=name, write=write)
            self._dirty_backend_parameters.add(name)
            return
        if not self._native_active:
            return
        self._write_native_actuator_parameter(name, write)

    def _flush_dirty_backend_parameters(self) -> None:
        """Coalesce pending implicit drive writes into one full-row PhysX call per property."""
        staging = self._backend_parameter_staging
        if staging is None or not self._dirty_backend_parameters:
            return
        articulation = self._articulation
        for name in tuple(self._dirty_backend_parameters):
            target = staging.target(ImplicitActuator, name)
            if name == "stiffness":
                backend_target = articulation._get_backend_ordered_joint_buffer(
                    target.warp, articulation.data._joint_stiffness_backend
                )
                wp.copy(articulation.data._joint_stiffness, target.warp)
                wp.copy(articulation._cpu_joint_stiffness, backend_target)
                articulation.root_view.set_dof_stiffnesses(
                    articulation._cpu_joint_stiffness, indices=articulation._cpu_env_ids_all
                )
            else:
                backend_target = articulation._get_backend_ordered_joint_buffer(
                    target.warp, articulation.data._joint_damping_backend
                )
                wp.copy(articulation.data._joint_damping, target.warp)
                wp.copy(articulation._cpu_joint_damping, backend_target)
                articulation.root_view.set_dof_dampings(
                    articulation._cpu_joint_damping, indices=articulation._cpu_env_ids_all
                )
        self._dirty_backend_parameters.clear()

    def resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        """Resolve an optional environment mask to a full Warp bool mask.

        PhysX's articulation-level mask resolution converts masks to int32
        indices for its index-only tensor API. The collection's mask write
        path consumes full bool masks instead, so normalize here.
        """
        return self._resolve_bool_mask(env_mask, "_all_env_mask", self.num_instances)

    def resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        """Resolve an optional joint mask to a full Warp bool mask."""
        return self._resolve_bool_mask(joint_mask, "_all_joint_mask", self.num_joints)

    def _resolve_bool_mask(self, mask: wp.array | None, cache_attr: str, size: int) -> wp.array:
        if mask is None:
            cached = getattr(self, cache_attr)
            if cached is None:
                cached = wp.ones(size, dtype=wp.bool, device=self.device)
                setattr(self, cache_attr, cached)
            return cached
        if isinstance(mask, wp.array) and mask.dtype == wp.bool:
            return mask
        # Legacy mask resolution accepted any nonzero-selectable mask; keep that.
        mask_torch = wp.to_torch(mask) if isinstance(mask, wp.array) else mask
        return wp.from_torch((mask_torch != 0).contiguous(), dtype=wp.bool)

    def _write_joint_friction_properties(self, actuator) -> None:
        articulation = self._articulation
        super()._write_joint_friction_properties(actuator)
        articulation.write_joint_dynamic_friction_coefficient_to_sim_index(
            joint_dynamic_friction_coeff=actuator.dynamic_friction,
            joint_ids=actuator.joint_indices,
        )
        articulation.write_joint_viscous_friction_coefficient_to_sim_index(
            joint_viscous_friction_coeff=actuator.viscous_friction,
            joint_ids=actuator.joint_indices,
        )

    def discover_native_actuators(self, actuator_cfgs: dict) -> set[str]:
        """Classify hosted-Newton groups without mutating candidate or backend state."""
        articulation = self._articulation
        use_newton_actuators = getattr(articulation._sim_cfg, "use_newton_actuators", False)
        if use_newton_actuators and not _HAS_NEWTON_ACTUATORS:
            logger.warning(
                "use_newton_actuators is enabled but 'isaaclab_newton.actuators' is not available."
                " Newton-native actuators will be disabled and the simulation will fall back to the"
                " Isaac Lab actuator path. Install the isaaclab_newton extension to enable the fast path."
            )
            return set()
        if not (use_newton_actuators and _HAS_NEWTON_ACTUATORS):
            return set()
        return {name for name, actuator_cfg in actuator_cfgs.items() if not self._is_implicit_cfg(actuator_cfg)}

    def prepare_actuator_binding(self, binding: _ArticulationBinding) -> None:
        """Build hosted-Newton wrappers from private candidate binding aliases."""
        super().prepare_actuator_binding(binding)
        self._backend_parameter_staging = binding.backend_parameter_staging
        self._clear_native_actuator_state()
        articulation = self._articulation
        use_newton_actuators = getattr(articulation._sim_cfg, "use_newton_actuators", False)
        if not (use_newton_actuators and _HAS_NEWTON_ACTUATORS):
            return
        self._native_active = True
        articulation._has_newton_actuators = True

        from isaaclab_newton.actuators import (  # noqa: PLC0415
            NewtonActuatorAdapter,
            PhysxActuatorWrapper,
            build_implicit_dof_mask,
            build_native_dof_mask,
        )

        from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415

        self._physx_actuator_wrapper = PhysxActuatorWrapper.create(
            num_envs=self.num_instances,
            num_joints=self.num_joints,
            device=self.device,
        )
        articulation._physx_actuator_wrapper = self._physx_actuator_wrapper
        assert binding.groups is not None
        articulation._native_dof_mask, articulation._native_dof_mask_owner = build_native_dof_mask(
            dict(binding.groups), binding.native_group_names, self.num_joints, self.device
        )
        if binding.native_group_names:
            first_prim = find_first_matching_prim(articulation.cfg.prim_path)
            art_prim_path = str(first_prim.GetPath()) if first_prim is not None else None
            adapter = NewtonActuatorAdapter.from_usd(
                stage=get_current_stage(),
                joint_names=articulation.joint_names,
                num_envs=self.num_instances,
                num_joints=self.num_joints,
                device=self.device,
                articulation_prim_path=art_prim_path,
            )
            wrapper = self._physx_actuator_wrapper
            wrapper.joint_q = articulation._data.joint_pos.warp.reshape(-1)
            wrapper.joint_qd = articulation._data.joint_vel.warp.reshape(-1)
            assert binding.command is not None
            assert binding.joint_command is not None
            wrapper.joint_target_pos = binding.joint_command.position.warp.reshape(-1)
            wrapper.joint_target_vel = binding.joint_command.velocity.warp.reshape(-1)
            wrapper.joint_act = binding.joint_command.effort.warp.reshape(-1)
            wrapper.joint_f_2d = binding.joint_command.effort.warp
            wrapper.joint_f = wrapper.joint_f_2d.reshape(-1)
            adapter.finalize(wrapper)
            articulation.newton_actuator_adapter = adapter
            assert binding.groups is not None
            native_binding = adapter.bind_articulation(
                binding,
                dof_offset=0,
            )
            articulation._newton_native_ranges = native_binding.ranges
            articulation._implicit_dof_mask = native_binding.implicit_dof_mask
            articulation._implicit_dof_mask_owner = native_binding.implicit_dof_mask_owner
            articulation._data._sim_bind_joint_computed_effort = native_binding.computed_effort_view
            return

        assert binding.groups is not None
        articulation._implicit_dof_mask, articulation._implicit_dof_mask_owner = build_implicit_dof_mask(
            dict(binding.groups), self.num_joints, self.device
        )
        assert binding.computed_effort is not None
        articulation._data._sim_bind_joint_computed_effort = binding.computed_effort.warp

    def _clear_native_actuator_state(self) -> None:
        """Clear all hosted-Newton fields installed by candidate preparation."""
        self._native_active = False
        self._physx_actuator_wrapper = None
        self._native_actuator_graphs = None
        self._native_actuator_graph_index = 0
        articulation = self._articulation
        adapter = getattr(articulation, "newton_actuator_adapter", None)
        ranges = getattr(articulation, "_newton_native_ranges", None)
        if adapter is not None and ranges:
            adapter.unregister_articulation_ranges(ranges)
        articulation._physx_actuator_wrapper = None
        articulation.newton_actuator_adapter = None
        articulation._newton_native_ranges = None
        articulation._implicit_dof_mask = None
        articulation._implicit_dof_mask_owner = None
        articulation._native_dof_mask = None
        articulation._native_dof_mask_owner = None
        articulation._has_newton_actuators = False
        data = getattr(articulation, "_data", None)
        if data is not None:
            data._sim_bind_joint_computed_effort = None

    def _write_native_actuator_parameter(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Patch hosted-Newton controller arrays without allocating or synchronizing."""
        canonical = write.canonical
        owner_slots = write.backend_owner_slots
        adapter = getattr(self._articulation, "newton_actuator_adapter", None)
        if canonical is None or owner_slots is None or adapter is None:
            return
        use_masks = write.env_mask is not None or write.joint_mask is not None
        if use_masks:
            env_mask = self._all_env_mask if write.env_mask is None else write.env_mask
            joint_mask = self._all_joint_mask if write.joint_mask is None else write.joint_mask
        else:
            env_mask = self._resolve_native_parameter_selection(
                write.env_ids, self._all_env_mask, self._native_env_selection
            )
            joint_mask = self._resolve_native_parameter_selection(
                write.joint_ids, self._all_joint_mask, self._native_joint_selection
            )
        for actuator in adapter.actuators:
            for destination in self._native_parameter_destinations(actuator, name):
                wp.launch(
                    _patch_native_parameter_mask,
                    dim=actuator.indices.shape[0],
                    inputs=[
                        actuator.indices,
                        canonical.warp,
                        owner_slots,
                        env_mask,
                        joint_mask,
                        self.num_joints,
                    ],
                    outputs=[destination],
                    device=self.device,
                )

    def _resolve_native_parameter_selection(
        self,
        ids: torch.Tensor | wp.array | None,
        all_mask: wp.array,
        scratch_mask: wp.array,
    ) -> wp.array:
        """Resolve signed indices through stable, capture-safe selection storage."""
        if ids is None:
            return all_mask
        scratch_mask.zero_()
        wp.launch(
            _MARK_NATIVE_PARAMETER_SELECTION.select(ids),
            dim=ids.shape[0],
            inputs=[ids],
            outputs=[scratch_mask],
            device=self.device,
        )
        return scratch_mask

    @staticmethod
    def _native_parameter_destinations(actuator, name: str) -> tuple[wp.array, ...]:
        """Return per-DOF Newton arrays implementing one exact-type parameter."""
        controller_attrs = {"stiffness": ("kp",), "damping": ("kd",)}
        clamping_attrs = {
            "effort_limit": ("max_effort", "max_motor_effort"),
            "velocity_limit": ("velocity_limit",),
            "saturation_effort": ("saturation_effort",),
        }
        destinations = []
        for attr in controller_attrs.get(name, ()):
            value = getattr(actuator.controller, attr, None)
            if isinstance(value, wp.array):
                destinations.append(value)
        for component in getattr(actuator, "clamping", None) or ():
            for attr in clamping_attrs.get(name, ()):
                value = getattr(component, attr, None)
                if isinstance(value, wp.array):
                    destinations.append(value)
        return tuple(destinations)

    def finalize_native_actuators(self, collection: ActuatorCollection) -> None:
        """Deprecated collection-constructor hook retained for third-party callers."""
        del collection
        return

    def compute_native_actuators(self, collection: ActuatorCollection.ArticulationView, dt: float) -> None:
        if not self._native_active:
            return

        articulation = self._articulation
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        wp.launch(
            actuator_kernels.merge_native_command_fields,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                collection.command.position.warp,
                collection.command.velocity.warp,
                collection.command.effort.warp,
                articulation._native_dof_mask,
            ],
            outputs=[
                collection.joint_command.position.warp,
                collection.joint_command.velocity.warp,
                collection.joint_command.effort.warp,
            ],
            device=self.device,
        )
        if articulation.newton_actuator_adapter is not None:
            adapter = articulation.newton_actuator_adapter
            device = wp.get_device(self.device)
            if device.is_cuda and device.is_capturing and adapter.is_stateful:
                raise RuntimeError(
                    "stateful Newton actuators cannot run inside an outer CUDA graph capture; "
                    "let PhysX capture their alternating state graphs automatically"
                )
            if articulation.data.has_joint_ordering:
                # ``wrapper.joint_q``/``joint_qd`` were bound once (at actuator setup) to
                # ``_data.joint_pos``/``joint_vel``. With identity ordering those bindings alias
                # PhysX-owned memory directly and are always current. With non-identity ordering
                # they alias an owned shadow buffer that is only refreshed when the public getters
                # run -- which otherwise would not happen until the telemetry kernel below reads
                # them, one step too late for the adapter. Force the refresh here so the adapter
                # sees this step's state instead of a stale one-step-old shadow.
                articulation._data._refresh_joint_pos()
                articulation._data._refresh_joint_vel()
            if adapter.is_all_graphable and device.is_cuda:
                if not device.is_capturing:
                    if self._native_actuator_graphs is None:
                        self._capture_native_actuator_graphs(collection)
                    if self._native_actuator_graphs:
                        wp.capture_launch(self._native_actuator_graphs[self._native_actuator_graph_index])
                        adapter._swap_state_buffers()
                        self._native_actuator_graph_index ^= 1
                        return

        self._run_native_actuator_kernels(collection)

    def _run_native_actuator_kernels(self, collection: ActuatorCollection.ArticulationView) -> None:
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        wrapper = self._physx_actuator_wrapper
        if articulation.newton_actuator_adapter is not None:
            articulation.newton_actuator_adapter.gather_staged_ranges(articulation._newton_native_ranges or ())
            articulation.newton_actuator_adapter.step(wrapper, wrapper, SimulationManager.get_physics_dt())

        wp.launch(
            actuator_kernels.sync_torque_telemetry,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                articulation._data.joint_pos.warp,
                articulation._data.joint_vel.warp,
                collection.command.position.warp,
                collection.command.velocity.warp,
                articulation._data.joint_stiffness.warp,
                articulation._data.joint_damping.warp,
                articulation._data.joint_effort_limits.warp,
                articulation._implicit_dof_mask,
                articulation._native_dof_mask,
                wrapper.joint_f_2d,
                articulation._data._sim_bind_joint_computed_effort,
                articulation._ALL_JOINT_INDICES,
                False,
            ],
            outputs=[
                collection.computed_effort.warp,
                collection.applied_effort.warp,
            ],
            device=self.device,
        )

    def _capture_native_actuator_graphs(self, collection: ActuatorCollection.ArticulationView) -> None:
        adapter = self._articulation.newton_actuator_adapter
        if adapter is None:
            return
        states_a = adapter._states_a
        states_b = adapter._states_b
        graphs = []
        try:
            for _ in range(2):
                with wp.ScopedCapture(device=self.device, force_module_load=True) as capture:
                    self._run_native_actuator_kernels(collection)
                graphs.append(capture.graph)
        except Exception as exc:
            logger.warning("PhysX Newton-actuator CUDA graph capture failed; using eager execution: %s", exc)
            graphs = []
        finally:
            adapter._states_a = states_a
            adapter._states_b = states_b
        self._native_actuator_graphs = tuple(graphs) if graphs else ()
        self._native_actuator_graph_index = 0

    def submit_commands(self, collection: ActuatorCollection.ArticulationView) -> None:
        articulation = self._articulation
        self._flush_dirty_backend_parameters()
        # Gate on the articulation-level mirrors (kept in lockstep with
        # ``self._native_active`` / ``self._physx_actuator_wrapper`` by
        # :meth:`prepare_native_actuators`) exactly as the pre-collection
        # ``write_data_to_sim`` body did: subclasses that override
        # ``_process_actuators_cfg`` and tests stub these articulation attributes.
        if getattr(articulation, "_has_newton_actuators", False):
            # Newton fast path: pos/vel targets pass straight through; ``joint_f_2d`` already
            # merges Newton's explicit-DOF output with user feedforward.
            user_effort = articulation._physx_actuator_wrapper.joint_f_2d
            user_pos_target = collection.command.position.warp
            user_vel_target = collection.command.velocity.warp
        else:
            # Standard Lab actuator path: push the processed staging buffers PhysX-side.
            user_effort = collection.joint_command.effort.warp
            user_pos_target = collection.joint_command.position.warp
            user_vel_target = collection.joint_command.velocity.warp

        if articulation.data.has_joint_ordering:
            # One fused gather replaces the per-target reorder launches. PhysX has no
            # direct-drive joint-act output, so its gated-off output is left unset.
            wp.launch(
                ordering_kernels.reorder_joint_targets_user_to_backend,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    user_effort,
                    user_pos_target,
                    user_vel_target,
                    articulation.data.joint_ordering.backend_to_user,
                    True,
                    articulation._has_implicit_actuators,
                    articulation._has_implicit_actuators,
                    False,
                ],
                outputs=[
                    None,
                    articulation._joint_pos_target_backend,
                    articulation._joint_vel_target_backend,
                    articulation._joint_effort_target_backend,
                ],
                device=self.device,
            )
            effort_target = articulation._joint_effort_target_backend
            pos_target = articulation._joint_pos_target_backend
            vel_target = articulation._joint_vel_target_backend
        else:
            effort_target = user_effort
            pos_target = user_pos_target
            vel_target = user_vel_target

        articulation.root_view.set_dof_actuation_forces(effort_target, articulation._ALL_INDICES)
        if articulation._has_implicit_actuators:
            articulation.root_view.set_dof_position_targets(pos_target, articulation._ALL_INDICES)
            articulation.root_view.set_dof_velocity_targets(vel_target, articulation._ALL_INDICES)

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
        if self._native_active and self._articulation.newton_actuator_adapter is not None:
            self._articulation.newton_actuator_adapter.reset(env_ids)

    def write_native_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        adapter = self._articulation.newton_actuator_adapter
        if adapter is None:
            return

        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        env_id_pos = torch.full((self.num_instances,), -1, dtype=torch.int32, device=self.device)
        env_id_pos[env_ids.to(self.device, dtype=torch.long)] = torch.arange(
            env_ids.shape[0],
            dtype=torch.int32,
            device=self.device,
        )
        joint_id_pos = torch.full((self.num_joints,), -1, dtype=torch.int32, device=self.device)
        joint_ids_local = joint_ids.to(self.device, dtype=torch.long)
        joint_id_pos[joint_ids_local] = torch.arange(
            joint_ids.shape[0],
            dtype=torch.int32,
            device=self.device,
        )

        values_wp = wp.from_torch(values.to(self.device, dtype=torch.float32).contiguous(), dtype=wp.float32)
        env_id_pos_wp = wp.from_torch(env_id_pos, dtype=wp.int32)
        joint_id_pos_wp = wp.from_torch(joint_id_pos, dtype=wp.int32)

        for actuator in adapter.actuators:
            ctrl = actuator.controller
            if not hasattr(ctrl, attr):
                continue
            wp.launch(
                actuator_kernels.patch_actuator_param_kernel,
                dim=actuator.indices.shape[0],
                inputs=[
                    actuator.indices,
                    env_id_pos_wp,
                    joint_id_pos_wp,
                    values_wp,
                    0,
                    self.num_joints,
                ],
                outputs=[getattr(ctrl, attr)],
                device=self.device,
            )
