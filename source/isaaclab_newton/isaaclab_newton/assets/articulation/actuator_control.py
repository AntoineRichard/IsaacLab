# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton actuator control adapter."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import ModelFlags

from isaaclab.actuators import ActuatorCollection, ImplicitActuator
from isaaclab.actuators.actuator_control import ArticulationActuatorControl, _ActuatorParameterWrite
from isaaclab.assets.articulation import ordering_kernels

from isaaclab_newton.physics import NewtonManager as SimulationManager

if TYPE_CHECKING:
    from isaaclab.actuators.actuator_base_cfg import ActuatorBaseCfg
    from isaaclab.actuators.actuator_collection import _ArticulationBinding
    from isaaclab.actuators.actuator_control import _ResolvedSolverProperties

    from .articulation import Articulation

_HAS_NEWTON_ACTUATORS = importlib.util.find_spec("isaaclab_newton.actuators") is not None

logger = logging.getLogger(__name__)


class NewtonActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the Newton backend."""

    def __init__(self, articulation: Articulation):
        """Initialize the control adapter.

        Args:
            articulation: Newton articulation that owns backend simulation handles.
        """
        super().__init__(articulation)
        self._native_active = False
        self._native_dof_offset: int | None = None
        self._post_actuator_callback = None
        self._solver_property_snapshots: list[tuple[wp.array, wp.array]] | None = None
        self._candidate_state_snapshot: tuple[tuple[object, str, object], ...] | None = None

    def discover_native_actuators(self, cfgs: Mapping[str, ActuatorBaseCfg]) -> set[str]:
        """Classify native groups without mutating solver properties.

        Args:
            cfgs: Logical actuator configurations in declaration order.

        Returns:
            Names of groups physically owned by Newton actuators.
        """
        articulation = self._articulation
        self._native_active = False
        articulation._has_newton_actuators = False

        use_newton_actuators = getattr(articulation._sim_cfg, "use_newton_actuators", False)
        if use_newton_actuators and not _HAS_NEWTON_ACTUATORS:
            logger.warning(
                "use_newton_actuators is enabled but 'newton.actuators' is not available. "
                "Newton-native actuators will be disabled. Upgrade Newton to >= 1.2.0rc1."
            )
            return set()
        if not (use_newton_actuators and _HAS_NEWTON_ACTUATORS):
            return set()

        self._native_active = True
        articulation._has_newton_actuators = True
        SimulationManager.activate_newton_actuator_path()

        native_group_names: set[str] = set()
        for actuator_name, actuator_cfg in cfgs.items():
            if self._is_implicit_cfg(actuator_cfg):
                continue
            native_group_names.add(actuator_name)

        return native_group_names

    def write_resolved_joint_properties_staged(self, properties: _ResolvedSolverProperties) -> None:
        """Write the final device-resident property rowset to Newton once per field.

        Args:
            properties: Clone-resolved candidate properties in user joint order.
        """
        property_writers = (
            ("stiffness", "write_joint_stiffness_to_sim_mask", "stiffness"),
            ("damping", "write_joint_damping_to_sim_mask", "damping"),
            ("effort_limit_sim", "write_joint_effort_limit_to_sim_mask", "limits"),
            ("velocity_limit_sim", "write_joint_velocity_limit_to_sim_mask", "limits"),
            ("armature", "write_joint_armature_to_sim_mask", "armature"),
            (
                "friction",
                "write_joint_friction_coefficient_to_sim_mask",
                "joint_friction_coeff",
            ),
        )
        resolved_writes: list[tuple[str, str, wp.array]] = []
        for property_name, writer_name, argument_name in property_writers:
            resolved = properties.properties[property_name]
            if resolved.transport != "device" or resolved.canonical_target is None:
                raise RuntimeError(f"Newton requires a device canonical target for solver property '{property_name}'.")
            resolved_writes.append((writer_name, argument_name, resolved.canonical_target.warp))

        if self._solver_property_snapshots is not None:
            raise RuntimeError("Newton solver-property staging is already active for this articulation.")
        data = self._articulation.data
        property_buffers = (
            ("_sim_bind_joint_stiffness_sim", "_joint_stiffness_user"),
            ("_sim_bind_joint_damping_sim", "_joint_damping_user"),
            ("_sim_bind_joint_effort_limits_sim", "_joint_effort_limits_user"),
            ("_sim_bind_joint_vel_limits_sim", "_joint_vel_limits_user"),
            ("_sim_bind_joint_armature", "_joint_armature_user"),
            ("_sim_bind_joint_friction_coeff", "_joint_friction_coeff_user"),
        )
        snapshots: list[tuple[wp.array, wp.array]] = []
        for backend_name, user_name in property_buffers:
            backend_buffer = getattr(data, backend_name)
            snapshots.append((backend_buffer, wp.clone(backend_buffer)))
            user_buffer = getattr(data, user_name)
            if user_buffer is not None:
                snapshots.append((user_buffer, wp.clone(user_buffer)))
        self._solver_property_snapshots = snapshots

        try:
            for writer_name, argument_name, value in resolved_writes:
                getattr(self._articulation, writer_name)(**{argument_name: value})
        except Exception:
            self.restore_resolved_joint_properties()
            raise

    def restore_resolved_joint_properties(self) -> None:
        """Restore the pre-candidate Newton property arrays after rollback."""
        snapshots = self._solver_property_snapshots
        if snapshots is None:
            return
        try:
            for target, snapshot in snapshots:
                target.assign(snapshot)
        finally:
            self._solver_property_snapshots = None

    def commit_resolved_joint_properties(self) -> None:
        """Release rollback snapshots after candidate publication."""
        self._solver_property_snapshots = None

    def prepare_actuator_binding(self, binding: _ArticulationBinding) -> None:
        """Prepare native adapter state from the private candidate binding.

        Args:
            binding: Unpublished articulation binding owned by the candidate generation.
        """
        self._unregister_post_actuator_callback()
        super().prepare_actuator_binding(binding)
        if not self._native_active:
            return

        self._snapshot_candidate_state()
        try:
            self._prepare_native_actuator_binding(binding)
        except Exception as error:
            for cleanup in (self._unregister_post_actuator_callback, self._restore_candidate_state):
                try:
                    cleanup()
                except Exception as cleanup_error:
                    error.add_note(f"Failed to roll back Newton actuator preparation: {cleanup_error}")
            raise

    def _prepare_native_actuator_binding(self, binding: _ArticulationBinding) -> None:
        """Install generation-specific native or fallback state."""

        from newton import Model as NewtonModel  # noqa: PLC0415

        from isaaclab_newton.actuators import build_implicit_dof_mask, build_native_dof_mask  # noqa: PLC0415
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        if (
            binding.groups is None
            or binding.command is None
            or binding.computed_effort is None
            or binding.applied_effort is None
        ):
            raise RuntimeError("Newton actuator preparation requires a complete private articulation binding.")
        articulation._native_dof_mask, articulation._native_dof_mask_owner = build_native_dof_mask(
            dict(binding.groups), getattr(binding, "native_group_names", frozenset()), self.num_joints, self.device
        )
        adapter = SimulationManager._adapter
        if adapter is not None:
            dof_layout = articulation._root_view.frequency_layouts[NewtonModel.AttributeFrequency.JOINT_DOF]
            if dof_layout.slice is not None:
                arti_start = dof_layout.slice.start
            elif dof_layout.indices is not None:
                arti_start = int(dof_layout.indices.numpy()[0])
            else:
                arti_start = 0
            joint_ordering = articulation.data.joint_ordering
            native_binding = adapter.bind_articulation(
                binding,
                dof_offset=arti_start,
                joint_user_to_backend_indices=(
                    joint_ordering.user_to_backend_indices if joint_ordering is not None else None
                ),
            )
            articulation.newton_actuator_adapter = adapter
            articulation._newton_native_ranges = native_binding.ranges
            articulation._implicit_dof_mask = native_binding.implicit_dof_mask
            articulation._implicit_dof_mask_owner = native_binding.implicit_dof_mask_owner
            articulation._data._sim_bind_joint_computed_effort = native_binding.computed_effort_view
            self._native_dof_offset = arti_start
        else:
            articulation._implicit_dof_mask, articulation._implicit_dof_mask_owner = build_implicit_dof_mask(
                dict(binding.groups), self.num_joints, self.device
            )
            articulation._data._sim_bind_joint_computed_effort = wp.zeros(
                (self.num_instances, self.num_joints), dtype=wp.float32, device=self.device
            )

        def _post_actuator() -> None:
            wp.launch(
                actuator_kernels.sync_torque_telemetry,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    articulation._data._sim_bind_joint_pos,
                    articulation._data._sim_bind_joint_vel,
                    binding.command.position.warp,
                    binding.command.velocity.warp,
                    articulation._data.joint_stiffness.warp,
                    articulation._data.joint_damping.warp,
                    articulation._data.joint_effort_limits.warp,
                    articulation._implicit_dof_mask,
                    articulation._native_dof_mask,
                    articulation._data._sim_bind_joint_effort,
                    articulation._data._sim_bind_joint_computed_effort,
                    articulation._joint_user_to_backend_map(),
                    articulation.data.has_joint_ordering,
                ],
                outputs=[
                    binding.computed_effort.warp,
                    binding.applied_effort.warp,
                ],
                device=self.device,
            )

        self._post_actuator_callback = _post_actuator
        SimulationManager.register_post_actuator_callback(_post_actuator)

    def invalidate_actuator_view(self) -> None:
        """Restore candidate state before releasing the private binding."""
        failures: list[Exception] = []
        for cleanup in (
            self._unregister_post_actuator_callback,
            self._unregister_native_ranges,
            self._restore_candidate_state,
        ):
            try:
                cleanup()
            except Exception as error:
                failures.append(error)
        try:
            super().invalidate_actuator_view()
        except Exception as error:
            failures.append(error)
        if failures:
            first, *remaining = failures
            for error in remaining:
                first.add_note(f"Additional Newton actuator invalidation failure: {error}")
            raise first

    def _unregister_native_ranges(self) -> None:
        """Drop this articulation's exact global adapter registrations."""
        articulation = self._articulation
        adapter = getattr(articulation, "newton_actuator_adapter", None)
        ranges = getattr(articulation, "_newton_native_ranges", None)
        if adapter is not None and ranges:
            adapter.unregister_articulation_ranges(ranges)
        articulation._newton_native_ranges = None

    def _snapshot_candidate_state(self) -> None:
        """Retain and clear every field replaced by native candidate preparation."""
        if self._candidate_state_snapshot is not None:
            raise RuntimeError("Newton actuator candidate state is already installed.")
        articulation = self._articulation
        state_fields = (
            (articulation, "newton_actuator_adapter"),
            (articulation, "_newton_native_ranges"),
            (articulation, "_native_dof_mask"),
            (articulation, "_native_dof_mask_owner"),
            (articulation, "_implicit_dof_mask"),
            (articulation, "_implicit_dof_mask_owner"),
            (articulation._data, "_sim_bind_joint_computed_effort"),
        )
        self._candidate_state_snapshot = tuple(
            (owner, name, getattr(owner, name, None)) for owner, name in state_fields
        )
        for owner, name in state_fields:
            setattr(owner, name, None)
        self._native_dof_offset = None

    def _restore_candidate_state(self) -> None:
        """Best-effort restore of every generation-specific articulation field."""
        snapshot = getattr(self, "_candidate_state_snapshot", None)
        self._native_dof_offset = None
        if snapshot is None:
            return
        self._candidate_state_snapshot = None
        failures: list[Exception] = []
        for owner, name, value in snapshot:
            try:
                setattr(owner, name, value)
            except Exception as error:
                failures.append(error)
        if failures:
            first, *remaining = failures
            for error in remaining:
                first.add_note(f"Additional Newton candidate-state restore failure: {error}")
            raise first

    def _unregister_post_actuator_callback(self) -> None:
        """Remove this control's exact telemetry callback, if still registered."""
        callback = self._post_actuator_callback
        if callback is not None:
            try:
                SimulationManager.unregister_post_actuator_callback(callback)
            finally:
                self._post_actuator_callback = None

    def write_actuator_parameter(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Apply one canonical parameter update to Newton-owned runtime state.

        Args:
            name: Canonical actuator parameter name.
            write: Exact-type canonical storage and backend ownership metadata.
        """
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        canonical = write.canonical
        if canonical is None:
            raise RuntimeError("Newton parameter writes require canonical exact-type storage.")
        owner_slots = write.backend_owner_slots
        if isinstance(owner_slots, torch.Tensor):
            if owner_slots.dtype is not torch.int32 or owner_slots.ndim != 1:
                raise TypeError("Newton backend owner slots must be a one-dimensional int32 tensor.")
            owner_slots_wp = wp.from_torch(owner_slots, dtype=wp.int32)
        elif isinstance(owner_slots, wp.array):
            owner_slots_wp = owner_slots
        else:
            raise TypeError("Newton parameter writes require backend owner slots.")

        if issubclass(write.actuator_type, ImplicitActuator) and name in {"stiffness", "damping"}:
            data = articulation.data
            if name == "stiffness":
                backend_buffer = data._sim_bind_joint_stiffness_sim
                user_buffer = data._joint_stiffness_user
            else:
                backend_buffer = data._sim_bind_joint_damping_sim
                user_buffer = data._joint_damping_user
            has_joint_ordering = data.has_joint_ordering
            wp.launch(
                actuator_kernels.patch_implicit_solver_parameter,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    canonical.warp,
                    owner_slots_wp,
                    articulation._joint_user_to_backend_map(),
                    has_joint_ordering,
                ],
                outputs=[backend_buffer if user_buffer is None else user_buffer, backend_buffer],
                device=self.device,
            )
            SimulationManager.add_model_change(ModelFlags.JOINT_DOF_PROPERTIES)
            return

        adapter = getattr(articulation, "newton_actuator_adapter", None)
        if adapter is None or self._native_dof_offset is None:
            return

        backend_to_user = articulation._joint_backend_to_user_map()
        has_joint_ordering = articulation.data.has_joint_ordering
        clamping_attributes = {
            "effort_limit": ("max_motor_effort", "max_effort"),
            "velocity_limit": ("velocity_limit",),
            "saturation_effort": ("saturation_effort",),
        }
        for actuator in adapter.actuators:
            targets: list[tuple[object, str]] = []
            controller_attribute = {"stiffness": "kp", "damping": "kd"}.get(name)
            if controller_attribute is not None and hasattr(actuator.controller, controller_attribute):
                targets.append((actuator.controller, controller_attribute))
            for clamping in actuator.clamping:
                for attribute in clamping_attributes.get(name, ()):
                    if hasattr(clamping, attribute):
                        targets.append((clamping, attribute))

            for component, attribute in targets:
                target = getattr(component, attribute)
                wp.launch(
                    actuator_kernels.patch_native_actuator_parameter,
                    dim=actuator.indices.shape[0],
                    inputs=[
                        actuator.indices,
                        canonical.warp,
                        owner_slots_wp,
                        backend_to_user,
                        self._native_dof_offset,
                        self.num_joints,
                        adapter.num_joints,
                        self.num_instances,
                        has_joint_ordering,
                    ],
                    outputs=[target],
                    device=self.device,
                )
                if attribute in {"max_motor_effort", "velocity_limit", "saturation_effort"} and all(
                    hasattr(component, field)
                    for field in ("saturation_effort", "velocity_limit", "max_motor_effort", "corner_velocity")
                ):
                    wp.launch(
                        actuator_kernels.recompute_dc_motor_corner_velocity,
                        dim=actuator.indices.shape[0],
                        inputs=[component.saturation_effort, component.velocity_limit, component.max_motor_effort],
                        outputs=[component.corner_velocity],
                        device=self.device,
                    )

    def compute_native_actuators(self, collection: ActuatorCollection.ArticulationView, dt: float) -> None:
        """Merge native raw command fields without physically stepping Newton."""
        del dt
        if not self._native_active:
            return
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        native_mask = getattr(self._articulation, "_native_dof_mask", None)
        if native_mask is None:
            return
        wp.launch(
            actuator_kernels.merge_native_command_fields,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                collection.command.position.warp,
                collection.command.velocity.warp,
                collection.command.effort.warp,
                native_mask,
            ],
            outputs=[
                collection.joint_command.position.warp,
                collection.joint_command.velocity.warp,
                collection.joint_command.effort.warp,
            ],
            device=self.device,
        )

    def submit_commands(self, collection: ActuatorCollection | ActuatorCollection.ArticulationView) -> None:
        articulation = self._articulation
        if self._native_active:
            # Raw targets go directly to Newton's control object. Newton PD
            # consumes ``joint_act`` for explicit (Newton-managed) joints; the
            # solver's built-in joint drive does the PD for implicit joints
            # (whose stiffness/damping are non-zero in sim) and adds whatever
            # is in ``joint_f`` as feedforward. Identity ordering copies the
            # targets directly, while non-identity ordering gathers all four
            # targets in one launch.
            if not articulation.data.has_joint_ordering:
                articulation.data._sim_bind_joint_position_target.assign(collection.command.position.warp)
                articulation.data._sim_bind_joint_velocity_target.assign(collection.command.velocity.warp)
                articulation.data._sim_bind_joint_act.assign(collection.command.effort.warp)
                articulation.data._sim_bind_joint_effort.assign(collection.command.effort.warp)
            else:
                wp.launch(
                    ordering_kernels.reorder_joint_targets_user_to_backend,
                    dim=(self.num_instances, self.num_joints),
                    inputs=[
                        collection.command.effort.warp,
                        collection.command.position.warp,
                        collection.command.velocity.warp,
                        articulation._joint_backend_to_user_map(),
                        True,
                        True,
                        True,
                        True,
                    ],
                    outputs=[
                        articulation.data._sim_bind_joint_effort,
                        articulation.data._sim_bind_joint_position_target,
                        articulation.data._sim_bind_joint_velocity_target,
                        articulation.data._sim_bind_joint_act,
                    ],
                    device=self.device,
                )
            return

        # Standard Lab actuator path. Identity ordering copies processed
        # targets directly; non-identity ordering gathers them in one launch.
        if not articulation.data.has_joint_ordering:
            articulation.data._sim_bind_joint_effort.assign(collection.joint_command.effort.warp)
            if collection.has_implicit_actuators:
                articulation.data._sim_bind_joint_position_target.assign(collection.joint_command.position.warp)
                articulation.data._sim_bind_joint_velocity_target.assign(collection.joint_command.velocity.warp)
        else:
            wp.launch(
                ordering_kernels.reorder_joint_targets_user_to_backend,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    collection.joint_command.effort.warp,
                    collection.joint_command.position.warp,
                    collection.joint_command.velocity.warp,
                    articulation._joint_backend_to_user_map(),
                    True,
                    collection.has_implicit_actuators,
                    collection.has_implicit_actuators,
                    False,
                ],
                outputs=[
                    articulation.data._sim_bind_joint_effort,
                    articulation.data._sim_bind_joint_position_target,
                    articulation.data._sim_bind_joint_velocity_target,
                    articulation.data._sim_bind_joint_act,
                ],
                device=self.device,
            )

    def reset_native_actuators(
        self,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | wp.array(dtype=wp.int64) | slice,
    ) -> None:
        """Reset Newton-native actuator state for selected environments.

        Args:
            env_ids: Environment indices to reset, or a full slice.
        """
        if self._native_active and SimulationManager._adapter is not None:
            SimulationManager._adapter.reset(env_ids)

    def write_native_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        # TODO: This routes through per-actuator torch indexing and has no mask
        # variant because the actuator gain buffers are per-actuator torch views
        # over arbitrary joint-index subsets. A single-launch warp path and a mask
        # variant need the actuator-side buffer layout rework, deferred to the
        # actuator rework built on this series.
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        adapter = articulation.newton_actuator_adapter
        if adapter is None:
            return

        env_ids_wp = wp.from_torch(env_ids.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        env_mask = wp.zeros(self.num_instances, dtype=wp.bool, device=self.device)
        wp.launch(
            actuator_kernels.set_mask_kernel,
            dim=env_ids_wp.shape[0],
            inputs=[env_mask, env_ids_wp],
            device=self.device,
        )

        env_ids_long = env_ids.to(self.device, dtype=torch.long).unsqueeze(1)
        joint_ids_backend = joint_ids.to(self.device, dtype=torch.long)
        if articulation.data.has_joint_ordering:
            joint_ids_backend = articulation._joint_user_to_backend_torch[joint_ids_backend]
        joint_ids_backend = joint_ids_backend.unsqueeze(0)

        for actuator in adapter.actuators:
            ctrl = actuator.controller
            if not hasattr(ctrl, attr):
                continue
            cur_wp = articulation._root_view.get_actuator_parameter(actuator, ctrl, attr)
            cur_torch = wp.to_torch(cur_wp)
            cur_torch[env_ids_long, joint_ids_backend] = values.to(cur_torch.device, dtype=cur_torch.dtype)
            articulation._root_view.set_actuator_parameter(
                actuator=actuator,
                component=ctrl,
                name=attr,
                values=cur_wp,
                mask=env_mask,
            )
