# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX actuator control adapter."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.actuators import ActuatorCollection, ImplicitActuator
from isaaclab.actuators.actuator_control import (
    ArticulationActuatorControl,
    _ActuatorParameterWrite,
    _ResolvedSolverProperties,
)
from isaaclab.assets.articulation import ordering_kernels

from isaaclab_ovphysx import tensor_types as TT


class OvPhysxActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the OVPhysX backend."""

    def __init__(self, articulation) -> None:
        """Initialize deferred CPU-drive writes for one articulation."""
        super().__init__(articulation)
        self._stiffness_dirty = False
        self._damping_dirty = False

    def write_resolved_joint_properties_staged(self, properties: _ResolvedSolverProperties) -> None:
        """Apply finalized solver properties before the facade is published."""
        resolved = properties.properties
        articulation = self._articulation

        def _target(name: str) -> wp.array:
            target = resolved[name].canonical_target
            if target is None:
                raise RuntimeError(f"OVPhysX requires a device target for resolved {name!r} properties.")
            return target.warp

        articulation.write_joint_effort_limit_to_sim_index(limits=_target("effort_limit_sim"))
        articulation.write_joint_velocity_limit_to_sim_index(limits=_target("velocity_limit_sim"))
        articulation.write_joint_armature_to_sim_index(armature=_target("armature"))
        articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=_target("friction"),
            joint_dynamic_friction_coeff=_target("dynamic_friction"),
            joint_viscous_friction_coeff=_target("viscous_friction"),
        )
        articulation.write_joint_stiffness_to_sim_index(stiffness=_target("stiffness"))
        articulation.write_joint_damping_to_sim_index(damping=_target("damping"))

    def preflight_actuator_parameter_write(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Reject implicit-drive mutations that would issue CPU binding writes in capture."""
        del write
        if name not in {"stiffness", "damping"}:
            return
        device = wp.get_device(self.device)
        if device.is_cuda and device.is_capturing:
            raise RuntimeError("OVPhysX implicit drive updates are not capture-safe.")

    def write_actuator_parameter(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Patch dense implicit-drive staging without synchronizing or writing the binding."""
        if name not in {"stiffness", "damping"} or write.backend_parameter_staging is None:
            return
        write.backend_parameter_staging.patch_write(
            actuator_type=write.actuator_type,
            name=name,
            write=write,
        )
        if name == "stiffness":
            self._stiffness_dirty = True
        else:
            self._damping_dirty = True

    def _write_joint_friction_properties(self, actuator) -> None:
        # OVPhysX packs the three friction components into the single ``DOF_FRICTION_PROPERTIES``
        # binding, so they are written together in one call rather than per-component.
        self._articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=actuator.friction,
            joint_dynamic_friction_coeff=actuator.dynamic_friction,
            joint_viscous_friction_coeff=actuator.viscous_friction,
            joint_ids=actuator.joint_indices,
        )

    def stage_user_command(
        self,
        command_name: str,
        collection: ActuatorCollection | ActuatorCollection.ArticulationView,
        env_ids: torch.Tensor | wp.array | None,
        joint_ids: torch.Tensor | wp.array | None,
        env_mask: wp.array | None,
        joint_mask: wp.array | None,
    ) -> None:
        """Push a raw user command into the OVPhysX binding when a user setter runs.

        Mirrors the eager per-setter ``set_attribute`` write the legacy target setters
        performed: the collection has already written the public-order command buffer, so
        reorder it into backend order and push the selected env / joint slice to the
        matching backend binding.
        """
        tensor_type, can_write, user_buffer, backend_buffer = self._command_buffers(command_name, collection)
        if not can_write:
            return
        articulation = self._articulation
        target_backend = articulation._get_backend_ordered_joint_buffer(user_buffer, backend_buffer)
        if env_mask is not None:
            articulation._root_view.set_attribute(tensor_type, target_backend, mask=env_mask)
        elif env_ids is not None:
            articulation._root_view.set_attribute(tensor_type, target_backend, indices=env_ids)
        else:
            articulation._root_view.set_attribute(tensor_type, target_backend)

    def submit_commands(self, collection: ActuatorCollection | ActuatorCollection.ArticulationView) -> None:
        articulation = self._articulation
        self._flush_implicit_drive_properties(collection)
        # Write actions into simulation (zeros are safe when no actuators are active).
        # ``_applied_torque`` is the actuator-computed output (may differ from the raw
        # commanded target, e.g. once clipped), so it is reordered into its own scratch
        # buffer rather than ``_joint_effort_target_backend``. The latter is the persistent
        # mirror of the raw target that partial writes rely on for their unselected joints.
        write_effort = articulation._can_write_effort
        # position and velocity targets only for implicit actuators.
        write_pos = articulation._has_implicit_actuators and articulation._can_write_pos_target
        write_vel = articulation._has_implicit_actuators and articulation._can_write_vel_target
        if articulation.data.has_joint_ordering:
            if write_effort or write_pos or write_vel:
                # One fused gather replaces the per-target reorder launches. The effort
                # scratch also backs the unused joint_act output (its flag is off, so it is
                # never indexed).
                wp.launch(
                    ordering_kernels.reorder_joint_targets_user_to_backend,
                    dim=(self.num_instances, self.num_joints),
                    inputs=[
                        collection.applied_effort.warp,
                        collection.joint_command.position.warp,
                        collection.joint_command.velocity.warp,
                        articulation.data.joint_ordering.backend_to_user,
                        write_effort,
                        write_pos,
                        write_vel,
                        False,
                    ],
                    outputs=[
                        None,
                        articulation._joint_pos_target_backend,
                        articulation._joint_vel_target_backend,
                        articulation._applied_torque_backend,
                    ],
                    device=self.device,
                )
            effort = articulation._applied_torque_backend
            pos_target = articulation._joint_pos_target_backend
            vel_target = articulation._joint_vel_target_backend
        else:
            effort = collection.applied_effort.warp
            pos_target = collection.joint_command.position.warp
            vel_target = collection.joint_command.velocity.warp
        if write_effort:
            articulation._root_view.set_attribute(TT.DOF_ACTUATION_FORCE, effort)
        if write_pos:
            articulation._root_view.set_attribute(TT.DOF_POSITION_TARGET, pos_target)
        if write_vel:
            articulation._root_view.set_attribute(TT.DOF_VELOCITY_TARGET, vel_target)

    def _command_buffers(
        self,
        command_name: str,
        collection: ActuatorCollection | ActuatorCollection.ArticulationView,
    ) -> tuple[TT.TensorType, bool, wp.array, wp.array | None]:
        articulation = self._articulation
        if command_name == "position":
            return (
                TT.DOF_POSITION_TARGET,
                articulation._can_write_pos_target,
                collection.command.position.warp,
                articulation._joint_pos_target_backend,
            )
        if command_name == "velocity":
            return (
                TT.DOF_VELOCITY_TARGET,
                articulation._can_write_vel_target,
                collection.command.velocity.warp,
                articulation._joint_vel_target_backend,
            )
        if command_name == "effort":
            return (
                TT.DOF_ACTUATION_FORCE,
                articulation._can_write_effort,
                collection.command.effort.warp,
                articulation._joint_effort_target_backend,
            )
        raise ValueError(f"Unsupported actuator command buffer '{command_name}'.")

    def _flush_implicit_drive_properties(
        self, collection: ActuatorCollection | ActuatorCollection.ArticulationView
    ) -> None:
        """Flush each dirty implicit property once through OVPhysX's CPU-only binding."""
        if not (self._stiffness_dirty or self._damping_dirty):
            return
        staging = getattr(collection, "_backend_parameter_staging", None)
        if staging is None:
            raise RuntimeError("OVPhysX implicit drive staging is unavailable for the published actuator view.")
        articulation = self._articulation
        data = articulation.data
        for name, dirty in (("stiffness", self._stiffness_dirty), ("damping", self._damping_dirty)):
            if not dirty:
                continue
            target = staging.target(ImplicitActuator, name).warp
            backend_buffer = getattr(data, f"_joint_{name}_backend")
            backend_target = articulation._get_backend_ordered_joint_buffer(target, backend_buffer)
            cpu_target = getattr(data, f"_cpu_joint_{name}")
            wp.copy(cpu_target, backend_target)
            tensor_type = TT.DOF_STIFFNESS if name == "stiffness" else TT.DOF_DAMPING
            articulation._root_view.set_attribute(tensor_type, cpu_target)
        self._stiffness_dirty = False
        self._damping_dirty = False
