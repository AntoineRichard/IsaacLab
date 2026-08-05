# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton actuator control adapter."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_control import ArticulationActuatorControl, _ActuatorParameterWrite
from isaaclab.assets.articulation import ordering_kernels

from isaaclab_newton.physics import NewtonManager as SimulationManager

if TYPE_CHECKING:
    from isaaclab.actuators.actuator_base_cfg import ActuatorBaseCfg
    from isaaclab.actuators.actuator_collection import _ArticulationBinding

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
        self._post_actuator_callback = None

    def discover_native_actuators(self, cfgs: dict[str, ActuatorBaseCfg]) -> set[str]:
        """Discover native groups and synchronously prepare solver drive ownership."""
        articulation = self._articulation
        articulation._has_newton_actuators = False
        articulation._implicit_dof_mask = None
        articulation.newton_actuator_adapter = None
        articulation.newton_default_stiffness = None
        articulation.newton_default_damping = None
        articulation.newton_managed_local_joints = None

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
        explicit_joint_ids: list[int] = []
        for actuator_name, actuator_cfg in cfgs.items():
            if self._is_implicit_cfg(actuator_cfg):
                continue
            native_group_names.add(actuator_name)
            joint_ids, _ = articulation.find_joints(actuator_cfg.joint_names_expr)
            explicit_joint_ids.extend(int(joint_id) for joint_id in joint_ids)

        if explicit_joint_ids:
            explicit_ids_t = torch.tensor(
                sorted(set(explicit_joint_ids)),
                dtype=torch.int32,
                device=self.device,
            )
            articulation.write_joint_stiffness_to_sim_index(stiffness=0.0, joint_ids=explicit_ids_t)
            articulation.write_joint_damping_to_sim_index(damping=0.0, joint_ids=explicit_ids_t)

        return native_group_names

    def prepare_actuator_binding(self, binding: _ArticulationBinding) -> None:
        """Prepare native adapter state from the private candidate binding."""
        self._unregister_post_actuator_callback()
        super().prepare_actuator_binding(binding)
        if not self._native_active:
            return

        from newton import Model as NewtonModel  # noqa: PLC0415

        from isaaclab_newton.actuators import build_implicit_dof_mask  # noqa: PLC0415
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        if (
            binding.groups is None
            or binding.command is None
            or binding.computed_effort is None
            or binding.applied_effort is None
        ):
            raise RuntimeError("Newton actuator preparation requires a complete private articulation binding.")
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
                lab_actuators=dict(binding.groups),
                dof_offset=arti_start,
                num_joints=self.num_joints,
                joint_user_to_backend_indices=(
                    joint_ordering.user_to_backend_indices if joint_ordering is not None else None
                ),
            )
            articulation.newton_actuator_adapter = adapter
            articulation.newton_default_stiffness = native_binding.stiffness
            articulation.newton_default_damping = native_binding.damping
            articulation.newton_managed_local_joints = native_binding.joint_indices
            articulation._implicit_dof_mask = native_binding.implicit_dof_mask
            articulation._implicit_dof_mask_owner = native_binding.implicit_dof_mask_owner
            articulation._data._sim_bind_joint_computed_effort = native_binding.computed_effort_view
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
        try:
            SimulationManager.register_post_actuator_callback(_post_actuator)
        except Exception:
            self._unregister_post_actuator_callback()
            raise

    def invalidate_actuator_view(self) -> None:
        """Deregister candidate telemetry before releasing its private binding."""
        self._unregister_post_actuator_callback()
        super().invalidate_actuator_view()

    def _unregister_post_actuator_callback(self) -> None:
        """Remove this control's exact telemetry callback, if still registered."""
        callback = self._post_actuator_callback
        if callback is not None:
            SimulationManager.unregister_post_actuator_callback(callback)
            self._post_actuator_callback = None

    def write_actuator_parameter(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Defer native controller parameter side effects until canonical pointer binding.

        The current Newton adapter owns copied gain/output snapshots.  Routing a
        canonical typed-storage update through it would require compacting the
        scoped selector, copying the selected values, and potentially syncing.
        Task 11 replaces those snapshots with direct candidate-store pointers;
        only then can this hook update native parameters allocation- and
        synchronization-free.
        """
        del name, write

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

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
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
