# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed rigid object collection asset."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object_collection import (
    BaseRigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.assets.kernels import _body_wrench_to_world
from isaaclab_ovphysx.physics import OvPhysxManager

from .rigid_object_collection_data import RigidObjectCollectionData


class RigidObjectCollection(BaseRigidObjectCollection):
    """OVPhysX-backed rigid object collection asset."""

    cfg: RigidObjectCollectionCfg
    """Configuration instance for the rigid object collection."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object collection."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """Initialize the rigid object collection.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        # Bindings are stored per tensor-type, per body.
        # Layout: _bindings[tensor_type] is a list of length num_bodies,
        # where _bindings[tensor_type][b] is the TensorBinding for body b.
        # Entries start as None and are populated lazily via _get_binding().
        self._bindings: dict[int, list[Any]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> RigidObjectCollectionData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        return list(self._body_names)

    @property
    def root_view(self):
        """Per-body TensorBinding dictionary.

        Returns the internal bindings dict keyed by TensorType constant.
        Each value is a list of length :attr:`num_bodies` containing one
        :class:`~isaaclab_ovphysx.TensorBinding` per body.
        """
        return self._bindings

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:  # type: ignore[override]
        """Returns the instantaneous wrench composer for the rigid object collection.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer will be applied for a single simulation step and then cleared.
        """
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:  # type: ignore[override]
        """Returns the permanent wrench composer for the rigid object collection.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer will be applied every simulation step until explicitly reset.
        """
        return self._permanent_wrench_composer

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reset(
        self,
        env_ids: Sequence[int] | wp.array | None = None,
        object_ids: slice | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Reset the wrench composers for the specified environments.

        Args:
            env_ids: Environment indices to reset. If None, all environments are reset.
            object_ids: Unused — included for interface compatibility with the base class.
            env_mask: Boolean environment mask. If provided, takes precedence over ``env_ids``.
        """
        self._instantaneous_wrench_composer.reset(env_ids=env_ids, env_mask=env_mask)
        self._permanent_wrench_composer.reset(env_ids=env_ids, env_mask=env_mask)

    def write_data_to_sim(self) -> None:  # type: ignore[override]
        """Write external wrench to the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        inst = self._instantaneous_wrench_composer
        perm = self._permanent_wrench_composer
        if not inst.active and not perm.active:
            return
        if inst.active:
            if perm.active:
                inst.add_raw_buffers_from(perm)
            force_b = inst.out_force_b.warp
            torque_b = inst.out_torque_b.warp
        else:
            force_b = perm.out_force_b.warp
            torque_b = perm.out_torque_b.warp

        poses = self._data.body_link_pose_w.warp  # (N, B) wp.transformf
        wp.launch(
            _body_wrench_to_world,
            dim=(self._num_instances, self._num_bodies),
            inputs=[force_b, torque_b, poses],
            outputs=[self._wrench_buf],
            device=self._device,
        )
        for b in range(self._num_bodies):
            binding = self._get_binding(TT.RIGID_BODY_WRENCH, body_idx=b)
            binding.write(self._wrench_buf_flat[b])
        inst.reset()

    def update(self, dt: float) -> None:  # type: ignore[override]
        self._data.update(dt)

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str]]:  # type: ignore[override]
        """Find bodies in the rigid body collection based on the name keys.

        Please check the :func:`isaaclab.utils.string.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        obj_ids, obj_names = resolve_matching_names(name_keys, self.body_names, preserve_order)
        return torch.tensor(obj_ids, device=self._device, dtype=torch.int32), obj_names

    # ------------------------------------------------------------------
    # Pose writers (3 pairs)
    # ------------------------------------------------------------------

    def write_body_pose_to_sim_index(
        self,
        *,
        body_poses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body pose over selected environment and body indices into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        For rigid bodies the actor frame coincides with the link frame, so this delegates to
        :meth:`write_body_link_pose_to_sim_index`.

        .. note::
            This method expects partial data.

        Args:
            body_poses: Body poses in simulation frame [m, rad]. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_body_link_pose_to_sim_index(body_poses=body_poses, body_ids=body_ids, env_ids=env_ids)

    def write_body_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body pose over selected environment and body masks into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        For rigid bodies the actor frame coincides with the link frame, so this delegates to
        :meth:`write_body_link_pose_to_sim_mask`.

        .. note::
            This method expects full data.

        Args:
            body_poses: Body poses in simulation frame [m, rad]. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.write_body_link_pose_to_sim_mask(body_poses=body_poses, body_mask=body_mask, env_mask=env_mask)

    def write_body_link_pose_to_sim_index(
        self,
        *,
        body_poses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body link pose over selected environment and body indices into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        Args:
            body_poses: Body link poses in simulation frame [m, rad]. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(body_poses, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "body_poses")
        wp.launch(
            shared_kernels.set_body_link_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_poses, env_ids, body_ids, False],
            outputs=[
                self.data._body_link_pose_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps so the next read recomposes them.
        self.data._body_com_pose_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        # Push updated per-body link poses to simulation via bindings.
        for b in self._iter_body_ids_cpu(body_ids):
            binding = self._get_binding(TT.RIGID_BODY_POSE, body_idx=b)
            binding.write(self.data._body_link_pose_w_flat[b].view(wp.float32), indices=env_ids)

    def write_body_link_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body link pose over selected environment and body masks into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        Args:
            body_poses: Body link poses in simulation frame [m, rad]. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_INDICES_ENV
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_INDICES_BODY
        self.assert_shape_and_dtype(body_poses, (self._num_instances, self._num_bodies), wp.transformf, "body_poses")
        wp.launch(
            shared_kernels.set_body_link_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_poses, env_ids, body_ids, True],
            outputs=[
                self.data._body_link_pose_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps so the next read recomposes them.
        self.data._body_com_pose_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        # Push updated per-body link poses to simulation via bindings.
        for b in self._iter_body_ids_cpu(body_ids):
            binding = self._get_binding(TT.RIGID_BODY_POSE, body_idx=b)
            binding.write(self.data._body_link_pose_w_flat[b].view(wp.float32), indices=env_ids)

    def write_body_com_pose_to_sim_index(
        self,
        *,
        body_poses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body center of mass pose over selected environment and body indices into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects partial data.

        Args:
            body_poses: Body center of mass poses in simulation frame [m, rad].
                Shape is (len(env_ids), len(body_ids), 7) or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(body_poses, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "body_poses")
        wp.launch(
            shared_kernels.set_body_com_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_poses, self.data.body_com_pose_b, env_ids, body_ids, False],
            outputs=[
                self.data._body_com_pose_w.data,
                self.data._body_link_pose_w.data,
                self.data._body_com_state_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps so the next read recomposes them.
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # Push updated per-body link poses to simulation via bindings (OVPhysX only exposes link frame).
        for b in self._iter_body_ids_cpu(body_ids):
            binding = self._get_binding(TT.RIGID_BODY_POSE, body_idx=b)
            binding.write(self.data._body_link_pose_w_flat[b].view(wp.float32), indices=env_ids)

    def write_body_com_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body center of mass pose over selected environment and body masks into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects full data.

        Args:
            body_poses: Body center of mass poses in simulation frame [m, rad].
                Shape is (num_instances, num_bodies, 7) or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_INDICES_ENV
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_INDICES_BODY
        self.assert_shape_and_dtype(body_poses, (self._num_instances, self._num_bodies), wp.transformf, "body_poses")
        wp.launch(
            shared_kernels.set_body_com_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_poses, self.data.body_com_pose_b, env_ids, body_ids, True],
            outputs=[
                self.data._body_com_pose_w.data,
                self.data._body_link_pose_w.data,
                self.data._body_com_state_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps so the next read recomposes them.
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # Push updated per-body link poses to simulation via bindings (OVPhysX only exposes link frame).
        for b in self._iter_body_ids_cpu(body_ids):
            binding = self._get_binding(TT.RIGID_BODY_POSE, body_idx=b)
            binding.write(self.data._body_link_pose_w_flat[b].view(wp.float32), indices=env_ids)

    # ------------------------------------------------------------------
    # Velocity writers (3 pairs)
    # ------------------------------------------------------------------

    def write_body_velocity_to_sim_index(
        self,
        *,
        body_velocities: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            For rigid bodies the actor frame coincides with the center of mass frame, so this
            delegates to :meth:`write_body_com_velocity_to_sim_index`.

        .. note::
            This method expects partial data.

        Args:
            body_velocities: Body velocities in simulation world frame [m/s, rad/s].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_body_com_velocity_to_sim_index(body_velocities=body_velocities, body_ids=body_ids, env_ids=env_ids)

    def write_body_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body velocity over selected environment and body masks into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            For rigid bodies the actor frame coincides with the center of mass frame, so this
            delegates to :meth:`write_body_com_velocity_to_sim_mask`.

        .. note::
            This method expects full data.

        Args:
            body_velocities: Body velocities in simulation world frame [m/s, rad/s].
                Shape is (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.write_body_com_velocity_to_sim_mask(
            body_velocities=body_velocities, body_mask=body_mask, env_mask=env_mask
        )

    def write_body_link_velocity_to_sim_index(
        self,
        *,
        body_velocities: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body link velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's frame rather than the body's center of mass.

        .. note::
            This method expects partial data.

        Args:
            body_velocities: Body link velocities in simulation world frame [m/s, rad/s].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(
            body_velocities, (env_ids.shape[0], body_ids.shape[0]), wp.spatial_vectorf, "body_velocities"
        )
        wp.launch(
            shared_kernels.set_body_link_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_velocities,
                self.data.body_com_pose_b,
                self.data.body_link_pose_w,
                env_ids,
                body_ids,
                False,
            ],
            outputs=[
                self.data._body_link_vel_w.data,
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
                self.data._body_com_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps so the next read recomposes them.
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # Push updated per-body COM velocities to simulation via bindings.
        for b in self._iter_body_ids_cpu(body_ids):
            binding = self._get_binding(TT.RIGID_BODY_VELOCITY, body_idx=b)
            binding.write(self.data._body_com_vel_w_flat[b].view(wp.float32), indices=env_ids)

    def write_body_link_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body link velocity over selected environment and body masks into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's frame rather than the body's center of mass.

        .. note::
            This method expects full data.

        Args:
            body_velocities: Body link velocities in simulation world frame [m/s, rad/s].
                Shape is (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_INDICES_ENV
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_INDICES_BODY
        self.assert_shape_and_dtype(
            body_velocities, (self._num_instances, self._num_bodies), wp.spatial_vectorf, "body_velocities"
        )
        wp.launch(
            shared_kernels.set_body_link_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_velocities,
                self.data.body_com_pose_b,
                self.data.body_link_pose_w,
                env_ids,
                body_ids,
                True,
            ],
            outputs=[
                self.data._body_link_vel_w.data,
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
                self.data._body_com_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps so the next read recomposes them.
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # Push updated per-body COM velocities to simulation via bindings.
        for b in self._iter_body_ids_cpu(body_ids):
            binding = self._get_binding(TT.RIGID_BODY_VELOCITY, body_idx=b)
            binding.write(self.data._body_com_vel_w_flat[b].view(wp.float32), indices=env_ids)

    def write_body_com_velocity_to_sim_index(
        self,
        *,
        body_velocities: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body center of mass velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects partial data.

        Args:
            body_velocities: Body center of mass velocities in simulation world frame [m/s, rad/s].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(
            body_velocities, (env_ids.shape[0], body_ids.shape[0]), wp.spatial_vectorf, "body_velocities"
        )
        wp.launch(
            shared_kernels.set_body_com_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_velocities, env_ids, body_ids, False],
            outputs=[
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                self.data._body_state_w.data,
                self.data._body_com_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps so the next read recomposes them.
        self.data._body_link_vel_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        # Push updated per-body COM velocities to simulation via bindings.
        for b in self._iter_body_ids_cpu(body_ids):
            binding = self._get_binding(TT.RIGID_BODY_VELOCITY, body_idx=b)
            binding.write(self.data._body_com_vel_w_flat[b].view(wp.float32), indices=env_ids)

    def write_body_com_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set the body center of mass velocity over selected environment and body masks into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects full data.

        Args:
            body_velocities: Body center of mass velocities in simulation world frame [m/s, rad/s].
                Shape is (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_INDICES_ENV
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_INDICES_BODY
        self.assert_shape_and_dtype(
            body_velocities, (self._num_instances, self._num_bodies), wp.spatial_vectorf, "body_velocities"
        )
        wp.launch(
            shared_kernels.set_body_com_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_velocities, env_ids, body_ids, True],
            outputs=[
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                self.data._body_state_w.data,
                self.data._body_com_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps so the next read recomposes them.
        self.data._body_link_vel_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        # Push updated per-body COM velocities to simulation via bindings.
        for b in self._iter_body_ids_cpu(body_ids):
            binding = self._get_binding(TT.RIGID_BODY_VELOCITY, body_idx=b)
            binding.write(self.data._body_com_vel_w_flat[b].view(wp.float32), indices=env_ids)

    # ------------------------------------------------------------------
    # Property setters (3 pairs)
    # ------------------------------------------------------------------

    def set_masses_index(
        self,
        *,
        masses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body masses over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``RIGID_BODY_MASS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        Args:
            masses: Body masses [kg]. Shape is (len(env_ids), len(body_ids))
                with dtype wp.float32.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(masses, (env_ids.shape[0], body_ids.shape[0]), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[masses, env_ids, body_ids],
            outputs=[self.data._body_mass],
            device=self._device,
        )
        # Push each body column to its CPU-only binding via pinned-host staging.
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        for b in self._iter_body_ids_cpu(body_ids):
            # _mass_flat_base is (B*N,) float32 body-major; body b occupies [b*N : (b+1)*N].
            gpu_col = wp.array(
                ptr=self.data._mass_flat_base.ptr + b * self._num_instances * 4,
                shape=(self._num_instances,),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            wp.copy(self._cpu_mass_staging[b], gpu_col)
            binding = self._get_binding(TT.RIGID_BODY_MASS, body_idx=b)
            binding.write(self._cpu_mass_staging[b], indices=cpu_env_ids)

    def set_masses_mask(
        self,
        *,
        masses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body masses over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``RIGID_BODY_MASS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        Args:
            masses: Body masses [kg]. Shape is (num_instances, num_bodies)
                with dtype wp.float32.
            body_mask: Body mask. If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_INDICES_ENV
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_INDICES_BODY
        self.assert_shape_and_dtype(masses, (self._num_instances, self._num_bodies), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[masses, self._resolve_env_mask(env_mask), self._resolve_body_mask(body_mask)],
            outputs=[self.data._body_mass],
            device=self._device,
        )
        # Push each body column to its CPU-only binding via pinned-host staging.
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        for b in self._iter_body_ids_cpu(body_ids):
            gpu_col = wp.array(
                ptr=self.data._mass_flat_base.ptr + b * self._num_instances * 4,
                shape=(self._num_instances,),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            wp.copy(self._cpu_mass_staging[b], gpu_col)
            binding = self._get_binding(TT.RIGID_BODY_MASS, body_idx=b)
            binding.write(self._cpu_mass_staging[b], indices=cpu_env_ids)

    def set_coms_index(
        self,
        *,
        coms: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body center-of-mass poses over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``RIGID_BODY_COM_POSE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        Args:
            coms: Body center-of-mass poses [m, quaternion (w, x, y, z)].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(coms, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "coms")
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[coms, env_ids, body_ids],
            outputs=[self.data._body_com_pose_b.data],
            device=self._device,
        )
        # Invalidate derived buffers that depend on body_com_pose_b.
        self.data._body_com_pose_w.timestamp = -1.0
        # Push each body column to its CPU-only binding via pinned-host staging.
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        tf_size = 7 * 4  # 7 floats × 4 bytes per wp.transformf
        for b in self._iter_body_ids_cpu(body_ids):
            # _body_com_pose_b_base is (B*N,) transformf body-major; view body b as (N*7,) float32.
            gpu_col_f32 = wp.array(
                ptr=self.data._body_com_pose_b_base.ptr + b * self._num_instances * tf_size,
                shape=(self._num_instances * 7,),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            # _cpu_com_staging[b] is (N, 7) float32; view as flat (N*7,) for the copy.
            cpu_col_flat = wp.array(
                ptr=self._cpu_com_staging[b].ptr,
                shape=(self._num_instances * 7,),
                dtype=wp.float32,
                device="cpu",
                copy=False,
            )
            wp.copy(cpu_col_flat, gpu_col_f32)
            binding = self._get_binding(TT.RIGID_BODY_COM_POSE, body_idx=b)
            binding.write(self._cpu_com_staging[b], indices=cpu_env_ids)

    def set_coms_mask(
        self,
        *,
        coms: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body center-of-mass poses over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``RIGID_BODY_COM_POSE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        Args:
            coms: Body center-of-mass poses [m, quaternion (w, x, y, z)].
                Shape is (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_INDICES_ENV
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_INDICES_BODY
        self.assert_shape_and_dtype(coms, (self._num_instances, self._num_bodies), wp.transformf, "coms")
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[coms, self._resolve_env_mask(env_mask), self._resolve_body_mask(body_mask)],
            outputs=[self.data._body_com_pose_b.data],
            device=self._device,
        )
        # Invalidate derived buffers that depend on body_com_pose_b.
        self.data._body_com_pose_w.timestamp = -1.0
        # Push each body column to its CPU-only binding via pinned-host staging.
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        tf_size = 7 * 4  # 7 floats × 4 bytes per wp.transformf
        for b in self._iter_body_ids_cpu(body_ids):
            gpu_col_f32 = wp.array(
                ptr=self.data._body_com_pose_b_base.ptr + b * self._num_instances * tf_size,
                shape=(self._num_instances * 7,),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            cpu_col_flat = wp.array(
                ptr=self._cpu_com_staging[b].ptr,
                shape=(self._num_instances * 7,),
                dtype=wp.float32,
                device="cpu",
                copy=False,
            )
            wp.copy(cpu_col_flat, gpu_col_f32)
            binding = self._get_binding(TT.RIGID_BODY_COM_POSE, body_idx=b)
            binding.write(self._cpu_com_staging[b], indices=cpu_env_ids)

    def set_inertias_index(
        self,
        *,
        inertias: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body inertia tensors over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``RIGID_BODY_INERTIA`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        Args:
            inertias: Body inertia tensors [kg·m²]. Shape is
                (len(env_ids), len(body_ids), 9) with dtype wp.float32.
                The 9 components are the row-major flatten of the 3×3 inertia
                matrix (Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz).
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(inertias, (env_ids.shape[0], body_ids.shape[0], 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[inertias, env_ids, body_ids],
            outputs=[self.data._body_inertia],
            device=self._device,
        )
        # Push each body column to its CPU-only binding via pinned-host staging.
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        for b in self._iter_body_ids_cpu(body_ids):
            # _inertia_flat_base is (B*N*9,) float32 body-major; body b occupies [b*N*9 : (b+1)*N*9].
            gpu_col = wp.array(
                ptr=self.data._inertia_flat_base.ptr + b * self._num_instances * 9 * 4,
                shape=(self._num_instances, 9),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            wp.copy(self._cpu_inertia_staging[b], gpu_col)
            binding = self._get_binding(TT.RIGID_BODY_INERTIA, body_idx=b)
            binding.write(self._cpu_inertia_staging[b], indices=cpu_env_ids)

    def set_inertias_mask(
        self,
        *,
        inertias: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body inertia tensors over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``RIGID_BODY_INERTIA`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        Args:
            inertias: Body inertia tensors [kg·m²]. Shape is
                (num_instances, num_bodies, 9) with dtype wp.float32.
                The 9 components are the row-major flatten of the 3×3 inertia
                matrix (Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz).
            body_mask: Body mask. If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_INDICES_ENV
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_INDICES_BODY
        self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[inertias, self._resolve_env_mask(env_mask), self._resolve_body_mask(body_mask)],
            outputs=[self.data._body_inertia],
            device=self._device,
        )
        # Push each body column to its CPU-only binding via pinned-host staging.
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        for b in self._iter_body_ids_cpu(body_ids):
            gpu_col = wp.array(
                ptr=self.data._inertia_flat_base.ptr + b * self._num_instances * 9 * 4,
                shape=(self._num_instances, 9),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            wp.copy(self._cpu_inertia_staging[b], gpu_col)
            binding = self._get_binding(TT.RIGID_BODY_INERTIA, body_idx=b)
            binding.write(self._cpu_inertia_staging[b], indices=cpu_env_ids)

    # ------------------------------------------------------------------
    # Deprecated state writers
    # ------------------------------------------------------------------

    def write_body_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:  # type: ignore[override]
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index` and
        :meth:`write_body_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_link_pose_to_sim_index' and 'write_body_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert wp.array to torch.Tensor for slicing.
        if isinstance(body_states, wp.array):
            body_states = wp.to_torch(body_states)
        self.write_body_link_pose_to_sim_index(body_poses=body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_com_velocity_to_sim_index(
            body_velocities=body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids
        )

    def write_body_link_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:  # type: ignore[override]
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index` and
        :meth:`write_body_link_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_link_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_link_pose_to_sim_index' and 'write_body_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert wp.array to torch.Tensor for slicing.
        if isinstance(body_states, wp.array):
            body_states = wp.to_torch(body_states)
        self.write_body_link_pose_to_sim_index(body_poses=body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_link_velocity_to_sim_index(
            body_velocities=body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids
        )

    def write_body_com_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:  # type: ignore[override]
        """Deprecated, same as :meth:`write_body_com_pose_to_sim_index` and
        :meth:`write_body_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_com_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_com_pose_to_sim_index' and 'write_body_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert wp.array to torch.Tensor for slicing.
        if isinstance(body_states, wp.array):
            body_states = wp.to_torch(body_states)
        self.write_body_com_pose_to_sim_index(body_poses=body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_com_velocity_to_sim_index(
            body_velocities=body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids
        )

    # ------------------------------------------------------------------
    # Internal hooks
    # ------------------------------------------------------------------

    def _initialize_impl(self) -> None:
        """Initialize the rigid object collection from the OVPhysX simulation backend.

        For each body in :attr:`cfg.rigid_objects`, validates the prim tree,
        converts the IsaacLab prim path to an fnmatch glob, and eagerly creates
        per-body TensorBindings for all standard rigid-body tensor types.
        Then creates the :class:`RigidObjectCollectionData` container and primes
        the asset-side buffers.
        """
        # Step 1: Acquire OVPhysX instance and device.
        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        self._ovphysx = physx_instance
        self._device = OvPhysxManager.get_device()

        # Step 2: Iterate over each body in the collection config.
        # Build per-body glob patterns and body names; validate the prim tree.
        self._binding_patterns: list[str] = []
        self._body_names: list[str] = []

        for name, obj_cfg in self.cfg.rigid_objects.items():
            # Convert IsaacLab prim-path notation to the fnmatch-style glob that
            # OVPhysX create_tensor_binding expects.  Two conventions are in use:
            #   /World/envs/env_.*/object   -- regex dot-star for any env index
            #   /World/envs/{ENV_REGEX_NS}/object -- explicit placeholder
            pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", obj_cfg.prim_path)
            pattern = re.sub(r"\.\*", "*", pattern)

            # Validate the prim tree before creating tensor bindings.
            # OVPhysX silently returns a zero-count binding when the pattern
            # matches nothing; fail fast here with a clear message instead.
            template_prim = sim_utils.find_first_matching_prim(obj_cfg.prim_path)
            if template_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{obj_cfg.prim_path}' (body '{name}').")
            template_prim_path = template_prim.GetPath().pathString

            root_prims = sim_utils.get_all_matching_child_prims(
                template_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
                traverse_instance_prims=False,
            )
            if len(root_prims) == 0:
                raise RuntimeError(
                    f"Failed to find a rigid body when resolving '{obj_cfg.prim_path}' (body '{name}')."
                    " Please ensure that the prim has 'USD RigidBodyAPI' applied."
                )
            if len(root_prims) > 1:
                raise RuntimeError(
                    f"Failed to find a single rigid body when resolving '{obj_cfg.prim_path}' (body '{name}')."
                    f" Found multiple '{root_prims}' under '{template_prim_path}'."
                    " Please ensure that there is only one rigid body in the prim path tree."
                )

            articulation_prims = sim_utils.get_all_matching_child_prims(
                template_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                traverse_instance_prims=False,
            )
            if len(articulation_prims) != 0:
                if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                    raise RuntimeError(
                        f"Found an articulation root when resolving '{obj_cfg.prim_path}' (body '{name}') in the"
                        f" rigid object collection. These are located at: '{articulation_prims}' under"
                        f" '{template_prim_path}'. Please disable the articulation root in the USD or from code by"
                        " setting the parameter 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the"
                        " spawn configuration."
                    )

            # Extend the glob to the RigidBodyAPI prim when it sits below the
            # template root (mirrors PhysX collection's root_prim_path_expr logic).
            root_prim_path = root_prims[0].GetPath().pathString
            suffix = root_prim_path[len(template_prim_path) :]
            if suffix:
                pattern = pattern + suffix

            self._binding_patterns.append(pattern)
            self._body_names.append(name)

        # Step 3: Total number of distinct body types.
        self._num_bodies = len(self._binding_patterns)

        # Step 4: Eagerly create per-body bindings for all standard tensor types.
        # This surfaces any wheel-side failures here, with a helpful message, rather
        # than as a raw exception on first write.
        for tt in (
            TT.RIGID_BODY_POSE,
            TT.RIGID_BODY_VELOCITY,
            TT.RIGID_BODY_WRENCH,
            TT.RIGID_BODY_MASS,
            TT.RIGID_BODY_COM_POSE,
            TT.RIGID_BODY_INERTIA,
        ):
            for b in range(self._num_bodies):
                try:
                    self._get_binding(tt, body_idx=b)
                except Exception as e:
                    raise RuntimeError(
                        f"OVPhysX could not create rigid-body binding {tt!r} for body"
                        f" '{self._body_names[b]}' (pattern={self._binding_patterns[b]!r})."
                        f" Check that the prim path matches at least one UsdPhysics.RigidBodyAPI"
                        f" prim and that the ovphysx wheel exposes the RIGID_BODY_* TensorType."
                    ) from e

        # Step 5: Read num_instances from the POSE binding of body 0 and validate
        # that all per-body bindings agree on the same count.
        ref_count = self._bindings[TT.RIGID_BODY_POSE][0].count
        for tt in self._bindings:
            for b, binding in enumerate(self._bindings[tt]):
                if binding.count != ref_count:
                    raise RuntimeError(
                        f"Per-body instance count mismatch for tensor type {tt!r}:"
                        f" body 0 has {ref_count} instances but body '{self._body_names[b]}'"
                        f" (index {b}) has {binding.count} instances."
                        " All bodies in the collection must have the same number of environment instances."
                    )
        self._num_instances = ref_count

        # Step 6: Create the data container.
        self._data = RigidObjectCollectionData(
            root_view=self._bindings,
            num_objects=self._num_bodies,
            device=self._device,
        )

        # Step 7: Pre-allocate asset-side buffers.
        self._create_buffers()

        # Step 8: Apply initial state from configuration.
        self._process_cfg()

        # Step 9: Prime buffers with zero acceleration history.
        self.update(0.0)

    def _create_buffers(self) -> None:
        """Pre-allocate asset-side index arrays and CPU staging buffers."""
        N = self._num_instances
        B = self._num_bodies

        self._ALL_INDICES_ENV = wp.array(np.arange(N), dtype=wp.int32, device=self._device)
        self._ALL_INDICES_BODY = wp.array(np.arange(B), dtype=wp.int32, device=self._device)

        # CPU copy of all-env indices used when calling CPU-only binding.write().
        self._cpu_all_env_ids = wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True)
        wp.copy(self._cpu_all_env_ids, self._ALL_INDICES_ENV)

        # Per-body pinned CPU staging buffers for CPU-only write paths.
        # Each entry holds one body-column worth of data for all N instances.
        # Used by set_masses_*, set_coms_*, set_inertias_* to stage GPU data
        # onto the CPU before calling binding.write() (RIGID_BODY_MASS /
        # RIGID_BODY_COM_POSE / RIGID_BODY_INERTIA are CPU-only bindings).
        # Shapes match the wheel's binding.write() expected layout:
        #   mass      -> (N,)     float32  (one scalar per instance)
        #   com_pose  -> (N, 7)   float32  (position + quaternion per instance)
        #   inertia   -> (N, 9)   float32  (row-major 3×3 matrix per instance)
        self._cpu_mass_staging = [wp.zeros(N, dtype=wp.float32, device="cpu", pinned=True) for _ in range(B)]
        self._cpu_com_staging = [wp.zeros((N, 7), dtype=wp.float32, device="cpu", pinned=True) for _ in range(B)]
        self._cpu_inertia_staging = [wp.zeros((N, 9), dtype=wp.float32, device="cpu", pinned=True) for _ in range(B)]

        # All-true boolean masks used as defaults in mask-based kernel calls.
        self._ALL_TRUE_ENV_MASK = wp.array(np.ones(N, dtype=bool), dtype=wp.bool, device=self._device)
        self._ALL_TRUE_BODY_MASK = wp.array(np.ones(B, dtype=bool), dtype=wp.bool, device=self._device)

        # External wrench buffers (mirrors rigid_object._create_buffers, extended to 2D).
        # Body-major flat base: body b occupies rows [b*N : (b+1)*N].
        # Per-body contiguous (N, 9) views alias the same allocation, so the
        # per-body binding writes never copy.  A strided (N, B, 9) view on top
        # lets the kernel write all bodies in a single launch.
        self._wrench_buf_base = wp.zeros(B * N * 9, dtype=wp.float32, device=self._device)
        # Per-body contiguous (N, 9) sub-views used by per-body binding.write().
        self._wrench_buf_flat = [
            wp.array(
                ptr=self._wrench_buf_base.ptr + b * N * 9 * 4,
                shape=(N, 9),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            for b in range(B)
        ]
        # Strided (N, B, 9) view: [i, j, k] → flat[j*N*9 + i*9 + k].
        self._wrench_buf = wp.array(
            ptr=self._wrench_buf_base.ptr,
            shape=(N, B, 9),
            dtype=wp.float32,
            strides=(9 * 4, N * 9 * 4, 4),
            device=self._device,
        )
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # Set body names into the data container (mirrors PhysX collection).
        self._data.body_names = self._body_names

    def _process_cfg(self) -> None:
        """Post-processing of configuration parameters.

        Reads the per-body initial state from :attr:`cfg.rigid_objects` and
        broadcasts it across all environment instances to produce
        ``(num_instances, num_bodies, data_size)`` default-state arrays.
        """
        default_body_poses = []
        default_body_vels = []

        for obj_cfg in self.cfg.rigid_objects.values():
            default_body_pose = tuple(obj_cfg.init_state.pos) + tuple(obj_cfg.init_state.rot)
            default_body_vel = tuple(obj_cfg.init_state.lin_vel) + tuple(obj_cfg.init_state.ang_vel)
            # Broadcast across num_instances: (data_size,) -> (num_instances, data_size)
            default_body_pose = np.tile(np.array(default_body_pose, dtype=np.float32), (self._num_instances, 1))
            default_body_vel = np.tile(np.array(default_body_vel, dtype=np.float32), (self._num_instances, 1))
            default_body_poses.append(default_body_pose)
            default_body_vels.append(default_body_vel)

        # Stack per-body arrays: each (num_instances, data_size) -> (num_instances, num_bodies, data_size)
        default_body_poses = np.stack(default_body_poses, axis=1)
        default_body_vels = np.stack(default_body_vels, axis=1)
        self._data.default_body_pose = wp.array(default_body_poses, dtype=wp.transformf, device=self._device)
        self._data.default_body_vel = wp.array(default_body_vels, dtype=wp.spatial_vectorf, device=self._device)

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_binding(self, tensor_type: int, body_idx: int):
        """Return a cached per-body TensorBinding, creating it on first access.

        Bindings are lightweight handles into OVPhysX's shared GPU buffers.
        Creating one does not allocate new GPU memory; the underlying buffers
        are allocated once by PhysX regardless of how many bindings reference
        them.

        The binding cache is stored in ``self._bindings`` as a dict from
        ``tensor_type`` to a list of length :attr:`num_bodies`.

        Args:
            tensor_type: The TensorType constant identifying which simulation
                buffer to bind (e.g. :attr:`~isaaclab_ovphysx.tensor_types.RIGID_BODY_POSE`).
            body_idx: The index of the body within the collection (0-based).

        Returns:
            The cached TensorBinding for ``tensor_type`` and ``body_idx``.

        Raises:
            Whatever the OVPhysX wheel raises if ``create_tensor_binding`` fails.
        """
        if tensor_type not in self._bindings:
            # Initialise with None placeholders for all bodies.
            self._bindings[tensor_type] = [None] * self._num_bodies
        binding = self._bindings[tensor_type][body_idx]
        if binding is not None:
            return binding
        binding = self._ovphysx.create_tensor_binding(pattern=self._binding_patterns[body_idx], tensor_type=tensor_type)
        self._bindings[tensor_type][body_idx] = binding
        return binding

    # ------------------------------------------------------------------
    # Internal helpers -- ID resolution
    # ------------------------------------------------------------------

    def _resolve_env_ids(self, env_ids) -> wp.array:
        """Resolve environment indices to a warp int32 array on ``self._device`` (mirrors PhysX).

        Tests sometimes hand us indices on CPU even when the sim runs on GPU; we move the
        resolved array onto ``self._device`` so kernel launches don't fail on a device
        mismatch.
        """
        if env_ids is None or env_ids == slice(None):
            return self._ALL_INDICES_ENV
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self._device)
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(env_ids, wp.array) and str(env_ids.device) != self._device:
            env_ids = wp.clone(env_ids, device=self._device)
        return env_ids

    def _resolve_body_ids(self, body_ids) -> wp.array:
        """Resolve body indices to a warp int32 array on ``self._device`` (mirrors PhysX)."""
        if body_ids is None or body_ids == slice(None):
            return self._ALL_INDICES_BODY
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self._device)
        return body_ids

    def _resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        """Resolve an environment mask to a ``wp.bool`` array on ``self._device``.

        ``None`` returns the pre-allocated all-true mask.

        Args:
            env_mask: Boolean environment mask or None. Shape is (num_instances,).

        Returns:
            A ``wp.bool`` array of shape (num_instances,) on ``self._device``.
        """
        if env_mask is None:
            return self._ALL_TRUE_ENV_MASK
        if isinstance(env_mask, torch.Tensor):
            return wp.from_torch(env_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(env_mask, wp.array) and str(env_mask.device) != self._device:
            env_mask = wp.clone(env_mask, device=self._device)
        return env_mask

    def _resolve_body_mask(self, body_mask: wp.array | None) -> wp.array:
        """Resolve a body mask to a ``wp.bool`` array on ``self._device``.

        ``None`` returns the pre-allocated all-true mask.

        Args:
            body_mask: Boolean body mask or None. Shape is (num_bodies,).

        Returns:
            A ``wp.bool`` array of shape (num_bodies,) on ``self._device``.
        """
        if body_mask is None:
            return self._ALL_TRUE_BODY_MASK
        if isinstance(body_mask, torch.Tensor):
            return wp.from_torch(body_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(body_mask, wp.array) and str(body_mask.device) != self._device:
            body_mask = wp.clone(body_mask, device=self._device)
        return body_mask

    def _iter_body_ids_cpu(self, body_ids: wp.array) -> Iterable[int]:
        """Iterate over body IDs on CPU.

        Warp arrays cannot drive Python control flow; this helper converts
        a warp int32 array to an iterator of Python ints on CPU.

        Args:
            body_ids: A warp array of int32 body indices.

        Returns:
            An iterator over the body IDs as Python integers.
        """
        cpu = wp.to_torch(body_ids).cpu().numpy().tolist()
        return iter(cpu)

    def _get_cpu_env_ids(self, env_ids: wp.array) -> wp.array:
        """Return CPU int32 env indices for CPU-only binding writes.

        Uses the pre-allocated pinned ``_cpu_all_env_ids`` fast path when
        *env_ids* covers all instances, otherwise clones to CPU.

        Args:
            env_ids: A warp int32 array of environment indices on any device.

        Returns:
            A warp int32 array guaranteed to live on ``"cpu"``.
        """
        if env_ids.ptr == self._ALL_INDICES_ENV.ptr:
            return self._cpu_all_env_ids
        return wp.clone(env_ids, device="cpu")
