# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed rigid object collection asset."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
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
from isaaclab_ovphysx.assets.kernels import _body_wrench_to_world, resolve_view_ids
from isaaclab_ovphysx.physics import OvPhysxManager

from .rigid_object_collection_data import RigidObjectCollectionData

# ---------------------------------------------------------------------------
# Internal adapter — presents B per-body RIGID_BODY bindings as a single
# (N, B, D) binding, preserving the dict-keyed interface the data class relies on.
# ---------------------------------------------------------------------------


class _FusedRigidBodyBinding:
    """Adapter wrapping B per-body ``RIGID_BODY_*`` bindings as one ``(N, B, D)`` view.

    The OVPhysX ``RIGID_BODY_*`` tensor types return flat ``(N, D)`` arrays for a
    single-body pattern.  A rigid-object collection contains *B* distinct body types,
    each backed by its own ``(N, D)`` binding.  This adapter presents the union as a
    single binding with ``shape = (N, B, D)`` and ``count = N``, satisfying the
    interface expected by :class:`RigidObjectCollectionData`.

    **Reads** assemble the B ``(N, D)`` staging reads into the caller-supplied
    ``(N, B, D)`` destination (warp array or NumPy array).

    **Writes** deassemble a ``(N, B, D)`` source into per-body ``(N, D)`` tensors and
    dispatch each to its corresponding per-body binding.

    The adapter is device-agnostic: staging buffers are allocated on the same device
    as the destination array at first use (GPU for GPU bindings, CPU for CPU-only
    bindings).  This avoids cross-device copies for CPU-resident quantities such as
    mass, COM pose, and inertia.

    Args:
        per_body_bindings: List of B ``TensorBinding`` objects, one per body type.
            Each must expose ``.count``, ``.shape``, ``.read()``, and ``.write()``.
        N: Number of environment instances.
        B: Number of body types.
        device: Simulation device string (e.g. ``"cuda:0"``).  Used as the default
            device for write staging; read staging is adapted to the destination device.
    """

    def __init__(self, per_body_bindings: list, N: int, B: int, device: str) -> None:
        if not per_body_bindings:
            raise ValueError("per_body_bindings must contain at least one binding.")
        self._per_body = per_body_bindings
        self._N = N
        self._B = B
        self._device = device
        # Infer D from the first per-body binding's shape.
        # RIGID_BODY_MASS has shape (N,); others have (N, D).
        first_shape = per_body_bindings[0].shape
        self._D: int = first_shape[1] if len(first_shape) > 1 else 1
        self._scalar = len(first_shape) == 1  # True for RIGID_BODY_MASS (shape (N,))
        # Public attributes mirroring TensorBinding.
        self.count: int = N
        self.shape: tuple = (N, B) if self._scalar else (N, B, self._D)
        # Per-device staging buffers for reads: keyed by device string.
        self._read_staging: dict[str, list[wp.array]] = {}
        # Per-device staging buffers for writes: keyed by device string.
        self._write_staging: dict[str, list[wp.array]] = {}

    # ------------------------------------------------------------------
    # Staging buffer helpers
    # ------------------------------------------------------------------

    def _get_staging(self, cache: dict, staging_device: str) -> list[wp.array]:
        """Return (creating if needed) a list of B staging arrays on *staging_device*."""
        if staging_device in cache:
            return cache[staging_device]
        if self._scalar:
            bufs = [wp.zeros((self._N,), dtype=wp.float32, device=staging_device) for _ in range(self._B)]
        else:
            bufs = [wp.zeros((self._N, self._D), dtype=wp.float32, device=staging_device) for _ in range(self._B)]
        cache[staging_device] = bufs
        return bufs

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, dst) -> None:
        """Read all per-body bindings and assemble into *dst*.

        Args:
            dst: Destination array.  May be a :class:`warp.array` (GPU or CPU) or a
                :class:`numpy.ndarray`.  Must have shape matching :attr:`shape`
                (``(N, B)`` for scalar quantities, ``(N, B, D)`` otherwise) and
                dtype ``float32``.
        """
        if isinstance(dst, np.ndarray):
            self._read_numpy(dst)
        else:
            self._read_warp(dst)

    def _read_numpy(self, dst: np.ndarray) -> None:
        """Read into a NumPy array (CPU path, used by ``_read_cpu`` in :class:`RigidObjectCollectionData`)."""
        for b, binding in enumerate(self._per_body):
            per_body_np = np.zeros(binding.shape, dtype=np.float32)
            binding.read(per_body_np)
            if self._scalar:
                dst[:, b] = per_body_np  # (N,) → column b of (N, B)
            else:
                dst[:, b, :] = per_body_np  # (N, D) → column b of (N, B, D)

    def _read_warp(self, dst: wp.array) -> None:
        """Read into a warp float32 array.

        Per-body staging buffers are allocated on the same device as *dst* so that
        GPU bindings write to GPU staging and CPU-only bindings write to CPU staging,
        avoiding illegal cross-device reads.
        """
        staging_device = str(dst.device)
        staging = self._get_staging(self._read_staging, staging_device)
        for b, binding in enumerate(self._per_body):
            binding.read(staging[b])
        # Assemble staging[b] into dst via torch zero-copy views.
        dst_t = wp.to_torch(dst)
        if self._scalar:
            dst_2d = dst_t.view(self._N, self._B)
            for b in range(self._B):
                dst_2d[:, b].copy_(wp.to_torch(staging[b]))
        else:
            dst_3d = dst_t.view(self._N, self._B, self._D)
            for b in range(self._B):
                dst_3d[:, b, :].copy_(wp.to_torch(staging[b]))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, tensor, indices=None, mask=None) -> None:
        """Deassemble *tensor* ``(N, B, D)`` and write to each per-body binding.

        Each per-body binding receives the full ``(N, D)`` column for its body index
        along with the optional *indices* or *mask* filter, matching the OVPhysX
        ``TensorBinding.write`` contract (full-shape tensor, selective row application).

        Args:
            tensor: Source warp float32 array with shape matching :attr:`shape`.
            indices: Optional int32 warp/torch array of environment row indices.
            mask: Optional bool warp array mask (takes precedence over *indices*).
        """
        write_device = str(tensor.device) if isinstance(tensor, wp.array) else self._device
        staging = self._get_staging(self._write_staging, write_device)
        src_t = wp.to_torch(tensor)
        if self._scalar:
            src_2d = src_t.view(self._N, self._B)
            for b, binding in enumerate(self._per_body):
                col = src_2d[:, b].contiguous()
                staging[b].assign(wp.from_torch(col, dtype=wp.float32))
                binding.write(staging[b], indices=indices, mask=mask)
        else:
            src_3d = src_t.view(self._N, self._B, self._D)
            for b, binding in enumerate(self._per_body):
                col = src_3d[:, b, :].contiguous()
                staging[b].assign(wp.from_torch(col, dtype=wp.float32))
                binding.write(staging[b], indices=indices, mask=mask)


class RigidObjectCollection(BaseRigidObjectCollection):
    """OVPhysX-backed rigid object collection asset.

    Uses one ``RIGID_BODY_*`` :class:`TensorBinding` per body type per tensor type.
    Each per-body binding covers all environment instances for that body type
    (shape ``(num_instances, D)``).  A :class:`_FusedRigidBodyBinding` adapter
    wraps the *B* per-body bindings so that the data class and write helpers see a
    single binding with shape ``(num_instances, num_bodies, D)`` — the same interface
    produced by the articulation-mode mock used in iface tests.
    """

    cfg: RigidObjectCollectionCfg
    """Configuration instance for the rigid object collection."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object collection."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """Initialize the rigid object collection.

        Args:
            cfg: A configuration instance.
        """
        # Note: We never call the parent constructor as it tries to call its own spawning which we don't want.
        # Mirrors :class:`isaaclab_physx.assets.RigidObjectCollection`.
        cfg.validate()
        self.cfg = cfg.copy()
        self._is_initialized = False
        # Spawn the rigid objects -- one prim per object_cfg.
        for rigid_body_cfg in self.cfg.rigid_objects.values():
            if rigid_body_cfg.spawn is not None:
                rigid_body_cfg.spawn.func(
                    rigid_body_cfg.prim_path,
                    rigid_body_cfg.spawn,
                    translation=rigid_body_cfg.init_state.pos,
                    orientation=rigid_body_cfg.init_state.rot,
                )
            matching_prims = sim_utils.find_matching_prims(rigid_body_cfg.prim_path)
            if len(matching_prims) == 0:
                raise RuntimeError(f"Could not find prim with path {rigid_body_cfg.prim_path}.")
        # Body name storage populated by ``_initialize_impl``.
        self._body_names_list: list[str] = []
        # Single binding per tensor type (mirrors Articulation).
        # Populated lazily via _get_binding() or eagerly in _initialize_impl().
        self._bindings: dict[int, Any] = {}
        # Register callbacks (initialize / invalidate / prim deletion).
        self._register_callbacks()
        self._debug_vis_handle = None

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
        return list(self._body_names_list)

    @property
    def root_view(self):
        """Fused TensorBinding dictionary.

        Returns the internal bindings dict keyed by TensorType constant.
        Each value is a single :class:`~isaaclab_ovphysx.TensorBinding` spanning
        all bodies in the collection, with shape
        ``(num_instances, num_bodies, D)``.
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
        binding = self._get_binding(TT.LINK_WRENCH)
        if binding is not None:
            binding.write(self._wrench_buf)
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
        # Mark the pose buffer as fresh and invalidate dependent timestamps so
        # the next read either uses the kernel-written value (within the same
        # sim step) or re-fetches from OVPhysX (after the next update() call).
        # Without the freshness mark, a stale timestamp causes the property to
        # re-read the post-step pose from OVPhysX before update() is called,
        # returning a physics-evolved position rather than the written one.
        self.data._body_link_pose_w.timestamp = self.data._sim_timestamp
        self.data._body_com_pose_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        # Push updated link poses to simulation via single fused binding.
        binding = self._get_binding(TT.LINK_POSE)
        view = self._make_float32_view(self.data._body_link_pose_w.data, binding)
        binding.write(view, indices=env_ids)

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
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_BODY_INDICES
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
        # Push updated link poses to simulation via single fused binding.
        binding = self._get_binding(TT.LINK_POSE)
        view = self._make_float32_view(self.data._body_link_pose_w.data, binding)
        binding.write(view, indices=env_ids)

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
        # Push updated link poses to simulation via single fused binding
        # (OVPhysX only exposes link frame).
        binding = self._get_binding(TT.LINK_POSE)
        view = self._make_float32_view(self.data._body_link_pose_w.data, binding)
        binding.write(view, indices=env_ids)

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
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_BODY_INDICES
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
        # Push updated link poses to simulation via single fused binding
        # (OVPhysX only exposes link frame).
        binding = self._get_binding(TT.LINK_POSE)
        view = self._make_float32_view(self.data._body_link_pose_w.data, binding)
        binding.write(view, indices=env_ids)

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
        # Push updated COM velocities to simulation via single fused binding.
        binding = self._get_binding(TT.LINK_VELOCITY)
        view = self._make_float32_view(self.data._body_com_vel_w.data, binding)
        binding.write(view, indices=env_ids)

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
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_BODY_INDICES
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
        # Push updated COM velocities to simulation via single fused binding.
        binding = self._get_binding(TT.LINK_VELOCITY)
        view = self._make_float32_view(self.data._body_com_vel_w.data, binding)
        binding.write(view, indices=env_ids)

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
        # Mark the velocity buffer as fresh so the next read returns the
        # kernel-written value rather than the post-step OVPhysX state.
        # See write_body_link_pose_to_sim_index for the full rationale.
        self.data._body_com_vel_w.timestamp = self.data._sim_timestamp
        self.data._body_link_vel_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        # Push updated COM velocities to simulation via single fused binding.
        binding = self._get_binding(TT.LINK_VELOCITY)
        view = self._make_float32_view(self.data._body_com_vel_w.data, binding)
        binding.write(view, indices=env_ids)

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
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_BODY_INDICES
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
        # Push updated COM velocities to simulation via single fused binding.
        binding = self._get_binding(TT.LINK_VELOCITY)
        view = self._make_float32_view(self.data._body_com_vel_w.data, binding)
        binding.write(view, indices=env_ids)

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
        ``BODY_MASS`` is a CPU-only OVPhysX binding.

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
            outputs=[self.data._body_mass.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_mass, self.data._body_mass.data)
        binding = self._get_binding(TT.BODY_MASS)
        binding.write(self.data._cpu_body_mass, indices=cpu_env_ids)

    def set_masses_mask(
        self,
        *,
        masses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body masses over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_MASS`` is a CPU-only OVPhysX binding.

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
            env_ids = self._ALL_ENV_INDICES
        self.assert_shape_and_dtype(masses, (self._num_instances, self._num_bodies), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[masses, self._resolve_env_mask(env_mask), self._resolve_body_mask(body_mask)],
            outputs=[self.data._body_mass.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_mass, self.data._body_mass.data)
        binding = self._get_binding(TT.BODY_MASS)
        binding.write(self.data._cpu_body_mass, indices=cpu_env_ids)

    def set_coms_index(
        self,
        *,
        coms: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body center-of-mass poses over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_COM_POSE`` is a CPU-only OVPhysX binding.

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
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_coms, self.data._body_com_pose_b.data)
        binding = self._get_binding(TT.BODY_COM_POSE)
        binding.write(self.data._cpu_body_coms, indices=cpu_env_ids)

    def set_coms_mask(
        self,
        *,
        coms: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body center-of-mass poses over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_COM_POSE`` is a CPU-only OVPhysX binding.

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
            env_ids = self._ALL_ENV_INDICES
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
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_coms, self.data._body_com_pose_b.data)
        binding = self._get_binding(TT.BODY_COM_POSE)
        binding.write(self.data._cpu_body_coms, indices=cpu_env_ids)

    def set_inertias_index(
        self,
        *,
        inertias: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body inertia tensors over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_INERTIA`` is a CPU-only OVPhysX binding.

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
            outputs=[self.data._body_inertia.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_inertia, self.data._body_inertia.data)
        binding = self._get_binding(TT.BODY_INERTIA)
        binding.write(self.data._cpu_body_inertia, indices=cpu_env_ids)

    def set_inertias_mask(
        self,
        *,
        inertias: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        """Set body inertia tensors over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_INERTIA`` is a CPU-only OVPhysX binding.

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
            env_ids = self._ALL_ENV_INDICES
        self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[inertias, self._resolve_env_mask(env_mask), self._resolve_body_mask(body_mask)],
            outputs=[self.data._body_inertia.data],
            device=self._device,
        )
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self.data._cpu_body_inertia, self.data._body_inertia.data)
        binding = self._get_binding(TT.BODY_INERTIA)
        binding.write(self.data._cpu_body_inertia, indices=cpu_env_ids)

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
        a single fused :class:`TensorBinding` per tensor type using the new
        ``prim_paths=[...]`` API introduced in ovphysx 0.4.3.

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
        self._prim_paths: list[str] = []
        self._body_names_list: list[str] = []

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

            self._prim_paths.append(pattern)
            self._body_names_list.append(name)

        # Step 3: Total number of distinct body types.
        self._num_bodies = len(self._prim_paths)

        # Step 4: For each supported tensor type, create one RIGID_BODY_* binding per
        # body type (pattern), then wrap the B per-body bindings in a
        # _FusedRigidBodyBinding adapter stored under the ARTICULATION_LINK_* key that
        # RigidObjectCollectionData uses.  This preserves the data-class interface while
        # querying the correct OVPhysX tensor type for non-articulated rigid bodies.
        #
        # Mapping: data-class key → (per-body RIGID_BODY tensor type, data class key)
        _TT_MAP = (
            (TT.LINK_POSE, TT.RIGID_BODY_POSE),
            (TT.LINK_VELOCITY, TT.RIGID_BODY_VELOCITY),
            (TT.LINK_WRENCH, TT.RIGID_BODY_WRENCH),
            (TT.BODY_MASS, TT.RIGID_BODY_MASS),
            (TT.BODY_COM_POSE, TT.RIGID_BODY_COM_POSE),
            (TT.BODY_INERTIA, TT.RIGID_BODY_INERTIA),
        )
        for fused_key, rb_tt in _TT_MAP:
            per_body = []
            for pattern in self._prim_paths:
                try:
                    b = self._ovphysx.create_tensor_binding(pattern=pattern, tensor_type=rb_tt)
                    per_body.append(b)
                except Exception as e:
                    raise RuntimeError(
                        f"OVPhysX could not create RIGID_BODY binding {rb_tt!r} for"
                        f" pattern {pattern!r}."
                        f" Check that the prim path matches at least one"
                        f" UsdPhysics.RigidBodyAPI prim."
                    ) from e
            # Determine N from the first binding's count.
            N = per_body[0].count
            B = len(per_body)
            self._bindings[fused_key] = _FusedRigidBodyBinding(per_body, N, B, self._device)

        # Step 5: Read num_instances from the LINK_POSE fused binding.
        # All fused bindings share the same N (verified implicitly by construction).
        self._num_instances = self._bindings[TT.LINK_POSE].count

        # Step 6: Create the data container.
        self._data = RigidObjectCollectionData(
            root_view=self._bindings,
            num_bodies=self._num_bodies,
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

        self._ALL_ENV_INDICES = wp.array(np.arange(N), dtype=wp.int32, device=self._device)
        self._ALL_BODY_INDICES = wp.array(np.arange(B), dtype=wp.int32, device=self._device)

        # CPU copy of all-env indices used when calling CPU-only binding.write().
        self._cpu_all_env_ids = wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True)
        wp.copy(self._cpu_all_env_ids, self._ALL_ENV_INDICES)

        # All-true boolean masks used as defaults in mask-based kernel calls.
        self._ALL_TRUE_ENV_MASK = wp.array(np.ones(N, dtype=bool), dtype=wp.bool, device=self._device)
        self._ALL_TRUE_BODY_MASK = wp.array(np.ones(B, dtype=bool), dtype=wp.bool, device=self._device)

        # External wrench buffer: direct (N, B, 9) contiguous allocation.
        # The fused LINK_WRENCH binding writes from a single (N, B, 9) buffer.
        self._wrench_buf = wp.zeros((N, B, 9), dtype=wp.float32, device=self._device)

        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # Set body names into the data container (mirrors PhysX collection).
        self._data.body_names = self._body_names_list

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

    def _get_binding(self, tensor_type: int):
        """Return the cached :class:`_FusedRigidBodyBinding` for *tensor_type*.

        All bindings are eagerly created in :meth:`_initialize_impl` and stored
        under the ``TT.LINK_*`` / ``TT.BODY_*`` keys that
        :class:`RigidObjectCollectionData` uses.  This method simply returns the
        cached entry.

        Args:
            tensor_type: The TensorType constant identifying which simulation
                buffer to bind (e.g. :attr:`~isaaclab_ovphysx.tensor_types.LINK_POSE`).

        Returns:
            The cached :class:`_FusedRigidBodyBinding`, or ``None`` if not found.
        """
        return self._bindings.get(tensor_type)

    def _make_float32_view(self, wp_array: wp.array, binding) -> wp.array:
        """Return a float32 view of *wp_array* matching the binding's flat shape.

        For structured-dtype buffers (e.g. ``wp.transformf``, ``wp.spatial_vectorf``),
        reinterprets the GPU memory as ``wp.float32`` with shape ``binding.shape``.
        For plain ``wp.float32`` buffers, returns the array as-is.

        Args:
            wp_array: Source warp array (may be structured dtype).
            binding: TensorBinding whose ``.shape`` gives the target float32 shape.

        Returns:
            A ``wp.float32`` view of the same memory.
        """
        if wp_array.dtype == wp.float32:
            return wp_array
        return wp.array(
            ptr=wp_array.ptr,
            shape=binding.shape,
            dtype=wp.float32,
            device=str(wp_array.device),
            copy=False,
        )

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
            return self._ALL_ENV_INDICES
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
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self._device)
        return body_ids

    def _env_body_ids_to_view_ids(
        self, env_ids: torch.Tensor | wp.array, body_ids: torch.Tensor | wp.array, device: str = "cuda:0"
    ) -> wp.array:
        """Convert environment and body indices to flat view indices (body-major ordering).

        Computes ``view_id = body_id * num_instances + env_id`` for each
        (env_id, body_id) pair.  The output array is laid out column-major over
        the (env, body) grid, matching the PhysX ``root_view`` ordering.

        Args:
            env_ids: Environment indices.
            body_ids: Body indices.
            device: Target device for the returned array.

        Returns:
            A :class:`wp.array` of shape ``(len(env_ids) * len(body_ids),)`` with
            flat view indices on *device*.
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(body_ids, torch.Tensor):
            body_ids = wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)
        if str(env_ids.device) != device:
            env_ids = wp.clone(env_ids, device=device)
        if str(body_ids.device) != device:
            body_ids = wp.clone(body_ids, device=device)
        num_query_envs = env_ids.shape[0]
        view_ids = wp.zeros(num_query_envs * body_ids.shape[0], dtype=wp.int32, device=device)
        wp.launch(
            resolve_view_ids,
            dim=(num_query_envs, body_ids.shape[0]),
            inputs=[env_ids, body_ids, num_query_envs, self.num_instances],
            outputs=[view_ids],
            device=device,
        )
        return view_ids

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

    def _get_cpu_env_ids(self, env_ids: wp.array) -> wp.array:
        """Return CPU int32 env indices for CPU-only binding writes.

        Uses the pre-allocated pinned ``_cpu_all_env_ids`` fast path when
        *env_ids* covers all instances, otherwise clones to CPU.

        Args:
            env_ids: A warp int32 array of environment indices on any device.

        Returns:
            A warp int32 array guaranteed to live on ``"cpu"``.
        """
        if env_ids.ptr == self._ALL_ENV_INDICES.ptr:
            return self._cpu_all_env_ids
        return wp.clone(env_ids, device="cpu")
