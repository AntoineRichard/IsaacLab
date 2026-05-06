# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed rigid object collection asset."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object_collection import (
    BaseRigidObjectCollection,
    RigidObjectCollectionCfg,
)

from isaaclab_ovphysx import tensor_types as TT
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
    def instantaneous_wrench_composer(self):  # type: ignore[override]
        raise NotImplementedError("phase 4")

    @property
    def permanent_wrench_composer(self):  # type: ignore[override]
        raise NotImplementedError("phase 4")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reset(
        self,
        env_ids: Sequence[int] | wp.array | None = None,
        object_ids: slice | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 4")

    def write_data_to_sim(self) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 4")

    def update(self, dt: float) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 2")

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False):  # type: ignore[override]
        raise NotImplementedError("phase 3")

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
        raise NotImplementedError("phase 3")

    def write_body_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_link_pose_to_sim_index(
        self,
        *,
        body_poses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_link_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_com_pose_to_sim_index(
        self,
        *,
        body_poses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_com_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

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
        raise NotImplementedError("phase 3")

    def write_body_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_link_velocity_to_sim_index(
        self,
        *,
        body_velocities: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_link_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_com_velocity_to_sim_index(
        self,
        *,
        body_velocities: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_com_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

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
        raise NotImplementedError("phase 3")

    def set_masses_mask(
        self,
        *,
        masses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def set_coms_index(
        self,
        *,
        coms: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def set_coms_mask(
        self,
        *,
        coms: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def set_inertias_index(
        self,
        *,
        inertias: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def set_inertias_mask(
        self,
        *,
        inertias: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    # ------------------------------------------------------------------
    # Deprecated state writers
    # ------------------------------------------------------------------

    def write_body_state_to_sim(
        self,
        body_states: wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
        body_ids: slice | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_link_state_to_sim(
        self,
        body_states: wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
        body_ids: slice | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

    def write_body_com_state_to_sim(
        self,
        body_states: wp.array,
        env_ids: Sequence[int] | wp.array | None = None,
        body_ids: slice | None = None,
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 3")

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

        # TODO(phase 2): self.update(0.0)

    def _create_buffers(self) -> None:
        """Pre-allocate asset-side index arrays."""
        self._ALL_INDICES_ENV = wp.array(np.arange(self._num_instances), dtype=wp.int32, device=self._device)
        self._ALL_INDICES_BODY = wp.array(np.arange(self._num_bodies), dtype=wp.int32, device=self._device)

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
