# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed rigid object collection asset."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from isaaclab.assets.rigid_object_collection import (
    BaseRigidObjectCollection,
    RigidObjectCollectionCfg,
)


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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self):  # type: ignore[override]
        raise NotImplementedError("phase 1.6")

    @property
    def num_instances(self) -> int:  # type: ignore[override]
        raise NotImplementedError("phase 1.6")

    @property
    def num_bodies(self) -> int:  # type: ignore[override]
        raise NotImplementedError("phase 1.6")

    @property
    def body_names(self) -> list[str]:  # type: ignore[override]
        raise NotImplementedError("phase 1.6")

    @property
    def root_view(self):  # type: ignore[override]
        raise NotImplementedError("phase 1.6")

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

    def _initialize_impl(self) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 1.6")

    def _create_buffers(self) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 1.6")

    def _process_cfg(self) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 1.6")

    def _invalidate_initialize_callback(self, event) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 1.6")
