# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed rigid object collection data container."""

from __future__ import annotations

from isaaclab.assets.rigid_object_collection import BaseRigidObjectCollectionData


class RigidObjectCollectionData(BaseRigidObjectCollectionData):
    """Data container for a rigid object collection.

    This class contains the data for a rigid object collection in the simulation.
    All properties are lazy and raise :exc:`NotImplementedError` until implemented
    in later phases.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object collection data."""

    def __init__(self, root_view, num_objects: int, device: str):
        """Initialize the rigid object collection data.

        Args:
            root_view: The OVPhysX tensor bindings dict or view for the collection.
            num_objects: The number of object types managed by the collection.
            device: The device used for processing.
        """
        super().__init__(root_view, num_objects, device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 2")

    # ------------------------------------------------------------------
    # Default state properties
    # ------------------------------------------------------------------

    @property
    def default_body_pose(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def default_body_vel(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def default_body_state(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    # ------------------------------------------------------------------
    # Body state properties
    # ------------------------------------------------------------------

    @property
    def body_mass(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_inertia(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_link_pose_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_link_vel_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_pose_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_vel_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_acc_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_pose_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_state_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_link_state_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_state_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    # ------------------------------------------------------------------
    # Sliced body properties (position, orientation, velocity)
    # ------------------------------------------------------------------

    @property
    def body_link_pos_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_link_quat_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_link_lin_vel_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_link_ang_vel_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_link_lin_vel_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_link_ang_vel_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_pos_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_quat_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_lin_vel_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_ang_vel_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_lin_vel_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_ang_vel_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_lin_acc_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_ang_acc_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_pos_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_quat_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def projected_gravity_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2: derived")

    @property
    def heading_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2: derived")
