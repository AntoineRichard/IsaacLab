# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed rigid object collection data container."""

from __future__ import annotations

from typing import Any

import warp as wp

from isaaclab.assets.rigid_object_collection import BaseRigidObjectCollectionData


class RigidObjectCollectionData(BaseRigidObjectCollectionData):
    """Data container for a rigid object collection.

    This class contains the data for a rigid object collection in the simulation.
    All properties are lazy and raise :exc:`NotImplementedError` until implemented
    in later phases.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object collection data."""

    def __init__(
        self,
        root_view: dict[int, list[Any]],
        num_objects: int,
        device: str,
    ):
        """Initialize the rigid object collection data.

        Args:
            root_view: Per-body TensorBinding dict, keyed by TensorType constant.
                Each value is a list of length ``num_objects`` containing one
                TensorBinding per body in the collection.
            num_objects: The number of object types managed by the collection.
            device: The device used for processing (e.g. ``"cuda:0"`` or ``"cpu"``).
        """
        super().__init__(root_view, num_objects, device)
        # Store the bindings dict for Phase 2 to consume.
        self._root_view = root_view
        self.num_bodies = num_objects

        # Placeholders populated by RigidObjectCollection._process_cfg().
        # Phase 2 will replace these with ProxyArray wrappers backed by proper
        # GPU buffers; for now they hold the raw wp.array values so that
        # test fixtures that read default_body_pose / default_body_vel
        # after _initialize_impl() work correctly.
        self._default_body_pose: wp.array | None = None
        self._default_body_vel: wp.array | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:  # type: ignore[override]
        raise NotImplementedError("phase 2")

    # ------------------------------------------------------------------
    # Default state properties
    # ------------------------------------------------------------------

    @property
    def default_body_pose(self) -> wp.array | None:
        """Default body pose ``[pos, quat]`` in local environment frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        Set by :meth:`~RigidObjectCollection._process_cfg` during initialization;
        Phase 2 will wrap this in a :class:`~isaaclab.utils.warp.ProxyArray`.
        """
        return self._default_body_pose

    @default_body_pose.setter
    def default_body_pose(self, value: wp.array) -> None:
        self._default_body_pose = value

    @property
    def default_body_vel(self) -> wp.array | None:
        """Default body velocity ``[lin_vel, ang_vel]`` in local environment frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        Set by :meth:`~RigidObjectCollection._process_cfg` during initialization;
        Phase 2 will wrap this in a :class:`~isaaclab.utils.warp.ProxyArray`.
        """
        return self._default_body_vel

    @default_body_vel.setter
    def default_body_vel(self, value: wp.array) -> None:
        self._default_body_vel = value

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
