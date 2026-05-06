# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed rigid object collection data container."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import warp as wp

from isaaclab.assets.rigid_object_collection import BaseRigidObjectCollectionData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.math import normalize
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.physics import OvPhysxManager as SimulationManager


class RigidObjectCollectionData(BaseRigidObjectCollectionData):
    """Data container for a rigid object collection.

    This class contains the data for a rigid object collection in the simulation.
    The data includes the state of all the bodies in the collection. The data is
    stored in the simulation world frame unless otherwise specified.

    The data is in the order ``(num_instances, num_bodies, data_size)``, where
    ``data_size`` is the size of the data for each body.

    For a rigid body, there are two frames of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically
      corresponds to the Xform prim with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the
      rigid body.

    Depending on the settings of the simulation, the actor frame and the center of
    mass frame may be the same. This needs to be taken into account when interpreting
    the data.

    The data is lazily updated, meaning that the data is only updated when it is
    accessed. The data is updated when the timestamp of the buffer is older than the
    current simulation timestamp.

    .. note::
        **Pull-to-refresh model.** Properties pull fresh data from the OVPhysX
        tensor API on first access per timestamp and cache the result.

    .. note::
        **Per-body bindings.** Unlike PhysX, OVPhysX does not expose a fused
        multi-prim view. Each body in the collection has its own
        :class:`~isaaclab_ovphysx.TensorBinding` keyed by tensor type. To produce
        ``(num_instances, num_bodies, D)`` arrays, properties loop over bodies and
        read each per-body binding into the appropriate slice of a pre-allocated
        contiguous buffer.

    .. note::
        **Buffer layout.** Internally, state buffers are allocated as flat
        ``(num_bodies * num_instances,)`` arrays in body-major order
        (body ``b``, instance ``i`` → index ``b * num_instances + i``).
        A strided :class:`warp.array` view with shape
        ``(num_instances, num_bodies)`` and strides
        ``(element_size, num_instances * element_size)`` is exposed to kernels,
        matching the transposition performed by the PhysX collection backend.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object collection data."""

    def __init__(
        self,
        root_view: dict[int, list[Any]],
        num_objects: int,
        device: str,
        check_shapes: bool = True,
    ):
        """Initialize the rigid object collection data.

        Args:
            root_view: Per-body TensorBinding dict, keyed by TensorType constant.
                Each value is a list of length ``num_objects`` containing one
                TensorBinding per body in the collection.
            num_objects: The number of object types managed by the collection.
            device: The device used for processing (e.g. ``"cuda:0"`` or ``"cpu"``).
            check_shapes: Whether to enforce internal shape/dtype invariants on
                lazy reads. Defaults to ``True``; production callers may thread
                this from
                :attr:`~isaaclab.assets.AssetBaseCfg.disable_shape_checks`.
        """
        super().__init__(root_view, num_objects, device)
        # Store the bindings dict (equivalent to the view in PhysX).
        self._root_view = root_view
        self.num_bodies = num_objects
        self._check_shapes = check_shapes
        # Set initial time stamp.
        self._sim_timestamp = 0.0
        self._is_primed = False

        # Read num_instances from the POSE binding of body 0.
        self.num_instances = self._root_view[TT.RIGID_BODY_POSE][0].count

        if SimulationManager._sim is not None and hasattr(SimulationManager._sim, "cfg"):
            gravity = SimulationManager._sim.cfg.gravity
        else:
            gravity = (0.0, 0.0, -9.81)

        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        if torch.linalg.norm(gravity_dir) > 0.0:
            gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir = gravity_dir.repeat(self.num_instances, self.num_bodies, 1)
        forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.num_instances, self.num_bodies, 1)

        # Initialize constants.
        self.GRAVITY_VEC_W = ProxyArray(wp.from_torch(gravity_dir, dtype=wp.vec3f))
        self.FORWARD_VEC_B = ProxyArray(wp.from_torch(forward_vec, dtype=wp.vec3f))

        # Placeholders populated by RigidObjectCollection._process_cfg().
        self._default_body_pose: wp.array | None = None
        self._default_body_vel: wp.array | None = None

        # Pinned-host staging buffers for CPU-only bindings on a non-CPU sim
        # (lazily allocated, keyed by (tensor_type, body_idx)).
        self._cpu_staging_buffers: dict[tuple[int, int], wp.array] = {}

        self._create_buffers()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_primed(self) -> bool:
        """Whether the rigid object collection data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        """Set whether the rigid object collection data is fully instantiated and ready to use.

        .. note::
            Once this quantity is set to True, it cannot be changed.

        Args:
            value: The primed state.

        Raises:
            ValueError: If the rigid object collection data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object collection data is already primed.")
        self._is_primed = value

    def update(self, dt: float) -> None:
        """Updates the data for the rigid object collection.

        Args:
            dt: The time step for the update [s]. This must be a positive value.
        """
        self._sim_timestamp += dt

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    # ------------------------------------------------------------------
    # Default state properties
    # ------------------------------------------------------------------

    @property
    def default_body_pose(self) -> wp.array | None:
        """Default body pose ``[pos, quat]`` in local environment frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        In torch this resolves to (num_instances, num_bodies, 7).
        Set by :meth:`~RigidObjectCollection._process_cfg` during initialization.
        """
        return self._default_body_pose

    @default_body_pose.setter
    def default_body_pose(self, value: wp.array) -> None:
        self._default_body_pose = value

    @property
    def default_body_vel(self) -> wp.array | None:
        """Default body velocity ``[lin_vel, ang_vel]`` in local environment frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        In torch this resolves to (num_instances, num_bodies, 6).
        Set by :meth:`~RigidObjectCollection._process_cfg` during initialization.
        """
        return self._default_body_vel

    @default_body_vel.setter
    def default_body_vel(self, value: wp.array) -> None:
        self._default_body_vel = value

    @property
    def default_body_state(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    # ------------------------------------------------------------------
    # Body state properties — raw reads
    # ------------------------------------------------------------------

    @property
    def body_link_pose_w(self) -> ProxyArray:
        """Body link pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        In torch this resolves to (num_instances, num_bodies, 7).
        This quantity is the pose of the actor frame of the rigid body relative to
        the world. The orientation is provided in (x, y, z, w) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            for b in range(self.num_bodies):
                self._read_binding_into(TT.RIGID_BODY_POSE, b, self._body_link_pose_w_flat[b])
            self._body_link_pose_w.timestamp = self._sim_timestamp
            # Invalidate sliced sub-component proxies so they are rebuilt from the
            # updated buffer on next access.
            self._body_link_pos_w_ta = None
            self._body_link_quat_w_ta = None
        if self._body_link_pose_w_ta is None:
            self._body_link_pose_w_ta = ProxyArray(self._body_link_pose_w.data)
        return self._body_link_pose_w_ta

    @property
    def body_link_vel_w(self) -> ProxyArray:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        In torch this resolves to (num_instances, num_bodies, 6).
        This quantity contains the linear and angular velocities of the actor frame
        of the rigid body relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_link_vel_from_body_com_vel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[
                    self.body_com_vel_w,
                    self.body_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[self._body_link_vel_w.data],
                device=self.device,
            )
            self._body_link_vel_w.timestamp = self._sim_timestamp
            self._body_link_lin_vel_w_ta = None
            self._body_link_ang_vel_w_ta = None
        if self._body_link_vel_w_ta is None:
            self._body_link_vel_w_ta = ProxyArray(self._body_link_vel_w.data)
        return self._body_link_vel_w_ta

    @property
    def body_com_pose_w(self) -> ProxyArray:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        In torch this resolves to (num_instances, num_bodies, 7).
        This quantity is the pose of the center of mass frame of the rigid body
        relative to the world. The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_com_pose_from_body_link_pose,
                dim=(self.num_instances, self.num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[self._body_com_pose_w.data],
                device=self.device,
            )
            self._body_com_pose_w.timestamp = self._sim_timestamp
            self._body_com_pos_w_ta = None
            self._body_com_quat_w_ta = None
        if self._body_com_pose_w_ta is None:
            self._body_com_pose_w_ta = ProxyArray(self._body_com_pose_w.data)
        return self._body_com_pose_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        In torch this resolves to (num_instances, num_bodies, 6).
        This quantity contains the linear and angular velocities of the rigid body's
        center of mass frame relative to the world.
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            for b in range(self.num_bodies):
                self._read_binding_into(TT.RIGID_BODY_VELOCITY, b, self._body_com_vel_w_flat[b])
            self._body_com_vel_w.timestamp = self._sim_timestamp
            self._body_com_lin_vel_w_ta = None
            self._body_com_ang_vel_w_ta = None
        if self._body_com_vel_w_ta is None:
            self._body_com_vel_w_ta = ProxyArray(self._body_com_vel_w.data)
        return self._body_com_vel_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_pose_b(self) -> ProxyArray:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body link frames [m, -].

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        In torch this resolves to (num_instances, num_bodies, 7).
        This quantity is the pose of the center of mass frame of the rigid body
        relative to the body's link frame. The orientation is provided in
        (x, y, z, w) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            for b in range(self.num_bodies):
                self._read_binding_into(TT.RIGID_BODY_COM_POSE, b, self._body_com_pose_b_flat[b])
            self._body_com_pose_b.timestamp = self._sim_timestamp
            self._body_com_pos_b_ta = None
            self._body_com_quat_b_ta = None
        if self._body_com_pose_b_ta is None:
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b.data)
        return self._body_com_pose_b_ta

    @property
    def body_mass(self) -> ProxyArray:
        """Mass of all bodies [kg].

        Shape is (num_instances, num_bodies), dtype = ``wp.float32``.
        In torch this resolves to (num_instances, num_bodies).
        """
        if self._body_mass_ta is None:
            self._body_mass_ta = ProxyArray(self._body_mass)
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Inertia tensor of all bodies, expressed at the center of mass [kg·m²].

        Shape is (num_instances, num_bodies, 9), dtype = ``wp.float32``.
        The 9 components are the row-major flatten of the 3×3 inertia matrix
        ``(Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz)``.
        In torch this resolves to (num_instances, num_bodies, 9).
        """
        if self._body_inertia_ta is None:
            self._body_inertia_ta = ProxyArray(self._body_inertia)
        return self._body_inertia_ta

    # ------------------------------------------------------------------
    # Deprecated state-concat properties
    # ------------------------------------------------------------------

    @property
    def body_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_com_vel_w`."""
        warnings.warn(
            "The `body_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_link_pose_w, self.body_com_vel_w],
                outputs=[self._body_state_w.data],
                device=self.device,
            )
            self._body_state_w.timestamp = self._sim_timestamp
        if self._body_state_w_ta is None:
            self._body_state_w_ta = ProxyArray(self._body_state_w.data)
        return self._body_state_w_ta

    @property
    def body_link_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_link_vel_w`."""
        warnings.warn(
            "The `body_link_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_link_pose_w, self.body_link_vel_w],
                outputs=[self._body_link_state_w.data],
                device=self.device,
            )
            self._body_link_state_w.timestamp = self._sim_timestamp
        if self._body_link_state_w_ta is None:
            self._body_link_state_w_ta = ProxyArray(self._body_link_state_w.data)
        return self._body_link_state_w_ta

    @property
    def body_com_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_com_pose_w` and :attr:`body_com_vel_w`."""
        warnings.warn(
            "The `body_com_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_com_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_com_pose_w, self.body_com_vel_w],
                outputs=[self._body_com_state_w.data],
                device=self.device,
            )
            self._body_com_state_w.timestamp = self._sim_timestamp
        if self._body_com_state_w_ta is None:
            self._body_com_state_w_ta = ProxyArray(self._body_com_state_w.data)
        return self._body_com_state_w_ta

    # ------------------------------------------------------------------
    # Sliced body properties (position, orientation, velocity)
    # ------------------------------------------------------------------

    @property
    def body_link_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame [m].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        This quantity is the position of the rigid bodies' actor frame relative to
        the world.
        """
        parent = self.body_link_pose_w
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.quatf``.
        In torch this resolves to (num_instances, num_bodies, 4).
        This quantity is the orientation of the rigid bodies' actor frame relative
        to the world.
        """
        parent = self.body_link_pose_w
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_link_quat_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame [m/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        This quantity is the linear velocity of the rigid bodies' actor frame
        relative to the world.
        """
        parent = self.body_link_vel_w
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame [rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        This quantity is the angular velocity of the rigid bodies' actor frame
        relative to the world.
        """
        parent = self.body_link_vel_w
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_link_ang_vel_w_ta

    @property
    def body_link_lin_vel_b(self) -> ProxyArray:
        """Linear velocity of all bodies in their respective body (actor) frames [m/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        if self._body_link_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_link_lin_vel_w, self.body_link_quat_w],
                outputs=[self._body_link_lin_vel_b.data],
                device=self.device,
            )
            self._body_link_lin_vel_b.timestamp = self._sim_timestamp
        if self._body_link_lin_vel_b_ta is None:
            self._body_link_lin_vel_b_ta = ProxyArray(self._body_link_lin_vel_b.data)
        return self._body_link_lin_vel_b_ta

    @property
    def body_link_ang_vel_b(self) -> ProxyArray:
        """Angular velocity of all bodies in their respective body (actor) frames [rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        if self._body_link_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_link_ang_vel_w, self.body_link_quat_w],
                outputs=[self._body_link_ang_vel_b.data],
                device=self.device,
            )
            self._body_link_ang_vel_b.timestamp = self._sim_timestamp
        if self._body_link_ang_vel_b_ta is None:
            self._body_link_ang_vel_b_ta = ProxyArray(self._body_link_ang_vel_b.data)
        return self._body_link_ang_vel_b_ta

    @property
    def body_com_pos_w(self) -> ProxyArray:
        """Positions of all bodies' center of mass in simulation world frame [m].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_pose_w
        if self._body_com_pos_w_ta is None:
            self._body_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_w_ta

    @property
    def body_com_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.quatf``.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_com_pose_w
        if self._body_com_quat_w_ta is None:
            self._body_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies' center of mass in simulation world frame [m/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_vel_w
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies' center of mass in simulation world frame [rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_vel_w
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_vel_w_ta

    @property
    def body_com_lin_vel_b(self) -> ProxyArray:
        """Linear velocity of all bodies' center of mass in their respective body (actor) frames [m/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        if self._body_com_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_com_lin_vel_w, self.body_link_quat_w],
                outputs=[self._body_com_lin_vel_b.data],
                device=self.device,
            )
            self._body_com_lin_vel_b.timestamp = self._sim_timestamp
        if self._body_com_lin_vel_b_ta is None:
            self._body_com_lin_vel_b_ta = ProxyArray(self._body_com_lin_vel_b.data)
        return self._body_com_lin_vel_b_ta

    @property
    def body_com_ang_vel_b(self) -> ProxyArray:
        """Angular velocity of all bodies' center of mass in their respective body (actor) frames [rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        if self._body_com_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_com_ang_vel_w, self.body_link_quat_w],
                outputs=[self._body_com_ang_vel_b.data],
                device=self.device,
            )
            self._body_com_ang_vel_b.timestamp = self._sim_timestamp
        if self._body_com_ang_vel_b_ta is None:
            self._body_com_ang_vel_b_ta = ProxyArray(self._body_com_ang_vel_b.data)
        return self._body_com_ang_vel_b_ta

    @property
    def body_com_lin_acc_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_ang_acc_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2")

    @property
    def body_com_pos_b(self) -> ProxyArray:
        """Center of mass position of all of the bodies in their respective link frames [m].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_pose_b
        if self._body_com_pos_b_ta is None:
            self._body_com_pos_b_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_b_ta

    @property
    def body_com_quat_b(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies
        in their respective link frames.

        Shape is (num_instances, num_bodies), dtype = ``wp.quatf``.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_com_pose_b
        if self._body_com_quat_b_ta is None:
            self._body_com_quat_b_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_b_ta

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def projected_gravity_b(self):  # type: ignore[override]
        raise NotImplementedError("phase 2: derived")

    @property
    def heading_w(self):  # type: ignore[override]
        raise NotImplementedError("phase 2: derived")

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _create_buffers(self) -> None:
        """Eagerly allocate every per-body TimestampedBuffer and the slots for
        cached :class:`ProxyArray` wrappers.

        Buffers for ``(num_instances, num_bodies)`` structured data use a
        body-major flat allocation (shape ``(num_bodies * num_instances,)``
        dtype ``T``) plus a strided 2D view (shape ``(num_instances, num_bodies)``
        strides ``(T_size, num_instances * T_size)``). This allows contiguous
        per-body reads from OVPhysX bindings while still giving kernels a proper
        2D indexed array.
        """
        super()._create_buffers()

        N = self.num_instances
        B = self.num_bodies

        # -- link frame w.r.t. world frame
        self._body_link_pose_w = TimestampedBuffer((N, B), self.device, wp.transformf)
        self._body_link_vel_w = TimestampedBuffer((N, B), self.device, wp.spatial_vectorf)
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer((N, B), self.device, wp.transformf)
        # -- com frame w.r.t. world frame
        self._body_com_pose_w = TimestampedBuffer((N, B), self.device, wp.transformf)
        self._body_com_vel_w = TimestampedBuffer((N, B), self.device, wp.spatial_vectorf)
        # -- combined state (cached, used by deprecated concat properties)
        self._body_state_w = TimestampedBuffer((N, B), self.device, shared_kernels.vec13f)
        self._body_link_state_w = TimestampedBuffer((N, B), self.device, shared_kernels.vec13f)
        self._body_com_state_w = TimestampedBuffer((N, B), self.device, shared_kernels.vec13f)
        # -- derived properties (in-body-frame velocities)
        self._body_link_lin_vel_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        self._body_link_ang_vel_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        self._body_com_lin_vel_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        self._body_com_ang_vel_b = TimestampedBuffer((N, B), self.device, wp.vec3f)

        # -- Per-body contiguous read buffers.
        #
        # OVPhysX ``binding.read()`` requires a flat contiguous float32 target;
        # it cannot write directly into a column of the 2D buffer above.
        # We allocate one contiguous ``(N,)`` buffer per body for each tensor type
        # that needs a per-body loop read, then expose strided 2D *views* of these
        # flat buffers to the rest of the class.
        #
        # Layout: ``_body_link_pose_w_flat[b]`` is a contiguous ``(N,)``
        # ``wp.transformf`` array for body ``b``.  The 2D strided view is built
        # on top of a flat ``(B * N,)`` base array allocated just below.
        #
        # The strided view uses:
        #   shape   = (N, B)
        #   strides = (T_size, N * T_size)    (body-major → instance-major transpose)
        #
        # Reading body ``b`` into ``_body_link_pose_w_flat[b]`` fills bytes
        # ``[b*N*T_size, (b+1)*N*T_size)`` of the flat base, which is exactly the
        # column ``[:, b]`` of the strided view.

        # transformf: 7 floats × 4 bytes = 28 bytes
        tf_size = wp.types.type_size_in_bytes(wp.transformf)  # 28
        # spatial_vectorf: 6 floats × 4 bytes = 24 bytes
        sv_size = wp.types.type_size_in_bytes(wp.spatial_vectorf)  # 24

        # Flat base arrays (body-major: body b occupies [b*N : (b+1)*N]).
        self._body_link_pose_w_base = wp.zeros(B * N, dtype=wp.transformf, device=self.device)
        self._body_com_vel_w_base = wp.zeros(B * N, dtype=wp.spatial_vectorf, device=self.device)
        self._body_com_pose_b_base = wp.zeros(B * N, dtype=wp.transformf, device=self.device)

        # Per-body contiguous sub-views (shape (N,), used by _read_binding_into).
        self._body_link_pose_w_flat = [
            wp.array(
                ptr=self._body_link_pose_w_base.ptr + b * N * tf_size,
                shape=(N,),
                dtype=wp.transformf,
                device=self.device,
                copy=False,
            )
            for b in range(B)
        ]
        self._body_com_vel_w_flat = [
            wp.array(
                ptr=self._body_com_vel_w_base.ptr + b * N * sv_size,
                shape=(N,),
                dtype=wp.spatial_vectorf,
                device=self.device,
                copy=False,
            )
            for b in range(B)
        ]
        self._body_com_pose_b_flat = [
            wp.array(
                ptr=self._body_com_pose_b_base.ptr + b * N * tf_size,
                shape=(N,),
                dtype=wp.transformf,
                device=self.device,
                copy=False,
            )
            for b in range(B)
        ]

        # Strided (N, B) 2D views on top of the flat base arrays — same memory,
        # transposed stride so that [i, b] → flat[b*N + i].
        # These replace the TimestampedBuffer.data for properties that use per-body reads.
        self._body_link_pose_w.data = wp.array(
            ptr=self._body_link_pose_w_base.ptr,
            shape=(N, B),
            dtype=wp.transformf,
            strides=(tf_size, N * tf_size),
            device=self.device,
        )
        self._body_com_vel_w.data = wp.array(
            ptr=self._body_com_vel_w_base.ptr,
            shape=(N, B),
            dtype=wp.spatial_vectorf,
            strides=(sv_size, N * sv_size),
            device=self.device,
        )
        self._body_com_pose_b.data = wp.array(
            ptr=self._body_com_pose_b_base.ptr,
            shape=(N, B),
            dtype=wp.transformf,
            strides=(tf_size, N * tf_size),
            device=self.device,
        )

        # -- Body properties: mass (N, B) and inertia (N, B, 9).
        # Read each body's binding (CPU-only types) into a staging buffer and pack.
        mass_flat = wp.zeros(B * N, dtype=wp.float32, device=self.device)
        inertia_flat = wp.zeros(B * N * 9, dtype=wp.float32, device=self.device)
        for b in range(B):
            inertia_binding = self._root_view[TT.RIGID_BODY_INERTIA][b]

            # Read mass (N floats) into column b of mass_flat.
            mass_col = wp.array(
                ptr=mass_flat.ptr + b * N * 4,
                shape=(N,),
                dtype=wp.float32,
                device=self.device,
                copy=False,
            )
            self._read_binding_into(TT.RIGID_BODY_MASS, b, mass_col)

            # Read inertia (N * 9 floats) into the appropriate block of inertia_flat.
            inertia_col = wp.array(
                ptr=inertia_flat.ptr + b * N * 9 * 4,
                shape=(N, 9),
                dtype=wp.float32,
                device=self.device,
                copy=False,
            )
            inertia_binding_shape = inertia_binding.shape
            if TT.RIGID_BODY_INERTIA in TT._CPU_ONLY_TYPES and str(inertia_col.device) != "cpu":
                staging_inertia = wp.zeros(inertia_binding_shape, dtype=wp.float32, device="cpu", pinned=True)
                inertia_binding.read(staging_inertia)
                wp.copy(inertia_col, staging_inertia.reshape((N, 9)))
            else:
                inertia_binding.read(inertia_col)

        # Strided (N, B) view for mass, (N, B, 9) view for inertia.
        self._body_mass = wp.array(
            ptr=mass_flat.ptr,
            shape=(N, B),
            dtype=wp.float32,
            strides=(4, N * 4),
            device=self.device,
        )
        self._body_inertia = wp.array(
            ptr=inertia_flat.ptr,
            shape=(N, B, 9),
            dtype=wp.float32,
            strides=(9 * 4, N * 9 * 4, 4),
            device=self.device,
        )

        # Keep references so the flat bases are not garbage-collected.
        self._mass_flat_base = mass_flat
        self._inertia_flat_base = inertia_flat

        # -- Defaults (set by _process_cfg after __init__).
        # These remain None until _process_cfg writes them.

        # Initialize ProxyArray wrappers.
        self._pin_proxy_arrays()

    def _pin_proxy_arrays(self) -> None:
        """Create pinned :class:`ProxyArray` wrappers for all data buffers.

        Called once from :meth:`_create_buffers`. OVPhysX tensor API buffers have
        stable GPU pointers across simulation steps, so no rebinding is needed
        (unlike Newton).
        """
        # Defaults
        self._default_body_pose_ta: ProxyArray | None = None
        self._default_body_vel_ta: ProxyArray | None = None
        # Body state (timestamped)
        self._body_link_pose_w_ta: ProxyArray | None = None
        self._body_link_vel_w_ta: ProxyArray | None = None
        self._body_com_pose_w_ta: ProxyArray | None = None
        self._body_com_vel_w_ta: ProxyArray | None = None
        self._body_com_pose_b_ta: ProxyArray | None = None
        # Body properties
        self._body_mass_ta: ProxyArray | None = None
        self._body_inertia_ta: ProxyArray | None = None
        # Derived properties (in-body-frame velocities)
        self._body_link_lin_vel_b_ta: ProxyArray | None = None
        self._body_link_ang_vel_b_ta: ProxyArray | None = None
        self._body_com_lin_vel_b_ta: ProxyArray | None = None
        self._body_com_ang_vel_b_ta: ProxyArray | None = None
        # Sliced properties (body link)
        self._body_link_pos_w_ta: ProxyArray | None = None
        self._body_link_quat_w_ta: ProxyArray | None = None
        self._body_link_lin_vel_w_ta: ProxyArray | None = None
        self._body_link_ang_vel_w_ta: ProxyArray | None = None
        # Sliced properties (body com)
        self._body_com_pos_w_ta: ProxyArray | None = None
        self._body_com_quat_w_ta: ProxyArray | None = None
        self._body_com_lin_vel_w_ta: ProxyArray | None = None
        self._body_com_ang_vel_w_ta: ProxyArray | None = None
        # Sliced properties (body com in body frame)
        self._body_com_pos_b_ta: ProxyArray | None = None
        self._body_com_quat_b_ta: ProxyArray | None = None
        # Deprecated state-concat properties
        self._default_body_state_ta: ProxyArray | None = None
        self._body_state_w_ta: ProxyArray | None = None
        self._body_link_state_w_ta: ProxyArray | None = None
        self._body_com_state_w_ta: ProxyArray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_binding(self, tensor_type: int, body_idx: int):
        """Return the binding for the given tensor type and body index, or None."""
        body_list = self._root_view.get(tensor_type)
        if body_list is None or body_idx >= len(body_list):
            return None
        return body_list[body_idx]

    def _read_binding_into(self, tensor_type: int, body_idx: int, dst: wp.array) -> None:
        """Read the OVPhysX TensorBinding for *tensor_type* and *body_idx* into *dst*.

        Adapter that replaces PhysX's view-getter pattern: the wheel exposes
        ``binding.read(target)`` rather than a getter returning a :class:`warp.array`,
        so we read into a flat float32 view of *dst*. CPU-only bindings on a non-CPU
        sim go through a lazily-allocated pinned-host ``wp.array`` to satisfy the
        wheel's device constraint.

        Args:
            tensor_type: The TensorType constant identifying which simulation buffer
                to read.
            body_idx: The body index within the collection (0-based).
            dst: The destination :class:`warp.array` to write into. Must have at
                least as many bytes as the binding.
        """
        binding = self._root_view[tensor_type][body_idx]
        if self._check_shapes:
            dst_bytes = dst.size * wp.types.type_size_in_bytes(dst.dtype)
            binding_bytes = 4 * math.prod(binding.shape)
            assert dst_bytes >= binding_bytes, (
                f"_read_binding_into: dst buffer too small for binding tt={tensor_type!r}, body={body_idx} "
                f"({dst_bytes} B < {binding_bytes} B). Caller allocated dst with "
                f"shape={tuple(dst.shape)}, dtype={dst.dtype}; binding shape={tuple(binding.shape)}."
            )
        # Build a flat float32 view of dst matching the binding's shape.
        if dst.dtype == wp.float32:
            view = dst
        else:
            view = wp.array(
                ptr=dst.ptr,
                shape=binding.shape,
                dtype=wp.float32,
                device=str(dst.device),
                copy=False,
            )
        key = (tensor_type, body_idx)
        if tensor_type in TT._CPU_ONLY_TYPES and str(view.device) != "cpu":
            staging = self._cpu_staging_buffers.get(key)
            if staging is None:
                staging = wp.zeros(binding.shape, dtype=wp.float32, device="cpu", pinned=True)
                self._cpu_staging_buffers[key] = staging
            binding.read(staging)
            wp.copy(view, staging)
        else:
            binding.read(view)

    def _get_pos_from_transform(self, transform: wp.array) -> wp.array:
        """Generates a position array from a transform array."""
        return wp.array(
            ptr=transform.ptr,
            shape=transform.shape,
            dtype=wp.vec3f,
            strides=transform.strides,
            device=self.device,
        )

    def _get_quat_from_transform(self, transform: wp.array) -> wp.array:
        """Generates a quaternion array from a transform array."""
        return wp.array(
            ptr=transform.ptr + 3 * 4,
            shape=transform.shape,
            dtype=wp.quatf,
            strides=transform.strides,
            device=self.device,
        )

    def _get_lin_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        """Generates a linear velocity array from a spatial vector array."""
        return wp.array(
            ptr=sv.ptr,
            shape=sv.shape,
            dtype=wp.vec3f,
            strides=sv.strides,
            device=self.device,
        )

    def _get_ang_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        """Generates an angular velocity array from a spatial vector array."""
        return wp.array(
            ptr=sv.ptr + 3 * 4,
            shape=sv.shape,
            dtype=wp.vec3f,
            strides=sv.strides,
            device=self.device,
        )
