# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed rigid object collection data container."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
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
    """Data container for a rigid object collection backed by OVPhysX.

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
        **Single fused binding.** OVPhysX 0.4.3+ exposes a fused multi-prim
        binding created with ``prim_paths=[...]``.  Each binding returns data of
        shape ``(num_instances, num_bodies, D)``, matching the Articulation body
        binding convention.  One binding read fills the entire
        ``(num_instances, num_bodies, D)`` buffer; no per-body Python loops are
        needed.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object collection data."""

    def __init__(
        self,
        root_view: dict[int, Any],
        num_bodies: int,
        device: str,
    ):
        """Initialize the rigid object collection data.

        Args:
            root_view: Fused TensorBinding dict, keyed by TensorType constant.
                Each value is a single :class:`TensorBinding` spanning all bodies
                in the collection (shape ``(num_instances, num_bodies, D)``).
            num_bodies: The number of object types managed by the collection.
            device: The device used for processing (e.g. ``"cuda:0"`` or ``"cpu"``).
        """
        super().__init__(root_view, num_bodies, device)
        # Store the bindings dict (equivalent to the view in PhysX).
        self._bindings = root_view
        self._binding_getter = None  # may be set externally after construction
        self.num_bodies = num_bodies
        self._num_bodies = num_bodies
        # Set initial time stamp.
        self._sim_timestamp = 0.0
        self._is_primed = False
        # Pinned-host staging buffers for CPU-only bindings (keyed by tensor_type).
        self._cpu_staging_buffers: dict[int, wp.array] = {}
        # Cache for float32 read views (keyed by (tensor_type, ptr)).
        self._read_view_cache: dict = {}

        # Read num_instances from the LINK_POSE binding.
        self.num_instances = self._bindings[TT.LINK_POSE].count
        self._num_instances = self.num_instances

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
        # Mirrors RigidObject's update() pattern.
        # Priming an FD-dependent derived property ensures the first read
        # returns sensible (zero) acceleration.
        _ = self.body_com_acc_w

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
    def default_body_state(self) -> ProxyArray:
        """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame.

        Deprecated. Use :attr:`default_body_pose` and :attr:`default_body_vel` instead.

        Shape is (num_instances, num_bodies), dtype = ``vec13f``.
        In torch this resolves to (num_instances, num_bodies, 13).
        """
        warnings.warn(
            "Reading the body state directly is deprecated since IsaacLab 3.0 and will be removed in a future version. "
            "Please use the default_body_pose and default_body_vel properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_body_state is None:
            self._default_body_state = wp.zeros(
                (self.num_instances, self.num_bodies), dtype=shared_kernels.vec13f, device=self.device
            )
            self._default_body_state_ta = ProxyArray(self._default_body_state)
        wp.launch(
            shared_kernels.concat_body_pose_and_vel_to_state,
            dim=(self.num_instances, self.num_bodies),
            inputs=[
                self._default_body_pose,
                self._default_body_vel,
            ],
            outputs=[
                self._default_body_state,
            ],
            device=self.device,
        )
        return self._default_body_state_ta

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
            self._read_transform_binding(TT.LINK_POSE, self._body_link_pose_w)
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
            self._read_spatial_vector_binding(TT.LINK_VELOCITY, self._body_com_vel_w)
            self._body_com_lin_vel_w_ta = None
            self._body_com_ang_vel_w_ta = None
        if self._body_com_vel_w_ta is None:
            self._body_com_vel_w_ta = ProxyArray(self._body_com_vel_w.data)
        return self._body_com_vel_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame [m/s², rad/s²].

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        In torch this resolves to (num_instances, num_bodies, 6).
        This quantity is the acceleration of the rigid bodies' center of mass frame relative
        to the world, derived by finite differencing consecutive COM velocities.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            if self._previous_body_com_vel is None:
                self._previous_body_com_vel = wp.clone(self.body_com_vel_w.warp)
            wp.launch(
                shared_kernels.derive_body_acceleration_from_body_com_velocities,
                dim=(self.num_instances, self.num_bodies),
                device=self.device,
                inputs=[
                    self.body_com_vel_w.warp,
                    SimulationManager.get_physics_dt(),
                    self._previous_body_com_vel,
                ],
                outputs=[
                    self._body_com_acc_w.data,
                ],
            )
            self._body_com_acc_w.timestamp = self._sim_timestamp
            self._body_com_lin_acc_w_ta = None
            self._body_com_ang_acc_w_ta = None
        if self._body_com_acc_w_ta is None:
            self._body_com_acc_w_ta = ProxyArray(self._body_com_acc_w.data)
        return self._body_com_acc_w_ta

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
            self._read_transform_binding(TT.BODY_COM_POSE, self._body_com_pose_b)
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
            self._body_mass_ta = ProxyArray(self._body_mass.data)
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
            self._body_inertia_ta = ProxyArray(self._body_inertia.data)
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
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Linear acceleration of all bodies' center of mass in simulation world frame [m/s²].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_acc_w
        if self._body_com_lin_acc_w_ta is None:
            self._body_com_lin_acc_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_acc_w_ta

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Angular acceleration of all bodies' center of mass in simulation world frame [rad/s²].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_acc_w
        if self._body_com_ang_acc_w_ta is None:
            self._body_com_ang_acc_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_acc_w_ta

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
    def projected_gravity_b(self) -> ProxyArray:
        """Projection of the gravity direction onto each body frame [-].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.GRAVITY_VEC_W, self.body_link_quat_w],
                outputs=[self._projected_gravity_b.data],
                device=self.device,
            )
            self._projected_gravity_b.timestamp = self._sim_timestamp
        if self._projected_gravity_b_ta is None:
            self._projected_gravity_b_ta = ProxyArray(self._projected_gravity_b.data)
        return self._projected_gravity_b_ta

    @property
    def heading_w(self) -> ProxyArray:
        """Yaw heading of each body frame [rad].

        Shape is (num_instances, num_bodies), dtype = ``wp.float32``.
        In torch this resolves to (num_instances, num_bodies).

        .. note::
            This quantity is computed by assuming that the forward-direction of each
            body frame is along the x-direction, i.e. :math:`(1, 0, 0)`.
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.body_heading_w,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.FORWARD_VEC_B, self.body_link_quat_w],
                outputs=[self._heading_w.data],
                device=self.device,
            )
            self._heading_w.timestamp = self._sim_timestamp
        if self._heading_w_ta is None:
            self._heading_w_ta = ProxyArray(self._heading_w.data)
        return self._heading_w_ta

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _create_buffers(self) -> None:
        """Eagerly allocate every per-body TimestampedBuffer and the slots for
        cached :class:`ProxyArray` wrappers.

        Buffers use direct ``(num_instances, num_bodies, D)`` shapes, matching
        the fused binding output.  No flat+strided tricks are needed because the
        fused binding returns a contiguous ``(N, B, D)`` array directly.
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
        # -- derived properties (acceleration via finite differencing)
        self._body_com_acc_w = TimestampedBuffer((N, B), self.device, wp.spatial_vectorf)
        # Holds the previous-step COM velocity for FD; initialised lazily on first access.
        self._previous_body_com_vel: wp.array | None = None
        # -- derived properties (projected gravity and heading)
        self._projected_gravity_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        self._heading_w = TimestampedBuffer((N, B), self.device, wp.float32)

        # -- Body properties: mass (N, B) and inertia (N, B, 9).
        # Initialised eagerly from the CPU-only bindings.
        self._body_mass = TimestampedBuffer((N, B), self.device, wp.float32)
        self._body_inertia = TimestampedBuffer((N, B, 9), self.device, wp.float32)

        # Pinned CPU staging buffers used by mass/com/inertia setters (mirrors Articulation).
        pinned = self.device != "cpu"
        self._cpu_body_mass = wp.zeros((N, B), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_body_coms = wp.zeros((N, B, 7), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_body_inertia = wp.zeros((N, B, 9), dtype=wp.float32, device="cpu", pinned=pinned)

        # Eagerly read mass and inertia (CPU-only bindings) at construction time.
        # Use numpy round-trip (same pattern as ArticulationData._create_buffers).
        def _read_cpu(tensor_type):
            binding = self._get_binding(tensor_type)
            if binding is None:
                return None
            np_buf = np.zeros(binding.shape, dtype=np.float32)
            binding.read(np_buf)
            return np_buf

        np_mass = _read_cpu(TT.BODY_MASS)
        if np_mass is not None:
            wp.copy(self._body_mass.data, wp.from_numpy(np_mass, dtype=wp.float32, device=self.device))
            self._body_mass.timestamp = self._sim_timestamp

        np_inertia = _read_cpu(TT.BODY_INERTIA)
        if np_inertia is not None:
            wp.copy(
                self._body_inertia.data,
                wp.from_numpy(np_inertia, dtype=wp.float32, device=self.device),
            )
            self._body_inertia.timestamp = self._sim_timestamp

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
        # Derived properties (FD acceleration)
        self._body_com_acc_w_ta: ProxyArray | None = None
        self._body_com_lin_acc_w_ta: ProxyArray | None = None
        self._body_com_ang_acc_w_ta: ProxyArray | None = None
        # Derived properties (projected gravity and heading)
        self._projected_gravity_b_ta: ProxyArray | None = None
        self._heading_w_ta: ProxyArray | None = None
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
        self._default_body_state: wp.array | None = None
        self._default_body_state_ta: ProxyArray | None = None
        self._body_state_w_ta: ProxyArray | None = None
        self._body_link_state_w_ta: ProxyArray | None = None
        self._body_com_state_w_ta: ProxyArray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_binding(self, tensor_type: int):
        """Return the binding for the given tensor type, or None.

        Mirrors :meth:`~isaaclab_ovphysx.assets.Articulation._get_binding` exactly:
        a single binding per tensor type, no body index.

        Args:
            tensor_type: The TensorType constant identifying which simulation buffer.

        Returns:
            The cached :class:`TensorBinding`, or ``None`` if not available.
        """
        b = self._bindings.get(tensor_type)
        if b is not None:
            return b
        if self._binding_getter is not None:
            b = self._binding_getter(tensor_type)
            if b is not None:
                self._bindings[tensor_type] = b
            return b
        return None

    def _binding_read(self, tensor_type: int, binding, dst: wp.array) -> None:
        """Read *binding* into *dst*, staging through pinned-host for CPU-only bindings.

        Mirrors :meth:`~isaaclab_ovphysx.assets.articulation.ArticulationData._binding_read`.

        Args:
            tensor_type: TensorType key identifying the binding.
            binding: OVPhysX TensorBinding whose ``read`` method is called.
            dst: Destination :class:`wp.array` on the simulation device.
        """
        if tensor_type not in TT._CPU_ONLY_TYPES or self.device == "cpu":
            binding.read(dst)
            return
        # Route through a lazily-allocated pinned-host staging buffer.
        staging = self._cpu_staging_buffers.get(tensor_type)
        if staging is None:
            staging = wp.zeros(binding.shape, dtype=wp.float32, device="cpu", pinned=True)
            self._cpu_staging_buffers[tensor_type] = staging
        binding.read(staging)
        # Build a flat float32 view of dst matching the binding's flat shape.
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
        wp.copy(view, staging)

    def _get_read_view(self, tensor_type: int, wp_array: wp.array, floats_per_elem: int = 0) -> wp.array | None:
        """Return a stable float32 view of a warp buffer for reading from a binding.

        For structured-dtype buffers (transformf, spatial_vectorf), the view
        reinterprets the same GPU memory as a flat float32 array matching the
        binding's shape.  For plain float32 buffers, returns the array as-is.

        The returned view is cached so that ``binding.read(view)`` sees the
        same object on every call.

        Mirrors :meth:`~isaaclab_ovphysx.assets.articulation.ArticulationData._get_read_view`.

        Args:
            tensor_type: TensorType key.
            wp_array: Destination warp array.
            floats_per_elem: Number of float32 elements per logical element
                (e.g. 7 for transformf, 6 for spatial_vectorf).  Pass 0 to
                return the array as-is.

        Returns:
            Float32 view suitable for ``binding.read()``, or ``None``.
        """
        cache_key = (tensor_type, wp_array.ptr)
        cached = self._read_view_cache.get(cache_key)
        if cached is not None:
            return cached

        binding = self._get_binding(tensor_type)
        if binding is None:
            self._read_view_cache[cache_key] = None
            return None

        if floats_per_elem > 0:
            view = wp.array(
                ptr=wp_array.ptr,
                shape=binding.shape,
                dtype=wp.float32,
                device=str(wp_array.device),
                copy=False,
            )
        else:
            view = wp_array

        self._read_view_cache[cache_key] = view
        return view

    def _read_transform_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a pose binding (float32 view of transformf buffer), skipping if fresh.

        CPU-only bindings (e.g. ``BODY_COM_POSE``) are routed through a
        pinned-host staging buffer via :meth:`_binding_read`.

        Args:
            tensor_type: TensorType key.
            buf: Timestamped :class:`wp.transformf` buffer to refresh.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        view = self._get_read_view(tensor_type, buf.data, 7)
        if view is None:
            return
        self._binding_read(tensor_type, binding, view)
        buf.timestamp = self._sim_timestamp

    def _read_spatial_vector_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a velocity binding (float32 view of spatial_vectorf buffer), skipping if fresh.

        Args:
            tensor_type: TensorType key.
            buf: Timestamped :class:`wp.spatial_vectorf` buffer to refresh.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        view = self._get_read_view(tensor_type, buf.data, 6)
        if view is None:
            return
        self._get_binding(tensor_type).read(view)
        buf.timestamp = self._sim_timestamp

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
